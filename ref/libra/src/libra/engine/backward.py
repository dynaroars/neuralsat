"""
Backward Analysis Engine
========================

:Author: Caterina Urban
"""

import itertools
import operator
import time
from copy import deepcopy
from functools import reduce
from multiprocessing import Value, Manager, Queue, Process, cpu_count, Lock
from typing import Tuple, Set, Dict, List, FrozenSet

from apronpy.box import PyBoxMPQManager
from apronpy.coeff import PyMPQScalarCoeff
from apronpy.interval import Interval
from apronpy.lincons0 import ConsTyp
from apronpy.manager import PyManager
from apronpy.polka import PyPolka, PyPolkaMPQstrictManager
from apronpy.tcons1 import PyTcons1, PyTcons1Array
from apronpy.texpr0 import TexprOp, TexprRtype, TexprRdir
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from pip._vendor.colorama import Fore, Style, Back

from libra.abstract_domains.bias_domain import BiasState
from libra.abstract_domains.state import State
from libra.core.cfg import Node, Function, Activation
from libra.core.expressions import BinaryComparisonOperation, Literal, VariableIdentifier, BinaryBooleanOperation
from libra.engine.interpreter import Interpreter
from libra.semantics.backward import DefaultBackwardSemantics
from libra.abstract_domains.interval2_domain import Box2State
from libra.abstract_domains.deeppoly_domain import DeepPolyState
from libra.abstract_domains.neurify_domain import NeurifyState
from libra.abstract_domains.symbolic3_domain import Symbolic3State
from libra.abstract_domains.product_domain import ProductState


rtype = TexprRtype.AP_RTYPE_REAL
rdir = TexprRdir.AP_RDIR_RND
OneHot1 = Tuple[VariableIdentifier, BinaryBooleanOperation]      # one-hot value for 1 feature
OneHotN = Tuple[OneHot1, ...]                                    # one-hot values for n features
lock = Lock()
NON_APRON_DOMAINS = (Box2State, DeepPolyState, NeurifyState, Symbolic3State, ProductState)


def one_hots(variables: List[VariableIdentifier]) -> Set[OneHot1]:
    """Compute all possible one-hots for a given list of variables

    :param variables: list of variables one-hot encoding a categorical input feature
    :return: set of Libra expressions corresponding to each possible value of the one-hot encoding
    (paired with the variable being one in the encoded value for convenience ---
    the variable is the first element of the tuple)
    """
    values: Set[OneHot1] = set()
    arity = len(variables)
    for i in range(arity):
        _value = dict()
        # the current variable has value one
        one = Literal('1')
        lower = BinaryComparisonOperation(one, BinaryComparisonOperation.Operator.LtE, variables[i])
        upper = BinaryComparisonOperation(variables[i], BinaryComparisonOperation.Operator.LtE, one)
        value = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
        _value[variables[i]] = (1, 1)
        # everything else has value zero
        zero = Literal('0')
        for j in range(0, i):
            lower = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, variables[j])
            upper = BinaryComparisonOperation(variables[j], BinaryComparisonOperation.Operator.LtE, zero)
            conj = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
            value = BinaryBooleanOperation(conj, BinaryBooleanOperation.Operator.And, value)
            _value[variables[j]] = (0, 0)
        for j in range(i + 1, arity):
            lower = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, variables[j])
            upper = BinaryComparisonOperation(variables[j], BinaryComparisonOperation.Operator.LtE, zero)
            conj = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
            value = BinaryBooleanOperation(value, BinaryBooleanOperation.Operator.And, conj)
            _value[variables[j]] = (0, 0)
        values.add((variables[i], value, tuple(_value.items())))
    return values


class BackwardInterpreter(Interpreter):
    """Backward control flow graph interpreter."""

    def __init__(self, cfg, manager, domain, semantics, specification, steps=None, minL=None, startL=0.25, startU=2, maxU=None, cpu=None, precursory=None):
        super().__init__(cfg, semantics, precursory=precursory)
        self.manager: PyManager = manager                               # manager to be used for the analysis
        from libra.engine.bias_analysis import AbstractDomain
        self.domain: AbstractDomain = domain
        self._initial: BiasState = None                                 # initial analysis state

        self.specification = specification                              # input specification file
        self.sensitive: List[VariableIdentifier] = None                 # sensitive feature
        self.values: List[OneHot1] = None                               # all one-hot sensitive values
        self.uncontroversial1: List[List[VariableIdentifier]] = None    # uncontroversial features / one-hot encoded
        self.uncontroversial2: List[VariableIdentifier] = None          # uncontroversial features / custom encoded
        self.bounds: BinaryBooleanOperation = None      # bound between 0 and 1 of sensitive and one-hot encoded

        self.outputs: Set[VariableIdentifier] = None                    # output classes

        self.activations = None                                         # activation nodes
        self.active = None                                              # always active activations
        self.inactive = None                                            # always inactive activations
        self.packs = Manager().dict()                                   # packing of 1-hot splits
        self.count = 0                                                  # 1-hot split count
        self.patterns = Manager().dict()                                # packing of abstract activation patterns
        self.discarded = Value('i', 0)
        self.partitions = Value('i', 0)

        self.autotuning = steps is not None
        self.steps = (1, 1) if steps is None else steps                 # autotuning heuristic
        self.minL = startL if minL is None else minL                    # minimum lower bound (default: starting lower bound)
        self.startL = startL                                            # starting lower bound (default: 0.25)
        self.startU = startU                                            # starting upper bound (default: 2)
        self.maxU = startU if maxU is None else maxU                    # maximum upper bound (default: maximum upper bound)
        self.lower = Value('d', 1)
        self.upper = Value('i', 0)

        self.fair = Value('d', 0.0)                                     # percentage that is proven fair by the pre-analysis
        self.biased = Value('d', 0.0)                                   # percentage that is biased
        self.feasible = Value('d', 0.0)                                 # percentage that could be analyzed
        self.explored = Value('d', 0.0)                                 # percentage that was explored
        self.analyzed = Value('i', 0)                                   # analyzed patterns

        self.cpu = cpu_count() if cpu is None else cpu

    @property
    def initial(self):
        """Initial analysis state

        :return: a deep copy of the initial analysis state
        """
        return deepcopy(self._initial)

    def feasibility(self, state, manager, disjuncts, key=None, chunk=None):
        """Determine feasibility (and activation patterns) for a partition of the input space

        :param state: state representing the partition of the input space
        :param manager: manager to be used for the (forward) analysis
        :param key: pre-determined activations
        :param do: whether to compute the activation patterns even if the analysis is not feasible
        :return: feasibility, patterns, last computed number of disjunctions
        """
        feasible = True
        patterns: List[Tuple[OneHot1, FrozenSet[Node], FrozenSet[Node]]] = list()
        outcomes = set()
        _disjunctions = None
        for idx, value in enumerate(self.values):
            if isinstance(state, NON_APRON_DOMAINS):
                result = deepcopy(state).assume(list(value[2]))
            else:
                result = deepcopy(state).assume({value[1]}, manager=manager)
            f_active = key[idx][0] if key else None
            f_inactive = key[idx][1] if key else None
            active, inactive, outcome = self.precursory.analyze(result, forced_active=f_active, forced_inactive=f_inactive, outputs=self.outputs)
            outcomes.add(outcome)
            disjunctions = len(self.activations) - len(active) - len(inactive)
            if disjunctions > disjuncts:
                _disjunctions = disjunctions
                feasible = False
            patterns.append((value, frozenset(active), frozenset(inactive)))
        if len(outcomes) <= 1 and None not in outcomes:
            classes = ', '.join(str(outcome) for outcome in outcomes)
            print(Fore.GREEN + '✔︎ No Bias ({}) in {}'.format(classes, chunk), Style.RESET_ALL)
            return True, list(), len(self.activations)
        return feasible, patterns, _disjunctions

    def producer(self, queue3):
        """Produce all possible combinations of one-hots for the one-hot encoded uncontroversial features

        :param queue3: queue in which to put the combinations
        """
        one_hotn = itertools.product(*(one_hots(encoding) for encoding in self.uncontroversial1))
        for one_hot in one_hotn:
            queue3.put(one_hot)
        queue3.put(None)

    def consumer(self, queue3, entry, manager):
        """Consume a combination of one-hots and put it in its abstract activation pattern pack

        :param queue3: queue from which to get the combination
        :param entry: state from which to start the (forward) analysis
        :param manager: manager to be used for the analysis
        """
        while True:
            one_hot = queue3.get(block=True)
            if one_hot is None:
                queue3.put(None)
                break
            result1 = deepcopy(entry)
            for item in one_hot:
                if isinstance(entry, NON_APRON_DOMAINS):
                    result1 = result1.assume(list(item[2]))
                else:
                    result1 = result1.assume({item[1]}, manager=manager)
            key = list()
            for value in self.values:
                if isinstance(entry, NON_APRON_DOMAINS):
                    result2 = deepcopy(result1).assume(list(value[2]))
                else:
                    result2 = deepcopy(result1).assume({value[1]}, manager=manager)
                active, inactive, _ = self.precursory.analyze(result2, earlystop=False, outputs=self.outputs)
                key.append((frozenset(active), frozenset(inactive)))
            _key = tuple(key)
            lock.acquire()
            curr = self.packs.get(_key, set())
            curr.add(one_hot)
            self.packs[_key] = curr
            lock.release()

    def packing(self, entry):
        """Pack all combinations of one-hots into abstract activation pattern packs

        :param entry: state from which to start the (forward) analysis
        """
        queue3 = Queue()
        start3 = time.time()
        processes = list()
        process = Process(target=self.producer, args=(queue3,))
        processes.append(process)
        for _ in range(self.cpu - 1):
            process = Process(target=self.consumer, args=(queue3, entry, PyBoxMPQManager()))
            processes.append(process)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        end3 = time.time()
        _count = sum(len(pack) for pack in self.packs.values())
        assert self.count == _count
        print(Fore.YELLOW + '\nFound {} Packs for {} 1-Hot Combinations:'.format(len(self.packs), _count))
        score = lambda k: sum(len(s[0]) + len(s[1]) for s in k)
        for key, pack in sorted(self.packs.items(), key=lambda v: score(v[0]) + len(v[1]), reverse=True):
            sset = lambda s: '{{{}}}'.format(', '.join('{}'.format(e) for e in s))
            skey = ' | '.join('{}, {}'.format(sset(pair[0]), sset(pair[1])) for pair in key)
            sscore = '(score: {})'.format(score(key) + len(pack))
            spack = ' | '.join('{}'.format(','.join('{}'.format(item[0]) for item in one_hot)) for one_hot in pack)
            print(Fore.YELLOW, skey, '->', spack, sscore, Style.RESET_ALL)
        print(Fore.YELLOW + '1-Hot Splitting Time: {}s\n'.format(end3 - start3), Style.RESET_ALL)

    def worker1(self, id, color, queue1, manager):
        """Partition the analysis into feasible chunks and pack them into abstract activation pattern packs

        :param id: id of the process
        :param color: color associated with the process (for logging)
        :param queue1: queue from which to get the current chunk
        :param manager: manager to be used for the (forward) analysis
        """
        while True:
            assumptions, steps, size, disjuncts, pivot1, unpacked, ranges, pivot2, splittable, percent, key = queue1.get(block=True)
            if assumptions is None:     # no more chunks
                queue1.put((None, None, None, None, None, None, None, None, None, None, None))
                break
            r_assumptions = '1-Hot: {}'.format(
                ', '.join('{}'.format('|'.join('{}'.format(var) for var in case)) for (case, _, _) in assumptions)
            ) if assumptions else ''
            r_ranges = 'Ranges: {}'.format(
                ', '.join('{} ∈ [{}, {}]'.format(feature, lower, upper) for feature, (lower, upper) in ranges)
            )
            r_partition = '{} | {}'.format(r_assumptions, r_ranges) if r_assumptions else '{}'.format(r_ranges)
            print(color + r_partition, Style.RESET_ALL)
            # bound the custom encoded uncontroversial features between their current lower and upper bounds
            bounds = self.bounds
            for feature, (lower, upper) in ranges:
                left = BinaryComparisonOperation(Literal(str(lower)), BinaryComparisonOperation.Operator.LtE, feature)
                right = BinaryComparisonOperation(feature, BinaryComparisonOperation.Operator.LtE, Literal(str(upper)))
                conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
                bounds = BinaryBooleanOperation(bounds, BinaryBooleanOperation.Operator.And, conj)
            if isinstance(self.initial.precursory, NON_APRON_DOMAINS):
                entry = self.initial.precursory.assume(ranges)
            else:
                entry = self.initial.precursory.assume({bounds}, manager=manager)
            # take into account the accumulated assumptions on the one-hot encoded uncontroversial features
            for (_, assumption, _assumption) in assumptions:
                if isinstance(entry, NON_APRON_DOMAINS):
                    entry = entry.assume(_assumption)
                else:
                    entry = entry.assume({assumption}, manager=manager)
            # determine chunk feasibility for each possible value of the sensitive feature
            feasibility = self.feasibility(entry, manager, disjuncts, key=key, chunk=r_partition)
            feasible: bool = feasibility[0]
            # pack the chunk, if feasible, or partition the space of values of all the uncontroversial features
            if feasible:    # the analysis is feasible
                with self.partitions.get_lock():
                    self.partitions.value += 1
                with self.feasible.get_lock():
                    self.feasible.value += percent
                with self.explored.get_lock():
                    self.explored.value += percent
                    if self.explored.value >= 100:
                        queue1.put((None, None, None, None, None, None, None, None, None, None, None))
                patterns: List[Tuple[OneHot1, Set[Node], Set[Node]]] = feasibility[1]
                if patterns:
                    key = list()
                    for _, active, inactive in patterns:
                        key.append((frozenset(active), frozenset(inactive)))
                    _key = tuple(key)
                    value = (frozenset(assumptions), frozenset(unpacked), frozenset(ranges), percent)
                    lock.acquire()
                    curr = self.patterns.get(_key, set())
                    curr.add(value)
                    self.patterns[_key] = curr
                    lock.release()
                    found = '‼ Possible Bias in {}'.format(r_partition)
                    print(Fore.LIGHTYELLOW_EX + found, Style.RESET_ALL)
                else:
                    with self.discarded.get_lock():
                        self.discarded.value += 1
                    with self.fair.get_lock():
                        self.fair.value += percent
                progress = 'Progress for #{}: {}% of {}% ({}% fair)'.format(id, self.feasible.value, self.explored.value, self.fair.value)
                print(Fore.YELLOW + progress, Style.RESET_ALL)
            else:  # too many disjunctions, we need to split further
                print('Too many disjunctions ({})!'.format(feasibility[2]))
                if pivot1 < len(self.uncontroversial1):  # we still have to split the one-hot encoded
                    print('1-hot splitting for: {}'.format(
                        ' | '.join(
                            ', '.join('{}'.format(var) for var in encoding) for encoding in self.uncontroversial1)
                    ))
                    self.packing(entry)     # pack the one-hot combinations
                    # run the analysis on the ranked packs
                    score = lambda k: sum(len(s[0]) + len(s[1]) for s in k)
                    for key, pack in sorted(self.packs.items(), key=lambda v: score(v[0]) + len(v[1]), reverse=True):
                        _assumptions = list(assumptions)
                        items: List[OneHotN] = list(pack)  # multiple one-hot values for n features
                        for i in range(len(items[0])):  # for each feature...
                            variables: Set[VariableIdentifier] = set()
                            var, case, _case = items[0][i]
                            variables.add(var)
                            for item in items[1:]:
                                var, nxt, _nxt = item[i]
                                variables.add(var)
                                case = BinaryBooleanOperation(case, BinaryBooleanOperation.Operator.Or, nxt)
                                _case = BinaryBooleanOperation(_case, BinaryBooleanOperation.Operator.Or, _nxt)
                            _assumptions.append((frozenset(variables), case, _case))
                        _unpacked = frozenset(frozenset(item) for item in pack)
                        _pivot1 = len(self.uncontroversial1)
                        _percent = percent * len(pack) / self.count
                        queue1.put((_assumptions, steps, size, disjuncts, _pivot1, _unpacked, ranges, pivot2, splittable, _percent, key))
                else:  # we can split the rest
                    if size == 0 and len(unpacked) > 1:  # unpack one-hots first if difference = 0
                        _percent = percent / len(unpacked)
                        print("Unpacking {}".format(r_assumptions))
                        for item in unpacked:
                            _assumptions = list()
                            for var, case, _case in item:
                                _assumptions.append((frozenset({var}), case, _case))
                            _unpacked = frozenset({item})
                            _assumptions = frozenset(_assumptions)
                            queue1.put((_assumptions, steps, size, disjuncts, pivot1, _unpacked, ranges, pivot2, splittable, _percent, None))
                    elif self.uncontroversial2 and splittable:
                        rangesdict = dict(ranges)
                        (lower, upper) = rangesdict[self.uncontroversial2[pivot2]]
                        if upper - lower <= size:
                            print('Cannot range split for {} anymore!'.format(self.uncontroversial2[pivot2]))
                            _splittable = list(splittable)
                            _splittable.remove(self.uncontroversial2[pivot2])
                            _pivot2 = (pivot2 + 1) % len(self.uncontroversial2)
                            _splittable = list(_splittable)
                            queue1.put((assumptions, steps, size, disjuncts, pivot1, unpacked, ranges, _pivot2, _splittable, percent, None))
                        else:
                            middle = lower + (upper - lower) / 2
                            print('Range split for {} at: {}'.format(self.uncontroversial2[pivot2], middle))
                            left = deepcopy(rangesdict)
                            left[self.uncontroversial2[pivot2]] = (lower, middle)
                            right = deepcopy(rangesdict)
                            right[self.uncontroversial2[pivot2]] = (middle, upper)
                            _pivot2 = (pivot2 + 1) % len(self.uncontroversial2)
                            _percent = percent / 2
                            _left, _right = list(left.items()), list(right.items())
                            queue1.put((assumptions, steps, size, disjuncts, pivot1, unpacked, _left, _pivot2, splittable, _percent, None))
                            queue1.put((assumptions, steps, size, disjuncts, pivot1, unpacked, _right, _pivot2, splittable, _percent, None))
                    elif len(unpacked) > 1:     # last resort: unpack the one-hot combinations
                        _percent = percent / len(unpacked)
                        print("Unpacking {}".format(r_assumptions))
                        for item in unpacked:
                            _assumptions = list()
                            for var, case, _case in item:
                                _assumptions.append((frozenset({var}), case, _case))
                            _unpacked = frozenset({item})
                            _assumptions = frozenset(_assumptions)
                            queue1.put((_assumptions, steps, size, disjuncts, pivot1, _unpacked, ranges, pivot2, splittable, _percent, None))
                    elif 2 * self.minL <= size or disjuncts < self.maxU:        # autotuning is possible
                        (stepsL, stepsU) = steps
                        if stepsU < self.steps[1] and disjuncts < self.maxU:
                            _stepsL, _stepsU = 0 if stepsU + 1 <= self.steps[1] else stepsL, stepsU + 1
                            _size, _pivot2, _splittable = size, pivot2, splittable
                            _disjuncts = disjuncts + 1
                            print(Fore.BLUE + "Upper bound increase from: {} to: {}".format(disjuncts, _disjuncts), Style.RESET_ALL)
                        elif stepsU < self.steps[1] and 2 * self.minL <= size:
                            _stepsL, _stepsU = stepsL, stepsU
                            _size, _pivot2, _splittable = size / 2, 0, list(self.uncontroversial2)
                            _disjuncts = disjuncts
                            print(Fore.BLUE + "Lower bound decrease from: {} to: {}".format(size, _size), Style.RESET_ALL)
                        elif stepsL < self.steps[0] and 2 * self.minL <= size:
                            _stepsL, _stepsU = stepsL + 1, 0 if stepsL + 1 <= self.steps[0] else stepsU
                            _size, _pivot2, _splittable = size / 2, 0, list(self.uncontroversial2)
                            _disjuncts = disjuncts
                            print(Fore.BLUE + "Lower bound decrease from: {} to: {}".format(size, _size), Style.RESET_ALL)
                        elif stepsL < self.steps[0] and disjuncts < self.maxU:
                            _stepsL, _stepsU = stepsL, stepsU
                            _size, _pivot2, _splittable = size, pivot2, splittable
                            _disjuncts = disjuncts + 1
                            print(Fore.BLUE + "Upper bound increase from: {} to: {}".format(disjuncts, _disjuncts), Style.RESET_ALL)
                        elif stepsL == self.steps[0] and stepsU == self.steps[1]:
                            _stepsL, _stepsU = stepsL, stepsU
                            if 2 * self.minL <= size:
                                _size, _pivot2, _splittable = size / 2, 0, list(self.uncontroversial2)
                                print(Fore.BLUE + "Lower bound decrease from: {} to: {}".format(size, _size), Style.RESET_ALL)
                            else:
                                _size, _pivot2, _splittable = size, pivot2, splittable
                            if disjuncts < self.maxU:
                                _disjuncts = disjuncts + 1
                                print(Fore.BLUE + "Upper bound increase from: {} to: {}".format(disjuncts, _disjuncts), Style.RESET_ALL)
                            else:
                                _disjuncts = disjuncts
                        else:
                            _stepsL, _stepsU = stepsL, stepsU
                            _size, _pivot2, _splittable = size, pivot2, splittable
                            _disjuncts = disjuncts
                        with self.lower.get_lock():
                            self.lower.value = min(self.lower.value, _size)
                        with self.upper.get_lock():
                            self.upper.value = max(self.upper.value, _disjuncts)
                        print(Fore.BLUE + "Autotuned to: L = {}, U = {}".format(_size, _disjuncts), Style.RESET_ALL)
                        queue1.put((assumptions, (_stepsL, _stepsU), _size, _disjuncts, pivot1, unpacked, ranges, _pivot2, _splittable, percent, key))
                    else:
                        with self.explored.get_lock():
                            self.explored.value += percent
                            if self.explored.value >= 100:
                                queue1.put((None, None, None, None, None, None, None, None, None, None, None))
                        found = '‼ Unchecked Bias in {}'.format(r_partition)
                        print(Fore.RED + found, Style.RESET_ALL)
                        progress = 'Progress for #{}: {}% of {}% ({}% fair)'.format(id, self.feasible.value, self.explored.value, self.fair.value)
                        print(Fore.YELLOW + progress, Style.RESET_ALL)

    def bias_check(self, chunk, result, ranges, percent):
        """Check for algorithmic bias

        :param chunk: chunk to be checked (string representation)
        :param result: result of the (backward) analysis for the current abstract activation pattern
        :param ranges: ranges for the custom encoded uncontroversial features in the chunk
        :param percent: percent of the input space covered by the chunk
        """
        nobias = True
        biases = set()
        b_ranges = dict()
        items = list(result.items())
        for i in range(len(items)):
            (outcome1, sensitive1), value1 = items[i]
            for j in range(i+1, len(items)):
                (outcome2, sensitive2), value2 = items[j]
                if outcome1 != outcome2 and sensitive1 != sensitive2:
                    for val1 in value1:
                        for val2 in value2:
                            intersection = deepcopy(val1).meet(val2)
                            for encoding in self.uncontroversial1:
                                intersection = intersection.forget(encoding)
                            for feature, (lower, upper) in ranges:
                                lte = BinaryComparisonOperation.Operator.LtE
                                left = BinaryComparisonOperation(Literal(str(lower)), lte, feature)
                                right = BinaryComparisonOperation(feature, lte, Literal(str(upper)))
                                conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
                                intersection = intersection.assume({conj}, manager=self.manager)
                            # for assumption in assumptions0:
                            #     intersection = intersection.assume(assumption)
                            representation = repr(intersection.polka)
                            if not representation.startswith('-1.0 >= 0') and not representation == '⊥':
                                nobias = False
                                if representation not in biases:
                                    for uncontroversial in self.uncontroversial2:
                                        itv: Interval = intersection.polka.bound_variable(PyVar(uncontroversial.name))
                                        lower = eval(str(itv.interval.contents.inf.contents))
                                        upper = eval(str(itv.interval.contents.sup.contents))
                                        if uncontroversial in b_ranges:
                                            inf, sup = b_ranges[uncontroversial]
                                            b_ranges[uncontroversial] = (min(lower, inf), max(upper, sup))
                                        else:
                                            b_ranges[uncontroversial] = (lower, upper)
                                    biases.add(representation)
                                    pair = '{}->{} vs {}->{}'.format(sensitive1, outcome1, sensitive2, outcome2)
                                    found = '✘ Bias Found ({})! in {}:\n{}'.format(pair, chunk, representation)
                                    print(Fore.RED + found, Style.RESET_ALL)
        if nobias:
            outcomes = set()
            for i in range(len(items)):
                (outcome, sensitive), value = items[i]
                for val in value:
                    for encoding in self.uncontroversial1:
                        val = val.forget(encoding)
                    for feature, (lower, upper) in ranges:
                        lte = BinaryComparisonOperation.Operator.LtE
                        left = BinaryComparisonOperation(Literal(str(lower)), lte, feature)
                        right = BinaryComparisonOperation(feature, lte, Literal(str(upper)))
                        conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
                        val = val.assume({conj}, manager=self.manager)
                        representation = repr(val.polka)
                        if not representation.startswith('-1.0 >= 0') and not representation == '⊥':
                            outcomes.add(outcome)
            classes = ', '.join(str(outcome) for outcome in outcomes)
            print(Fore.GREEN + '✔︎ No Bias ({}) in {}'.format(classes, chunk), Style.RESET_ALL)
        else:
            total_size = 1
            for _, (lower, upper) in ranges:
                total_size *= upper - lower
            biased_size = 1
            for (lower, upper) in b_ranges.values():
                biased_size *= upper - lower
            _percent = percent * biased_size / total_size
            with self.biased.get_lock():
                self.biased.value += _percent

    def from_node(self, node, initial, join):
        """Run the backward analysis

        :param node: node from which to start the (backward) analysis
        :param initial: state from which to start the (backward) analysis
        :param join: whether joins should be performed
        :return: the result of the (backward) analysis (at the beginning of the CFG)
        """
        state = initial
        if isinstance(node, Function):
            state = self.semantics.list_semantics(node.stmts, state)
            if state.is_bottom():
                yield None
            else:
                if self.cfg.predecessors(node):
                    yield from self.from_node(self.cfg.nodes[self.cfg.predecessors(node).pop()], state, join)
                else:
                    yield state
        elif isinstance(node, Activation):
            if node in self.active:  # only the active path is viable
                state = self.semantics.ReLU_call_semantics(node.stmts, state, self.manager, True)
                if state.is_bottom():
                    yield None
                else:
                    predecessor = self.cfg.nodes[self.cfg.predecessors(node).pop()]
                    yield from self.from_node(predecessor, state, join)
            elif node in self.inactive:  # only the inactive path is viable
                state = self.semantics.ReLU_call_semantics(node.stmts, state, self.manager, False)
                if state.is_bottom():
                    yield None
                else:
                    predecessor = self.cfg.nodes[self.cfg.predecessors(node).pop()]
                    yield from self.from_node(predecessor, state, join)
            else:  # both paths are viable
                active, inactive = deepcopy(state), deepcopy(state)
                state1 = self.semantics.ReLU_call_semantics(node.stmts, active, self.manager, True)
                state2 = self.semantics.ReLU_call_semantics(node.stmts, inactive, self.manager, False)
                if join:
                    state = state1.join(state2)
                    predecessor = self.cfg.nodes[self.cfg.predecessors(node).pop()]
                    yield from self.from_node(predecessor, state, join)
                else:
                    if state1.is_bottom():
                        if state2.is_bottom():
                            yield None
                        else:
                            predecessor = self.cfg.nodes[self.cfg.predecessors(node).pop()]
                            yield from self.from_node(predecessor, state2, join)
                    else:
                        predecessor = self.cfg.nodes[self.cfg.predecessors(node).pop()]
                        yield from self.from_node(predecessor, state1, join)
                        if state2.is_bottom():
                            yield None
                        else:
                            yield from self.from_node(predecessor, state2, join)
        else:
            for stmt in reversed(node.stmts):
                state = self.semantics.assume_call_semantics(stmt, state, self.manager)
            if state.is_bottom():
                yield None
            else:
                if self.cfg.predecessors(node):
                    yield from self.from_node(self.cfg.nodes[self.cfg.predecessors(node).pop()], state, join)
                else:
                    yield state

    def worker2(self, id, color, queue2, manager, total):
        """Run the analysis for an abstract activation pattern and check the corresponding chunks for algorithmic bias

        :param id: id of the process
        :param color: color associated with the process (for logging)
        :param queue2: queue from which to get the current abstract activation pattern and corresponding chunks
        :param manager: manager to be used for the (backward) analysis
        :param total: total number of abstract activation patterns
        """
        while True:
            idx, (key, pack) = queue2.get(block=True)
            if idx is None:     # no more abstract activation patterns
                queue2.put((None, (None, None)))
                break
            print(color + 'Pattern #{} of {} [{}]'.format(idx, total, len(pack)), Style.RESET_ALL)
            check: Dict[Tuple[VariableIdentifier, VariableIdentifier], Set[BiasState]] = dict()
            for idx, (case, value, _) in enumerate(self.values):
                self.active, self.inactive = key[idx]
                for chosen in self.outputs:
                    remaining = self.outputs - {chosen}
                    discarded = remaining.pop()
                    outcome = BinaryComparisonOperation(discarded, BinaryComparisonOperation.Operator.Lt, chosen)
                    for discarded in remaining:
                        cond = BinaryComparisonOperation(discarded, BinaryComparisonOperation.Operator.Lt, chosen)
                        outcome = BinaryBooleanOperation(outcome, BinaryBooleanOperation.Operator.And, cond)
                    result = self.initial.assume({outcome}, manager=manager, bwd=True)
                    check[(chosen, case)] = set()
                    for state in self.from_node(self.cfg.out_node, deepcopy(result), False):
                        if state:
                            state = state.assume({value}, manager=manager)
                            check[(chosen, case)].add(state)
            # check for bias
            for assumptions, unpacked, ranges, percent in pack:
                r_assumptions = '1-Hot: {}'.format(
                    ', '.join('{}'.format('|'.join('{}'.format(var) for var in case)) for (case, _, _) in assumptions)
                ) if assumptions else ''
                r_ranges = 'Ranges: {}'.format(
                    ', '.join('{} ∈ [{}, {}]'.format(feature, lower, upper) for feature, (lower, upper) in ranges)
                )
                r_partition = '{} | {}'.format(r_assumptions, r_ranges) if r_assumptions else '{}'.format(r_ranges)
                if unpacked:
                    _percent = percent / len(unpacked)
                    for item in unpacked:
                        partition = deepcopy(check)
                        for states in partition.values():
                            for state in states:
                                for (_, assumption, _) in item:
                                    state.assume({assumption}, manager=manager)
                                # forget the sensitive variables
                                state.forget(self.sensitive)
                        self.bias_check(r_partition, partition, ranges, _percent)
                else:
                    partition = deepcopy(check)
                    for states in partition.values():
                        for state in states:
                            # forget the sensitive variables
                            state.forget(self.sensitive)
                    self.bias_check(r_partition, partition, ranges, percent)
            with self.analyzed.get_lock():
                self.analyzed.value += len(pack)
            analyzed = self.analyzed.value
            discarded = self.discarded.value
            partitions = self.partitions.value
            considered = partitions - discarded
            biased = self.biased.value
            progress = 'Progress for #{}: {} of {} partitions ({}% biased)'.format(id, analyzed, considered, biased)
            print(Fore.YELLOW + progress, Style.RESET_ALL)

    def analyze(self, initial, inputs=None, outputs=None, activations=None, analysis=True):
        """Backward analysis checking for algorithmic bias

        :param initial: (BiasState) state from which to start the analysis
        :param inputs: (Set[VariableIdentifier]) input variables
        :param outputs: (Set[VariableIdentifier]) output variables
        :param activations: (Set[Node]) CFG nodes corresponding to activation functions
        """
        print(Fore.BLUE + '\n||==================================||')
        print('|| domain: {}'.format(self.domain))
        print('|| min_difference: {}'.format(self.minL))
        print('|| start_difference: {}'.format(self.startL))
        print('|| start_widening: {}'.format(self.startU))
        print('|| max_widening: {}'.format(self.maxU))
        print('||==================================||', Style.RESET_ALL)
        self._initial = initial
        with open(self.specification, 'r') as specification:
            """
            pick sensitive feature and fix its bounds / we assume one-hot encoding
            """
            arity = int(specification.readline().strip())
            self.sensitive = list()
            for i in range(arity):
                self.sensitive.append(VariableIdentifier(specification.readline().strip()))
            if arity > 1:   # the sensitive feature is one-hot encoded
                self.values: List[OneHot1] = list(one_hots(self.sensitive))
            else:           # the sensitive feature is continuous
                zero = Literal('0')
                pivot = specification.readline().strip()
                literal = Literal(pivot)
                one = Literal('1')
                self.values = list()
                _value1 = dict()
                variable1 = VariableIdentifier('{}[<{}]'.format(self.sensitive[0], literal))
                lower = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, self.sensitive[0])
                upper = BinaryComparisonOperation(self.sensitive[0], BinaryComparisonOperation.Operator.Lt, literal)
                value1 = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
                _value1[self.sensitive[0]] = (0, eval(pivot))
                self.values.append((variable1, value1, tuple(_value1.items())))
                _value2 = dict()
                variable2 = VariableIdentifier('{}[>={}]'.format(self.sensitive[0], literal))
                lower = BinaryComparisonOperation(literal, BinaryComparisonOperation.Operator.LtE, self.sensitive[0])
                upper = BinaryComparisonOperation(self.sensitive[0], BinaryComparisonOperation.Operator.LtE, one)
                value2 = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
                _value2[self.sensitive[0]] = (eval(pivot), 1)
                self.values.append((variable2, value2, tuple(_value2.items())))
            # bound the sensitive feature between 0 and 1
            zero = Literal('0')
            one = Literal('1')
            left = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, self.sensitive[0])
            right = BinaryComparisonOperation(self.sensitive[0], BinaryComparisonOperation.Operator.LtE, one)
            self.bounds = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
            for sensitive in self.sensitive[1:]:
                left = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, sensitive)
                right = BinaryComparisonOperation(sensitive, BinaryComparisonOperation.Operator.LtE, one)
                conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
                self.bounds = BinaryBooleanOperation(self.bounds, BinaryBooleanOperation.Operator.And, conj)
            """
            determine the one-hot encoded uncontroversial features and fix their bounds
            """
            self.uncontroversial1 = list()
            while True:
                try:
                    arity = specification.readline().strip()
                    uncontroversial = list()
                    for i in range(int(arity)):
                        uncontroversial.append(VariableIdentifier(specification.readline().strip()))
                    self.uncontroversial1.append(uncontroversial)
                except ValueError:
                    break
            self.count = reduce(operator.mul, (len(encoding) for encoding in self.uncontroversial1), 1)
            # bound the one-hot encoded uncontroversial features between 0 and 1
            for encoding in self.uncontroversial1:
                for uncontroversial in encoding:
                    left = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, uncontroversial)
                    right = BinaryComparisonOperation(uncontroversial, BinaryComparisonOperation.Operator.LtE, one)
                    conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
                    self.bounds = BinaryBooleanOperation(self.bounds, BinaryBooleanOperation.Operator.And, conj)
            """
            determine the custom encoded uncontroversial features and fix their ranges
            """
            self.uncontroversial2 = list(inputs - set(self.sensitive) - set(itertools.chain(*self.uncontroversial1)))
            ranges: Dict[VariableIdentifier, Tuple[int, int]] = dict()
            for uncontroversial in self.uncontroversial2:
                ranges[uncontroversial] = (0, 1)
            # for uncontroversial in self.uncontroversial2:
            #     left = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, uncontroversial)
            #     right = BinaryComparisonOperation(uncontroversial, BinaryComparisonOperation.Operator.LtE, one)
            #     conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
            #     self.bounds = BinaryBooleanOperation(self.bounds, BinaryBooleanOperation.Operator.And, conj)
        self.outputs = outputs
        self.activations = activations
        cpu = self.cpu
        print('\nAvailable CPUs: {}'.format(cpu))
        colors = [
            Fore.LIGHTMAGENTA_EX,
            Back.BLACK + Fore.WHITE,
            Back.LIGHTRED_EX + Fore.BLACK,
            Back.MAGENTA + Fore.BLACK,
            Back.BLUE + Fore.BLACK,
            Back.CYAN + Fore.BLACK,
            Back.LIGHTGREEN_EX + Fore.BLACK,
            Back.YELLOW + Fore.BLACK,
        ]
        """
        do the pre-analysis
        """
        print(Fore.BLUE + '\n||==============||')
        print('|| Pre-Analysis ||')
        print('||==============||\n', Style.RESET_ALL)
        # prepare the queue
        queue1 = Manager().Queue()
        queue1.put((list(), (0, 0), self.startL, self.startU, 0, list(), list(ranges.items()), 0, list(self.uncontroversial2), 100, None))
        # run the pre-analysis
        start1 = time.time()
        processes = list()
        for i in range(cpu):
            color = colors[i % len(colors)]
            process = Process(target=self.worker1, args=(i, color, queue1, PyBoxMPQManager()))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        end1 = time.time()
        if self.autotuning:
            print(Fore.BLUE + "\nAutotuned to: L = {}, U = {}".format(self.lower.value, self.upper.value), Style.RESET_ALL)
        #
        patterns = len(self.patterns)
        discarded = self.discarded.value
        partitions = self.partitions.value
        considered = partitions - discarded
        print(Fore.BLUE + '\nFound: {} patterns for {}[{}] partitions'.format(patterns, considered, partitions))
        prioritized = sorted(self.patterns.items(), key=lambda v: len(v[1]), reverse=True)
        for key, pack in prioritized:
            sset = lambda s: '{{{}}}'.format(', '.join('{}'.format(e) for e in s))
            skey = ' | '.join('{}, {}'.format(sset(pair[0]), sset(pair[1])) for pair in key)
            print(skey, '->', len(pack))
        #
        expanded = dict()
        idx = 0
        for key, pack in self.patterns.items():
            for value in pack:
                expanded[idx] = (key, {value})
                idx += 1
        if len(expanded) > len(self.patterns):
            print('Expanded to: {} patterns'.format(len(expanded)))
        #
        # compressed = dict()
        # for key1, pack1 in sorted(self.patterns.items(), key=lambda v: len(v[1]), reverse=False):
        #     unmerged = True
        #     for key2 in compressed:
        #         mergeable1, mergeable2 = True, True
        #         for (s11, s12), (s21, s22) in zip(key1, key2):
        #             if (not s21.issubset(s11)) or (not s22.issubset(s12)):
        #                 mergeable1 = False
        #             if (not s11.issubset(s21)) or (not s12.issubset(s22)):
        #                 mergeable2 = False
        #         if mergeable1:
        #             unmerged = False
        #             compressed[key2] = compressed[key2].union(pack1)
        #             break
        #         if mergeable2:
        #             unmerged = False
        #             compressed[key1] = compressed[key2].union(pack1)
        #             del compressed[key2]
        #             break
        #     if unmerged:
        #         compressed[key1] = pack1
        # def max_disj(key):
        #     a1, i1 = key[0]
        #     a2, i2 = key[1]
        #     return max(len(self.activations) - len(a1) - len(i1), len(self.activations) - len(a2) - len(i2))
        # prioritized = sorted(compressed.items(), key=lambda v: max_disj(v[0]) + len(v[1]), reverse=True)
        # if len(compressed) < len(self.patterns):
        #     print('Compressed to: {} patterns'.format(len(compressed)))
        #     for key, pack in prioritized:
        #         sset = lambda s: '{{{}}}'.format(', '.join('{}'.format(e) for e in s))
        #         skey = ' | '.join('{}, {}'.format(sset(pair[0]), sset(pair[1])) for pair in key)
        #         print(skey, '->', len(pack))
        #
        result = '\nPre-Analysis Result: {}% fair ({}% feasible)'.format(self.fair.value, self.feasible.value)
        print(Fore.BLUE + result)
        print('Pre-Analysis Time: {}s'.format(end1 - start1), Style.RESET_ALL)

        """
        do the analysis
        """
        if analysis:
            print(Fore.BLUE + '\n||==========||')
            print('|| Analysis ||')
            print('||==========||\n', Style.RESET_ALL)
            # prepare the queue
            queue2 = Queue()
            for idx, idxkeypack in enumerate(expanded.items()):
                (_, (key, pack)) = idxkeypack
            # for idx, (key, pack) in enumerate(prioritized):
                queue2.put((idx+1, (key, pack)))
            queue2.put((None, (None, None)))
            # run the analysis
            start2 = time.time()
            processes = list()
            for i in range(cpu):
                color = colors[i % len(colors)]
                man = PyPolkaMPQstrictManager()
                process = Process(target=self.worker2, args=(i, color, queue2, man, len(expanded)))
                # process = Process(target=self.worker2, args=(i, color, queue2, man, len(compressed)))
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            end2 = time.time()
            #
            result = '\nResult: {}% of {}% ({}% biased)'.format(self.feasible.value, self.explored.value, self.biased.value)
            print(Fore.BLUE + result)
            print('Pre-Analysis Time: {}s'.format(end1 - start1))
            print('Analysis Time: {}s'.format(end2 - start2), Style.RESET_ALL)

            log = '{} ({}% certified in the pre-analysis, {}% biased) {}s {}s'.format(self.feasible.value, self.fair.value, self.biased.value, end1 - start1, end2 - start2)
        else:
            log = '{} ({}% certified in the pre-analysis) {}s'.format(self.feasible.value, self.fair.value, end1 - start1)
        print('\nDone!')
        return log


class BiasBackwardSemantics(DefaultBackwardSemantics):

    def assume_call_semantics(self, stmt, state, manager: PyManager = None) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument, manager=manager)
        state.result = set()
        return state

    def list_semantics(self, stmt, state) -> State:
        state.polka = state.polka.substitute(stmt[0], stmt[1])
        return state

    def ReLU_call_semantics(self, stmt, state, manager: PyManager = None, active: bool = True) -> State:
        assert manager is not None
        if active:  # assume h >= 0
            expr = PyTexpr1.var(state.environment, stmt)
            cond = PyTcons1.make(expr, ConsTyp.AP_CONS_SUPEQ)
            abstract1 = PyPolka(manager, state.environment, array=PyTcons1Array([cond]))
            state.polka = state.polka.meet(abstract1)
            return state
        else:  # assign h = 0, assume h < 0
            expr = PyTexpr1.var(state.environment, stmt)
            zero = PyTexpr1.cst(state.environment, PyMPQScalarCoeff(0.0))
            neg = PyTexpr1.binop(TexprOp.AP_TEXPR_SUB, zero, expr, rtype, rdir)
            cond = PyTcons1.make(neg, ConsTyp.AP_CONS_SUP)
            abstract1 = PyPolka(manager, state.environment, array=PyTcons1Array([cond]))
            zero = PyTexpr1.cst(state.environment, PyMPQScalarCoeff(0.0))
            state.polka = state.polka.substitute(stmt, zero).meet(abstract1)
            return state
