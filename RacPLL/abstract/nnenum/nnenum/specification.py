'''
Stanley Bak
Specification container object
'''

import time

import numpy as np

from nnenum.util import Freezable
from nnenum.timerutil import Timers

class DisjunctiveSpec(Freezable):
    'disjunctive specification'

    def __init__(self, spec_list):
        # unsafe is any of the specs is unsafe

        assert spec_list, "spec_list must have > 0 Specficiation objects"

        for spec in spec_list:
            assert isinstance(spec, Specification), "expected flat (non-nested) DisjunctiveSpec"

        self.spec_list = spec_list

        # sanity check expected variables matches
        expected_vars = self.spec_list[0].get_num_expected_variables()

        for spec in spec_list[1:]:
            assert spec.get_num_expected_variables() == expected_vars

        self.freeze_attrs()

    def __str__(self):
        s = ""

        for spec in self.spec_list:
            if s:
                s += " or "

            s += str(spec)

        return s

    def get_num_expected_variables(self):
        'get the number of expected variables for this spec'

        return self.spec_list[0].get_num_expected_variables()

    def is_violation(self, state):
        'does this concrete state violate the specification?'

        rv = False

        for spec in self.spec_list:
            if spec.is_violation(state):
                rv = True
                break

        return rv
    
    def distance(self, state):
        '''get the minimum distance (l-1 norm) between this state and the boundary of the unsafe states
        0 = violation
        '''

        return min([spec.distance(state) for spec in self.spec_list])

    def zono_might_violate_spec(self, zono):
        '''is it possible that the zonotope violates the spec?

        returns True or False
        '''

        # strategy: check if each row individually can have a violation... necessary condition for intersection

        rv = False

        for spec in self.spec_list:
            if spec.zono_might_violate_spec(zono):
                rv = True
                break

        return rv

    def get_violation_star(self, lp_star, safe_spec_list=None, normalize=True, domain_contraction=True):
        '''does this lp_star violate the spec?

        if so, return a new, non-empty star object with the violation region
        '''

        Timers.tic('disjunctive.get_violation_star')

        res = None

        for i, spec in enumerate(self.spec_list):
            if safe_spec_list is not None and safe_spec_list[i]:
                # skip parts of the disjunctive spec that are already safe
                continue
            
            res = spec.get_violation_star(lp_star, normalize=normalize, domain_contraction=domain_contraction)

            if res is not None:
                break
            
        Timers.toc('disjunctive.get_violation_star')

        return res            

class Specification(Freezable):
    'specification container'

    def __init__(self, mat, rhs):
        # unsafe if there is some state where, mat * state <= rhs

        if not isinstance(mat, np.ndarray):
            assert isinstance(rhs, list)
            mat = np.array(mat, dtype=float)

        if not isinstance(rhs, np.ndarray):
            assert isinstance(rhs, list)
            rhs = np.array(rhs, dtype=float)

        self.mat = mat
        self.rhs = rhs

        assert len(rhs.shape) == 1
        assert len(rhs) == mat.shape[0]

    def __str__(self):

        s = ""

        for i, row in enumerate(self.mat):
            if s:
                s += " & "

            s += f"{row} <= {self.rhs[i]}"

        return s

    def get_num_expected_variables(self):
        'get the number of expected variables for this spec'

        return self.mat.shape[1]

    def is_violation(self, state, tol_rhs=0.0):
        'does this concrete state violate the specification?'

        res = np.dot(self.mat, state)

        rv = True

        for got, ub in zip(res, self.rhs):
            if got > ub + tol_rhs:
                rv = False
                break

        return rv

    def distance(self, state):
        '''get the minimum distance (l-inf norm) between this state and the boundary of the unsafe states
        0 = violation
        '''

        res = np.dot(self.mat, state)
        rv = -np.inf

        for got, ub in zip(res, self.rhs):
            rv = max(rv, got - ub)

        return rv

    def zono_might_violate_spec(self, zono):
        '''is it possible that the zonotope violates the spec?

        sometimes we can prove it's impossible. If this returns True, though, it doesn't mean there's an
        intersection (except in the case of single-row specifications)

        returns True or False
        '''

        # strategy: check if each row individually can have a violation... necessary condition for intersection

        Timers.tic('zono_might_violate_spec')

        might_violate = True

        for i, row in enumerate(self.mat):
            min_dot = zono.minimize_val(row)
            
            if min_dot > self.rhs[i]:
                might_violate = False
                break

        Timers.toc('zono_might_violate_spec')

        return might_violate

    def get_violation_star(self, lp_star, safe_spec_list=None, normalize=True, domain_contraction=True):
        '''does this lp_star violate the spec?

        if so, return a new, non-empty star object with the violation region
        '''

        assert safe_spec_list is None, "single spec doesn't expect safe_spec_list"

        Timers.tic('get_violation_star')
        rv = None

        # constructing a new star and do exact check
        copy = lp_star.copy()
        
        # add constraints on the outputs

        # output = a_mat.tranpose * input_col

        # a_mat.transpose * self.mat.transpose
        # same as (self.mat * a_mat).transpose

        init_spec = np.dot(self.mat, copy.a_mat)
        lpi = copy.lpi

        init_bias = np.dot(self.mat, copy.bias)
        hs_list = []
        rhs_list = []

        for i, row in enumerate(init_spec):
            rhs = self.rhs[i] - init_bias[i]
            hs_list.append(row)
            rhs_list.append(rhs)
            lpi.add_dense_row(row, rhs, normalize=normalize)

        winput = lpi.minimize(None, fail_on_unsat=False)

        if winput is None:
            # when we check all the specification directions at the same time, there is no violaton
            is_violation = False
        else:
            is_violation = True
            rv = copy
            #woutput = np.dot(copy.a_mat, winput) + copy.bias
            #assert self.is_violation(woutput), f"witness output {woutput} was not a violation of {self}"

            # also comput input box bounds
            if domain_contraction:
                Timers.tic('violation_update_input_box_bounds')
                rv.update_input_box_bounds(hs_list, rhs_list)
                Timers.toc('violation_update_input_box_bounds')

        Timers.toc('get_violation_star')

        return rv if is_violation else None
