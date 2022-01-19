from solver.theory_solver import TheorySolver
from simplex.simplex import Simplex
from simplex.parser import Parser
from pprint import pprint
import numpy as np
import copy

from sat_solver.sat_solver_2 import SATSolver2
from real_solver.real_solver import RealSolver
from dnn_solver.helpers import Utils
from solver.solver import Solver

class TheorySolver(Solver):

    def __init__(self, formula, meta, first_var=None, max_new_clauses=float('inf'), halving_period=10000):
        super().__init__()

        self._solver = SATSolver2(formula,
                                  meta=meta,
                                  first_var=first_var,
                                  max_new_clauses=max_new_clauses,
                                  halving_period=halving_period,
                                  theory_solver=self)

    def get_assignment(self) -> dict:
        pass

    def solve(self) -> bool:
        return self._solver.solve()




class DNNSolver(TheorySolver):

    def __init__(self, meta, conditions):


        # self.parsed_input = Parser.parse(formula_str)
        # print(self.parsed_input.formula)
        # print(self.parsed_input.cnf)
        # pprint(self.parsed_input.vars_dict)
        # xi_int = self.parsed_input.vars_dict['xi']

        super().__init__(formula=None, meta=meta)

        # for clause in self._non_boolean_clauses:
        #     self._c = np.zeros(len(clause[1]), dtype=np.float64)
        #     break

        # self._tseitin_variable_to_np = {}
        # for clause in self._non_boolean_clauses:
        #     self._tseitin_variable_to_np[subterm_to_tseitin_variable[clause]] = {
        #         True: (np.array(clause[1], dtype=np.float64), np.array(clause[2], dtype=np.float64)),
        #         False: (-np.array(clause[1], dtype=np.float64), -np.array(clause[2] + epsilon, dtype=np.float64))
        #     }

        # pprint(self._tseitin_variable_to_np)
        self.conditions = conditions
        self.meta = meta

    def propagate(self):
        print('- Theory propagate\n')

        # print('_mapping_vars', self._solver._mapping_vars)
        # print('_assignment', self._solver._assignment)
        input_conds = self.conditions['in']
        output_conds = self.conditions['out']

        new_assignments = []
        conflict_clause = set()

        in_conds_tmp = copy.deepcopy(input_conds)
        in_conds_tmp2 = copy.deepcopy(input_conds)
        for key, value in self.meta.items():
            # print(key, value)
            if key.startswith('a'):
                in_conds_tmp = Utils.And(in_conds_tmp, f'({key} = {value[1]})')
                if self._solver._mapping_vars[key] in self._solver._assignment:
                    if self._solver._assignment[self._solver._mapping_vars[key]]['value']:
                        in_conds_tmp = Utils.And(in_conds_tmp, f'(n{key[1:]} = {value[1]})')
                        in_conds_tmp2 = Utils.And(in_conds_tmp2, f'(and (n{key[1:]} >= 0) (n{key[1:]} = {value[1]}))')
                    else:
                        in_conds_tmp = Utils.And(in_conds_tmp, f'(n{key[1:]} = 0)')
                        in_conds_tmp2 = Utils.And(in_conds_tmp2, f'(n{key[1:]} = 0)')
            else:
                in_conds_tmp = Utils.And(in_conds_tmp, f'({key} = {value})')
                in_conds_tmp2 = Utils.And(in_conds_tmp2, f'({key} = {value})')

        stat = RealSolver(Utils.Prove(in_conds_tmp2, Utils.Not(output_conds))).solve()
        # print(stat)
        if not stat[0]:
            print('    - Check T-SAT: `UNSAT`')
            for variable, value in self._solver.iterable_assignment():
                conflict_clause.add(-variable if value else variable)
            conflict_clause = frozenset(conflict_clause)
            print(f'    - Conflict clause: `{list(conflict_clause)}`')
            print()
            return conflict_clause, []

        print('    - Check T-SAT: `SAT`')
        conflict_clause = None
        # deduce next layers
        tmp = copy.deepcopy(in_conds_tmp)
        for key, value in self.meta.items():
            if key.startswith('a'):
                if self._solver._mapping_vars[key] in self._solver._assignment:
                    if self._solver._assignment[self._solver._mapping_vars[key]]['value']:
                        tmp = Utils.And(tmp, f'({key} > 0)')
                    else:
                        tmp = Utils.And(tmp, f'({key} <= 0)')
                    continue

                # print(Utils.Prove(tmp, f'({key} < 0)'))
                stat_neg = RealSolver(Utils.Prove(tmp, f'({key} <= 0)')).solve()
                if not stat_neg[0]:

                    aa = Utils.Prove(tmp, f'({key} <= 0)')
                    print(f'    - Constraints: `{aa}`')
                    print(f'    - Deduction: `{key} <= 0`')
                    new_assignments.append(-self._solver._mapping_vars[key])
                    continue

                stat_pos = RealSolver(Utils.Prove(tmp, f'({key} > 0)')).solve()
                if not stat_pos[0]:
                    aa = Utils.Prove(tmp, f'({key} > 0)')
                    print(f'    - Constraints: `{aa}`')
                    print(f'    - Deduction: `{key} > 0`')
                    # print(f'[+] deducing:\n{key} >= 0')
                    new_assignments.append(self._solver._mapping_vars[key])

        print(f'    - New assignment: `{new_assignments}`')
        print()
        return conflict_clause, new_assignments




    def get_assignment(self) -> dict:
        pass
