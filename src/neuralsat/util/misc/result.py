from collections import namedtuple


AbstractResults = namedtuple(
    'AbstractResults', 
    (
        'objective_ids',
        'output_lbs', 
        'masks', 'lAs', 'histories', 
        'lower_bounds', 'upper_bounds', 
        'input_lowers', 'input_uppers',
        'slopes', 'betas', 
        'cs', 'rhs',
        'sat_solvers',
        'last_decisions',
    ), 
    defaults=(None,) * 15
)


class ReturnStatus:

    UNSAT   = 'unsat'
    SAT     = 'sat'
    UNKNOWN = 'unknown'
    TIMEOUT = 'timeout'
    RESTART = 'restart'
