from collections import namedtuple

AbstractResults = namedtuple('AbstractResults', ('output_lbs', 
                                                 'masks', 'lAs', 'histories', 
                                                 'lower_bounds', 'upper_bounds', 
                                                 'input_lowers', 'input_uppers',
                                                 'slopes', 'betas', 
                                                 'cs', 'rhs',
                                                 'sat_solvers'), 
                             defaults=(None,) * 13)


class ReturnStatus:

    UNSAT   = 'UNSAT'
    SAT     = 'SAT'
    UNKNOWN = 'UNKNOWN'
    TIMEOUT = 'TIMEOUT'
    RESTART = 'RESTART'
