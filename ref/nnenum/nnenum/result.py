'''Enumeration result object definition

This defines the object returned by enumerate_network
'''

import multiprocessing

from nnenum.util import Freezable

class Result(Freezable):
    'computation result object'

    manager = multiprocessing.Manager()

    # possible result strings in result_str
    results = ["none", "error", "timeout", "safe", "unsafe (unconfirmed)", "unsafe"]

    def __init__(self, nn, quick=False):
        # result string, one of Result.results
        # can be safe/unsafe only if a spec is provided to verification problem
        # "unsafe (unconfirmed)" means that the output set appeared to violate the spec, but no concrete trace
        # could be found, which can happen due to numerical accuracy in LP solving and during network execution
        self.result_str = "none"

        # total verification time, in seconds
        self.total_secs = None

        # total number of times LP solver was called during enumeration (statistic)
        self.total_lps_enum = 0

        # total number of times LP solver was called during enumeration and verification / plotting (statistic)
        self.total_lps = 0

        # total number of stars explored during path enumeration
        self.total_stars = 0

        # data (3-tuple) about problem progress: (finished_stars, unfinished_stars, finished_work_frac)
        self.progress_tuple = (0, 0, 0)

        ##### assigned if cls.RESULT_SAVE_TIMERS is nonempty. Map of timer_name -> total_seconds
        self.timers = {}

        if not quick:
            ###### assigned if Settings.RESULT_SAVE_POLYS = True. Each entry is polygon (list of 2-d points), ######
            self.polys = Result.manager.list()

            ###### assigned if Settings.RESULT_SAVE_STARS = True. Each entry is an LpStar ######
            self.stars = Result.manager.list()

            ###### below are assigned used if spec is not None and property is unsafe ######
            # counter-example boolean flags
            self.found_counterexample = multiprocessing.Value('i', 0)
            self.found_confirmed_counterexample = multiprocessing.Value('i', 0) # found counter-example with concrete input

            # concrete counter-example input and output
            self.coutput = multiprocessing.Array('d', nn.get_num_outputs())
            self.cinput = multiprocessing.Array('d', nn.get_num_inputs())
        else:
            # types may be different hmmm...
            self.polys = None
            self.stars = None
            self.found_counterexample = 0
            self.found_confirmed_counterexample = 0
            self.cinput = None
            self.coutput = None

        self.freeze_attrs()
