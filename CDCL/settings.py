# constant
SUPPORTED_DECIDER = ['ORDERED', 'VSIDS', 'MINISAT']
SUPPORTED_RESTARTER = ['GEOMETRIC', 'LUBY']

# variable for decider
DECIDER = 'ORDERED'

INCREASE = 1 # Score increases when literal is found in a conflict clause
DECAY = 0.85 # Scores decays after each conflict

# variable for restarter
RESTARTER = None

CONFLICT_LIMIT = 512
LUBY_BASE = 512
LIMIT_MULT = 2 # Conflict limit will be multiplied after each restart

# logging
DEBUG = False