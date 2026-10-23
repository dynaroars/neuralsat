'''Shared settings of the experiment and the plots.'''

DATA_DIR = 'writeup/data'
CSV_DIR = f'{DATA_DIR}/csv'  # one {search}_{abstraction}.csv per config
FIGURE_DIR = f'{DATA_DIR}/figure'

SEARCHES = ('bab', 'dpll')
ABSTRACTIONS = ('crown', 'neuralsat')

THREADS = 23  # threads per run; see CLAUDE.md (Resource limits)
N_INSTANCES = 40
TIMEOUT = 2.0  # seconds per run

# instance generator: random FC + ReLU nets with convex input constraints
SIZES = (8, 24, 24, 3)
FACTORS = (1.02, 1.05, 1.1, 1.2)  # eps as a multiple of the CROWN radius
