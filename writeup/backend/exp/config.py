'''Shared settings of the experiment and the plots.'''

DATA_DIR = 'writeup/backend/data'  # raw results, not shown by the site
CSV_DIR = f'{DATA_DIR}/csv'  # one {search}_{abstraction}.csv per config
# files the website reads (the frontend is a Hugo site)
SITE_DIR = 'writeup/frontend'
FIGURE_DIR = f'{SITE_DIR}/static/figure'
IMG_DIR = f'{SITE_DIR}/static/img'
SUMMARY_PATH = f'{SITE_DIR}/data/summary.json'
CONTENT_DIR = f'{SITE_DIR}/content'

SEARCHES = ('bab', 'dpll')
ABSTRACTIONS = ('ibp', 'zonotope', 'crown', 'alphacrown', 'neuralsat')

THREADS = 23  # threads per run; see CLAUDE.md (Resource limits)
N_INSTANCES = 40
TIMEOUT = 2.0  # seconds per run

# instance generator: random FC + ReLU nets with convex input constraints
SIZES = (8, 24, 24, 3)
FACTORS = (1.02, 1.05, 1.1, 1.2)  # eps as a multiple of the CROWN radius
