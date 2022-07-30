'''
Utilities

Stanley Bak, 2018
'''

from collections import deque
import queue

import numpy as np
from threadpoolctl import threadpool_info

def check_openblas_threads():
    'make sure openblas is running single-threaded'

    d = threadpool_info()

    #pprint(d)
    assert len(d) > 0, f"numpy didn't use blas api?: {d}"

    for entry in d:
        assert entry['num_threads'] == 1, "expected single thread libraries (export OPENBLAS_NUM_THREADS=1 and " + \
            "export OMP_NUM_THREADS=1 or use Settings.CHECK_SINGLE_THREAD_BLAS" + \
            f" = False to skip): {entry}\nAll entries:{d}"

class Freezable():
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise AttributeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

class FreezableMeta(type):
    'metaclass used to initialize and prevent new settings after initialization. class must define reset().'

    def __new__(mcs, name, bases, dct):
        rv = type.__new__(mcs, name, bases, dct)
        rv._FROZEN = False
        rv.reset() # freezable class must define reset() classmethod
        rv._FROZEN = True

        return rv

    def __setattr__(cls, key, value):
        if key != "_FROZEN" and cls._FROZEN and not hasattr(cls, key):
            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{key}' (object was frozen)")

        return super(FreezableMeta, cls).__setattr__(key, value)

class FakeQueue(Freezable):
    'a queue class wrapper with the same interface as multiprocessing.queue but no sync'

    def __init__(self):

        self.d = deque()

        self.freeze_attrs()

    def put(self, item):
        'add an item to the queue'

        self.d.append(item)

    def get(self, block=True, timeout=None):
        'get an item from the queue'

        if self.d:
            rv = self.d.popleft()
        else:
            raise queue.Empty()

        return rv

    def qsize(self):
        'queue size'

        return len(self.d)

def pt_almost_in(pt, pt_list, tol=1e-9):
    'check if a pt is in a pt list (up to small tolerance)'

    rv = False

    for existing_pt in pt_list:
        if np.allclose(existing_pt, pt, atol=tol):
            rv = True
            break

    return rv

def are_verts_equal(verts, check_list, tol=1e-5):
    '''check that the two lists of vertices are the same

    returns a boolean
    '''

    rv = True

    for v in check_list:
        if not pt_almost_in(v, verts, tol):
            rv = False
            break

    if rv:
        for v in verts:
            if not pt_almost_in(v, check_list, tol):
                rv = False
                break

    return rv

def assert_verts_equals(verts, check_list, tol=1e-5):
    '''check that the two lists of vertices are the same using asserts'''

    for v in check_list:
        assert pt_almost_in(v, verts, tol), "{} was not found in verts: {}".format(v, verts)

    for v in verts:
        assert pt_almost_in(v, check_list, tol), "verts contains {}, which was not in check_list: {}".format(
            v, check_list)

def assert_verts_is_box(verts, box, tol=1e-5):
    '''check that a list of verts is almost equal to the passed-in box using assertions

    box is [[xmin, xmax], [ymin, ymax]]
    '''

    pts = [(box[0][0], box[1][0]), (box[0][1], box[1][0]), (box[0][1], box[1][1]), (box[0][0], box[1][1])]

    assert_verts_equals(verts, pts, tol)

def compress_init_box(input_box, tol=1e-9):
    '''compress all constant values in the init set into a single input

    returns init_bm, init_bias, new_input_box
    '''

    inputs = len(input_box)

    dtype = type(input_box[0][0])
    assert dtype in [float, np.float64, np.float32], f"input_box dtype should be float32/64, got {dtype}"
    cur_bias = np.array([0] * inputs, dtype=dtype)

    cur_bm_transpose = []
    new_input_box = []

    for dim, (lb, ub) in enumerate(input_box):
        mid = (lb + ub) / 2.0

        if abs(ub-lb) < tol:
            # equal, update cur_bias
            cur_bias[dim] = mid
        else:
            new_input_box.append((lb, ub))

            # add column from identity matrix to cur_bm
            cur_bm_transpose.append([1 if d == dim else 0 for d in range(inputs)])

    cur_bm = np.array(cur_bm_transpose, dtype=dtype).transpose()

    return cur_bm, cur_bias, new_input_box

def to_time_str(secs):
    'return a string representation of the number of seconds'

    divisors = [1, 60, 60*60, 24*60*60, 7*24*60*60, 365*24*60*60, np.inf]
    labels = ["sec", "min", "hr", "days", "weeks", "years"]
    bounds = divisors[1:]
    digits = [1, 2, 3, 4, 4, 4]
    time_str = ""

    for divisor, digit, label, bound in zip(divisors, digits, labels, bounds):
        if secs < bound:
            time_str = f"{round(secs / divisor, digit)} {label}"
            break

    return time_str
