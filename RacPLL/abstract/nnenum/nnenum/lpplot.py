'''
Stanley Bak
Functions related to plotting a set of lp constriants

get_verts is probably the main one to use

make_plot_vecs is useful for controlling the accuracy (and decreasing overhead compared w/ not passing it to get_verts)
'''

import numpy as np

import nnenum.kamenev as kamenev
from nnenum.timerutil import Timers

def get_verts_nd(lpi, dims):
    '''
    get an the n-dimensional vertices

    if dims is an int, this uses the first dim coordinates of the lpi
    '''

    assert isinstance(dims, list), f"unsupported dims type: {type(dims)}"
    dim_list = dims

    for dim in dim_list:
        assert dim < lpi.dims, f"lpi has {lpi.dims} dims, but requested dim_list was {dim_list}"

    # algorithm: Kamenev's method in n-d

    def supp_point_nd(vec):
        'return a supporting point for the given direction (maximize)'

        assert len(vec) == len(dim_list)
        assert lpi.dims > 0

        Timers.tic('construct')
        d = np.zeros((lpi.dims,), dtype=float)
        # negative here because we want to MAXIMIZE not minimize

        for i, dim_index in enumerate(dim_list):
            d[dim_index] = -vec[i]

        Timers.toc('construct')

        Timers.tic('set_minimize_dir')
        lpi.set_minimize_direction(d)
        Timers.toc('set_minimize_dir')

        Timers.tic('lpi.minimize')
        res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(lpi.dims)])
        Timers.toc('lpi.minimize')

        Timers.tic('make res')
        rv = []

        for dim in dim_list:
            rv.append(res[dim])
        
        rv = np.array(rv, dtype=float)
        Timers.toc('make res')

        return rv

    Timers.tic('kamenev.get_verts')
    verts = kamenev.get_verts(len(dim_list), supp_point_nd)
    Timers.toc('kamenev.get_verts')

    return verts
