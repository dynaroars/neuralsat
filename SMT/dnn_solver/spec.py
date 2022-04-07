from dnn_solver.utils import DNFConstraint
import itertools
import random
import torch

def get_acasxu_bounds(p):
    if p == 0:
        return { # debug
            'lbs' : [-30, -30, -30],
            'ubs' : [30, 30, 30],
        }

    if p == 1:
        return {
            'lbs' : [0.6,        -0.5, -0.5, 0.45, -0.5],
            'ubs' : [0.679857769, 0.5,  0.5, 0.5,  -0.45],
        }

    if p == 2:
        return {
            'lbs' : [0.6,        -0.5, -0.5, 0.45, -0.5],
            'ubs' : [0.679857769, 0.5,  0.5, 0.5,  -0.45],
        } 

    if p == 3:
        return {
            'lbs' : [-0.303531156, -0.009549297, 0.493380324, 0.3, 0.3],
            'ubs' : [-0.298552812,  0.009549297, 0.5,         0.5, 0.5],
        }

    if p == 4:
        return {
            'lbs' : [-0.303531156, -0.009549297, 0.0, 0.318181818, 0.083333333],
            'ubs' : [-0.298552812,  0.009549297, 0.0, 0.5,         0.166666667],
        }

    if p == 5:
        return {
            'lbs' : [-0.324274257, 0.031830989, -0.499999896, -0.5,         -0.5],
            'ubs' : [-0.321785085, 0.063661977, -0.499204121, -0.227272727, -0.166666667],
        }

    if p == 7:
        return {
            'lbs' : [-0.328422877, -0.499999896, -0.499999896, -0.5, -0.5],
            'ubs' : [ 0.679857769,  0.499999896,  0.499999896,  0.5,  0.5],
        }

    if p == 8:
        return {
            'lbs' : [-0.328422877, -0.499999896, -0.015915494, -0.045454545, 0.0],
            'ubs' : [ 0.679857769, -0.374999922,  0.015915494,  0.5,         0.5],
        }

    if p == 9:
        return {
            'lbs' : [-0.295233916, -0.063661977, -0.499999896, -0.5,         -0.5],
            'ubs' : [-0.212261512, -0.022281692, -0.498408347, -0.454545455, -0.375],
        }

    if p == 10:
        return {
            'lbs' : [0.268978427, 0.11140846,  -0.499999896, 0.227272727, 0.0],
            'ubs' : [0.679857769, 0.499999896, -0.498408347, 0.5,         0.5],
        }

    raise NotImplementedError


def split_bounds(bounds, steps=3):
    lower = bounds['lbs']
    upper = bounds['ubs']

    bs = [(l, u) for l, u in zip(lower, upper)]
    bs = [torch.linspace(b[0], b[1], steps=steps) for b in bs]
    bs = [[torch.Tensor([b[i], b[i+1]]) for i in range(b.shape[0] - 1)] for b in bs]
    bs = itertools.product(*bs)
    splits = [{'lbs': torch.Tensor([_[0] for _ in b]),
               'ubs': torch.Tensor([_[1] for _ in b])} for b in bs]
    random.shuffle(splits)
    return splits

class Specification:

    def __init__(self, p, bounds):
        self.lower = bounds['lbs']
        self.upper = bounds['ubs']
        self.p = p

    def get_output_property(self, output):
        if self.p == 0: # debug
            return output[0] >= 1e-6

        if self.p == 1: # Unsafe if COC >= 1500. (1500 - 7.518884) / 373.94992 = 3.991125
            return output[0] >= 3.991125645861615

        if self.p == 2: # Unsafe if COC is maximal
            return [output[0] >= output[1],
                    output[0] >= output[2],
                    output[0] >= output[3],
                    output[0] >= output[4]]

        if self.p == 3 or self.p == 4: # Unsafe if COC is minimal
            return [output[0] <= output[1],
                    output[0] <= output[2],
                    output[0] <= output[3],
                    output[0] <= output[4]]

        if self.p == 5: # unsafe if strong right is not minimal
            return DNFConstraint([ # or
                [output[0] <= output[4]], # and
                [output[1] <= output[4]], # and
                [output[2] <= output[4]], # and
                [output[3] <= output[4]], # and
            ]) 

        if self.p == 7: # unsafe if strong left is minimial or strong right is minimal
            return DNFConstraint([ # or
                [ # and
                    output[3] <= output[0],
                    output[3] <= output[1],
                    output[3] <= output[2],
                ], 
                [ # and
                    output[4] <= output[0],
                    output[4] <= output[1],
                    output[4] <= output[2],

                ], 
            ])

        if self.p == 8: # weak left is minimal or COC is minimal
            return DNFConstraint([ # or
                [ # and
                    output[2] <= output[0],
                    output[2] <= output[1],
                ], 
                [ # and
                    output[3] <= output[0],
                    output[3] <= output[1],
                ], 
                [ # and
                    output[4] <= output[0],
                    output[4] <= output[1],
                ], 
            ])

        if self.p == 9: # strong left should be minimal
            return DNFConstraint([ # or
                [output[0] <= output[3]], 
                [output[1] <= output[3]], 
                [output[2] <= output[3]], 
                [output[4] <= output[3]], 
            ]) 


        if self.p == 10: # unsafe if coc is not minimal
            return DNFConstraint([ # or
                [output[1] <= output[0]], 
                [output[2] <= output[0]], 
                [output[3] <= output[0]], 
                [output[4] <= output[0]], 
            ]) 

        raise NotImplementedError


    def check_output_reachability(self, lbs, ubs):
        if self.p == 0:
            return ubs[0] >= 1e-6

        if self.p == 1:
            return ubs[0] >= 3.991125645861615

        if self.p == 2:
            return all([ubs[0] >= lbs[1],
                        ubs[0] >= lbs[2],
                        ubs[0] >= lbs[3],
                        ubs[0] >= lbs[4]])
            
        if self.p == 3 or self.p == 4:
            return all([ubs[1] >= lbs[0],
                        ubs[2] >= lbs[0],
                        ubs[3] >= lbs[0],
                        ubs[4] >= lbs[0]])

        if self.p == 5:
            return any([ubs[4] >= lbs[0],
                        ubs[4] >= lbs[1],
                        ubs[4] >= lbs[2],
                        ubs[4] >= lbs[3]])

        if self.p == 7:
            return any([ # or
                all([ubs[0] >= lbs[3], # and
                     ubs[1] >= lbs[3],
                     ubs[2] >= lbs[3]]
                ), 
                all([ubs[0] >= lbs[4], # and
                     ubs[1] >= lbs[4],
                     ubs[2] >= lbs[4]]
                ),
            ])

        if self.p == 8:
            return any([ # or
                all([ubs[0] >= lbs[2], # and
                     ubs[1] >= lbs[2]]
                ), 
                all([ubs[0] >= lbs[3], # and
                     ubs[1] >= lbs[3]]
                ), 
                all([ubs[0] >= lbs[4], # and
                     ubs[1] >= lbs[4]]
                ),
            ])

        if self.p == 9:
            return any([ubs[3] >= lbs[0],
                        ubs[3] >= lbs[1],
                        ubs[3] >= lbs[2],
                        ubs[3] >= lbs[4]])

        if self.p == 10:
            return any([ubs[0] >= lbs[1],
                        ubs[0] >= lbs[2],
                        ubs[0] >= lbs[3],
                        ubs[0] >= lbs[4]])

        raise NotImplementedError
