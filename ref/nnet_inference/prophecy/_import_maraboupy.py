import os
import platform
import sys
import warnings

__all__ = [
    'MARABOU_ROOT',
    'import_marabou'
]

MARABOU_ROOT = ''

# default settings, put your machine's path here
if platform.node() == 'Server2':
    MARABOU_ROOT = '/home/nhatth/code/dynaroars/Marabou'


def import_marabou(marabou_root=MARABOU_ROOT):
    if marabou_root == '':
        warnings.warn('Marabou not found, please install following '
                      'https://github.com/NeuralNetworkVerification/Marabou, '
                      'and then specify the path import_marabou(PATH_TO_MARABOU). '
                      'In which PATH_TO_MARABOU contains maraboupy folder.')
    elif marabou_root is None or not os.path.isdir(marabou_root):
        raise ValueError(f'Specified path to Marabou {marabou_root} is invalid.')
    sys.path.append(os.path.abspath(marabou_root))
    return __import__('maraboupy')
