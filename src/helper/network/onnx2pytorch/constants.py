from torch.nn.modules.conv import _ConvNd
from torch import nn

from .operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    Loop,
    Split,
    TopK,
)
from .convert.layer import LSTMUnrolledImpl, GRUUnrolledImpl


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (Loop, LSTMUnrolledImpl, GRUUnrolledImpl, Split, TopK)
STANDARD_LAYERS = (
    _ConvNd,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMUnrolledImpl,
    GRUUnrolledImpl,
    nn.Linear,
)
