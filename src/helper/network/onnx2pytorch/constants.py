from torch.nn.modules.conv import _ConvNd
from torch import nn

from .operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    Loop,
    LSTMWrapper,
    Split,
    TopK,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (Loop, LSTMWrapper, Split, TopK)
STANDARD_LAYERS = (
    _ConvNd,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)
