import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


class Pad(Operator):
    def __init__(self, mode="constant", padding=None, constant=0):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        assert input.ndim == 4
        if len(pads) > 4:
            # pads should be [0,0,pad_top,pad_left,0,0,pad_bottom,pad_right]
            assert pads[0] == pads[1] == pads[4] == pads[5] == 0
            pads = [int(pads[3]), int(pads[7]), int(pads[2]), int(pads[6])]
        out = F.pad(input, list(pads), mode=self.mode, value=value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
