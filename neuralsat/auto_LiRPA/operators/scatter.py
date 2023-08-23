from .base import *

class BoundScatterND(Bound):
    
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, *x):
        raise

    def bound_backward(self, last_lA, last_uA, x):
        raise
        exit(1)

class BoundReduceProd(Bound):
    
    def __init__(self, attr, inputs, output_index, options):
        self.device = attr['device']
        self.keepdims = attr['keepdims']
        if isinstance(self.keepdims, int):
            self.keepdims = self.keepdims == 1
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        raise
        # print('[BoundReduceProd]', x)
        # final = torch.prod(x)
        # print('[BoundReduceProd]', final)
        # return final
        
    def bound_backward(self, last_lA, last_uA, x):
        raise

