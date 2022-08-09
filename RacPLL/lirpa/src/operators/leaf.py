from .base import Bound

class BoundedInput(Bound):

    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name)
        self.value = value
        self.perturbation = perturbation
        self.from_input = True



class BoundedBuffers(BoundedInput):
    
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_buffer('buffer', value.clone().detach())

    @Bound.save_io_shape
    def forward(self):
        return self.buffer




class BoundedParams(BoundedInput):

    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_parameter('param', value)
        self.from_input = False
        self.initializing = False

    def register_parameter(self, name, param):
        "Override `register_parameter()` hook to register only needed parameters."

        if name == 'param':
            # self._parameters[name] = param  # cannot contain '.' in name, it will cause error when loading state_dict
            return super().register_parameter(name, param)
        else:
            # Just register it as a normal property of class.
            object.__setattr__(self, name, param)

    def init(self, initializing=False):
        self.initializing = initializing

    @Bound.save_io_shape
    def forward(self):
        if self.initializing:
            return self.param_init
        else:
            return self.param

    def infer_batch_dim(self, batch_size, *x):
        return -1