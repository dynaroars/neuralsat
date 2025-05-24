from .base import *

class BoundShape(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.never_perturbed = True

    @staticmethod
    def shape(x):
        return x.shape if isinstance(x, Tensor) else torch.tensor(x).shape

    def forward(self, x):
        self.from_input = False
        return BoundShape.shape(x)

    def bound_forward(self, dim_in, x):
        return self.forward_value

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if not isinstance(v[0], Tensor):
            # e.g., v[0] input shape (8, 7, 7) => output its shape (1, 8, 7, 7)
            gvars_array = np.array(v[0])
            self.solver_vars = torch.tensor(np.expand_dims(gvars_array, axis=0).shape).long()
        else:
            self.solver_vars = torch.tensor(self.forward(v[0])).long()
