"""
forward runner to toy example
"""
import sys

from apronpy.var import PyVar
from libra.core.cfg import Activation

from libra.engine.forward_runner import ForwardAnalysis
from libra.engine.bias_analysis import AbstractDomain
from libra.main import checker

spec = 'tests/census/census.txt'
nn = 'tests/census/20.py'
if len(sys.argv) > 1:
    domain = checker(sys.argv[1])
else:
    domain = AbstractDomain.NEURIFY_SYMBOLIC3 # default
print(f"> Domain chosen: '{domain}'")
b = ForwardAnalysis(spec, domain=domain, log=True)
# forced_active = {
#     Activation(23+3, PyVar("x10")), 
#     Activation(23+5, PyVar("x12")), 
#     Activation(23+6, PyVar("x13")), 
#     Activation(23+9, PyVar("x20")), 
#     Activation(23+12, PyVar("x23"))
# }
# forced_inactive = {
#     Activation(23+17, PyVar("x32")), 
#     Activation(23+15, PyVar("x30")), 
#     Activation(23+13, PyVar("x24")), 
#     Activation(23+7, PyVar("x14"))
# }
b.main(nn) # , forced_active_names=forced_active, forced_inactive_names=forced_inactive)

"""
By-hand analysis (exact, using rational numbers) using Neurify in the 'tests/oy.py' neural network:

Note that, variables endowed with a tick denote the variable state before the ReLU activation function.

{eq_table_low:                          {eq_table_up:
x01   -> 0                              x01   -> +1.0
x02   -> 0                              x02   -> +1.0
x11'  -> -0.31x01 + 0.99x02 -0.63       x11'  -> -0.31x01 + 0.99x02 -0.63
x11   -> 0.27692307692307694x11'        x11   -> 0.27692307692307694x11' +0.2603076923076923
x12'  -> -1.25x01 - 0.64x02 +1.88       x12'  -> -1.25x01 - 0.64x02 +1.88
x12   -> 0.9947089947089947x12'         x12   -> 0.9947089947089947x12' +0.009947089947089947
x21'  -> 0.4x11 + 1.21x12               x21'  -> 0.4x11 + 1.21x12
x21   -> 0.9970458806685601x21'         x21   -> x21'
x22'  -> 0.64x11 + 0.69x12 -0.39        x22'  -> 0.64x11 + 0.69x12 -0.39
x22   -> 0.670257895935289x22'          x22   -> 0.8176726234369361x22' +0.17542474259004248
x31   -> 0.26x21 + 0.33x22 +0.45        x31   -> 0.26x21 + 0.33x22 +0.45
x32   -> 1.42x21 + 0.4x22 -0.45         x32   -> 1.42x21 + 0.4x22 -0.45
bounds:
x01: (0.0, 0.0) (1.0, 1.0)
x02: (0.0, 0.0) (1.0, 1.0)
x11': (-0.94, 0.36) (-0.94, 0.36)
x11: (-0.2603076923076923, _) (_, 0.36)
x12': (-0.01, 1.88) (-0.01, 1.88)
x12: (-0.009947089947089947, _) (_, 1.88)
x21': (-0.006497517297517298, 2.192979405779406) (0.10966153846153846, 2.3091384615384616)
x21: (-0.006478322856062337, _) (_, 2.3091384615384616)
x22': (-0.3880019536019536, 0.7886811233211233) (-0.21454153846153845, 0.9621415384615385)
x22: (-0.260061373040027, _) (_, 0.9621415384615385)
x31: (0.36249538295421485, 1.192934799152831) (0.478512, 1.3678827076923077)
x32: (-0.5632237676716193, 2.8662794378635863) (-0.29428061538461536, 3.2138332307692306)

By-hand analysis (exact, using rational numbers) using the product DeepPoly x Neurify in the 'tests/deeppoly_custom.py' neural network,
a custom version of the example neural network presented in their paper, the only difference is that x01 ranges in [0, 1] instead of [-1, 1].

We use a numerical workaround to obtain an equivalent neural network starting from inputs in [0, 1] (we add the implicit constraint x02 := 2*x02 - 1)
since x02 ranges in [-1, 1] in the custom example.

bounds:
x01: (0.0, 0.0) (1.0, 1.0)
x02: (-1.0, -1.0) (1.0, 1.0)
x11': (-1.0, 2.0) (-1.0, 2.0)
x11: (-0.6666666666666666, _) (_, 2.0)
x12': (-1.0, 2.0) (-1.0, 2.0)
x12: (-0.6666666666666666, _) (_, 2.0)
x21': (0.0, 1.3333333333333333) (1.3333333333333333, 2.6666666666666665)
x21: (0.0, _) (_, 2.6666666666666665)
x22': (-2.0, 0.6666666666666666) (-0.6666666666666666, 2.0)
x22: (-0.5, _) (_, 2.0)
x31: (0.6666666666666667, 2.6666666666666665) (1.833333333333333, 5.166666666666666)
x32: (-0.5, 0.16666666666666666) (0.0, 2.0)

"""