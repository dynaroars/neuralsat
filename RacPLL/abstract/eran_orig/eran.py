import sys
sys.path.insert(0, '/home/hai/Desktop/research/survey/tools/eran/tf_verify')
sys.path.insert(0, '/home/hai/Desktop/research/survey/tools/eran/ELINA/python_interface/')
import re
import numpy as np
import itertools
import torch

from read_net_file import read_onnx_net
from config import config
from onnx_translator import ONNXTranslator
from optimizer import Optimizer
from analyzer import Analyzer, layers


class ERAN:

    def __init__(self, model_path, domain='deepzono'):
        model, _ = read_onnx_net(model_path)
        translator = ONNXTranslator(model, False)
        operations, resources = translator.translate()
        self.input_shape = resources[0]["deeppoly"][2]
        self.optimizer  = Optimizer(operations, resources)
        self.domain = domain        

    def __call__(self, specLB, specUB, timeout_lp=1, timeout_milp=1, use_default_heuristic=True,
                output_constraints=None, lexpr_weights= None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None,
                uexpr_cst=None, uexpr_dim=None, expr_size=0, testing = False,label=-1, prop = -1,
                spatial_constraints=None, K=0, s=0, timeout_final_lp=100, timeout_final_milp=100, use_milp=False,
                complete=False, terminate_on_failure=True, partial_milp=False, max_milp_neurons=30, approx_k=True):
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        if self.domain == 'deepzono' or self.domain == 'refinezono':
            execute_list, output_info = self.optimizer.get_deepzono(nn, specLB, specUB)
            analyzer = Analyzer(execute_list, nn, self.domain, timeout_lp, timeout_milp, output_constraints,
                                use_default_heuristic, label, prop, testing, K=K, s=s,
                                timeout_final_lp=timeout_final_lp, timeout_final_milp=timeout_final_milp,
                                use_milp=use_milp, complete=complete,
                                partial_milp=partial_milp, max_milp_neurons=max_milp_neurons)
        elif self.domain == 'deeppoly' or self.domain == 'refinepoly':
            execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints)
            analyzer = Analyzer(execute_list, nn, self.domain, timeout_lp, timeout_milp, output_constraints,
                                use_default_heuristic, label, prop, testing, K=K, s=s,
                                timeout_final_lp=timeout_final_lp, timeout_final_milp=timeout_final_milp,
                                use_milp=use_milp, complete=complete,
                                partial_milp=partial_milp, max_milp_neurons=max_milp_neurons,
                                approx_k=approx_k)
        element, nlb, nub = analyzer.get_abstract0()
        hidden_bounds = []
        for i in range(0, len(nlb)-1, 2):
            # print('lower:', nlb[i])
            # print('upper:', nub[i])
            # print()
            hidden_bounds.append((nlb[i], nub[i]))
        # print('-------------')

        return (torch.Tensor(nlb[-1]), torch.Tensor(nub[-1])), hidden_bounds


if __name__ == '__main__':
    
    path = '/home/hai/Desktop/research/survey/tools/eran/data/corina/net.onnx'

    specLB = [-5, -4]
    specUB = [-1, -2]

    eran = ERAN(path, 'refinezono')

    lbs, ubs = eran(specLB, specUB)
    print(lbs)
    print(ubs)
