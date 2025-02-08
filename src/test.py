
import warnings
warnings.filterwarnings(action='ignore')
import unittest
import logging

from verifier.objective import Objective, DnfObjectives
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.verifier import Verifier 
from util.misc.result import ReturnStatus
from util.misc.logger import logger
from setting import Settings



def extract_instance(net_path, vnnlib_path):
    vnnlibs = read_vnnlib(vnnlib_path)
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives

def reset_settings():
    Settings.__init__()
    Settings.setup(args=None)


class TestVerifier(unittest.TestCase):

    def test_mnist_sat(self):
        reset_settings()
        net_path = 'example/onnx/mnist-net_256x2.onnx'
        vnnlib_path = 'example/vnnlib/prop_1_0.05.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.SAT)
        
        
    def test_mnist_gdvb_unsat(self):
        reset_settings()
        net_path = 'example/onnx/mnistfc-medium-net-151.onnx'
        vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
        
    def test_mnist_unsat(self):
        reset_settings()
        net_path = 'example/onnx/mnist-net_256x2.onnx'
        vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
        
    def test_mnist_unsat_w_restart(self):
        reset_settings()
        Settings.use_mip_tightening = False
        Settings.use_restart = True
        Settings.max_hidden_visited_branches = 100
        
        net_path = 'example/onnx/mnist-net_256x2.onnx'
        vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
    def test_mnist_unsat_w_stablize(self):
        reset_settings()
        Settings.use_mip_tightening = True
        Settings.use_restart = False
        Settings.mip_tightening_timeout_per_neuron = 2.0
        
        net_path = 'example/onnx/mnist-net_256x2.onnx'
        vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
    def test_mnist_unsat_w_stablize_and_restart(self):
        reset_settings()
        Settings.use_mip_tightening = True
        Settings.use_restart = True
        Settings.mip_tightening_timeout_per_neuron = 2.0
        Settings.max_hidden_visited_branches = 100
        
        net_path = 'example/onnx/mnist-net_256x2.onnx'
        vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
    def test_acas_unsat(self):
        reset_settings()
        net_path = 'example/onnx/ACASXU_run2a_1_1_batch_2000.onnx'
        vnnlib_path = 'example/vnnlib/prop_6.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
    def test_acas_sat(self):
        reset_settings()
        net_path = 'example/onnx/ACASXU_run2a_1_9_batch_2000.onnx'
        vnnlib_path = 'example/vnnlib/prop_7.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.SAT)
        
        
    def test_nn4sys_unsat_1(self):
        reset_settings()
        net_path = 'example/onnx/mscn_128d.onnx'
        vnnlib_path = 'example/vnnlib/cardinality_0_100_128.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
        
    def test_nn4sys_unsat_2(self):
        reset_settings()
        net_path = 'example/onnx/pensieve_big_parallel.onnx'
        vnnlib_path = 'example/vnnlib/pensieve_parallel_55.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
    
    def test_cifar_unsat_1(self):
        reset_settings()
        net_path = 'example/onnx/cifar10_2_255_simplified.onnx'
        vnnlib_path = 'example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=200,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
    
    
    
    def test_cifar_unsat_2(self):
        reset_settings()
        net_path = 'example/onnx/convBigRELU__PGD.onnx'
        vnnlib_path = 'example/vnnlib/cifar10_spec_idx_95_eps_0.00784.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=200,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
    def test_cgan_unsat(self):
        reset_settings()
        net_path = 'example/onnx/cGAN_imgSz32_nCh_1.onnx'
        vnnlib_path = 'example/vnnlib/cGAN_imgSz32_nCh_1_prop_2_input_eps_0.020_output_eps_0.025.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=10,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
        
    def test_dist_shift_unsat(self):
        reset_settings()
        net_path = 'example/onnx/mnist_concat.onnx'
        vnnlib_path = 'example/vnnlib/index188_delta0.13.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)

        
        
    def test_tllverifybench_unsat(self):
        reset_settings()
        net_path = 'example/onnx/tllBench_n=2_N=M=16_m=1_instance_1_1.onnx'
        vnnlib_path = 'example/vnnlib/property_N=16_1.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
        
        
    def test_vit_unsat(self):
        reset_settings()
        net_path = 'example/onnx/pgd_2_3_16.onnx'
        vnnlib_path = 'example/vnnlib/pgd_2_3_16_4021.vnnlib'
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=1000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        
    
        
if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    unittest.main()
    