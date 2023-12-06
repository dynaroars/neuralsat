from pathlib import Path
import unittest
import os

import warnings
warnings.filterwarnings(action='ignore')

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

class TestVerifier(unittest.TestCase):


    def test_mnist1(self):
        net_path = 'example/mnistfc-medium-net-554.onnx'
        vnnlib_path = Path('example/test.vnnlib')
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
        # self.assertEqual(verifier.iteration, 40)
        

    def test_mnist2(self):
        net_path = 'example/mnistfc-medium-net-151.onnx'
        vnnlib_path = Path('example/prop_2_0.03.vnnlib')
        device = 'cpu'

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
        # self.assertTrue(verifier.iteration in [24])
        
    
    def test_mnist3(self):
        net_path = 'example/mnist-net_256x2.onnx'
        vnnlib_path = Path('example/prop_1_0.05.vnnlib')
        device = 'cpu'

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
        
        
    def test_acas1(self):
        net_path = 'example/ACASXU_run2a_1_1_batch_2000.onnx'
        vnnlib_path = Path('example/prop_3.vnnlib')
        device = 'cuda'

        print('\n\nRunning test with', net_path, vnnlib_path)
        
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        
        verifier = Verifier(
            net=model, 
            input_shape=input_shape, 
            batch=2000,
            device=device,
        )
        
        status = verifier.verify(objectives)
        
        self.assertEqual(status, ReturnStatus.UNSAT)
        # self.assertTrue(verifier.iteration in [180, 184])
        
        
    
    def test_acas2(self):
        net_path = 'example/ACASXU_run2a_1_1_batch_2000.onnx'
        vnnlib_path = Path('example/prop_6.vnnlib')
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
        # self.assertTrue(verifier.iteration in [40, 177])
        
        
    def test_acas3(self):
        net_path = 'example/ACASXU_run2a_1_9_batch_2000.onnx'
        vnnlib_path = Path('example/prop_7.vnnlib')
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
        
        
    def test_nn4sys(self):
        net_path = 'example/mscn_128d.onnx'
        vnnlib_path = Path('example/cardinality_0_100_128.vnnlib')
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
        # self.assertTrue(verifier.iteration in [5, 158])
        
        
    
    def test_cifar1(self):
        net_path = 'example/cifar10_2_255_simplified.onnx'
        vnnlib_path = Path('example/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib')
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
        # self.assertTrue(verifier.iteration in [20])
        
        
    def test_cgan1(self):
        return True
        net_path = 'example/cGAN_imgSz32_nCh_1.onnx'
        vnnlib_path = Path('example/cGAN_imgSz32_nCh_1_prop_2_input_eps_0.020_output_eps_0.025.vnnlib')
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
        
        
        
    def test_dist_shift1(self):
        net_path = 'example/mnist_concat.onnx'
        vnnlib_path = Path('example/index188_delta0.13.vnnlib')
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

        
        
    def test_tllverifybench1(self):
        net_path = 'example/tllBench_n=2_N=M=16_m=1_instance_1_1.onnx'
        vnnlib_path = Path('example/property_N=16_1.vnnlib')
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
        
        
    def test_vggnet161(self):
        return True
        net_path = 'example/vgg16-7.onnx'
        if not os.path.exists(net_path):
            return True
        
        vnnlib_path = Path('example/spec0_screw.vnnlib')
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
    Settings.setup(args=None)
    print(Settings)
    
    logger.setLevel(1)
    unittest.main()
    