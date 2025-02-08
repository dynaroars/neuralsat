from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
import torch

from verifier.verifier import Verifier 
from .test_function import extract_instance
from util.misc.logger import logger
from setting import Settings



    
    
if __name__ == "__main__":
    device = 'cuda'
    # model = models.resnet18().to(device)
    # inputs = torch.randn(50, 3, 224, 224).to(device)
    
    net_path = 'example/backup/motivation_example_159.onnx'
    vnnlib_path = 'example/backup/motivation_example_159.vnnlib'
    
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    
    logger.setLevel(1)
    # Settings.setup_test()
    Settings.setup(args=None)
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof:
        with record_function("test"):
            # model(inputs)
            verifier.verify(objectives)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))