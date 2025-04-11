from helper.network.read_onnx import parse_onnx
from helper.spec.objective import parse_vnnlib
from helper.misc.result import ReturnStatus
from verifier.verifier import Verifier 
from attacker.attacker import Attacker
from helper.misc.logger import logger
from setting import Settings

def neuralsat_verify(onnx_path, vnnlib_path, device, timeout=10.0, verbose=False, force_split=None):
    if verbose:
        logger.setLevel(2)
    else:
        logger.setLevel(0)
        
    Settings.setup(None)
    Settings.use_attack = 0
    Settings.use_mip_verify = 0
    Settings.use_restart = 0
    Settings.use_mip_tightening = 0
    
    model, input_shape, output_shape = parse_onnx(onnx_path)
    model = model.to(device)
    model.eval()
    objectives = parse_vnnlib(vnnlib_path, input_shape)
    if verbose:
        print(model)
        
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=500,
        device=device,
    )
    
    status = verifier.verify(objectives, timeout=timeout, force_split=force_split)
    return status
    
    
def neuralsat_falsify(onnx_path, vnnlib_path, device, timeout=10.0, use_attack=True, verbose=False):
    
    model, input_shape, output_shape = parse_onnx(onnx_path)
    model.eval()
    model = model.to(device)
    
    objectives = parse_vnnlib(vnnlib_path, input_shape)
    if verbose:
        print(model)
        
    is_attacked, adv = Attacker(model, objectives, input_shape, device=device).run(timeout=timeout)
    if is_attacked:
        return ReturnStatus.SAT, adv.to(objectives.lower_bounds)
    return ReturnStatus.UNKNOWN, None
