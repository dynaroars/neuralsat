
# unpack tuple, dict, list into one single list
# TODO: not sure if the order matches graph.inputs()
def unpack_inputs(inputs, device=None):
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        res = []
        for item in inputs: 
            res += unpack_inputs(item, device=device)
        return res
    else:
        if device is not None:
            inputs = inputs.to(device)
        return [inputs]