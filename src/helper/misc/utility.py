import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
            

def print_w_b(model):
    for layer in model.modules():
        if hasattr(layer, 'weight'):
            print(layer)
            print('\t[+] w:', layer.weight.data.detach().flatten())
            print('\t[+] b:', layer.bias.data.detach().flatten())
            print()