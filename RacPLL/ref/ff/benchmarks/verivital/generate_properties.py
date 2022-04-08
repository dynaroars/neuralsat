#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_data(img_num: int = 20, seed: int = 0):
    
    
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor(),
        download = True
    )
    
    loader = DataLoader(test_data,batch_size=10000)#num_workers=1 shuffle=True,
    
    sample_images = next(iter(loader))
    images, labels = sample_images
    #print(len(labels))

    image_counter = 0
    final_images, final_labels = [],[]
    
    sess_avg = onnxruntime.InferenceSession("Convnet_avgpool.onnx")
    sess_max = onnxruntime.InferenceSession("Convnet_maxpool.onnx")

    i = seed-1
    while image_counter < img_num:

        i += 1
        correctly_classified = True

        input_image_avg = {sess_avg.get_inputs()[0].name: to_numpy(images[i:i+1])}
        output_avg = np.argmax(sess_avg.run(None, input_image_avg))
        
        input_image_max = {sess_max.get_inputs()[0].name: to_numpy(images[i:i+1])}
        output_max = np.argmax(sess_max.run(None, input_image_max))
        
        if output_avg != labels[i:i+1].item() or output_max != labels[i:i+1].item():
            correctly_classified = False

        if output_avg == labels[i:i+1].item() and output_max == labels[i:i+1].item():
            correctly_classified = True
            #print(i)
            image_counter += 1
            final_images.append(images[i:i+1])
            final_labels.append(labels[i:i+1])
    
    return final_images, final_labels

def upper_lower_bounds(image: torch.Tensor, epsilon: float):

    upper_bound = torch.clamp((image + epsilon), 0, 1)
    upper_bound = torch.reshape(upper_bound, (1, 784))
    lower_bound = torch.clamp((image - epsilon), 0, 1)
    lower_bound = torch.reshape(lower_bound, (1, 784))
    
    return upper_bound, lower_bound

def write_vnnlib_spec(upper_bound: torch.Tensor, lower_bound: torch.Tensor, correct_label: int, path: str):

    output_class = 10
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:

        f.write(f"; Mnist property with label: {correct_label}.\n")

        # input variables.
        f.write(f"; Input variables:\n")
        f.write("\n")
        for i in range(upper_bound.shape[1]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # output variables.
        f.write(f"; Output variables:\n")
        f.write("\n")
        for i in range(output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(upper_bound.shape[1]):
            f.write(f"(assert (<= X_{i} {upper_bound[0][i].item()}))\n")
            f.write(f"(assert (>= X_{i} {lower_bound[0][i].item()}))\n")
            f.write("\n")
        f.write("\n")

        # output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(output_class):
            if i != correct_label:
                f.write(f"    (and (>= Y_{i} Y_{correct_label}))\n")
        f.write("))")

def csv_instances(num_props: int = 25, path: str = "verivital_instances.csv"):

    nets = ["Convnet_avgpool.onnx",
            "Convnet_maxpool.onnx"]

    properties_avg = [f"prop_{i}_0.02.vnnlib" for i in range(num_props)]
    properties_avg += [f"prop_{i}_0.04.vnnlib" for i in range(num_props)]
    
    properties_max = [f"prop_{i}_0.004.vnnlib" for i in range(num_props)]
    with open(path, "w") as f:

        for net in nets:
            timeout = 300 if net == "Convnet_avgpool.onnx" else 420
            if net == "Convnet_avgpool.onnx":
                for prop in properties_avg:

                    if prop == properties_avg[-1]:
                        f.write(f"{net},./specs/avgpool_specs/{prop},{timeout}\n")
                    else:
                        f.write(f"{net},./specs/avgpool_specs/{prop},{timeout}\n")
                        
            if net == "Convnet_maxpool.onnx":
                for prop in properties_max:
                    f.write(f"{net},./specs/maxpool_specs/{prop},{timeout}\n")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Specification Generrator: vnnlib format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed',type = int,default= 0, help='seed is selected for image selection')
    args = parser.parse_args()
    
    num_images = 20
    epsilon_avg = [0.02, 0.04]
    epsilon_max = [0.004]
    
    torch.manual_seed(args.seed)
    idx = torch.randint(0,9000,(1,)).item()
    #print(idx)

    images, labels = get_data(img_num=num_images,seed = idx)

    for eps in epsilon_avg:

        for i in range(num_images):
            #print(images[i])
            image, label = images[i], labels[i].item()
            image
            upper_bound, lower_bound = upper_lower_bounds(image, eps)
            path = f"./specs/avgpool_specs/prop_{i}_{eps:.2f}.vnnlib"

            write_vnnlib_spec(upper_bound, lower_bound, label, path)
            
    for eps in epsilon_max:

        for i in range(num_images):

            image, label = images[i], labels[i].item()
            upper_bound, lower_bound = upper_lower_bounds(image, eps)
            path = f"./specs/maxpool_specs/prop_{i}_{eps:.3f}.vnnlib"

            write_vnnlib_spec(upper_bound, lower_bound, label, path)

    csv_instances(num_images)

