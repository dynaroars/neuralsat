import random
import torch
import os


def write_vnnlib_classify_single(
    spec_path: str,  
    data_lb: float, 
    data_ub: float, 
    prediction: torch.Tensor,
    negate_spec=False):
    # input bounds
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    for y_i in range(n_class):
        if y_i == y:
            continue
        
        spec_name = f'{spec_path}_output_{y_i}.vnnlib'
        # print(f'{spec_name=}')
        with open(spec_name, "w") as f:
            f.write(f"; Specification for class {int(y)}\n")

            f.write(f"\n; Definition of input variables\n")
            for i in range(len(x_ub)):
                f.write(f"(declare-const X_{i} Real)\n")

            f.write(f"\n; Definition of output variables\n")
            for i in range(n_class):
                f.write(f"(declare-const Y_{i} Real)\n")

            f.write(f"\n; Definition of input constraints\n")
            for i in range(len(x_ub)):
                f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
                f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

            f.write(f"\n; Definition of output constraints\n")
            if not negate_spec:
                f.write(f"(assert (<= Y_{y_i} Y_{y}))\n")
            else:
                f.write(f"(assert (or\n")
                f.write(f"\t(and (>= Y_{y_i} Y_{y}))\n")
                f.write(f"))\n")
        yield os.path.basename(spec_name)


def write_vnnlib_classify(
    spec_path: str,  
    data_lb: float, data_ub: float, 
    prediction: torch.Tensor,
    negate_spec=False,
    single_output=False) -> str:
    # input bounds
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    with open(spec_path, "w") as f:
        f.write(f"; Specification for class {int(y)}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        if not negate_spec:
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"(assert (<= Y_{i} Y_{y}))\n")
                if single_output:
                    break
        else:
            f.write(f"(assert (or\n")
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
                if single_output:
                    break
            f.write(f"))\n")
    return spec_path



def write_vnnlib_recon_robust(
    spec_path: str,  
    center: torch.Tensor, 
    input_radius: float, 
    output_radius: float,
    num_out_prop: int,
    seed: int,
    negate_spec=False) -> str:

    flattened_center = center.flatten()
    # input bounds
    x_lb = flattened_center - input_radius
    x_ub = flattened_center + input_radius
    
    y_lb = flattened_center - output_radius
    y_ub = flattened_center + output_radius
    
    # outputs
    n_class = len(flattened_center)
    indices = random.sample(range(n_class), num_out_prop)
    
    # print(f'{indices=}')
    
    with open(spec_path, "w") as f:
        f.write(f"; Specification for reconstruction for {seed=} {indices=} {input_radius=} {output_radius=}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        if not negate_spec:
            for i in indices:
                f.write(f"(assert (<= Y_{i} {y_ub[i]}))\n")
                f.write(f"(assert (>= Y_{i} {y_lb[i]}))\n")
        else:
            f.write(f"(assert (or\n")
            for i in indices:
                # f.write(f"\t(and (>= Y_{i} {y_ub[i]}))\n")
                f.write(f"\t(and (<= Y_{i} {y_lb[i]}))\n")
            f.write(f"))\n")
    return spec_path

def write_vnnlib_recon_relation(
    spec_path: str,  
    center: torch.Tensor, 
    input_radius: float, 
    select_pairs: list[tuple[int, int]],
    negate_spec=False) -> str:

    flattened_center = center.flatten()
    # input bounds
    x_lb = flattened_center - input_radius
    x_ub = flattened_center + input_radius
    
    # outputs
    n_class = len(flattened_center)
    
    # print(f'{indices=}')
    
    with open(spec_path, "w") as f:
        f.write(f"; Specification for reconstruction for {select_pairs=} {input_radius=}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        if not negate_spec:
            for pair in select_pairs:
                assert flattened_center[pair[0]] + 2 * input_radius < flattened_center[pair[1]]
                f.write(f"(assert (<= Y_{pair[0]} Y_{pair[1]}))\n")
        else:
            f.write(f"(assert (or\n")
            for pair in select_pairs:
                assert flattened_center[pair[0]] + 2 * input_radius < flattened_center[pair[1]]
                f.write(f"\t(and (>= Y_{pair[0]} Y_{pair[1]}))\n")
            f.write(f"))\n")
    return spec_path

if __name__ == "__main__":
    center = torch.randn(1, 3, 5, 5)
    input_radius = 0.01
    output_radius = 0.05
    
    write_vnnlib_recon_robust(
        spec_path='./test_recon.vnnlib',
        center=center,
        input_radius=input_radius,
        output_radius=output_radius,
        num_out_prop=5,
        negate_spec=True,
    )