"""
Scripts used for benchmarking a few of the ACAS properties

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import time
import os

import numpy as np
from tqdm import tqdm

from verinet.verification.verinet import VeriNet
from verinet.parsers.onnx_parser import ONNXParser
from verinet.verification.objective import Objective

networks_path = "../../resources/models/onnx/acas/"
networks_path = "resources/models/onnx/acas/"
networks = sorted(os.listdir(networks_path))


# noinspection PyTypeChecker,DuplicatedCode
def prop_1(result_file) -> tuple:

    print("Benchmarking property 1...")
    result_file.write(f"Benchmarking for property 1: \n\n")
    tic = time.time()

    for i in tqdm(range(len(networks))):
        network = networks[i]

        result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

        parser = ONNXParser(os.path.join(networks_path, network))
        model = parser.to_pytorch()

        input_bounds = np.array([[55947.691, 60760.0], [-3.141593, 3.141593],
                                 [-3.141593, 3.141593], [1145, 1200.0], [0, 60]])

        input_bounds = norm_input(input_bounds)

        objective = Objective(input_bounds, output_size=5, model=model)
        out_vars = objective.output_vars

        objective.add_constraints(out_vars[0] <= 3.9911256459)

        benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_2(result_file) -> tuple:

    print("Benchmarking property 2...")
    result_file.write(f"Benchmarking for property 2: \n\n")
    tic = time.time()

    networks2 = []
    for network in networks:

        id1, id2 = int(network.split('_')[2]), int(network.split('_')[3])  # Network id
        if not (id1 == 1 or (id1 == 3 and id2 == 3) or (id1 == 4 and id2 == 2) or (id1 == 5 and id2 == 3)):
            networks2.append(network)

    for i in tqdm(range(len(networks2))):
        network = networks2[i]

        result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

        parser = ONNXParser(os.path.join(networks_path, network))
        model = parser.to_pytorch()

        input_bounds = np.array([[55947.691, 60760.0], [-3.141593, 3.141593],
                                 [-3.141593, 3.141593], [1145, 1200.0], [0, 60]])

        input_bounds = norm_input(input_bounds)

        objective = Objective(input_bounds, output_size=5, model=model)
        out_vars = objective.output_vars

        constr = out_vars[0] <= out_vars[1]
        for j in [2, 3, 4]:
            constr = constr | (out_vars[0] <= out_vars[j])

        objective.add_constraints(constr)
        benchmark(objective, result_file)


    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")

# noinspection PyTypeChecker,DuplicatedCode
def prop_3(result_file) -> tuple:

    print("Benchmarking property 3...")
    result_file.write(f"Benchmarking for property 3: \n\n")
    tic = time.time()

    networks3 = []
    for network in networks:

        id1, id2 = int(network.split('_')[2]), int(network.split('_')[3])  # Network id
        if not ((id1 == 1 and id2 == 7) or (id1 == 1 and id2 == 8) or (id1 == 1 and id2 == 9)):
            networks3.append(network)

    for i in tqdm(range(len(networks3))):
        network = networks3[i]

        result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

        parser = ONNXParser(os.path.join(networks_path, network))
        model = parser.to_pytorch()

        input_bounds = np.array([[1500, 1800], [-0.06, 0.06],
                                 [3.10, 3.141593], [980, 1200], [960, 1200]])

        input_bounds = norm_input(input_bounds)

        objective = Objective(input_bounds, output_size=5, model=model)
        out_vars = objective.output_vars

        constr = out_vars[0] >= out_vars[1]
        for j in [2, 3, 4]:
            constr = constr | (out_vars[0] >= out_vars[j])

        objective.add_constraints(constr)
        benchmark(objective, result_file)


    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_4(result_file) -> tuple:

    print("Benchmarking property 4...")
    result_file.write(f"Benchmarking for property 4: \n\n")
    tic = time.time()

    networks4 = []
    for network in networks:

        id1, id2 = int(network.split('_')[2]), int(network.split('_')[3])  # Network id
        if not ((id1 == 1 and id2 == 7) or (id1 == 1 and id2 == 8) or (id1 == 1 and id2 == 9)):
            networks4.append(network)

    for i in tqdm(range(len(networks4))):
        network = networks4[i]

        result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

        parser = ONNXParser(os.path.join(networks_path, network))
        model = parser.to_pytorch()

        input_bounds = np.array([[1500, 1800], [-0.06, 0.06],
                                 [0, 0], [1000, 1200], [700, 800]])

        input_bounds = norm_input(input_bounds)

        objective = Objective(input_bounds, output_size=5, model=model)
        out_vars = objective.output_vars

        constr = out_vars[0] >= out_vars[1]
        for j in [2, 3, 4]:
            constr = constr | (out_vars[0] >= out_vars[j])

        objective.add_constraints(constr)
        benchmark(objective, result_file)


    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_5(result_file) -> tuple:

    print("Benchmarking property 5...")
    result_file.write(f"Benchmarking for property 5: \n\n")
    tic = time.time()

    network = networks[0]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[250, 400], [0.2, 0.4], [-3.141592, -3.141592 + 0.005], [100, 400], [0, 400]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [0, 1, 2, 3]:
        objective.add_constraints(out_vars[4] <= out_vars[i])

    benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_6(result_file) -> tuple:

    print("Benchmarking property 6...")
    result_file.write(f"Benchmarking for property 6: \n\n")
    tic = time.time()

    network = networks[0]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[12000, 62000], [0.7, 3.141592], [-3.141592, -3.141592 + 0.005], [100, 1200], [0, 1200]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [1, 2, 3, 4]:
        objective.add_constraints(out_vars[0] <= out_vars[i])

    benchmark(objective, result_file)

    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    input_bounds = np.array([[12000, 62000], [-3.141592, -0.7], [-3.141592, -3.141592 + 0.005], [100, 1200], [0, 1200]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [1, 2, 3, 4]:
        objective.add_constraints(out_vars[0] <= out_vars[i])

    benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")

# noinspection PyTypeChecker,DuplicatedCode
def prop_7(result_file):

    print("Benchmarking property 7...")
    result_file.write(f"Benchmarking for property 7: \n\n")
    tic = time.time()

    network = networks[8]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[0, 60760], [-3.141592, 3.141592], [-3.141592, 3.141592], [100, 1200], [0, 1200]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    constr_strong_left = (out_vars[3] >= out_vars[0]) | (out_vars[3] >= out_vars[1]) | (out_vars[3] >= out_vars[2])
    constr_strong_right = (out_vars[4] >= out_vars[0]) | (out_vars[4] >= out_vars[1]) | (out_vars[4] >= out_vars[2])

    objective.add_constraints(constr_strong_left)
    objective.add_constraints(constr_strong_right)

    benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_8(result_file):

    print("Benchmarking property 8...")
    result_file.write(f"Benchmarking for property 8: \n\n")
    tic = time.time()

    network = networks[17]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[0, 60760], [-3.141592, -0.75 * 3.141592], [-0.1, 0.1], [600, 1200], [600, 1200]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    constr1 = (out_vars[0] >= out_vars[2]) | (out_vars[1] >= out_vars[2])
    constr2 = (out_vars[0] >= out_vars[3]) | (out_vars[1] >= out_vars[3])
    constr3 = (out_vars[0] >= out_vars[4]) | (out_vars[1] >= out_vars[4])

    objective.add_constraints(constr1)
    objective.add_constraints(constr2)
    objective.add_constraints(constr3)

    benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")
    
# noinspection PyTypeChecker,DuplicatedCode
def prop_9(result_file):

    print("Benchmarking property 9...")
    result_file.write(f"Benchmarking for property 9: \n\n")
    tic = time.time()

    network = networks[20]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[2000, 7000], [-0.4, -0.14], [-3.141592, -3.141592 + 0.01], [100, 150], [0, 150]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [0, 1, 2, 4]:
        objective.add_constraints(out_vars[3] <= out_vars[i])

    benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")

# noinspection PyTypeChecker,DuplicatedCode
def prop_10(result_file):

    print("Benchmarking property 10...")
    result_file.write(f"Benchmarking for property 10: \n\n")
    tic = time.time()

    network = networks[31]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[36000, 60760], [0.7, 3.141592], [-3.141592, -3.141592 + 0.01], [900, 1200], [600, 1200]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [1, 2, 3, 4]:
        objective.add_constraints(out_vars[0] <= out_vars[i])

    benchmark(objective, result_file)

    t = time.time() - tic
    result_file.write(f'Total time: {t}')
    result_file.write("\n\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_11(result_file):

    print("Benchmarking property 11...")
    result_file.write(f"Benchmarking for property 11: \n\n")

    network = networks[4]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[250, 400], [0.2, 0.4], [-3.141592, -3.141592 + 0.005], [100, 400], [0, 400]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    objective.add_constraints(out_vars[4] <= out_vars[0])

    benchmark(objective, result_file)
    result_file.write("\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_12(result_file):

    print("Benchmarking property 12...")
    result_file.write(f"Benchmarking for property 12: \n\n")

    network = networks[20]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[55947.691, 60760.0], [-3.141593, 3.141593],
                             [-3.141593, 3.141593], [1145, 1200.0], [0, 60]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [1, 2, 3, 4]:
        objective.add_constraints(out_vars[0] <= out_vars[i])

    benchmark(objective, result_file)
    result_file.write("\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_13(result_file):

    print("Benchmarking property 13...")
    result_file.write(f"Benchmarking for property 13: \n\n")

    network = networks[0]
    result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")

    parser = ONNXParser(os.path.join(networks_path, network))
    model = parser.to_pytorch()

    input_bounds = np.array([[60000, 60760.0], [-3.141593, 3.141593],
                             [-3.141593, 3.141593], [0, 360], [0, 360]])

    input_bounds = norm_input(input_bounds)

    objective = Objective(input_bounds, output_size=5, model=model)
    out_vars = objective.output_vars

    for i in [1, 2, 3, 4]:
        objective.add_constraints(out_vars[0] <= out_vars[i])

    benchmark(objective, result_file)
    result_file.write("\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_14(result_file):

    print("Benchmarking property 14...")
    result_file.write(f"Benchmarking for property 14: \n\n")

    for network in [networks[27], networks[36]]:

        result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")
        network = os.path.join(networks_path, network)

        parser = ONNXParser(os.path.join(networks_path, network))
        model = parser.to_pytorch()

        input_bounds = np.array([[250, 400], [0.2, 0.4],
                                 [-3.141593, -3.141593+0.005], [100, 400], [0, 400]])

        input_bounds = norm_input(input_bounds)

        objective = Objective(input_bounds, output_size=5, model=model)
        out_vars = objective.output_vars

        for i in [0, 1, 2, 3]:
            objective.add_constraints(out_vars[4] <= out_vars[i])

        benchmark(objective, result_file)
    result_file.write("\n")


# noinspection PyTypeChecker,DuplicatedCode
def prop_15(result_file):

    print("Benchmarking property 15...")
    result_file.write(f"Benchmarking for property 15: \n\n")

    for network in [networks[27], networks[36]]:

        result_file.write(f"Network_{network.split('_')[2]}_{network.split('_')[3]}, ")
        network = os.path.join(networks_path, network)

        parser = ONNXParser(os.path.join(networks_path, network))
        model = parser.to_pytorch()

        input_bounds = np.array([[250, 400], [-0.4, -0.2],
                                 [-3.141593, -3.141593+0.005], [100, 400], [0, 400]])

        input_bounds = norm_input(input_bounds)

        objective = Objective(input_bounds, output_size=5, model=model)
        out_vars = objective.output_vars

        for i in [0, 1, 2, 4]:
            objective.add_constraints(out_vars[3] <= out_vars[i])

        benchmark(objective, result_file)
    result_file.write("\n")


# noinspection PyTypeChecker,DuplicatedCode
def benchmark(objective: Objective, result_file):

    start_time = time.time()
    status = solver.verify(objective=objective, timeout=7200)
    result_file.write(f"Status: {status}, branches explored: {solver.branches_explored}, "
                      f"max depth: {solver.max_depth}, time: {time.time() - start_time:.2f} seconds\n")


def norm_input(inputs: np.array):

    mean = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    std = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])

    for i in range(inputs.shape[0]):

        inputs[i] = (inputs[i] - mean[i]) / std[i]

    return inputs


if __name__ == '__main__':

    global solver
    solver = VeriNet(max_procs=1, use_gpu=False)

    with open(f"acas.txt", 'w', buffering=1) as file:
        # prop_1(file)
        # prop_2(file)
        # prop_3(file)
        # prop_4(file)
        # prop_5(file)
        prop_6(file)
        # prop_7(file)
        # prop_8(file)
        # prop_9(file)
        # prop_10(file)
        # prop_11(file)
        # prop_12(file)
        # prop_13(file)
        # prop_14(file)
        # prop_15(file)

    del solver
