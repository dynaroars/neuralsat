from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import onnxruntime as rt
import os


def generateNBenchmarks(network, epsilon, numBenchmarks, timeout, csvFile, specsDir, seed):
    print(f"Generating {numBenchmarks} benchmarks for {network} with epsilon {epsilon}")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    imgList = list(range(len(test_images)))
    random.Random(seed).shuffle(imgList)
    numBenchmarksSoFar = 0
    for index in imgList:
        if numBenchmarksSoFar == numBenchmarks:
            return
        if getBenchmark(network, test_images, test_labels, epsilon, index, timeout, csvFile, specsDir):
            numBenchmarksSoFar += 1


def getBenchmark(network, test_images, test_labels, epsilon, index, timeout, csvFile, specsDir):
    image = (test_images[index]/255).astype(np.float32).reshape(1,32,32,3)
    label = test_labels[index][0]

    sess = rt.InferenceSession(network)
    input_name = sess.get_inputs()[0].name
    output = sess.run([sess.get_outputs()[0].name], {input_name: image})
    prediction = np.argmax(output)
    if (prediction != label):
        return False

    image = image.flatten()
    targetLabel = (label + 1) % 10

    vnnLibFile = "{}/network{}_index{}_eps{}_target{}_orig{}.vnnlib".format(specsDir, os.path.basename(network).split(".")[0],
                                                                            index, epsilon,
                                                                            targetLabel, label)
    if os.path.isfile(vnnLibFile):
        return False

    f = open(vnnLibFile, 'w')
    print("; index {}, correct label {}, target label {}, epsilon {}".format(index,label, targetLabel,epsilon), file = f)

    for i in range(3*32*32):
        print("(declare-const X_{} Real)".format(i), file=f)

    for i in range(10):
        print("(declare-const Y_{} Real)".format(i), file=f)

    print( "; Input constraints", file=f)
    for i in range(3*32*32):
        val = image[i]
        lb = max(0, val - epsilon)
        ub = min(1, val + epsilon)
        print("(assert (<= X_{} {}))".format(i, ub), file=f)
        print("(assert (>= X_{} {}))".format(i, lb), file=f)

    print( "; Output constraints",file=f)
    for i in range(10):
        if i != targetLabel:
            print("(assert (>= Y_{} Y_{}))".format(targetLabel, i), file=f)
    f.close()
    print(f"{network},{vnnLibFile},{timeout}", file=csvFile)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--networks', type=str, default="./nets/", help='The network directory')
    parser.add_argument('--specs', type=str, default="./specs", help='The spec directory')
    parser.add_argument('--timeout', type=int, default=300, help='Number of benchmarks per network-epsilon pair')
    parser.add_argument('--csv', type=str, default="./marabou-cifar10_instances.csv", help='csv file to write to')

    args = parser.parse_args()
    seed = args.seed
    networks = args.networks
    specsDir = args.specs
    timeout = args.timeout
    numBenchmarks = 6 * 3600 / timeout / 3 / 2
    csv = args.csv

    csvFile = open(csv, "w")
    for network in os.listdir(networks):
        network = os.path.join(networks, network)
        for epsilon in [0.012, 0.024]:
            generateNBenchmarks(network, epsilon, numBenchmarks, timeout, csvFile, specsDir, seed)
    csvFile.close()

if __name__ == "__main__":
    main()
