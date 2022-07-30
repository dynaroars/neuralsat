import argparse
import numpy as np
from generate_specs import get_mnist, get_cifar10, get_sample_idx
import onnxruntime as rt


def main():
    parser = argparse.ArgumentParser(description='Evaluate Network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, choices=["mnist", "cifar10"],
                        help='The dataset to generate specs for')
    parser.add_argument('--n', type=int, default=25, help='The number of specs to generate')
    parser.add_argument('--block', action="store_true", default=False, help='Generate specs in a block')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for idx generation')
    parser.add_argument('--start_idx', type=int, default=None, help='Enforce block mode and return deterministic indices')
    parser.add_argument("--network", type=str,required=True, help="Network to evaluate as .onnx file.")
    parser.add_argument("--debug", action="store_true", default=False, help='Print more.')
    parser.add_argument('--mean', nargs='+', type=float, default=0.0, help='the mean used to normalize the data with')
    parser.add_argument('--std', nargs='+', type=float, default=1.0, help='the standard deviation used to normalize the data with')
    args = parser.parse_args()

    if args.start_idx is not None:
        args.block = True
        print(f"Producing deterministic {args.n} indices starting from {args.start_idx}.")

    if args.dataset == "mnist":
        dataset = get_mnist()
    elif args.dataset == "cifar10":
        dataset = get_cifar10()
    else:
        assert False, "Unkown dataset" # Should be unreachable

    idxs = get_sample_idx(args.n, block=args.block, seed=args.seed, n_max=len(dataset), start_idx=args.start_idx)

    sess = rt.InferenceSession(args.network)
    input_name = sess.get_inputs()[0].name

    mean = np.array(args.mean).reshape((1,-1,1,1)).astype(np.float32)
    std = np.array(args.std).reshape((1,-1,1,1)).astype(np.float32)

    acc = 0
    for idx in idxs:
        x, y = dataset[idx]
        x = x.unsqueeze(0).numpy().astype(np.float32)
        x = (x-mean)/std
        pred_onx = sess.run(None, {input_name: x})[0]
        acc += (y == np.argmax(pred_onx,axis=-1)).sum()
        if args.debug:
            print(f"Sample idx {idx}, predicted class: {int(np.argmax(pred_onx,axis=-1))}, true class {y}:\n", pred_onx)
    print(f"Accuracy: {acc}/{len(idxs)}")

if __name__ == "__main__":
    main()