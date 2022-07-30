"""
Libra Static Neural Network Analyzer
====================================

:Author: Caterina Urban
"""
import argparse
from multiprocessing import cpu_count

from libra.engine.bias_analysis import BiasAnalysis, AbstractDomain


def checker(domain):
    if domain == 'boxes':
        return AbstractDomain.BOXES1
    elif domain == 'symbolic':
        return AbstractDomain.SYMBOLIC2
    elif domain == 'deeppoly':
        return AbstractDomain.DEEPPOLY
    elif domain == 'neurify':
        return AbstractDomain.NEURIFY
    elif domain == 'boxes_deeppoly':
        return AbstractDomain.BOXES2_DEEPPOLY
    elif domain == 'boxes_neurify':
        return AbstractDomain.BOXES2_NEURIFY
    elif domain == 'deeppoly_neurify':
        return AbstractDomain.DEEPPOLY_NEURIFY
    elif domain == 'deeppoly_symbolic':
        return AbstractDomain.DEEPPOLY_SYMBOLIC3
    elif domain == 'neurify_symbolic':
        return AbstractDomain.NEURIFY_SYMBOLIC3
    elif domain == 'boxes_deeppoly_neurify':
        return AbstractDomain.BOXES2_DEEPPOLY_NEURIFY
    elif domain == 'deeppoly_neurify_symbolic':
        return AbstractDomain.DEEPPOLY_NEURIFY_SYMBOLIC3

    error = "Invalid abstract domain! Valid domains are 'boxes', 'symbolic', " + \
            "'deeppoly', 'neurify', 'boxes_deeppoly', 'boxes_neurify'" + \
            "'deeppoly_symbolic', 'deeppoly_neurify', 'neurify_symbolic'" + \
            "'boxes_deeppoly_neurify', or 'deeppoly_neurify_symbolic'."
    raise argparse.ArgumentTypeError(error)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'specification',
        help='input specification')
    parser.add_argument(
        'neural_network',
        help='neural network (.py file) to analyze')

    parser.add_argument(
        '--domain',
        help='abstract domain to be used for the forward pre-analysis (boxes, symbolic, deeppoly, or neurify)',
        type=checker,
        default='symbolic')

    parser.add_argument(
        '--min_lower',
        help='minimum lower bound for the forward pre-analysis',
        type=float,
        default=None)
    parser.add_argument(
        '--lower',
        help='lower bound for the forward pre-analysis',
        type=float,
        default=1)

    parser.add_argument(
        '--upper',
        help='upper bound for the forward pre-analysis',
        type=int,
        default=0)
    parser.add_argument(
        '--max_upper',
        help='upper bound for the forward pre-analysis',
        type=int,
        default=None)

    parser.add_argument(
        '--cpu',
        help='number of CPUs to be used for the analysis',
        type=int,
        default=cpu_count())

    args = parser.parse_args()

    spec = args.specification
    domain = args.domain
    minL = args.lower if args.min_lower is None else args.min_lower
    L = args.lower
    U = args.upper
    maxU = args.upper if args.max_upper is None else args.max_upper
    cpu = args.cpu
    nn = args.neural_network

    BiasAnalysis(spec, domain=domain, minL=minL, startL=L, startU=U, maxU=maxU, cpu=cpu).main(nn)

if __name__ == '__main__':
    main()
