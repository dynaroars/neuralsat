#!/bin/bash

INSTANCES_NAME="cifar2020_instances"

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GENERATE_PATH=$SCRIPT_PATH"/generate_specs.py"


if [ "$#" == 1 ]; then
    python3 $GENERATE_PATH --dataset cifar10 --n 72 --seed $1 --epsilon 0.00784313725 --network $SCRIPT_PATH/../nets/cifar10_2_255_simplified.onnx --new_instances --instances $SCRIPT_PATH/../$INSTANCES_NAME.csv --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --time_out 100
    python3 $GENERATE_PATH --dataset cifar10 --n 72 --seed $( expr $1 + $1 + 1) --epsilon 0.03137254901 --network $SCRIPT_PATH/../nets/cifar10_8_255_simplified.onnx --instances $SCRIPT_PATH/../$INSTANCES_NAME.csv --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --time_out 100
    python3 $GENERATE_PATH --dataset cifar10 --n 72 --seed $( expr $1 + $1 + $1 + 2) --epsilon 0.00784313725 --network $SCRIPT_PATH/../nets/convBigRELU__PGD.onnx  --instances $SCRIPT_PATH/../$INSTANCES_NAME.csv --time_out 100
    exit 0
elif [ "$#" == 0 ]; then
    python3 $GENERATE_PATH --dataset cifar10 --n 100 --start_idx 0 --epsilon 0.00784313725 --network $SCRIPT_PATH/../nets/cifar10_2_255_simplified.onnx --new_instances --instances $SCRIPT_PATH/../$INSTANCES_NAME.csv --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --dont_extend --time_out 100
    python3 $GENERATE_PATH --dataset cifar10 --n 100 --start_idx 0 --epsilon 0.03137254901 --network $SCRIPT_PATH/../nets/cifar10_8_255_simplified.onnx --instances $SCRIPT_PATH/../$INSTANCES_NAME.csv --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --dont_extend --time_out 100
    python3 $GENERATE_PATH --dataset cifar10 --n 100 --start_idx 0 --epsilon 0.00784313725 --network $SCRIPT_PATH/../nets/convBigRELU__PGD.onnx --instances $SCRIPT_PATH/../$INSTANCES_NAME.csv --dont_extend --time_out 100
    exit 0
else
  echo "Expected either no argument for deterministic generation or just a random seed. (got $# arguments)"
  exit 1
fi

