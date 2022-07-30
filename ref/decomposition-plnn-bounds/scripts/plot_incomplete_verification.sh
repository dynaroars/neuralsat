#!/bin/bash
TEMP_DIR=./results/plots

mkdir -p $TEMP_DIR
TO_LOAD="Gurobi-fromintermediate-fixed.txt,Gurobi_anytime-400steps_equivalent-fromintermediate-fixed.txt,Naive-fromintermediate.txt,KW-fromintermediate.txt"
TO_LOAD="$TO_LOAD,DJ_Adam_520-fromintermediate.txt,DJ_Adam_toDecomposition_520-fromintermediate.txt,Adam_fixed_370-fromintermediate.txt,Proximal_momentum_200-fromintermediate.txt"
TO_LOAD="$TO_LOAD,DJ_Adam_1040-fromintermediate.txt,DJ_Adam_toDecomposition_1040-fromintermediate.txt,Adam_fixed_740-fromintermediate.txt,Proximal_momentum_400-fromintermediate.txt"

NAMES="Gurobi,Gurobi-TL,IP,WK"
NAMES="$NAMES,DSG+\n520 steps,Dec-DSG+\n520 steps,Supergradient\n370 steps,Proximal\n200 steps"
NAMES="$NAMES,DSG+\n1040 steps,Dec-DSG+\n1040 steps,Supergradient\n740 steps,Proximal\n400 steps"

taskset -c 3,5 python tools/parse_bounds.py ./results/madry8/ $TEMP_DIR/madry8_cr "$TO_LOAD" "$NAMES"
taskset -c 3,5 python tools/parse_bounds.py ./results/sgd8/ $TEMP_DIR/sgd8_cr "$TO_LOAD" "$NAMES"