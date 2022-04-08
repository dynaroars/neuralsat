#! /usr/bin/env bash

if [ -d specs ]
then
    rm -rf specs
fi
mkdir specs

seed=$1
python3.8 generate_linf_robustness_query.py --seed $seed
