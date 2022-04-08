#!/bin/bash -e
# generate random benchmarks based on seed

SEED=0

if [ "$#" -eq 1 ]; then
    SEED=$1
fi


pushd mnistfc
#pip3 install torch==1.8.1
python3 generate_properties.py $SEED
popd


pushd eran/src
rm -f ../specs/mnist/*.vnnlib
./specs_from_seed.sh $SEED
popd

pushd cifar2020/src
rm -f ../specs/cifar10/*.vnnlib
#./specs_from_seed.sh $SEED
./specs_from_seed.sh
echo "WARNING: CIFAR2020 not using random seed (first 100 images)"
popd

pushd oval21
rm -f vnnlib/*.vnnlib
CUBLAS_WORKSPACE_CONFIG=:4096:8 eval 'python3 generate_properties.py --seed $SEED'
popd

pushd cifar10_resnet/pytorch_model
rm -f ../vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/*.vnnlib
rm -f ../vnnlib_properties_pgd_filtered/resnet4b_pgd_filtered/*.vnnlib
python3 generate_properties_pgd.py --seed $SEED --device cpu
popd

pushd marabou-cifar10
./generate_benchmarks.sh $SEED
popd

pushd verivital
python3 generate_properties.py --seed $SEED
popd

echo "Successfully generated all random instances."
echo "WARNING: CIFAR2020 not using random seed (first 100 images)"
