#!/bin/bash

VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

set -e
set -x

SCRIPT_DIR=`dirname $(readlink -f $0)`

sudo apt-get update

sudo apt-get remove -y python2.7 python3.6
sudo apt-get autoremove -y
sudo apt-get install -y python3.8 python3.8-dev gfortran python3-pip bc

python3.8 -m pip install pip
sudo -H python3.8 -m pip install -U pipenv
cd src
pipenv install
pipenv_python=`pipenv run which python`

cd ~/
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/opt/openblas install
sudo sh -c 'echo "/opt/openblas/lib" > /etc/ld.so.conf.d/openblas.conf'

$pipenv_python -m pip install cython
cd ~/
git clone https://github.com/numpy/numpy
cd numpy/
git checkout v1.21.0
echo "[default]
include_dirs = /opt/openblas/include
library_dirs = /opt/openblas/lib
 
[openblas]
openblas_libs = openblas
library_dirs = /opt/openblas/lib
 
[lapack]
lapack_libs = openblas
library_dirs = /opt/openblas/lib" > site.cfg
$pipenv_python setup.py build
$pipenv_python setup.py install

cd ~/
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xzvf gurobi9.1.2_linux64.tar.gz 
rm gurobi9.1.2_linux64.tar.gz 
sudo mv gurobi912/ /opt/ 
cd /opt/gurobi912/linux64/
$pipenv_python setup.py install
