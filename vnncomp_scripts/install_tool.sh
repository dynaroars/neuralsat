TOOL_NAME=neuralsat
VERSION_STRING=v1

CONDA_HOME=~/conda
CONDA=$CONDA_HOME/bin/conda

NEURALSAT_CONDA_HOME=~/.conda/envs/neuralsat
NEURALSAT_PY=$NEURALSAT_CONDA_HOME/bin/python

NEURALSAT_HOME=$(dirname $(pwd))
NEURALSAT_MAIN=$NEURALSAT_HOME/neuralsat/main.py


# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi


echo "======= Installing $TOOL_NAME ======="
echo "Neuralsat home is                           $NEURALSAT_HOME"
echo "Conda will be installed to                  $CONDA_HOME"
echo "Neuralsat's conda env will be installed at  $NEURALSAT_CONDA_HOME"



wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
chmod 755 conda.sh

# install conda
rm -rf $CONDA_HOME
./conda.sh -bf -p $CONDA_HOME
rm ./conda.sh

# install env
echo "======= Installing conda env ======="
$CONDA env remove -p $NEURALSAT_CONDA_HOME
rm -rf $NEURALSAT_CONDA_HOME
$CONDA  env create -p $NEURALSAT_CONDA_HOME -f ../env.yaml


# setup alias
echo "======= Exporting env variables ======="
alias neuralsat='$NEURALSAT_PY $NEURALSAT_MAIN'
sed -i '/neuralsat/Id' ~/.bashrc
echo "####         Neuralsat stuff            ####" >> ~/.bashrc
echo "alias neuralsat='$NEURALSAT_PY $NEURALSAT_MAIN'" >> ~/.bashrc

export NEURALSAT_CONDA_HOME
export NEURALSAT_HOME
export NEURALSAT_PY
export NEURALSAT_MAIN
echo "export NEURALSAT_CONDA_HOME=$NEURALSAT_CONDA_HOME" >> ~/.bashrc
echo "export NEURALSAT_HOME=$NEURALSAT_HOME" >>  ~/.bashrc
echo "export NEURALSAT_PY=$NEURALSAT_PY" >> ~/.bashrc
echo "export NEURALSAT_MAIN=$NEURALSAT_MAIN" >> ~/.bashrc

echo "####       End of Neuralsat stuff       ####" >> ~/.bashrc

# install gurobi license
echo "======= Installing Gurobi License ======="
if [ -e ~/gurobi.lic ]; then
    echo "Gurobi license detected ~/gurobi.lic"
else
    echo "No Gurobi license found...enter it now"
    echo "enter WLSACCESSID"
    read ACCESSID
    echo "enter WLSSECRET"
    read SECRET
    echo "enter LICENSEID"
    read LICENSEID
    echo "WLSACCESSID=$ACCESSID" > ~/gurobi.lic
    echo "WLSSECRET=$SECRET" >> ~/gurobi.lic
    echo "LICENSEID=$LICENSEID" >> ~/gurobi.lic
fi

