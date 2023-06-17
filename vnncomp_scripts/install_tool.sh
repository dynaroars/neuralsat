TOOL_NAME=neuralsat
VERSION_STRING=v1
CONDA_HOME = ~/conda

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi


echo "Installing $TOOL_NAME"


wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
chmod 755 conda.sh
./conda.sh -p $CONDA_HOME

