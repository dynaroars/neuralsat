#!/bin/bash
# four arguments, first is "v1", second is a benchmark category identifier string such as "acasxu", third is path to the .onnx file and fourth is path to .vnnlib file

TOOL_NAME=oval
VERSION_STRING=v1
NON_SUPPORTED_CAT=cifar10_resnet

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

# return 1 for the non-supported cifar10_resnet category
if [ "$2" = ${NON_SUPPORTED_CAT} ]; then
	echo "'$2' benchmark category not supported"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

echo "Preparing $TOOL_NAME for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"
DIR=$(dirname $(dirname $(realpath $0)))

# kill any zombie processes
killall -q python

# Convert the network into the canonical form (see https://arxiv.org/pdf/1909.06588.pdf, section 3.1).
eval "$(conda shell.bash hook)"  # Allow for conda activate within bash script
conda activate oval
python "${DIR}/tools/bab_tools/bab_from_vnnlib.py" --mode prepare --onnx "${ONNX_FILE}" --vnnlib "${VNNLIB_FILE}"
conda deactivate

# script returns a 0 exit code if successful.
exit 0