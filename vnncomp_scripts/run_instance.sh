#!/bin/bash

TOOL_NAME=NeuralSAT
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

# python 
NEURALSAT_PY=python3

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
NEURALSAT_MAIN=$TOOL_DIR/neuralsat-pt201/main.py

# remove old result
if [ -f $RESULTS_FILE ]; then
	rm $RESULTS_FILE
fi

export NEURALSAT_DEBUG=1

echo ""
echo "Running '$TOOL_NAME' in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file '$RESULTS_FILE', and timeout '$TIMEOUT'"
echo ""

$NEURALSAT_PY $NEURALSAT_MAIN --net $ONNX_FILE --spec $VNNLIB_FILE --timeout $TIMEOUT --verbosity=2 --result_file $RESULTS_FILE
