#!/bin/bash

TOOL_NAME=NeuralSAT
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
SCRIPT_DIR=$(dirname $(realpath $0))
OUTPUT_DIR=$TOOL_DIR/tmp_simplified_model_output
NEURALSAT_MAIN=$TOOL_DIR/neuralsat-pt201/main.py

# remove old result
if [ -f $RESULTS_FILE ]; then
	rm $RESULTS_FILE
fi

echo ""
echo "Running '$TOOL_NAME' in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file '$RESULTS_FILE', and timeout '$TIMEOUT'"
echo ""

python3 $NEURALSAT_MAIN --net $ONNX_FILE --spec $VNNLIB_FILE --timeout $TIMEOUT --verbosity=2 --result_file $RESULTS_FILE

# success
if [ -f $RESULTS_FILE ]; then
	exit 0
fi

echo ""
echo "[!] Result file '$RESULTS_FILE' doesn't exist. Attempting to simplify."
echo ""

# simplify network if needed
$SCRIPT_DIR/prepare_instance.sh $CATEGORY $ONNX_FILE

# DNNV version
DNNV_ONNX_SIM=$OUTPUT_DIR/model-simplified-dnnv.onnx
if [ -f $DNNV_ONNX_SIM ]; then
	echo ""
	echo "Running '$TOOL_NAME' in category '$CATEGORY' with onnx file '$DNNV_ONNX_SIM', vnnlib file '$VNNLIB_FILE', results file '$RESULTS_FILE', and timeout '$TIMEOUT'"
	echo ""

	python3 $NEURALSAT_MAIN --net $DNNV_ONNX_SIM --spec $VNNLIB_FILE --timeout $TIMEOUT --verbosity=2 --result_file $RESULTS_FILE

	# success
	if [ -f $RESULTS_FILE ]; then
	    rm -r $OUTPUT_DIR
		exit 0
	fi
fi

# ONNXSIM version
ONNXSIM_ONNX_SIM=$OUTPUT_DIR/model-simplified-onnxsim.onnx
if [ -f $ONNXSIM_ONNX_SIM ]; then
	echo ""
	echo "Running '$TOOL_NAME' in category '$CATEGORY' with onnx file '$ONNXSIM_ONNX_SIM', vnnlib file '$VNNLIB_FILE', results file '$RESULTS_FILE', and timeout '$TIMEOUT'"
	echo ""
	
	# success
	python3 $NEURALSAT_MAIN --net $ONNXSIM_ONNX_SIM --spec $VNNLIB_FILE --timeout $TIMEOUT --verbosity=2 --result_file $RESULTS_FILE

	if [ -f $RESULTS_FILE ]; then
	    rm -r $OUTPUT_DIR
		exit 0
	fi
fi

if [ -d $OUTPUT_DIR ]; then
	rm -r $OUTPUT_DIR
fi

