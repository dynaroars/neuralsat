#!/bin/bash

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

echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
#DIR=$(dirname $(realpath $0))
#export PYTHONPATH="$PYTHONPATH:$DIR/src"

# run the tool to produce the results file

current_dir=`pwd`
cd $(dirname $0)/src
cat /dev/null > "$current_dir/$RESULTS_FILE"
pipenv run python -O scripts/benchmark_vnn.py "$current_dir/$ONNX_FILE" "$current_dir/$VNNLIB_FILE" "$current_dir/$RESULTS_FILE" $TIMEOUT
