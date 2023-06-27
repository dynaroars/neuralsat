#!/bin/bash

VERSION_STRING=v1

if [[ -z "${NEURALSAT_PY}" ]]; then
	NEURALSAT_PY=~/.conda/envs/neuralsat/bin/python
fi

if [[ -z "${NEURALSAT_MAIN}" ]]; then
        NEURALSAT_MAIN=$(dirname $(pwd))/neuralsat/main.py
fi


# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

#echo $NEURALSAT_PY
#echo $NEURALSAT_MAIN


CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running Neuralsat in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"


rm -f $RESULTS_FILE

$NEURALSAT_PY $NEURALSAT_MAIN --net $ONNX_FILE --spec $VNNLIB_FILE --timeout $TIMEOUT --verbosity=0  >  $RESULTS_FILE  2>/dev/null

if [ $? -ne 0 ]; then
  echo error > $RESULTS_FILE
fi
