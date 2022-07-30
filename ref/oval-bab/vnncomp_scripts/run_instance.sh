#!/bin/bash
# six arguments, first is "v1", second is a benchmark category itentifier string such as "acasxu", third is path to the .onnx file, fourth is path to .vnnlib file, fifth is a path to the results file, and sixth is a timeout in seconds.

TOOL_NAME=oval
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

echo "Running $TOOL_NAME on benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

DIR=$(dirname $(dirname $(realpath $0)))
CONFIGS_PATH="${DIR}/bab_configs/"

# Run OVAL BaB on the prepared network/property in canonical form.
eval "$(conda shell.bash hook)"  # Allow for conda activate within bash script
conda activate oval
python "${DIR}/tools/bab_tools/bab_from_vnnlib.py" --mode run_instance --from_pickle --onnx "${ONNX_FILE}" --vnnlib "${VNNLIB_FILE}" --vnncomp_category "${CATEGORY}" --configs_path "${CONFIGS_PATH}" --result_file "${RESULTS_FILE}" --instance_timeout "${TIMEOUT}"
conda deactivate