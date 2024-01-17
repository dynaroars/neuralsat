if [[ -z "${DNNV_PYTHON}" ]]; then
	DNNV_PYTHON=${HOME}/anaconda3/envs/dnnv/bin/python3
fi

# echo $DNNV_PYTHON
$DNNV_PYTHON -c "import dnnv; print('DNNV version', dnnv.__version__)"
$DNNV_PYTHON -c "import onnxsim; print('ONNXSIM version', onnxsim.__version__)"

CATEGORY=$1
ONNX_FILE=$2

echo "Simplifying for benchmark '$CATEGORY' with onnx file '$ONNX_FILE'"

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
OUTPUT_DIR=$TOOL_DIR/tmp_simplified_model_output

echo TOOL_DIR = $TOOL_DIR
echo OUTPUT_DIR = $OUTPUT_DIR

if [ -d $OUTPUT_DIR ]; then
	rm -r $OUTPUT_DIR
fi

if [ "${CATEGORY,,}" == "vggnet16" ] || [ "${CATEGORY,,}" == "cgan" ]; then
    $DNNV_PYTHON $TOOL_DIR/neuralsat-pt201/util/network/simplify_onnx.py $ONNX_FILE $OUTPUT_DIR/model-simplified
fi
exit 0
