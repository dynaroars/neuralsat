from __future__ import annotations

import onnx
import numpy as np
from onnx import numpy_helper as nh


def _create_initializer_tensor(name: str, tensor_array: np.ndarray) -> onnx.TensorProto:
    if tensor_array.dtype in ('float32', 'float64'):
        data_type = onnx.TensorProto.FLOAT
    elif tensor_array.dtype == 'int64':
        data_type = onnx.TensorProto.INT64
    else:
        data_type = onnx.TensorProto.FLOAT
    return onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )


def _create_new_initializers(node, initializers: dict) -> list:
    new_initializers = []
    for old_init in node.input:
        if old_init in initializers:
            new_initializers.append(_create_initializer_tensor(
                name=old_init, tensor_array=initializers[old_init]))
    return new_initializers


def optimize_remove_matmul_inplace(
        onnx_model: onnx.ModelProto,
        save_path: str | None = None,
) -> onnx.ModelProto:
    """Fuse Transpose-MatMul-Transpose patterns into a single MatMul."""
    initializers = {init.name: nh.to_array(init) for init in onnx_model.graph.initializer}
    nodes = list(onnx_model.graph.node)

    to_transpose_const: dict[int, int] = {}
    temp_dict: dict[str, int] = {}
    for i, node in enumerate(nodes):
        if node.op_type == 'Constant':
            temp_dict[node.output[0]] = i
        matmul_case1 = (
            node.op_type == 'MatMul' and i - 1 >= 0 and i + 1 < len(nodes)
            and nodes[i - 1].op_type == 'Transpose'
            and nodes[i + 1].op_type == 'Transpose'
            and nodes[i - 1].output[0] == node.input[1]
        )
        matmul_case2 = (
            node.op_type == 'MatMul' and i - 2 >= 0 and i + 1 < len(nodes)
            and nodes[i - 2].op_type == 'Transpose'
            and nodes[i + 1].op_type == 'Transpose'
            and nodes[i - 2].output[0] == node.input[1]
        )
        if matmul_case1 or matmul_case2:
            to_transpose_const[i] = temp_dict[node.input[0]]

    new_nodes: list = []
    new_initializers: list = []
    cnt = 0

    for i, node in enumerate(nodes):
        transpose_case1 = (
            node.op_type == 'Transpose' and i - 2 >= 0
            and nodes[i - 1].op_type == 'MatMul' and nodes[i - 2].op_type == 'Transpose'
        )
        transpose_case2 = (
            node.op_type == 'Transpose' and i - 3 >= 0
            and nodes[i - 1].op_type == 'MatMul' and nodes[i - 3].op_type == 'Transpose'
        )
        transpose_case3 = (
            node.op_type == 'Transpose' and i + 2 < len(nodes)
            and nodes[i + 1].op_type == 'MatMul' and nodes[i + 2].op_type == 'Transpose'
        )
        transpose_case4 = (
            node.op_type == 'Transpose' and i + 3 < len(nodes)
            and nodes[i + 2].op_type == 'MatMul' and nodes[i + 3].op_type == 'Transpose'
        )
        matmul_case1 = (
            node.op_type == 'MatMul' and i - 1 >= 0 and i + 1 < len(nodes)
            and nodes[i - 1].op_type == 'Transpose'
            and nodes[i + 1].op_type == 'Transpose'
            and nodes[i - 1].output[0] == node.input[1]
        )
        matmul_case2 = (
            node.op_type == 'MatMul' and i - 2 >= 0 and i + 1 < len(nodes)
            and nodes[i - 2].op_type == 'Transpose'
            and nodes[i + 1].op_type == 'Transpose'
            and nodes[i - 2].output[0] == node.input[1]
        )

        if transpose_case1 or transpose_case2 or transpose_case3 or transpose_case4:
            continue
        if node.op_type == 'Constant' and i in list(to_transpose_const.values()):
            continue
        if matmul_case1 or matmul_case2:
            source_input = (
                nodes[i - 1].input[0] if matmul_case1 else nodes[i - 2].input[0]
            )
            const_node = nodes[to_transpose_const[i]]
            val = nh.to_array(const_node.attribute[0].t).transpose(1, 0)
            new_tensor = _create_initializer_tensor(name=f'Constant{cnt}', tensor_array=val)
            matmul_node = onnx.helper.make_node(
                name=f'linear{cnt}_MatMul',
                op_type='MatMul',
                inputs=[source_input, f'Constant{cnt}'],
                outputs=[nodes[i + 1].output[0]],
            )
            new_nodes.append(matmul_node)
            new_initializers.append(new_tensor)
            cnt += 1
        else:
            new_nodes.append(node)
            new_initializers.extend(_create_new_initializers(node, initializers))

    seen = set()
    deduped_initializers = []
    for init in new_initializers:
        if init.name not in seen:
            seen.add(init.name)
            deduped_initializers.append(init)

    new_graph = onnx.helper.make_graph(
        name='OptimizedNet',
        nodes=new_nodes,
        inputs=list(onnx_model.graph.input),
        outputs=list(onnx_model.graph.output),
        initializer=deduped_initializers,
    )
    model_def = onnx.helper.make_model(new_graph, producer_name='neuralsat_onnx_opt')
    model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    if save_path is not None:
        onnx.save(model_def, save_path)
    return model_def


def optimize_fix_gtrsb(
        onnx_model: onnx.ModelProto,
        save_path: str | None = None,
) -> onnx.ModelProto:
    """Convert GTRSB/traffic-sign NHWC ONNX graphs to NCHW for pytorch."""
    import copy
    initializers = {init.name: nh.to_array(init) for init in onnx_model.graph.initializer}
    nodes = list(onnx_model.graph.node)
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    place_holder = [d.dim_value if d.dim_value > 0 else 1 for d in dims]

    new_nodes: list = []
    new_initializers: list = []
    cnt = 0
    existing_init_names: set[str] = set()

    for i, node in enumerate(nodes):
        if node.op_type == 'Transpose':
            if i + 1 < len(nodes):
                nodes[i + 1].input[0] = node.input[0]
            continue
        if node.op_type == 'MatMul' and cnt == 0:
            w_cur = initializers[node.input[1]]
            shape_now = w_cur.shape
            if shape_now[0] == 23328:
                w_cur = w_cur.reshape(1, 27, 27, 32, -1)
            elif shape_now[0] == 3136:
                w_cur = w_cur.reshape(1, 7, 7, 64, -1)
            elif shape_now[0] == 1600:
                w_cur = w_cur.reshape(1, 5, 5, 64, -1)
            else:
                raise ValueError(f'Unexpected GTRSB MatMul weight shape: {shape_now}')
            w = w_cur.transpose(0, 3, 1, 2, 4).reshape(-1, shape_now[-1])
            init_name = f'matmul{cnt}_W'
            new_initializers.append(_create_initializer_tensor(name=init_name, tensor_array=w))
            existing_init_names.add(init_name)
            new_nodes.append(onnx.helper.make_node(
                name=f'matmul{cnt}_MatMul',
                op_type='MatMul',
                inputs=[node.input[0], init_name],
                outputs=[node.output[0]],
            ))
            cnt += 1
        else:
            new_node = copy.deepcopy(node)
            for j in range(len(new_node.input)):
                if new_node.input[j] in initializers and new_node.input[j] not in existing_init_names:
                    new_initializers.append(_create_initializer_tensor(
                        name=new_node.input[j], tensor_array=initializers[new_node.input[j]]))
                    existing_init_names.add(new_node.input[j])
            new_nodes.append(new_node)

    for i, node in enumerate(new_nodes):
        if node.op_type == 'Reshape':
            tmp = node.input[0]
            node.input[0] = new_nodes[i + 3].output[0]
            new_nodes[i + 1].input[0] = tmp
            new_nodes[i + 4].input[0] = node.output[0]
            new_nodes[i] = new_nodes[i + 1]
            new_nodes[i + 1] = new_nodes[i + 2]
            new_nodes[i + 2] = new_nodes[i + 3]
            new_nodes[i + 3] = node
            break

    input_node = onnx.helper.make_tensor_value_info(
        name=onnx_model.graph.input[0].name,
        elem_type=onnx.TensorProto.FLOAT,
        shape=('unk__195', place_holder[3], place_holder[1], place_holder[2]),
    )
    model_def = onnx.helper.make_model(
        onnx.helper.make_graph(
            name='GtrsbNet',
            nodes=new_nodes,
            inputs=[input_node],
            outputs=list(onnx_model.graph.output),
            initializer=new_initializers,
        ),
        producer_name='neuralsat_onnx_opt',
    )
    model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    if save_path is not None:
        onnx.save(model_def, save_path)
    return model_def
