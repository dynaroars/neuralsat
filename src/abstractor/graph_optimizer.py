"""Graph optimizers for BoundedModule."""

from .auto_LiRPA.bound_ops import (
    BoundReduceSum, BoundRelu, BoundConstant, BoundBuffers, BoundMul,
    BoundSub, BoundUnsqueeze, BoundMultiPiecewiseNonlinear, BoundAdd,
)
from .auto_LiRPA.operators.relu import BoundSign, BoundSignMerge


def merge_relu_lookup_table(model):
    """Merge ReLU-based lookup tables into BoundMultiPiecewiseNonlinear."""
    merged = 0
    nodes = list(model.nodes())
    for node in nodes:
        if (isinstance(node, BoundReduceSum)
                and isinstance(node.inputs[0], BoundMul)
                and isinstance(node.inputs[1], BoundConstant)
                and node.inputs[1].value.item() == -1):
            node_mul = node.inputs[0]
            if (isinstance(node_mul.inputs[1], BoundBuffers)
                    and node_mul.inputs[1].buffer.ndim == 1
                    and isinstance(node_mul.inputs[0], BoundRelu)
                    and isinstance(node_mul.inputs[0].inputs[0], BoundSub)):
                node_sub = node_mul.inputs[0].inputs[0]
                node_weight = node_mul.inputs[1]
                if (isinstance(node_sub.inputs[1], BoundBuffers)
                        and node_sub.inputs[1].buffer.ndim == 1
                        and isinstance(node_sub.inputs[0], BoundUnsqueeze)
                        and not node_sub.inputs[0].inputs[1].perturbed
                        and node_sub.inputs[0].inputs[1].value == -1
                        and len(node_sub.inputs[0].inputs[0].output_shape) == 2):
                    node_offset = node_sub.inputs[1]
                    node_input = node_sub.inputs[0].inputs[0]
                    node_merged = BoundMultiPiecewiseNonlinear(
                        inputs=[node_input, node_weight, node_offset])
                    node_merged.name = f'{node.name}/merged'
                    model.add_nodes([node_merged])
                    model.replace_node(node, node_merged)
                    merged += 1
    return merged


def merge_sign(model):
    """Merge Sign-Add-Sign STE patterns into BoundSignMerge."""
    merged = 0
    nodes = list(model.nodes())
    for i, node in enumerate(nodes):
        if (i + 2 < len(nodes) and isinstance(node, BoundSign)
                and isinstance(nodes[i + 1], BoundAdd)
                and isinstance(nodes[i + 2], BoundSign)):
            node_merge = BoundSignMerge(inputs=[node.inputs[0]], options=model.bound_opts)
            node_merge.name = f'{node.name}/merge'
            model.add_nodes([node_merge])
            model.replace_node(node, node_merge)
            model.replace_node(nodes[i + 1], node_merge)
            model.replace_node(nodes[i + 2], node_merge)
            merged += 1
    return merged
