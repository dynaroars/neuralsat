import networkx as nx
import numpy as np
import torch

def _prepare_linear(nx_graph, pre_nodes, layer, layer_id, input_shape, output_shape, offset_node_id, node_to_id, cs=None):
    nodes_array = np.array(pre_nodes)
    assert nodes_array.shape == input_shape
    assert nodes_array.shape[0] == 1

    this_layer_weight = layer.weight.detach().cpu()
    this_layer_bias = layer.bias.detach().cpu()
    if cs is not None:
        this_layer_weight = cs.squeeze(1).mm(this_layer_weight)
        this_layer_bias = cs.squeeze(1).mm(this_layer_bias.unsqueeze(-1)).view(-1)
        output_shape = (1, len(this_layer_bias))
        
    node_labels = []
    for node_id in range(len(this_layer_weight)):
        # bias = this_layer_bias[node_id].item()
        coeffs = this_layer_weight[node_id]
        nodes = nodes_array[0]

        node_label = f'linear_{layer_id}_{node_id}'
        node_to_id[node_label] = node_id + offset_node_id
        nx_graph.add_node(node_to_id[node_label], layer=layer_id)
        assert len(coeffs) == len(nodes)
        for c_, n_ in zip(coeffs, nodes):
            nx_graph.add_edge(node_to_id[node_label], node_to_id[n_], label=c_)
            nx_graph.add_edge(node_to_id[n_], node_to_id[node_label], label=c_)
        # self loop for bias
        c_ = this_layer_bias[node_id]
        nx_graph.add_edge(node_to_id[node_label], node_to_id[node_label], label=c_)
        # assert isinstance()
        node_labels.append(node_label)

    node_labels = np.array(node_labels)
    node_labels = node_labels.reshape(output_shape)
    return node_labels

def _prepare_conv2d(nx_graph, pre_nodes, layer, layer_id, input_shape, output_shape, offset_node_id, node_to_id):
    assert layer.dilation == (1, 1)
    assert len(layer.padding) == 2

    nodes_array = np.array(pre_nodes)
    assert nodes_array.shape == input_shape

    this_layer_weight = layer.weight.detach().cpu().numpy()
    this_layer_bias = layer.bias.detach().cpu().numpy()

    # compute row mapping: from current row to input rows
    in_row_idx_mins = np.arange(output_shape[2]) * layer.stride[0] - layer.padding[0]
    in_row_idx_maxs = in_row_idx_mins + this_layer_weight.shape[2] - 1
    ker_row_mins = np.zeros(output_shape[2], dtype=int)
    ker_row_maxs = np.ones(output_shape[2], dtype=int) * this_layer_weight.shape[2]
    ker_row_mins[in_row_idx_mins < 0] = -in_row_idx_mins[in_row_idx_mins < 0]
    ker_row_maxs[in_row_idx_maxs >= input_shape[2]] = ker_row_maxs[in_row_idx_maxs >= input_shape[2]] - in_row_idx_maxs[in_row_idx_maxs >= input_shape[2]] + input_shape[2] - 1
    in_row_idx_mins = np.maximum(in_row_idx_mins, 0)
    in_row_idx_maxs = np.minimum(in_row_idx_maxs, input_shape[2] - 1)

    # compute column mapping: from current column to input columns
    in_col_idx_mins = np.arange(output_shape[3]) * layer.stride[1] - layer.padding[1]
    in_col_idx_maxs = in_col_idx_mins + this_layer_weight.shape[3] - 1
    ker_col_mins = np.zeros(output_shape[3], dtype=int)
    ker_col_maxs = np.ones(output_shape[3], dtype=int) * this_layer_weight.shape[3]
    ker_col_mins[in_col_idx_mins < 0] = -in_col_idx_mins[in_col_idx_mins < 0]
    ker_col_maxs[in_col_idx_maxs >= input_shape[3]] = ker_col_maxs[in_col_idx_maxs >= input_shape[3]] - in_col_idx_maxs[in_col_idx_maxs >= input_shape[3]] + input_shape[3] - 1
    in_col_idx_mins = np.maximum(in_col_idx_mins, 0)
    in_col_idx_maxs = np.minimum(in_col_idx_maxs, input_shape[3] - 1)

    node_id = 0
    node_labels = []
    for out_chan_idx in range(output_shape[1]):
        out_chan_vars = []
        for out_row_idx in range(output_shape[2]):
            out_row_vars = []
            # get row index range from precomputed arrays
            ker_row_min, ker_row_max = ker_row_mins[out_row_idx], ker_row_maxs[out_row_idx]
            in_row_idx_min, in_row_idx_max = in_row_idx_mins[out_row_idx], in_row_idx_maxs[out_row_idx]
            for out_col_idx in range(output_shape[3]):
                # get col index range from precomputed arrays
                ker_col_min, ker_col_max = ker_col_mins[out_col_idx], ker_col_maxs[out_col_idx]
                in_col_idx_min, in_col_idx_max = in_col_idx_mins[out_col_idx], in_col_idx_maxs[out_col_idx]

                node_label = f'conv2d_{layer_id}_{node_id}'
                node_to_id[node_label] = node_id + offset_node_id
                nx_graph.add_node(node_to_id[node_label], layer=layer_id)

                # init linear constraint LHS implied by the conv operation
                for in_chan_idx in range(this_layer_weight.shape[1]):
                    coeffs = this_layer_weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)
                    # print(f'{coeffs=}')
                    nodes = nodes_array[0][in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                    # print(f'{nodes=}')
                    assert len(coeffs) == len(nodes)
                    for c_, n_ in zip(coeffs, nodes):
                        nx_graph.add_edge(node_to_id[node_label], node_to_id[n_], label=c_)
                        nx_graph.add_edge(node_to_id[n_], node_to_id[node_label], label=c_)
                c_ = this_layer_bias[out_chan_idx]
                nx_graph.add_edge(node_to_id[node_label], node_to_id[node_label], label=c_)

                out_row_vars.append(node_label)
                node_id += 1
            out_chan_vars.append(out_row_vars)
        node_labels.append(out_chan_vars)

    node_labels = np.array(node_labels)
    assert node_labels.shape == output_shape[1:], f'{node_labels.shape=} {output_shape=}'
    node_labels = node_labels.reshape(output_shape)
    return node_labels

def _prepare_input(nx_graph, input_shape, node_to_id):
    assert input_shape[0] == 1

    node_labels = []
    for node_id in range(np.prod(input_shape)):
        node_label = f'input_{node_id}'
        node_to_id[node_label] = node_id
        nx_graph.add_node(node_to_id[node_label], layer=0)
        node_labels.append(node_label)

    node_labels = np.array(node_labels)
    node_labels = node_labels.reshape(input_shape)
    return node_labels

def prepare_graph(net, input_shape, objective):
    G = nx.DiGraph()
    node_to_id = {}

    # process input layer
    layer_node_labels = _prepare_input(G, input_shape, node_to_id=node_to_id)

    # process hidden layers
    cs = None
    layer_id = 1
    pre = torch.randn(input_shape)
    for layer in list(net.modules())[1:]: # TODO: update
        print(f'Processing {layer=} {layer_id=}')
        post = layer(pre)
        if isinstance(layer, torch.nn.Linear):
            if layer_id == len(list(net.modules())[1:]) - 2:
                cs = objective.cs
                print(f'{cs=} {cs.shape=}')
                
            layer_node_labels = _prepare_linear(
                nx_graph=G,
                pre_nodes=layer_node_labels,
                layer=layer,
                layer_id=layer_id,
                input_shape=pre.shape,
                output_shape=post.shape,
                offset_node_id=len(node_to_id),
                node_to_id=node_to_id,
                cs=cs,
            )
        elif isinstance(layer, torch.nn.Conv2d):
            layer_node_labels = _prepare_conv2d(
                nx_graph=G,
                pre_nodes=layer_node_labels,
                layer=layer,
                layer_id=layer_id,
                input_shape=pre.shape,
                output_shape=post.shape,
                offset_node_id=len(node_to_id),
                node_to_id=node_to_id,
            )
        elif isinstance(layer, torch.nn.ReLU):
            continue
        elif isinstance(layer, torch.nn.Flatten):
            layer_node_labels = layer_node_labels.reshape(1, -1)
        else:
            print('Unsupported:', layer)
            raise

        layer_id += 1
        pre = post
        # print(layer, f'{layer_node_labels.shape=}, {pre.shape=}, {post.shape=} {len(node_to_id)=}')

    assert len(list(set(node_to_id.keys()))) == len(node_to_id) # no duplicate names
    assert len(list(set(node_to_id.values()))) == len(node_to_id) # no duplicate values

    return G

def get_edge_weight(G):
    edge_weight = []
    for u, v, data in G.edges(data=True):
        edge_weight.append(data.get('label'))
    return torch.abs(torch.tensor(edge_weight))

def get_edge_index(G):
    node_to_index = {}
    for i, v in enumerate(G.nodes()):
        node_to_index[v] = i
    edge_index = []
    for u, v in G.edges:
        edge_index.append([node_to_index[u], node_to_index[v]])
    return torch.tensor(edge_index).T
