'''
functions related to loading onnx networks
'''

import time
import numpy as np

from scipy.sparse import csc_matrix, csr_matrix
from scipy import sparse

import onnx
import onnxruntime as ort

from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, select_model_inputs_outputs
from onnx.helper import ValueInfoProto, make_graph, make_model

from nnenum.network import NeuralNetwork, AddLayer, FlattenLayer, ReluLayer, MatMulLayer, FullyConnectedLayer
from nnenum.network import nn_unflatten, nn_flatten
from nnenum.settings import Settings

from nnenum.util import Freezable

class LinearOnnxSubnetworkLayer(Freezable):
    '''a linear layer consisting of multiple onnx operators

    this uses the onnx runtime to execute
    '''

    def __init__(self, layer_num, onnx_submodel):

        self.layer_num = layer_num
        self.network = None # populated later when constructing network

        initializers = [i.name for i in onnx_submodel.graph.initializer]
        inputs = [i for i in onnx_submodel.graph.input if i.name not in initializers]
    
        assert len(inputs) == 1
        assert len(onnx_submodel.graph.output) == 1
                
        self.input_name = inputs[0].name

        self.model_str = onnx_submodel.SerializeToString()
        self.sess = ort.InferenceSession(self.model_str)

        # execute to find output shape and zero_output
        t = inputs[0].type.tensor_type.elem_type

        if t == onnx.TensorProto.FLOAT:
            self.dtype = np.float32
        else:
            assert t == onnx.TensorProto.DOUBLE
            self.dtype = np.float64
        
        input_map = {}

        inp = inputs[0]
        
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        input_map[inp.name] = np.zeros(shape, dtype=self.dtype)

        assert input_map, "didn't find input?"
        self.input_shape = input_map[inp.name].shape
        
        self.zero_output = self.sess.run(None, input_map)[0]
        self.output_shape = self.zero_output.shape

        self.freeze_attrs()

    def __str__(self):
        return f'[LinearOnnxSubnetworkLayer with input {self.get_input_shape()} and output {self.get_output_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.input_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.output_shape

    def transform_star(self, star):
        'transform the star'

        if star.a_mat is None:
            dims = star.lpi.get_num_cols()
            star.a_mat = np.identity(dims, dtype=self.dtype)
            star.bias = np.zeros(dims, dtype=self.dtype)

        cols = []

        for col in range(star.a_mat.shape[1]):
            #print(f".transforming star: {col} / {star.a_mat.shape[1]})")
            vec = star.a_mat[:, col]
            vec = nn_unflatten(vec, self.input_shape)
            
            res = self.execute(vec)
            res = res - self.zero_output
            res = nn_flatten(res)

            cols.append(res)

        dtype = star.bias.dtype
        star.a_mat = np.array(cols, dtype=dtype).transpose()

        vec = nn_unflatten(star.bias, self.input_shape)
        res = self.execute(vec)
        star.bias = nn_flatten(res)

    def transform_zono(self, zono):
        'transform the zono'

        cols = []

        for col in range(zono.mat_t.shape[1]):
            #print(f".transforming zono: {col} / {zono.mat_t.shape[1]})")
            vec = zono.mat_t[:, col]
            vec = nn_unflatten(vec, self.input_shape)

            res = self.execute(vec)

            res = res - self.zero_output
            res = nn_flatten(res)

            cols.append(res)

        dtype = zono.center.dtype
        zono.mat_t = np.array(cols, dtype=dtype).transpose()

        start_center = nn_unflatten(zono.center, self.input_shape)
        end_center = self.execute(start_center)
        zono.center = nn_flatten(end_center)

    def execute(self, state):
        '''execute on a concrete state
 
        returns output
        '''

        assert state.dtype == self.dtype, f"onnx subgraph dtype was {self.dtype}, execute() input was {state.dtype}"
        assert state.shape == self.input_shape, f"expected input shape {self.input_shape}, got {state.shape}"

        rv = self.sess.run(None, {self.input_name: state})[0]

        assert rv.shape == self.output_shape

        return rv

def find_node_with_input(graph, input_name):
    'find the unique onnx node with the given input, can return None'

    rv = None

    for n in graph.node:
        for i in n.input:
            if i == input_name:
                assert rv is None, f"multiple onnx nodes accept network input {input_name}"
                rv = n

    return rv

def convert_model_type_unused(model, from_type=onnx.TensorProto.FLOAT, to_type=onnx.TensorProto.DOUBLE, check_model=True):
    'convert a float32 model to a float64 one'

    assert from_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]
    assert to_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]

    # make a copy
    s = model.SerializeToString()
    model = onnx.ModelProto.FromString(s)

    # we must convert inputs, outputs, initiaizers, and nodes
    new_nodes = []
    new_inputs = []
    new_outputs = []
    new_init = []

    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == from_type:
            inp.type.tensor_type.elem_type = to_type
            
        new_inputs.append(inp)

    for out in model.graph.output:
        if out.type.tensor_type.elem_type == from_type:
            out.type.tensor_type.elem_type = to_type
            
        new_outputs.append(out)

    for node in model.graph.node:
    #    for a in node.attribute:
    #        print(f"attribute:\n{a}")
    #        if a.type == from_type:
    #            a.type = to_type
           
        new_nodes.append(node)

    for init in model.graph.initializer:
        if init.data_type == from_type:
            init.data_type = to_type

            from_dtype = '<f4' if from_type == onnx.TensorProto.FLOAT else '<f8'
            to_dtype = '<f4' if to_type == onnx.TensorProto.FLOAT else '<f8'
            
            # convert raw_data
            b = np.frombuffer(init.raw_data, dtype=from_dtype)
            init.raw_data = b.astype(np.dtype(to_dtype)).tobytes()

        new_init.append(init)

    graph = make_graph(new_nodes, model.graph.name, new_inputs,
                        new_outputs, new_init)

    onnx_model = make_model_with_graph(model, graph, check_model=check_model)

    return onnx_model

def load_onnx_network_optimized(filename):
    '''load an onnx network from a filename and return the NeuralNetwork object

    this has optimized implementation for linear transformations, but it supports less
    layer types than the non-optimized version
    '''

    model = onnx.load(filename)
    onnx.checker.check_model(model)

    graph = model.graph

    #print(graph)

    # find the node with input "input"
    all_input_names = sum([[str(i) for i in n.input] for n in graph.node], [])

    all_initializer_names = [i.name for i in graph.initializer]
    all_output_names = sum([[str(o) for o in n.output] for n in graph.node], [])

    # the input to the network is the one not in all_inputs_list and not in all_outputs_list
    network_input = None
    
    for i in all_input_names:
        if i not in all_initializer_names and i not in all_output_names:
            assert network_input is None, f"multiple onnx network inputs {network_input} and {i}"        
            network_input = i

    assert network_input, "did not find onnx network input"

    assert len(graph.output) == 1, "onnx network defined multiple outputs"
    network_output = graph.output[0].name

    #print(f"input: '{network_input}', output: '{network_output}'")
    
    #assert network_input == graph.input[0].name, \
    #    f"network_input ({network_input}) != graph.input[0].name ({graph.input[0].name})"
    ##########

    # map names -> structs
    input_map = {i.name: i for i in graph.input}
    init_map = {i.name: i for i in graph.initializer}

    i = input_map[network_input]

    # find the node which takes the input (probably node 0)
    cur_node = find_node_with_input(graph, network_input)
    cur_input_name = network_input

    # ok! now proceed recusively
    layers = []

    # data types
    onnx_type_float = 1
    onnx_type_int = 2

    while cur_node is not None:
        assert cur_node.input[0] == cur_input_name, \
            f"cur_node.input[0] ({cur_node.input[0]}) should be previous output ({cur_input_name}) in " + \
            f"node:\n{cur_node.name}"
        
        op = cur_node.op_type
        layer = None

        if layers:
            prev_shape = layers[-1].get_output_shape()
        else:
            s_node = graph.input[0].type.tensor_type.shape
            prev_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in s_node.dim)
            
        if op in ['Add', 'Sub']:
            assert len(cur_node.input) == 2
            init = init_map[cur_node.input[1]]
            assert init.data_type == onnx_type_float

            b = np.frombuffer(init.raw_data, dtype='<f4') # little endian float32
                # note shapes are not reversed here... acasxu input is 1, 1, 1, 5, but dim_value is 1, 1, 1, 5
            shape = tuple(d for d in init.dims) # note dims reversed, acasxu has 5, 50 but want 5 cols
            b = nn_unflatten(b, shape, order='F')

            if op == 'Sub':
                b = -1 * b

            layer = AddLayer(len(layers), b)
        elif op == 'Flatten':
            assert cur_node.attribute[0].i == 1 # flatten along columns

            layer = FlattenLayer(len(layers), prev_shape)
            
        elif op == 'MatMul':
            assert len(cur_node.input) == 2
            init = init_map[cur_node.input[1]]
            
            assert init.data_type == onnx_type_float

            b = np.frombuffer(init.raw_data, dtype='<f4') # little endian float32
            shape = tuple(d for d in reversed(init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols

            b = nn_unflatten(b, shape, order='F')

            layer = MatMulLayer(len(layers), b, prev_shape)
            
        elif op == 'Relu':
            assert layers, "expected previous layer before relu layer"
            
            layer = ReluLayer(len(layers), prev_shape)
        elif op == 'Gemm':
            assert len(cur_node.input) == 3
            
            weight_init = init_map[cur_node.input[1]]
            bias_init = init_map[cur_node.input[2]]

            # weight
            assert weight_init.data_type == onnx_type_float
            b = np.frombuffer(weight_init.raw_data, dtype='<f4') # little endian float32
            shape = tuple(d for d in reversed(weight_init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols
            weight_mat = nn_unflatten(b, shape, order='F')

            # bias
            assert bias_init.data_type == onnx_type_float
            b = np.frombuffer(bias_init.raw_data, dtype='<f4') # little endian float32
            shape = tuple(d for d in reversed(bias_init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols
            bias_vec = nn_unflatten(b, shape, order='F')

            for a in cur_node.attribute:
                assert a.name in ['alpha', 'beta', 'transB'], "general Gemm node unsupported"

                if a.name in ['alpha', 'beta']:
                    assert a.f == 1.0
                    assert a.type == onnx_type_float
                elif a.name == 'transB':
                    assert a.type == onnx_type_int
                    assert a.i == 1
                    weight_mat = weight_mat.transpose().copy()

            layer = FullyConnectedLayer(len(layers), weight_mat, bias_vec, prev_shape)
        else:
            assert False, f"unsupported onnx op_type {op} in node {cur_node.name}"

        assert layer is not None
        layers.append(layer)

        assert len(cur_node.output) == 1, f"multiple output at onnx node {cur_node.name}"
        cur_input_name = cur_node.output[0]

        #print(f"{cur_node.name} -> {cur_input_name}")
        cur_node = find_node_with_input(graph, cur_input_name)

    assert cur_input_name == network_output, \
        f"output witout node {cur_input_name} is not network output {network_output}"

    return NeuralNetwork(layers)

def stan_select_model_inputs_outputs(model, dtype, inputs, outputs, io_shapes):
    """
    a modificiation of select_model_input_outputs from sklearn-on

    Takes a model and changes its inputs and outputs
    :param model: *ONNX* model
    :param inputs: new inputs
    :return: modified model
    The function removes unneeded nodes.
    """

    if dtype == np.float32:
        elem_type = onnx.TensorProto.FLOAT
    else:
        assert dtype == np.float64
        elem_type = onnx.TensorProto.DOUBLE
    
    
    if inputs is None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    
    if not isinstance(inputs, list):
        inputs = [inputs]

    if not isinstance(outputs, list):
        outputs = [outputs]

    ##########

    mark_var = {} # keys are (input or node output) names, vals 1 = keep, 0 = delete
    
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
        
    for inp in model.graph.input:
        mark_var[inp.name] = 0

    for out in outputs:
        if out not in mark_var:
            raise ValueError("Desired Output '{}' not found in model.".format(out))

    initializers = [i.name for i in model.graph.initializer]
        
    for inp in inputs:
        if inp not in mark_var:
            raise ValueError("Desired Input '{}' not found in model.".format(inp))

        if inp not in initializers:
            mark_var[inp] = 1

    nodes = list(enumerate(model.graph.node))
    
    mark_op = {} # these are the marks for the node indices, 1 = keep, 0 = delete
    for node in nodes:
        mark_op[node[0]] = 0

    # We mark all the nodes we need to keep.
    nb = 1 # number marked... used as a termination condition

    keep_initializers = []

    while nb > 0:
        nb = 0

        for index, node in nodes:
            
            if mark_op[index] == 1: # node was already processed, skip
                continue
            
            mod = False # is this a newly-marked node?

            node_initializers = []
            
            for inp in node.input:
                if inp in outputs:
                    continue
                
                if not inp in mark_var or mark_var.get(inp, 0) == 0:
                    node_initializers.append(inp) # was initializer
                elif mark_var[inp] == 1:
                    # make the node because its input was marked
                    mark_op[index] = 1
                    mod = True

            for out in node.output:
                if out in inputs:
                    continue
                
                if mark_var[out] == 1:
                    # mark the node because the output was marked
                    mark_op[index] = 1
                    mod = True
                
            if not mod: # none of the node's inputs were marked, skip it
                continue

            keep_initializers += node_initializers

            nb += 1 # mark the node and all its inputs / outputs
            
            for out in node.output:
                if mark_var.get(out, 0) == 1:
                    continue
                
                if out in outputs:
                    continue

                mark_var[out] = 1
                nb += 1

            for inp in node.input:
                if mark_var.get(inp, 0) == 1:
                    continue
                
                if inp in inputs:
                    continue

                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node[1] for node in nodes if mark_op[node[0]] == 1]

    var_in = []
    for inp in inputs:
        nt = onnx.TypeProto()
        nt.tensor_type.elem_type = elem_type

        # inputs need shape info, which is not in the graph!
        shape = io_shapes[inp]

        for s in shape:
            nt.tensor_type.shape.dim.add()
            nt.tensor_type.shape.dim[-1].dim_value = s

        value_info = ValueInfoProto(type=nt)
        value_info.name = inp
        
        var_in.append(value_info)

    # add initializers to inputs
    for i in model.graph.input:
        if i.name in keep_initializers:
            var_in.append(i)

    var_out = []
    for out in outputs:
        nt = onnx.TypeProto()
        nt.tensor_type.elem_type = elem_type

        # inputs need shape info, which is not in the graph!
        shape = io_shapes[out]

        for s in shape:
            nt.tensor_type.shape.dim.add()
            nt.tensor_type.shape.dim[-1].dim_value = s

        value_info = ValueInfoProto(type=nt)
            
        value_info.name = out
        var_out.append(value_info)
            

    init_out = [init for init in model.graph.initializer if init.name in keep_initializers] 

    graph = make_graph(keep_nodes, model.graph.name, var_in,
                       var_out, init_out)

    #print(f"making model with inputs {inputs} / outputs {outputs} and nodes len: {len(keep_nodes)}")
    onnx_model = make_model_with_graph(model, graph)
        
    return onnx_model

def extract_ordered_relus(model, start):
    '''extract the relu nodes in a topological order

    returns an ordered list of relu nodes (from model.graph.node)
    '''

    relu_nodes = []
    marked_values = [start]
    marked_nodes = []

    modified = True

    while modified:
        modified = False

        for index, node in enumerate(model.graph.node):

            # node was already processed
            if index in marked_nodes:
                continue

            should_process = False

            for inp in node.input:
                if inp in marked_values:
                    should_process = True
                    break

            # none of the node's inputs were marked
            if not should_process:
                continue

            # process the node!
            modified = True
            marked_nodes.append(index)

            if node.op_type == 'Relu':
                relu_nodes.append(node)

            for out in node.output:
                if out in marked_values:
                    continue

                marked_values.append(out)

    return relu_nodes

def make_model_with_graph(model, graph, check_model=True):
    'copy a model with a new graph'

    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    #if len(onnx_model.graph.input) != len(model.graph.input):
    #    raise RuntimeError("Input mismatch {} != {}".format(
    #                    len(onnx_model.input), len(model.input)))

    # fix opset import
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    #print(". making model -------------")
    #onnx.save_model(onnx_model, 'temp.onnx')
    #with open('temp.txt', 'w') as f:
    #    f.write(str(graph))

    if check_model:
        onnx.checker.check_model(onnx_model, full_check=True)

    return onnx_model

def get_io_shapes(model):
    """returns map io_name -> shape"""

    rv = {}

    intermediate_outputs = list(enumerate_model_node_outputs(model))

    initializers = [i.name for i in model.graph.initializer]
    inputs = [i for i in model.graph.input if i.name not in initializers]
    assert len(inputs) == 1

    t = inputs[0].type.tensor_type.elem_type
    assert t == onnx.TensorProto.FLOAT
    dtype = np.float32

    if dtype == np.float32:
        elem_type = onnx.TensorProto.FLOAT
    else:
        assert dtype == np.float64
        elem_type = onnx.TensorProto.DOUBLE

    # create inputs as zero tensors
    input_map = {}

    for inp in inputs:            
        shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
        
        input_map[inp.name] = np.zeros(shape, dtype=dtype)

        # also save it's shape
        rv[inp.name] = shape

    new_out = []

    # add all old outputs
    for out in model.graph.output:
        new_out.append(out)
        
    for out_name in intermediate_outputs:
        if out_name in rv: # inputs were already added
            continue

        # create new output
        #nt = onnx.TypeProto()
        #nt.tensor_type.elem_type = elem_type

        value_info = ValueInfoProto()
        value_info.name = out_name
        new_out.append(value_info)

    # ok run once and get all outputs
    graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
                       new_out, model.graph.initializer)

    # this model is not a valud model since the outputs don't have shape type info...
    # but it still will execute! skip the check_model step
    new_onnx_model = make_model_with_graph(model, graph, check_model=False)
    
    sess = ort.InferenceSession(new_onnx_model.SerializeToString())

    res = sess.run(None, input_map)
    names = [o.name for o in sess.get_outputs()]
    out_map = {name: output for name, output in zip(names, res)}

    for out_name in intermediate_outputs:
        if out_name in rv: # inputs were already added
            continue

        rv[out_name] = out_map[out_name].shape
        
    return rv

def remove_unused_initializers(model):
    'return a modified model'

    new_init = []

    for init in model.graph.initializer:
        found = False
        
        for node in model.graph.node:
            for i in node.input:
                if init.name == i:
                    found = True
                    break

            if found:
                break

        if found:
            new_init.append(init)

    graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
                        model.graph.output, new_init)

    onnx_model = make_model_with_graph(model, graph)

    return onnx_model
    

def load_onnx_network(filename):
    '''load an onnx network from a filename and return the NeuralNetwork object

    This newer version will use the onnx runtime to execute all linear layers,
    spliting the network into parts on the relu layers
    '''

    model = onnx.load(filename)
    onnx.checker.check_model(model)

    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    
    model = remove_unused_initializers(model)
    onnx.checker.check_model(model)

    io_shapes = get_io_shapes(model)

    dtype = np.float32

    initializers = [i.name for i in model.graph.initializer]
    inputs = [i.name for i in model.graph.input if i.name not in initializers]
    
    assert len(inputs) == 1, f"expected one input in onnx model, got {inputs}"
    assert len(model.graph.output) == 1, "expected single output in onnx model"

    graph = model.graph

    for node in graph.node:
        o = node.op_type
        assert o not in Settings.ONNX_BLACKLIST, f"Onnx model contains node with op {o}, which is unsupported. " + \
          "Updated Settings.BLACKLIST if you want to override this."

        assert o in Settings.ONNX_WHITELIST, f"Onnx model contains node with op {o}, which may not be a linear operation. " + \
          "Updated Settings.WHITELIST if you want to override this."
    
    # check to see if only linear nodes are being used
    relus = extract_ordered_relus(model, inputs[0])
    assert relus, "expected at least one relu layer in network"
    
    # split the network input parts along the relus
    prev_input = inputs[0]

    layers = []

    while relus:
        r = relus[0]

        assert r.op_type == 'Relu'
        assert len(r.input) == 1 and len(r.output) == 1
        end_node = r.input[0]

        # in case network starts with relu or there are two relus in a row
        if end_node != prev_input:
            submodel = stan_select_model_inputs_outputs(model, dtype, prev_input, end_node, io_shapes)

            l = LinearOnnxSubnetworkLayer(len(layers), submodel)
            layers.append(l)

        end_shape = io_shapes[end_node]
        l = ReluLayer(len(layers), end_shape)
        layers.append(l)
        
        prev_input = r.output[0]

        # next!
        relus = relus[1:]

    # done with relus... do last subnetwork
    if prev_input != model.graph.output[0].name:
        end_node = model.graph.output[0].name

        submodel = stan_select_model_inputs_outputs(model, dtype, prev_input, end_node, io_shapes)

        l = LinearOnnxSubnetworkLayer(len(layers), submodel)
        layers.append(l)

    return NeuralNetwork(layers)
