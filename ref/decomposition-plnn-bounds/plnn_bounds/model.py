import math
import scipy.io
import torch

from collections import Counter, defaultdict
from plnn_bounds.modules import View, Flatten
from plnn_bounds.naive_approximation import NaiveNetwork
from torch import nn

def no_grad(mod):
    for param in mod.parameters():
        param.requires_grad = False

def cifar_model_large():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model



def load_adversarial_problem(filename, cls):
    if filename.endswith('mini.pth'):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 2, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8*4*4,50),
            nn.ReLU(),
            nn.Linear(50,10),
        )
        model.load_state_dict(torch.load(filename)['state_dict'][0])
        no_grad(model)
        dataset = torch.load('./data/mini_mnist_test.pt')

    elif filename.endswith('small.pth'):
        model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32*7*7,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        model.load_state_dict(torch.load(filename)['state_dict'][0])
        no_grad(model)
        # from torchvision import datasets, transforms
        # ds = datasets.MNIST('./data', train=True, download=True)
        # train_ds = {'data': ds.train_data.unsqueeze(1).float()/255.0,
        #             'labels': ds.train_labels}
        # torch.save(train_ds, './data/mnist_train.pt')
        # ds = datasets.MNIST('./data', train=False, download=True)
        # test_ds = {'data': ds.test_data.unsqueeze(1).float() / 255.0,
        #             'labels': ds.test_labels}
        # torch.save(test_ds, './data/mnist_test.pt')

        dataset = torch.load('./data/mnist_test.pt')
    else:
        raise NotImplementedError

    data = dataset['data']
    labels = dataset['labels']

    sample = data[0].type(torch.Tensor().type())
    label = int(labels[0])
    adv_label = 0
    if label == adv_label:
        adv_label += 1
    eps = 0.1

    # Create the input domain to the verification
    domain = torch.stack([torch.clamp(sample - eps, 0, None),
                          torch.clamp(sample + eps, None, 1.0)], -1)
    # Adjust the convolutional bound so as to make it mono-objective, just for
    # the target label.
    layers = [lay for lay in model]
    assert isinstance(layers[-1], nn.Linear)
    old_last = layers[-1]
    new_last = nn.Linear(old_last.in_features, 1)
    no_grad(new_last)

    new_last.weight.copy_(old_last.weight[label] - old_last.weight[adv_label])
    new_last.bias.copy_(old_last.bias[label] - old_last.bias[adv_label])

    layers[-1] = new_last

    return cls(layers), domain


class AcasNetwork:
    def __init__(self, rpx_infile):
        readline = lambda: rpx_infile.readline().strip()

        line = readline()

        # Ignore the comments
        while line.startswith('//'):
            line = readline()

        # Parse the dimensions
        all_dims = [int(dim) for dim in line.split(',')
                    if dim != '']
        self.nb_layers, self.input_size, \
            self.output_size, self.max_lay_size = all_dims

        # Get the layers size
        line = readline()
        self.nodes_in_layer = [int(l_size_str) for l_size_str in line.split(',')
                               if l_size_str != '']
        assert(self.input_size == self.nodes_in_layer[0])
        assert(self.output_size == self.nodes_in_layer[-1])

        # Load the symmetric parameter
        line = readline()
        is_symmetric = int(line.split(',')[0]) != 0
        # if symmetric == 1, enforce that psi (input[2]) is positive
        # if to do so, it needs to be flipped, input[1] is also adjusted
        # In practice, all the networks released with Reluxplex 1.0 have it as 0
        # so we will just ignore it.


        # Load Min/Max/Mean/Range values of inputs
        line = readline()
        self.inp_mins = [float(min_str) for min_str in line.split(',')
                         if min_str != '']
        line = readline()
        self.inp_maxs = [float(max_str) for max_str in line.split(',')
                         if max_str != '']
        line = readline()
        self.inpout_means = [float(mean_str) for mean_str in line.split(',')
                             if mean_str != '']
        line = readline()
        self.inpout_ranges = [float(range_str) for range_str in line.split(',')
                              if range_str != '']
        assert(len(self.inp_mins) == len(self.inp_maxs))
        assert(len(self.inpout_means) == len(self.inpout_ranges))
        assert(len(self.inpout_means) == (len(self.inp_mins) + 1))

        # Load the weights
        self.parameters = []
        for layer_idx in range(self.nb_layers):
            # Gather weight matrix
            weights = []
            biases = []
            for tgt_neuron in range(self.nodes_in_layer[layer_idx+1]):
                line = readline()
                to_neuron_weights = [float(wgt_str) for wgt_str in line.split(',')
                                     if wgt_str != '']
                assert(len(to_neuron_weights) == self.nodes_in_layer[layer_idx])
                weights.append(to_neuron_weights)
            for tgt_neuron in range(self.nodes_in_layer[layer_idx+1]):
                line = readline()
                neuron_biases = [float(bias_str) for bias_str in line.split(',')
                                 if bias_str != '']
                assert(len(neuron_biases) == 1)
                biases.append(neuron_biases[0])
            assert(len(weights) == len(biases))
            self.parameters.append((weights, biases))

    def write_rlv_file(self, rlv_outfile):
        write_line = lambda x: rlv_outfile.write(x + '\n')

        layers_var_name = []

        # Write down all the inputs
        inp_layer = []
        for inp_idx in range(self.input_size):
            new_var_name = f"in_{inp_idx}"
            inp_layer.append(new_var_name)
            write_line(f"Input {new_var_name}")
        layers_var_name.append(inp_layer)

        # Write down the rescaled version of the inputs
        resc_inp_layer = []
        for inp_idx in range(self.input_size):
            new_var_name = f"resc_inX{inp_idx}"
            resc_inp_layer.append(new_var_name)

            scale = 1.0 / self.inpout_ranges[inp_idx]
            bias = - scale * self.inpout_means[inp_idx]
            prev_var = layers_var_name[-1][inp_idx]
            write_line(f"Linear {new_var_name} {bias} {scale} {prev_var}")
        layers_var_name.append(resc_inp_layer)


        # Write down the linear/ReLU layers
        for layer_idx in range(self.nb_layers):
            lin_weights, bias = self.parameters[layer_idx]

            layer_type = "Linear" if (layer_idx == self.nb_layers-1) else "ReLU"
            name_prefix = "outnormed" if (layer_idx == self.nb_layers-1) else "relu"
            prev_lay_vars = layers_var_name[-1]
            nb_nodes_from = self.nodes_in_layer[layer_idx]
            nb_nodes_to_write = self.nodes_in_layer[layer_idx+1]

            assert(len(lin_weights) == nb_nodes_to_write)
            assert(len(bias) == nb_nodes_to_write)
            for node_weight in lin_weights:
                assert(len(node_weight) == nb_nodes_from)
            assert(len(node_weight) == len(prev_lay_vars))

            relu_layer = []
            for neur_idx in range(nb_nodes_to_write):
                new_var_name = f"{name_prefix}_{layer_idx}X{neur_idx}"

                node_line = f"{layer_type} {new_var_name}"

                node_bias = bias[neur_idx]
                node_line += f" {node_bias}"
                for edge_weight, prev_var in zip(lin_weights[neur_idx],
                                                 prev_lay_vars):
                    node_line += f" {edge_weight} {prev_var}"

                relu_layer.append(new_var_name)
                write_line(node_line)
            layers_var_name.append(relu_layer)

        # Write down the output rescaling
        unscaled_outvar = layers_var_name[-1]
        assert(len(unscaled_outvar) == self.output_size)

        # The means/ranges are given as:
        # in0 in1 ... inLast out ??? ???
        # There is a bunch of random variables at the end that are useless
        output_bias = self.inpout_means[self.input_size]
        output_scale = self.inpout_ranges[self.input_size]
        out_vars = []
        for out_idx in range(self.output_size):
            new_var_name = f"out_{out_idx}"
            prev_var = unscaled_outvar[out_idx]

            out_vars.append(new_var_name)
            write_line(f"Linear {new_var_name} {output_bias} {output_scale} {prev_var}")
        layers_var_name.append(out_vars)

        # Write down the constraints that we know
        inp_vars = layers_var_name[0]
        for inp_idx in range(self.input_size):
            var_name = inp_vars[inp_idx]
            # Min-constraint
            min_val = self.inp_mins[inp_idx]
            min_constr = f"Assert <= {min_val} 1.0 {var_name}"
            write_line(min_constr)
            # Max-constraint
            max_val = self.inp_maxs[inp_idx]
            max_constr = f"Assert >= {max_val} 1.0 {var_name}"
            write_line(max_constr)


GE='>='
LE='<='
COMPS = [GE, LE]

def load_rlv(rlv_infile):
    # This parser only makes really sense in the case where the network is a
    # feedforward network, organised in layers. It's most certainly wrong in
    # all the other situations.

    # What we will return:
    # -> The layers of a network in pytorch, corresponding to the network
    #    described in the .rlv
    # -> An input domain on which the property should be proved
    # -> A set of layers to stack on top of the network so as to transform
    #    the proof problem into a minimization problem.
    readline = lambda: rlv_infile.readline().strip().split(' ')

    all_layers = []
    layer_type = []
    nb_neuron_in_layer = Counter()
    neuron_depth = {}
    neuron_idx_in_layer = {}
    weight_from_neuron = defaultdict(dict)
    pool_parents = {}
    bias_on_neuron = {}
    network_depth = 0
    input_domain = []
    to_prove = []

    while True:
        line = readline()
        if line[0] == '':
            break
        if line[0] == "Input":
            n_name = line[1]
            n_depth = 0

            neuron_depth[n_name] = n_depth
            if n_depth >= len(all_layers):
                all_layers.append([])
                layer_type.append("Input")
            all_layers[n_depth].append(n_name)
            neuron_idx_in_layer[n_name] = nb_neuron_in_layer[n_depth]
            nb_neuron_in_layer[n_depth] += 1
            input_domain.append((-float('inf'), float('inf')))
        elif line[0] in ["Linear", "ReLU"]:
            n_name = line[1]
            n_bias = line[2]
            parents = [(line[i], line[i+1]) for i in range(3, len(line), 2)]

            deduced_depth = [neuron_depth[parent_name] + 1
                             for (_, parent_name) in parents]
            # Check that all the deduced depth are the same. This wouldn't be
            # the case for a ResNet type network but let's say we don't support
            # it for now :)
            for d in deduced_depth:
                assert d == deduced_depth[0], "Non Supported architecture"
            # If we are here, the deduced depth is probably correct
            n_depth = deduced_depth[0]

            neuron_depth[n_name] = n_depth
            if n_depth >= len(all_layers):
                # This is the first Neuron that we see of this layer
                all_layers.append([])
                layer_type.append(line[0])
                network_depth = n_depth
            else:
                # This is not the first neuron of this layer, let's make sure
                # the layer type is consistent
                assert line[0] == layer_type[n_depth]
            all_layers[n_depth].append(n_name)
            neuron_idx_in_layer[n_name] = nb_neuron_in_layer[n_depth]
            nb_neuron_in_layer[n_depth] += 1
            for weight_from_parent, parent_name in parents:
                weight_from_neuron[parent_name][n_name] = float(weight_from_parent)
            bias_on_neuron[n_name] = float(n_bias)
        elif line[0] == "Assert":
            # Ignore for now that there is some assert,
            # I'll figure out later how to deal with them
            ineq_symb = line[1]
            assert ineq_symb in COMPS
            off = float(line[2])
            parents = [(float(line[i]), line[i+1])
                       for i in range(3, len(line), 2)]

            if len(parents) == 1:
                # This is a constraint on a single variable, probably a simple bound.
                p_name = parents[0][1]
                depth = neuron_depth[p_name]
                pos_in_layer = neuron_idx_in_layer[p_name]
                weight = parents[0][0]
                # Normalise things a bit
                if weight < 0:
                    off = -off
                    weight = -weight
                    ineq_symb = LE if ineq_symb == GE else GE
                if weight != 1:
                    off = off / weight
                    weight = 1

                if depth == 0:
                    # This is a limiting bound on the input, let's update the
                    # domain
                    known_bounds = input_domain[pos_in_layer]
                    if ineq_symb == GE:
                        # The offset needs to be greater or equal than the
                        # value, this is an upper bound
                        new_bounds = (known_bounds[0], min(off, known_bounds[1]))
                    else:
                        # The offset needs to be less or equal than the value
                        # so this is a lower bound
                        new_bounds = (max(off, known_bounds[0]), known_bounds[1])
                    input_domain[pos_in_layer] = new_bounds
                elif depth == network_depth:
                    # If this is not on the input layer, this should be on the
                    # output layer. Imposing constraints on inner-hidden units
                    # is not supported for now.
                    to_prove.append(([(1.0, pos_in_layer)], off, ineq_symb))
                else:
                    raise Exception(f"Can't handle this line: {line}")
            else:
                parents_depth = [neuron_depth[parent_name] for _, parent_name in parents]
                assert all(network_depth == pdepth for pdepth in parents_depth), \
                "Only linear constraints on the output have been implemented."

                art_weights = [(weight, neuron_idx_in_layer[parent_name])
                               for (weight, parent_name) in parents]
                to_prove.append((art_weights, off, ineq_symb))
        elif line[0] == "MaxPool":
            n_name = line[1]
            parents = line[2:]
            deduced_depth = [neuron_depth[parent_name] + 1
                             for parent_name in parents]
            # Check that all the deduced depth are the same. This wouldn't be
            # the case for a ResNet type network but let's say we don't support
            # it for now :)
            for d in deduced_depth:
                assert d == deduced_depth[0], "Non Supported architecture"
            # If we are here, the deduced depth is probably correct
            n_depth = deduced_depth[0]
            if n_depth >= len(all_layers):
                # This is the first Neuron that we see of this layer
                all_layers.append([])
                layer_type.append(line[0])
            else:
                # This is not the first neuron of this layer, let's make sure
                # the layer type is consistent
                assert line[0] == layer_type[n_depth]
            all_layers[n_depth].append(n_name)
            neuron_idx_in_layer[n_name] = nb_neuron_in_layer[n_depth]
            nb_neuron_in_layer[n_depth] += 1

            neuron_depth[n_name] = n_depth
            pool_parents[n_name] = parents
        else:
            print("Unknown start of line.")
            raise NotImplementedError

    # Check that we have a properly defined input domain
    for var_bounds in input_domain:
        assert not math.isinf(var_bounds[0]), "No lower bound for one of the variable"
        assert not math.isinf(var_bounds[1]), "No upper bound for one of the variable"
        assert var_bounds[1] >= var_bounds[0], "No feasible value for one variable"
    # TODO maybe: If we have a constraint that is an equality exactly, it might
    # be worth it to deal with this better than just representing it by two
    # inequality constraints. A solution might be to just modify the network so
    # that it takes one less input, and to fold the contribution into the bias.
    # Note that property 4 of Reluplex is such a property.

    # Construct the network layers
    net_layers = []
    nb_layers = len(all_layers) - 1
    for from_lay_idx in range(nb_layers):
        to_lay_idx = from_lay_idx + 1

        l_type = layer_type[to_lay_idx]
        nb_from = len(all_layers[from_lay_idx])
        nb_to = len(all_layers[to_lay_idx])

        if l_type in ["Linear", "ReLU"]:
            # If it's linear or ReLU, we're going to get a nn.Linear to
            # represent the Linear part, and eventually a nn.ReLU if necessary
            new_layer = torch.nn.Linear(nb_from, nb_to, bias=True)
            lin_weight = new_layer.weight.data
            # nb_to x nb_from
            bias = new_layer.bias.data
            # nb_to

            lin_weight.zero_()
            bias.zero_()

            for from_idx, from_name in enumerate(all_layers[from_lay_idx]):
                weight_from = weight_from_neuron[from_name]
                for to_name, weight_value in weight_from.items():
                    to_idx = neuron_idx_in_layer[to_name]
                    lin_weight[to_idx, from_idx] = weight_value
            for to_idx, to_name in enumerate(all_layers[to_lay_idx]):
                bias_value = bias_on_neuron[to_name]
                bias[to_idx] = bias_value

            net_layers.append(new_layer)
            if l_type == "ReLU":
                net_layers.append(torch.nn.ReLU())
        elif l_type == "MaxPool":
            # We need to identify what kind of MaxPooling we are
            # considering.
            # Not sure how robust this really is though :/
            pool_dims_estimated = []
            first_index = []
            nb_parents = []
            for to_idx, to_name in enumerate(all_layers[to_lay_idx]):
                parents = pool_parents[to_name]
                parents_idx = [neuron_idx_in_layer[p_name]
                               for p_name in parents]
                # Let's try to identify the pattern for the max_pooling
                off_with_prev = [parents_idx[i+1] - parents_idx[i]
                                 for i in range(len(parents_idx)-1)]
                diff_offsets = set(off_with_prev)
                # The number of differents offset should mostly correspond to
                # the number of dimensions of the pooling operation, maybe???
                pool_dims_estimated.append(len(diff_offsets))
                nb_parents.append(len(parents_idx))
                first_index.append(parents_idx[0])
            assert all(pde == pool_dims_estimated[0]
                       for pde in pool_dims_estimated), "Can't identify pooling dim"
            assert all(p_nb == nb_parents[0]
                       for p_nb in nb_parents), "Can't identify the kernel size"
            # Can we identify a constant stride?
            stride_candidates = [first_index[i+1] - first_index[i]
                                 for i in range(len(first_index)-1)]
            assert all(sc == stride_candidates[0]
                       for sc in stride_candidates), "Can't identify stride."

            pool_dim = pool_dims_estimated[0]
            stride = stride_candidates[0]
            kernel_size = nb_parents[0]
            if pool_dim == 1:
                net_layers.append(View((1, nb_neuron_in_layer[from_lay_idx])))
                net_layers.append(torch.nn.MaxPool1d(kernel_size,
                                                     stride=stride))
                net_layers.append(View((nb_neuron_in_layer[to_lay_idx],)))
            else:
                raise Exception("Not implemented yet")
        else:
            raise Exception("Not implemented")

    # The .rlv files contains the specifications that we need to satisfy for
    # obtaining a counterexample

    # We will add extra layers on top that will makes it so that finding the
    # minimum of the resulting network is equivalent to performing the proof.

    # The way we do it:
    # -> For each constraint, we transform it into a canonical representation
    #    `offset GreaterOrEqual linear_fun`
    # -> Create a new neuron with a value of `linear_fun - offset`
    # -> If this neuron is negative, this constraint is satisfied
    # -> We add a Max over all of these constraint outputs.
    #    If the output of the max is negative, that means that all of the
    #    constraints have been satisfied and therefore we have a counterexample

    # So, when we minimize this network,
    # * if we obtain a negative minimum,
    #     -> We have a counterexample
    # * if we obtain a positive minimum,
    #     -> There is no input which gives a negative value, and therefore no
    #        counterexamples

    prop_layers = []
    ## Add the linear to compute the value of each constraint
    nb_final = len(all_layers[network_depth])
    nb_constr = len(to_prove)
    constr_val_layer = torch.nn.Linear(nb_final, nb_constr, bias=True)
    constr_weight = constr_val_layer.weight.data
    # nb_to x nb_from
    constr_bias = constr_val_layer.bias.data
    # nb_to
    constr_weight.zero_()
    constr_bias.zero_()
    for constr_idx, out_constr in enumerate(to_prove):
        art_weights, off, ineq_symb = out_constr
        if ineq_symb == LE:
            # Flip all the weights and the offset, and flip the LE to a GE
            art_weights = [(-weight, idx) for weight, idx in art_weights]
            off = - off
            ineq_symb = GE
        constr_bias[constr_idx] = -off
        for w, parent_idx in art_weights:
            constr_weight[constr_idx, parent_idx] = w
    prop_layers.append(constr_val_layer)

    ## Add a Maxpooling layer
    # We take a max over all the element
    nb_elt = nb_constr
    kernel_size = nb_constr

    prop_layers.append(View((1, nb_elt)))
    prop_layers.append(torch.nn.MaxPool1d(kernel_size))
    prop_layers.append(View((1,)))

    # Make input_domain into a Tensor
    input_domain = torch.Tensor(input_domain)

    return net_layers, input_domain, prop_layers


def simplify_network(all_layers):
    '''
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    '''
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            new_all_layers.append(joint_layer)
        elif (type(layer) is nn.MaxPool1d) and (layer.kernel_size == 1) and (layer.stride == 1):
            # This is just a spurious Maxpooling because the kernel_size is 1
            # We will do nothing
            pass
        elif (type(layer) is View) and (type(new_all_layers[-1]) is View):
            # No point in viewing twice in a row
            del new_all_layers[-1]

            # Figure out what was the last thing that imposed a shape
            # and if this shape was the proper one.
            prev_layer_idx = -1
            lay_nb_dim_inp = 0
            while True:
                parent_lay = new_all_layers[prev_layer_idx]
                prev_layer_idx -= 1
                if type(parent_lay) is nn.ReLU:
                    # Can't say anything, ReLU is flexible in dimension
                    continue
                elif type(parent_lay) is nn.Linear:
                    lay_nb_dim_inp = 1
                    break
                elif type(parent_lay) is nn.MaxPool1d:
                    lay_nb_dim_inp = 2
                    break
                else:
                    raise NotImplementedError
            if len(layer.out_shape) != lay_nb_dim_inp:
                # If the View is actually necessary, add the change
                new_all_layers.append(layer)
                # Otherwise do nothing
        else:
            new_all_layers.append(layer)
    return new_all_layers


def load_and_simplify(rlv_file, net_cls):
    '''
    Take as argument a .rlv file `rlv_file`,
    loads the corresponding network and its property,
    simplify it and instantiate it as an object with the `net_cls` class

    Returns the `net_cls` object and the domain of the proof
    '''
    net_layers, domain, prop_layers = load_rlv(rlv_file)
    all_layers = net_layers + prop_layers
    all_layers = simplify_network(all_layers)
    network = net_cls(all_layers)

    return network, domain


def load_mat_network(mat_file):
    '''
    Take as argument the path to a matlab file,
    loads the network and return its layers.
    '''
    weights = scipy.io.loadmat(mat_file)
    all_weight_keys = sorted(key for key in weights.keys() if 'weight' in key)
    all_bias_keys = sorted(key for key in weights.keys() if 'bias' in key)

    all_layers = []
    for w_key, b_key in zip(all_weight_keys, all_bias_keys):
        linear_weight = weights[w_key]
        linear_bias = weights[b_key]

        feat_from, feat_to = linear_weight.shape
        new_linear = nn.Linear(feat_from, feat_to, bias=True)

        new_linear.weight.data.copy_(torch.FloatTensor(linear_weight.T))
        new_linear.bias.data.copy_(torch.FloatTensor(linear_bias))

        all_layers.append(new_linear)
        all_layers.append(nn.ReLU())

    # Remove the extra ReLU at the end
    del all_layers[-1]

    return all_layers


def reluify_maxpool(layers, domain):
    '''
    Remove all the Maxpool units of a feedforward network represented by
    `layers` and replace them by an equivalent combination of ReLU + Linear

    This is only valid over the domain `domain` because we use some knowledge
    about upper and lower bounds of certain neurons
    '''
    naive_net = NaiveNetwork(layers)
    naive_net.do_interval_analysis(domain)
    lbs = naive_net.lower_bounds
    layers = layers[:]

    new_all_layers = []

    idx_of_inp_lbs = 0
    layer_idx = 0
    while layer_idx < len(layers):
        layer = layers[layer_idx]
        if type(layer) is nn.MaxPool1d:
            # We need to decompose this MaxPool until it only has a size of 2
            assert layer.padding == 0
            assert layer.dilation == 1
            if layer.kernel_size > 2:
                assert layer.kernel_size % 2 == 0, "Not supported yet"
                assert layer.stride % 2 == 0, "Not supported yet"
                # We're going to decompose this maxpooling into two maxpooling
                # max(     in_1, in_2 ,      in_3, in_4)
                # will become
                # max( max(in_1, in_2),  max(in_3, in_4))
                first_mp = nn.MaxPool1d(2, stride=2)
                second_mp = nn.MaxPool1d(layer.kernel_size // 2,
                                         stride=layer.stride // 2)
                # We will replace the Maxpooling that was originally there with
                # those two layers
                # We need to add a corresponding layer of lower bounds
                first_lbs = lbs[idx_of_inp_lbs]
                intermediate_lbs = []
                for pair_idx in range(len(first_lbs) // 2):
                    intermediate_lbs.append(max(first_lbs[2*pair_idx],
                                                first_lbs[2*pair_idx+1]))
                # Do the replacement
                del layers[layer_idx]
                layers.insert(layer_idx, first_mp)
                layers.insert(layer_idx+1, second_mp)
                lbs.insert(idx_of_inp_lbs+1, intermediate_lbs)

                # Now continue so that we re-go through the loop with the now
                # simplified maxpool
                continue
            elif layer.kernel_size == 2:
                # Each pair need two in the intermediate layers that is going
                # to be Relu-ified
                pre_nb_inp_lin = len(lbs[idx_of_inp_lbs])
                # How many starting position can we fit in?
                # 1 + how many stride we can fit before we're too late in the array to fit a kernel_size
                pre_nb_out_lin = (1 + ((pre_nb_inp_lin - layer.kernel_size) // layer.stride)) * 2
                pre_relu_lin = nn.Linear(pre_nb_inp_lin, pre_nb_out_lin, bias=True)
                pre_relu_weight = pre_relu_lin.weight.data
                pre_relu_bias = pre_relu_lin.bias.data
                pre_relu_weight.zero_()
                pre_relu_bias.zero_()
                # For each of (x, y) that needs to be transformed to max(x, y)
                # We create (x-y, y-y_lb)
                first_in_index = 0
                first_out_index = 0
                while first_in_index + 1 < pre_nb_inp_lin:
                    pre_relu_weight[first_out_index, first_in_index] = 1
                    pre_relu_weight[first_out_index, first_in_index+1] = -1

                    pre_relu_weight[first_out_index+1, first_in_index+1] = 1
                    pre_relu_bias[first_out_index+1] = -lbs[idx_of_inp_lbs][first_in_index + 1]

                    # Now shift
                    first_in_index += layer.stride
                    first_out_index += 2
                new_all_layers.append(pre_relu_lin)
                new_all_layers.append(nn.ReLU())

                # We now need to create the second layer
                # It will sum [max(x-y, 0)], [max(y - y_lb, 0)] and y_lb
                post_nb_inp_lin = pre_nb_out_lin
                post_nb_out_lin = post_nb_inp_lin // 2
                post_relu_lin = nn.Linear(post_nb_inp_lin, post_nb_out_lin)
                post_relu_weight = post_relu_lin.weight.data
                post_relu_bias = post_relu_lin.bias.data
                post_relu_weight.zero_()
                post_relu_bias.zero_()
                first_in_index = 0
                out_index = 0
                while first_in_index + 1 < post_nb_inp_lin:
                    post_relu_weight[out_index, first_in_index] = 1
                    post_relu_weight[out_index, first_in_index+1] = 1
                    post_relu_bias[out_index] = lbs[idx_of_inp_lbs][layer.stride*out_index+1]
                    first_in_index += 2
                    out_index += 1
                new_all_layers.append(post_relu_lin)
                idx_of_inp_lbs += 1
            else:
                # This should have been cleaned up in one of the simplify passes
                raise NotImplementedError
        elif type(layer) is nn.Linear:
            new_all_layers.append(layer)
            idx_of_inp_lbs += 1
        elif type(layer) is nn.ReLU:
            new_all_layers.append(layer)
        elif type(layer) is View:
            # We shouldn't add the view as we are getting rid of them
            pass
        layer_idx += 1
    return new_all_layers


def dump_rlv(rlv_outfile, layers, domain, transform_maxpool=False):
    '''
    Dump the networks represented by the series of `layers`
    into the `rlv_outfile` file.
    If `transform_maxpool` is set to True, replace the Maxpool layer
    by a combination of ReLUs
    '''
    writeline = lambda x: rlv_outfile.write(x + '\n')

    if transform_maxpool:
        new_layers = simplify_network(layers)
        new_layers = reluify_maxpool(new_layers, domain)
        new_layers = simplify_network(new_layers)

        max_net = nn.Sequential(*layers)
        relu_net = nn.Sequential(*new_layers)

        assert_network_equivalence(max_net, relu_net, domain)
        layers = new_layers

    var_names = []

    # Define the input
    inp_layer_var_names = []
    for inp_idx, (inp_lb, inp_ub) in enumerate(domain):
        var_name = f"inX{inp_idx}"
        writeline(f"Input {var_name}")
        writeline(f"Assert <= {inp_lb} 1.0 {var_name}")
        writeline(f"Assert >= {inp_ub} 1.0 {var_name}")
        inp_layer_var_names.append(var_name)
    var_names.append(inp_layer_var_names)

    layer_idx = 0
    out_layer_idx = 1
    while layer_idx < len(layers):
        layer = layers[layer_idx]
        new_layer_var_names = []
        if type(layer) is nn.Linear:
            # Should we write it as a Linear or as a ReLU?
            is_relu = False
            # If the next layer is a ReLU, write it as ReLU
            # Otherwise, as Linear
            if (layer_idx + 1 < len(layers)) and (type(layers[layer_idx+1]) is nn.ReLU):
                is_relu = True
            line_header = "ReLU" if is_relu else "Linear"
            var_pattern = "relu" if is_relu else "linear"

            prev_var_names = var_names[-1]

            for out_n_idx in range(layer.out_features):
                var_name = f"{var_pattern}_{out_layer_idx}-{out_n_idx}"
                bias = layer.bias.data[out_n_idx]
                weight_str = " ".join([f"{w} {pre_var}" for w, pre_var
                                       in zip(layer.weight.data[out_n_idx, :],
                                              prev_var_names)])
                writeline(f"{line_header} {var_name} {bias} {weight_str}")
                new_layer_var_names.append(var_name)

            out_layer_idx += 1
            var_names.append(new_layer_var_names)
        elif type(layer) is nn.ReLU:
            assert layer_idx > 0, "A ReLU is the first layer, that's weird"
            assert type(layers[layer_idx-1]) is nn.Linear, "There was no linear before this ReLU, this script might be wrong in this case"
        elif type(layer) is View:
            pass
        elif type(layer) is nn.MaxPool1d:
            assert not transform_maxpool
        else:
            raise NotImplementedError
        layer_idx += 1

    # Given that we have standardized the property to amount to
    # Prove that the output is less than zero,
    writeline(f"Assert >= 0.0 1.0 {var_names[-1][0]}")


def dump_nnet(nnet_outfile, layers, domain):
    '''
    Dump the networks represented by the series of `layers`
    into the `nnet_outfile` file.
    This is a valid dump only on the domain `domain`, because
    we use some knowledge about bounds on the value of some neurons
    to guarantee that we are passing the ReLU.
    '''
    writeline = lambda x: nnet_outfile.write(x + '\n')
    make_comma_separated_line = lambda tab: ",".join(map(str, tab))+","

    new_layers = simplify_network(layers)
    new_layers = reluify_maxpool(new_layers, domain)
    new_layers = simplify_network(new_layers)

    max_net = nn.Sequential(*layers)
    relu_net = nn.Sequential(*new_layers)

    assert_network_equivalence(max_net, relu_net, domain)
    layers = new_layers

    var_names = []

    # Global parameters of the networks
    nb_layers = 0
    max_lay_size = 0
    for layer in layers:
        if type(layer) is nn.Linear:
            nb_layers += 1
            max_lay_size = max(max_lay_size, layer.out_features)
    nb_input = layers[0].in_features
    output_size = 1
    writeline(f"{nb_layers},{nb_input},{output_size},{max_lay_size},")

    # Layer sizes
    layer_sizes = [nb_input]
    for layer in layers:
        if type(layer) is nn.Linear:
            layer_sizes.append(layer.out_features)

    layer_size_str = ",".join(map(str, layer_sizes))
    writeline(make_comma_separated_line(layer_sizes))

    # Symmetric parameter
    writeline("0")

    # Write down the mins of the input of the network
    inp_lbs = domain[:, 0]
    writeline(make_comma_separated_line(inp_lbs))

    # Write down the maxes of the input of the network
    inp_ubs = domain[:, 1]
    writeline(make_comma_separated_line(inp_ubs))

    # Write down the mean of the input of the network.
    # We're not going to do any conditioning
    # Note that there is one additional that is for the output
    writeline(make_comma_separated_line([0]*(nb_input+1)))

    # Write down the ranges of the input of the network
    # We're not going to do any conditioning
    # Note that there is one additional that is for the output
    writeline(make_comma_separated_line([1]*(nb_input+1)))

    for layer in layers:
        if type(layer) is not nn.Linear:
            # The ReLU is implicit, and we have removed all the linear layers
            continue
        # Write the weight coming to each neuron
        for neuron_out_idx in range(layer.out_features):
            to_neuron_weight = layer.weight.data[neuron_out_idx, :]
            writeline(make_comma_separated_line(to_neuron_weight))
        # Write the bias for each neuron
        for neuron_out_idx in range(layer.out_features):
            neuron_bias = layer.bias.data[neuron_out_idx]
            writeline(f"{neuron_bias},")


def assert_network_equivalence(net1, net2, domain):
    nb_samples = 1024 * 1024
    nb_inp = domain.size(0)
    rand_samples = torch.Tensor(nb_samples, nb_inp)
    rand_samples.uniform_(0, 1)

    domain_lb = domain.select(1, 0).contiguous()
    domain_ub = domain.select(1, 1).contiguous()
    domain_width = domain_ub - domain_lb

    domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
    domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

    inps = domain_lb + domain_width * rand_samples

    with torch.no_grad():
        net1_out = net1(inps)
        net2_out = net2(inps)
        diff = net1_out - net2_out
        max_diff = torch.abs(diff).max()

    assert max_diff <= 1e-8, "The network rewrite is incorrect"
