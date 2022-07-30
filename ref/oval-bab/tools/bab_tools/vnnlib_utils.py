import onnx
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import re, os
from tools.custom_torch_modules import supported_transforms, Flatten
from plnn.model import simplify_network, remove_maxpools
from plnn.proxlp_solver.propagation import Propagation
from tools.bab_tools.model_utils import reluified_max_pool
from tools.bab_tools.onnx_reader import OnnxConverter, torch_type_tab, ResNetError
import onnxruntime as ort


def predict_with_onnxruntime(model_def, *inputs):

    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = {name: inp for name, inp in zip(names, inputs)}

    res = sess.run(None, inp)
    names = [o.name for o in sess.get_outputs()]

    return {name: output for name, output in zip(names, res)}


def create_identity_layer(n, dtype=torch.float32):
    id_layer = nn.Linear(n, n, bias=True)
    id_layer.weight = nn.Parameter(torch.eye(n, dtype=dtype), requires_grad=False)
    id_layer.bias = nn.Parameter(torch.zeros((n,), dtype=dtype), requires_grad=False)
    return id_layer


def onnx_to_pytorch(onnx_path, numerical_tolerance=1e-3):
    if onnx_path.endswith(".gz"):
        os.system(f"gzip -dfk {onnx_path}")
        onnx_path = onnx_path.replace(".gz", "")
    onnx_model = onnx.load(onnx_path)
    try:
        torch_layers = [param for param in OnnxConverter(onnx_model).children()]
    except ResNetError as resneterror:
        print(resneterror)
        return None, None, None, None, False
    except KeyError:
        import onnxsim
        print("Operations on constants are not supported by the onnx converter, "
              "simplifying the model using the external onnxsim library (please install it if missing).")
        model_simp, check = onnxsim.simplify(onnx_model)
        assert check, "the onnx simplification library failed"
        torch_layers = [param for param in OnnxConverter(model_simp).children()]

    onnx_input_shape = [cdim.dim_value if cdim.dim_value > 0 else 1 for cdim in
                        onnx_model.graph.input[0].type.tensor_type.shape.dim]
    onnx_output_shape = [cdim.dim_value if cdim.dim_value > 0 else 1 for cdim in
                         onnx_model.graph.output[0].type.tensor_type.shape.dim]
    dtype = torch_type_tab[onnx_model.graph.input[0].type.tensor_type.elem_type]

    # Simplify the torch model
    to_remove = []
    for idx, clayer in enumerate(torch_layers):
        # Remove constant paddings that have no effect.
        if isinstance(clayer, (nn.ConstantPad1d, nn.ConstantPad2d)):
            if sum(clayer.padding) == 0:
                to_remove.append(idx)
    for idx in to_remove:
        torch_layers.pop(idx)

    if not (isinstance(torch_layers[-1], nn.Linear) or isinstance(torch_layers[-1], nn.Conv2d)):
        # OVAL-BaB expects the network to end with a linear layer: adding it.
        linear_layers = list(filter(lambda x: isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d), torch_layers))
        if isinstance(linear_layers[-1], nn.Conv2d):
            torch_layers.append(Flatten())
        torch_layers.append(create_identity_layer(np.prod(onnx_output_shape[1:])))
    torch_model = nn.Sequential(*torch_layers)

    # Execute a random image on both the onnx and the converted pytorch model and check whether they're reasonably close
    image = torch.rand(onnx_input_shape, dtype=dtype)
    onnx_in = image.numpy()
    onnx_out = predict_with_onnxruntime(onnx_model, onnx_in)
    onnx_out = torch.tensor(onnx_out[list(onnx_out.keys())[0]])
    torch_out = torch_model(image)
    correctness = ((onnx_out - torch_out).abs().max() < numerical_tolerance)

    return torch_model, onnx_input_shape[1:] if onnx_input_shape[1:] else onnx_input_shape, onnx_output_shape[1:], \
           dtype, correctness


def add_canonical_layers_from_outspecs(domain, out_specs, model, max_solver_batch=5e3, gpu=True, dtype=torch.float32):
    """
    Before doing anything, one needs to negate the property, from counter-example to the propery itself.
    Out_specs is in form ((a <= b) AND (c <= d)) OR (...).
    Let's negate it: NOT (((a <= b) AND (c <= d)) OR (...)) = (NOT ( (a >= b) AND (c >= d) )) AND ( NOT ...)
        = ( (a > b) OR (c > d) ) AND ( NOT ...)
    """
    model_layers = list(model)

    # Add identity linear layer at the end if the last layer is an activation.
    if type(model[-1]) not in [nn.Linear, nn.Conv2d]:
        model_layers += [create_identity_layer(model[-2].out_features, dtype=dtype)]

    def get_lbs_from_layers(clayers):
        # use tighter bounds than IBP to reduce numerical imprecisions
        lb_layers = [deepcopy(lay).cuda() for lay in clayers] if gpu else clayers
        with torch.no_grad():
            lb_net = Propagation(lb_layers, max_batch=max_solver_batch, type="crown")
            lb_domain = domain.cuda().unsqueeze(0) if gpu else domain.unsqueeze(0)
            lb_net.define_linear_approximation(lb_domain)
            out = lb_net.lower_bounds[-1].squeeze(0).cpu() if gpu else lb_net.lower_bounds[-1].squeeze(0)
            lb_net.unbuild()
        return out

    def linear_layer_skeleton(n_in, n_out):
        return nn.Linear(n_in, n_out, bias=True), torch.zeros((n_out, n_in), dtype=dtype), \
               torch.zeros((n_out,), dtype=dtype)

    nested_or = (sum([cspec["out_coeffs"].shape[0] for cspec in out_specs]) > len(out_specs))
    if not nested_or:
        # Add layer before the final AND.
        n_in, and_size = model_layers[-1].out_features, len(out_specs)
        premin_layer, premin_weights, premin_bias = linear_layer_skeleton(n_in, and_size)
        # The counter example specification is in the form out_specs[i]["out_coeffs"]^T last_layer <= out_specs[i]["rhs"]
        # However, to get the canonical form, we are negating the counter-example SAT specification and build a network
        # capturing (out_specs[i]["out_coeffs"]^T last_layer - out_specs[i]["rhs"]) -- which must be > 0
        for idx, cspec in enumerate(out_specs):
            premin_weights[idx, :] = torch.tensor(cspec["out_coeffs"], dtype=dtype)
            premin_bias[idx] = -torch.tensor(cspec["rhs"], dtype=dtype)
        premin_layer.weight = nn.Parameter(premin_weights, requires_grad=False)
        premin_layer.bias = nn.Parameter(premin_bias, requires_grad=False)
        model_layers.append(premin_layer)
        model_layers = simplify_network(model_layers)
    else:
        # Before the final AND, represent the nested OR.
        and_size = len(out_specs)
        and_layers = {}  # dict storing for each and input, the list of layers representing it
        max_or_depth = 0  # max depth of OR subnetwork
        for idx, cspec in enumerate(out_specs):
            # Add layer before the current OR.
            or_size = cspec["out_coeffs"].shape[0]
            n_in, n_out = model_layers[-1].out_features, or_size
            premax_layer, premax_weights, premax_bias = linear_layer_skeleton(n_in, n_out)
            for or_idx in range(or_size):
                premax_weights[or_idx, :] = torch.tensor(cspec["out_coeffs"][or_idx], dtype=dtype)
                premax_bias[or_idx] = -torch.tensor(cspec["rhs"][or_idx], dtype=dtype)
            premax_layer.weight = nn.Parameter(premax_weights, requires_grad=False)
            premax_layer.bias = nn.Parameter(premax_bias, requires_grad=False)
            # Add reluified maxpool representing the current OR.
            lbs = get_lbs_from_layers(simplify_network(model_layers + [premax_layer]))
            max_pool_layers = reluified_max_pool(n_out, lbs, dtype=dtype)
            and_layers[idx] = simplify_network([premax_layer] + max_pool_layers)
            max_or_depth = max(max_or_depth, len(and_layers[idx]))

        # First make all the OR subnetworks of the same length by adding dummy identity layers (+ReLU, which will
        # always be passing as it is applied to a ReLU output) after the first ReLU
        # TODO: add skip connections support.
        for idx in range(and_size):
            while len(and_layers[idx]) < max_or_depth:
                n_id = and_layers[idx][0].out_features
                identity_layer = create_identity_layer(n_id, dtype=dtype)
                and_layers[idx] = and_layers[idx][:2] + [identity_layer, nn.ReLU()] + and_layers[idx][2:]

        # Aggregate the subnetworks.
        stacked_layers = []
        previous_cumul_sizes = None
        for layer_n in range(max_or_depth):
            if isinstance(and_layers[0][layer_n], nn.Linear):
                # as sizes of the subnetworks might be irregular, compute the cumulative indices
                cumul_sizes = [0, and_layers[0][layer_n].out_features]
                for idx in range(1, and_size):
                    cumul_sizes.append(cumul_sizes[idx] + and_layers[idx][layer_n].out_features)

                n_out = cumul_sizes[-1]
                n_in = and_layers[0][layer_n].in_features if layer_n == 0 else stacked_layers[layer_n-2].out_features
                stacked_layer, stacked_weights, stacked_bias = linear_layer_skeleton(n_in, n_out)
                for idx in range(and_size):
                    vrange = (cumul_sizes[idx], cumul_sizes[idx+1])
                    stacked_bias[vrange[0]:vrange[1]] = and_layers[idx][layer_n].bias
                    # only the first layer is stacked vertically, the others are in a block-matrix form
                    if previous_cumul_sizes is None:
                        stacked_weights[vrange[0]:vrange[1]] = and_layers[idx][layer_n].weight
                    else:
                        hrange = (previous_cumul_sizes[idx], previous_cumul_sizes[idx+1])
                        stacked_weights[vrange[0]:vrange[1], hrange[0]:hrange[1]] = and_layers[idx][layer_n].weight

                stacked_layer.weight = nn.Parameter(stacked_weights, requires_grad=False)
                stacked_layer.bias = nn.Parameter(stacked_bias, requires_grad=False)
                stacked_layers.append(stacked_layer)
                previous_cumul_sizes = cumul_sizes
            else:
                # Append ReLU
                stacked_layers.append(and_layers[0][layer_n])

        model_layers = simplify_network(model_layers + stacked_layers)

    if and_size > 1:
        # Add AND over the last layer by negating the layer, then adding a negated ReLUified maxpool (equivalent to min)
        # Negate last layer.
        model_layers[-1].weight = nn.Parameter(-model_layers[-1].weight, requires_grad=False)
        model_layers[-1].bias = nn.Parameter(-model_layers[-1].bias, requires_grad=False)
        # Compute LBs for ReLUified maxpool.
        lbs = get_lbs_from_layers(model_layers)
        # Add ReLUified maxpool.
        max_pool_layers = reluified_max_pool(and_size, lbs, flip_out_sign=True, dtype=dtype)
        simplified_layers = simplify_network(model_layers[-1:] + max_pool_layers)
        model_layers = model_layers[:-1] + simplified_layers

    return model_layers


def get_ground_truth(out_specs):
    # Retrieve ground truth class for adversarial robustness specifications.
    ground_truth = np.where(out_specs[0]["out_coeffs"][0] == 1)[0].item()
    for cspec in out_specs:
        assert cspec["out_coeffs"].shape[0] == 1, "k-ary ANDs must be absent in adversarial robustness representation"
        assert ground_truth == np.where(cspec["out_coeffs"][0] == 1)[0].item()
    return ground_truth


def is_supported_model(model):
    supported = True
    accepted_layers = [nn.Flatten, nn.Linear, nn.Conv2d, nn.ReLU, nn.AvgPool2d, nn.ConstantPad2d,
                       *supported_transforms, nn.MaxPool1d, nn.MaxPool2d]
    for clayer in model:
        if type(clayer) not in accepted_layers:
            supported = False
    return supported


def load_vnnlib_property(onnx_filename, vnnlib_filename, is_adversarial=False, gpu=True):
    """
    if is_adversarial is True, return also the model not in canonical form, the input lower bounds, and the
    classification ground truth.
    :return: a list of (input_bounds, verif_layers, UB info for adv. robustness). Each entry corresponds to an input
        disjunction
    """
    model, in_shape, out_shape, dtype, model_correctness = onnx_to_pytorch(onnx_filename)

    if not model_correctness:
        return None, False

    # Assert that the model specification is currently supported.
    supported = is_supported_model(model)

    if not supported:
        return None, False

    num_inputs = int(np.array(in_shape).prod())
    num_outputs = int(np.array(out_shape).prod())
    vnnlib_parser = VNNLibParser(vnnlib_filename, num_inputs, num_outputs)
    parsed_spec = vnnlib_parser.parse()

    # Iterate over input disjunctions
    simplification_correctness = True
    in_disjunctions = []
    for in_disjunction in parsed_spec:
        input_bounds = torch.tensor(in_disjunction["input_box"], dtype=dtype).view(in_shape + [2])
        out_specs = in_disjunction["output_constraints"]

        # ReLUify any maxpool...
        with torch.no_grad():
            simplified_model = remove_maxpools(deepcopy(list(model.children())), input_bounds, dtype=dtype)
        # ... and check that the simplification is formally correct
        simplification_correctness = ((torch.nn.Sequential(*simplified_model)(input_bounds.select(-1, 0).unsqueeze(0))
                                       - model((input_bounds.select(-1, 0).unsqueeze(0)))).abs().max() < 1e-5)
        if not simplification_correctness:
            return None, False

        verif_layers = add_canonical_layers_from_outspecs(input_bounds, out_specs, simplified_model, dtype=dtype,
                                                          gpu=gpu)
        in_disjunctions.append((input_bounds, verif_layers))

    return in_disjunctions, supported and model_correctness and simplification_correctness


class VNNLibParser:
    """
    Adaptation of vnnlib parser by Stanley Bak, see:
    https://github.com/stanleybak/simple_adversarial_generator/blob/main/src/agen/vnnlib.py
    """
    def __init__(self, vnnlib_filename, num_inputs, num_outputs):
        self.vnnlib_filename = vnnlib_filename
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def read_statements(self):
        """
        Process vnnlib and return a list of strings (statements)
        useful to get rid of comments and blank lines and combine multi-line statements
        """

        with open(self.vnnlib_filename, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]
        assert len(lines) > 0

        # combine lines if case a single command spans multiple lines
        open_parentheses = 0
        statements = []
        current_statement = ''

        for line in lines:
            comment_index = line.find(';')

            if comment_index != -1:
                line = line[:comment_index].rstrip()

            if not line:
                continue

            new_open = line.count('(')
            new_close = line.count(')')

            open_parentheses += new_open - new_close

            assert open_parentheses >= 0, "mismatched parenthesis in vnnlib file"

            # add space
            current_statement += ' ' if current_statement else ''
            current_statement += line

            if open_parentheses == 0:
                statements.append(current_statement)
                current_statement = ''

        if current_statement:
            statements.append(current_statement)

        # remove repeated whitespace characters
        statements = [" ".join(s.split()) for s in statements]

        # remove space after '('
        statements = [s.replace('( ', '(') for s in statements]

        # remove space after ')'
        statements = [s.replace(') ', ')') for s in statements]

        return statements

    def update_rv_tuple(self, rv_tuple, op, first, second):
        """
        Update tuple from rv in read_vnnlib_simple, with the passed in constraint (op first second)
        """

        if first.startswith("X_"):
            # Input constraints
            index = int(first[2:])

            assert not second.startswith("X") and not second.startswith("Y"), \
                f"input constraints must be box ({op} {first} {second})"
            assert 0 <= index < self.num_inputs

            limits = rv_tuple[0][index]

            if op == "<=":
                limits[1] = min(float(second), limits[1])
            else:
                limits[0] = max(float(second), limits[0])

            assert limits[0] <= limits[1], f"{first} range is empty: {limits}"

        else:
            # output constraint
            if op == ">=":
                # swap order if op is >=
                first, second = second, first

            row = [0.0] * self.num_outputs
            rhs = 0.0

            # assume op is <=
            if first.startswith("Y_") and second.startswith("Y_"):
                index1 = int(first[2:])
                index2 = int(second[2:])

                row[index1] = 1
                row[index2] = -1
            elif first.startswith("Y_"):
                index1 = int(first[2:])
                row[index1] = 1
                rhs = float(second)
            else:
                assert second.startswith("Y_")
                index2 = int(second[2:])
                row[index2] = -1
                rhs = -1 * float(first)

            mat, rhs_list = rv_tuple[1], rv_tuple[2]
            mat.append(row)
            rhs_list.append(rhs)

    def parse(self):
        """
        Process in a vnnlib file: not a general parser, and assumes files are provided in a 'nice' format.
        Only a single disjunction is allowed.

        output a list (a list entry for each input disjunction)containing a dict:
            "input_box": input ranges (box), list of pairs for each input variable
            "output_constraints": specification, provided as a list of dicts
                {"out_coeffs": mat, "rhs": rhs}, as in: mat * y <= rhs,
                where y is the output. Each element in the list is a term in a disjunction for the specification.
                Conjunctions within each disjunction are represented as batched tensors.
        """

        # example: "(declare-const X_0 Real)"
        regex_declare = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

        # comparison sub-expression
        # example: "(<= Y_0 Y_1)" or "(<= Y_0 10.5)"
        comparison_str = r"\((<=|>=) (\S+) (\S+)\)"

        # example: "(and (<= Y_0 Y_2)(<= Y_1 Y_2))"
        dnf_clause_str = r"\(and (" + comparison_str + r")+\)"

        # example: "(assert (<= Y_0 Y_1))"
        regex_simple_assert = re.compile(r"^\(assert " + comparison_str + r"\)$")

        # disjunctive-normal-form
        # (assert (or (and (<= Y_3 Y_0)(<= Y_3 Y_1)(<= Y_3 Y_2))(and (<= Y_4 Y_0)(<= Y_4 Y_1)(<= Y_4 Y_2))))
        regex_dnf = re.compile(r"^\(assert \(or (" + dnf_clause_str + r")+\)\)$")

        rv = []  # list of 3-tuples, (box-dict, mat, rhs)
        rv.append(({i: [-np.inf, np.inf] for i in range(self.num_inputs)}, [], []))

        lines = self.read_statements()

        for line in lines:
            # print(f"Line: {line}")

            if len(regex_declare.findall(line)) > 0:
                continue

            groups = regex_simple_assert.findall(line)

            if groups:
                assert len(groups[0]) == 3, f"groups was {groups}: {line}"
                op, first, second = groups[0]

                for rv_tuple in rv:
                    self.update_rv_tuple(rv_tuple, op, first, second)

                continue

            ################
            groups = regex_dnf.findall(line)
            assert groups, f"failed parsing line: {line}"

            tokens = line.replace("(", " ").replace(")", " ").split()
            tokens = tokens[2:]  # skip 'assert' and 'or'

            conjuncts = " ".join(tokens).split("and")[1:]

            old_rv = rv
            rv = []

            for rv_tuple in old_rv:
                for c in conjuncts:
                    rv_tuple_copy = deepcopy(rv_tuple)
                    rv.append(rv_tuple_copy)

                    c_tokens = [s for s in c.split(" ") if len(s) > 0]

                    count = len(c_tokens) // 3

                    for i in range(count):
                        op, first, second = c_tokens[3 * i:3 * (i + 1)]

                        self.update_rv_tuple(rv_tuple_copy, op, first, second)

        # merge elements of rv with the same input spec
        merged_rv = {}

        for rv_tuple in rv:
            boxdict = rv_tuple[0]
            matrhs = (rv_tuple[1], rv_tuple[2])

            key = str(boxdict)  # merge based on string representation of input box... accurate enough for now

            if key in merged_rv:
                merged_rv[key][1].append(matrhs)
            else:
                merged_rv[key] = (boxdict, [matrhs])

        # finalize objects (convert dicts to lists and lists to np.array)
        final_rv = []

        for rv_tuple in merged_rv.values():
            box_dict = rv_tuple[0]

            box = []

            for d in range(self.num_inputs):
                r = box_dict[d]

                assert r[0] != -np.inf and r[1] != np.inf, f"input X_{d} was unbounded: {r}"
                box.append(r)

            spec_list = []

            for matrhs in rv_tuple[1]:
                mat = np.array(matrhs[0], dtype=float)
                rhs = np.array(matrhs[1], dtype=float)
                spec_list.append({"out_coeffs": mat, "rhs": rhs})

            final_rv.append({"input_box": box, "output_constraints": spec_list})

        return final_rv


if __name__ == "__main__":

    # 1vsall adversarial vulnerability example, tested.
    onnx_filename = "./models/onnx/cifar_base_kw.onnx"
    vnnlib_filename = "../vnncomp2021/benchmarks/oval21/vnnlib/cifar_base_kw-img8493-eps0.03320261437908497.vnnlib"

    # Contains input disjunction, tested.
    # onnx_filename = "../vnncomp2021/benchmarks/acasxu/ACASXU_run2a_1_1_batch_2000.onnx"
    # vnnlib_filename = "../vnncomp2021/benchmarks/acasxu/prop_6.vnnlib"

    # Contains output disjunction + conjunction, tested.
    # onnx_filename = "../vnncomp2021/benchmarks/acasxu/ACASXU_run2a_1_9_batch_2000.onnx"
    # vnnlib_filename = "../vnncomp2021/benchmarks/acasxu/prop_7.vnnlib"

    # Contains joint input-output properties, to test.
    # onnx_filename = "../vnncomp2021/benchmarks/nn4sys/nets/lognormal_1000.onnx"
    # vnnlib_filename = "../vnncomp2021/benchmarks/nn4sys/specs/lognormal_1000_72_73_75_76_77_78_80_81_82_83_85_86_87" \
    #                   "_88_91_92_93_94_96_97_98_99_100_103_104_105_106_109_110_111_112_113_117_118_119_120_121_122_" \
    #                   "123_127_128_129_130_131_132_133_134_142_143_144.vnnlib"

    # onnx_filename = "../vnncomp2021/benchmarks/verivital/Convnet_maxpool.onnx"
    # vnnlib_filename = "../vnncomp2021/benchmarks/verivital/specs/maxpool_specs/prop_3_0.004.vnnlib"

    _, supported = load_vnnlib_property(onnx_filename, vnnlib_filename, gpu=False)

    print(f"VNNLib property supported? --> {supported}")
