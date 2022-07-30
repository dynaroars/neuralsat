import time, os, argparse, csv, json, gc, pickle, traceback
import pandas as pd
import torch
from tools.bab_tools.bab_runner import bab_from_json, bab_output_from_return_dict
from tools.bab_tools.vnnlib_utils import load_vnnlib_property


def bab_from_vnnlib_dataset(dataset_csv_filename, json_filename, result_path, is_adversarial=False):
    # run a whole dataset locally loading a csv file

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(dataset_csv_filename, 'r') as csvfile:
        dataset_spec = list(csv.reader(csvfile, delimiter=','))
    base_path = os.path.dirname(dataset_csv_filename) + "/"

    # Prepare pickle for results.
    json_name = os.path.basename(json_filename.replace(".json", ""))
    record_name = result_path + os.path.basename(dataset_csv_filename.replace(".csv", "")) + "_" + json_name + ".pkl"
    if os.path.isfile(record_name):
        results_table = pd.read_pickle(record_name)
    else:
        indices = list(range(len(dataset_spec)))
        results_table = pd.DataFrame(
            index=indices, columns=["prop"] + [f'BSAT_{json_name}', f'BBran_{json_name}', f'BTime_{json_name}'])
        results_table.to_pickle(record_name)

    for idx, (onnx_filename, vnnlib_filename, timeout) in enumerate(dataset_spec):

        torch.cuda.empty_cache()
        gc.collect()  # Garbage-collect cpu memory.

        if pd.isna(results_table.loc[idx]["prop"]) == False:
            print(f'the {idx}th element is done')
            continue

        # Run BaB
        bab_out, bab_nb_states, bab_time = bab_from_vnnlib(
            base_path + onnx_filename, base_path + vnnlib_filename, int(float(timeout)), json_filename,
            is_adversarial=is_adversarial)

        results_table.loc[idx]["prop"] = os.path.basename(vnnlib_filename.replace(".vnnlib", ""))
        results_table.loc[idx][f"BSAT_{json_name}"] = bab_out
        results_table.loc[idx][f"BBran_{json_name}"] = bab_nb_states
        results_table.loc[idx][f"BTime_{json_name}"] = bab_time
        results_table.to_pickle(record_name)


def bab_from_vnnlib(onnx_filename, vnnlib_filename, timeout, json_filename, is_adversarial=False, time_results=True,
                    in_disjunctions=None, supported=None, start_time=None):

    found_counter_example = lambda x: x == "True"
    timed_out = lambda x: x == "timeout"
    is_error = lambda x: x not in ["True", "False", "timeout"]

    # The specifications for the network in canonical form can be optionally passed as arguments directly.
    if in_disjunctions is None or supported is None:
        # Convert the network and property specification into the canonical form
        in_disjunctions, supported = load_vnnlib_property(onnx_filename, vnnlib_filename, is_adversarial=is_adversarial)

    if not supported:
        return "babError", 0, -1

    if time_results:
        torch.cuda.synchronize()
        start_time = time.time()
    else:
        bab_time = -1

    # Iterate over input disjunctions running BaB separately, then merge the results.
    bab_nb_states = 0
    bab_out = -1
    has_timed_out = False
    for domain, verif_layers in in_disjunctions:

        # the network name for the json is only the net name w/o extensions
        nn_name = os.path.basename(onnx_filename.replace(".onnx", ""))
        return_dict = dict()

        with open(json_filename) as json_file:
            json_params = json.load(json_file)
        bab_from_json(json_params, verif_layers, domain, return_dict, nn_name,
                      instance_timeout=timeout / len(in_disjunctions), start_time=start_time)
        del json_params
        c_bab_out, c_bab_nb_states = bab_output_from_return_dict(return_dict)
        bab_nb_states += c_bab_nb_states

        # The only ignored case from the if-else below is "False", which, in order to be returned, must be set at the
        # first disjunction and never altered.
        if bab_out == -1:
            # Set the BaB output to the first result from input disjunctions.
            bab_out = c_bab_out
        if is_error(c_bab_out) or found_counter_example(c_bab_out):
            # Break the loop at the first error, or counter-example.
            bab_out = c_bab_out
            break
        elif timed_out(c_bab_out):
            # If it ever timed-out, we can't exclude the existence of counter-examples.
            has_timed_out = True
    if has_timed_out and bab_out == "False":
        bab_out = "timeout"

    if time_results:
        bab_time = time.time() - start_time
        print('total time required: ', bab_time)

    return bab_out, bab_nb_states, bab_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modulo', type=int, help='Numbers of jobs to split the dataset over (e.g., one per GPU).',
                        default=1)
    parser.add_argument('--modulo_do', type=int, help='Which job_id is this one.', default=0)
    parser.add_argument('--dataset', type=str, help='Which .csv experiment to run')
    parser.add_argument('--is_adversarial', action='store_true', help='retrieve info for adv. robustness specifications')
    parser.add_argument('--json', type=str, help='json file storing the BaB settings')
    parser.add_argument('--vnncomp_category', type=str, help='Name of the VNNCOMP21 category being run')
    parser.add_argument('--mode', type=str, choices=["run_dataset", "run_instance", "prepare"], default="run_dataset")
    parser.add_argument('--result_path', type=str, help='path where to store results')
    parser.add_argument('--configs_path', type=str, help='path where to look for json files for VNNCOMP21')
    parser.add_argument('--result_file', type=str, help='file where to store results')
    parser.add_argument('--onnx', type=str, help='location of onnx file for the network')
    parser.add_argument('--vnnlib', type=str, help='location of vnnlib file for the property')
    parser.add_argument('--from_pickle', action='store_true')
    parser.add_argument('--instance_timeout', type=float)
    args = parser.parse_args()

    if args.mode == "run_dataset":
        # Given a csv file with the list of onnx and vnnlib pairs defining the benchmark, and a .json configuration
        # file, run BaB on all the properties, storing the results into a pickle file in folder args.result_path.
        bab_from_vnnlib_dataset(args.dataset, args.json, args.result_path, is_adversarial=args.is_adversarial)
    elif args.mode in ["prepare", "run_instance"]:
        onnx_name = os.path.basename(args.onnx).replace(".onnx", "").replace(".gz", "")
        spec_name = os.path.basename(args.vnnlib).replace(".vnnlib", "")
        pickle_folder = os.path.dirname(args.vnnlib) + "/"
        pickle_name = pickle_folder + onnx_name + "-" + spec_name + ".pkl"

        if args.mode == "prepare":
            # Given the onnx net and the vnnlib specification, convert the property in canonical form and pickle it
            # (see https://arxiv.org/pdf/1909.06588.pdf, section 3.1)
            in_disjunctions, supported = load_vnnlib_property(args.onnx, args.vnnlib)
            with open(pickle_name, 'wb') as pfile:
                pickle.dump((in_disjunctions, supported), pfile)

        elif args.mode == "run_instance":
            # Run BaB on a single verification property defined by the onnx and vnnlib specification. Write the
            # result to a file specified via args.result_file, in the vnncomp format
            # Requires a json file containing BaB parameters, indicated by:
            #  (args.vnncomp_category + args.configs_path) OR (args.json)

            start_time = time.time()  # Pass start time to BaB execution in order to exit before the timeout.

            try:
                if args.from_pickle:
                    # Retrieve the property in canonical form from pickle
                    with open(pickle_name, 'rb') as pfile:
                        in_disjunctions, supported = pickle.load(pfile)
                    os.remove(pickle_name)
                else:
                    in_disjunctions, supported = load_vnnlib_property(args.onnx, args.vnnlib)

                json_filename = args.json if args.json else f"{args.configs_path}{args.vnncomp_category}_vnncomp21.json"
                bab_out, _, _ = bab_from_vnnlib(args.onnx, args.vnnlib, args.instance_timeout, json_filename,
                                                time_results=False, in_disjunctions=in_disjunctions,
                                                supported=supported, start_time=start_time)

                # Vnncomp out format: just write the out status on a single line.
                conversion_dict = {
                    "timeout": "timeout",
                    "True": "violated",
                    "False": "holds",
                    "babError": "unknown",  # this means the network architecture is not supported
                }
                result = conversion_dict[bab_out] if bab_out in conversion_dict else "error"
            except:
                # Catch any exception, print its stack, and return error
                traceback.print_exc()
                result = "error"

            with open(args.result_file, "w") as file:
                file.write(result)
