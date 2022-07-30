import numpy as np
import pandas as pd
import datetime
import time
import glob
import os

# log path
path = "logs"
# table 1 or table 2 or table 3 (untargeted)
table = 2
# Lp norm 

index = []
if table == 1:
    # small networks
    for network in ["mnist"]:
        for layer in ["2", "3"]:
            for norm in ["Li", "L2", "L1"]:
                for target in ["top2", "random", "least"]:
                    index.append("{}-{}-{}-{}".format(network, layer, norm, target))
    neuron_ub = 100
    neuron_lb = 0
    method = ["ours", "lip", "lp", "lpfull"]
elif table == 2:
    # targeted table
    for network in ["mnist"]:
        for layer in ["2", "3", "4"]:
            for norm in ["Li", "L2", "L1"]:
                for target in ["top2", "random", "least"]:
                    index.append("{}-{}-{}-{}".format(network, layer, norm, target))
    for network in ["cifar"]:
        for layer in ["5", "6", "7"]:
            for norm in ["Li", "L2", "L1"]:
                for target in ["top2", "random", "least"]:
                    index.append("{}-{}-{}-{}".format(network, layer, norm, target))
    neuron_ub = 3000
    neuron_lb = 1000
    method = ["ours", "lip", "lp"]
elif table == 3:
    # untargeted table
    network = "mnist"
    layer = "3"
    target = "untargeted"
    for norm in ["Li", "L2", "L1"]:
        index.append("{}-{}-{}-{}".format(network, layer, norm, target))
    network = "cifar"
    layer = "5"
    target = "untargeted"
    for norm in ["Li", "L2", "L1"]:
        index.append("{}-{}-{}-{}".format(network, layer, norm, target))
    neuron_ub = 3000
    neuron_lb = 1000
    method = ["ours", "lip", "lp"]
else:
    raise RuntimeError("unknown table " + str(table))


# create the table
dist_df = pd.DataFrame(dtype=np.float64,index=index, columns=method+["ratio"])
time_df = pd.DataFrame(dtype=np.float64,index=index, columns=method+["speedup"])
epoch_df = pd.DataFrame(dtype=np.int64,index=index, columns=method)
# table for output
output_dist_df = pd.DataFrame(dtype=str,index=index, columns=method+["ratio"])
output_time_df = pd.DataFrame(dtype=str,index=index, columns=method+["speedup"])

# fill in tables
for filename in glob.iglob(path + '/**/*.log', recursive=True):
    fn = os.path.basename(filename).split('_')
    network = fn[0]
    layer = fn[2]
    neuron = int(fn[3])
    norm = fn[4]
    method = fn[5]
    target = fn[6]
    date = fn[8]
    ltime = fn[9].split('.')[0]
    epoch = int(datetime.datetime.strptime('18'+date+ltime, '%y%m%d%H%M%S').timestamp())
    row_name = "{}-{}-{}-{}".format(network, layer, norm, target)
    if row_name in dist_df.index and neuron > neuron_lb and neuron < neuron_ub:
        if method not in dist_df.columns:
            continue
        old_number = dist_df.loc[row_name][method]
        update = True
        if old_number == old_number:
            print("WARNING: conflicting", filename, "row name is", row_name)
            prev_epoch = epoch_df.loc[row_name][method]
            epoch_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))
            prev_epoch_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(prev_epoch))
            print("Time stamp: prev {} current {}".format(prev_epoch_str, epoch_str))
            if epoch < prev_epoch:
                print("Skipped outdated file")
                update = False
        if update:
            with open(filename) as f:
                lines = f.readlines()
                result = lines[-1]
                if result[:5] != "[L0] ":
                    print("skipping incomplete", filename, "row name is", row_name)
                else:
                    result = result.split(",")
                    robustness = float(result[1].split("=")[1])
                    numimg = int(result[2].split("=")[1])
                    runtime = float(result[3].split("=")[1])
                    per_img_time = runtime / numimg
                    print("filename = {}\n row = {}, robustness = {}, time = {}".format(filename, row_name, robustness, per_img_time))
                    dist_df.loc[row_name][method] = robustness
                    time_df.loc[row_name][method] = per_img_time
                    epoch_df.loc[row_name][method] = epoch

    else:
        # print("skipping", filename, "row name is", row_name)
        pass

pd.set_option('display.width', 1000) 
print(dist_df)
print(time_df)
def format_time(t):
    if t != t:
        return "-"
    if t >= 1.0:
        if t < 100:
            return "{:#.3g} s".format(t)
        else:
            return "{:.3g} s".format(t)
    else:
        t *= 1000
        if t < 100:
            return "{:#.3g} ms".format(t)
        else:
            return "{:.3g} ms".format(t)
def format_norm(n):
    if n == n:
        return "{:.5f}".format(n)
    else:
        return "-"

# compute ratios
for idx in dist_df.index.tolist():
    ours = dist_df.loc[idx]["ours"]
    lip = dist_df.loc[idx]["lip"]
    lp = dist_df.loc[idx]["lp"]
    if "lpfull" in dist_df:
        lpfull = dist_df.loc[idx]["lpfull"]
        if lpfull > lp:
            ref = lpfull
            ref_time = time_df.loc[idx]["lpfull"]
            output_dist_df.loc[idx]["lp"] = format_norm(lp)
            output_time_df.loc[idx]["lp"] = format_time(time_df.loc[idx]["lp"])
            output_dist_df.loc[idx]["lpfull"] = "\\bf "+format_norm(lpfull)
            output_time_df.loc[idx]["lpfull"] = "\\bf "+format_time(time_df.loc[idx]["lpfull"])
        else:
            ref = lp
            ref_time = time_df.loc[idx]["lp"]
            output_dist_df.loc[idx]["lp"] = "\\bf "+format_norm(lp)
            output_time_df.loc[idx]["lp"] = "\\bf "+format_time(time_df.loc[idx]["lp"])
            output_dist_df.loc[idx]["lpfull"] = format_norm(lpfull)
            output_time_df.loc[idx]["lpfull"] = format_time(time_df.loc[idx]["lpfull"])
    else:
        ref = lp
        ref_time = time_df.loc[idx]["lp"]
        if lp == lp:
            output_dist_df.loc[idx]["lp"] = "\\bf "+format_norm(lp)
            output_time_df.loc[idx]["lp"] = "\\bf "+format_time(time_df.loc[idx]["lp"])
        else:
            output_dist_df.loc[idx]["lp"] = format_norm(lp)
            output_time_df.loc[idx]["lp"] = format_time(time_df.loc[idx]["lp"])
    if ours < lip:
        new = lip
        new_time = time_df.loc[idx]["lip"]
        output_dist_df.loc[idx]["ours"] = format_norm(ours)
        output_time_df.loc[idx]["ours"] = format_time(time_df.loc[idx]["ours"])
        output_dist_df.loc[idx]["lip"] = "\\bf "+format_norm(lip)
        output_time_df.loc[idx]["lip"] = "\\bf "+format_time(time_df.loc[idx]["lip"])
    else:
        new = ours
        new_time = time_df.loc[idx]["ours"]
        output_dist_df.loc[idx]["ours"] = "\\bf "+format_norm(ours)
        output_time_df.loc[idx]["ours"] = "\\bf "+format_time(time_df.loc[idx]["ours"])
        output_dist_df.loc[idx]["lip"] = format_norm(lip)
        output_time_df.loc[idx]["lip"] = format_time(time_df.loc[idx]["lip"])
    ratio = (new - ref) / ref
    speedup = ref_time / new_time
    dist_df.loc[idx]["ratio"] = ratio
    if ratio == ratio:
        output_dist_df.loc[idx]["ratio"] = "{:+.1%}".format(ratio).replace("%", "\%")
    else:
        output_dist_df.loc[idx]["ratio"] = "-"
    time_df.loc[idx]["speedup"] = speedup
    if speedup == speedup:
        output_time_df.loc[idx]["speedup"] = "{}X".format(int(np.around(speedup)))
    else:
        output_time_df.loc[idx]["speedup"] = "-"

# add Latex &
for idx in output_dist_df.index.tolist():
    for c in output_dist_df.columns.values.tolist():
        output_dist_df.loc[idx][c] += " &"
for idx in output_time_df.index.tolist():
    for c in output_time_df.columns.values.tolist():
        output_time_df.loc[idx][c] += " &"

pd.set_option('display.width', 1000) 
print(dist_df)
print(time_df)
print(output_dist_df)
print(output_time_df)


