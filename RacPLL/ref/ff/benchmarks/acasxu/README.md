These is the acasxu benchmark category. The folder contains .onnx and .vnnlib files used for the category. The acasxu_instances.csv containts the full list of benchmark instances, one per line: onnx_file,vnn_lib_file,timeout_secs
 
This benchmark uses the ACAS Xu networks (from "Reluplex: An efficient SMT solver for verifying deep neural networks"), properties 1-4 run on all networks.

The .vnnlib and .csv files were created with the included generate.py script.


ACASXU-HARD: following instances are considered as ACAS-HARD

net 4-6, prop 1
net 4-8, prop 1
net 3-3, prop 2
net 4-2, prop 2
net 4-9, prop 2
net 5-3, prop 2
net 3-6, prop 3
net 5-1, prop 3
net 1-9, prop 7
net 3-3, prop 9
