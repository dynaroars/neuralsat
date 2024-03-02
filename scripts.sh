# easy - hidden split
python3 profile_.py --net example/onnx/mnist-net_256x2.onnx --spec example/vnnlib/prop_1_0.03.vnnlib

# easy - input split
python3 profile_.py --net example/onnx/ACASXU_run2a_1_1_batch_2000.onnx --spec example/vnnlib/prop_1.vnnlib

# hard - hidden split
python3 profile_.py --net example/onnx/mnist-net_256x4.onnx --spec example/vnnlib/prop_1_0.03.vnnlib

# hard - input split
python3 profile_.py --net example/onnx/ACASXU_run2a_3_3_batch_2000.onnx --spec example/vnnlib/prop_2.vnnlib

# proton-viewer 
proton-viewer -m time/ms output.hatchet
