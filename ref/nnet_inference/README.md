### Loss

##### Installation Notes:

Please install [Marabou](https://github.com/NeuralNetworkVerification/Marabou) from source with python binding.
This is done by specifying `-DBUILD_PYTHON=ON` when invoking CMake 
([documentation here](https://neuralnetworkverification.github.io/Marabou/Setup/0_Installation.html)). \
Then preferably, define the `MARABOU_ROOT` variable in `prophecy/_import_maraboupy.py`.
