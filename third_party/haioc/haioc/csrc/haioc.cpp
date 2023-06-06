#ifdef USE_PYTHON
#include <Python.h>
#endif // USE_PYTHON

#include <torch/script.h>
#include "haioc.h"

#ifdef WITH_CUDA

#include <cuda.h>

#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension.
// For PyMODINIT_FUNC to work, we need to include Python.h
#ifdef _WIN32
#ifdef USE_PYTHON
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    return NULL;
}
#endif // USE_PYTHON
#endif // _WIN32

namespace haioc {
    int64_t cuda_version() {
#ifdef WITH_CUDA
        return CUDA_VERSION;
#else
        return -1;
#endif
    }

    TORCH_LIBRARY_FRAGMENT(haioc, m) {
        m.def("_cuda_version", &cuda_version);
    }
}
