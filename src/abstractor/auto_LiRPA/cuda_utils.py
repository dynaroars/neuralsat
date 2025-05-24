class DummyCudaClass:
    
    """A dummy class with error message when a CUDA function is called."""
    
    def __getattr__(self, attr):
        if attr == "double2float":
            # When CUDA module is not built successfully, use a workaround.
            def _f(x, d):
                print('WARNING: Missing CUDA kernels. Please enable CUDA build by setting environment variable AUTOLIRPA_ENABLE_CUDA_BUILD=1 for the correct behavior!')
                return x.float()
            return _f
        def _f(*args, **kwargs):
            raise RuntimeError(f"method {attr} not available because CUDA module was not built.")
        return _f

_cuda_utils = DummyCudaClass()

double2float = _cuda_utils.double2float
