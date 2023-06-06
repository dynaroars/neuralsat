"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/extension.py
"""

import ctypes
import importlib
import os
import sys
from warnings import warn

import torch


def _get_extension_path(lib_name):
    lib_dir = os.path.dirname(__file__)
    if os.name == 'nt':
        # Register the main library location on the default DLL path
        import ctypes
        import sys

        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        if sys.version_info >= (3, 8):
            os.add_dll_directory(lib_dir)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(lib_dir)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                raise err

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError

    return ext_specs.origin


_HAS_OPS = False


def _has_ops():
    return False


try:
    # On Windows Python-3.8.x has `os.add_dll_directory` call,
    # which is called to configure dll search path.
    # To find cuda related dlls we need to make sure the
    # conda environment/bin path is configured Please take a look:
    # https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python
    # Please note: if some path can't be added using add_dll_directory we simply ignore this path
    if os.name == 'nt' and (3, 8) <= sys.version_info < (3, 9):
        env_path = os.environ['PATH']
        path_arr = env_path.split(';')
        for path in path_arr:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)  # type: ignore[attr-defined]
                except Exception:
                    pass

    lib_path = _get_extension_path('_C')
    torch.ops.load_library(lib_path)
    _HAS_OPS = True


    def _has_ops():  # noqa: F811
        return True

except (ImportError, OSError):
    pass


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            'Couldn\'t load custom C++ ops. Recompile C++ extension with:\n'
            '\tpython setup.py build_ext --inplace'
        )


def _check_cuda_version(minor=True):
    """
    Make sure that CUDA versions match between the pytorch install and C++ extension install

    Args:
        minor (bool): If ``False``, ignore minor version difference.
         Defaults to ``True``.
    """
    if not _HAS_OPS:
        return -1
    from torch.version import cuda as torch_version_cuda

    _version = torch.ops.haioc._cuda_version()
    if _version != -1 and torch_version_cuda is not None:
        ext_version = str(_version)
        if int(ext_version) < 10000:
            ext_major = int(ext_version[0])
            ext_minor = int(ext_version[2])
        else:
            ext_major = int(ext_version[0:2])
            ext_minor = int(ext_version[3])
        t_version = torch_version_cuda.split('.')
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != ext_major or (minor and t_minor != ext_minor):
            raise RuntimeError(
                'Detected that PyTorch and Extension were compiled with different CUDA versions. '
                f'PyTorch has CUDA Version={t_major}.{t_minor} and '
                f'Extension has CUDA Version={ext_major}.{ext_minor}. '
                'Please reinstall the Extension that matches your PyTorch install.'
            )
    return _version


def _load_library(lib_name):
    lib_path = _get_extension_path(lib_name)
    # On Windows Python-3.8+ has `os.add_dll_directory` call,
    # which is called from _get_extension_path to configure dll search path
    # Condition below adds a workaround for older versions by
    # explicitly calling `LoadLibraryExW` with the following flags:
    #  - LOAD_LIBRARY_SEARCH_DEFAULT_DIRS (0x1000)
    #  - LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR (0x100)
    if os.name == 'nt' and sys.version_info < (3, 8):
        _kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        if hasattr(_kernel32, 'LoadLibraryExW'):
            _kernel32.LoadLibraryExW(lib_path, None, 0x00001100)
        else:
            warn('LoadLibraryExW is missing in kernel32.dll')

    torch.ops.load_library(lib_path)


_check_cuda_version(False)


###################
# Exposed functions
###################
def has_ops():
    """
    Check if C++ extension is successfully compiled.
    """
    return _HAS_OPS


def cuda_version():
    """
    Get compiled Cuda version.
    """
    if not _HAS_OPS:
        return -1
    return torch.ops.haioc._cuda_version()


def with_cuda():
    """
    Check if C++ extension is compiled with Cuda.
    """
    return cuda_version() != -1
