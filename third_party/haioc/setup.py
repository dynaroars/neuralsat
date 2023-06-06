import glob
import os
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension

PACKAGE_ROOT = 'haioc'


def get_version(version_file='_version.py'):
    import importlib.util
    version_file_path = os.path.abspath(os.path.join(PACKAGE_ROOT, version_file))
    try:
        spec = importlib.util.spec_from_file_location('_version', version_file_path)
        version_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version_module)
        return str(version_module.__version__)
    except:
        return '0.0.0'


def get_extensions():
    extensions_dir = os.path.join(PACKAGE_ROOT, 'csrc')

    main_file = (glob.glob(os.path.join(extensions_dir, '*.cpp')) +
                 glob.glob(os.path.join(extensions_dir, 'ops', '*.cpp'))
                 )

    source_cpu = (glob.glob(os.path.join(extensions_dir, 'ops', 'cpu', '*.cpp')) +
                  glob.glob(os.path.join(extensions_dir, 'ops', 'dispatch', '*.cpp')) +
                  glob.glob(os.path.join(extensions_dir, 'ops', 'autograd', '*.cpp')) +
                  glob.glob(os.path.join(extensions_dir, 'ops', 'quantized', 'cpu', '*.cpp'))
                  )

    source_cuda = glob.glob(os.path.join(extensions_dir, 'ops', 'cuda', '*.cu'))
    source_cuda += glob.glob(os.path.join(extensions_dir, 'ops', 'autocast', '*.cpp'))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []

    print('Compiling extensions with following flags:')
    force_cuda = os.getenv('FORCE_CUDA', '0') == '1'
    print(f'  FORCE_CUDA: {force_cuda}')
    debug_mode = os.getenv('DEBUG', '0') == '1'
    print(f'  DEBUG: {debug_mode}')

    nvcc_flags = os.getenv('NVCC_FLAGS', '')
    print(f'  NVCC_FLAGS: {nvcc_flags}')

    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args['nvcc'] = nvcc_flags

    if sys.platform == 'win32':
        define_macros += [(f'{PACKAGE_ROOT}_EXPORTS', None)]
        define_macros += [('USE_PYTHON', None)]
        extra_compile_args['cxx'].append('/MP')

    if debug_mode:
        print('Compiling in debug mode')
        extra_compile_args['cxx'].append('-g')
        extra_compile_args['cxx'].append('-O0')
        if 'nvcc' in extra_compile_args:
            # we have to remove '-OX' and '-g' flag if exists and append
            nvcc_flags = extra_compile_args['nvcc']
            extra_compile_args['nvcc'] = [f for f in nvcc_flags if not ('-O' in f or '-g' in f)]
            extra_compile_args['nvcc'].append('-O0')
            extra_compile_args['nvcc'].append('-g')

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            f'{PACKAGE_ROOT}._C',
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


def setup_package():
    setup(
        version=get_version(),
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)
        },
    )


if __name__ == '__main__':
    setup_package()
