import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

for p in POLICIES:
    p['lib_whitelist'].extend([
        'libtorch.so',
        'libtorch_cpu.so',
        'libtorch_cuda.so',
        'libtorch_python.so',
        'libc10.so',
        'libc10_cuda.so',
    ])

if __name__ == "__main__":
    sys.exit(main())
