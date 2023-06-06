#pragma once

// index type
#define HAIOC_DISPATCH_INDEX_TYPE(N_KERNELS, ...)     \
  if (((int64_t)N_KERNELS) > (1 << 31)) {             \
    using index_t = int64_t;                          \
    __VA_ARGS__();                                    \
  }                                                   \
  else {                                              \
    using index_t = int;                              \
    __VA_ARGS__();                                    \
  }
