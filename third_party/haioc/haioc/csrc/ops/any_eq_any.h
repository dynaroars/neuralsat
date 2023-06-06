#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace haioc {
    namespace ops {
        HAIOC_API at::Tensor any_eq_any(
                const at::Tensor &input,
                const at::Tensor &other);
    }
}
