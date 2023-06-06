#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace haioc {
    namespace ops {
        HAIOC_API at::Tensor fill_if_eq_any(
                at::Tensor &input,
                const at::Tensor &other,
                double fill_value,
                bool inplace);
    }
}
