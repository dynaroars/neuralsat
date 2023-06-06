#include "fill_if_eq_any.h"

#include <torch/types.h>

namespace haioc {
    namespace ops {
        at::Tensor fill_if_eq_any(
                at::Tensor &input,
                const at::Tensor &other,
                const double fill_value = 0.,
                const bool inplace = false) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("haioc::fill_if_eq_any", "")
                    .typed<decltype(fill_if_eq_any)>();
            return op.call(
                    input,
                    other,
                    fill_value,
                    inplace);
        }

        TORCH_LIBRARY_FRAGMENT(haioc, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "haioc::fill_if_eq_any(Tensor input, Tensor other, float fill_value, bool inplace) -> Tensor")
            );
        }
    } // namespace ops
} // namespace haioc
