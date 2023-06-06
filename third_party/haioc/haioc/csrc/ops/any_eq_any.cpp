#include "any_eq_any.h"

#include <torch/types.h>

namespace haioc {
    namespace ops {
        at::Tensor any_eq_any(
                const at::Tensor &input,
                const at::Tensor &other) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("haioc::any_eq_any", "")
                    .typed<decltype(any_eq_any)>();
            return op.call(
                    input,
                    other);
        }

        TORCH_LIBRARY_FRAGMENT(haioc, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "haioc::any_eq_any(Tensor input, Tensor other) -> Tensor")
            );
        }
    } // namespace ops
} // namespace haioc
