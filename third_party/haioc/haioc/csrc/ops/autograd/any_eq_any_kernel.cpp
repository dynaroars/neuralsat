#include "../any_eq_any.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            class AnyEqAnyFunction
                    : public torch::autograd::Function<AnyEqAnyFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &input,
                        const torch::autograd::Variable &other) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = any_eq_any(
                            input,
                            other);

                    return {
                            output,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    TORCH_CHECK(0, "backwards on any_eq_any not supported")
                }
            };
        } // namespace

        at::Tensor any_eq_any_autograd(
                const at::Tensor &input,
                const at::Tensor &other) {
            return AnyEqAnyFunction::apply(
                    input,
                    other
            )[0];
        }

        TORCH_LIBRARY_IMPL(haioc, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::any_eq_any"),
                    TORCH_FN(any_eq_any_autograd));
        }
    }
}
