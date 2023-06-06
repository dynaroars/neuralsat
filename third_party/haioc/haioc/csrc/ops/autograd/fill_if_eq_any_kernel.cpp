#include "../fill_if_eq_any.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            class FillIfEqAnyFunction
                    : public torch::autograd::Function<FillIfEqAnyFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        torch::autograd::Variable &input,
                        const torch::autograd::Variable &other,
                        const double fill_value,
                        const bool inplace) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = fill_if_eq_any(
                            input,
                            other,
                            fill_value,
                            inplace);

                    return {
                            output,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    TORCH_CHECK(0, "backwards on fill_if_eq_any not supported")
                }
            };
        } // namespace

        at::Tensor fill_if_eq_any_autograd(
                at::Tensor &input,
                const at::Tensor &other,
                const double fill_value,
                const bool inplace) {
            return FillIfEqAnyFunction::apply(
                    input,
                    other,
                    fill_value,
                    inplace
            )[0];
        }

        TORCH_LIBRARY_IMPL(haioc, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::fill_if_eq_any"),
                    TORCH_FN(fill_if_eq_any_autograd));
        }
    }
}
