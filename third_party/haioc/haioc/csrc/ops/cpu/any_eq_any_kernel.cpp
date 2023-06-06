#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace haioc {
    namespace ops {
        namespace {
            template<typename scalar_t, typename index_t>
            static void any_eq_any_forward_kernel_impl(
                    const at::TensorAccessor<scalar_t, 2> input,
                    const at::TensorAccessor<scalar_t, 1> other,
                    at::TensorAccessor<bool, 1> output) {
                CPU_1D_KERNEL_LOOP_T(i, input.size(0), index_t) {
                    for (index_t j = 0; j < input.size(1); j++) {
                        for (index_t k = 0; k < other.size(0); k++) {
                            if (input[i][j] == other[k]) {
                                output[i] = true;
                                goto next;
                            }
                        }
                    }
                    next:
                    continue;
                }
            }

            at::Tensor any_eq_any_forward_kernel(
                    const at::Tensor &input,
                    const at::Tensor &other) {
                at::CheckedFrom c = "any_eq_any_forward";
                auto args = {
                        at::TensorArg(input, "input", 1),
                        at::TensorArg(other, "other", 2)};
                at::checkAllSameType(c, args);

                TORCH_CHECK(input.ndimension() > 1,
                            "input must be Nd tensor with N > 1. Got input.ndimension() = ",
                            input.ndimension())
                const int64_t n_kernels = input.size(0);
                at::Tensor output = at::zeros({input.size(0)},
                                              at::TensorOptions().dtype(at::kBool).device(at::kCPU));

                auto input_flatten = input.flatten(1);
                auto other_flatten = other.flatten();

                AT_DISPATCH_ALL_TYPES(
                        input.scalar_type(), "any_eq_any_forward_cpu", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto output_accessor =
                                output.accessor<bool, 1>();
                        any_eq_any_forward_kernel_impl<scalar_t, index_t>(
                                input_flatten.accessor<scalar_t, 2>(),
                                other_flatten.accessor<scalar_t, 1>(),
                                output_accessor);
                    }));
                }));
                return output;
            }
        }

        TORCH_LIBRARY_IMPL(haioc, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::any_eq_any"),
                    TORCH_FN(any_eq_any_forward_kernel));
        }
    }
}
