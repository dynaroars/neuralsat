#include <ATen/ATen.h>
#include <torch/library.h>

#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace haioc {
    namespace ops {
        namespace {
            inline unsigned int GET_THREADS() {
                return 1024;
            }

            template<typename scalar_t, typename index_t>
            static __global__ void any_eq_any_forward_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, index_t> input,
                    const at::GenericPackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> other,
                    at::GenericPackedTensorAccessor<bool, 1, at::RestrictPtrTraits, index_t> output) {
                CUDA_1D_KERNEL_LOOP_T(i, input.size(0), index_t) {
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
                at::checkAllSameGPU(c, args);

                TORCH_CHECK(input.ndimension() > 1,
                            "input must be Nd tensor with N > 1. Got input.ndimension() = ",
                            input.ndimension())
                at::cuda::CUDAGuard device_guard(input.get_device());
                const int64_t n_kernels = input.size(0);
                at::Tensor output = at::zeros({input.size(0)},
                                              at::TensorOptions().dtype(at::kBool).device(input.device()));

                auto input_flatten = input.flatten(1);
                auto other_flatten = other.flatten();

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                AT_DISPATCH_ALL_TYPES(
                        input.scalar_type(), "any_eq_any_forward_cuda", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto output_accessor =
                                output.generic_packed_accessor<bool, 1, at::RestrictPtrTraits, index_t>();
                        any_eq_any_forward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                input_flatten.generic_packed_accessor<scalar_t, 2, at::RestrictPtrTraits, index_t>(),
                                other_flatten.generic_packed_accessor<scalar_t, 1, at::RestrictPtrTraits, index_t>(),
                                output_accessor);
                    }));
                }));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return output;
            }
        }

        TORCH_LIBRARY_IMPL(haioc, CUDA, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::any_eq_any"),
                    TORCH_FN(any_eq_any_forward_kernel));
        }
    }
}
