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
            static __global__ void fill_if_eq_any_forward_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 1> input,
                    const at::GenericPackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> other,
                    scalar_t fill_value,
                    at::GenericPackedTensorAccessor<scalar_t, 1> output) {
                CUDA_1D_KERNEL_LOOP_T(i, input.size(0), index_t) {
                    for (index_t j = 0; j < other.size(0); j++) {
                        if (input[i] == other[j]) {
                            output[i] = fill_value;
                            continue;
                        }
                    }
                }
            }

            at::Tensor fill_if_eq_any_forward_kernel(
                    at::Tensor &input,
                    const at::Tensor &other,
                    const double fill_value,
                    const bool inplace) {
                at::CheckedFrom c = "fill_if_eq_any_forward";
                auto args = {
                        at::TensorArg(input, "input", 1),
                        at::TensorArg(other, "other", 2)};
                at::checkAllSameType(c, args);
                at::checkAllSameGPU(c, args);

                at::cuda::CUDAGuard device_guard(input.get_device());
                const int64_t n_kernels = input.numel();
                at::Tensor output;
                if (!inplace)
                    output = input.clone();
                else
                    output = input;

                auto input_flatten = input.flatten();
                auto other_flatten = other.flatten();
                auto output_flatten = output.flatten();

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                AT_DISPATCH_ALL_TYPES(
                        input.scalar_type(), "fill_if_eq_any_forward_cuda", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto output_accessor = output_flatten.generic_packed_accessor<scalar_t, 1>();
                        fill_if_eq_any_forward_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                input_flatten.generic_packed_accessor<scalar_t, 1>(),
                                other_flatten.generic_packed_accessor<scalar_t, 1, at::RestrictPtrTraits, index_t>(),
                                static_cast<scalar_t>(fill_value),
                                output_accessor);
                    }));
                }));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return output;
            }
        }

        TORCH_LIBRARY_IMPL(haioc, CUDA, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::fill_if_eq_any"),
                    TORCH_FN(fill_if_eq_any_forward_kernel));
        }
    }
}
