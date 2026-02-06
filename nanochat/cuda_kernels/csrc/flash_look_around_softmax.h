// flash_look_around_softmax.h - Online softmax with 5-tap convolution
// Extends FlashAttention-3's softmax pattern to handle look-around attention
//
// The 5-tap convolution formula:
//   P_conv[j] = w0*P[j+2] + w1*P[j+1] + w2*P[j] + w3*P[j-1] + w4*P[j-2]
//
// This requires finding a shared max across all 5 shifted score positions
// to ensure numerical stability when applying softmax to convolved outputs.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cmath>

namespace flash_look_around {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Thread-level reduce max across a tensor
template <bool Is_first, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void reduce_max(Tensor0 const& tensor, Tensor1& max_val) {
    static_assert(decltype(size(tensor))::value == decltype(size(max_val))::value);
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        if constexpr (Is_first) {
            max_val(i) = tensor(i);
        } else {
            max_val(i) = max(max_val(i), tensor(i));
        }
    }
}

// Thread-level reduce sum across a tensor
template <bool Is_first, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void reduce_sum(Tensor0 const& tensor, Tensor1& sum_val) {
    static_assert(decltype(size(tensor))::value == decltype(size(sum_val))::value);
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        if constexpr (Is_first) {
            sum_val(i) = tensor(i);
        } else {
            sum_val(i) += tensor(i);
        }
    }
}

// Apply scale and exp2 to scores: scores = exp2((scores - max_val) * scale_log2)
template <bool Scale_max = true, bool Check_inf = false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void scale_apply_exp2(
    Tensor0& tensor,
    Tensor1 const& max_val,
    float const scale_log2
) {
    static_assert(decltype(size(tensor))::value == decltype(size(max_val))::value);
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        float scaled_score;
        if constexpr (Scale_max) {
            scaled_score = (tensor(i) - max_val(i)) * scale_log2;
        } else {
            scaled_score = tensor(i) * scale_log2;
        }
        // Clamp to avoid exp2 overflow
        if constexpr (Check_inf) {
            if (scaled_score < -100.0f) {
                tensor(i) = 0.0f;
                continue;
            }
        }
        tensor(i) = exp2f(scaled_score);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Online softmax with 5-tap convolution for look-around attention
//
// This struct maintains running max and sum across K-blocks, supporting
// the incremental computation pattern of FlashAttention while applying
// the 5-tap convolution to attention weights.
//
// Template parameters:
//   kNRows: Number of rows (query positions) per thread
//   kNColsPerRow: Number of columns (key positions) per row per thread
template <int kNRows, int kNColsPerRow = 1>
struct SoftmaxConv {

    using TensorT = Tensor<float, Shape<Int<kNRows>>>;

    TensorT row_max;    // Running max for each row
    TensorT row_sum;    // Running sum for each row
    float softmax_scale_log2;  // scale * log2(e) for exp2

    // Convolution weights (loaded once per head)
    float w0, w1, w2, w3, w4;

    __forceinline__ __device__ SoftmaxConv(float scale, float const* conv_weights)
        : softmax_scale_log2(scale * float(M_LOG2E))
        , w0(conv_weights[0])  // Weight for shift +2
        , w1(conv_weights[1])  // Weight for shift +1
        , w2(conv_weights[2])  // Weight for shift 0 (center)
        , w3(conv_weights[3])  // Weight for shift -1
        , w4(conv_weights[4])  // Weight for shift -2
    {
        // Initialize with -inf max and zero sum
        #pragma unroll
        for (int i = 0; i < kNRows; ++i) {
            row_max(i) = -INFINITY;
            row_sum(i) = 0.0f;
        }
    }

    // Find max across all 5 shifted score positions
    // scores_halo has shape (kNRows, kNColsPerRow + 4) to include halo
    // Returns the max for positions [2, kNColsPerRow+2) after considering all shifts
    template <bool Is_first, typename TensorScores>
    __forceinline__ __device__ TensorT max_get_scale_5shifts(TensorScores& scores_halo) {
        TensorT new_max;

        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float m = Is_first ? -INFINITY : row_max(mi);

            // For each output position, find max across all 5 contributing shifts
            #pragma unroll
            for (int ni = 0; ni < kNColsPerRow; ++ni) {
                // Positions contributing to output ni (centered at ni+2 in halo):
                // shift -2: scores_halo[mi, ni]     -> P[ni-2] * w4
                // shift -1: scores_halo[mi, ni+1]   -> P[ni-1] * w3
                // shift  0: scores_halo[mi, ni+2]   -> P[ni]   * w2
                // shift +1: scores_halo[mi, ni+3]   -> P[ni+1] * w1
                // shift +2: scores_halo[mi, ni+4]   -> P[ni+2] * w0
                m = max(m, scores_halo(mi, ni));
                m = max(m, scores_halo(mi, ni + 1));
                m = max(m, scores_halo(mi, ni + 2));
                m = max(m, scores_halo(mi, ni + 3));
                m = max(m, scores_halo(mi, ni + 4));
            }
            new_max(mi) = m;
        }

        // Compute rescale factors for previous accumulator
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            if constexpr (!Is_first) {
                float scale = exp2f((row_max(mi) - new_max(mi)) * softmax_scale_log2);
                scores_scale(mi) = scale;
                row_sum(mi) *= scale;
            } else {
                scores_scale(mi) = 1.0f;
            }
            row_max(mi) = new_max(mi);
        }

        return scores_scale;
    }

    // Apply softmax with 5-tap convolution
    // Input: scores_halo of shape (kNRows, kNColsPerRow + 4)
    // Output: p_conv of shape (kNRows, kNColsPerRow) - convolved attention weights
    // Also updates row_sum
    template <bool Is_first, bool Check_inf = false, typename TensorScores, typename TensorPConv>
    __forceinline__ __device__ void online_softmax_conv(
        TensorScores& scores_halo,
        TensorPConv& p_conv
    ) {
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float max_scaled = row_max(mi) * softmax_scale_log2;
            float l_ij = 0.0f;

            #pragma unroll
            for (int ni = 0; ni < kNColsPerRow; ++ni) {
                // Compute exp2((score - max) * scale_log2) for all 5 shifts
                float s_m2 = scores_halo(mi, ni);
                float s_m1 = scores_halo(mi, ni + 1);
                float s_0  = scores_halo(mi, ni + 2);
                float s_p1 = scores_halo(mi, ni + 3);
                float s_p2 = scores_halo(mi, ni + 4);

                // Apply exp2 with max subtraction
                auto safe_exp2 = [&](float s) {
                    float scaled = s * softmax_scale_log2 - max_scaled;
                    if constexpr (Check_inf) {
                        return (scaled < -100.0f) ? 0.0f : exp2f(scaled);
                    } else {
                        return (s > -1e20f) ? exp2f(scaled) : 0.0f;
                    }
                };

                float p_m2 = safe_exp2(s_m2);
                float p_m1 = safe_exp2(s_m1);
                float p_0  = safe_exp2(s_0);
                float p_p1 = safe_exp2(s_p1);
                float p_p2 = safe_exp2(s_p2);

                // Apply 5-tap convolution
                float conv = w0 * p_p2 + w1 * p_p1 + w2 * p_0 + w3 * p_m1 + w4 * p_m2;

                p_conv(mi, ni) = conv;
                l_ij += conv;
            }

            if constexpr (Is_first) {
                row_sum(mi) = l_ij;
            } else {
                row_sum(mi) += l_ij;
            }
        }
    }

    // Combined max finding and softmax with conv in one pass (for simpler cases)
    template <bool Is_first, bool Check_inf = false, typename TensorScores, typename TensorPConv>
    __forceinline__ __device__ TensorT online_softmax_conv_fused(
        TensorScores& scores_halo,
        TensorPConv& p_conv
    ) {
        // Step 1: Find max across all shifts
        TensorT scores_scale = max_get_scale_5shifts<Is_first>(scores_halo);

        // Step 2: Apply softmax and convolution
        online_softmax_conv<Is_first, Check_inf>(scores_halo, p_conv);

        return scores_scale;
    }

    // Rescale output accumulator by scale factors
    template <typename TensorO>
    __forceinline__ __device__ void rescale_o(TensorO& o_acc, TensorT const& scores_scale) {
        // o_acc has shape (kNRows, HeadDim)
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float scale = scores_scale(mi);
            #pragma unroll
            for (int di = 0; di < size<1>(o_acc); ++di) {
                o_acc(mi, di) *= scale;
            }
        }
    }

    // Finalize: compute 1/row_sum for final normalization
    __forceinline__ __device__ TensorT finalize() {
        TensorT final_scale;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            final_scale(mi) = 1.0f / max(row_sum(mi), 1e-9f);
        }
        return final_scale;
    }

    // Get LSE (log-sum-exp) for backward pass
    __forceinline__ __device__ TensorT get_lse() {
        TensorT lse;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            // LSE = max + log(sum) = max + log(sum * exp(-max*scale)) / scale
            // Since we used exp2 with scale_log2: LSE = max + log2(sum) / log2(e)
            lse(mi) = row_max(mi) + logf(max(row_sum(mi), 1e-9f));
        }
        return lse;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Simplified softmax for backward pass (no convolution, just standard softmax)
template <int kNRows>
struct SoftmaxSimple {
    using TensorT = Tensor<float, Shape<Int<kNRows>>>;

    TensorT row_max;
    TensorT row_sum;
    float softmax_scale_log2;

    __forceinline__ __device__ SoftmaxSimple(float scale)
        : softmax_scale_log2(scale * float(M_LOG2E))
    {
        #pragma unroll
        for (int i = 0; i < kNRows; ++i) {
            row_max(i) = -INFINITY;
            row_sum(i) = 0.0f;
        }
    }

    template <bool Is_first, typename TensorScores>
    __forceinline__ __device__ TensorT online_softmax(TensorScores& scores) {
        TensorT new_max;

        // Find max
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float m = Is_first ? -INFINITY : row_max(mi);
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                m = max(m, scores(mi, ni));
            }
            new_max(mi) = m;
        }

        // Compute scale factors
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            if constexpr (!Is_first) {
                scores_scale(mi) = exp2f((row_max(mi) - new_max(mi)) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            } else {
                scores_scale(mi) = 1.0f;
            }
            row_max(mi) = new_max(mi);
        }

        // Apply exp and accumulate sum
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float max_scaled = new_max(mi) * softmax_scale_log2;
            float l_ij = 0.0f;
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                float s = scores(mi, ni);
                float p = (s > -1e20f) ? exp2f(s * softmax_scale_log2 - max_scaled) : 0.0f;
                scores(mi, ni) = p;
                l_ij += p;
            }
            if constexpr (Is_first) {
                row_sum(mi) = l_ij;
            } else {
                row_sum(mi) += l_ij;
            }
        }

        return scores_scale;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash_look_around
