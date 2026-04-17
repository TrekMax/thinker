#ifndef _LINEARINT_LUNA_H_
#define _LINEARINT_LUNA_H_

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/include/cache.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_define.h"


/**
 * @brief Linear transformation with integer quantization
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (optional)
 * @param attrs Linear transformation attributes
 * @param workspace Temporary workspace tensor
 * @param output Output tensor
 * @return int32_t Operation status
 */
int32_t linearint_luna(tTensor *input, tTensor *weight, tTensor *bias, LinearIntAttrs *attrs, tTensor *workspace, tTensor *output) {
    tShape new_shape;

    // Reshape input tensor for 2D processing
    if (input->shape_.ndim_ == 1) {
        new_shape.ndim_ = 2;
        new_shape.dims_[1] = input->shape_.dims_[0];
        new_shape.dims_[0] = 1;
    } else if (input->shape_.ndim_ == 3) {
        new_shape.ndim_ = 2;
        new_shape.dims_[0] = input->shape_.dims_[0] * input->shape_.dims_[1];
        new_shape.dims_[1] = input->shape_.dims_[2];
    } else {
        new_shape = input->shape_;
    }

    // Check if input and output are in PSram
    int32_t x_in_psram = (input->mem_.type_ != 2);
    int32_t y_in_psram = (output->mem_.type_ != 2);

    // Validate input data type
    if (input->dtype_ != Int8 || weight->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }

    if (attrs->transB != 1) {
        return T_ERR_INVALID_PARA;
    }

    // Determine output index and tensor dimensions
    int32_t ou_idx = (output->dtype_ & 0xF) >> 1;
    int32_t n_dim = new_shape.ndim_;
    int32_t M = new_shape.dims_[n_dim - 2];
    int32_t N = new_shape.dims_[n_dim - 1];
    int32_t L = weight->shape_.dims_[0];
    int32_t input_size =  M * N;
    int32_t output_size = M * L;

    // Validate weight dimensions
    if (weight->shape_.dims_[n_dim - 1] != new_shape.dims_[n_dim - 1]) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Data pointers
    int8_t *src = (int8_t *)input->dptr_;
    int8_t *p_weight = (int8_t *)weight->dptr_;
    int32_t *p_bias = bias ? (int32_t *)bias->dptr_ : NULL;
    int8_t *dst = (int8_t *)output->dptr_;
    int32_t workspace_size = (workspace != NULL) ? workspace->shape_.dims_[0] : 0;

    // Quantization scales and shift
    int32_t q_i = (int32_t)input->scale_;
    int32_t q_w = (int32_t)weight->scale_;
    int32_t q_o = (int32_t)output->scale_;
    int32_t shift = q_i + q_w - q_o;

    // Check shift validity
    if ((shift < 0) && (output->dtype_ == Int8)) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Temporary workspace pointer
    int8_t *p_tmp = (workspace != NULL) ? (int8_t *)workspace->dptr_ : NULL;
    int8_t *p_src = p_tmp;
    int32_t offset = input_size;

    // Main computation based on data types
    if (ALIGN4(M) * ALIGN8(N) <= 65536) {
        THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(src, p_src, M, N), "luna_mat_trans_i8o8");
    }
    else if (x_in_psram == 1) {
        int8_t *src_tmp = p_tmp + offset;
        THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src, src_tmp, M * N), "luna_memcpy_i8o8");
        THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src_tmp, p_src, M, N), "luna_split_mat_trans_i8o8");
    }
    else {
        THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src, p_src, M, N), "luna_split_mat_trans_i8o8");
    }

    // Execute matrix multiplication and bias addition based on data types
    if ((weight->dtype_ == Int4 || weight->dtype_ == Int8)  && output->dtype_ == Int8) {
        if (y_in_psram) {
            int8_t *dst_tmp = p_tmp + input_size;
            if (ALIGN4(L) * ALIGN8(M) > 65536) {
                dst_tmp = p_tmp + MAX(input_size, output_size);
            }
            if (weight->dtype_ == Int4)
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i4i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, ALIGN2(N), M, shift), "luna_split_mat_mul_bias_i4i8i32o8");
            else
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o8");
            if (ALIGN4(L) * ALIGN8(M) <= 65536) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i8o8");
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
            }
            else {
                int8_t *dst_tmp1 = p_tmp;
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, dst_tmp1, L, M), "luna_split_mat_trans_i8o8");
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
            }
        }
        else {
            int8_t *dst_tmp = (int8_t *)output->dptr_;
            if (ALIGN4(L) * ALIGN8(M) > 65536) {
                dst_tmp = p_tmp + input_size;
            }
            if (weight->dtype_ == Int4)
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i4i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, ALIGN2(N), M, shift), "luna_split_mat_mul_bias_i4i8i32o8");
            else
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o8");

            if (ALIGN4(L) * ALIGN8(M) <= 65536) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)((int8_t *)dst_tmp, (int8_t *)output->dptr_, L, M), "luna_mat_trans_i8o8");
            } 
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, (int8_t *)output->dptr_, L, M), "luna_split_mat_trans_i8o8");
            }
        }
    } 
    else if (weight->dtype_ == Int8 && output->dtype_ == Int16) {
        if (y_in_psram) {
            int16_t *dst_tmp = (int16_t *)(p_tmp + input_size * 2);
            if (ALIGN4(L) * ALIGN4(M) > 65536) {
                dst_tmp = (int16_t *)(p_tmp + MAX(input_size, output_size) * 2);
            }
            if (shift < 0) {
                int32_t scale_out = 1UL << (-shift);
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o16");
                THINKER_RET_CHECK(API_LIB(scale_i16i16o16)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i16i16o16");
            }
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o16");
            }

            if (ALIGN4(L) * ALIGN4(M) <= 65536) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i16o16");
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size * 2);
            }
            else {
                int16_t *dst_tmp1 = (int16_t *)p_tmp;
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)((int16_t *)dst_tmp, (int16_t *)dst_tmp1, L, M), "luna_split_mat_trans_i16o16");
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size * 2);
            }
        }
        else {
            int16_t *dst_tmp = (int16_t *)output->dptr_;
            if (ALIGN4(L) * ALIGN4(M) > 65536) {
                dst_tmp = (int16_t *)(p_tmp + input_size * 2);
            }

            if (shift < 0) {
                int32_t scale_out = 1UL << (-shift);
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o16");
                THINKER_RET_CHECK(API_LIB(scale_i16i16o16)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i16i16o16");
            }
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o16");
            }

            if (ALIGN4(L) * ALIGN4(M) <= 65536) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)(dst_tmp, (int16_t *)output->dptr_, L, M), "luna_mat_trans_i16o16");
            }
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)(dst_tmp,  (int16_t *)output->dptr_, L, M), "luna_split_mat_trans_i16o16");
            }
        }
    }
    else if (weight->dtype_ == Int8 && output->dtype_ == Int32) {
        if (y_in_psram) {
            int32_t *dst_tmp = (int32_t *)(p_tmp + input_size * 4);
            if (ALIGN2(L) * ALIGN4(M) > 32768) {
                dst_tmp = (int32_t *)(p_tmp + 4 * MAX(input_size, output_size));
            }
            if (shift < 0) {
                int32_t scale_out = 1UL << (-shift);
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o32");
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
            }
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o32");
            }

            if (ALIGN2(L) * ALIGN4(M) <= 32768) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp, L, M), "luna_mat_trans_i32o32");
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size * 4);
            }
            else {
                int32_t *dst_tmp1 = (int32_t *)p_tmp;
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp1, L, M), "luna_split_mat_trans_i32o32");
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size * 4);
            }
        }
        else {
            int32_t *dst_tmp = (int32_t *)output->dptr_;
            if (ALIGN2(L) * ALIGN4(M) > 32768) {
                dst_tmp = (int32_t *)(p_tmp + 4 * input_size);
            }
            if (shift < 0) {
                int32_t scale_out = 1UL << (-shift);
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o32");
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
            }
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o32");
            }
            if (ALIGN2(L) * ALIGN4(M) <= 32768) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_mat_trans_i32o32");
            }
            else {
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
            }
        }
    }
    else {
        return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
}

#endif  // _LINEARINT_LUNA_H_