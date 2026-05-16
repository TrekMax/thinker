#ifndef _GLU_LUNA_H_
#define _GLU_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

static int32_t gluint_dtype_size(uint16_t dtype) {
    if (dtype == Int8) {
        return 1;
    } else if (dtype == Int16) {
        return 2;
    } else if (dtype == Int32) {
        return 4;
    }
    return 0;
}

static int32_t gluint_scale_i32_inplace(int32_t *src, int32_t size, int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
        scalar = 1 << (uint32_t)shift;
    } else {
        rshift = (uint32_t)(-shift);
    }
    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, scalar, src, size, rshift),
                      "luna_scale_i32i32o32");
    return T_SUCCESS;
}

static int32_t gluint_mul_i32_to_dtype(const int32_t *src1, const int32_t *src2,
                                       void *dst, uint16_t dtype, int32_t size,
                                       int32_t shift) {
    uint32_t rshift = 0;
    if (shift > 0) {
        rshift = (uint32_t)shift;
    }

    if (dtype == Int8) {
        THINKER_RET_CHECK(API_LIB(mul_i32i32o8)(src1, src2, (int8_t *)dst, size, rshift),
                          "luna_mul_i32i32o8");
    } else if (dtype == Int16) {
        THINKER_RET_CHECK(API_LIB(mul_i32i32o16)(src1, src2, (int16_t *)dst, size, rshift),
                          "luna_mul_i32i32o16");
    } else if (dtype == Int32) {
        THINKER_RET_CHECK(API_LIB(mul_i32i32o32)(src1, src2, (int32_t *)dst, size, rshift),
                          "luna_mul_i32i32o32");
    } else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

static int32_t gluint_workspace_bytes(int32_t size) {
    return ALIGN4(size * (int32_t)sizeof(int16_t)) +
           size * (int32_t)sizeof(int32_t) * 2;
}

static int32_t gluint_split_size(int32_t output_size, int32_t workspace_size) {
    int32_t split = workspace_size / 10;
    if (split > output_size) {
        split = output_size;
    }
    while (split > 0 && gluint_workspace_bytes(split) > workspace_size) {
        split--;
    }
    return split;
}

/// Gated Linear Unit (Glu) operation implementation
int32_t gluint_luna(tTensor *X, tTensor *Y, tTensor *workspace, GluIntAttrs *attr) {
    const int32_t Q_SIGMOID_IN = 27;
    const int32_t Q_SIGMOID_OUT = 15;

    if (X->dtype_ != Int8 || gluint_dtype_size(Y->dtype_) == 0) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (workspace == NULL) {
        return T_ERR_NO_WORKSPACE;
    }

    int32_t axis = attr->axis;
    axis = (axis < 0) ? ((int32_t)X->shape_.ndim_ + axis) : axis;
    if (axis < 0 || axis >= (int32_t)X->shape_.ndim_) {
        return T_ERR_INVALID_PARA;
    }

    int32_t axis_size = (int32_t)X->shape_.dims_[axis];
    if ((axis_size & 1) != 0) {
        return T_ERR_INVALID_PARA;
    }

    int32_t outer = 1;
    int32_t inner = 1;
    for (int32_t i = 0; i < axis; ++i) {
        outer *= X->shape_.dims_[i];
    }
    for (int32_t i = axis + 1; i < (int32_t)X->shape_.ndim_; ++i) {
        inner *= X->shape_.dims_[i];
    }

    int32_t half_axis = axis_size / 2;
    int32_t block_size = half_axis * inner;
    int32_t workspace_size = (int32_t)getTensorSize(workspace) * workspace->byte_;
    int32_t split_size = gluint_split_size(block_size, workspace_size);
    if (split_size <= 0) {
        return T_ERR_NO_WORKSPACE;
    }

    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t sigmoid_shift = Q_SIGMOID_IN - x_q;
    int32_t mul_shift = Q_SIGMOID_OUT + x_q - y_q;
    int32_t out_bytes = gluint_dtype_size(Y->dtype_);
    int32_t y_in_psram = (Y->mem_.type_ != 2);

    for (int32_t o = 0; o < outer; ++o) {
        int8_t *input_base = (int8_t *)X->dptr_ + o * axis_size * inner;
        int8_t *a_base = input_base;
        int8_t *b_base = input_base + block_size;
        int8_t *out_base = (int8_t *)Y->dptr_ + o * block_size * out_bytes;

        for (int32_t offset = 0; offset < block_size; offset += split_size) {
            int32_t cur_size = block_size - offset;
            if (cur_size > split_size) {
                cur_size = split_size;
            }

            int8_t *tmp = (int8_t *)workspace->dptr_;
            int16_t *tmp_i16 = (int16_t *)tmp;
            int32_t *a_i32 = (int32_t *)(tmp + ALIGN4(cur_size * (int32_t)sizeof(int16_t)));
            int32_t *b_i32 = a_i32 + cur_size;
            void *dst = (void *)(out_base + offset * out_bytes);

            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(a_base + offset, 1, tmp_i16, cur_size, 0),
                              "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(tmp_i16, 1, a_i32, cur_size, 0),
                              "luna_scale_i16i16o32");
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(b_base + offset, 1, tmp_i16, cur_size, 0),
                              "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(tmp_i16, 1, b_i32, cur_size, 0),
                              "luna_scale_i16i16o32");
            THINKER_RET_CHECK(gluint_scale_i32_inplace(b_i32, cur_size, sigmoid_shift),
                              "gluint_scale_i32_inplace");
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o32)(b_i32, b_i32, cur_size),
                              "luna_sigmoid_i32o32");
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(b_i32, 1, b_i32, cur_size, 16),
                              "luna_scale_i32i32o32");

            if (mul_shift < 0) {
                THINKER_RET_CHECK(gluint_scale_i32_inplace(a_i32, cur_size, -mul_shift),
                                  "gluint_scale_i32_inplace");
            }

            if (y_in_psram) {
                dst = (Y->dtype_ == Int32) ? (void *)a_i32 : (void *)tmp;
            }

            THINKER_RET_CHECK(gluint_mul_i32_to_dtype(a_i32, b_i32, dst, Y->dtype_,
                                                      cur_size, mul_shift),
                              "gluint_mul_i32_to_dtype");

            if (y_in_psram) {
                opi_psram_cpy_out(out_base + offset * out_bytes, dst, cur_size * out_bytes);
            }
        }
    }

    return T_SUCCESS;
}

#endif  //_GLU_LUNA_H_