#ifndef _SOFTMAXINT_LUNA_H_
#define _SOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Perform integer softmax operation
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Temporary workspace for calculations
 * @param attrs Softmax attributes containing axis and scaling parameters
 * @return Execution status
 */
int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, SoftmaxIntAttrs *attrs) {
    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;

    int32_t leading = 1, stride = 1;
    int32_t axis = attrs->axis < 0 ? (data->shape_.ndim_ + attrs->axis) : attrs->axis;

    // Calculate leading and stride dimensions based on axis
    for (int32_t i = 0; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (int32_t i = axis; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }
    int32_t data_size = leading * stride;

    if (Int8 != data->dtype_ && Int16 != data->dtype_ && Int32 != data->dtype_) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (Int8 != out->dtype_ && Int16 != out->dtype_ && Int32 != out->dtype_) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Check if output is in PSRAM
    int32_t x_in_psram = (data->mem_.type_ != 2) ? 1 : 0;
    int32_t y_in_psram = (out->mem_.type_ != 2) ? 1 : 0;
    int32_t workspace_size = Workspace ? Workspace->shape_.dims_[0] : 0;
    int32_t x_scale = (int32_t)data->scale_;
    int32_t y_scale = (int32_t)out->scale_;

    // Process based on input data type
    if ((data->dtype_ == Int8) && (out->dtype_ == Int8)) {
        int32_t split_leading = 0;
        int32_t paste_leading = 0;
        while ((ALIGN4(split_leading * stride * 2) + split_leading * stride * 4) < workspace_size)
            split_leading++;
        while(paste_leading < leading) {
            int32_t remain_leading = leading - paste_leading;
            int32_t cur_leading = remain_leading > split_leading ? split_leading : remain_leading;
            int32_t cur_size = cur_leading * stride;
            int16_t *p_tmp0 = (int16_t *)Workspace->dptr_;
            int32_t *p_tmp1 = (int32_t *)((int8_t *)Workspace->dptr_ + ALIGN4(cur_size * 2));
            int32_t *dst_tmp = p_tmp1;
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)((int8_t *)data->dptr_ + paste_leading * stride, 1, p_tmp0, cur_size, 0), "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(p_tmp0, 1, p_tmp1, cur_size, 0), "luna_scale_i16i16o32");
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp1, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp1, cur_size, 0), "luna_scale_i32i32o32");

            // Compute softmax
            for (int32_t l = 0; l < cur_leading; ++l) {
                int32_t offset = l * stride;
                THINKER_RET_CHECK(API_LIB(softmax_i32o32)(p_tmp1 + offset, p_tmp1 + offset, stride), "luna_softmax_i32o32");
            }

            if (y_in_psram) {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)dst_tmp, cur_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o8");
                opi_psram_cpy_out(dst_tmp, (int8_t *)out->dptr_ + paste_leading * stride, cur_size);
            } 
            else {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_ + paste_leading * stride, cur_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o32");
            }
            paste_leading += cur_leading;
        }
    } else if (data->dtype_ == Int16) {
        int32_t *p_tmp = (int32_t *)Workspace->dptr_;
        int32_t *dst_tmp = p_tmp + 4 * data_size;

        // Scale from Int16 to Int32
        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)((int16_t *)data->dptr_, 1, p_tmp, data_size, 0), "luna_scale_i16i16o32");
        // Apply input scaling
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_tmp, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp, data_size, 0), "luna_scale_i32i32o32");

        // Compute softmax
        for (int32_t l = 0; l < leading; ++l) {
            int32_t offset = l * stride;
            THINKER_RET_CHECK(API_LIB(softmax_i32o32)(p_tmp + offset, (int32_t *)dst_tmp + offset, stride), "luna_softmax_i32o32");
        }

        // Scale output based on output data type
        if (out->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o8");
        } else if (out->dtype_ == Int16) {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o16)((int32_t *)dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o16");
        } else {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o32");
        }
    } else if (data->dtype_ == Int32) {
        int32_t *p_tmp = (int32_t *)Workspace->dptr_;
        int32_t *dst_tmp = p_tmp + 4 * stride;

        // Apply input scaling
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)data->dptr_, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp, data_size, 0), "luna_scale_i32i32o32");

        // Compute softmax
        for (int32_t l = 0; l < leading; ++l) {
            int32_t offset = l * stride;
            THINKER_RET_CHECK(API_LIB(softmax_i32o32)(p_tmp + offset, (int32_t *)dst_tmp + offset, stride), "luna_softmax_i32o32");
        }

        // Scale output based on output data type
        if (out->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o8");
        } else if (out->dtype_ == Int16) {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o16)((int32_t *)dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o16");
        } else {
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale)), "luna_scale_i32i32o32");
        }
    }

    return T_SUCCESS;
}

#endif