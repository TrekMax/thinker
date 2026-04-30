#ifndef _SWISH_LUNA_H_
#define _SWISH_LUNA_H_

#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

/**
 * @brief Quantized Swish activation function implementation (β=1)
 *        Swish(x) = x * sigmoid(x), also known as SiLU
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t swish_luna(tTensor *X, tTensor *Y, tTensor *Temp) {
    // Debug: check workspace
    if (Temp == NULL) {
        // THINKER_LOG_ERROR("swish_luna: workspace tensor is required but NULL");
        return T_ERR_NO_WORKSPACE;
    }

    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = getTensorSize(Temp);

#ifdef RUNTIME_PARAM_CHECK
    if ((Y->mem_.type_ != 2) || (Y->dtype_ != Int8))
        return T_ERR_INVALID_DATATYPE;
#endif

    // Quantization parameters (Swish uses Q27 input for sigmoid computation)
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 27;
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    int8_t *dst = (int8_t *)Y->dptr_;
    int32_t shift_i = Q_INPUT - x_q;
    int32_t shift_o = Q_OUTPUT - y_q;

    if (X->dtype_ == Int8) {
        if (workspace_size < input_size * 6) {
            return T_ERR_NO_WORKSPACE;
        }
        int8_t *src = (int8_t *)X->dptr_;
        int16_t *tmp = (int16_t *)Temp->dptr_;
        int32_t *tmp1 = (int32_t *)(tmp + input_size);

        THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(src, 1, tmp, input_size, 0), "luna_scale_i8i8o16");
        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(tmp, 1, tmp1, input_size, 0), "luna_scale_i16i16o32");
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(tmp1, 1UL << shift_i, tmp1, input_size, 0), "luna_scale_i32i32o32");
        THINKER_RET_CHECK(API_LIB(swish_i32o32)(tmp1, tmp1, input_size), "luna_swish_i32o32");
        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(tmp1, 1, dst, input_size, shift_o), "luna_scale_i32i32o8");
    }
    else if (X->dtype_ == Int16) {
        if (workspace_size < input_size * 4) {
            return T_ERR_NO_WORKSPACE;
        }
        int16_t *src = (int16_t *)X->dptr_;
        int32_t *tmp = (int32_t *)Temp->dptr_;

        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src, 1 << shift_i, tmp, input_size, 0), "luna_scale_i16i16o32");
        THINKER_RET_CHECK(API_LIB(swish_i32o32)(tmp, tmp, input_size), "luna_swish_i32o32");
        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(tmp, 1, dst, input_size, shift_o), "luna_scale_i32i32o8");
    }
    else if (X->dtype_ == Int32) {
        if (workspace_size < input_size * 4) {
            return T_ERR_NO_WORKSPACE;
        }
        int32_t *src = (int32_t *)X->dptr_;
        int32_t *tmp = (int32_t *)Temp->dptr_;

        if (shift_i != 0) {
            uint32_t shift1 = shift_i > 0 ? shift_i : 0;
            uint32_t shift2 = shift_i > 0 ? 0 : -shift_i;

            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, 1 << shift1, tmp, input_size, shift2), "luna_scale_i32i32o32");
            src = tmp;
        }
        THINKER_RET_CHECK(API_LIB(swish_i32o32)(src, tmp, input_size), "luna_swish_i32o32");
        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(tmp, 1, dst, input_size, shift_o), "luna_scale_i32i32o8");
    }
    else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

#endif  // _SWISH_LUNA_H_