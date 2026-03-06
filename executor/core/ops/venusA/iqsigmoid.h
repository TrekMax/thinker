#ifndef _SIGMOID_LUNA_H_
#define _SIGMOID_LUNA_H_

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
 * @brief Quantized sigmoid activation function implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t iqsigmoid(tTensor *X, tTensor *Y, tTensor *Temp) {
    uint32_t input_size = getTensorSize(X);
    uint32_t workspace_size = getTensorSize(Temp);

#ifdef RUNTIME_PARAM_CHECK
    /*Check the storage locations for input and output, 
    as it is unnecessary because they have already been limited in tpacker.*/
    if ((Y->mem_.type_ != 2)|| (Y->dtype_ != Int8)) 
        return T_ERR_INVALID_DATATYPE;
#endif
    // Quantization parameters
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    // Pointers to tensor data
    int8_t *dst = (int8_t *)Y->dptr_;

    // Quantization shift
    int32_t shift = Q_INPUT - x_q;

    // Perform quantized sigmoid computation
    if (X->dtype_ == Int8) {
        // Check if temporary workspace is sufficient
        if (workspace_size < input_size * 6) {
            return T_ERR_NO_WORKSPACE;
        }
        int8_t *src = (int8_t *)X->dptr_;
        int16_t *tmp = (int16_t *)Temp->dptr_;
        int32_t *tmp1 = (int32_t *)(tmp + input_size);
        THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(src, 1, tmp, input_size, 0), "luna_scale_i8i8o16");  // Convert Int8 to Int16, venusA not support Int8 => Int32
        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(tmp, 1UL << shift, tmp1, input_size, 0), "luna_scale_i16i16o32");  // Scale to Int32
        THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(tmp1, dst, input_size), "luna_sigmoid_i32o8");  // Apply sigmoid activation and convert to Int8
    }
    else if (X->dtype_ == Int16) {
        // Check if temporary workspace is sufficient
        if (workspace_size < input_size * 4) {
            return T_ERR_NO_WORKSPACE;
        }
        int16_t *src = (int16_t *)X->dptr_;
        int32_t *tmp = (int32_t *)Temp->dptr_;
        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src, 1 << shift, tmp, input_size, 0), "luna_scale_i16i16o32");  // Scale to Int32
        THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(tmp, dst, input_size), "luna_sigmoid_i32o8");  // Apply sigmoid activation and convert to Int8
    }
    else if (X->dtype_ == Int32) {
        int32_t *src = (int32_t *)X->dptr_;
        if (shift != 0) {
            // Check if temporary workspace is sufficient
            if (workspace_size < input_size * 4) {
                return T_ERR_NO_WORKSPACE;
            }
            int32_t *src = (int32_t *)X->dptr_;
            int32_t *tmp = (int32_t *)Temp->dptr_;
            uint32_t shift1 = shift > 0 ? shift : 0;
            uint32_t shift2 = shift > 0 ? 0 : -shift;
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, 1 << shift1, tmp, input_size, shift2), "luna_scale_i32i32o32");  // Scale to Int32
            src = tmp;
        }
        THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(src, dst, input_size), "luna_sigmoid_i32o8");  // Apply sigmoid activation and convert to Int8
    }
    else
        return T_ERR_INVALID_DATATYPE;

    return T_SUCCESS;
}

#endif  // _SIGMOID_LUNA_H_