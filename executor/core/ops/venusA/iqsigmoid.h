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
    uint32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;

#ifdef RUNTIME_PARAM_CHECK
    if (Y->dtype_ != Int8)
        return T_ERR_INVALID_DATATYPE;
#endif
    // Quantization parameters
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 7;
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

        // Determine memory types
    bool y_in_psram = (Y->mem_.type_ != 2);
    bool x_in_psram = (X->mem_.type_ != 2);

    // Quantization shift
    int32_t shift = Q_INPUT - x_q;

    // Perform quantized sigmoid computation
    if (X->dtype_ == Int8) {
        uint32_t past_size = 0;
        uint32_t chunk_size = workspace_size / 6;  // Max chunk size
        while ((ALIGN4(chunk_size*2) + chunk_size*4) > workspace_size)
            chunk_size -= 1;
        while (past_size < input_size) {
            uint32_t remain_size = input_size - past_size;
            uint32_t cur_size = chunk_size < remain_size ? chunk_size : remain_size;

            int8_t *src_chunk = (int8_t *)X->dptr_ + past_size;
            int16_t *src_chunk_int16 = (int16_t *)Temp->dptr_;
            int32_t *src_chunk_int32 = (int32_t *)((int8_t *)Temp->dptr_ + ALIGN4(cur_size * 2));
            int8_t *dst_temp = y_in_psram ? (int8_t *)Temp->dptr_ : (int8_t *)Y->dptr_ + past_size;

            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(src_chunk, 1, src_chunk_int16, cur_size, 0), "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src_chunk_int16, 1, src_chunk_int32, cur_size, 0), "luna_scale_i16i16o32");
            if (shift != 0) {
                uint32_t shift1 = shift > 0 ? (uint32_t)shift : 0U;
                uint32_t shift2 = shift > 0 ? 0U : (uint32_t)(-shift);
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src_chunk_int32, 1UL << shift1, src_chunk_int32, cur_size, shift2), "luna_scale_i32i32o32");
            }
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(src_chunk_int32, dst_temp, cur_size), "luna_sigmoid_i32o8");
            if (y_in_psram)
                opi_psram_cpy_out((void *)(Y->dptr_ + past_size), dst_temp, cur_size * sizeof(int8_t));
            past_size += cur_size;
        }
    }
    else if (X->dtype_ == Int16) {
        uint32_t past_size = 0;
        uint32_t chunk_size = workspace_size / 4;  // Max chunk size
        while (past_size < input_size) {
            uint32_t remain_size = input_size - past_size;
            uint32_t cur_size = chunk_size < remain_size ? chunk_size : remain_size;

            int16_t *src_chunk = (int16_t *)X->dptr_ + past_size;
            int32_t *src_chunk_int32 = (int32_t *)Temp->dptr_;
            int8_t *dst_temp = y_in_psram ? (int8_t *)Temp->dptr_ : (int8_t *)Y->dptr_ + past_size;

            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src_chunk, 1, src_chunk_int32, cur_size, 0), "luna_scale_i16i16o32");
            if (shift != 0) {
                uint32_t shift1 = shift > 0 ? (uint32_t)shift : 0U;
                uint32_t shift2 = shift > 0 ? 0U : (uint32_t)(-shift);
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src_chunk_int32, 1UL << shift1, src_chunk_int32, cur_size, shift2), "luna_scale_i32i32o32");
            }
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(src_chunk_int32, dst_temp, cur_size), "luna_sigmoid_i32o8");
            if (y_in_psram)
                opi_psram_cpy_out((void *)(Y->dptr_ + past_size), dst_temp, cur_size * sizeof(int8_t));
            past_size += cur_size;
        }
    }
    else if (X->dtype_ == Int32) {
        if (!y_in_psram && shift == 0) {
            THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)((int32_t *)X->dptr_, (int8_t *)Y->dptr_, input_size), "luna_sigmoid_i32o8");
        }
        else if (shift == 0) {
            uint32_t past_size = 0;
            uint32_t chunk_size = workspace_size;  // Max chunk size
            while (past_size < input_size) {
                uint32_t remain_size = input_size - past_size;
                uint32_t cur_size = chunk_size < remain_size ? chunk_size : remain_size;

                int32_t *src_chunk = (int32_t *)X->dptr_ + past_size;
                int8_t *dst_temp = (int8_t *)Temp->dptr_;
                THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(src_chunk, dst_temp, cur_size), "luna_sigmoid_i32o8");
                opi_psram_cpy_out((void *)(Y->dptr_ + past_size), dst_temp, cur_size * sizeof(int8_t));
                past_size += cur_size;
            }
        }        
        else {
            uint32_t past_size = 0;
            uint32_t chunk_size = workspace_size / 4;  // Max chunk size
            while (past_size < input_size) {
                uint32_t remain_size = input_size - past_size;
                uint32_t cur_size = chunk_size < remain_size ? chunk_size : remain_size;

                int32_t *src_chunk = (int32_t *)X->dptr_ + past_size;
                int32_t *tmp_chunk = (int32_t *)Temp->dptr_;
                int8_t *dst_temp = y_in_psram ? (int8_t *)Temp->dptr_ : (int8_t *)Y->dptr_ + past_size;
                uint32_t shift1 = shift > 0 ? shift : 0;
                uint32_t shift2 = shift > 0 ? 0 : -shift;
                THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src_chunk, 1UL << shift1, tmp_chunk, cur_size, shift2), "luna_scale_i32i32o32");
                THINKER_RET_CHECK(API_LIB(sigmoid_i32o8)(tmp_chunk, dst_temp, cur_size), "luna_sigmoid_i32o8");
                if (y_in_psram)
                    opi_psram_cpy_out((void *)(Y->dptr_ + past_size), dst_temp, cur_size * sizeof(int8_t));
                past_size += cur_size;
            }
        }
    }
    else
        return T_ERR_INVALID_DATATYPE;

    return T_SUCCESS;
}

#endif  // _SIGMOID_LUNA_H_
