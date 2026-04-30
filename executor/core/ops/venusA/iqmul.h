#ifndef _MUL_LUNA_H_
#define _MUL_LUNA_H_

#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Performs vector multiplication with broadcast for specific tensor shapes
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param shift Quantization shift
 * @return int32_t Operation status
 */
int32_t calc_vec_mul_luna_b2b2_broadcast_h1w1(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, int32_t shift) {
    int32_t c = lhs->shape_.dims_[1];
    int32_t h = lhs->shape_.dims_[2];
    int32_t w = lhs->shape_.dims_[3];
    int8_t *p_tmp1 = (int8_t *)Temp->dptr_;
    int8_t *p_tmp2 = p_tmp1 + c;

    THINKER_RET_CHECK(API_LIB(memset_i8o8)(p_tmp1, 1, h * w), "luna_memset_i8o8");
    THINKER_RET_CHECK(API_LIB(mat_mul_i8i8o8)((int8_t *)rhs->dptr_, p_tmp1, p_tmp2, c, 1, h * w, 0), "luna_mul_i8i8o8");
    THINKER_RET_CHECK(API_LIB(mul_i8i8o8)((int8_t *)lhs->dptr_, p_tmp2, (int8_t *)Y->dptr_, c * h * w, shift), "luna_mul_i8i8o8");

    return T_SUCCESS;
}

/**
 * @brief Quantized multiplication operation implementation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param attrs Operation attributes
 * @return int32_t Operation status
 */
int32_t iqmul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, iqBinaryAttrs *attrs) {
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = x1_q + x2_q - y_q;
    size_t size = getTensorSize(lhs);

    if (shift < 0) {
        return T_ERR_INVALID_PARA;
    }

    if ((lhs->dtype_ != rhs->dtype_) || (lhs->dtype_ != Y->dtype_))
        return T_ERR_INVALID_DATATYPE;

    if (lhs->shape_.ndim_ == 4 && rhs->shape_.ndim_ == 4 &&
        lhs->shape_.dims_[1] == rhs->shape_.dims_[1] &&
        rhs->shape_.dims_[2] == 1 && rhs->shape_.dims_[3] == 1) {
        if (lhs->dtype_ == Int8)
            THINKER_RET_CHECK(calc_vec_mul_luna_b2b2_broadcast_h1w1(lhs, rhs, Y, Temp, shift), "calc_vec_mul_luna_b2b2_broadcast_h1w1");
        else
            return T_ERR_INVALID_DATATYPE;
    } 
    else if (rhs->shape_.ndim_ == 0) {
        if (rhs->dtype_ == Int8) {
            int8_t scalar = *(int8_t *)rhs->dptr_;
            THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)lhs->dptr_, scalar, (int8_t *)Y->dptr_, size, shift), "luna_scalar_i8i8o8");
        } 
        else if (rhs->dtype_ == Int16) {
            int16_t scalar = *(int16_t *)rhs->dptr_;
            THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)lhs->dptr_, scalar, (int16_t *)Y->dptr_, size, shift), "luna_scalar_i16i16o16");
        } 
        else if (rhs->dtype_ == Float32) {
            int32_t scalar = (int32_t)*(float *)rhs->dptr_;
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)lhs->dptr_, scalar, (int32_t *)Y->dptr_, size, shift), "luna_scalar_i32i32o32");
        } 
        else {
            return T_ERR_INVALID_DATATYPE;
        }
    } 
    else {
        if (rhs->dtype_ == Int8) {
            THINKER_RET_CHECK(API_LIB(mul_i8i8o8)((int8_t *)lhs->dptr_, (int8_t *)rhs->dptr_, (int8_t *)Y->dptr_, size, shift), "luna_mul_i8i8o8");
        } 
        else if (rhs->dtype_ == Int16) {
            THINKER_RET_CHECK(API_LIB(mul_i16i16o16)((int16_t *)lhs->dptr_, (int16_t *)rhs->dptr_, (int16_t *)Y->dptr_, size, shift), "luna_mul_i16i16o16");
        } 
        else if (rhs->dtype_ == Float32) {
            THINKER_RET_CHECK(API_LIB(mul_i32i32o32)((int32_t *)lhs->dptr_, (int32_t *)rhs->dptr_, (int32_t *)Y->dptr_, size, shift), "luna_mul_i32i32o32");
        } 
        else {
            return T_ERR_INVALID_DATATYPE;
        }
    }

    return T_SUCCESS;
}

#endif  // _MUL_LUNA_H_