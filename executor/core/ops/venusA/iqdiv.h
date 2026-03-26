#ifndef _DIV_LUNA_H_
#define _DIV_LUNA_H_

#include "c_api/thinker_define.h"
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
 * @brief Quantized division operation implementation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) {
    if (lhs->dtype_ != Int32 || rhs->dtype_ != Int32 || Y->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }

    if (Y->mem_.mem_type_ != 2)
        return T_ERR_NO_IMPLEMENTED;

    size_t size = getTensorSize(lhs);

    // Calculate quantization shift
    int32_t lhs_scale = (int32_t)lhs->scale_;
    int32_t rhs_scale = (int32_t)rhs->scale_;
    int32_t output_scale = (int32_t)Y->scale_;
    int32_t shift = output_scale - (lhs_scale - rhs_scale);

    // Check if right-hand side is a scalar
    if (rhs->shape_.ndim_ == 0) {
        int32_t scalar = (int32_t)(*(int32_t *)rhs->dptr_);
        return luna_div_scalar_i32i32o32((const int32_t *)lhs->dptr_, scalar, (int32_t *)Y->dptr_, size, shift);
    } 
    else {
        return API_LIB(div_i32i32o32)((const int32_t *)lhs->dptr_, (const int32_t *)rhs->dptr_, (int32_t *)Y->dptr_, size, shift);
    }

    return T_SUCCESS;
}

#endif  // _DIV_LUNA_H_