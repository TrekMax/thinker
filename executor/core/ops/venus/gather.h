#ifndef _GATHER_LUNA_H_
#define _GATHER_LUNA_H_

#include <math.h>
#include "core/operator_attrs.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Perform gather operation on input tensor based on indices
 * @param X Pointer to input tensor
 * @param indices Pointer to indices tensor
 * @param Y Pointer to output tensor
 * @param attr Pointer to GatherAttrs containing gather attributes
 * @return int32_t Return status (T_SUCCESS if successful)
 */
int32_t gather_luna(tTensor *X, tTensor *indices, tTensor *Y, GatherAttrs *attr) {
    int32_t axis = attr->axis;
    axis = (axis < 0) ? (X->shape_.ndim_ + axis) : axis;

    // Calculate the total number of elements in indices
    int32_t ndim = 1;
    for (uint32_t i = 0; i < indices->shape_.ndim_; ++i) {
        ndim *= indices->shape_.dims_[i];
    }
    if (ndim == 0) {
        ndim = 1;
    }

    // Calculate tensor dimensions
    int32_t leading = 1;
    uint32_t dim_index = 0;
    for (; dim_index < axis; ++dim_index) {
        leading *= X->shape_.dims_[dim_index];
    }
    int32_t middle = X->shape_.dims_[dim_index++];
    int32_t tail = 1;
    for (; dim_index < X->shape_.ndim_; ++dim_index) {
        tail *= X->shape_.dims_[dim_index];
    }

    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;

    if (indices->dtype_ == Int64) {
        int64_t *index = (int64_t *)indices->dptr_;
        for (int32_t l = 0; l < leading; ++l) {
            for (int32_t j = 0; j < ndim; ++j) {
                int32_t idx = index[j];
                if (idx == -1) {
                    idx = middle - 1;
                }
                if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
                    memcpy(output + (l * ndim * tail + j * tail) * X->byte_,
                        input + (l * middle * tail + idx * tail) * X->byte_,
                        X->byte_ * tail);
                } else {
                    THINKER_RET_CHECK(API_LIB(memcpy)(output + (l * ndim * tail + j * tail) * X->byte_,
                                input + (l * middle * tail + idx * tail) * X->byte_,
                                X->byte_ * tail), "luna_memcpy");
                }
            }
        }
    }
    else if (indices->dtype_ == Int32) {
        int32_t *index = (int32_t *)indices->dptr_;
        for (int32_t l = 0; l < leading; ++l)
            for (int32_t j = 0; j < ndim; ++j)
            {
                int32_t idx = index[j] == -1 ? X->shape_.dims_[axis] - 1 : index[j];
                if (Y->mem_.type_ == 2 && X->mem_.type_ != 2)
                    THINKER_RET_CHECK(API_LIB(memcpy)(output + (l * ndim * tail + j * tail) * X->byte_,
                                              input + (l * middle * tail + idx * tail) * X->byte_,
                                              X->byte_ * tail), "luna_memcpy_i8o8");
                else {
                    memcpy(output + (l * ndim * tail + j * tail) * X->byte_,
                                     input + (l * middle * tail + idx * tail) * X->byte_,
                                     X->byte_ * tail);
                }
            }
    }
    else
        return T_ERR_INVALID_DATATYPE;

    return T_SUCCESS;
}

#endif  // _GATHER_LUNA_H_