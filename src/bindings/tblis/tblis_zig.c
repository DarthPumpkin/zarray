#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#include "tblis_zig.h"

#include "tblis/frame/base/basic_types.h"
#include "tblis/frame/1t/add.h"
#include "tblis/frame/1t/dot.h"
#include "tblis/frame/1t/reduce.h"
#include "tblis/frame/1t/scale.h"
#include "tblis/frame/1t/set.h"
#include "tblis/frame/1t/shift.h"
#include "tblis/frame/3t/mult.h"


/*
 * Internal helpers.
 */

static int dim_for_alloc(int dim)
{
    return dim > 0 ? dim : 1;
}

static void convert_len_to_tblis(const zig_len_type* zig_len, len_type* len_conv, int dim)
{
    for (int i = 0; i < dim; i++) {
        len_conv[i] = (len_type)zig_len[i];
    }
}

static void convert_stride_to_tblis(const zig_stride_type* zig_stride, stride_type* stride_conv, int dim)
{
    for (int i = 0; i < dim; i++) {
        stride_conv[i] = (stride_type)zig_stride[i];
    }
}

static void convert_len_to_zig(const len_type* len_conv, zig_len_type* zig_len, int dim)
{
    for (int i = 0; i < dim; i++) {
        zig_len[i] = (zig_len_type)len_conv[i];
    }
}

tblis_scalar convert_scalar(const tblis_zig_scalar scalar) {
    tblis_scalar scalar_internal;
    switch (scalar.type) {
        case TYPE_SINGLE:
            tblis_init_scalar_s(&scalar_internal, scalar.data.s);
            break;
        case TYPE_DOUBLE:
            tblis_init_scalar_d(&scalar_internal, scalar.data.d);
            break;
        case TYPE_SCOMPLEX: {
            scomplex_zig c = scalar.data.c;
            tblis_init_scalar_c(&scalar_internal, CMPLX(c.re, c.im));
            break;
        }
        case TYPE_DCOMPLEX: {
            dcomplex_zig z = scalar.data.z;
            tblis_init_scalar_z(&scalar_internal, CMPLX(z.re, z.im));
            break;
        }
        default:
            printf("Unrecognized scalar type: %d", scalar.type);
            exit(-1);
    }
    return scalar_internal;
}


void update_zig_scalar(tblis_zig_scalar *scalar, const tblis_scalar *new_val) {
    switch (scalar->type) {
    case ZIG_TYPE_SINGLE: {
        scalar->data.s = new_val->data.s;
        break;
    }
    case ZIG_TYPE_DOUBLE: {
        scalar->data.d = new_val->data.d;
        break;
    }
    case ZIG_TYPE_SCOMPLEX: {
        float re = crealf(new_val->data.c);
        float im = cimagf(new_val->data.c);
        scalar->data.c.re = re;
        scalar->data.c.im = im;
        break;
    }
    case ZIG_TYPE_DCOMPLEX: {
        double re = creal(new_val->data.z);
        double im = cimag(new_val->data.z);
        scalar->data.z.re = re;
        scalar->data.z.im = im;
        break;
    }
    default:
        printf("Unrecognized scalar type: %d", scalar->type);
        exit(-1);
    }
}


tblis_tensor convert_tensor(const tblis_zig_tensor *tensor, len_type *len_conv, stride_type *stride_conv) {
    int dim = tensor->ndim;
    convert_len_to_tblis(tensor->len, len_conv, dim);
    convert_stride_to_tblis(tensor->stride, stride_conv, dim);

    tblis_scalar scalar_conv = convert_scalar(tensor->scalar);
    tblis_tensor tensor_conv = {
        .type = tensor->type,
        .conj = tensor->conj,
        .scalar = scalar_conv,
        .data = tensor->data,
        .ndim = dim,
        .len = len_conv,
        .stride = stride_conv
    };
    return tensor_conv;
}


/*
 * Here is where the actual API starts
 */

void tblis_zig_tensor_add(
    const tblis_comm* comm,
    const tblis_zig_config* cntx,
    const tblis_zig_tensor* A,
    const zig_label_type* idx_A,
          tblis_zig_tensor* B,
    const zig_label_type* idx_B
) {
    int dimA = A->ndim;
    int dimB = B->ndim;

    len_type A_len[dim_for_alloc(dimA)];
    stride_type A_stride[dim_for_alloc(dimA)];
    len_type B_len[dim_for_alloc(dimB)];
    stride_type B_stride[dim_for_alloc(dimB)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);
    tblis_tensor B_conv = convert_tensor(B, B_len, B_stride);
    tblis_tensor_add(comm, cntx, &A_conv, idx_A, &B_conv, idx_B);
    update_zig_scalar(&B->scalar, &B_conv.scalar);
}

void tblis_zig_tensor_dot(
    const tblis_comm* comm,
    const tblis_zig_config* cntx,
    const tblis_zig_tensor* A,
    const zig_label_type* idx_A,
    const tblis_zig_tensor* B,
    const zig_label_type* idx_B,
    tblis_zig_scalar* result
) {
    int rank_A = A->ndim;
    int rank_B = B->ndim;

    len_type A_len[dim_for_alloc(rank_A)];
    stride_type A_stride[dim_for_alloc(rank_A)];
    len_type B_len[dim_for_alloc(rank_B)];
    stride_type B_stride[dim_for_alloc(rank_B)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);
    tblis_tensor B_conv = convert_tensor(B, B_len, B_stride);

    tblis_scalar result_conv;
    result_conv.type = A->type;

    tblis_tensor_dot(comm, cntx, &A_conv, idx_A, &B_conv, idx_B, &result_conv);

    update_zig_scalar(result, &result_conv);
}


void tblis_zig_tensor_reduce(
    const tblis_comm* comm,
    const tblis_zig_config* cntx,
    zig_reduce_t op,
    const tblis_zig_tensor* A, const zig_label_type* idx_A,
    tblis_zig_scalar* result, zig_len_type* result_idx
) {
    reduce_t op_conv = (reduce_t) op;

    int rank_A = A->ndim;

    len_type A_len[dim_for_alloc(rank_A)];
    stride_type A_stride[dim_for_alloc(rank_A)];
    len_type result_idx_conv[dim_for_alloc(rank_A)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);

    tblis_scalar result_conv;
    result_conv.type = result->type;

    tblis_tensor_reduce(
        comm, cntx,
        op_conv,
        &A_conv, idx_A,
        &result_conv, result_idx_conv
    );

    update_zig_scalar(result, &result_conv);
    convert_len_to_zig(result_idx_conv, result_idx, rank_A);
}


void tblis_zig_tensor_scale(
    const tblis_comm* comm, const tblis_zig_config* cntx,
    tblis_zig_tensor* A, const zig_label_type* idx_A
) {
    int rank_A = A->ndim;
    len_type A_len[dim_for_alloc(rank_A)];
    stride_type A_stride[dim_for_alloc(rank_A)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);

    tblis_tensor_scale(
        comm, cntx,
        &A_conv, idx_A
    );

    update_zig_scalar(&A->scalar, &A_conv.scalar);
}

void tblis_zig_tensor_set(
    const tblis_comm* comm,
    const tblis_zig_config* cntx,
    const tblis_zig_scalar* alpha,
    tblis_zig_tensor* A,
    const zig_label_type* idx_A
) {
    tblis_scalar alpha_conv = convert_scalar(*alpha);

    int rank_A = A->ndim;
    len_type A_len[dim_for_alloc(rank_A)];
    stride_type A_stride[dim_for_alloc(rank_A)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);

    tblis_tensor_set(
        comm, cntx,
        &alpha_conv,
        &A_conv, idx_A
    );
    update_zig_scalar(&A->scalar, &A_conv.scalar);
}

void tblis_zig_tensor_shift(
    const tblis_comm* comm, const tblis_zig_config* cntx,
    const tblis_zig_scalar* alpha,
    tblis_zig_tensor* A, const zig_label_type* idx_A
) {
    tblis_scalar alpha_conv = convert_scalar(*alpha);

    int rank_A = A->ndim;
    len_type A_len[dim_for_alloc(rank_A)];
    stride_type A_stride[dim_for_alloc(rank_A)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);

    tblis_tensor_shift(
        comm, cntx,
        &alpha_conv,
        &A_conv, idx_A
    );
    update_zig_scalar(&A->scalar, &A_conv.scalar);
}

void tblis_zig_tensor_mult(
    const tblis_comm* comm, const tblis_zig_config* cntx,
    const tblis_zig_tensor* A, const zig_label_type* idx_A,
    const tblis_zig_tensor* B, const zig_label_type* idx_B,
    tblis_zig_tensor* C, const zig_label_type* idx_C
) {
    int rank_A = A->ndim;
    int rank_B = B->ndim;
    int rank_C = C->ndim;

    len_type A_len[dim_for_alloc(rank_A)];
    stride_type A_stride[dim_for_alloc(rank_A)];
    len_type B_len[dim_for_alloc(rank_B)];
    stride_type B_stride[dim_for_alloc(rank_B)];
    len_type C_len[dim_for_alloc(rank_C)];
    stride_type C_stride[dim_for_alloc(rank_C)];

    tblis_tensor A_conv = convert_tensor(A, A_len, A_stride);
    tblis_tensor B_conv = convert_tensor(B, B_len, B_stride);
    tblis_tensor C_conv = convert_tensor(C, C_len, C_stride);

    tblis_tensor_mult(comm, cntx, &A_conv, idx_A, &B_conv, idx_B, &C_conv, idx_C);
}
