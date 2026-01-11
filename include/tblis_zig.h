#pragma once

#include "tci.h"
// Try to read config, otherwise choose defaults.
#ifdef _TBLIS_CONFIG_H_
    typedef TBLIS_LABEL_TYPE zig_label_type;
    typedef TBLIS_LEN_TYPE zig_len_type;
    typedef TBLIS_STRIDE_TYPE zig_stride_type;
#else
    #warning "tblis_config.h not found, using defaults."
    typedef char zig_label_type;
    typedef int zig_len_type;
    typedef int zig_stride_type;
#endif


/*
 * Unchanged types
 */
typedef void tblis_zig_config;
typedef enum
{
    ZIG_REDUCE_SUM      = 0,
    ZIG_REDUCE_SUM_ABS  = 1,
    ZIG_REDUCE_MAX      = 2,
    ZIG_REDUCE_MAX_ABS  = 3,
    ZIG_REDUCE_MIN      = 4,
    ZIG_REDUCE_MIN_ABS  = 5,
    ZIG_REDUCE_NORM_1   = ZIG_REDUCE_SUM_ABS,
    ZIG_REDUCE_NORM_2   = 6,
    ZIG_REDUCE_NORM_INF = ZIG_REDUCE_MAX_ABS
} zig_reduce_t;

typedef int zig_type_t;
static const zig_type_t ZIG_TYPE_SINGLE   = 0;
static const zig_type_t ZIG_TYPE_FLOAT    = ZIG_TYPE_SINGLE;
static const zig_type_t ZIG_TYPE_DOUBLE   = 2;
static const zig_type_t ZIG_TYPE_SCOMPLEX = 1;
static const zig_type_t ZIG_TYPE_DCOMPLEX = 3;

/*
 * Structs to replace the unrepresentable C complex types.
 * Identical memory layout to Zig's Complex(f32) and Complex(f64)
 */
typedef struct {float re, im;} scomplex_zig;
typedef struct {double re, im;} dcomplex_zig;

/*
 * Structs adapted to the zig-friendly complex types.
 */
union zig_scalar
{
    float s;
    double d;
    scomplex_zig c;
    dcomplex_zig z;
};

typedef struct
{
    union zig_scalar data;
    zig_type_t type;
} tblis_zig_scalar;

typedef struct
{
    zig_type_t type;
    int conj;
    tblis_zig_scalar scalar;
    void* data;
    int ndim;
    const zig_len_type* len;
    const zig_stride_type* stride;
} tblis_zig_tensor;


/*
 * thread.h unchanged
 */
typedef tci_comm tblis_comm;
extern const tblis_comm* const tblis_single;
unsigned tblis_get_num_threads();
void tblis_set_num_threads(unsigned num_threads);

/*
 * add.h adapted
 */
/// `B_j <- alpha A_i + beta B_j`,
/// where the index sequences $i, j$ have the same elements, but can be in a different order.
/// Returns garbage when $i, j$ have different elements.
/// For instance, I checked that 'ab b' as well as 'a b' return garbage.
void tblis_zig_tensor_add(const tblis_comm* comm,
                    const tblis_zig_config* cntx,
                    const tblis_zig_tensor* A,
                    const zig_label_type* idx_A,
                            tblis_zig_tensor* B,
                    const zig_label_type* idx_B);

/*
 * dot.h adapted
 */
void tblis_zig_tensor_dot(const tblis_comm* comm,
                      const tblis_zig_config* cntx,
                      const tblis_zig_tensor* A,
                      const zig_label_type* idx_A,
                      const tblis_zig_tensor* B,
                      const zig_label_type* idx_B,
                      tblis_zig_scalar* result);

/*
 * reduce.h adapted
 */
 void tblis_zig_tensor_reduce(const tblis_comm* comm,
                          const tblis_zig_config* cntx,
                          zig_reduce_t op,
                          const tblis_zig_tensor* A,
                          const zig_label_type* idx_A,
                          tblis_zig_scalar* result,
                          zig_len_type* idx);

/*
 * scale.h adapted
 */
 void tblis_zig_tensor_scale(const tblis_comm* comm,
                         const tblis_zig_config* cntx,
                               tblis_zig_tensor* A,
                         const zig_label_type* idx_A);

/*
 * set.h adapted
 */
 void tblis_zig_tensor_set(const tblis_comm* comm,
                       const tblis_zig_config* cntx,
                       const tblis_zig_scalar* alpha,
                             tblis_zig_tensor* A,
                       const zig_label_type* idx_A);

/*
 * shift.h adapted
 */
 void tblis_zig_tensor_shift(const tblis_comm* comm,
                         const tblis_zig_config* cntx,
                         const tblis_zig_scalar* alpha,
                               tblis_zig_tensor* A,
                         const zig_label_type* idx_A);

/*
 * mult.h adapted
 */
 void tblis_zig_tensor_mult(const tblis_comm* comm, const tblis_zig_config* cntx,
                        const tblis_zig_tensor* A, const zig_label_type* idx_A,
                        const tblis_zig_tensor* B, const zig_label_type* idx_B,
                              tblis_zig_tensor* C, const zig_label_type* idx_C);
