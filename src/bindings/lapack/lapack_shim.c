/* Header-verified forwarders to Accelerate's complex LAPACK routines.
 *
 * Each forwarder receives primitive pointers from Zig and forwards to the real
 * `c*_`/`z*_` entry point, casting the interleaved-float buffers to the C
 * `_Complex` type. Because this translation unit `#include`s <vecLib/lapack.h>,
 * the C compiler checks every forwarding call against Apple's own prototypes:
 * a wrong argument list (a dropped `rwork`, a mis-split complex `w`, a
 * confused real/complex output) fails the build here rather than corrupting
 * memory at runtime. Including "lapack_shim.h" additionally checks each
 * forwarder's definition against its declared (primitive) prototype.
 *
 * See lapack_shim.h for the rationale and naming conventions.
 */
#define ACCELERATE_NEW_LAPACK 1
#include <vecLib/lapack.h>

#include "lapack_shim.h"

typedef __LAPACK_float_complex cf;
typedef __LAPACK_double_complex zf;

/* ---- LU family ---- */
void zarray_cgetrf(const int *m, const int *n, float *a, const int *lda,
                   int *ipiv, int *info) {
    cgetrf_(m, n, (cf *)a, lda, ipiv, info);
}
void zarray_zgetrf(const int *m, const int *n, double *a, const int *lda,
                   int *ipiv, int *info) {
    zgetrf_(m, n, (zf *)a, lda, ipiv, info);
}

void zarray_cgetrs(const char *trans, const int *n, const int *nrhs, float *a,
                   const int *lda, const int *ipiv, float *b, const int *ldb,
                   int *info) {
    cgetrs_(trans, n, nrhs, (const cf *)a, lda, ipiv, (cf *)b, ldb, info);
}
void zarray_zgetrs(const char *trans, const int *n, const int *nrhs, double *a,
                   const int *lda, const int *ipiv, double *b, const int *ldb,
                   int *info) {
    zgetrs_(trans, n, nrhs, (const zf *)a, lda, ipiv, (zf *)b, ldb, info);
}

void zarray_cgetri(const int *n, float *a, const int *lda, const int *ipiv,
                   float *work, const int *lwork, int *info) {
    cgetri_(n, (cf *)a, lda, ipiv, (cf *)work, lwork, info);
}
void zarray_zgetri(const int *n, double *a, const int *lda, const int *ipiv,
                   double *work, const int *lwork, int *info) {
    zgetri_(n, (zf *)a, lda, ipiv, (zf *)work, lwork, info);
}

/* ---- Cholesky family ---- */
void zarray_cpotrf(const char *uplo, const int *n, float *a, const int *lda,
                   int *info) {
    cpotrf_(uplo, n, (cf *)a, lda, info);
}
void zarray_zpotrf(const char *uplo, const int *n, double *a, const int *lda,
                   int *info) {
    zpotrf_(uplo, n, (zf *)a, lda, info);
}

void zarray_cpotrs(const char *uplo, const int *n, const int *nrhs, float *a,
                   const int *lda, float *b, const int *ldb, int *info) {
    cpotrs_(uplo, n, nrhs, (const cf *)a, lda, (cf *)b, ldb, info);
}
void zarray_zpotrs(const char *uplo, const int *n, const int *nrhs, double *a,
                   const int *lda, double *b, const int *ldb, int *info) {
    zpotrs_(uplo, n, nrhs, (const zf *)a, lda, (zf *)b, ldb, info);
}

/* ---- Hermitian eigenvalues (heev): w/rwork stay real ---- */
void zarray_cheev(const char *jobz, const char *uplo, const int *n, float *a,
                  const int *lda, float *w, float *work, const int *lwork,
                  float *rwork, int *info) {
    cheev_(jobz, uplo, n, (cf *)a, lda, w, (cf *)work, lwork, rwork, info);
}
void zarray_zheev(const char *jobz, const char *uplo, const int *n, double *a,
                  const int *lda, double *w, double *work, const int *lwork,
                  double *rwork, int *info) {
    zheev_(jobz, uplo, n, (zf *)a, lda, w, (zf *)work, lwork, rwork, info);
}

/* ---- General eigenvalues (geev): single complex w, real rwork ---- */
void zarray_cgeev(const char *jobvl, const char *jobvr, const int *n, float *a,
                  const int *lda, float *w, float *vl, const int *ldvl,
                  float *vr, const int *ldvr, float *work, const int *lwork,
                  float *rwork, int *info) {
    cgeev_(jobvl, jobvr, n, (cf *)a, lda, (cf *)w, (cf *)vl, ldvl, (cf *)vr,
           ldvr, (cf *)work, lwork, rwork, info);
}
void zarray_zgeev(const char *jobvl, const char *jobvr, const int *n, double *a,
                  const int *lda, double *w, double *vl, const int *ldvl,
                  double *vr, const int *ldvr, double *work, const int *lwork,
                  double *rwork, int *info) {
    zgeev_(jobvl, jobvr, n, (zf *)a, lda, (zf *)w, (zf *)vl, ldvl, (zf *)vr,
           ldvr, (zf *)work, lwork, rwork, info);
}

/* ---- SVD (gesdd): s/rwork stay real, u/vt complex ---- */
void zarray_cgesdd(const char *jobz, const int *m, const int *n, float *a,
                   const int *lda, float *s, float *u, const int *ldu,
                   float *vt, const int *ldvt, float *work, const int *lwork,
                   float *rwork, int *iwork, int *info) {
    cgesdd_(jobz, m, n, (cf *)a, lda, s, (cf *)u, ldu, (cf *)vt, ldvt,
            (cf *)work, lwork, rwork, iwork, info);
}
void zarray_zgesdd(const char *jobz, const int *m, const int *n, double *a,
                   const int *lda, double *s, double *u, const int *ldu,
                   double *vt, const int *ldvt, double *work, const int *lwork,
                   double *rwork, int *iwork, int *info) {
    zgesdd_(jobz, m, n, (zf *)a, lda, s, (zf *)u, ldu, (zf *)vt, ldvt,
            (zf *)work, lwork, rwork, iwork, info);
}

/* ---- Least squares (gels) ---- */
void zarray_cgels(const char *trans, const int *m, const int *n,
                  const int *nrhs, float *a, const int *lda, float *b,
                  const int *ldb, float *work, const int *lwork, int *info) {
    cgels_(trans, m, n, nrhs, (cf *)a, lda, (cf *)b, ldb, (cf *)work, lwork,
           info);
}
void zarray_zgels(const char *trans, const int *m, const int *n,
                  const int *nrhs, double *a, const int *lda, double *b,
                  const int *ldb, double *work, const int *lwork, int *info) {
    zgels_(trans, m, n, nrhs, (zf *)a, lda, (zf *)b, ldb, (zf *)work, lwork,
           info);
}

/* ---- QR (geqrf) + unitary Q (ungqr) ---- */
void zarray_cgeqrf(const int *m, const int *n, float *a, const int *lda,
                   float *tau, float *work, const int *lwork, int *info) {
    cgeqrf_(m, n, (cf *)a, lda, (cf *)tau, (cf *)work, lwork, info);
}
void zarray_zgeqrf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info) {
    zgeqrf_(m, n, (zf *)a, lda, (zf *)tau, (zf *)work, lwork, info);
}

void zarray_cungqr(const int *m, const int *n, const int *k, float *a,
                   const int *lda, float *tau, float *work, const int *lwork,
                   int *info) {
    cungqr_(m, n, k, (cf *)a, lda, (const cf *)tau, (cf *)work, lwork, info);
}
void zarray_zungqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, double *tau, double *work, const int *lwork,
                   int *info) {
    zungqr_(m, n, k, (zf *)a, lda, (const zf *)tau, (zf *)work, lwork, info);
}

/* ---- Generalized Hermitian-definite eigenproblem (hegv): w/rwork real ---- */
void zarray_chegv(const int *itype, const char *jobz, const char *uplo,
                  const int *n, float *a, const int *lda, float *b,
                  const int *ldb, float *w, float *work, const int *lwork,
                  float *rwork, int *info) {
    chegv_(itype, jobz, uplo, n, (cf *)a, lda, (cf *)b, ldb, w, (cf *)work,
           lwork, rwork, info);
}
void zarray_zhegv(const int *itype, const char *jobz, const char *uplo,
                  const int *n, double *a, const int *lda, double *b,
                  const int *ldb, double *w, double *work, const int *lwork,
                  double *rwork, int *info) {
    zhegv_(itype, jobz, uplo, n, (zf *)a, lda, (zf *)b, ldb, w, (zf *)work,
           lwork, rwork, info);
}

/* ---- Rank-deficient least squares via SVD (gelsd): s/rcond/rwork real ---- */
void zarray_cgelsd(const int *m, const int *n, const int *nrhs, float *a,
                   const int *lda, float *b, const int *ldb, float *s,
                   const float *rcond, int *rank, float *work,
                   const int *lwork, float *rwork, int *iwork, int *info) {
    cgelsd_(m, n, nrhs, (cf *)a, lda, (cf *)b, ldb, s, rcond, rank, (cf *)work,
            lwork, rwork, iwork, info);
}
void zarray_zgelsd(const int *m, const int *n, const int *nrhs, double *a,
                   const int *lda, double *b, const int *ldb, double *s,
                   const double *rcond, int *rank, double *work,
                   const int *lwork, double *rwork, int *iwork, int *info) {
    zgelsd_(m, n, nrhs, (zf *)a, lda, (zf *)b, ldb, s, rcond, rank, (zf *)work,
            lwork, rwork, iwork, info);
}
