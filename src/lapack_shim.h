/* Primitive-typed forwarders to Accelerate's complex LAPACK entry points.
 *
 * translate-c cannot model C `double _Complex` / `float _Complex`, so the
 * complex LAPACK symbols are unreachable from Zig's @cImport. Rather than
 * hand-writing `extern fn` declarations that guess at the (structurally
 * divergent) complex argument lists — an easy way to silently corrupt memory —
 * these forwarders let the C compiler verify every prototype against
 * <vecLib/lapack.h> at build time (see lapack_shim.c). The Zig<->shim boundary
 * then speaks only primitive pointer types: a complex array is a pointer to
 * interleaved re/im floats (`float*` / `double*`), so there is no complex
 * ambiguity, and the risky ABI matching lives next to the header-checked call.
 *
 * Naming: `zarray_<lapackname>`. `c` = single complex, `z` = double complex.
 * Integer arguments are plain `int` (== Apple's `__LAPACK_int` in the default
 * LP64 build); a mismatch would fail the C compile — a feature.
 */
#ifndef ZARRAY_LAPACK_SHIM_H
#define ZARRAY_LAPACK_SHIM_H

/* ---- LU family (getrf / getrs / getri) ---- */
void zarray_cgetrf(const int *m, const int *n, float *a, const int *lda,
                   int *ipiv, int *info);
void zarray_zgetrf(const int *m, const int *n, double *a, const int *lda,
                   int *ipiv, int *info);

void zarray_cgetrs(const char *trans, const int *n, const int *nrhs, float *a,
                   const int *lda, const int *ipiv, float *b, const int *ldb,
                   int *info);
void zarray_zgetrs(const char *trans, const int *n, const int *nrhs, double *a,
                   const int *lda, const int *ipiv, double *b, const int *ldb,
                   int *info);

void zarray_cgetri(const int *n, float *a, const int *lda, const int *ipiv,
                   float *work, const int *lwork, int *info);
void zarray_zgetri(const int *n, double *a, const int *lda, const int *ipiv,
                   double *work, const int *lwork, int *info);

/* ---- Cholesky family (potrf / potrs) ---- */
void zarray_cpotrf(const char *uplo, const int *n, float *a, const int *lda,
                   int *info);
void zarray_zpotrf(const char *uplo, const int *n, double *a, const int *lda,
                   int *info);

void zarray_cpotrs(const char *uplo, const int *n, const int *nrhs, float *a,
                   const int *lda, float *b, const int *ldb, int *info);
void zarray_zpotrs(const char *uplo, const int *n, const int *nrhs, double *a,
                   const int *lda, double *b, const int *ldb, int *info);

/* ---- Hermitian eigenvalues (heev) — real w, complex work, real rwork ---- */
void zarray_cheev(const char *jobz, const char *uplo, const int *n, float *a,
                  const int *lda, float *w, float *work, const int *lwork,
                  float *rwork, int *info);
void zarray_zheev(const char *jobz, const char *uplo, const int *n, double *a,
                  const int *lda, double *w, double *work, const int *lwork,
                  double *rwork, int *info);

/* ---- General eigenvalues (geev) — single complex w, complex vl/vr, rwork ---- */
void zarray_cgeev(const char *jobvl, const char *jobvr, const int *n, float *a,
                  const int *lda, float *w, float *vl, const int *ldvl,
                  float *vr, const int *ldvr, float *work, const int *lwork,
                  float *rwork, int *info);
void zarray_zgeev(const char *jobvl, const char *jobvr, const int *n, double *a,
                  const int *lda, double *w, double *vl, const int *ldvl,
                  double *vr, const int *ldvr, double *work, const int *lwork,
                  double *rwork, int *info);

/* ---- SVD (gesdd) — real s, complex u/vt, complex work, real rwork, iwork ---- */
void zarray_cgesdd(const char *jobz, const int *m, const int *n, float *a,
                   const int *lda, float *s, float *u, const int *ldu,
                   float *vt, const int *ldvt, float *work, const int *lwork,
                   float *rwork, int *iwork, int *info);
void zarray_zgesdd(const char *jobz, const int *m, const int *n, double *a,
                   const int *lda, double *s, double *u, const int *ldu,
                   double *vt, const int *ldvt, double *work, const int *lwork,
                   double *rwork, int *iwork, int *info);

/* ---- Least squares (gels) — all complex, trans in {'N','C'} ---- */
void zarray_cgels(const char *trans, const int *m, const int *n,
                  const int *nrhs, float *a, const int *lda, float *b,
                  const int *ldb, float *work, const int *lwork, int *info);
void zarray_zgels(const char *trans, const int *m, const int *n,
                  const int *nrhs, double *a, const int *lda, double *b,
                  const int *ldb, double *work, const int *lwork, int *info);

/* ---- QR factor (geqrf) and unitary Q assembly (ungqr, complex analog of orgqr) ---- */
void zarray_cgeqrf(const int *m, const int *n, float *a, const int *lda,
                   float *tau, float *work, const int *lwork, int *info);
void zarray_zgeqrf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);

void zarray_cungqr(const int *m, const int *n, const int *k, float *a,
                   const int *lda, float *tau, float *work, const int *lwork,
                   int *info);
void zarray_zungqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, double *tau, double *work, const int *lwork,
                   int *info);

/* ---- Generalized Hermitian-definite eigenproblem (hegv) — real w/rwork ---- */
void zarray_chegv(const int *itype, const char *jobz, const char *uplo,
                  const int *n, float *a, const int *lda, float *b,
                  const int *ldb, float *w, float *work, const int *lwork,
                  float *rwork, int *info);
void zarray_zhegv(const int *itype, const char *jobz, const char *uplo,
                  const int *n, double *a, const int *lda, double *b,
                  const int *ldb, double *w, double *work, const int *lwork,
                  double *rwork, int *info);

/* ---- Rank-deficient least squares via SVD (gelsd) — real s/rcond/rwork ---- */
void zarray_cgelsd(const int *m, const int *n, const int *nrhs, float *a,
                   const int *lda, float *b, const int *ldb, float *s,
                   const float *rcond, int *rank, float *work,
                   const int *lwork, float *rwork, int *iwork, int *info);
void zarray_zgelsd(const int *m, const int *n, const int *nrhs, double *a,
                   const int *lda, double *b, const int *ldb, double *s,
                   const double *rcond, int *rank, double *work,
                   const int *lwork, double *rwork, int *iwork, int *info);

#endif /* ZARRAY_LAPACK_SHIM_H */
