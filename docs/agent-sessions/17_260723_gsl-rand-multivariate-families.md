# GSL — Session 17 (`rand` multivariate families: multivariate Gaussian + Wishart)

**Status:** **Complete.** Finishes the `rand` module by binding its two remaining
deferred families — the multivariate Gaussian and Wishart distributions — plus a
`choleskyLower` helper. Both were blocked on matrix/vector bindings (session-8
deferral); the session-16 `gsl_matrix` bridge unblocked them, and this is that
bridge's first external consumer.

**Tests (latest run):**
- `zig test src/gsl_rand.zig -lgsl -lblas -llapack -I /opt/homebrew/include -L /opt/homebrew/lib -framework Accelerate`
  → **All 211 tests pass**.
- `zig build test --summary all` → **577/577 pass** (was 572; +5).
- `diagnostics` on `src/gsl_rand.zig` → clean.

---

## 1. What shipped (`src/gsl_rand.zig`, `gsl.rand`)

- **`MultivariateGaussian`** `{ mu: []const f64, l: gsl.Matrix(f64) }`:
  `sample(r, out)`, `pdf(x, work)`, `logPdf(x, work)`, plus the static
  estimators `mean(X, mu_hat)` and `vcov(X, sigma_hat)` over an `n×k` data
  matrix. Covariance is supplied as its **lower Cholesky factor** `L`
  (`Σ = L Lᵀ`); only the lower triangle is read.
- **`Wishart`** `{ df: f64, l: gsl.Matrix(f64) }`: `sample(r, out, work)`,
  `pdf(X, L_X, work)`, `logPdf(X, L_X, work)` over symmetric PD `k×k` matrices.
- **`choleskyLower(a: gsl.MatrixMut(f64))`** — in-place lower Cholesky via
  `gsl_linalg_cholesky_decomp1`, so callers can turn a covariance/scale matrix
  into the `L` these families want without leaving the module. Returns
  `error.NotPositiveDefinite` on a non-PD input.
- Added `@cInclude("gsl/gsl_linalg.h")` to the module's `@cImport` (for the
  Cholesky helper).
- Removed the "deferred until matrix bindings exist" note from the module
  docstring's Omissions.

## 2. Design decisions (as ratified)
- **`L`, not `Σ`.** The core APIs take the lower Cholesky factor directly
  (faithful to GSL), *plus* the `choleskyLower` convenience so the module is
  self-contained (decision (a)+(b) from scoping). `gsl_linalg_cholesky_decomp1`
  also writes `Lᵀ` into the upper triangle; the families read only the lower
  triangle, so that is harmless.
- **Caller-provided `work` buffers.** `pdf`/`logPdf` (length-`k` vector) and
  Wishart (`k×k`) take scratch from the caller, keeping the module
  allocation-free at call sites, consistent with the rest of `rand`.
- **Fallible methods.** Unlike the scalar families (which can't fail), these
  cross into linear algebra and return `Error!` (`NotPositiveDefinite`,
  `DimensionMismatch`, `Failed`, `Unspecified`) via a new module-local
  `Error` + `check`. GSL's non-aborting handler is armed with `gsl.ensureHandler`.
- **Views.** Vectors bridged with `gsl.constVectorViewOf`/`mutVectorViewOf`
  (over `Strided`/`StridedMut` slices); matrices with the session-16
  `gsl.constMatrixViewOf`/`mutMatrixViewOf` (over `gsl.Matrix`/`MatrixMut`).
  Small file-local `constVec`/`mutVec`/`constMat`/`mutMat` wrappers keep the
  call sites readable.

## 3. Tests added (5)
- `choleskyLower` reconstructs `Σ` from `L Lᵀ`; and rejects an indefinite matrix
  (`error.NotPositiveDefinite`).
- Multivariate Gaussian: 40k draws into an `n×2` data matrix, then `mean`/`vcov`
  recover `μ` and `Σ` (loose tolerances); `pdf` peaks at the mean (`= 1/(2π)`
  for `Σ = I`), decreases off-mean, and `logPdf == log(pdf)`.
- Wishart: 20k draws are symmetric with `E[W₀₀] ≈ df·V₀₀`; `pdf > 0` and
  `logPdf == log(pdf)` at a sampled point (its own `L_X` via `choleskyLower`).

Statistical checks stay loose (they verify wiring, not GSL's numerics), per the
module's testing conventions.

## 4. Files changed
- **Modified:** `src/gsl_rand.zig` — `gsl_linalg.h` import; `Error`/`check`;
  `choleskyLower`; `MultivariateGaussian`; `Wishart`; view helpers; 5 tests;
  docstring omission removed.
- No `build.zig`/shim changes; no changes to `gsl.zig` (reused the session-16
  matrix bridge as-is).

## 5. Notes / follow-ups
- With these two families bound, the `rand` module's deferrals are cleared. The
  remaining `rand` omissions (legacy generator zoo, redundant samplers, bulk
  `*_array` helpers, `FILE*` forms) are all **deliberate** — reachable via `c`.
- The session-16 `gsl_matrix` bridge now has a second consumer (after
  `gsl_multifit_nlinear`), confirming the "wait for a real consumer" call and
  the `Matrix`/`MatrixMut` shape.
