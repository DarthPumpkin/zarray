# GSL — Session 16 (`gsl_matrix` bridge + `gsl_multifit_nlinear`)

**Status:** **Complete.** Lands the last deferred callback chapter,
`gsl_multifit_nlinear` (trust-region nonlinear least squares / curve fitting),
together with the `gsl_matrix` borrowed-view bridge it was blocked on (D-cb5).
Reached as `gsl.nlinear`.

**Tests (latest run):**
- `zig build test --summary all` → **572/572 pass** (was 564; +8 net).
- `zig test src/gsl_multifit_nlinear.zig -lgsl -lblas -llapack -I /opt/homebrew/include -L /opt/homebrew/lib -framework Accelerate`
  → **All 206 tests pass** (chapter pulls in all of `gsl.zig`).
- `diagnostics` on `gsl.zig`, `gsl_callback.zig`, `gsl_multifit_nlinear.zig` →
  clean.

---

## 1. What shipped

### The `gsl_matrix` bridge (`src/gsl.zig`)
The 2-D analogue of the existing `gsl_vector` helpers. GSL's `gsl_matrix` is
row-major with a leading dimension `tda` (element `(i,j)` at `data[i*tda + j]`,
`tda >= cols`); a borrowed view sets `block = null, owner = 0`.

- `Matrix(T)` / `MatrixMut(T)` — borrowed row-major views `{ptr, rows, cols, tda}`
  with `init`, `fromSlice`, `get(i,j)`, and (mut) `set(i,j,v)` / `asConst`.
- `constMatrixViewOf(Mat, m)` / `mutMatrixViewOf(Mat, m)` — generic over each
  `@cImport`'s distinct `gsl_matrix` type (same pattern as
  `constVectorViewOf`/`mutVectorViewOf`).
- Also added `at`/`set`/`asConst` ergonomics to `Strided`/`StridedMut` (needed
  by the strided residual — see §3).

Kept in `gsl.zig` alongside the vector helpers, **not** `view.zig` — GSL stays
decoupled from `NamedArray` by design. `analyze2d` still has exactly two
consumers (LAPACK, BLAS).

### The callback bridge (`src/gsl_callback.zig`)
- `multifitFdf(Fdf, Vec, n, p, ctx)` — residual only (`df = null`, GSL uses
  finite differences). Requires `pub fn residual`.
- `multifitFdfWithJacobian(Fdf, Vec, Mat, n, p, ctx)` — residual + analytic
  Jacobian. Requires `pub fn residual` and `pub fn jacobian`.
- Both mirror the ODE bridge: `void`-or-`c_int` returns, generic over the
  chapter's C types, zero global state. Counters (`nevalf`/…) zeroed.
- 3 `Mock`-based bridge tests (residual-only wiring, tda-aware Jacobian fill
  with a *padded* leading dimension, `c_int` status passthrough).

### The chapter (`src/gsl_multifit_nlinear.zig`, `gsl.nlinear`)
- `Type` (`.trust`), `Trs` (lm/lmaccel/dogleg/ddogleg/subspace2d),
  `Scale` (levenberg/marquardt/more), `Solver` (cholesky/mcholesky/qr/svd),
  `FdType` (forward/central), and `Parameters` (defaults from
  `gsl_multifit_nlinear_default_parameters`, override only what you need).
- `Problem` — the fdf bundle (`initCtx` / `initCtxWithJacobian`).
- `Workspace` — `init`/`deinit`, `initSolution`/`initSolutionWeighted`,
  one-shot `driver(max_iter, conv)` → `DriverResult{converged_by, iterations}`,
  manual `iterate` + `testConvergence` loop, and results:
  `solutionInto`, `residualInto`, `niter`, `name`, `trsName`, `rcond`,
  `covariance(epsrel, out)` (p×p, via `mutMatrixViewOf` over caller memory).
- Error set maps `EINVAL/EBADFUNC/ENOPROG/EFAILED/EMAXITER/ENOMEM`; `Invalid`
  also guards `n < p` / zero-sized problems before `alloc`.
- 5 chapter tests: FD-Jacobian linear fit, analytic-Jacobian fit (+ residual
  check), manual iterate/test loop, covariance/rcond availability, and
  length/shape validation.

Wired into `gsl.zig` (`pub const nlinear`) and the `test { _ = … }` discovery
block.

## 2. Where `gsl_matrix` actually touches the surface
Only three places (traced through `gsl_multifit_nlinear.h`):
1. **Jacobian callback** `df(x, params, gsl_matrix *J)` — user *writes*;
   **optional** (NULL ⇒ finite difference).
2. **`jac(w)`** result — workspace-owned, read for covariance.
3. **`covar(J, epsrel, covar)`** — user-allocated `p×p` output.

So the bridge is needed regardless (2 & 3), but the analytic Jacobian is
optional, exactly like the ODE Jacobian.

## 3. The one surprise: the residual output is **strided**

Scoping assumed all callback vectors are contiguous, so we planned to present
the residual as a plain `[]f64`. That was **wrong** and blew up at first run
(`std.debug.assert(stride == 1)` panic). In finite-difference mode GSL evaluates
the residual **directly into a strided column view of the Jacobian**
(`stride == tda`; observed `stride=2` for a 2-parameter fit).

Correction (kept zero-copy): the residual output is presented as
`gsl.StridedMut(f64)` (write with `f.set(i, v)`), not a slice. The parameter
vector `x` *is* always contiguous, so it stays `[]const f64` (debug-asserted).
The Jacobian is `gsl.MatrixMut(f64)` (already `tda`-aware). This is why
`Strided`/`StridedMut` gained `at`/`set`. The chapter doc calls this out
explicitly under "The residual is presented as a strided view".

(This is a deviation from the "slices for both" call ratified during scoping;
the ratification was predicated on the contiguity assumption, which GSL's
`fdjac.c` violates.)

## 4. Design decisions (as implemented)
- **Vectors:** `x` as `[]const f64` (contiguous params); residual `f` as
  `StridedMut(f64)` (see §3). Jacobian as `MatrixMut(f64)`.
- **MVP scope:** analytic + finite-difference Jacobians in the same pass (the
  matrix bridge is the whole point).
- **Namespace:** `gsl.nlinear`.
- **Bridge location:** matrix views in `gsl.zig` (decoupled from `view.zig`).
- **Driver + manual loop:** both exposed (`driver`; `iterate`/`testConvergence`).
- **`fvv` (geodesic acceleration):** not wired; reachable via raw `c`.

## 5. Files changed
- **Modified:** `src/gsl.zig` — `Matrix`/`MatrixMut`,
  `constMatrixViewOf`/`mutMatrixViewOf`; `at`/`set`/`asConst` on the strided
  views; `nlinear` re-export + test discovery.
- **Modified:** `src/gsl_callback.zig` — import `gsl.zig`; `multifitFdf` /
  `multifitFdfWithJacobian` + helpers; `Mock` structs and 3 tests.
- **New:** `src/gsl_multifit_nlinear.zig` — the chapter (`gsl.nlinear`).
- No `build.zig` / shim changes (plain `@import`, GSL already linked).

## 6. Notes / follow-ups
- `gsl_multifit_nlinear` was the **last** deferred callback chapter (session 11,
  D-cb5). The GSL callback surface is now complete.
- The `gsl_matrix` bridge now has its first consumer; the multivariate-Gaussian
  `randist` families (also deferred on the matrix story) could reuse
  `Matrix`/`MatrixMut` if/when a consumer appears.
- Session-11 header was updated (this session's start) to reflect that
  increments 1–7 shipped; with `gsl_multifit_nlinear` now done, the only thing
  left in that design doc is truly closed out.
