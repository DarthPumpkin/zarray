# GSL — Session 18 (`multimin` + `multiroots`: multidimensional optimization & root finding)

**Status:** **Complete.** Adds two new GSL chapters — multidimensional
minimization (`gsl_multimin`) and multidimensional root finding
(`gsl_multiroots`) — as idiomatic Zig bindings, each with both a derivative-free
and a derivative-supplied solver family. Both ride on the callback trampolines
that were already added to `gsl_callback.zig` in the prior session and on the
`gsl_matrix`/`gsl_vector` view bridge from session 16.

**Tests (latest run):**
- `zig build test --summary all` → **586/586 pass** (was 577; +9).
- `diagnostics` on `src/gsl_multimin.zig` and `src/gsl_multiroots.zig` → clean.

---

## 1. What shipped

### `src/gsl_multimin.zig` (`gsl.multimin`)
- **`Minimizer`** (derivative-free, `gsl_multimin_fminimizer`): Nelder-Mead
  simplex. `Method` = `{ nmsimplex2, nmsimplex, nmsimplex2rand }`.
  `init(method, n)`, `set(func, x0, step)` (vector step), `iterate`,
  `xInto`, `minimum`, `size`, `name`, `testSize(epsabs)`.
- **`GradientMinimizer`** (gradient-based, `gsl_multimin_fdfminimizer`):
  `GradientMethod` = `{ conjugate_fr, conjugate_pr, vector_bfgs, vector_bfgs2,
  steepest_descent }`. `set(func, x0, step_size, tol)` (scalar step + line-search
  tol), `iterate`, `xInto`, `gradientInto`, `minimum`, `restart`, `name`,
  `testGradient(epsabs)`.
- **`Function`** `{ raw: gsl_multimin_function }` via `callback.multiminF`;
  context declares `pub fn eval(self, x: []const f64) f64`.
- **`FunctionFdf`** `{ raw: gsl_multimin_function_fdf }` via `callback.multiminFdf`;
  context declares `eval` + `gradient(self, x, g: []f64)`, and optionally a fused
  `evalGradient(self, x, f: *f64, g: []f64)` (a dedicated test confirms the fused
  path is taken when present).

### `src/gsl_multiroots.zig` (`gsl.multiroots`)
- **`Solver`** (derivative-free, `gsl_multiroot_fsolver`): `Method` =
  `{ hybrids, hybrid, dnewton, broyden }`. `init(method, n)`, `set(sys, x0)`,
  `iterate`, `rootInto`, `fInto`, `dxInto`, `name`, `testResidual(epsabs)`,
  `testDelta(epsabs, epsrel)`.
- **`DerivSolver`** (Jacobian-based, `gsl_multiroot_fdfsolver`): `DerivMethod` =
  `{ hybridsj, hybridj, newton, gnewton }`. Analogous surface.
- **`System`** `{ raw: gsl_multiroot_function }` via `callback.multirootF`;
  context declares `pub fn equations(self, x: []const f64, f: gsl.StridedMut(f64))`.
- **`DerivSystem`** `{ raw: gsl_multiroot_function_fdf }` via
  `callback.multirootFdf`; context also declares
  `pub fn jacobian(self, x, J: gsl.MatrixMut(f64))`.

### `src/gsl.zig`
- Added `pub const multimin` / `pub const multiroots` re-exports (with chapter
  doc comments) after `nlinear`, and added `_ = multimin; _ = multiroots;` to the
  `test { … }` discovery block.

## 2. Design (mirrors `gsl_min.zig` / `gsl_multifit_nlinear.zig`)

- Each file: header doc + own `@cImport` (`gsl_errno.h` + the chapter header);
  re-export `disableDefaultErrorHandler`/`strerror`; own `Error` set + `check`;
  `gsl.ensureHandler()` before each fallible entry point.
- **Lifetime:** the callback bundle lives in a caller-owned `Function`/`System`
  value; the solver stores only the raw GSL pointer plus `n`, and `set` passes
  `&func.raw` (GSL retains that pointer across iterations). Callers must keep the
  bundle + its context alive and unmoved between `set` and the last `iterate`.
  This differs slightly from `gsl_min.zig` (which stores the callback *inline* in
  the solver) because these `_set` calls take separate `x`/`step` vectors and the
  bundle is naturally caller-held; keeping it external matched the
  `nlinear.Workspace`/`Problem` split already in the tree.
- **Convergence tests** share a small `testStatus` helper: `GSL_SUCCESS`→`true`,
  `GSL_CONTINUE`→`false`, else map to `Error`.
- **Vector I/O:** inputs built with `gsl.constVectorViewOf` +
  `gsl.Strided(f64).fromSlice`; result vectors read via a per-file `copyVec`
  honoring stride (identical to `nlinear`'s).
- `Error` sets cover the codes these chapters actually emit: `GSL_EINVAL`,
  `GSL_EBADFUNC`, `GSL_ENOPROG` (both), `GSL_ENOPROGJ` (multiroots only),
  `GSL_EMAXITER`, `GSL_ENOMEM`.

## 3. Tests

- **multimin (5):** all three simplex methods and all five gradient methods
  recover the minimum of a shifted paraboloid `(x−a)²+(y−b)²`; gradient at the
  optimum is ~0; the fused `evalGradient` path is exercised via a flag; `restart`
  + `name`; length validation on `set`.
- **multiroots (4):** all four derivative-free and all four Jacobian methods find
  the diagonal point on the unit circle (`x₀²+x₁²=1, x₀=x₁` → `(√½, √½)`);
  residual ~0; `testDelta` convergence with `dx`/`name` accessors; length
  validation on `set`.
- Loose tolerances for the simplex methods (locating a multidim minimum is only
  accurate to ~√eps); tight ones where an analytic gradient/Jacobian is supplied.

## 4. Files changed
- **Added:** `src/gsl_multimin.zig`, `src/gsl_multiroots.zig`.
- **Modified:** `src/gsl.zig` — two re-exports + two test-discovery lines.
- No changes needed to `src/gsl_callback.zig` (the `multiminF/Fdf` and
  `multirootF/Fdf` builders + shared `vecSliceConst`/`vecSliceMut`/
  `stridedMutView`/`matrixMutView` helpers were added in the prior session and
  used as-is).

## 5. Notes / follow-ups
- Not wrapped (raw `c` API remains available): `gsl_multimin_fminimizer_set` with
  precomputed values, and any solver-specific tuning beyond method selection.
- These two chapters are the last of the previously-discussed
  optimization/root-finding surface. Remaining optional GSL territory (e.g.
  wavelet transforms, BSplines) is unclaimed and only worth doing on demand.
