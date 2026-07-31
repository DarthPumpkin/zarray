# LAPACK Bindings — Session Handover

**Status:** LAPACK "workhorse" surface implemented, tested, and wired into the
build. Awaiting design decisions on several follow-ups (see [Open questions](#open-questions--next-steps)).

**Audience:** An agent picking up the linear-algebra binding work in `zarray`
(a.k.a. `ndarray_zig`).

---

## 1. Context & goal

`zarray` is a named-axis n-dimensional array library for Zig (see
`docs/adr/ADR-0001-...md` and `Readme.md`). It wraps mature C numeric libraries
as explicit, per-library bindings over its `NamedArray` view type.

Existing bindings before this session:
- `src/accelerate.zig` — **BLAS** levels 1–3 (complete, heavily tested). LAPACK
  was a `@compileError` stub.
- `src/tblis.zig` — TBLIS tensor ops (einsum-style contraction/reduction). Uses
  a **hand-written C shim** (`src/tblis_zig.c` + `include/tblis_zig.h`).
- `src/gsl.zig` (+ `src/gsl_sf.zig`) — GSL rng/dist/stats/special functions.
  **Another agent is actively working in `gsl.zig`/`gsl_sf.zig`** — avoid editing
  those to prevent write collisions.

The session began as a strategy discussion about which C libraries to bind next,
concluded that **LAPACK was the highest-value gap** (already linked via
Accelerate; unblocks a real linear-algebra story; complements the finished
BLAS), and then implemented the LAPACK workhorse surface in a new file.

---

## 2. What was implemented

New file **`src/lapack.zig`**, exposed as `libs.lapack` in `src/root.zig`. The
old `pub const lapack = struct { @compileError(...) };` stub was removed from
`src/accelerate.zig`.

| Function | LAPACK routine(s) | Scalars | Allocates? | Layout handling |
|---|---|---|---|---|
| `solve` | getrf + getrs | s/d/c/z | no (caller `ipiv`) | absorbs A layout via `trans`; B must be col-major or a vector |
| `lu` | getrf | s/d/c/z | no | in-place factor |
| `luSolve` | getrs | s/d/c/z | no | reuses `lu` factors |
| `det` | getrf | s/d/c/z | no | layout-transparent; singular ⇒ returns 0 |
| `inv` | getrf + getri | s/d/c/z | allocator | layout-transparent, in place |
| `cholesky` | potrf | s/d/c/z | no | absorbs row-major by flipping `uplo` |
| `choleskySolve` | potrs | s/d/c/z | no | same |
| `lstsq` | gels | s/d | allocator | **requires column-major** |
| `qr` | geqrf + orgqr | s/d | allocator | **requires column-major** |
| `eigSym` | syev | s/d | allocator | absorbs row-major via `uplo` flip |
| `eig` | geev | s/d | allocator | layout-transparent |
| `svd` | gesdd | s/d | allocator | layout-transparent |

**Important scope limits (deliberate, first pass):**
- `eig` / `eigSym` / `svd` return **values only** — no eigenvectors / singular
  vectors. (Values are transpose-invariant, which is what makes those routines
  layout-tolerant.)
- Decompositions (`lstsq`/`qr`/`eig`/`eigSym`/`svd`) are **real-only** (`f32`/`f64`).
  Complex is available for the LU/Cholesky family only.
- `qr` and `lstsq` reject non-column-major input with `error.NotColumnMajor`.
  Note the library's default `initContiguous` layout is **row-major**, so these
  two currently error on default-constructed arrays.

---

## 3. Key technical decisions

### 3.1 New LAPACK interface (portable ABI)
Uses Accelerate's **new** LAPACK, opted in via a preprocessor define, rather than
the deprecated legacy CLAPACK:

```zig
pub const c = @cImport({
    @cDefine("ACCELERATE_NEW_LAPACK", "1");
    @cInclude("vecLib/lapack.h"); // NOT the Accelerate umbrella (vImage breaks translate-c)
});
```

- New LAPACK is the **reference Fortran ABI** (`dgesv_(n, nrhs, a, lda, ipiv, b,
  ldb, info)`, all-pointer args, `info` out-param, trailing-underscore symbols).
  This is the same ABI as reference LAPACK / OpenBLAS, so the binding is
  **portable off macOS** later.
- `__LAPACK_int` is `c_int` in the default (LP64) build. `ACCELERATE_LAPACK_ILP64`
  (64-bit ints) is intentionally **not** enabled.
- Requires macOS 13.3+ (fine; project is already macOS-only).
- Char args (`trans`, `uplo`, `jobz`) are `const char *` → passed as `*const u8`
  to a `u8` local (e.g. `var trans: u8 = 'N'; ... &trans`).

### 3.2 Complex support via hand-declared `extern fn`s
Zig's translate-c **cannot model C `_Complex`**, so the cImport yields
`@compileError` for every `c*`/`z*` routine. Worked around by declaring the ~10
complex symbols we use directly (Fortran ABI is all-pointer, and a pointer to two
contiguous floats is exactly `*Complex(T)`):

```zig
extern fn zgetrf_(m: *const c_int, n: *const c_int, a: [*]Complex(f64), lda: *const c_int, ipiv: [*]c_int, info: *c_int) void;
// ... c/z variants of getrf, getrs, getri, potrf, potrs
```

**Contrast with TBLIS:** TBLIS uses a full C shim (`tblis_zig.c`) that
`#include`s real headers, so the C compiler validates signatures. The LAPACK
approach here is lighter (no C file, no build wiring) but the complex signatures
are **hand-transcribed and not C-compiler-checked** — a transcription error is a
silent ABI mismatch (UB). If complex support expands much (see Q6), consider
switching the complex routines to a TBLIS-style C shim for signature safety.

**Complex decompositions are a binding-scope limit, not a LAPACK limit.** LAPACK
has `zgesdd`/`zheev`/`zgeev`/`zgeqrf`+`zungqr`/`zgels`; they just need extra
`rwork` real arrays and a reshaped eigenvalue output.

### 3.3 Layout handling (why some routines tolerate row-major and two don't)
The default `NamedArray` layout is **row-major**. LAPACK is column-major. A
row-major buffer is bit-identical to the column-major transpose, so routines
absorb layout **for free** when a mathematical symmetry allows:

- `solve`/`lu`/`luSolve`: `getrs` has a `trans` flag → feed the row-major buffer
  (= Aᵀ) and set `trans='T'`. Use **`'T'` not `'C'`** even for complex
  (reinterpretation transposes but does not conjugate).
- `inv`: `inv(Aᵀ) = inv(A)ᵀ`, in place.
- `det`: `det(Aᵀ) = det(A)`.
- `cholesky`/`choleskySolve`/`eigSym`: symmetric ⇒ row-major upper = col-major
  lower ⇒ flip `uplo`.
- `eig`/`svd`: values-only, and eigen/singular values are transpose-invariant.

The two holdouts:
- **`qr` genuinely cannot** absorb row-major without copying: `geqrf` has no
  `trans` flag and QR(Aᵀ) is unrelated to QR(A).
- **`lstsq` could in principle** (`gels` has a `trans` flag, so the `solve`-style
  trick applies) but was left column-major-only as a **conservative choice** —
  the m/n/ldb/solution-shape bookkeeping under the swap is error-prone and
  untested. Not a fundamental limit.

The matrix adapter is the private `describe()` function (returns
`{ layout, m, n, lda, ptr }`, or `error.NotContiguous`). It is the **second**
instance of the "NamedArray → column-major matrix" adapter, the first being
`Blas2d`/`Blas2dMut` in `accelerate.zig`. The shared abstraction is now visible
in two places but was **intentionally not extracted** (per the maintainer's
bottom-up "draft per-library, abstract later" plan).

### 3.4 Error taxonomy: recoverable → error, unrecoverable → panic
- **Recoverable → returned error** (`LapackError`): `NotContiguous` (doubly
  strided / negative stride), `NotColumnMajor`, `RhsNotColumnMajor`, `Singular`,
  `NotPositiveDefinite`, `ConvergenceFailure`. Caller can copy/retry or handle
  the numerical condition.
- **Unrecoverable programmer bug → `@panic`**: non-square A, mismatched row
  extents, too-short `ipiv`, `m < n` for `qr`/`lstsq`.
- **Structural → `@compileError`**: wrong axis count, no shared axis name.
- No redundant "panic wrapper" over the error-returning path (the maintainer
  argued, correctly, that `catch @panic` is the caller's one-liner).

### 3.5 Named-axis conventions
- `solve`/`luSolve`/`choleskySolve` take `A: NamedArray(MatAxis, T)` and
  `B: NamedArray(RhsAxis, T)` that share **exactly one axis name** (the "row"
  axis), enforced at comptime. Solving `A·X = B` produces X with axes
  `{A's other axis, B's other axis}`.
- These return the solution as a **view aliasing B's buffer**, with B's row axis
  relabeled to A's column axis (synthesized enum via `axis_meta.KeyEnum`).
  ⚠️ The returned view and `b` alias the same memory.
- `qr` returns a `QrResult(T)` struct with `q`/`r` as freshly-allocated arrays
  using **synthesized axis names** `{q_rows,q_cols}` / `{r_rows,r_cols}`.
  `deinit(allocator)` frees both.
- `eigSym`/`eig`/`svd` return **bare slices** (`[]T` / `[]Complex(RealOf(T))`),
  caller-owned, not `NamedArray`s.
- `lstsq` returns `void`, overwriting `b` in place (solution in its first n rows).
  ⚠️ Inconsistent with `solve`'s renamed-view convention (see Q3).

---

## 4. Test status

- `zig test src/lapack.zig -framework Accelerate` → **all pass** (15 LAPACK tests
  covering solve row/col-major/multi-RHS/singular, lu+luSolve, det 2×2/3×3, inv,
  cholesky+solve + not-PD, eigSym, svd, eig (complex spectrum of a rotation), qr
  (reconstruction + orthonormality), lstsq (overdetermined exact fit), and a
  complex solve).
- `zig build test` → **passes** (full project, including the other agent's
  `gsl_sf`).

Tests use hand-built `NamedIndex` strides to construct column-major inputs
(`colMajor` helper in the test section) since `initContiguous` is row-major.

---

## 5. Files changed

- **Added:** `src/lapack.zig` (~1100 lines incl. tests).
- **Modified:** `src/root.zig` — `libs.lapack = @import("lapack.zig");` (was a
  commented-out line pointing at `accelerate.zig`).
- **Modified:** `src/accelerate.zig` — removed the `pub const lapack` compileError
  stub.

No `build.zig` / `build_config.zig` changes were needed: `lapack` and the
`Accelerate` framework were already linked.

---

## 6. Open questions / next steps

Ordered roughly by value. Q1 and Q2 are the functional gaps.

1. **Eigen/singular VECTORS.** The biggest gap. `eigh`/`eig`/`svd` currently
   return values only. Adding vectors forces a decision on how to present the
   vector matrices across row/col-major storage (likely: allocate column-major
   internally) and interacts with complex (Q6).
2. **`qr`/`lstsq` on row-major (default) arrays.** Since both allocate anyway,
   have them internally pack a column-major copy instead of erroring. For `qr`
   this is the only option; for `lstsq` it's safer than the `trans` trick.
   Recommended: yes.
3. **Result-shaping consistency.** Three conventions are currently mixed:
   - `eigSym`/`eig`/`svd` return bare slices vs. wrapping as 1-axis `NamedArray`.
   - `qr` invents axis names vs. caller-supplied output axis enums (einsum-style
     explicit `AxisOut`).
   - `lstsq` returns `void`+in-place vs. `solve`'s renamed-view. Unify?
4. **In-place destruction of inputs.** `det`/`lu` overwrite A with LU factors.
   Consider copying `*Alloc` conveniences (esp. `det`, where it's surprising).
   Maintainer said "ok" to current behavior; revisit if adding conveniences.
   Also the proposed-but-unbuilt `solveAlloc` (copying variant for multi-column
   row-major B) lives here.
5. **Scratch-sizing helper.** Optionally expose `pivotLen(n) == n` (GSL
   `*WorkLen` style) so callers size `ipiv` without reading docs. Maintainer said
   "ok" to documenting "length ≥ n".
6. **Complex decompositions.** Add `heev`/`gesdd`/`geev`/`gels` complex variants
   (need `rwork` + reshaped outputs). Given complex symbols are already
   hand-declared, consider moving them to a C shim (§3.2) for signature safety
   before expanding.

### Broader roadmap (beyond LAPACK)
From the strategy discussion, the agreed next targets after LAPACK were:
- **`gsl_sf`** (special functions) — elementwise, easy, high-coverage. *Note: an
  agent may already be on this (`src/gsl_sf.zig` exists).*
- **`gsl_fft`** (FFT) — high scientific value, fits the strided-view pattern.
- Then interpolation/splines, fitting/regression, and callback-driven families
  (integration/roots/minimization/ODE — these need a Zig-idiomatic callback
  convention designed once).
- **Deferred:** extracting the shared "NamedArray → column-major matrix" adapter
  (`describe` + `Blas2d`) into one `view.zig` layer, once a third consumer makes
  the right abstraction clear.

---

## 7. Coordination notes

- **Do not edit `src/gsl.zig` / `src/gsl_sf.zig`** — another agent is active
  there. LAPACK work is isolated in `src/lapack.zig`.
- The `describe` adapter in `lapack.zig` duplicates logic from `Blas2d` in
  `accelerate.zig` by design (bottom-up). Keep them mechanically similar so a
  future extraction is a delete-duplication job, not a reconcile-conventions job.
- Verify complex extern-decl signatures against `vecLib/lapack.h` if you touch
  them — they are not C-compiler-checked.
