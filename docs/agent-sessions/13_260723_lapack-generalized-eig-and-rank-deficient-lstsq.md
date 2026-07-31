# LAPACK — Session 13 (generalized eig + rank-deficient lstsq)

**Status:** **Complete.** Implements the two demand-driven LAPACK extensions
recommended in the session-12 roadmap assessment: the symmetric/Hermitian-
definite **generalized eigenproblem** (`sygv`/`hegv`) and **rank-deficient least
squares** via divide-and-conquer SVD (`gelsd`). Real + complex, values + vectors,
following the session-10 input-preserving-default / `*Inplace` convention.

**Tests (latest run):**
- `zig test src/lapack.zig src/lapack_shim.c -framework Accelerate -lc` →
  **All 147 tests pass** (134 prior + 13 new).
- `zig build test` → **passes** (whole project).
- `diagnostics` on `src/lapack.zig` → clean.

---

## 1. Context

Session 12's roadmap assessment concluded the dense LAPACK workhorse surface was
complete and recommended *freezing* it, with two optional demand-driven additions
worth doing: the generalized eigenproblem and rank-deficient `lstsq`. The
maintainer chose to implement both now (then defer the `view.zig` extraction; the
GSL callback track is handled separately). This session delivers them.

## 2. What landed

### New public API (`src/lapack.zig`)

| Function | LAPACK | Scalars | Returns |
|---|---|---|---|
| `eigSymGen` / `eigSymGenInplace` | `sygv`/`hegv` (`jobz='N'`) | s/d/c/z | `![]RealOf(T)` (eigenvalues, ascending) |
| `eigSymGenVectors` | `sygv`/`hegv` (`jobz='V'`) | s/d/c/z | `EighResult(Axis, T)` (values + B-orthonormal vectors) |
| `lstsqSvd` / `lstsqSvdInplace` | `gelsd` | s/d/c/z | `LstsqSvdResult` = `{ x, rank, singular_values }` |

- **`GenEigProblem`** enum (`enum(c_int)`) selects LAPACK's `itype`:
  `.a_bx = 1` (A·x = λ·B·x, the standard form; B-orthonormal eigenvectors),
  `.ab_x = 2` (A·B·x = λ·x), `.ba_x = 3` (B·A·x = λ·x). B must be positive
  definite; a non-PD B returns `error.NotPositiveDefinite`.
- **`eigSymGenVectors`** reuses the existing `EighResult(Axis, T)` type — vectors
  carry `{C, eig}` axes (C = A's column axis, `eig` the shared synthesized inner
  axis), exactly like `eigSymVectors`. It is inherently copy-based (like
  `eigSymVectors`), so there is **no** `eigSymGenVectorsInplace`.
- **`lstsqSvd`** returns the minimum-norm solution `x` (a view aliasing `b`'s
  first n rows, axes `{C, B.rhs}` — same vocabulary as `lstsq`/`solve`), the
  effective numerical `rank`, and the owned `singular_values` (descending,
  length min(m,n)). The `rcond` cutoff thresholds small singular values
  (negative ⇒ machine precision). `res.deinit(allocator)` frees
  `singular_values` (the `x` view aliases `b`).

### Shim (`src/lapack_shim.{c,h}`)
Added header-verified forwarders `zarray_{c,z}hegv` and `zarray_{c,z}gelsd`
next to the existing complex forwarders. The `.c` TU includes
`<vecLib/lapack.h>`, so clang checks every complex prototype at build time
(verified: `clang -c … -Wall -Wextra` compiles clean, confirming `chegv_`/
`zhegv_`/`cgelsd_`/`zgelsd_` exist and my argument lists match Apple's).

### Zig dispatch (`src/lapack.zig`)
- Hand-declared `extern fn zarray_{c,z}hegv` / `zarray_{c,z}gelsd` (primitive
  pointer types, mirroring the existing complex externs).
- New dispatch wrappers: `xsygv` (real) / `xhegv` (complex, real `w`+`rwork`);
  `xgelsd` (real) / `xgelsdc` (complex, real `s`/`rcond`/`rwork`). The real arms
  call `c.ssygv_`/`c.dsygv_`/`c.sgelsd_`/`c.dgelsd_` from the cImport; the complex
  arms `@ptrCast` to `[*]f32`/`[*]f64` and route through the shim. This mirrors
  the existing `xsyev`/`xheev` and `xgesdd`/`xgesddc` split (real vs. complex have
  structurally different argument lists — the complex forms carry `rwork`).

## 3. Key implementation decisions

- **Single shared `Axis` for A and B** in the generalized routines. A and B are
  both operators on the same n-dim space, so requiring the same 2-axis enum is
  the natural (and simplest) contract; the result vectors reuse A's column axis.
- **Layout handling for `eigSymGen*`.** `sygv`/`hegv` take one `uplo` for *both*
  A and B, so a per-matrix `uplo`-flip trick (as in `eigSym`) can't absorb mixed
  layouts. Instead:
  - `eigSymGenInplace` normalizes each of A and B to column-major via
    `toColMajorSquare` (in-place transpose of a dense row-major block yields
    genuine column-major logical data ⇒ `tri` is used as-is for both). This
    mutates the caller's A/B — the documented `*Inplace` behavior.
  - `eigSymGen` (preserving) copies A and B with `toContiguous` first.
  - `eigSymGenVectors` makes faithful column-major copies via `readElem` (A's
    copy doubles as the eigenvector output buffer; B's is overwritten with its
    Cholesky factor), so both inputs are preserved.
- **`gelsd` has no transpose flag**, so A must be column-major: `lstsqSvdInplace`
  factors a column-major A in place, or packs a column-major copy for any other
  layout (preserving A in that case) — the same pattern `qr` uses. `lstsqSvd`
  always copies first. B must be column-major (or a single RHS), like `lstsq`.
- **`gelsd` workspace query returns all three sizes** (`work[0]`, `iwork[0]`, and
  for complex `rwork[0]`). The query scratch (`iwq`/`rwq`) is initialized to 0 so
  a non-writing implementation can't feed garbage, and each size is floored at 1.
  This matches the codebase's existing "trust the query" pattern for `lwork`.
- **`gvInfo` helper** maps `sygv`/`hegv` `info`: `<0` panics (binding bug),
  `1..n` ⇒ `ConvergenceFailure`, `>n` ⇒ `NotPositiveDefinite` (the leading minor
  of order `info-n` of B is not PD). No new error variants were needed —
  `NotPositiveDefinite`/`ConvergenceFailure` already exist.

## 4. Tests added (13)

Every scalar dispatch arm is exercised (`ssygv`/`dsygv`/`chegv`/`zhegv`,
`sgelsd`/`dgelsd`/`cgelsd`/`zgelsd`):
- **`eigSymGen`** clean generalized eigenvalues (A=2I, B=diag(2,8) ⇒ {0.25, 1});
  A/B preservation (byte-for-byte); `eigSymGenInplace` matches `eigSymGen`.
- **`eigSymGenVectors`** real residual `‖A v − λ B v‖ ≈ 0` per eigenpair;
  complex Hermitian with B=I reducing to the ordinary eig {1, 3} (+ residual).
- **`lstsqSvd`** overdetermined full-rank (rank 2, exact fit, descending σ);
  **rank-deficient min-norm** (A with a doubled column ⇒ rank 1, min-norm
  solution [0.2, 0.4], second σ ≈ 0); row-major A preservation; complex full
  rank (x = [1, i]).
- **f32 / Complex(f32)** coverage for both `eigSymGen` and `lstsqSvd` to light up
  the single-precision arms (matching the session-10 testing philosophy).

## 5. Files changed
- **Modified:** `src/lapack.zig` (+extern decls, +4 dispatch wrappers,
  `LstsqSvdResult` + `lstsqSvd`/`lstsqSvdInplace`, `GenEigProblem` + `gvInfo` +
  `eigSymGen`/`eigSymGenInplace`/`eigSymGenVectors`, module-header bullet, 13
  tests).
- **Modified:** `src/lapack_shim.h`, `src/lapack_shim.c` (hegv/gelsd forwarders).
- No `build.zig` change (the shim `.c` is already compiled; only symbols added).

## 6. Notes / follow-ups
- The dense LAPACK workhorse surface is now genuinely feature-complete for the
  common scientific cases (the two session-11 "worth-it on demand" gaps are
  closed). Remaining unbound LAPACK is the narrow specialization tail
  (banded/packed storage, Schur, condition estimation, pivoted QR, Sylvester) —
  leave frozen unless a concrete need appears.
- Deferred (maintainer's sequencing): the `view.zig` adapter extraction
  (`Blas2d`/`describe`/GSL vector-views → one layer). The `describe`/`wrapMat`
  mechanics remain unchanged, so that stays a delete-duplication job.
- `lstsqSvd` rank-deficient handling supersedes `lstsq`'s `error.Singular` path
  as the recommended route for possibly-rank-deficient systems.
- Open (raised post-session): the input-preserving routines take `NamedArray`,
  not `NamedArrayConst`. **Resolved in session 14** — the whole input-preserving
  surface now takes `NamedArrayConst`.
