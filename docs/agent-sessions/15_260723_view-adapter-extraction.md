# view.zig — Session 15 (the shared matrix-geometry adapter extraction)

**Status:** **Complete (2D core).** Extracts the long-deferred "`NamedArray` →
column-major matrix" adapter into a shared *mechanism* layer, `src/view.zig`, and
rewires the LAPACK and BLAS facades onto it. GSL (the third/forward-looking
consumer) is designed-for but not yet wired (no `gsl_matrix` binding exists yet).

**Tests:** `zig build test --summary all` → **564/564 pass**. Diagnostics clean
on `view.zig`, `lapack.zig`, `accelerate.zig`.

---

## 1. The design decision (why not one common type)

The old roadmap framed this as "delete duplication → one shared adapter," implying
a single descriptor type all backends consume. That's wrong: the three libraries
have genuinely different *requirements* (BLAS: both layouts via a `CBLAS_ORDER`
arg; LAPACK: column-major-native + `trans`; GSL: row-major only, plus
`block`/`owner`). A single shared descriptor type would be a lowest-common
denominator serving none.

**Resolution — a two-layer split:**
- **Mechanism (shared, `view.zig`):** the *stride→geometry* analysis, which is
  identical everywhere. `analyze2d(idx, rows, cols, prefer)` answers the
  policy-free question "can these two axes be seen as a single-`lda` strided
  matrix, and if so: row/col-major, dims, and major-axis stride?" Returns a plain
  `Geometry2d { layout, rows, cols, lda }` or `error.NotContiguous`.
- **Policy (per-library types):** each backend keeps its own descriptor type that
  wraps `analyze2d` and enforces its own rules — ABI enums, integer widths,
  error-vs-panic, ownership bookkeeping, and the offset-aware element pointer.

`analyze2d` deliberately takes only the **index** (not the array), so it stays
type-agnostic (no `Scalar`/const generics) and is unit-testable with a bare
index. Each policy layer derives its own base pointer via `arr.at(zeroes)` — the
one type-dependent line, and where const-ness (`[*]T` vs `[*]const T`) is decided.

## 2. Two things dropped/fixed along the way

- **No `offset` field.** An earlier sketch put `offset` in `Geometry2d`; it's
  redundant. `arr.at(std.mem.zeroes(Axes))` already returns the offset-applied
  pointer (`linear()` folds `offset` in), so facades derive the base directly.
- **`Blas2d` offset bug fixed.** The old `Blas2d`/`Blas2dMut` set the base as
  `@ptrCast(arr.buf.ptr)` — the *start of the buffer*, ignoring `idx.offset`. Any
  sliced/strided submatrix view (nonzero offset) passed to `gemm`/`gemv`/etc.
  pointed at the wrong element. (Tell: `Blas1d` in the same file was already
  offset-aware via `arr.at`.) The rewrite uses `arr.at(zeroes)`, fixing it.

## 3. The subtle part: the tie-break is *policy*, not mechanism

A row/column **vector** (one extent == 1) is layout-ambiguous — the same memory
is validly both row- and column-major. The two old adapters had **opposite**
tie-breaks: LAPACK's `describeGeom` tested col-major first; `Blas2d` tested
row-major first. They were never actually identical here.

A single fixed order broke `her2k`: a 2×1 operand (`k=1`) flipped to `col_major`
while its 2×2 sibling stayed `row_major`, tripping BLAS's "operands share one
physical layout" assertion. So `analyze2d` takes a `comptime prefer: Layout` that
decides the tie **only** when an extent is 1 (never for a genuine matrix, where at
most one layout is contiguous):
- LAPACK passes `.col_major` (matches its old behavior + column-major nature).
- BLAS passes `.row_major` (keeps a lone column/row labeled like its row-major
  neighbors).

This preserves each backend's tested behavior exactly while sharing the kernel.

## 4. Orthogonal questions raised and resolved

- **BLAS stays on CBLAS** (not migrated to LAPACK's Fortran ABI). The shared
  layer carries a *neutral* `Layout` enum; each facade maps it to its ABI. CBLAS
  is actually the better fit — it accepts both layouts natively (its order arg),
  so BLAS keeps its zero-copy row-major path; the Fortran BLAS ABI would lose
  that. Migration is a separate, non-blocking concern.
- **GSL 2D fit (forward-looking).** No `gsl_matrix` binding exists yet (it blocks
  `multifit_nlinear`). `gsl_matrix` is strictly **row-major** (`tda` = row
  stride ≥ cols), with no storage-level transpose flag. The neutral descriptor is
  a clean superset: a future GSL facade calls `analyze2d(..., .row_major)`,
  requires `layout == .row_major` (rejecting or transpose-copying col-major),
  sets `tda = lda`, and attaches `block=null, owner=0`. The only real
  incompatibility — GSL can't express column-major — lives entirely in that
  facade, not the core.

## 5. Files changed
- **Added:** `src/view.zig` — `Layout`, `Geometry2d`, `analyze2d`, `Error`;
  8 unit tests (dense col/row, padded submatrix, single-column clamp, vector
  tie-break, doubly-strided/zero-size/broadcast rejection).
- **Modified:** `src/lapack.zig` — import `view.zig` (as `mat_view`); `Layout`
  now aliases `view.Layout`; removed the private `DescGeom`/`describeGeom`;
  `describe`/`describeConst` call `analyze2d(..., .col_major)` and cast dims to
  `c_int`.
- **Modified:** `src/accelerate.zig` — import `view.zig`; `Blas2d`/`Blas2dMut`
  `init` call `analyze2d(..., .row_major)`, map `layout → CBLAS_ORDER`, and take
  the offset-aware base pointer (bug fix).
- No `build.zig` change (plain `@import`).

## 6. Notes / follow-ups
- **1D not unified.** The GSL vector-view helpers (`constVectorViewOf` /
  `mutVectorViewOf`) and `Blas1d`/`Blas1dMut` are the *1D* analog and were left
  as-is — they parallel `analyze2d` but for vectors. A `Vector1d` kernel is the
  natural next step if desired.
- **GSL 2D facade** lands with the eventual `gsl_matrix` binding, per §4.
