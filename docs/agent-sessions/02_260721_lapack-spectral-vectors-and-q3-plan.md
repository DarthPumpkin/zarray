# LAPACK Bindings — Session 2 Handover

**Status:** Q1 (eigen/singular vectors) and Q2 (row-major `qr`/`lstsq`) are
**implemented, tested, and merged into `src/lapack.zig`**. A full ownership pass
was done on all docstrings. **Q3 (result-shaping consistency) is fully designed
and signed off but NOT yet implemented** — that is the primary task for the next
session. See [§7](#7-q3-implementation-plan-do-this-next).

**Audience:** An agent continuing the linear-algebra binding work in `zarray`
(a.k.a. `ndarray_zig`). Read session 1 first:
`docs/agent-sessions/01_260721_roadmap-and-lapack-bindings.md`.

**Tests:** `zig test src/lapack.zig -framework Accelerate` → **108 pass**.
`zig build test` → **passes** (whole project, incl. the other agent's `gsl_sf`).

---

## 1. Where session 1 left off

Session 1 implemented the LAPACK "workhorse" surface in `src/lapack.zig` (LU /
Cholesky / lstsq / qr / eigSym / eig / svd), values-only for the spectral
routines, and left **6 open questions**. This session worked through **Q1**,
**Q2**, and the **design of Q3**.

---

## 2. What was implemented this session

### 2.1 Q1 — eigen/singular VECTORS (done)

Added three vector-returning routines alongside the untouched values-only
`eigSym`/`eig`/`svd`:

| Function | LAPACK | Returns | Config |
|---|---|---|---|
| `eigSymVectors` | `syev` (`jobz='V'`) | `EighResult(T)` = `{ values: []T, vectors }` | `tri: Triangle` |
| `eigVectors` | `geev` | `EigResult(T, sides)` (comptime-shaped) | `sides: EigSides` **comptime** → `.right`/`.left`/`.both` |
| `svdVectors` | `gesdd` | `SvdResult(T)` = `{ s: []T, u, vt }` | `mode: SvdMode` **runtime** → `.thin`/`.full` |

**Design decisions (all settled with the maintainer):**

- **Allocation-free trio on the input side.** Vectors are *not*
  transpose-invariant, so each routine first gets A into LAPACK's column-major
  orientation with the cheapest correct trick:
  - `eigSymVectors`: makes a column-major **copy** of A (symmetric ⇒ `uplo`
    picks the triangle); **A is left unmodified**. The copy doubles as the
    eigenvector output buffer.
  - `eigVectors`: **transposes A in place** when it isn't column-major (packs a
    copy only for a *padded* row-major view). A is used as scratch (overwritten).
  - `svdVectors`: factors a row-major A as its transpose and undoes the U↔V swap
    by **relabeling strides** (no data copy). A is used as scratch.
- **`sides` is comptime and changes the return type** (invalid states
  unrepresentable — you can't reach `.left` when you asked for `.right`).
  **`mode` is runtime** because `.thin`/`.full` share the same fields (only
  runtime dims differ). Rule: *comptime iff it changes the return type.*
- **Complex-conjugate eigenvector pairs are assembled** from `geev`'s packed real
  columns into `Complex(T)` columns (`assembleEigvecs`), matching the complex
  eigenvalue output.
- **Real-only** (`f32`/`f64`); complex deferred to Q6 (via `@compileError`).
- Result matrices carry **synthesized axis names** (like `qr`) — *this is exactly
  what Q3 revises; see §7.*

New private helpers added (reused by Q3 later): `readElem`, `wrapMat`,
`ColMajorSquare`/`toColMajorSquare`, `assembleEigvecs`.

### 2.2 Q2 — row-major `qr`/`lstsq` (done)

Both previously rejected non-column-major input with `error.NotColumnMajor`.

- **`lstsq` — zero-copy (a real trick).** `gels` has a `trans` flag, so a
  row-major A (bit-identical to Aᵀ column-major) is absorbed for free: set
  `trans='T'` and swap the logical `m`/`n` handed to `gels`. **No allocation
  added.** Plain `'T'` (never `'C'`) is correct — lstsq is real-only. B still
  must be column-major (independent constraint, unchanged).
- **`qr` — packed copy (no cheap trick).** `geqrf` has no transpose flag and QR
  is not transpose-invariant, so a non-column-major A can't be absorbed for free.
  The only zero-copy route is an LQ factorization of Aᵀ (`gelqf`/`orglq`) — *not*
  bound (deliberately). Instead: column-major A is factored in place; a
  non-column-major A is packed into a column-major work buffer (one m×n alloc)
  and **the caller's A is left intact**.

### 2.3 Docstring ownership pass (done)

Every public function's docstring now explicitly states **whether and how to free
the return value**, in an `Ownership:` line:
- View-aliasing results (`solve`/`luSolve`/`choleskySolve`) → "shares `b`'s
  memory — nothing to free".
- Void/in-place (`lu`/`det`/`inv`/`cholesky`/`lstsq`) → "nothing to free; internal
  scratch freed before returning".
- Owned slices (`eigSym`/`eig`/`svd`) → "free with `allocator.free(slice)`".
- Result structs (`qr`, `eigSymVectors`, `eigVectors`, `svdVectors`) → "free with
  `res.deinit(allocator)`"; each `deinit` also got a doc comment naming the
  fields it releases.

---

## 3. Current public API surface (`src/lapack.zig`)

```
solve, lu, luSolve, det, inv                 (LU family; s/d/c/z)
cholesky, choleskySolve                       (Cholesky family; s/d/c/z)
lstsq                                          (gels; real; any layout for A)
qr → QrResult                                  (geqrf+orgqr; real; any layout)
eigSym, eig, svd                               (values only; real; layout-transparent)
eigSymVectors → EighResult                     (syev jobz=V; real)
eigVectors → EigResult(T, sides)               (geev; real; comptime sides)
svdVectors → SvdResult                         (gesdd; real; runtime mode)
```

Enums/types: `Triangle`, `EigSides`, `SvdMode`, `LapackError`, `Solution`,
`QrResult`, `EighResult`, `EigResult`, `SvdResult`.

Complex support is still LU/Cholesky-family only. Everything spectral/
factorization is real-only (Q6).

---

## 4. Test status

`zig test src/lapack.zig -framework Accelerate` → **108 tests pass**. New tests
this session cover:
- vectors: `A·v=λv` + unit norm (eigSym), real & complex-conjugate right
  eigenvectors, `.both` left-eigenvector identity, SVD reconstruction
  `A≈U·diag(s)·Vᵀ` across thin/full and both native (col-major) and
  output-swap (row-major, m<n) paths.
- Q2: `lstsq` row-major exact fit + row/col agreement; `qr` row-major 2×2
  (reconstruction + A-preserved) and tall row-major 3×2 (reconstruction + QᵀQ=I).

Gotchas hit (avoid re-hitting): loop vars named `c`, `u0`, `u1` **shadow
primitives / the `c` cImport handle** — Zig errors. Use `cc`, `uu0`, etc.

---

## 5. Files changed this session

- **Modified:** `src/lapack.zig` only. (~+560 lines incl. tests.) Added the three
  vector routines + result types + helpers; rewrote `lstsq`/`qr` layout handling;
  ownership docstrings throughout.
- No `root.zig` / `build.zig` changes. `src/gsl.zig` / `src/gsl_sf.zig`
  **untouched** (other agent still active there — do not edit).

---

## 6. Q3 — the decision (context for §7)

Q3 asks whether to unify result-shaping. The maintainer settled on **one rule**:

> For each **output axis**, ask: does it correspond to an input axis?
> **Yes → reuse that input axis's label. No → synthesize a custom name** — and
> give the *same* synthesized name to both factors that contract over it, so they
> compose by name. **1-D value outputs stay bare slices** regardless.

Rationale: LAPACK decompositions introduce genuinely new axes (rank index,
eigenvalue index, singular index) that map to no input axis, but their
*component* axes (rows/cols of the vectors) DO correspond to A's row/col spaces.
Reusing those labels makes results compose with the caller's vocabulary (and with
`tblis` contractions) instead of forcing the caller to learn opaque names like
`q_cols`. Which input axis a component reuses is **fixed by what gets contracted**
in the defining identity (see table below) — it is not a free choice.

The maintainer explicitly confirmed the three judgment calls in §7.3.

---

## 7. Q3 IMPLEMENTATION PLAN (do this next)

### 7.1 Target axis layout per result

Let `R` = A's row-axis label, `C` = A's col-axis label (both known at comptime
from the caller's `Axis` enum). `k`/`e`/`l` are **synthesized inner axes**.

| Function | Defining identity | New result axes |
|---|---|---|
| `solve`/`luSolve`/`choleskySolve` | A·X=B | `X = {C, B.rhs}` — **already correct, leave as-is** |
| `lstsq` | min‖A·X−B‖ | `X = {C, B.rhs}`, as a **view of `b`'s first n rows** (was `void`) |
| `qr` | A(i,j)=Σₖ Q(i,k)R(k,j) | `Q = {R, k}`, `Rfactor = {k, C}` |
| `eigSymVectors` | A·v=λv | `vectors = {C, e}` |
| `eigVectors` | A·vr=λvr ; Aᵀ·vl=λ̄vl | `right = {C, e}`, `left = {R, e}` (share `e`) |
| `svdVectors` | A(i,j)=Σₗ U(i,l)s(l)Vt(l,j) | `U = {R, l}`, `Vt = {l, C}` (share `l`) |

**Why each component axis reuses what it does (derive, don't guess):**
- `A·v` contracts A's **columns** ⇒ a right/eigSym eigenvector indexes into `C`.
- `vlᴴ·A` contracts A's **rows** ⇒ a left eigenvector indexes into `R`.
- Reconstruction `A=UΣVᵀ`: free row index `i` ⇒ `U`'s row = `R`; free col index
  `j` ⇒ `Vt`'s col = `C`; the summed `l` is the inner (shared U.col ≡ Vt.row).
- QR same shape as SVD: `Q.row=R`, `Rfactor.col=C`, shared inner `k`
  (Q.col ≡ Rfactor.row). **This fixes today's mismatch** where `q_cols` ≠
  `r_rows` even though they are the same contracted axis.

### 7.2 Concrete construction notes

- **Inner axes are a single shared synthesized name per result.** Build the
  result axis enums as `KeyEnum(&.{ R, inner })` / `KeyEnum(&.{ inner, C })`
  using the caller's `R`/`C` field-name strings (available via
  `meta.fieldNames(Axis)`), mirroring how `SolutionAxis` already composes names.
- **Suggested inner names:** `qr` → `"qr_rank"`; eig → `"eig"`; svd → `"sv"`
  (bikeshed as you like, but keep them distinctive).
- **Collision guard (judgment call 3, approved):** if `R` or `C` equals the
  synthesized inner name, `KeyEnum` produces a duplicate-field enum (confusing
  error). Add an explicit `@compileError` with a clear message
  ("axis name '…' collides with the synthesized … axis; rename your input axis").
  Do **not** make the inner name caller-supplied (that contradicts "keep
  synthesized").
- **`lstsq` view construction:** the solution occupies the first `n` rows of `b`
  (column-major). Build the view by slicing `b`'s row axis to `[0, n)` then
  renaming row→`C`:
  - `b.idx.sliceAxis(comptime @field(RhsAxis, row), 0, nn)` → a `NamedIndex`
    with row extent `nn` (offset unchanged, strides unchanged);
  - wrap as `NamedArray(RhsAxis, T).init(sliced, b.buf)`, then `.renameAxes(
    SolutionAxis(MatAxis, RhsAxis), &.{ .{ .old = row, .new = col } })`.
  - Return type becomes `Solution(MatAxis, RhsAxis, T)` (already defined; same as
    `solve`). Verify `sliceAxis` asserts hold (`0 < nn <= m`).
- **Values stay bare (judgment call 2, approved):** `eigSym`/`eig`/`svd` and the
  `values`/`s` struct fields remain `[]T`/`[]Complex`. This creates a deliberate
  asymmetry (e.g. `svdVectors` returns `s: []T` but `U`/`Vt` with a named `l`
  axis). Intended — do not wrap the values.

### 7.3 The three judgment calls (all APPROVED by maintainer)

1. **`lstsq` `void → Solution` view.** Its axes `{C, B.rhs}` correspond to inputs,
   so reuse (no brand-new enum). Accepted as a signature/behavior change.
2. **Values stay bare slices** even though vectors gain a named `e`/`l` axis.
3. **Collision handled via `@compileError`**, inner names stay synthesized.

### 7.4 Work items / checklist

1. `qr`: rename `QrResult`'s axes from `{q_rows,q_cols}`/`{r_rows,r_cols}` to
   `{R, k}` / `{k, C}` (build from `Axis` field names). Update the R/Q extraction
   `.at(...)` keys and the `deinit` (unchanged) accordingly.
2. `eigSymVectors`: `EighResult` vectors axis `{evec_rows,evec_cols}` → `{C, e}`.
3. `eigVectors`: `EigResult` `right`/`left` axes → `{C, e}` / `{R, e}` (shared
   `e`). Note all three `sides` struct variants.
4. `svdVectors`: `SvdResult` `u`/`vt` axes → `{R, l}` / `{l, C}` (shared `l`).
5. `lstsq`: return the renamed view (see §7.2).
6. Add the collision `@compileError` guards.
7. **Update every test** that indexes results by the old synthesized names
   (`.q_rows`, `.r_cols`, `.evec_rows`, `.rvec_*`, `.lvec_*`, `.u_rows`,
   `.vt_cols`, …) to the new `{R,k}` / `{C,e}` / etc. names. Since tests use
   `IJ = enum { i, j }` for A, the new names will be `.i`/`.j` plus the inner
   axis (e.g. `res.vectors.at(.{ .j = …, .eig = … })`).
8. Update docstrings to describe the new axis names.
9. Re-run `zig test src/lapack.zig -framework Accelerate` and `zig build test`.

### 7.5 Watch out for

- `KeyEnum` requires **distinct** field names → the collision guard matters.
- The shared inner name means `Q` and `Rfactor` (and `U`/`Vt`, `right`/`left`)
  reference the **same** axis label; that's the intended composability, not a bug.
- `solve`/`luSolve`/`choleskySolve` are already correct (`SolutionAxis` derives
  from inputs). **Do not touch them.**
- Keep `describe`/`wrapMat` mechanics identical between BLAS-adjacent code and
  LAPACK so the deferred adapter extraction stays a delete-duplication job.

---

## 8. Remaining open questions (after Q3)

Unchanged from session 1; ordered by value:

- **Q4 — In-place destruction of inputs.** `det`/`lu` overwrite A. Consider
  `*Alloc` copying conveniences (esp. `det`). Maintainer said current behavior is
  OK; revisit if adding conveniences. (Note: `eigSymVectors`/`qr`-row-major now
  *preserve* A; `eig`/`svd`/`eigVectors`/`svdVectors` use A as scratch.)
- **Q5 — Scratch-sizing helper.** Optionally expose `pivotLen(n)==n` etc. so
  callers size `ipiv`/workspace without docs. For the allocating spectral
  routines this was discussed: outputs need heap (runtime shapes) and optimal
  workspace needs the `lwork=-1` query, so a fully no-alloc API would use the
  *minimum* workspace formulas + caller buffers. Maintainer OK with documenting
  "length ≥ n" for now.
- **Q6 — Complex decompositions.** Add `heev`/`gesdd`/`geev`/`gels` complex
  variants (need `rwork` + reshaped outputs). Complex symbols are currently
  hand-declared `extern fn`s (not C-compiler-checked); consider a TBLIS-style C
  shim for signature safety before expanding. This also unlocks complex
  eigen/singular VECTORS (the vector routines currently `@compileError` on
  complex T).

### Broader roadmap (beyond LAPACK)
Post-LAPACK targets from session 1 stand: `gsl_sf` (in progress by another
agent), `gsl_fft`, then interpolation/splines, fitting/regression, and
callback-driven families (integration/roots/minimization/ODE). Deferred: extract
the shared "NamedArray → column-major matrix" adapter (`describe` +
`Blas2d`) once a third consumer clarifies the right abstraction.

---

## 9. Coordination notes

- **Do not edit `src/gsl.zig` / `src/gsl_sf.zig`** — another agent is active there.
- LAPACK work stays isolated in `src/lapack.zig`.
- Complex `extern fn` signatures are **not** C-compiler-checked; verify against
  `vecLib/lapack.h` if you touch them.
- Watch Zig name-shadowing: `c` (the cImport handle), and primitive type names
  like `u0`/`u1`/`i8` are illegal as identifiers.
