# LAPACK Bindings — Session 4 Handover (Q3 + Q4 done)

**Status:** **Q3 (result-shaping consistency) and Q4 (input-preserving `*Alloc`
conveniences + ownership doc pass) are implemented, tested, and merged into
`src/lapack.zig`.** Remaining open questions are **Q5/Q6** (unchanged; see §5).

**Audience:** An agent continuing the linear-algebra binding work in `zarray`
(a.k.a. `ndarray_zig`). Read sessions 1 and 2 first.

**Tests:** `zig test src/lapack.zig -framework Accelerate` → **116 pass**.
`zig build test` → **passes** (whole project).

---

## 1. What Q3 changed

Result axes now follow **one rule**: for each output axis, reuse the
corresponding input axis label if one exists, otherwise use a single *shared*
synthesized inner name. 1-D value outputs stay bare slices. Concretely, with `R`
= A's row-axis label and `C` = A's col-axis label (taken from the caller's
`Axis` enum via `meta.fieldNames`):

| Function | Old axes | New axes |
|---|---|---|
| `qr` → `QrResult(Axis, T)` | `{q_rows,q_cols}` / `{r_rows,r_cols}` | `q = {R, qr_rank}`, `r = {qr_rank, C}` |
| `eigSymVectors` → `EighResult(Axis, T)` | `{evec_rows,evec_cols}` | `vectors = {C, eig}` |
| `eigVectors` → `EigResult(Axis, T, sides)` | `{rvec_*}` / `{lvec_*}` | `right = {C, eig}`, `left = {R, eig}` (share `eig`) |
| `svdVectors` → `SvdResult(Axis, T)` | `{u_*}` / `{vt_*}` | `u = {R, sv}`, `vt = {sv, C}` (share `sv`) |
| `lstsq` | returned `void` | returns `Solution(MatAxis, RhsAxis, T)` = `{C, B.rhs}` |

**Why each component axis reuses what it does** (derived from the defining
identity, not chosen freely):
- `A·v` contracts A's **columns** ⇒ right/eigSym eigenvectors index into `C`.
- `uᴴ·A` contracts A's **rows** ⇒ left eigenvectors index into `R`.
- `A = UΣVᵀ`: free row `i` ⇒ `u.row = R`; free col `j` ⇒ `vt.col = C`; summed
  index is the shared inner `sv`.
- `A = Q·R`: `q.row = R`, `r.col = C`, shared inner `qr_rank` (`q.col ≡ r.row`).
  This fixed the old mismatch where `q_cols ≠ r_rows` despite being the same
  contracted axis.

The two factors that contract over an inner axis now carry the **same**
synthesized name, so they compose by name (e.g. with `tblis` contractions).

### Synthesized inner names (in `src/lapack.zig`)
```
const qr_inner  = "qr_rank";
const eig_inner = "eig";
const svd_inner = "sv";
```

### Collision guard
`assertInnerFree(Axis, inner)` `@compileError`s if a caller's input axis name
equals the synthesized inner name (which would otherwise make `KeyEnum` mint a
duplicate-field enum). It is invoked inside each `*Result` factory. The inner
names are **not** caller-supplied (deliberately — keeps them synthesized).

---

## 2. Implementation notes / mechanics

- Each `*Result` type factory now takes `Axis` as a leading comptime param and
  builds its axis enums with `KeyEnum(&.{ R, inner })` / `KeyEnum(&.{ inner, C })`
  from `meta.fieldNames(Axis)`. The producing function rebuilds the *same*
  `KeyEnum(...)` locally for its `wrapMat` calls; `KeyEnum` is comptime-memoized,
  so the locally-built type is identical to the one in the result struct.
- `qr` was reworked to write R/Q into raw column-major buffers (`@memset` zero +
  index copy) and wrap them with `wrapMat`, instead of `initAlloc` + `.at(...)`
  with literal field names (literal field names are impossible once the names are
  comptime strings). Behavior/allocations are otherwise unchanged.
- `lstsq` now slices `b`'s first `n` rows (`b.idx.sliceAxis(@field(RhsAxis, row),
  0, n)`), wraps them, and `renameAxes` row→`C`, returning the same
  `Solution(...)` view type `solve`/`choleskySolve` return. The view **aliases
  `b`** — nothing new to free; writes through it mutate `b`. The old in-place
  contract (solution in `b`'s first n rows) still holds, so pre-existing callers
  that read `b` directly keep working; they just must not ignore the returned
  value (`_ = try lstsq(...)` or bind it).
- **Values stay bare** (`eigSym`/`eig`/`svd` and the `values`/`s` fields remain
  `[]T` / `[]Complex`). This is a deliberate asymmetry: `svdVectors` returns
  `s: []T` but `u`/`vt` with a named `sv` axis. Intended — do not wrap values.

---

## 3. API surface after Q3

```
solve, lu, luSolve, det, inv                 (LU family; s/d/c/z)
cholesky, choleskySolve                       (Cholesky family; s/d/c/z)
lstsq → Solution(MatAxis, RhsAxis, T)         (gels; real; any layout for A; view of b)
qr → QrResult(Axis, T)                        (geqrf+orgqr; real; any layout)
eigSym, eig, svd                              (values only; real; layout-transparent)
eigSymVectors → EighResult(Axis, T)           (syev jobz=V; real)
eigVectors → EigResult(Axis, T, sides)        (geev; real; comptime sides)
svdVectors → SvdResult(Axis, T)               (gesdd; real; runtime mode)
```

All the `*Result` factories and `lstsq`/`qr`/spectral routines are unchanged in
signature *except* for the added leading `Axis` param on the result types and
`lstsq`'s non-void return.

---

## 4. Tests updated

All result-indexing tests were migrated from the old synthesized names to the
new `{R,k}`/`{C,e}`/etc. names. Since the tests use `IJ = enum { i, j }` for A:
- `qr`: `.q_rows→.i`, `.q_cols→.qr_rank`, `.r_rows→.qr_rank`, `.r_cols→.j`.
- `eigSymVectors`: `.evec_rows→.j` (=C), `.evec_cols→.eig`.
- `eigVectors` right: `.rvec_rows→.j`, `.rvec_cols→.eig`; left: `.lvec_rows→.i`
  (=R), `.lvec_cols→.eig`.
- `svdVectors`: `.u_rows→.i`, `.u_cols→.sv`, `.vt_rows→.sv`, `.vt_cols→.j`.
- `lstsq` tests now bind the returned `Solution` view and read `x.at(.{ .j=…,
  .k=… })` (the renamed row→`j` axis).

---

## 4b. Q4 — input-preserving `*Alloc` conveniences + ownership doc pass

### The audit that drove it
Every public routine was checked for whether it writes into the caller's `a`
(or `b`). Three categories:
- **Mutation is the deliverable** (`lu`, `inv`, `cholesky`): `a` becomes the
  factor/inverse the caller wants — left as-is.
- **`a` overwritten purely as scratch** (`det`, `lstsq`, `eigSym`, `eig`, `svd`,
  `eigVectors`, `svdVectors`, and *column-major* `qr`): the real output is
  separate, so clobbering `a` is an unwanted side effect → got `*Alloc` variants.
- **`a` not destroyed** (`luSolve`, `choleskySolve`, `eigSymVectors`, and
  *row-major* `qr` which already packs a copy).

Note the handover from session 2 under-stated this as “only `det`/`lu`” — it is
the whole scratch set above.

### What was added (chosen: option “1 + 3”)
**1 — `*Alloc` copying conveniences** for every scratch-destroying routine. Each
is a thin wrapper: `const copy = try a.toContiguous(allocator); defer
allocator.free(copy.buf); return <base>(…, copy, …);`. `toContiguous` makes a
fresh row-major copy (base routines already handle row-major), so the caller's
`a` is untouched.

| Base (consumes `a`) | Preserving variant |
|---|---|
| `det` | `detAlloc` (also allocates its own `ipiv`) |
| `lstsq` | `lstsqAlloc` (still writes X into `b` — that's the result buffer) |
| `eigSym` | `eigSymAlloc` |
| `eig` | `eigAlloc` |
| `svd` | `svdAlloc` |
| `eigVectors` | `eigVectorsAlloc` |
| `svdVectors` | `svdVectorsAlloc` |
| `qr` | `qrAlloc` (always preserves; plain `qr` only preserves non-col-major) |

`eigSymVectors` is already input-preserving (copies internally), so it has **no**
`*Alloc` twin — documented as the one inherent exception. `lu`/`inv`/`cholesky`
get no twin (mutation is the deliverable). `solve` gets none either (use
`lu`+`luSolve` to keep `a`).

**3 — ownership/`Input:` doc pass.** Every public routine's docstring now carries
an explicit `Input:` line stating whether `a`/`b` is overwritten or preserved,
and pointing to the `*Alloc` twin where one exists. This sits alongside the
existing `Ownership:` line from session 2.

### Mechanics / gotchas
- The `*Alloc` variants add one O(size-of-`a`) copy. For `qr` specifically a
  col-major input is copied (row-major) then packed again by `qr` (col-major) — a
  double copy, accepted as a convenience cost.
- The returned results never alias the internal copy: value routines return owned
  slices, `qr`/`svdVectors`/`eigVectors` wrap freshly-allocated result buffers,
  and `lstsqAlloc` returns a view of the caller's `b` (not the copy). So freeing
  the copy in the wrapper is always safe.
- New tests (8) assert byte-for-byte input preservation via
  `expectEqualSlices` plus a correctness check for each variant.

---

## 5. Remaining open questions

- **Q5 — Scratch-sizing helper.** Optionally expose min-workspace formulas so
  callers can size buffers without the `lwork=-1` query.
- **Q6 — Complex decompositions.** Add `heev`/`gesdd`/`geev`/`gels` complex
  variants (need `rwork` + reshaped outputs) and unlock complex eigen/singular
  vectors (currently `@compileError` on complex T). Consider a TBLIS-style C shim
  for signature safety first — complex symbols are currently hand-declared
  `extern fn`s (not C-compiler-checked).

---

## 6. Coordination notes

- **Do not edit `src/gsl.zig` / `src/gsl_sf.zig` / gsl-fft work** — other agents
  active. LAPACK work stays isolated in `src/lapack.zig`.
- Watch Zig name-shadowing: `c` (the cImport handle) and primitive type names
  like `u0`/`u1`. In `EigResult` the complex type was renamed `Cx` to free up
  `C` for the column-axis label string.
