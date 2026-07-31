# LAPACK Bindings — Session 9 (Q6 complex + Q5 scratch: implemented)

**Status:** **Q6 and Q5 are implemented, tested, and green.** This executes the
plan in `06_260722_lapack-q6-complex-and-q5-scratch-plan.md` after the maintainer
confirmed all four §3 open decisions.

**Tests:**
- `zig test src/lapack.zig src/lapack_shim.c -framework Accelerate -lc` →
  **All 127 tests pass** (116 prior + 11 new). Note the shim `.c` must be on the
  command line for the `zarray_*` symbols to link when testing the file directly.
- `zig build test` → **passes** (a clean-cache full rebuild; see note below).

---

## 1. What landed

### New files
- **`src/lapack_shim.h`** — primitive-typed (`float*`/`double*`/`int*`/`char*`)
  prototypes for one `zarray_<name>` forwarder per complex LAPACK entry point.
- **`src/lapack_shim.c`** — `#include <vecLib/lapack.h>` + `#include
  "lapack_shim.h"`; each forwarder casts the interleaved-float buffers to
  `__LAPACK_{float,double}_complex*` and calls the real `c*_`/`z*_`. Because the
  TU includes Apple's header, **the C compiler checks every complex prototype at
  build time** (verified: compiles clean under `-Wall -Wextra`).

### `build.zig`
- Added a second `addCSourceFile` for `src/lapack_shim.c` next to the TBLIS one.

### `src/lapack.zig`
- **Deleted the unchecked hand-written `extern fn` block** (getrf/getrs/getri/
  potrf/potrs). All complex symbols — including the LU/Cholesky family — now go
  through the shim (decision §3.2 confirmed). The Zig side declares `extern fn
  zarray_*` with primitive `[*]f32`/`[*]f64` params; complex arrays are
  `@ptrCast` from `[*]Complex(T)` in the dispatch wrappers.
- **Dispatch wrappers:** complex arms added to `xgetr*`/`xpotr*`/`xgels`/`xgeqrf`/
  `xorgqr` (the last calls `ungqr` for complex). Added `xheev`, `xgeevc`,
  `xgesddc` for the structurally divergent complex signatures (`rwork`, single
  complex `w`, real `s`/`w`). The real `xsyev`/`xgeev`/`xgesdd` stay real-only.
- **Complex code paths + `RealOf(T)` generalization** (decision §3.1 confirmed):
  - `eigSym`/`eigSymAlloc` → `![]RealOf(T)`; complex uses `heev` (Hermitian),
    with `rwork` of length `3n−2`.
  - `eigSymVectors` → `EighResult.values: []RealOf(T)`, complex vectors; complex
    uses `heev` `jobz='V'`. Guard removed.
  - `eig`/`eigAlloc` → complex `geev` returns a single complex `w` directly
    (`rwork` `2n`); no `wr/wi` recombination.
  - `eigVectors`/`eigVectorsAlloc` → complex vectors come out **directly** from
    `geev` (no `assembleEigvecs`, which is now real-only). Guard removed.
  - `svd`/`svdAlloc` → `![]RealOf(T)`; complex `gesdd` with `rwork`.
  - `svdVectors`/`svdVectorsAlloc` → `SvdResult.s: []RealOf(T)`, complex `u`/`vt`.
    Guard removed.
  - `qr`/`qrAlloc` → complex `geqrf` + `ungqr` (unitary Q).
- **Gotchas handled:**
  - `lstsq` complex row-major: **column-major pack + `trans='N'`** (decision §3.3
    confirmed) — the real `trans='T'` reinterpretation trick needs an
    unconjugated transpose that complex `gels` (`'N'`/`'C'` only) can't express.
    Real path unchanged. `a` is left unmodified in the packed case.
  - `svdVectors` complex row-major: the pure relabel/stride-swap identities used
    for real inputs would yield the SVD of the **conjugate** (row-major
    reinterpretation is an unconjugated transpose). Fix: **pack a column-major
    copy and factor natively** for complex row-major; complex column-major is
    still factored in place; real path (relabel trick) untouched.
  - `gesdd` `rwork` length: closed-form helper (query doesn't return it).
- **Latent bug fixed:** `det`'s complex-only branch used `product.scale(sign)`,
  but `std.math.Complex` has no `scale` in this Zig (0.16). Replaced with
  `T.init(product.re*sign, product.im*sign)`. This branch had never been
  instantiated before (no complex `det` caller), so it compiled but would have
  failed the moment someone used it.

### Q5 sizing helpers (Option A, decision §3.4)
Pure, allocation-free `pub fn`s (used internally too, so numbers live in one
place):
- Index lengths: `pivotLen(n)`, `svdIworkLen(m,n)`.
- Complex `rwork`: `eigSymRworkLen(n)` (`3n−2`), `eigRworkLen(n)` (`2n`),
  `svdRworkLen(m,n,want_vectors)` (documented safe upper bound, symmetric in
  m/n).
- Min `lwork` closed forms: `invMinWork`, `lstsqMinWork`, `qrMinWork`,
  `eigSymMinWork(T,n)` (`syev` 3n−1 / `heev` 2n−1), `eigMinWork(T,n,vectors)`.

**Deferred (as the plan allowed):** the optional `…QueryWork(dims)` optimal-size
query helpers, and Q5 options **B** (split scratch/output allocator) and **C**
(`*Into` BYO-buffer). The routines still issue their own `lwork=-1` query
internally for the optimal size; the closed-form minima above let a caller
pre-size a `FixedBufferAllocator` allocation-free when a suboptimal workspace is
acceptable.

---

## 2. Tests added (§1.7)
`eigSym` (Hermitian, real λ), `eigSymVectors` (`A v = λv`, real λ), `eig`
(complex spectrum), `eigVectors` (complex `A v = λv`), `svd` (real σ),
`svdVectors` reconstruction `A ≈ U·diag(s)·Vᴴ` **column-major and row-major**
(the pack path), `qr` (`A = QR`, `Qᴴ Q = I`), `lstsq` complex exact fit
**column-major and row-major fallback** (incl. A-preservation), and a `Q5`
helper self-check. All existing real tests stay green (`RealOf(T)==T`).

---

## 3. Notes / follow-ups
- `zig build test` hit a **transient** `tblis_zig.o: FileNotFound` cache race
  once (two C sources compiled in parallel); a rerun / clean-cache build passes.
  Not a code issue. If it recurs in CI, consider serializing the C compiles.
- Complex `det`/`inv`/`cholesky` now all route through the shim but only `solve`/
  `det`(fixed)/LU/Cholesky have complex tests via the existing suite; adding a
  complex `det`/`inv` round-trip test would be cheap insurance (not required).
- If a hot-loop caller appears, revisit Q5 option B; if a strict no-`alloc`
  guarantee is needed, option C. Neither is needed today.
