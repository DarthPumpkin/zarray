# LAPACK Bindings — Session 5 Plan (Q6 complex + Q5 scratch)

**Status:** **Q1–Q4 are implemented, tested, and merged** (see sessions 1–4).
This doc is a **design/handover plan for Q6 and Q5** — *no implementation yet*.
It records the agreed direction so the next session can execute.

**Audience:** An agent continuing the linear-algebra binding work in `zarray`
(a.k.a. `ndarray_zig`). Read sessions 1–4 first; most relevant is
`04_260722_lapack-q3-result-shaping.md` (Q3 + Q4).

**Tests today:** `zig test src/lapack.zig -framework Accelerate` → **116 pass**.
`zig build test` → **passes**.

---

## 0. Ordering decision

**Do Q6 (complex decompositions) first, then Q5 (scratch sizing).** Q6 is the
higher-demand feature and it *defines the final set of routines and their scratch
needs* (esp. the extra `rwork` arrays). Designing Q5's sizing helpers before the
complex scratch requirements exist would mean revisiting them immediately after.

---

## 1. Q6 — complex decompositions (heev / geev / gesdd / gels + ungqr)

### 1.1 Current state

- Complex is bound **only** for the LU + Cholesky families (`getrf/getrs/getri`,
  `potrf/potrs`) via **hand-written `extern fn`** blocks in `src/lapack.zig`
  (lines ~80–91). These are **not checked** against any header.
- Everything spectral/factorization (`qr`, `eigSym`, `eig`, `svd`, and the vector
  routines) is real-only, guarded by `@compileError` in the `x*` dispatch
  switches and/or at the top of each public fn.
- `Complex(f32/f64)` is layout-compatible with C `_Complex`, so element typing is
  *not* the hard part.

### 1.2 Why a C shim here (the real justification)

The complex-type marshalling is easy (layout compat). The hazard is that the
complex decomposition routines have **structurally different argument lists** than
their real namesakes — extra `rwork`, a merged complex `w`, real outputs for
Hermitian/SVD. A hand `extern fn` that drops `rwork` or mis-splits `w` compiles
fine and corrupts memory at runtime.

A shim whose body does `#include <vecLib/lapack.h>` and forwards to the real
`zheev_`/`zgeev_`/… makes the **C compiler verify every prototype against Apple's
header at build time**. The Zig↔shim boundary then uses only primitive pointer
types (`[*]f64`, `[*]c_int`, `*const u8`), so there's no complex ambiguity and the
risky ABI matching lives right next to the verified call.

Precedent: `src/tblis_zig.c` is already compiled via
`root_mod.addCSourceFile(.{ .file = .{ .cwd_relative = "src/tblis_zig.c" }, … })`
in `build.zig` (~line 78). The LAPACK shim is *much* smaller than TBLIS — no
malloc/marshalling, just forwarders.

**Honest tradeoff:** round-trip tests would likely also catch a bad prototype
(crash/garbage), so the shim is compile-time defense-in-depth rather than strictly
required. Maintainer chose the shim — accepted; it's most valuable exactly for
these divergent, `rwork`-carrying signatures. Bonus: we can route the existing
getrf/potrf hand-externs through the shim too and **delete the unchecked
`extern fn` block** for uniform verification.

### 1.3 Shim design

New files:
- `src/lapack_shim.c` — one forwarder per complex entry point.
- `src/lapack_shim.h` — prototypes of the forwarders (primitive types only), for
  the Zig side to `@cInclude` **or** to mirror as `extern fn` (either works; a
  header keeps them in one place).

Wiring: add a second `addCSourceFile` for `src/lapack_shim.c` next to the TBLIS
one in `build.zig`. The shim TU defines `ACCELERATE_NEW_LAPACK` and includes
`<vecLib/lapack.h>`.

Sketch:

```c
// src/lapack_shim.c
#define ACCELERATE_NEW_LAPACK 1
#include <vecLib/lapack.h>

void zheev_shim(const char* jobz, const char* uplo, const int* n,
                double* a, const int* lda, double* w,
                double* work, const int* lwork, double* rwork, int* info) {
    zheev_(jobz, uplo, n, (__LAPACK_double_complex*)a, lda, w,
           (__LAPACK_double_complex*)work, lwork, rwork, info);
}
// … cheev_shim, cgeev_shim/zgeev_shim, cgesdd_shim/zgesdd_shim,
//    cgels_shim/zgels_shim, cungqr_shim/zungqr_shim, and optionally the
//    getrf/getrs/getri/potrf/potrs family for uniform checking.
```

Zig side: extend the `x*` dispatch switches (`xsyev`, `xgeev`, `xgesdd`, `xgels`,
`xorgqr`, plus a new `xheev`) to call the `_shim` symbols for the complex arms and
drop the `@compileError` else-branches. Complex arrays are passed as `[*]f32/f64`
via `@ptrCast` from `[*]Complex(T)`.

### 1.4 Complex signature reference (what differs from the real form)

| Routine (real → complex) | Key differences vs real | Extra scratch |
|---|---|---|
| `ssyev`/`dsyev` → `cheev`/`zheev` | eigenvalues `w` stay **real** (`RealOf(T)`); Hermitian, not symmetric | `rwork` length `max(1, 3n−2)` |
| `sgeev`/`dgeev` → `cgeev`/`zgeev` | **single complex `w`** (no `wr`/`wi` split); vectors come out complex directly | `rwork` length `2n` |
| `sgesdd`/`dgesdd` → `cgesdd`/`zgesdd` | singular values `s` stay **real**; `u`/`vt` complex | `rwork` (see gotcha 1.6.2) + `iwork` `8·min(m,n)` |
| `sgels`/`dgels` → `cgels`/`zgels` | `trans ∈ {'N','C'}` — **no plain `'T'`** | none (no `rwork`) |
| `sgeqrf`/`dgeqrf` → `cgeqrf`/`zgeqrf` | same shape, complex `tau` | (query) |
| `sorgqr`/`dorgqr` → **`cungqr`/`zungqr`** | unitary, not orthogonal; name changes | (query) |

The `lwork = -1` query still works for complex: `work[0]` returns the optimal
`lwork` as a complex scalar whose **real part** is the size. `lworkFrom` already
takes `.re` for complex, so it's ready.

### 1.5 Per-routine changes + the type story

Making the value/vector types depend on `RealOf(T)` is a clean generalization —
**for real `T`, `RealOf(T) == T`, so real callers are unaffected.**

- **`eigSym` / `eigSymAlloc`** → dispatch to `heev` for complex. Return type
  becomes `![]RealOf(T)` (real eigenvalues). Add `xheev`. `tri` now selects the
  Hermitian triangle.
- **`eigSymVectors`** → `EighResult(Axis, T)`: `values: []RealOf(T)`,
  `vectors: NamedArray(_, T)` (complex vectors). Uses `heev` `jobz='V'`.
- **`eig` / `eigAlloc`** → for complex, `cgeev`/`zgeev` return a single complex
  `w`; skip the `wr/wi` recombination. Return type stays
  `![]Complex(RealOf(T))`.
- **`eigVectors` / `eigVectorsAlloc`** → complex path gets vectors **directly**
  (no `assembleEigvecs` packing). `assembleEigvecs` becomes real-only. `values`
  stay `[]Complex(RealOf(T))`; vectors `Complex(RealOf(T))` (already the element
  type used).
- **`svd` / `svdAlloc`** → `s` becomes `![]RealOf(T)`. Complex `gesdd` adds
  `rwork`.
- **`svdVectors` / `svdVectorsAlloc`** → `SvdResult(Axis, T)`: `s: []RealOf(T)`,
  `u`/`vt` complex `T`. The row-major output-swap relabel trick is layout-only, so
  it carries over — **but** for complex the swap must also account for conjugation
  (the transposed factorization gives `Uᴴ`/`Vᴴ` relationships, not plain `ᵀ`);
  verify the relabel identities on a complex example before trusting them.
- **`qr` / `qrAlloc`** → complex `geqrf` + **`ungqr`**. Add complex arms to
  `xgeqrf`/`xorgqr` (or a new `xungqr`). `QrResult(Axis, T)` already generic.
- Remove the `isComplex(T)` `@compileError` guards from each of the above as its
  complex path lands.

### 1.6 Gotchas to bake in

1. **`lstsq` row-major zero-copy trick breaks for complex.** A row-major complex
   `A` reinterpreted column-major is `Aᵀ`, but `zgels` only offers `'N'`/`'C'`
   (conjugate transpose `Aᴴ`), not plain `'T'`. So the free reinterpretation reals
   enjoy is unavailable. Fallback: **pack a column-major copy for the complex
   row-major case** (mirror what `qr` already does), keeping `'N'`. `solve` is
   **fine** — `getrs` supports `'T'`, so complex row-major `solve` already works.
2. **`gesdd`'s `lrwork` is not returned by the query.** Unlike `lwork`, the real
   `rwork` length must come from LAPACK's **documented closed-form** (a function of
   `m`, `n`, and `jobz`). Compute it explicitly; do not try to query it.
3. **Hermitian ≠ symmetric.** `heev` reads a triangle and assumes the imaginary
   part of the diagonal is zero. Document that `eigSym`(complex) means *Hermitian*.
4. **Hand-extern deletion.** Once the shim covers getrf/potrf too, delete the
   `cf32/cf64` + `extern fn` block; keep a note that all complex symbols are now
   header-verified via the shim.

### 1.7 Tests

- Round-trip identities per routine on small complex matrices:
  `A·v = λv` (heev, real λ; geev, complex λ), `A ≈ U·diag(s)·Vᴴ` (gesdd),
  `A = Q·R` with `Qᴴ Q = I` (geqrf/ungqr), `min‖A·X−B‖` (gels), incl. the
  **row-major complex `lstsq` fallback** path.
- A Hermitian case verifying eigenvalues come back real (`[]RealOf(T)`).
- Keep the existing real tests green (RealOf(T)==T ⇒ no signature churn).

### 1.8 Checklist

1. Add `src/lapack_shim.c` + `.h`; wire into `build.zig`.
2. Add complex arms + `xheev`/`xungqr` to the dispatch switches (via shim).
3. Generalize value/`s` types to `RealOf(T)`; adjust `EighResult`/`SvdResult`.
4. Complex branches: `eig`/`eigVectors` (direct complex `w`/vectors),
   `svdVectors` (verify complex relabel/conjugation), `qr` (`ungqr`).
5. `lstsq` complex row-major fallback (column-major pack).
6. Remove real-only `@compileError` guards as each path lands.
7. Optionally route getrf/potrf through the shim; delete hand-externs.
8. Tests (1.7). Re-run both test commands.

---

## 2. Q5 — scratch sizing (revised: **no `Workspace` object**)

### 2.1 The allocator-vs-Workspace question (resolved)

A reusable `Workspace` object was considered and **rejected**. Rationale
(maintainer's point, agreed):

- **An allocator already handles memory reuse.** Callers who care about a hot
  loop wrap a reused buffer in a `FixedBufferAllocator`, or reset an arena each
  iteration — alloc/free then costs ~nothing, and the *caller* chooses the
  strategy. A `Workspace` would reinvent that, less flexibly.
- **The `lwork = -1` query** is a computation, not an allocation, so an allocator
  was never the thing helping there — and the query is cheap next to the O(n³)
  factorization, so caching it buys little.
- Net: a `Workspace` mostly duplicates the allocator interface. Keep taking an
  `Allocator`.

**The only thing a plain allocator cannot provide** is *knowledge of how big the
scratch must be* (`ipiv`, `iwork`, `rwork`, min `lwork`). That knowledge is
LAPACK-specific and currently hidden. Exposing it is the whole of Q5.

### 2.2 What Q5 should expose

- **Pivot / index lengths** (closed form): `ipiv` length `= n` for the
  LU family; `iwork` length `= 8·min(m,n)` for `gesdd`. These are the
  caller-managed buffers in `solve`/`lu`/`det`/`luSolve` today — the only pain is
  "how long?", which is a doc line now.
- **Workspace sizing** — two flavors:
  - **Minimum (closed form):** LAPACK documents a min `lwork` per routine; expose
    it so callers can pre-size without the query (suboptimal but allocation-free).
  - **Optimal (query helper):** optionally expose `…QueryWork(dims) → usize`
    wrapping the `lwork=-1` call, for callers who want the optimal size once and
    reuse the buffer.
- **`rwork` lengths** for the complex routines (from Q6): closed-form helpers
  (`heev` `3n−2`, `geev` `2n`, `gesdd` per formula).

### 2.3 Options (in increasing ambition)

- **A — Sizing helpers only (recommended core).** Pure functions
  (`pivotLen(n)`, `svdIworkLen(m,n)`, `eigSymMinWork(n)`, `…QueryWork(...)`, plus
  the `rwork` helpers). No behavior change. This alone lets a caller size a
  `FixedBufferAllocator` correctly and get a **bounded/zero-heap** path *through
  the existing allocator-taking API* (FBA satisfies the internal `alloc` from a
  fixed buffer). This is the highest value-per-surface option.
- **B — Split scratch vs output allocator (light, optional).** Add an optional
  `scratch_allocator` (default = the main allocator) so transient buffers can come
  from an arena/FBA while *owned outputs* (which outlive the call) come from a
  durable allocator. Still "just allocators," no new object. Useful because
  outputs can't come from a per-iteration-reset arena but scratch can.
- **C — `*Into` bring-your-own-buffer variants (probably unnecessary).** Fully
  no-internal-alloc twins taking caller `work`/`ipiv`/output buffers. Big surface;
  option A + `FixedBufferAllocator` usually covers the same need without it. Only
  pursue if a real caller needs a hard "no `alloc` call at all" guarantee.

### 2.4 Recommendation

**Option A** (sizing helpers), folding in the Q6 `rwork` formulas so they land
together. Consider **B** if a concrete hot-loop caller appears. Skip **C** unless
a strict allocation-free requirement shows up. **Do not add a `Workspace`
object.**

---

## 3. Open decisions for the maintainer

1. Q6: confirm the **`RealOf(T)` return-type generalization** for
   `eigSym`/`svd`/`eigSymVectors`/`svdVectors` values (real callers unaffected).
2. Q6: OK to **route getrf/potrf through the shim and delete the hand-externs**,
   or keep the shim scoped to only the new complex decompositions?
3. Q6: `lstsq` complex row-major — confirm the **column-major-pack fallback**
   (vs. a conjugation trick) is the preferred approach.
4. Q5: confirm **Option A only** for now (helpers), defer B/C.

---

## 4. Coordination notes

- **Do not edit `src/gsl.zig` / `src/gsl_sf.zig` / gsl-fft work** — other agents
  active. LAPACK work stays isolated to `src/lapack.zig` (+ the new
  `src/lapack_shim.{c,h}` and its `build.zig` line).
- The shim is the first LAPACK C source; mirror the TBLIS `addCSourceFile`
  pattern. Verify `<vecLib/lapack.h>` exposes the needed complex symbols at
  C-compile time (a missing symbol fails the build immediately — a feature).
- Watch Zig name-shadowing (`c` cImport handle; primitives `u0`/`u1`). In
  `EigResult` the complex type is already aliased `Cx` to free `C` for the
  column-axis label.
