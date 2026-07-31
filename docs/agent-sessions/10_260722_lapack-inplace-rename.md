# LAPACK Bindings — Session 10 (default/`*Inplace` rename)

**Status:** **Complete.** This session is a naming/ergonomics pass on
`src/lapack.zig` following the Q6/Q5 implementation (session 9). No numerical
behavior changes.

**Tests (latest run):**
- `zig test src/lapack.zig src/lapack_shim.c -framework Accelerate -lc` →
  **All 134 tests pass** (127 prior + 7 new coverage tests).
- `zig build test` → **passes**.
- `diagnostics` on `src/lapack.zig` → clean.

---

## 1. Motivation

The maintainer flagged that `lstsq` and `lstsqAlloc` **both take an allocator**,
so the `Alloc` suffix was the wrong axis to name on: it implied "this is the one
that allocates" when in fact *both* allocate (the base for LAPACK workspace, the
`*Alloc` variant *additionally* for the input copy). The real distinction is
**input mutation**: the base overwrote the caller's `a` (used it as scratch);
`*Alloc` factored a private copy and left `a` untouched.

This convention spanned **8 pairs** in `src/lapack.zig`, not just `lstsq`.

## 2. Decision

**Flip the default** so the plain name is the safe, input-preserving one, and the
destructive variant is explicitly flagged `Inplace` (maintainer chose `Inplace`
over `InPlace` for typing ease):

| destructive (overwrites `a`) | input-preserving default |
|---|---|
| `detInplace` | `det` |
| `lstsqInplace` | `lstsq` |
| `qrInplace` | `qr` |
| `eigSymInplace` | `eigSym` |
| `eigInplace` | `eig` |
| `svdInplace` | `svd` |
| `eigVectorsInplace` | `eigVectors` |
| `svdVectorsInplace` | `svdVectors` |

Rationale: principle of least surprise — `svd(a)` no longer silently clobbers
`a`; mutation is opt-in via `svdInplace`. Both forms still take an `allocator`
(for LAPACK workspace), so the allocator is *not* the distinguishing feature —
the name is.

`eigSymVectors` is inherently copy-based (always preserves `a`) and has **no**
in-place form, so it was left unchanged.

## 3. What landed

### Mechanical rename (`src/lapack.zig`)
Done with two `perl` passes using `\b` word boundaries:
1. destructive `X(` → `XInplace(` — word boundaries avoided corrupting
   `xorgqr(` / `zarray_cungqr(` (which end in `qr(`); `det` was restricted to its
   actual call sites so the **math** comment `det(Aᵀ) = det(A)` stayed intact.
2. `XAlloc` → `X`.

Also updated by hand:
- Docstrings on the new defaults reframed from the self-referential
  "Input-preserving `svd`:" to "Input-preserving (default) variant of
  `svdInplace`: …"; the `*Inplace` docstrings' cross-refs now correctly read
  "Use `svd` to preserve `a`".
- The `eigSymVectors` note "…there is no `eigSymVectorsAlloc`" →
  "…no in-place `eigSymVectorsInplace` variant".
- Four preservation-test comments that read "a plain `X` would overwrite" →
  "`XInplace` would overwrite".
- **Module header:** added a new bullet *"Input-preserving default, `*Inplace`
  opt-out"* documenting the pair convention and that the allocator does not
  distinguish the two; corrected the vector-variant mechanics to name the
  `*Inplace` forms as the ones that use `a` as scratch. Restored the
  "No hidden allocation in the strict routines" bullet (had been accidentally
  dropped) with `detInplace` in the no-alloc list.

### Test retitling
Destructive-path tests (those that call an `*Inplace` routine) were retitled
`test "XInplace: …"` so the title names the exact function under test (26 tests).
The eight input-preserving tests keep their plain-name titles
(`test "X: preserves …"`), which now correctly describe the default.

### Header accuracy fix
The stale claim that "`qr`/`lstsq` currently require column-major input" (a
session-1 limitation, lifted in session 2) was corrected to describe the actual
any-layout behavior and where the zero-copy absorption lives (`*Inplace` forms).
The LU-family "no copy" line now names `detInplace` (the no-alloc, in-place form)
rather than `det` (which copies).

## 4. Notes / follow-ups
- Historical session logs under `docs/agent-sessions/` were intentionally left
  untouched (they are a record of past work); only `src/lapack.zig` and this new
  report changed.

## 5. Test-coverage pass

Audited coverage and found the double-precision surface well covered but three
gaps. Outcome:

### Added (7 tests)
Everything below tests **our** code (dispatch `switch(T)` arms, C-shim casts, our
complex arithmetic) — not LAPACK's math:

- **Single precision (`f32`)** — 3 tests (`LU family`, `eigSym/eig/svd`,
  `qr/lstsq`) that light up the real `s*` dispatch arms, previously unrun (all
  prior numeric tests used `f64`/`Complex(f64)`).
- **`Complex(f32)`** — 3 tests covering every single-precision `zarray_c*` shim
  forwarder (`cgetrf/cgetrs/cgetri`, `cpotrf/cpotrs`, `cheev`, `cgeev`,
  `cgesdd`, `cgels`, `cgeqrf/cungqr`). The shim's compile-time header check
  verifies signatures; these verify the runtime interleaved-buffer casts.
- **`Complex(f64)` det/inv/cholesky** — 1 test covering `zgetri`, `zpotrf/zpotrs`,
  and the complex determinant sign×product branch (where the `Complex.scale`
  latent bug from session 9 hid; now runtime-tested, incl. a pivot-swap case).

All use small closed-form matrices with known answers (`f32` tol `1e-4`).
Added generic test helpers `rowMajorG`/`colMajorG`/`cmaddT` for any scalar type.

### Dropped: “validate the Q5 sizing helpers”
On reflection this isn't a real target — there's no independent oracle we own.
The "ground truth" for a workspace size is LAPACK's documented contract, which
the helpers *copy*. Re-asserting the formula (the existing self-check) is a
tautology; running a decomposition through a helper-sized buffer tests LAPACK,
not us. The load-bearing `rwork`/`iwork` sizers are already covered transitively
(the complex tests would corrupt if they were wrong), which *is* a valid wrapper
test.

### (b) `*MinWork` helpers — removed (Q5 option A)
Surfaced a real inconsistency: the allocator-taking routines query for the
*optimal* `lwork` and allocate that, never the `*MinWork` minimum, so the
advertised "size a `FixedBufferAllocator` from these" workflow can fail with
`error.OutOfMemory`. The `*MinWork` helpers were the caller-facing half of a
zero-alloc feature whose other half (a BYO-buffer `*Into` API) was never built —
so they had no consumer and were a mild footgun.

Decision: **remove them** (`invMinWork`/`lstsqMinWork`/`qrMinWork`/
`eigSymMinWork`/`eigMinWork`) rather than keep speculative public surface. Also
trimmed their assertions from the Q5 test and simplified the Q5 header doc.
Kept the genuinely-useful/load-bearing helpers: `pivotLen` (sizes the caller's
mandatory `ipiv`) and the `rwork`/`iwork` sizers (used internally). If a no-heap
caller ever appears, reintroduce the minima *together with* an `*Into` API
(Q5 option C).

### Not added (lower value, noted for later)
`lstsq` rank-deficient `error.Singular`, the layout-error returns
(`error.NotColumnMajor` / `error.RhsNotColumnMajor`), an `inv` round-trip
(`A·A⁻¹≈I`), and full-mode *rectangular* SVD remain unexercised.
