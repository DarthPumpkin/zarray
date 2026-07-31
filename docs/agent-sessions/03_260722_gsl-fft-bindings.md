# GSL FFT Bindings — Session 3 Report

**Status:** The GSL Fast Fourier Transform chapter is **fully bound, tested, and
integrated**. A new module `src/gsl_fft.zig` wraps all three GSL FFT sub-modules
(complex, real, half-complex) in both precisions (`f32`/`f64`), exposed to
callers as `gsl.fft`. **No open implementation work remains** — the only
unbound symbols are niche or semantically redundant (see [§6](#6-coverage-what-is-and-isnt-bound)).

**Audience:** An agent continuing the GSL binding work in `zarray` (a.k.a.
`ndarray_zig`). This is the next "post-LAPACK target" flagged in session 2's
roadmap (`docs/agent-sessions/02_260721_lapack-spectral-vectors-and-q3-plan.md`,
§8).

**Tests:** `zig test src/gsl_fft.zig -I/opt/homebrew/opt/gsl/include
-L/opt/homebrew/opt/gsl/lib -lgsl -lgslcblas -lc` → **59 pass**.
`zig build test` → **passes** (whole project).

---

## 1. Task

Add FFT bindings for GSL in a **new file**, following the conventions of the
existing bindings (`gsl.zig`, BLAS/LAPACK style): explicit allocation ownership,
no hidden allocation in core calls, strided views as first-class inputs, and
clear comptime/runtime contract checks. Consult the GSL headers/docs directly
rather than asking for input.

---

## 2. What was implemented

New module **`src/gsl_fft.zig`** with three comptime factories that select the
element precision `T` (`f32` or `f64`), one per GSL sub-module:

| Factory | GSL module | Purpose |
|---|---|---|
| `complex(T)` | `gsl_fft_complex[_float]` | in-place complex→complex FFTs over packed `Complex(T)` |
| `real(T)` | `gsl_fft_real[_float]` | forward transform of real data (real→half-complex) |
| `halfcomplex(T)` | `gsl_fft_halfcomplex[_float]` | inverse/backward transform (half-complex→real) + `unpack` |

Each factory exposes:

- **Radix-2 free functions** (`radix2*`) — in place, no scratch space, but the
  length must be a power of two. Complex has
  `radix2Forward`/`radix2Backward`/`radix2Inverse`/`radix2Transform`; real has
  `radix2Forward`; half-complex has `radix2Backward`/`radix2Inverse` plus
  `radix2Unpack`.
- **A mixed-radix `Plan`** — works for *any* length. Owns GSL's trigonometric
  wavetable, and via `init(n)` its own scratch `Workspace` too; the common path
  is one self-contained object with workspace-free calls (`plan.forward(data)`).
  `Plan.factors()` exposes GSL's chosen factorization.
- **`unpack` helpers** — expand a real / half-complex buffer into an ordinary
  `[]Complex(T)` for inspection (half-complex spectra are stored compactly in
  `n` reals because a real signal's spectrum is conjugate-symmetric).

Shared/support surface:

- `Direction` enum (`forward = -1`, `backward = 1`) matching `gsl_fft_direction`;
  only the `transform` entry points take it, the `forward`/`backward`/`inverse`
  shorthands fix it.
- `Error` set (see §4).
- Re-exports from `gsl.zig`: `Strided`, `StridedMut`, `Complex`
  (`std.math.Complex`), `disableDefaultErrorHandler`, `strerror`.

---

## 3. Key design decisions (all settled with the maintainer)

### 3.1 Namespace placement — `gsl.fft`

FFT lives in its own file because it needs the `gsl_fft_*` C headers that
`gsl.zig` deliberately does not `@cInclude`. It is surfaced as
`pub const fft = @import("gsl_fft.zig");` in `src/gsl.zig`, so callers reach it
as `gsl.fft.complex(f64)` etc. A redundant direct `libs.gsl_fft` export was
removed from `root.zig` so there is exactly one access path.

### 3.2 Packed complex = `std.math.Complex(T)` (zero-copy)

GSL's complex routines operate on interleaved `re, im, re, im, ...` floats.
`std.math.Complex(T)` is an `extern struct { re: T, im: T }` with exactly that
layout, so a `[]Complex(T)` passes straight through with no repacking.

### 3.3 Strided views as first-class inputs

Every transform takes a `StridedMut(...)` view (re-exported from `gsl.zig`), so a
single row/column/axis of a larger array is transformed in place without
copying. `stride`/`len` are in **element** units — complex numbers for
`complex(T)`, reals for `real(T)`/`halfcomplex(T)`. `.fromSlice(buf)` covers the
contiguous case.

### 3.4 Workspace ownership & the sharing model (the main API iteration)

A mixed-radix transform needs two things: a **wavetable** (trig tables, per
length *and* transform kind) and a **workspace** (scratch, per length).

- `Plan.init(n)` → the plan owns **both**; `deinit` frees both. This is the
  common, self-contained path.
- `Plan.initWithWorkspace(n, ws)` → the plan owns the wavetable but **borrows** a
  caller-owned `Workspace`; `deinit` frees only the wavetable. This lets a
  `real(T).Plan` (forward) and a `halfcomplex(T).Plan` (inverse) of the same
  length share **one** workspace instead of allocating a redundant second one.

Enabler: GSL's half-complex inverse reuses the **real** workspace type, so
`real(T).Workspace` and `halfcomplex(T).Workspace` are literally the same type.
`initWithWorkspace` validates the borrowed workspace's length and rejects a
mismatch with `error.WorkspaceLengthMismatch`.

### 3.5 Could the workspace use a Zig allocator instead of GSL's? (evaluated, declined)

Investigated at the maintainer's request. Finding:

- The **workspace** struct (`{ size_t n; double *scratch; }`) is fully
  transparent — GSL only reads/writes `scratch` (size `2*n` doubles for complex,
  `n` for real). It **could** be Zig-allocated. The catch is that the scratch
  size is an *internal GSL implementation detail*, not a documented API contract.
- The **wavetable** is **not** just memory: `..._wavetable_alloc(n)` factorizes
  `n` and *computes* all twiddle factors. There is no public GSL entry point to
  initialize a caller-provided wavetable buffer, so it must use GSL's allocator.

**Decision:** because the wavetable can't be Zig-allocated, keeping the workspace
on GSL's allocator too (uniform ownership, no reliance on an undocumented scratch
size) is the cleaner choice. Current behavior retained. (Revisit only if a strong
need for arena/leak-tracked workspace memory appears — the workspace half is
mechanically feasible.)

### 3.6 Error philosophy — pre-validate in Zig, generic fallback for the rest

Checkable contract violations are rejected **before** calling GSL, so they
surface as Zig errors regardless of which GSL error handler is installed
(`NotPowerOfTwo`, `LengthMismatch`, `ZeroLength`, `WorkspaceLengthMismatch`). Any
other nonzero GSL status becomes `error.TransformFailed`. Precision is guarded at
comptime (`@compileError` for `T` ∉ {`f32`,`f64`}). Rich per-code GSL error
mapping was considered and **declined** as over-engineering for this surface.

Note recorded in the docs: GSL's *default* error handler calls `abort()`, so a
genuine internal GSL error never returns a code — install the non-aborting
handler once at startup via `disableDefaultErrorHandler()` for fully fallible
behavior. The pre-validation above keeps the common mistakes off that path.

---

## 4. Public API surface (`src/gsl_fft.zig`, reached as `gsl.fft`)

```
complex(T)      → Value=Complex(T), View, Workspace,
                  radix2Forward/Backward/Inverse/Transform,
                  Plan{ init, initWithWorkspace, deinit,
                        forward, backward, inverse, transform, factors }
real(T)         → Value=T, ComplexValue=Complex(T), View, Workspace,
                  radix2Forward, unpack,
                  Plan{ init, initWithWorkspace, deinit, forward, factors }
halfcomplex(T)  → Value=T, ComplexValue=Complex(T), View, Workspace,
                  radix2Inverse/Backward, radix2Unpack, unpack,
                  Plan{ init, initWithWorkspace, deinit, inverse, backward, factors }
```

Shared: `Direction`, `Error`, `Complex`, `Strided`, `StridedMut`,
`disableDefaultErrorHandler`, `strerror`. `T` ∈ {`f32`, `f64`} (comptime-guarded).

---

## 5. Test status

`zig test src/gsl_fft.zig …` → **59 pass**. `zig build test` → passes. Coverage
includes:

- forward→inverse round-trips (radix-2 and mixed-radix);
- known FFT properties (impulse ↔ flat spectrum; constant ↔ DC impulse;
  `backward` is the unscaled inverse, scaling by `n`);
- explicit-`Direction` `transform` matching `forward`;
- pre-validation: radix-2 rejects non-power-of-two *before* calling GSL; plan
  rejects a data-length mismatch;
- mixed-radix non-power-of-two length, incl. a **large prime** length (exercises
  GSL's slow general O(n²) fallback module);
- `f32` precision round-trip;
- **strided path** (transform every other element in place);
- the **shared-workspace** story: real forward + half-complex inverse sharing one
  `Workspace`, and `initWithWorkspace` rejecting a wrong-length workspace;
- `unpack` agreement (half-complex unpack matches a full complex FFT of the same
  signal; real unpack embeds reals with zero imaginary parts);
- comptime precision guard rejecting unsupported element types.

---

## 6. Coverage: what is (and isn't) bound

All three headers were cross-checked against the bindings. **Every distinct
transform, precision, direction, and the strided/workspace-sharing model is
covered.** The only unbound symbols are niche or redundant (and remain reachable
via the raw `c` handle):

1. **`gsl_fft_complex_radix2_dif_{forward,backward,inverse,transform}`** —
   decimation-in-frequency radix-2 variants. Same results as the bound
   decimation-in-time routines, but they leave output in *bit-reversed* order
   (skipping the final bit-reversal). A specialist optimization for chained
   transforms; not surfaced.
2. **`gsl_fft_complex_memcpy`** — clones a precomputed wavetable. Pure
   convenience; allocate a fresh `Plan` instead.
3. **`gsl_fft_halfcomplex_transform` / `halfcomplex_radix2_transform`** —
   *semantically redundant*: half-complex has no direction parameter, so in GSL
   `..._backward` is just a wrapper over `..._transform`. Already exposed as
   `backward`; only the alternate name is absent.
4. **`long double`** — GSL ships no long-double FFT module, so only `f32`/`f64`
   exist.

Conclusion: the FFT chapter is complete. Item (1) (DIF) is the only genuinely-new
*capability* that could ever be added, and only if a chained-transform
optimization use case appears — a "when needed" item, not a gap.

---

## 7. Files changed this session

- **Added:** `src/gsl_fft.zig` (bindings + docs + 59 tests).
- **Modified:** `src/gsl.zig` — added `pub const fft = @import("gsl_fft.zig");`
  with a doc comment.
- **Modified:** `root.zig` — removed the redundant direct `gsl_fft` export so
  access is solely through `gsl.fft`. (`root.zig` was being edited concurrently
  by other agents for lapack/sf; edits were made carefully and re-validated.)
- No `build.zig` changes.

---

## 8. Coordination notes / gotchas

- The GSL FFT headers are **not** pulled in by `gsl.zig`; `gsl_fft.zig` owns its
  own `c = @cImport({...})` over the eight `gsl_fft_*` headers. Keep that split.
- Standalone test invocation needs the GSL include/lib paths explicitly
  (Homebrew): `-I/opt/homebrew/opt/gsl/include -L/opt/homebrew/opt/gsl/lib -lgsl
  -lgslcblas -lc`. (`pkg-config` is not installed on this machine.) `zig build
  test` handles linking itself.
- `root.zig` / `gsl.zig` are shared with other active agents — coordinate before
  editing; keep FFT logic isolated in `src/gsl_fft.zig`.
- Watch Zig name-shadowing (same as the LAPACK sessions): `c` is the cImport
  handle; primitive type names (`u0`, `u1`, `i8`, …) are illegal identifiers.

---

## 9. Remaining work

**None required.** The FFT bindings are complete and validated. Optional
follow-ups, only if desired:

- Add usage examples/docs contrasting the default `init` path vs. the
  `initWithWorkspace` sharing path.
- Bind the DIF radix-2 variants if a chained-transform optimization ever needs
  bit-reversed output.

Per session 2's roadmap (§8), the next post-LAPACK GSL targets after FFT are
interpolation/splines, fitting/regression, and the callback-driven families
(integration/roots/minimization/ODE).
