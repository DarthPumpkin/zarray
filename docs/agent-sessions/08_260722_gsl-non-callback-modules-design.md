# GSL Non-Callback Modules — Design Sketch (Session 8)

**Status:** *Design ratified; **all five modules implemented, tested,
integrated.*** interpolation/splines (`gsl.interp`), histograms
(`gsl.histogram`), polynomials (`gsl.poly`), digital filters (`gsl.filter`),
quasi-random sequences (`gsl.qrng`). A **clean** `zig build test` → **460 pass**
(see the cache-staleness note in §D). A follow-up hardening pass added the
divided-difference/Hermite poly API, the impulse outlier mask, a non-empty-coeffs
guard, per-module *Omissions* docs, and — crucially — caught & fixed a by-value
`gsl_complex` ABI bug (see §D "Hardening pass").

Sketches the API for every **non-callback** GSL module worth binding next,
ordered by value, to surface the design decisions they share **before**
implementing any of them.

**Scope (this pass):** interpolation/splines, histograms, polynomials, digital
filters, quasi-random sequences. Deliberately excludes callback-based modules
(integration, roots, minimization, ODE, multifit, monte, deriv) — those come
later behind a shared `gsl_function` bridge.

**Prior art it must match:** `rand`/`stats`/`rstat`/`movstat` in `src/gsl.zig`,
`src/gsl_fft.zig`, `src/gsl_sf.zig`. Established conventions:
- Workspace-owning types = `init()`/`deinit()` structs, `init` returns
  `error{OutOfMemory}!Self` (`rand.Rng`, `movstat.Window`).
- Algorithm selector = Zig enum → `const gsl_*_type *` pointer (`rand.Generator`).
- Each subsystem = its own file, own `@cImport`, **own module-specific `Error`
  set**; re-exports shared vocab (`Strided`/`StridedMut`/handler/`strerror`) from
  `gsl.zig`. (`gsl_fft.zig`, `gsl_sf.zig` both do this.)
- Complex = `std.math.Complex(T)` (packed layout = `gsl_complex`), per `gsl_fft`.
- Strided view iff GSL takes `gsl_vector`/exposes a stride; else contiguous
  slice (per sf `_array` vs `movstat`).

---

## Part A — Per-module API sketches (ordered by value)

### A1. Interpolation & splines (`gsl_spline`/`gsl_interp`) — highest value

Bind **`gsl_spline`** (which stores its own copy of the x/y data, so `eval`
takes only `x`) rather than the lower-level `gsl_interp` (which re-takes the
arrays on every call). Types: linear, polynomial, cspline(+periodic),
akima(+periodic), steffen.

```zig
pub const interp = struct {
    pub const Type = enum { linear, polynomial, cspline, cspline_periodic,
                            akima, akima_periodic, steffen };

    /// Optional O(1) lookup cache for repeated evals (gsl_interp_accel).
    pub const Accel = struct { ptr, init()/deinit()/reset() };

    pub const Spline = struct {
        ptr: *c.gsl_spline,
        // fused alloc(type,n)+init(xa,ya): xs.len == ys.len, sorted xs
        pub fn init(t: Type, xs: []const f64, ys: []const f64) Error!Spline;
        pub fn deinit(self) void;
        pub fn reinit(self, xs: []const f64, ys: []const f64) Error!void; // reuse alloc
        pub fn name(self) [:0]const u8;
        pub fn minSize(self) u32;
        // eval family; accel optional (?*Accel or bundled — see D8)
        pub fn eval(self, x: f64) Error!f64;
        pub fn evalDeriv(self, x: f64) Error!f64;
        pub fn evalDeriv2(self, x: f64) Error!f64;
        pub fn evalInteg(self, a: f64, b: f64) Error!f64;
    };
};
```

### A2. Histograms (`gsl_histogram` + `gsl_histogram2d`) — Tier 1, no callbacks

```zig
pub const histogram = struct {
    pub const Histogram = struct {
        ptr: *c.gsl_histogram,
        pub fn init(bins: usize) Error!Histogram;                 // alloc (unset ranges)
        pub fn initUniform(bins: usize, min: f64, max: f64) Error!Histogram;
        pub fn initWithRanges(ranges: []const f64) Error!Histogram; // len = bins+1
        pub fn deinit(self) void;
        pub fn clone(self) Error!Histogram;
        pub fn reset(self) void;
        pub fn setRangesUniform(self, min: f64, max: f64) Error!void;
        pub fn setRanges(self, ranges: []const f64) Error!void;
        // fill
        pub fn increment(self, x: f64) Error!void;
        pub fn accumulate(self, x: f64, weight: f64) Error!void;
        // query
        pub fn bins(self) usize;
        pub fn get(self, i: usize) f64;
        pub fn getRange(self, i: usize) Error!struct { lower: f64, upper: f64 };
        pub fn find(self, x: f64) ?usize;
        pub fn min(self) f64;   pub fn max(self) f64;
        pub fn maxVal/minVal/maxBin/minBin/mean/sigma/sum(self) ...;
        // whole-histogram arithmetic (in place on self)
        pub fn add/sub/mul/div(self, other: Histogram) Error!void;
        pub fn scale(self, s: f64) void;  pub fn shift(self, s: f64) void;
        pub fn equalBins(self, other: Histogram) bool;
    };
    /// Sample the histogram as an empirical distribution (gsl_histogram_pdf).
    pub const Pdf = struct {
        ptr: *c.gsl_histogram_pdf,
        pub fn init(h: Histogram) Error!Pdf;
        pub fn deinit(self) void;
        pub fn sample(self, r: f64) f64;    // r in [0,1) (e.g. rng.uniform())
    };
    // Histogram2d/Pdf2d mirror the above with (x,y) pairs; include in same pass.
};
```

### A3. Polynomials (`gsl_poly`) — easy, high utility

```zig
pub const poly = struct {
    pub fn eval(coeffs: []const f64, x: f64) f64;               // Horner (infallible)
    pub fn evalDerivs(coeffs: []const f64, x: f64, out: []f64) Error!void;

    /// Real roots only; n = number of real roots.
    pub fn solveQuadratic(a: f64, b: f64, c: f64) struct { n: usize, roots: [2]f64 };
    pub fn solveCubic(a: f64, b: f64, c: f64) struct { n: usize, roots: [3]f64 };
    /// Complex roots (std.math.Complex(f64), packed = gsl_complex).
    pub fn solveQuadraticComplex(a, b, c) [2]Complex(f64);
    pub fn solveCubicComplex(a, b, c) [3]Complex(f64);

    /// General degree-(n-1) polynomial root finding (companion matrix).
    pub const ComplexSolver = struct {
        ptr: *c.gsl_poly_complex_workspace,
        pub fn init(n: usize) Error!ComplexSolver;   // n = #coeffs
        pub fn deinit(self) void;
        pub fn solve(self, coeffs: []const f64, roots: []Complex(f64)) Error!void; // roots.len == n-1
    };
    // Newton divided-difference (gsl_poly_dd_*) — optional, defer unless wanted.
};
```

### A4. Digital filters (`gsl_filter`) — follow-on to `movstat`, no callbacks

Same `gsl_vector` shape as `movstat`, so it reuses the borrowed-view helpers.
Its `End` enum has the *same underlying values* as `movstat.End`.

```zig
pub const filter = struct {
    pub const End = enum { pad_zero, pad_value, truncate }; // == movstat values
    pub const Scale = enum { mad, iqr, sn, qn };            // impulse scale estimator

    pub const Gaussian = struct {
        ptr: *c.gsl_filter_gaussian_workspace,
        pub fn init(k: usize) Error!Gaussian;
        pub fn deinit(self) void;
        pub fn apply(self, end: End, alpha: f64, order: usize,
                     x: Strided(f64), y: StridedMut(f64)) Error!void;
        pub fn kernel(alpha: f64, order: usize, normalize: bool,
                      out: StridedMut(f64)) Error!void;      // static
    };
    pub const Median = struct { init(k)/deinit/apply(end,x,y) };          // rank
    pub const RecursiveMedian = struct { init(k)/deinit/apply(end,x,y) }; // rmedian
    pub const Impulse = struct {
        ptr,
        pub fn init(k: usize) Error!Impulse;
        pub fn deinit(self) void;
        pub fn apply(self, end: End, scale: Scale, t: f64, x: Strided(f64),
                     xmedian: StridedMut(f64), xsigma: StridedMut(f64),
                     y: StridedMut(f64)) Error!usize;         // returns #outliers
    };
};
```

### A5. Quasi-random sequences (`gsl_qrng`) — easy; replaces the reserved stub

```zig
pub const qrng = struct {  // replaces the current @compileError placeholder
    pub const Type = enum { sobol, halton, reverse_halton, niederreiter2 };
    pub const Sequence = struct {
        ptr: *c.gsl_qrng,
        pub fn init(t: Type, dim: u32) error{OutOfMemory}!Sequence;
        pub fn deinit(self) void;
        pub fn clone(self) error{OutOfMemory}!Sequence;
        pub fn reset(self) void;
        pub fn name(self) [:0]const u8;
        pub fn get(self, out: []f64) Error!void;   // out.len == dim
        pub fn saveState(self, buf: []u8) []u8;    // via size()/state(), like rand.Rng
        pub fn loadState(self, snapshot: []const u8) void;
    };
};
```

**Lower value, likely skip:** `gsl_sort` (overlaps `std.sort` + `stats` already
has select/median), `gsl_permutation`/`gsl_combination`/`gsl_multiset` (niche),
`gsl_chebyshev`/`gsl_deriv` (callback-based, deferred with the rest).

---

## Part B — Shared design decisions (the point of this pass)

Recommended default in **bold**; ★ = wants maintainer ratification.

- **D1. File layout.** One file per subsystem with its own `@cImport`, matching
  `gsl_fft`/`gsl_sf`: `gsl_interp.zig`, `gsl_histogram.zig`, `gsl_poly.zig`,
  `gsl_filter.zig`. **qrng** is small + rng-adjacent → flesh out in place in
  `gsl.zig` (replacing the reserved stub). ★ (granularity; whether to fold tiny
  `poly` into `gsl.zig` too).
- **D2. Workspace pattern.** All owning types = `init`/`deinit`, `init` returns
  `error{OutOfMemory}!Self` (or the module `Error` when it also validates
  inputs). No Zig allocator (GSL owns its allocations) — same rationale as
  `rstat`/`movstat`.
- **D3. Type selector = enum → GSL type pointer.** `interp.Type`, `qrng.Type`
  map to `const gsl_*_type *` via a private `typePtr()`, exactly like
  `rand.Generator`. `filter.Scale` maps to `gsl_filter_scale_t`.
- **D4. Per-module `Error`; shared handler/views.** Each file defines its own
  minimal `Error` set (fft/sf precedent) and re-exports `Strided`/`StridedMut`/
  `disableDefaultErrorHandler`/`strerror` from `gsl.zig`. Auto-install the
  non-aborting handler on first fallible use. (Confirms: **do not** consolidate
  the error sets — per-module is the house style.)
- **D5. Fuse alloc+init, add `reinit`/`set*` for reuse.** GSL separates
  allocation from data-loading and lets you re-load same-size data without
  realloc (perf). Fuse into one `init(...)` for ergonomics; expose `reinit`
  (spline) / `setRanges` (histogram) for the reuse path. Recurs in interp +
  histogram.
- **D6. Array-shape rule.** Contiguous `[]const f64`/`[]f64` where GSL takes a
  bare pointer+size (interp, poly, histogram ranges); `Strided`/`StridedMut`
  where GSL takes `gsl_vector` (filter). Length-check and return `BadLength`.
- **D7. `Error!f64` eval on the `_e` form.** Where GSL offers `_e`+natural forms
  (interp/spline eval), wrap the `_e` form as `Error!f64` (sf precedent); the
  bare `f64` stays reachable via `c`.
- **D8. Optional stateful companion (accel). ★** `gsl_interp_accel` is a mutable,
  non-thread-safe eval cache. Options: (a) `Spline` owns one internally and
  `eval` uses it (simplest; single-thread only) + an `evalWith(x, accel)` escape
  hatch for concurrent readers; (b) always pass `?*Accel`. **Lean (a).**
- **D9. Multi-real-output → `struct { n: usize, roots: [N]f64 }`.** For poly
  quadratic/cubic real roots (precedent: `movstat.minMax`, `sf.elljac`).
- **D10. Complex = `std.math.Complex(f64)`.** Packed layout matches `gsl_complex`
  and `gsl_fft`; `[]Complex(f64)` casts straight through. Bind the complex poly
  solvers in the first pass (that's poly's main value). ★ (include vs defer).
- **D11. Skip `FILE*` serialization; expose buffer save/load + clone.** Omit the
  `fread`/`fwrite`/`fprintf` forms (histogram, qrng) as non-idiomatic (reach via
  `c`); expose `clone`/`reset`, and for qrng a `saveState`/`loadState` mirroring
  `rand.Rng`. (rand already omits the `FILE*` forms.)
- **D12. Shared `gsl_vector` view helpers. ★** `filter` needs
  `constVectorView`/`mutVectorView`, currently file-private in `gsl.zig`. Either
  promote them to shared (internal, re-exported) helpers for `gsl_filter.zig`, or
  keep `filter` inside `gsl.zig` next to `movstat`. **Lean: promote the helpers**
  (pub-but-underscored) so `filter` can live in its own file.

---

## Part C — Proposed build order

1. **interp/splines** (`gsl_interp.zig`) — highest value, zero new infra, only
   new decision is D8 (accel).
2. **histograms** (`gsl_histogram.zig`) — Tier 1, no callbacks, exercises D5.
3. **poly** (`gsl_poly.zig` or in `gsl.zig`) — small, exercises D9/D10 (complex).
4. **filter** (`gsl_filter.zig`) — reuses movstat infra, exercises D12.
5. **qrng** (flesh out in `gsl.zig`) — small, closes an existing reserved stub.

Open decisions needing a call before coding: **D1** (file granularity), **D8**
(accel bundling), **D10** (bind complex poly solvers now?), **D12** (promote the
vector-view helpers).

---

## Part D — Resolutions & implementation log

### Decisions (ratified by maintainer)

- **D1 → one file per module**, nothing new implemented in `gsl.zig` (only the
  `pub const <mod> = @import(...)` re-export glue, matching `fft`/`sf`). The
  maintainer also plans to later extract `rand`/`stats`/`rstat`/`movstat` out of
  `gsl.zig` into their own files, leaving it a thin aggregator.
- **D8 → `eval` always takes an explicit `accel: ?Accel`.** No real drawback: a
  `Spline` is `const` during eval and only the `Accel` mutates, so an explicit
  per-thread accel is the *correct* thread-safety model (bundling one inside
  `Spline` would create a shared-mutable data race). `null` covers the one-off
  case (GSL falls back to binary search).
- **D10 → yes**, bind the complex poly solvers with `std.math.Complex(f64)`.
- **D12 → yes**, share the `gsl_vector` view helpers internally (needed when
  `filter` lands). Since each file has its own `@cImport` (distinct `gsl_vector`
  types), the shared helper must be **generic over the target vector type**:
  `fn view(comptime Vec: type, s) Vec` — the struct literal coerces to each
  caller's `c.gsl_vector`. To do when implementing `filter`.
- **D13 (new convention) → never cross the `@cImport` ABI with a by-value
  `gsl_complex`.** Zig's `@cImport` does **not** reliably reproduce the C ABI for
  functions that pass or *return* `gsl_complex` by value (16-byte HFA); such a
  call silently returns garbage (see the `evalComplex` bug in the Hardening
  pass). Rule for all current and future GSL bindings:
    1. If a GSL routine takes/returns `gsl_complex` **by value**, do **not** call
       it through `c`; reimplement it natively over `std.math.Complex(f64)`
       (e.g. `poly.evalComplex`/`evalComplexPoly` are native Horner loops), or
       route through an equivalent **pointer-based** form.
    2. **Pointer-based** complex I/O is safe: `gsl_complex *` outputs (the poly
       `solve*Complex` solvers), packed `double[]` arrays (`gsl_fft`, and
       `gsl_poly_complex_solve` via `gsl_complex_packed_ptr`), and
       `gsl_sf_result *` structs. Prefer these.
    3. **Validate complex results against nonzero expected values**, never
       magnitude-only — a magnitude≈0-at-the-roots check is exactly what let the
       `evalComplex` bug hide. Add a constant/off-root oracle case.
  Audit result (this session): the *only* by-value `gsl_complex` site in the
  whole GSL surface was `poly.evalComplex`/`evalComplexPoly` (now native). `fft`
  (packed `double[]`), `sf` (omits complex; pointer `gsl_sf_result`), and the
  poly pointer solvers are all safe. Mirrors the rationale behind
  `lapack_shim.c`, which exists to tame the same complex-ABI problem for
  Accelerate's LAPACK.

### Implemented so far

- **`gsl.interp`** (`src/gsl_interp.zig`): `Type` (7 methods → `gsl_interp_*`),
  `Accel` (init/deinit/reset), `Spline` (init/deinit/reinit/name +
  eval/evalDeriv/evalDeriv2/evalInteg, each `Error!f64` taking `accel: ?Accel`).
  Own `@cImport`, own `Error` set, re-exports `disableDefaultErrorHandler`/
  `strerror` from `gsl.zig`, and calls the now-`pub gsl.ensureHandler()`.
  Re-exported as `gsl.interp`. 7 tests (line/cspline oracle checks, domain
  error, length checks, reinit, accel==null equivalence, every-type sweep).
- **`gsl.ensureHandler` made `pub`** in `gsl.zig` (shared one-flag handler
  install) so submodules reuse a single install guard instead of each carrying
  their own. (`gsl_sf.zig` still has its own; not refactored.)
- Gotcha found: `gsl_spline_alloc` returns NULL below a type's `min_size`
  (indistinguishable from OOM), so `Spline.init` pre-checks `xs.len <
  t.minSize()` and returns `BadLength`.
- **`gsl.histogram`** (`src/gsl_histogram.zig`): 1-D `Histogram` +
  empirical-sampling `Pdf`, and 2-D `Histogram2d` + `Pdf2d`. Fused
  alloc+range-set into `init`/`initUniform`/`initWithRanges` (D5), with
  `setRanges`/`setRangesUniform`/`reset`/`clone` for reuse. Full query surface
  (`bins`/`get`/`getRange`/`find`/min/max/maxVal/minVal/maxBin/minBin/mean/
  sigma/sum), in-place arithmetic (`add`/`sub`/`mul`/`div`/`scale`/`shift`),
  and `equalBins`. Own `@cImport`, own `Error`, re-exports the shared handler.
  `FILE*` forms deliberately omitted (D11). Named `Range`/`BinIndex`/`Point`
  return structs. 9 tests. Re-exported as `gsl.histogram`.
  - Zig gotcha: a fn **parameter may not share a name with a method** of the
    same struct (e.g. `initUniform(bins,…)` vs the `bins()` method); renamed
    the params (`n_bins`/`lo`/`hi`/`x_lo`…). Also two anonymous `struct {…}`
    return types are **distinct types**, so `Pdf2d.sample` returns a *named*
    `Point`.
- **`gsl.poly`** (`src/gsl_poly.zig`): `eval` (Horner), `evalComplex`
  (real poly / complex point), `evalComplexPoly` (complex poly / complex
  point), `evalDerivs`; closed-form `solveQuadratic`/`solveCubic` returning
  `{ n, roots }` (D9, via named `RealRoots2`/`RealRoots3`) and the complex
  `solveQuadraticComplex`/`solveCubicComplex`; and the general-degree
  `ComplexSolver` (companion-matrix workspace). Complex = `std.math.Complex(f64)`
  (D10) with `toComplex`/`fromComplex` bit-compatible converters; the
  `[]Complex(f64)` roots slice `@ptrCast`s straight to GSL's packed `double*`.
  Own `@cImport`, own `Error` (adds `Failure` = `GSL_EFAILED`). 8 tests.
  Re-exported as `gsl.poly`.
  - Naming: to avoid shadowing the module-level `pub const c` (the cimport),
    root-solver coefficients are named by power — `c2·x² + c1·x + c0` (and the
    *monic* cubic `x³ + c2·x² + c1·x + c0`, matching GSL's convention).
- **`gsl.filter`** (`src/gsl_filter.zig`): the digital-filter follow-on to
  `movstat`. `Gaussian` (smoothing/derivative + static `kernel`), `Median`,
  `RecursiveMedian`, `Impulse` (returns the outlier count). `End` mirrors
  `movstat.End` (same underlying values; a test asserts this); `Scale` selects
  the impulse robust-scale estimator (`mad`/`iqr`/`sn`/`qn`). Same
  `Strided`/`StridedMut` shape as `movstat`, fed zero-copy via the now-shared
  generic view helpers (D12 realized — see below). Own `@cImport`, own `Error`.
  7 tests. Re-exported as `gsl.filter`.
- **`gsl.qrng`** (`src/gsl_qrng.zig`): replaces the `@compileError` placeholder
  in `gsl.zig`. `Type` (sobol/halton/reverse_halton/niederreiter2 →
  `gsl_qrng_*`, with `maxDimension`) and `Sequence`
  (init/deinit/clone/reset/name/dimension/get + `saveState`/`loadState`/
  `stateSize` mirroring `rand.Rng`). `init` validates `dim ∈ 1..=maxDimension`.
  Own `@cImport`, own `Error`. 6 tests. Re-exported as `gsl.qrng`.
  - Test gotcha: `gsl_qrng_size` (state blob) is large for niederreiter2 — the
    round-trip test sizes its buffer from `stateSize()` (via `testing.allocator`)
    rather than a fixed stack array.

### D12 realized — shared generic `gsl_vector` view helpers

Promoted the file-private `constVectorView`/`mutVectorView` in `gsl.zig` to
**public generic** `constVectorViewOf(comptime Vec, Strided(f64))` /
`mutVectorViewOf(comptime Vec, StridedMut(f64))`. The struct literal coerces to
whichever caller's `c.gsl_vector` is passed (each submodule's `@cImport` yields a
*distinct* `gsl_vector` type). `movstat`'s private helpers now delegate to them;
`gsl_filter.zig` calls them with its own `c.gsl_vector`. Zero-copy borrowed
views (`block = null, owner = 0`) as before.

### Cache-staleness gotcha (important, tooling)

Zig 0.16's incremental cache went **stale** mid-session: after adding
`gsl_poly.zig`, `zig build test` reported bogus errors in the *untracked*
`src/lapack.zig` (e.g. `const Cx = Complex(f64)` at a line that actually reads
`const Cf = …`, and a phantom `.scale` method) — the reported source lines did
not match the file on disk. `rm -rf .zig-cache && zig build test` cleared it and
all tests passed. **Takeaway:** when errors reference line/symbol text that
doesn't match the current file (especially in files you didn't touch), suspect
the incremental cache and do a clean build before chasing the "bug." The clean
count (440) is the authoritative one; the earlier incremental 374/413/421
numbers were likely under-reported by partial test discovery.

### Test-discovery gap found & fixed (important)

`zig build test` only collects tests from *analyzed* files. `gsl.zig` re-exports
submodules via `pub const sf/fft/interp = @import(...)` but **never referenced
them**, and it can't `refAllDecls(@This())` because `constants`/`qrng` are
`@compileError` placeholders. Result: **`gsl_sf.zig` (13) and `gsl_fft.zig` (18)
tests had silently stopped running** (regressed when their direct `libs.gsl_sf`/
re-exports were removed in earlier sessions). Fixed with a discovery `test { _ =
fft; _ = sf; _ = interp; }` block in `gsl.zig`. Test count jumped 374 → 413
(+7 interp, +31 recovered, +1 the discovery block). New submodule files must be
added to this block.

### Build order status

1. ✅ interp/splines — done.
2. ✅ histograms (`gsl_histogram.zig`) — done (1-D + 2-D + PDFs).
3. ✅ poly (`gsl_poly.zig`) — done (D9 real-root structs, D10 complex solvers).
4. ✅ filter (`gsl_filter.zig`) — done (D12 generic shared view helper realized).
5. ✅ qrng (`gsl_qrng.zig`) — done (replaced the reserved `@compileError` stub).

**All five modules complete.** Clean `zig build test` → **460 pass**. The only
remaining `@compileError` placeholder in `gsl.zig` is `constants` (intentional,
points users at `std.math`), so the explicit `test { _ = fft; _ = sf; _ = interp;
_ = histogram; _ = poly; _ = filter; _ = qrng; }` discovery block is still
required. Next natural step (future session): the callback-based modules
(integration, roots, minimization, ODE, multifit, monte, deriv) behind a shared
`gsl_function` bridge, and/or the planned extraction of `rand`/`stats`/`rstat`/
`movstat` out of `gsl.zig` into their own files.

### Hardening pass (omissions review)

After the five modules landed, a review of *what was left unbound* drove a
follow-up pass:

- **poly non-empty guard (the "a" fix).** `eval`/`evalComplex`/`evalComplexPoly`/
  `ddEval` now assert a non-empty coefficient slice (GSL's Horner reads
  `c[len-1]`; an empty slice was UB). `evalDerivs`/`dd*` return `BadLength`.
- **poly divided differences (the "c" fix).** Added `ddInit`/`ddEval`/`ddTaylor`/
  `ddHermiteInit` — Newton divided-difference + Hermite interpolation, kept
  allocation-free via caller-supplied buffers (matches poly's pure-function
  ethos). Hermite is evaluated with the doubled abscissae `za`, not `xs`.
- **filter impulse mask (the "b" fix).** `Impulse.apply` gained an optional
  `outliers: ?StridedMut(i32)` that receives GSL's per-sample `ioutlier` flag
  vector; the `gsl_vector_int` view is stack-built inline (the shared f64 view
  helpers don't cover the int vector).
- **`gsl_complex` by-value ABI bug found & fixed.** The new poly constant-eval
  test exposed that `gsl_poly_complex_eval`/`gsl_complex_poly_complex_eval` —
  which pass/return `gsl_complex` **by value** — are miscompiled through Zig's
  `@cImport` (returned ~0 instead of the constant term). The earlier root-finder
  test had only checked `magnitude ≈ 0` at the roots, so it never validated a
  *nonzero* result and masked the bug. Fix: reimplement both as native Zig
  Horner loops over `std.math.Complex(f64)` (identical algorithm, no by-value
  ABI). **Lesson: the pointer-based complex solvers are fine (GSL writes through
  `gsl_complex *`); only by-value `gsl_complex` args/returns are unsafe via
  `@cImport` — wrap those natively.**
- **Omissions documented** in every module's top-of-file docstring (interp,
  histogram, poly, filter, qrng): low-level `gsl_interp`/bare non-`_e` forms;
  `FILE*` serialization; `*_memcpy`-into-existing (only `clone`); `get`
  returning 0 out of range; the by-value complex-eval caveat.
- **Coverage expanded**: poly dd/Hermite + constant-poly boundary; filter
  impulse mask across all four `Scale` estimators (mad/iqr/sn/qn) with
  count==flags cross-check; histogram `sub`/`mul`/`div`/`minVal`/`minBin`/`mean`/
  `sigma` and 2-D `getYRange`/`xmin..ymax`/`minBin`/`shift`; qrng `maxDimension`
  bounds. Net +7 tests over the 453 baseline (some replaced) → 460.

### Coverage gap sweep (follow-up)

A per-module audit of *untested public branches* (not just untested modules)
found a handful of gaps in otherwise-strong coverage; all were closed:

- **filter: Gaussian derivative orders.** Every prior `apply`/`kernel` call used
  `order = 0`. Added an order-1 filter test (constant slope on a ramp; zero on a
  flat signal) and a derivative-kernel test (antisymmetric, zero centre tap,
  sums to zero).
- **filter: `End.pad_zero` never flowed through an `apply`.** Added a median
  filter test on the `.pad_zero` path (interior of a constant signal intact).
- **poly: `evalComplex` (real coeffs at a complex point)** was only hit at the
  constant boundary. Added `p(x)=1+2x+3x²` at `z=1+i` → `3+8i`.
- **poly: error/edge branches.** Added `evalDerivs([]empty) → BadLength` and the
  zero-discriminant quadratic (`(x−3)²` → `n==2`, both roots 3).

Net +6 tests → **474 passing** (`zig build test --summary all`).

### Follow-up structural work (post-design, completed)

Three of the "recommended next steps" from this design were carried out after
the modules landed (the first two were handed to sub-agents via written
handovers under `/tmp/gsl-handover/`; results reviewed and accepted):

- **`gsl.zig` module split (pure refactor).** Extracted the four inline
  chapters (`rand`/`stats`/`rstat`/`movstat`) and their tests into
  `gsl_rand.zig`/`gsl_stats.zig`/`gsl_rstat.zig`/`gsl_movstat.zig`, re-exported
  from `gsl.zig` (public paths unchanged; `stats` re-exported as
  `stats_ref.stats`). Shared infra stayed in the hub; `gsl.zig`'s own `c` was
  trimmed to just `gsl_errno.h`. No behavior change — clean-cache rebuild held
  at 474/474.
- **Container chapters implemented.** Added `gsl_sort.zig` (typed `sort(T)`:
  sort/sort2/sortIndex/smallest/largest + index forms), `gsl_permutation.zig`
  (owning `Permutation` + the `gsl_permute` family via `permute(T)`),
  `gsl_combination.zig`, and `gsl_multiset.zig`. `next`/`prev`/`valid` surface
  exhaustion as `false` (not an error): `next`/`prev` typed `Error!bool`. Two
  shared symbol-mapping helpers (`numericModuleStem`/`numericModuleInfix`) were
  hoisted into `gsl.zig` and now back both `stats` and `sort`. → **486 passing**.
- **Hub docstring rewritten.** `gsl.zig`'s top-level `//!` doc now describes its
  role as the aggregating hub: all 15 chapters listed, a "shared
  infrastructure" section, and the corrected note that each chapter owns its own
  `c` (the hub's `c` is only `gsl_errno.h`).

### Remaining known omissions (deliberate)

- `FILE*` serialization forms (all modules).
- `*_memcpy`-into-an-existing-instance (histogram/qrng) — `clone` covers the
  allocate-new case.
- Low-level `gsl_interp` layer and bare non-`_e` spline eval forms.
- Broad chapters still out of scope: **callback modules** (integration, roots,
  min, ODE, multifit, monte, deriv, Chebyshev) — now sketched in Session 11
  (`11_260722_gsl-callback-bindings-design.md`) and the next thing to build.
  (The `gsl_sort`/`permutation`/`combination`/`multiset` container chapters,
  previously listed here, are now implemented — see "Follow-up structural
  work" above.)
