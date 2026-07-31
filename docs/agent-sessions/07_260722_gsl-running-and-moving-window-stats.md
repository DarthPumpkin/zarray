# GSL Running & Moving-Window Statistics — Session 7 Report

**Status:** **Implemented, tested, and integrated.** `gsl.rstat` (running/
streaming statistics) and `gsl.movstat` (moving-window statistics) are bound in
`src/gsl.zig` as two new top-level namespaces alongside the descriptive
`stats(T)`. A shared `Error`/`check`/`ensureHandler` error layer was added to
`gsl.zig` (it previously had none), plus borrowed-`gsl_vector` view helpers.
`zig build test` passes (373 tests; 5 new).

**Audience:** An agent continuing the GSL binding work in `zarray` (a.k.a.
`ndarray_zig`). Complements the descriptive stats / RNG bindings in
`src/gsl.zig`, the FFT bindings in `src/gsl_fft.zig`
(`docs/agent-sessions/03_...`), and the special-function bindings in
`src/gsl_sf.zig` (`docs/agent-sessions/05_...`).

**Tests:** `zig build test` → **passes** (374 total; 6 new in `gsl.zig`). File is
diagnostic-clean.

---

## 1. Task

The maintainer asked whether the current `stats` module covers **running** and
**moving-window** statistics; it does not. Goal: decide whether to add them and,
if so, design the scope + API. Two design questions had to be answered first
(see §3). A session report (this file) is to be written and updated throughout.

---

## 2. Findings: what `stats` covers vs. what's missing

- `stats(T)` (in `src/gsl.zig`) wraps only `gsl_stats_*` — **whole-array
  descriptive statistics** over a `Strided(T)` view already held in memory.
  Allocation-free; generic over `f32`/`f64`/integer element types.
- **Not covered**, and each a separate GSL module (headers confirmed present at
  `/opt/homebrew/include/gsl/`):
  - `gsl_rstat.h` — **running/streaming** stats: an accumulator fed one value at
    a time, O(1) memory, querying mean/variance/sd/rms/min/max/median/skew/
    kurtosis. `double`-only. Plus a standalone P²-algorithm single-quantile
    estimator (`gsl_rstat_quantile_*`).
  - `gsl_movstat.h` — **moving-window** stats: slides a width-`K` window over a
    signal, emits a same-length output series. `double`-only; operates on
    `gsl_vector *`. Functions: mean, variance, sd, median, min, max, minmax,
    sum, mad, mad0, qqr, Sn, Qn; plus user-accumulator `apply`/`apply_accum`.
  - `gsl_filter.h` — adjacent digital filters (Gaussian, median, RMF, impulse).
    **Out of scope for this session** (larger, more "signal processing").

---

## 3. Design questions answered (pre-implementation)

### 3.1 `Strided` vs `gsl_vector`; what is `block`?

- Our `Strided(T)` = `{ ptr, stride, len }`: a **pure non-owning view**, exactly
  the `(data, stride, len)` triple `gsl_stats_*` takes as three scalar args.
- `gsl_vector` = **same triple + ownership bookkeeping**:
  `{ size, stride, data, gsl_block *block, int owner }`. `block` is GSL's owning
  heap record (`gsl_block = { size_t size; double *data; }`) that actually holds
  the `malloc`'d buffer; `owner` says whether `gsl_vector_free` should free it.
- A `gsl_vector_view` (borrowed) has `block = NULL, owner = 0` — **that is the
  analog of our `Strided`.** So `movstat`'s `gsl_vector *` API can be fed
  zero-copy by **stack-constructing a borrowed vector** from a `Strided`/
  `StridedMut`:
  ```zig
  var v: c.gsl_vector = .{ .size = s.len, .stride = s.stride,
                           .data = @constCast(s.ptr), .block = null, .owner = 0 };
  ```
  No `gsl_block`/`gsl_vector` allocation, no new public type. `@constCast` on the
  read-only input path is safe (GSL takes `const gsl_vector *`).

### 3.2 Zig allocator or GSL allocation for the workspaces?

**Use GSL's allocation** (`gsl_movstat_alloc`/`_free`, `gsl_rstat_alloc`/`_free`),
wrapped in `init`/`deinit`. A Zig allocator is not a clean option:
- `gsl_movstat_alloc(K)` also computes accumulator `state_size` and allocates
  internal `work`(K)/`state` sub-buffers + function pointers; `gsl_rstat_alloc()`
  allocates a *nested* quantile workspace. Reproducing that from a Zig allocator
  means replicating GSL-internal (`.c`-file, version-dependent) layout — fragile.
- No allocator-injection hook; ownership can't be mixed (GSL-`malloc`'d ⇒ must be
  `gsl_*_free`'d).
- Matches existing precedent: `rand.Rng`/`rand.General` wrap `gsl_*_alloc`/`_free`
  directly. Follow `rand.Rng.init` exactly:
  `init(...) error{OutOfMemory}!Self` via `orelse return error.OutOfMemory`;
  `deinit` calls the GSL free.

---

## 4. Proposed scope & API (NOT yet ratified)

Two new **top-level sibling namespaces** in `gsl.zig` (not nested in `stats(T)`:
these are `f64`-only and stateful, which would break `stats`'s generic,
allocation-free shape). Names mirror GSL modules, consistent with `fft`/`sf`.

### 4.1 `gsl.rstat` — running statistics
- `Accumulator` (wraps `gsl_rstat_workspace`): `init()`/`deinit()`/`reset()`,
  `add(x)`, `addSlice(xs)` (convenience), `count()`, and getters
  `mean/variance/sd/sdMean/rms/skew/kurtosis/min/max/median`.
- `Quantile` (wraps `gsl_rstat_quantile_workspace`, P² algorithm):
  `init(p)`/`deinit()`/`reset()`, `add(x)`, `get()`.

### 4.2 `gsl.movstat` — moving-window statistics
- `End` enum: `pad_zero`/`pad_value`/`truncate` (← `gsl_movstat_end_t`).
- `Window` (wraps `gsl_movstat_workspace`): `init(k)` (symmetric K),
  `initAsymmetric(back, forward)` (H, J), `deinit()`.
- Methods take `Strided(f64)` in, `StridedMut(f64)` out, return `Error!void`
  (length mismatch ⇒ `Error.BadLength`), auto-installing the non-aborting
  handler on first use: `mean/variance/sd/median/min/max/sum`,
  `minMax(...y_min, y_max)`, `mad`/`mad0` (expose intermediate `xmedian` out),
  `qqr(q)`, `Sn`, `Qn`.
- **Deferred (reach via `c`):** `apply`/`apply_accum` custom-accumulator
  callback surface; the whole `gsl_filter` module.

---

## 5. Implementation status (DONE)

All of §4 landed in `src/gsl.zig` (single file, no new module):

- **`@cImport`** extended with `gsl/gsl_vector.h`, `gsl/gsl_rstat.h`,
  `gsl/gsl_movstat.h`.
- **Error layer added to `gsl.zig`** (it had none — confirmed): `pub const Error`
  (`Domain`/`Range`/`Invalid`/`BadLength`/`OutOfMemory`/`Unspecified`),
  `pub fn check(c_int) Error!void`, and a lazy `ensureHandler()` +
  `handler_installed` atomic mirroring `gsl_sf.zig`. `disableDefaultErrorHandler`
  now also sets the installed flag.
- **Borrowed vector views:** file-private `constVectorView`/`mutVectorView`
  stack-construct a `c.gsl_vector` (`block = null, owner = 0`) over a
  `Strided`/`StridedMut` — zero-copy, no `gsl_block`, no new public type.
- **`gsl.rstat`:** `Accumulator` (init/deinit/reset/add/addSlice/count +
  mean/variance/sd/sdMean/rms/norm/skew/kurtosis/min/max/median) and `Quantile`
  (init(p)/deinit/reset/add/get). `init` returns `error{OutOfMemory}!Self` like
  `rand.Rng`; getters are infallible bare `f64`.
- **`gsl.movstat`:** `End` enum (`enum(c.gsl_movstat_end_t)`:
  pad_zero/pad_value/truncate) and `Window` (init(k)/initAsymmetric(back,
  forward)/deinit) with `mean/variance/sd/median/min/max/sum/minMax/mad/mad0/
  qqr/Sn/Qn`, all `Error!void`, length-checked (`BadLength`), auto-installing the
  handler via private `run1`/`run2` drivers.
- **Deferred (documented in the `movstat` doc comment):**
  `apply`/`apply_accum`/`fill` (custom-accumulator surface) and the whole
  `gsl_filter.h` module.

### Tests added (6, in the file's binding-layer style)

- `rstat: streaming accumulator matches whole-array stats` — cross-checks
  mean/variance/sd/skew/kurtosis/min/max against `stats(f64)` as oracle;
  rms/norm against manual sum-of-squares; sdMean = sd/√n (note: **not** the same
  as `stats.sdMean`, which is the `_m` precomputed-mean form — naming collision);
  median within tolerance; reset round-trip.
- `rstat: P² quantile estimator approaches the true quantile` — 5000 GSL-RNG
  uniforms, 0.5/0.9 estimates within 0.03; reset round-trip.
- `movstat: moving statistics on a known signal (truncate ends)` — hand-computed
  mean/sum/min/max/median for a width-3 window; pad_zero boundary check.
- `movstat: multi-output, robust, and asymmetric routines run` — minMax ordering,
  mad/mad0/qqr/Sn/Qn/variance/sd finiteness/non-negativity, asymmetric window.
- `movstat: strided views and length checks` — stride-2 in/out borrowed views
  hit only the strided slots; length mismatch returns `Error.BadLength`.
- `rstat/movstat: every wrapped method is invoked (symbol + arity coverage)` —
  exhaustive `inline for` sweep calling every `Accumulator`/`Quantile`/`Window`
  method (movstat across all three `End` variants), mirroring the equivalent
  guard in `gsl_sf.zig`; forces every extern symbol to link and each signature
  to compile.

---

## 6b. Open question raised: consolidate the error layer with `gsl_sf.zig`?

`gsl.zig` and `gsl_sf.zig` now each carry a near-duplicate
`handler_installed`/`ensureHandler` + `Error` + `check`. Assessment: **fine to
keep separate for now.** Each file deliberately owns its own `@cImport`, and
`check` is tied to *that* import's `c.GSL_*` constants; the two `Error` sets
differ in scope (sf exposes math-specific `Overflow`/`Underflow`/`Roundoff`/
`MaxIterations`/`LossOfAccuracy` that the rng/stats/movstat surface never
raises). The only genuine (harmless, idempotent) redundancy is two handler-
install flags guarding one process-global handler. If we later want to DRY it, a
tiny shared internal module owning just `gsl_errno.h` + the handler install +
a superset `Error`/`check` is the clean route — not scheduled.

---

## 6. Design notes / gotchas discovered while implementing

- **`rstat.sdMean` ≠ `stats.sdMean`.** GSL reuses the name: `gsl_rstat_sd_mean`
  is the standard error of the mean (`sd/√n`), while `gsl_stats_sd_m` (our
  `stats.sdMean`) is the sd about a *precomputed* mean. Cross-check accordingly.
- **P² is genuinely approximate** and order-sensitive; test it with a real i.i.d.
  sample (GSL RNG), not a hand-rolled LCG modulo sequence.
- `gsl_rstat` getters are infallible bare `double`s, so `rstat` needs only
  `error{OutOfMemory}` on `init` (no `check`); only `movstat` needed the `Error`
  layer. It was added at `gsl.zig` top level so any future fallible binding here
  can reuse it.

---

## 7. Coordination notes

- `gsl_movstat`/`gsl_rstat` are **`double`-only** — plain namespaces, not
  `fn(comptime T) type`.
- Homebrew GSL headers: `/opt/homebrew/include/gsl/`; build already links `gsl`.
- The new `Error`/`check` in `gsl.zig` is intentionally its own set (not shared
  with `gsl_sf.zig`'s, which lives in that file); they mirror each other.
