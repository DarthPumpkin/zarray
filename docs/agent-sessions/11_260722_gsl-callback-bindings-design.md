## GSL Callback Bindings — Design + Implementation (Session 11)

> **Status update:** this started as a design sketch but was **implemented in
> increments 1–7** — see the "Implementation log" at the bottom of this file.
> Shipped: `gsl_deriv`, `gsl_integration`, `gsl_roots`, `gsl_min`,
> `gsl_chebyshev`, `gsl_monte`, `gsl_odeiv2` (ODEs, incl. the Jacobian path),
> plus the shared `gsl_callback.zig` bridge. The last chapter,
> `gsl_multifit_nlinear` (deferred here as D-cb5, blocked on a `gsl_matrix`
> bridge), shipped in **session 16** together with that bridge — so the GSL
> callback surface is now complete.

Design pass for the **callback-based** GSL chapters, the large block deliberately
deferred while Sessions 3–8 bound the callback-free surface (FFT, special
functions, RNG/stats/rstat/movstat, interp, histogram, poly, filter, qrng).

Every chapter here shares one problem: GSL wants a **C function pointer + opaque
`void *params`**, and we want to hand it an idiomatic Zig callable (a bare
`fn(f64) f64`, or a closure carrying captured state). The point of this pass is
to nail the *bridge* once, then let each chapter be a thin wrapper on top of it.

The design below was subsequently implemented (increments 1–7; see the
Implementation log). Conventions inherited from Session 8 apply throughout: per-file `@cImport`; `gsl.ensureHandler()` on fallible entry points;
chapter-local `Error` + `check`; `Strided`/`StridedMut` views; the generic
`constVectorViewOf`/`mutVectorViewOf` helpers (D12); and **never** cross a
`@cImport` boundary with a by-value `gsl_complex` (D13).

---

## Part 0 — The callback zoo (what we actually have to bridge)

GSL does not have one callback type; it has a family. All are plain structs of
function pointers + `void *params` (confirmed in the installed headers):

| Struct | Signature | Used by |
|--------|-----------|---------|
| `gsl_function` | `double (*)(double x, void *p)` | integration, roots (bracketing), min, deriv, chebyshev |
| `gsl_function_fdf` | `f`, `df`, and `fdf` (both at once), `void *p` | root *polishing* (Newton/secant/Steffenson) |
| `gsl_monte_function` | `double (*)(double *x, size_t dim, void *p)` + `dim` | monte carlo (plain/miser/vegas) |
| `gsl_odeiv2_system` | `function(t, y[], dydt[], p)` + optional `jacobian(t, y[], dfdy, dfdt[], p)` + `dimension` | ODE |
| `gsl_multifit_nlinear_fdf` | residual `f(x, p, → f_vec)` + `df` Jacobian + `n`,`p` sizes | nonlinear least squares |

Definitions live in `gsl_math.h` (`gsl_function`, `gsl_function_fdf`,
`gsl_function_vec`), `gsl_monte.h`, `gsl_odeiv2.h`, `gsl_multifit_nlinear.h`.
There is **no** `gsl_function.h`; pull `gsl_math.h`.

So the bridge is not one type — it's a small *pattern* instantiated five ways.

---

## Part A — The bridge (the whole ballgame)

### A1. The trampoline

Zig can emit a `callconv(.c)` function that GSL calls directly. The idiom:

```zig
/// Wrap a Zig context `*Ctx` (with `pub fn eval(self, x: f64) f64`) as a
/// stack-built gsl_function. The gsl_function borrows `ctx`; it must not
/// outlive it, and must not escape the calling scope.
pub fn function(ctx: anytype) c.gsl_function {
    const Ptr = @TypeOf(ctx);                 // *Ctx (or *const Ctx)
    const Ctx = @typeInfo(Ptr).pointer.child;
    const Trampoline = struct {
        fn call(x: f64, params: ?*anyopaque) callconv(.c) f64 {
            const self: Ptr = @ptrCast(@alignCast(params));
            return self.eval(x);
        }
    };
    return .{ .function = &Trampoline.call, .params = @constCast(ctx) };
}
```

Two ways to construct the callback, exposed as **named factory methods on a
per-chapter `Callback` value** (ratified D-cb2 — not a merged `anytype`
dispatcher, and not a doubled `X`/`XCtx` routine pair; the factory name makes
the caller's obligation explicit while keeping one function per routine):

1. **Plain function** — `Callback.initFn(f)` where `f: *const fn(f64) f64`. A
   bare top-level function coerces to that pointer type. The pointer is carried
   in `params`; a small trampoline casts it back and calls it.
   ```zig
   fn f(x: f64) f64 { return x * x - 2; }
   const d = try gsl.deriv.central(.initFn(f), 1.5, 1e-8);
   ```

2. **Context / captured state** — `Callback.initCtx(ctx)` where `ctx` is a
   pointer to a struct with `pub fn eval(self, x: f64) f64`. The caller writes a
   small struct holding the captured fields; the context is caller-owned memory
   that outlives the GSL call. A comptime guard reports a clear error if `eval`
   is missing.
   ```zig
   const Wave = struct { k: f64,
       pub fn eval(self: *const @This(), x: f64) f64 { return @sin(self.k * x); } };
   var w = Wave{ .k = 3.0 };
   const d = try gsl.deriv.central(.initCtx(&w), 0.5, 1e-6);
   ```

Because distinct `@cImport` blocks yield distinct `gsl_function` types, the
`Callback` value cannot be one shared type: each chapter aliases the generic
`callback.Function(c.gsl_function)` (and `callback.FunctionFdf(...)` for root
polishing). A routine takes the value by value and passes `&cb.gf` to GSL. The
leading-dot decl literal (`.initFn(...)`) resolves against the routine's
parameter type, so the call site stays terse. The low-level builders
`function`/`context`/`functionFdf` remain available for direct use.

### A2. Error propagation out of a callback — DECISION NEEDED

GSL callbacks return a bare `double` with **no status channel**, and GSL calls
them deep inside its own loops. A Zig callback that wants to fail has three
options; pick one project-wide:

- **(a) Sentinel + post-check (recommended default).** The callback returns
  `NaN`/`Inf` on failure; the *context struct* also stashes a `caught_error:
  ?anyerror`. After the GSL routine returns, the wrapper checks the context and
  surfaces it as a Zig `Error`. Works for the closure form; the bare-fn form only
  gets the sentinel.
- **(b) Callbacks are infallible.** Document that callbacks must return a finite
  `f64` and never fail; anything else is UB/GSL-domain-error territory. Simplest,
  and true for the overwhelming majority of math callbacks (`x*x - 2`).
- **(c) longjmp/panic.** Rejected — crosses the C frame, unsafe, non-portable.

Recommendation: support **(b) as the baseline** and offer **(a)** for the
closure form via an optional `caught: *?anyerror` field convention. Ratify
before building.

### A3. Lifetime & re-entrancy rules (to document, not enforce)

- The `gsl_function` value is built on the stack immediately before the GSL call
  and never stored. Its `params` borrows caller memory; caller keeps the context
  alive across the call. This mirrors how `filter`/`movstat` build borrowed
  `gsl_vector` views inline.
- Trampolines are pure forwarders with no global state → naturally reentrant and
  thread-safe as long as each thread builds its own `gsl_function` over its own
  context (same rule as `interp.Accel`).

### A4. Where the bridge lives

New file `src/gsl_callback.zig`, re-exported as `gsl.callback`, holding
`function`, `functionFdf`, `monteFunction`, `odeSystem`, `multifitFdf`, plus the
bare-fn conveniences. Every callback chapter imports it. This keeps the five
trampoline variants in one audited place (they're the only `callconv(.c)` +
pointer-casting code in the project).

---

## Part B — Per-module API sketches (ordered by value)

### B1. Numerical integration (`gsl_integration`) — highest value

```
pub const integration = struct {
    pub const Workspace = struct { ptr, init(limit)/deinit };
    pub const Result = struct { value: f64, abserr: f64 };
    // QAG adaptive on [a,b] with a key (GAUSS15..61):
    pub fn qag(f, a, b, epsabs, epsrel, key, ws) Error!Result;
    pub fn qags(f, a, b, epsabs, epsrel, ws) Error!Result;   // + singularities
    pub fn qagi(f, epsabs, epsrel, ws) Error!Result;         // (-inf, +inf)
    pub fn qagiu(f, a, …) / qagil(f, b, …);                  // half-infinite
    pub fn qng(f, a, b, epsabs, epsrel) Error!Result;        // non-adaptive, no ws
    // f is a gsl.callback context (or bare fn) → gsl_function.
};
```
Non-adaptive `qng` needs no workspace — nice first target to validate the
bridge end-to-end. CQUAD/Romberg (`gsl_integration_cquad`/`romberg`) are
follow-ons with their own workspaces.

### B2. One-dim root finding (`gsl_roots`) — high value

```
pub const roots = struct {
    pub const Bracket = enum { bisection, brent, falsepos };   // gsl_root_fsolver_*
    pub const Solver = struct {
        ptr, init(Bracket)/deinit,
        set(f, lo, hi) Error!void,          // gsl_function
        iterate() Error!void, root() f64, interval() struct{lo,hi:f64},
    };
    pub const Polish = enum { newton, secant, steffenson }; // gsl_root_fdfsolver_*
    pub const PolishSolver = struct {
        ptr, init(Polish)/deinit,
        set(fdf, guess) Error!void,         // gsl_function_fdf
        iterate() Error!void, root() f64,
    };
    // Convergence helpers: testInterval / testDelta / testResidual.
};
```
This is the first consumer of **`gsl_function_fdf`** — the caller supplies `f`,
`df`, and a combined `fdf`. The bridge's `functionFdf` needs a context with
`eval`, `deriv`, and `evalDeriv` methods (or synthesize `fdf` from the other two
by default).

### B3. Minimization (`gsl_min`) — same shape as roots

`gsl_min_fminimizer` with `goldensection`/`brent`/`quad_golden`, `set(f, x_min,
x_lo, x_hi)`, `iterate`, `minimum`, `interval`. Trivial once B2 exists.

### B4. Numerical differentiation (`gsl_deriv`) — tiny, no workspace

```
pub fn central(f, x, h) Error!Result;    // Result{value, abserr}
pub fn forward(f, x, h) Error!Result;
pub fn backward(f, x, h) Error!Result;
```
Three one-liners over `gsl_function`. Good smoke test for the bare-fn front door.

### B5. Chebyshev approximation (`gsl_chebyshev`)

```
pub const Chebyshev = struct {
    ptr, init(order)/deinit,
    fit(f, a, b) Error!void,             // gsl_cheb_init over gsl_function
    eval(x) f64, evalErr(x) Result, evalN(n, x) f64,
    deriv() Chebyshev, integ() Chebyshev,  // derived series
};
```

### B6. Monte Carlo integration (`gsl_monte`) — first multidim callback

```
pub const monte = struct {
    pub const Plain  = struct { ptr, init(dim)/deinit };
    pub const Miser  = struct { ptr, init(dim)/deinit };
    pub const Vegas  = struct { ptr, init(dim)/deinit, chisq()… };
    pub fn integrate(engine, f, lo: []const f64, hi: []const f64,
                     calls: usize, rng: gsl.rand.Rng) Error!Result;
};
```
Uses **`gsl_monte_function`** (`f(x[], dim, p)`), and — nicely — reuses the
existing `gsl.rand.Rng`. The multidim trampoline casts `x` to `[]const f64` of
length `dim` before calling `ctx.eval(point)`.

### B7. ODE solvers (`gsl_odeiv2`) — highest complexity

```
pub const ode = struct {
    pub const Step = enum { rk4, rkf45, rk8pd, msadams, bsimp, … };
    pub const System = struct { … };   // built via gsl.callback.odeSystem
    pub const Driver = struct {
        ptr, initY(system, step, h_start, epsabs, epsrel)/deinit,
        apply(t: *f64, t1: f64, y: []f64) Error!void,   // integrate to t1
        applyFixedStep(…), reset(),
    };
};
```
Uses **`gsl_odeiv2_system`** (RHS + optional Jacobian + `dimension`). Implicit
steppers (`bsimp`, `msbdf`) need the Jacobian; explicit ones don't. Per D-cb4
the bridge exposes **two separate builders** — `odeSystem(ctx)` (context needs
`rhs`) and `odeSystemWithJacobian(ctx)` (context needs `rhs` *and* `jacobian`)
— so the call site states intent explicitly rather than relying on `@hasDecl`.
Biggest single chapter; do it last.

### B8. Nonlinear least squares (`gsl_multifit_nlinear`) — depends on vectors

Uses **`gsl_multifit_nlinear_fdf`** (residual vector + Jacobian matrix over
`gsl_vector`/`gsl_matrix`). This chapter needs real `gsl_matrix` support, which
the project does not yet have a first-class binding for. **Defer until a
`gsl_matrix` view story exists** (parallels the Session-8 deferral of the
multivariate-Gaussian `randist` families for the same reason). List as
out-of-scope for the first callback pass.

---

## Part C — Proposed build order

1. **`gsl_callback.zig`** — `function` + bare-fn `wrap` + the error-propagation
   decision (A2). Nothing else compiles without this.
2. **`gsl_deriv.zig`** (B4) — smallest real consumer; validates the bridge with
   no workspace.
3. **`gsl_integration.zig`** (B1) — start with `qng` (no workspace), then the
   `qag*`/`qags`/`qagi*` workspace forms.
4. **`gsl_roots.zig`** (B2) — introduces `gsl_function_fdf` (`functionFdf`).
5. **`gsl_min.zig`** (B3) — reuses roots' machinery.
6. **`gsl_chebyshev.zig`** (B5).
7. **`gsl_monte.zig`** (B6) — introduces the multidim trampoline; reuses
   `gsl.rand.Rng`.
8. **`gsl_odeiv2.zig`** (B7) — the big one; optional Jacobian via `@hasDecl`.
9. *(deferred)* `gsl_multifit_nlinear` (B8) — blocked on `gsl_matrix` bindings.

Each step: own file, re-export from `gsl.zig`, add to the `test { _ = …; }`
discovery block, test correctness against closed-form answers (∫₀^π sin = 2;
root of x²−2 = √2; min of (x−2)² at 2; d/dx sin at known points; etc.),
plus error paths (bad interval, non-convergence → surfaced as `Error`, not
abort).

---
## Part D — Decisions (ratified by maintainer)

1. **D-cb1 (error propagation, A2): RATIFIED — infallible baseline + opt-in
   `caught`.** Callbacks must return a finite `f64` and are assumed not to fail.
   For fallible user code, the *closure* form may carry a `caught: *?anyerror`
   field it sets on failure (returning `NaN`); the caller checks that variable
   after the GSL routine returns and prefers it over GSL's generic error. The
   bridge itself stays unaware of `caught` — it's a pure caller-side convention.
2. **D-cb2 (front doors): RATIFIED — a `Callback` value with named factory
   methods.** Rather than a single `anytype` dispatcher *or* a doubled
   `X`/`XCtx` routine pair, each chapter exposes one `Callback` value type
   (`callback.Function(c.gsl_function)`) constructed with `.initFn(f)` (plain
   `*const fn(f64) f64`; a bare `fn` coerces in) or `.initCtx(ctx)` (a
   `*context` with `pub fn eval`, comptime-validated). Routines take one
   `Callback` argument, so the call site reads `central(.initFn(myFn), …)` /
   `central(.initCtx(&w), …)` — explicit *and* one function per routine (better
   scaling for chapters with many routines). The low-level `function`/`context`
   builders remain as primitives.
3. **D-cb3 (fdf synthesis): RATIFIED — synthesize when absent.** For
   `gsl_function_fdf`, if the context provides `evalDeriv(x, *f64, *f64)` use it;
   otherwise synthesize the combined `fdf` from `eval` + `deriv`
   (`@hasDecl(Ctx, "evalDeriv")`).
4. **D-cb4 (ODE Jacobian): RATIFIED — two separate types (explicitness).**
   *Reversed* the sketch's `@hasDecl` recommendation. The bridge exposes two
   distinct builders — `odeSystem(ctx)` (RHS only) and
   `odeSystemWithJacobian(ctx)` (RHS + Jacobian) — so the caller states intent
   at the call site and the required methods are unambiguous. B7 below is
   updated to reflect this.
5. **D-cb5 (scope): RATIFIED — defer.** `gsl_multifit_nlinear` (and any other
   `gsl_matrix`-typed callback chapter) is out of scope for the first callback
   pass; it waits until first-class matrix-view bindings exist.

---


## Implementation log

### Increment 1 — the bridge + `gsl_deriv` (steps 1–2 of Part C)

- **`src/gsl_callback.zig`** — the bridge, implemented `@cImport`-free and
  generic over the target C struct type (each chapter passes its own
  `c.gsl_function`), sidestepping the distinct-`@cImport`-types problem the way
  the `*VectorViewOf` helpers do for `gsl_vector`. Per D-cb2 it exposes the
  generic value type **`Function(GF)`** with `.initFn` / `.initCtx` factory
  methods (and `FunctionFdf(GF)` with `.initCtx` for root polishing), layered
  over the low-level primitives `function` / `context` / `functionFdf`. Context
  structs are validated by a comptime guard (`requireMethods`). Trampolines are
  `callconv(.c)`. Unit-tested against mock structs matching the C layout, so the
  bridge is exercised with no GSL dependency.
- **`src/gsl_deriv.zig`** — first real consumer: `central`/`forward`/`backward`
  each take one `Callback` (= `callback.Function(c.gsl_function)`), so the call
  site reads `central(.initFn(myFn), x, h)` or `central(.initCtx(&w), x, h)`.
  Returns `Result{ value, abserr }`. Tested against closed forms (d/dx x²,
  d/dx sin), a parameter-capturing context, and build-once/reuse of a
  `Callback`.
- **Wiring:** both re-exported from `gsl.zig` (`gsl.callback`, `gsl.deriv`),
  added to the test-discovery block, and listed in the hub docstring.
- **Gotcha found:** cross-file trampolines require the context's `eval`/`deriv`
  methods to be **`pub`** (the trampoline lives in `gsl_callback.zig` and calls
  them across the module boundary). Documented in the bridge header.
- **Validation:** `zig build test --summary all` → **496/496 passed**.

### Increment 2 — `gsl_integration` (step 3 of Part C)

- **`src/gsl_integration.zig`** — second callback consumer and the first that
  pairs a `Callback` (= `callback.Function(c.gsl_function)`) with a reusable
  workspace. Surface:
    - `qng(cb, a, b, epsabs, epsrel)` — non-adaptive, no workspace (the
      end-to-end validation target for the bridge against a real quadrature
      routine).
    - `qag(cb, a, b, epsabs, epsrel, key, ws)` — adaptive Gauss-Kronrod with a
      `Key` enum (`gauss15`..`gauss61`).
    - `qags(cb, a, b, epsabs, epsrel, ws)` — adaptive + extrapolation
      (integrable singularities).
    - `qagi`/`qagiu`/`qagil` — infinite / half-infinite ranges.
  - `Workspace` owns the `gsl_integration_workspace` (`init(max_intervals)` /
    `deinit`); routines pass its allocated `limit` as the subdivision cap, so
    callers never juggle a separate `limit` argument. `limit()` exposes it.
  - `Result{ value, abserr }` mirrors `deriv`.
- **Gotcha found:** `qagi`/`qagiu`/`qagil` take a *non-const* `gsl_function *`
  (unlike `qng`/`qag`/`qags`), so `&cb.gf` needs `@constCast` at those three
  call sites.
- **Error mapping:** extended the chapter `check` beyond deriv's set to cover
  the quadrature statuses (`ETOL`/`EMAXITER`/`EROUND`/`ESING`/`EDIVERGE`) so a
  non-convergent call surfaces as a typed `Error` rather than aborting.
- **Wiring:** re-exported from `gsl.zig` (`gsl.integration`), added to the
  test-discovery block and the hub docstring chapter list + callback-chapter
  note.
- **Tests:** closed forms (∫₀^π sin = 2 across all six rules; ∫₀¹ 1/√x = 2 via
  `qags`; ∫ e^{−x²} over the line = √π, split into equal halves by
  `qagiu`/`qagil`), a parameter-capturing context, `Callback`+`Workspace`
  reuse, workspace `limit`/zero-rejection, and a non-convergent case surfacing
  as `Error.MaxIterations`.
- **Validation:** `zig build test --summary all` → **505/505 passed** (+9).

#### Refinements (post-review, maintainer-agreed)

- **`Tol{ .abs, .rel }` param struct.** The finite-interval routines had four
  adjacent `f64` args (`a, b, epsabs, epsrel`); the two tolerances are the real
  hazard (dimensionally alike, adjacent, a silent bug if swapped). Collapsed
  both tolerances into one named-field `Tol` value used by *all six* routines,
  defaulting to a purely relative target (`.{ .rel = 1e-9 }`, `abs = 0`). Call
  sites now read `qng(.initFn(f), 0, pi, .{ .rel = 1e-9 })`. Left `a`/`b`
  positional (a reversed interval is self-evident and GSL just negates). +1
  test for the `Tol` default path.
- **`@constCast` on `qagi`/`qagiu`/`qagil` documented.** These three declare a
  *non-const* `gsl_function *` — not because GSL mutates the callback
  (`GSL_FN_EVAL` is read-only) but because the infinite-range transform stores
  the pointer in a non-const internal helper field, and the public signature
  matches. Added a comment at each call site recording that the cast is safe.
- **Validation after refinements:** **506/506 passed**.

### Increment 3 — `gsl_roots` (step 4 of Part C)

- **`src/gsl_roots.zig`** — first consumer of the bridge's *derivative-bearing*
  form (`callback.FunctionFdf`). Two stateful solver families:
    - `Solver` (bracketing: `bisection`/`brent`/`falsepos`) — `init(Bracket)` /
      `deinit`, `set(Callback, lower, upper)`, `iterate`, `root`, `interval`
      (→ `Interval{ lower, upper }`), `name`.
    - `PolishSolver` (derivative: `newton`/`secant`/`steffenson`) —
      `init(Polish)` / `deinit`, `set(CallbackFdf, guess)`, `iterate`, `root`,
      `name`.
  - Convergence helpers `testInterval` / `testDelta` / `testResidual` return
    `Error!bool` (`true` = converged, `false` = `GSL_CONTINUE`, error for a bad
    tolerance). Reused the swap-safe `Tol{ .abs, .rel }` pattern (here
    defaulting to a purely *absolute* target, the natural choice for roots).
  - `Callback = callback.Function(c.gsl_function)` and
    `CallbackFdf = callback.FunctionFdf(c.gsl_function_fdf)`.
- **Lifetime design (important):** unlike `deriv`/`integration` (synchronous
  one-shot calls), a solver retains a *pointer* to the `gsl_function`/`_fdf`
  across many `iterate` calls. So each solver stores the callback **by value**
  (`cb: Callback = undefined`) and hands GSL `&self.cb.gf` — a temporary would
  dangle. Documented rule: don't move a solver between `set` and the last
  `iterate`, and keep any `.initCtx` context alive across that span. Because
  `self` is a `*Solver`, `&self.cb.gf` is already mutable — no `@constCast`
  needed (`gsl_root_fsolver_set` takes non-const `gsl_function *`).
- **`converged` helper:** maps `GSL_SUCCESS`→true, `GSL_CONTINUE`→false, else
  routes through `check` (with an `unreachable` after, since any other status
  is always an error).
- **Wiring:** re-exported from `gsl.zig` (`gsl.roots`), added to the
  test-discovery block and the hub docstring chapter list + callback-chapter
  note.
- **Tests:** all three bracketing solvers converge to √2 of x²−2 on [0,2]; all
  three polishing solvers refine √2 from a guess of 1.5 (exercising the
  `functionFdf` synthesis path); a non-bracketing interval → `Error.Invalid`;
  a parameter-capturing context; solver `name`s; the three convergence helpers
  (continue-then-success + negative-tolerance → `Error.BadTolerance`); and
  `set`-reuse onto a second problem (cos root = π/2).
- **Validation:** `zig build test --summary all` → **513/513 passed** (+7).

### Increment 4 — `gsl_min` (step 5 of Part C)

- **`src/gsl_min.zig`** — a close twin of `gsl_roots`'s bracketing solver, for
  one-dimensional function *minimization*. `Minimizer` wraps
  `gsl_min_fminimizer`:
    - `init(Method)` (`goldensection`/`brent`/`quad_golden`) / `deinit`,
    - `set(Callback, guess, lower, upper)` — requires `f(guess)` strictly below
      both endpoints (else `Error.Invalid`),
    - `iterate`, `minimum()` (x location), `fMinimum()` (value there),
      `interval()` (→ `Interval{ lower, upper }`), `name`.
  - Same by-value callback storage and lifetime rule as `roots` (GSL retains a
    pointer across iterations).
  - `testInterval(lower, upper, Tol)` convergence helper, same swap-safe `Tol`
    (absolute default).
- **Numerics gotcha found (documented in the test):** locating a 1-D minimum is
  only accurate to ~√eps because `f` is locally quadratic, so an over-tight
  *interval* tolerance is physically unreachable — `goldensection` bottoms out
  around a bracket width of ~1e-7 and returns `GSL_FAILURE`. Tests use a
  realistic `abs = 1e-5`; mapped both `GSL_FAILURE` and `GSL_EFAILED` to
  `Error.Failed`. Also switched the all-methods test from an exact parabola
  (which lets interpolating methods jump to the vertex and stall the bracket)
  to `cos` (min at π).
- **Wiring:** re-exported from `gsl.zig` (`gsl.min`), added to the
  test-discovery block and the hub docstring chapter list + callback-chapter
  note.
- **Tests:** all three methods find the `cos` minimum at π; a non-trapping
  triple → `Error.Invalid`; a parameter-capturing context; algorithm `name`;
  `set`-reuse onto a second problem; and `testInterval` continue-then-success
  plus negative-tolerance → `Error.BadTolerance`.
- **Validation:** `zig build test --summary all` → **519/519 passed** (+6).

### Increment 5 — `gsl_chebyshev` (step 6 of Part C)

- **`src/gsl_chebyshev.zig`** — Chebyshev-series approximation, reached as
  `gsl.cheb`. `Chebyshev` owns a `gsl_cheb_series`:
    - `init(max_order)` / `deinit`,
    - `fit(Callback, a, b)` — samples the function over `[a, b]` and computes
      the coefficients,
    - `eval(x)` / `evalErr(x)` (→ `Result{ value, abserr }`) / `evalN(n, x)` /
      `evalNErr(n, x)`,
    - `order()`, `coeffs()` (borrowed `[]const f64` of length `order + 1`),
    - `deriv()` / `integ()` — each returns a *new* caller-owned `Chebyshev`
      (same order/interval) for `f'` and `∫ₐˣ f`.
- **Lifetime note (differs from roots/min):** `gsl_cheb_init` samples the
  function *synchronously* and stores only coefficients — it does not retain a
  function pointer. So the `Callback` is transient (like `deriv`/`integration`),
  needing no by-value storage; `fit` just keeps a local copy alive for the
  duration of the call. Documented in the header.
- **`evalErr`/`evalNErr`:** GSL guarantees these cannot fail for a fitted
  series, so they return `Result` directly (status ignored) rather than
  `Error!Result`, keeping the eval path allocation- and error-free like `eval`.
- **Wiring:** re-exported from `gsl.zig` (`gsl.cheb`), added to the
  test-discovery block and the hub docstring chapter list + callback-chapter
  note.
- **Tests (7):** exp fit accurate to ~1e-12 across `[0,1]`; `coeffs().len ==
  order + 1`; `evalErr` value+bound; a truncated `evalN(4, ·)` is strictly
  coarser than the full series; the `deriv` series reproduces exp; the `integ`
  series reproduces `exp(x) − 1` (zero at `a`); a parameter-capturing context
  (`sin(k·x)`); and re-`fit` onto a new function/interval.
- **Validation:** `zig build test --summary all` → **539/539 passed** (7 new
  `cheb` tests; the remaining delta from prior increments is unrelated,
  concurrent, non-callback work in the tree — a new untracked `src/lapack.zig`
  plus edits to `build.zig`/`accelerate.zig`/`root.zig` — which this increment
  did not touch).

### Increment 6 — `gsl_monte` + bridge multidim trampoline (step 7 of Part C)

- **`src/gsl_callback.zig` (bridge extension):** added the *multidimensional*
  form for `gsl_monte_function` (`f(x[], dim, params)`):
    - `MonteFunction(MF)` value type with `.initFn(f)` (a
      `*const fn([]const f64) f64`) and `.initCtx(ctx)` (a struct with
      `pub fn eval(self, x: []const f64) f64`), over the low-level
      `monteFunction`/`monteContext` builders.
    - The trampoline reconstructs the `dim`-length **slice** from GSL's raw
      `[*c]f64` + `dim` before calling. The struct's `dim` field is left 0 at
      construction and filled in by the consumer immediately before the call
      (it equals the integration dimension), so the caller never repeats the
      dimension when building the callback.
    - Unit-tested against a `MockMonte` extern struct (plain fn + context).
- **`src/gsl_monte.zig`** — three engines, each owning its GSL state:
    - `Plain` / `Miser` / `Vegas`, `init(dim)` / `deinit`, `reset()`, and
      `integrate(Callback, xl, xu, calls, rng) Error!Result`.
    - `Vegas` additionally exposes `chisq()` and `runval()` (cumulative
      weighted average + σ across passes; re-run `integrate` to refine the grid).
    - A shared comptime-generic `run` helper (over the engine's C
      `integrate_fn` + typed state ptr) holds the common body; all three C
      entry points share the same argument order.
- **Cross-`@cImport` `gsl_rng` identity:** the `gsl_rng *` this file sees is a
  *distinct* C type from `gsl.rand`'s (each `@cImport` of `gsl_rng.h` yields its
  own), so the shared `gsl.rand.Rng`'s pointer is reinterpreted via
  `@ptrCast` (a `rngPtr` helper) — the same identity workaround the vector-view
  helpers use. Reuses the existing RNG rather than re-wrapping one.
- **Callback lifetime:** transient like `integration` — GSL samples during the
  `integrate` call only; `run` fills `dim` on a local copy and passes
  `&f.mf`. `xl`/`xu`/point slices are borrowed for the call's duration.
- **Validation gotcha found:** `gsl_monte_vegas_runval` returns the *cumulative*
  weighted average across passes, which is *not* bit-identical to the last
  `integrate` call's returned value — the test asserts both land near the known
  integral rather than equal to each other.
- **Wiring:** re-exported from `gsl.zig` (`gsl.monte`), added to the
  test-discovery block and the hub docstring chapter list + callback-chapter
  note.
- **Tests:** the classic Γ(1/4)⁴/(4π³) box integrand near its known value for
  `Plain`/`Miser`/`Vegas`; VEGAS `chisq`/`runval`; a parameter-capturing
  context (`a·x + b·y`); a bounds/dimension mismatch → `Error.BadLength`; and
  an exact constant integrand (unit-cube volume = 1). RNG is the seeded
  `mt19937`, so results are deterministic.
- **Validation:** `zig build test --summary all` → **547/547 passed** (+2
  bridge, +6 monte; total also includes the unrelated concurrent tree work).

### Increment 7 — `gsl_odeiv2` + bridge ODE builders (step 8 of Part C)

- **`src/gsl_callback.zig` (bridge extension):** added the ODE builders ratified
  in D-cb4:
    - `odeSystem(Sys, dimension, ctx)` — RHS-only (`pub fn rhs(...)`).
    - `odeSystemWithJacobian(Sys, dimension, ctx)` — RHS + Jacobian
      (`pub fn rhs(...)` + `pub fn jacobian(...)`).
    - Both methods accept `void` or `c_int` returns; `void` is treated as
      `GSL_SUCCESS`, nonzero `c_int` is forwarded to GSL.
    - Added `MockOde` bridge tests for RHS-only wiring, Jacobian wiring, and
      status passthrough.
- **`src/gsl_odeiv2.zig`** — new ODE chapter (`gsl.ode`):
    - `Step` enum covering `gsl_odeiv2_step_*` steppers (`rk2`, `rk4`, `rkf45`,
      `rkck`, `rk8pd`, `rk2imp`, `rk4imp`, `bsimp`, `rk1imp`, `msadams`,
      `msbdf`).
    - `System` callback bundle with explicit constructors:
      `System.initCtx(&ctx, dim)` and
      `System.initCtxWithJacobian(&ctx, dim)`.
    - `Driver` wrapper around `gsl_odeiv2_driver` with:
      `initY` / `initYp`, `apply`, `applyFixedStep`, `reset`, `resetHStart`,
      `setHMin`, `setHMax`, `setNMax`, and `stepCount`.
    - Length validation (`Error.BadLength`) for state slices and chapter-local
      error mapping (`BadFunction`, `NoProgress`, `MaxIterations`, etc.).
- **Lifetime rule documented explicitly:** `gsl_odeiv2_driver` stores a pointer
  to `gsl_odeiv2_system`, so `Driver` takes `*System`; the `System` must stay
  alive and unmoved for the driver's lifetime.
- **Wiring:** re-exported from `gsl.zig` as `gsl.ode`, added to top-level
  chapter docs and test discovery.
- **Tests:** explicit solver on `dy/dt = -k y`, implicit `bsimp` with Jacobian,
  fixed-step integration + reset helpers, bad-length rejection, callback-returned
  `EBADFUNC` propagation, and zero-dimension rejection.
- **Validation:** `zig build test --summary all` → **556/556 passed**.

### Next

~~`gsl_multifit_nlinear` remains deferred (D-cb5) until first-class matrix-view
bindings land.~~ **Done in session 16:** the `gsl_matrix` borrowed-view bridge
(`Matrix`/`MatrixMut` in `gsl.zig`) and `gsl_multifit_nlinear` (`gsl.nlinear`)
both shipped. See `16_260723_gsl-matrix-bridge-and-multifit-nlinear.md`. The GSL
callback surface is now complete.
