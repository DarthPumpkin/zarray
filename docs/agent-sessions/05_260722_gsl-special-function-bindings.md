# GSL Special-Function Bindings — Session 5 Report

**Status:** The GSL special-function chapter (`gsl_sf_*`) is **bound, tested,
and integrated**. A module `src/gsl_sf.zig` wraps ~200 scalar special functions
across 25+ families (each returning `Error!f64`), the 1-D sequence `_array`
fillers, the Jacobian elliptic functions, and a generic `evaluate` for the
value+error-estimate path. It is exposed as **`gsl.sf`** (re-exported from
`gsl.zig`).

**Update (session 5, discussion round 2):** all open design questions were
**resolved** with the maintainer (see [§7](#7-design-decisions-resolved)) and the
redesign is now **implemented and green** (see
[§9](#9-remaining-work--implementation-plan)): every scalar wrapper returns
`Error!f64`, the non-aborting handler auto-installs on first use, a reserved
(compile-error) `result` namespace holds the val+err tier, the 1-D `_array`
families are bound, an exhaustive coverage test invokes every function, and the
module is now reached as `gsl.sf` (the direct `libs.gsl_sf` export was removed).
`zig build test` passes.

**Audience:** An agent continuing the GSL binding work in `zarray` (a.k.a.
`ndarray_zig`). This complements the RNG/stats bindings in `src/gsl.zig`
(session-independent) and the FFT bindings in `src/gsl_fft.zig`
(`docs/agent-sessions/03_260722_gsl-fft-bindings.md`).

**Tests:** `zig build test` → **passes** (whole project, 343 tests; 15 new in
`gsl_sf.zig`). The new file is diagnostic-clean.

---

## 1. Task

Expand the existing GSL bindings to cover **special functions**, in a **new
file**, following the conventions of `gsl.zig` (and consulting BLAS/LAPACK style
+ `docs/`). Do not modify the LAPACK bindings (another agent is actively working
on them). Implement autonomously, then summarize decisions and raise design
questions for a follow-up iteration.

---

## 2. What was implemented

New module **`src/gsl_sf.zig`**, wired into `src/root.zig` as
`libs.gsl_sf = @import("gsl_sf.zig");`.

It owns its own `c = @cImport({...})` over `gsl/gsl_errno.h`, `gsl/gsl_mode.h`,
and the umbrella `gsl/gsl_sf.h` (which pulls in every `gsl_sf_*.h` header). This
keeps the special-function C surface out of `gsl.zig`, mirroring how
`gsl_fft.zig` owns its own cImport.

### Coverage (natural `f64` forms, grouped by GSL header into namespaces)

| Namespace | Family |
|---|---|
| `erf` | erf, erfc, log_erfc, Gaussian Z/Q, hazard |
| `gamma` | Γ, lnΓ, Γ*, 1/Γ, factorials, double factorials, choose, Pochhammer, incomplete Γ (P/Q/non-norm), beta, incomplete beta |
| `psi` | digamma, trigamma, polygamma (int + real args) |
| `zeta` | Riemann ζ, ζ−1, Hurwitz ζ, Dirichlet η (int + real) |
| `bessel` | J/Y/I/K (integer + fractional `nu` + scaled), spherical j/y/i/k, zeros |
| `airy` | Ai/Bi, derivatives, scaled variants, zeros (mode-carrying) |
| `ellint` | Legendre K/E/Π/D (complete + incomplete) + Carlson RC/RD/RF/RJ (mode-carrying); plus `elljac` (sn/cn/dn) |
| `exp` | exp, exp_mult, expm1, exprel, exprel_2, exprel_n |
| `log` | log, log_abs, log_1plusx, log_1plusx_mx |
| `trig` | sin, cos, hypot, sinc, sin_pi/cos_pi, lnsinh, lncosh, angle reduction |
| `expint` | E1/E2/En, Ei, Shi, Chi, expint_3, Si, Ci, atanint |
| `fermiDirac` | F_{-1..2}, F_int, F_{±1/2}, F_{3/2}, incomplete F_0 |
| `debye` | D_1..D_6 |
| `transport` | J_2..J_5 |
| `synchrotron` | first/second synchrotron functions |
| `legendre` | P_l, Q_0/Q_1/Q_l, P_l^m, sphPlm, conical P, H3d |
| `laguerre` | L_1..L_3, L_n^a |
| `gegenbauer` | C_1..C_3, C_n^λ |
| `hermite` | physicist H_n, probabilist He_n, Hermite function ψ_n, derivatives, zeros |
| `coulomb` | hydrogenic radial R_1, R_{n,l} |
| `coupling` | Wigner 3j/6j/9j |
| `hyperg` | 0F1, 1F1 (int + real), U (int + real), 2F1 (+conj/renorm), 2F0 |
| (top-level fns) | `dawson`, `clausen`, `dilog`, `powInt`, `multiply` |

### Error-bounded infrastructure

- `Result = struct { val: f64, err: f64 }` — mirrors `gsl_sf_result`.
- `Error` — Zig error set mapping the common GSL status codes (`Domain`,
  `Range`, `Overflow`, `Underflow`, `LossOfAccuracy`, `Roundoff`,
  `MaxIterations`, `Unspecified`).
- `check(status: c_int) Error!void` — code → error mapper.
- `evaluate(comptime ef, args) Error!Result` — the generic escape hatch that
  turns **any** `gsl_sf_*_e` symbol into `Error!Result`, supplying the trailing
  `gsl_sf_result *` itself. E.g. `try sf.evaluate(sf.c.gsl_sf_gamma_e, .{5.0})`.
- `disableDefaultErrorHandler()` — toggles GSL's process-wide handler (shared
  with `gsl.zig`).
- `Mode` enum (`double`/`single`/`approx` → `GSL_PREC_*`) for the `gsl_mode_t`
  families (Airy, elliptic integrals).

---

## 3. Key design decisions (first-draft; not yet ratified)

### 3.1 New separate file, `gsl.zig` untouched

Special functions are a self-contained GSL subsystem with a uniform two-form
calling convention. Kept in `src/gsl_sf.zig` with its own cImport. `gsl.zig` and
the LAPACK code were not modified.

### 3.2 Natural `f64` form is the primary API

The bulk of the surface returns bare `f64`, grouped by GSL header into
namespaces (`sf.gamma.gamma(x)`, `sf.bessel.J0(x)`, `sf.zeta.zeta(2.0)`). This
matches the established `gsl.zig` style (distributions/stats return bare values
and rely on GSL's abort-by-default handler) and lets these compose like
`std.math`.

### 3.3 Error-bounded `_e` form exposed generically, not per-function

Rather than hand-writing a second wrapper for each of ~200 functions, the `_e`
form is reached through the single generic `evaluate(ef, args)` helper. This
gives full access to GSL's error estimate + status codes without doubling the
surface. **This is the biggest open design question** (see §7.1).

### 3.4 `Mode` as a required parameter

Airy / elliptic-integral functions take `mode: Mode` explicitly (Zig has no
default args). Pass `.double` for full precision.

### 3.5 Multi-output-only routines surfaced as structs

`elljac(u, m)` (GSL provides no natural form) returns
`Error!struct { sn, cn, dn }` — the pattern future multi-output bindings should
follow.

### 3.6 C integer types at the boundary

`c_int`/`c_uint` for ABI-exact integer params (comptime literals coerce
naturally); `f64` elsewhere.

### 3.7 Scope boundary documented

Following `gsl.zig`'s "Omitted from GSL" convention, the module header lists what
is deferred to raw `c`: array-filling variants, the `gsl_sf_alf.h` workspace
Legendre API, continuum Coulomb waves, Mathieu functions, complex/`result_e10`
forms, and `_err` forms.

---

## 4. Public API surface (`src/gsl_sf.zig`, reached as `libs.gsl_sf`)

```
Result, Error, Mode, check, evaluate, disableDefaultErrorHandler
Top-level fns: multiply, powInt, dawson, clausen, dilog, elljac
Namespaces:    trig, exp, log, erf, gamma, psi, zeta, debye, lambert,
               synchrotron, transport, expint, fermiDirac, bessel, airy,
               ellint, legendre, laguerre, gegenbauer, hermite, coulomb,
               coupling, hyperg
```

---

## 5. Test status

`zig build test` → passes (343 total). The 15 new tests spot-check reference
values against closed forms:

- Γ/factorials/choose/beta at integer points;
- erf reference values + `erf + erfc == 1`;
- ζ(2)=π²/6, η(1)=ln2, Li₂(1)=π²/6;
- exp/log careful variants (expm1 ↔ log1plusx inverse near 0);
- Bessel identities, a tabulated J₀ zero, `Jn == Jnu` cross-check;
- Airy Ai(0) closed form, elliptic K(0)=E(0)=π/2 (explicit `.double` mode);
- `elljac` reducing to sin/cos/1 at m=0;
- orthogonal polynomials (Legendre P₂, Hermite H₃, Gegenbauer C₁, Laguerre L₁);
- Wigner 3j reference value;
- `evaluate` returning value + small error estimate and agreeing with the
  natural wrapper (1- and 2-arg symbols);
- `evaluate` surfacing a domain error as `Error.Domain` (after
  `disableDefaultErrorHandler`).

Spot-checks only — see §7.7 re: an exhaustive `refAllDecls`-style symbol guard.

---

## 6. Files changed this session

- **Added:** `src/gsl_sf.zig` (bindings + docs + tests).
- **Modified:** `src/gsl.zig` — added `pub const sf = @import("gsl_sf.zig");`
  with a doc comment, next to the existing `pub const fft`.
- **Modified:** `src/root.zig` — the interim `libs.gsl_sf` line was added and
  then **removed** once `gsl.sf` became the single access path. (`root.zig` is
  shared with the concurrent LAPACK agent, who has since moved LAPACK into its
  own `src/lapack.zig`; that line was left untouched.)
- No `build.zig` changes.

---

## 7. Design decisions (RESOLVED)

All questions from the first turn were resolved with the maintainer in a second
discussion round. Outcomes:

1. **Error handling → `Error!f64` primary + reserved `result` namespace.** Every
   scalar wrapper will return `Error!f64` (idiomatic, safe, ergonomic value
   access; internally calls the `_e` symbol and maps status → `Error`). The
   val+err (`Result`) tier gets its **own namespace** modeled on
   `stats.weighted`, but is **postponed**: implement it as
   `pub const result = @compileError("... reserved; use evaluate() meanwhile ...")`
   and document it. `evaluate` stays as the interim way to obtain the error
   estimate. Rationale for accepting the `try` cost: no extra numerical work
   (GSL's natural functions are themselves implemented on top of the `_e` form
   and discard `.err`); Zig-side overhead is one well-predicted branch per call,
   negligible next to a special function's own flops. The bare-`f64` GSL
   functions remain reachable via `sf.c.gsl_sf_*` for hot loops that want to opt
   out.
2. **Abort footgun → auto-install the non-aborting handler.** Because GSL's `_e`
   functions still invoke the process-global handler (default: `abort()`) before
   returning a code, the `Error!f64` contract only holds with the no-op handler
   installed. Decision: **auto-install it lazily on first `sf` use** (one-time,
   thread-safe). It is a process-global change to *failure* behavior only (never
   numerics) and also benefits `gsl.zig`/`gsl_fft`; documented as such.
3. **Naming/structure → keep as-is** (grouped by GSL header, math casing).
4. **`Mode` → keep the explicit `mode: Mode` parameter.**
5. **Array/workspace coverage:**
   - **1-D `_array` sequence families → bind now, contiguous `[]f64` out.** Key
     realization: unlike the stats module (which *reads* strided input), GSL's
     `_array` fillers *write a contiguous run with no stride parameter*, so
     there is no zero-copy `Strided` path (honoring a stride would need an
     intermediate copy, violating the no-hidden-allocation ADR). So these take a
     contiguous `[]f64` output and return `Error!void` with a length check.
     Scope: Bessel `Jn/Yn/In/Kn` (± scaled), spherical `jl/yl/il/kl`, Legendre
     `Pl`, Gegenbauer.
   - **`alf` (associated Legendre), `mathieu`, continuum Coulomb waves, and
     complex/`_e10` results → deferred**, documented as reserved. Rationale
     (maintainer delegated the call, weighing implementation complexity): `alf`
     is inherently *multi-axis* (a packed `(l, m)` triangular array — exactly the
     "more than one axis" exclusion), `mathieu` needs an allocated workspace and
     is niche, Coulomb waves are multi-output + array + overflow bookkeeping, and
     complex/`_e10` are a different result shape. All reachable via `sf.c`.
6. **Integration → re-export as `gsl.sf`.** Add `pub const sf =
   @import("gsl_sf.zig");` to `gsl.zig` and **remove** the `libs.gsl_sf` direct
   export, matching the `gsl.fft` precedent (single access path).
7. **Tests → add the exhaustive `refAllDecls`-style pass** exercising every
   declared function, in addition to the closed-form spot-checks.

---

## 8. Coordination notes / gotchas

- `src/gsl_sf.zig` owns its own `c = @cImport({...})` over the special-function
  headers; `gsl.zig` deliberately does not include them. Keep that split (same
  pattern as `gsl_fft.zig`).
- `root.zig` is shared with the concurrent LAPACK agent — coordinate before
  editing; keep special-function logic isolated in `src/gsl_sf.zig`. The pending
  redesign will also touch `gsl.zig` (add `pub const sf`) and `root.zig` (drop
  the `libs.gsl_sf` line) — both single-line, additive/subtractive.
- Watch Zig name-shadowing: `c` is the cImport handle; a namespace and a function
  can share a name across scopes (e.g. `gamma.gamma`, `zeta.zeta`) but avoid
  primitive type names as identifiers.
- Homebrew GSL headers: `/opt/homebrew/include/gsl/`, libs `/opt/homebrew/lib/`.
  `build_config.zig` already links `gsl`. `zig build test` handles linking.
- The 1-D `_array` families write **contiguous** output only — GSL exposes no
  stride parameter for them, so do not try to thread `Strided`/`StridedMut`
  through; take a plain `[]f64`.

---

## 9. Remaining work / implementation plan

The redesign agreed in §7 is **implemented and validated** (`zig build test`
passes). Completed:

1. ✅ Every scalar wrapper returns `Error!f64` via a shared `call(ef, args)`
   driver (calls the `_e` symbol, `try check(status)`, returns `r.val`); all
   tests updated to `try`.
2. ✅ Lazy auto-install of the non-aborting handler through `ensureHandler()`
   (an atomic-guarded, benign-race one-time init), invoked by `call`,
   `evaluate`, `elljac`, and the array fillers.
3. ✅ Reserved `result` namespace as a documented `@compileError` stub (safe
   because Zig analyzes unreferenced decls lazily, same as `qrng`); `evaluate`
   is the interim val+err path.
4. ✅ Re-exported as `gsl.sf` in `gsl.zig`; the `libs.gsl_sf` export was removed.
5. ✅ 1-D `_array` families bound (contiguous `[]f64` out, `Error!void`,
   length-checked via `fillN`/`fillL`): Bessel `Jn/Yn/In/Kn` ± scaled, spherical
   `jl/yl/il/kl`, Legendre `Pl`, Gegenbauer.
6. ✅ Exhaustive coverage test (`sf: every wrapped function is invoked`) calls
   every binding with benign inputs, discarding the value and any domain error,
   to force every extern symbol to link and every signature to compile.
7. ✅ `alf`/`mathieu`/Coulomb-waves/complex/`_e10` remain deferred and documented
   as reserved.

Possible future follow-ups (not scheduled):

- Implement the `result` (val+err) namespace once its ergonomics are wanted.
- Bind the deferred `alf`/`mathieu` families (workspace-owning, `Plan`-style).
- Revisit `_array` output to also accept `NamedArray`/strided targets (would
  require an intermediate contiguous copy, so only if a real need appears).
