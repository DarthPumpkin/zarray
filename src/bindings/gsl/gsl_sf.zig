//! Idiomatic Zig bindings for the GNU Scientific Library's special-function
//! module (`gsl_sf_*`). This is the companion to `gsl.zig` (which covers
//! `gsl_rng`/`gsl_randist`/`gsl_cdf`/`gsl_statistics`); it is kept in a separate
//! file because the special functions are a self-contained GSL subsystem with a
//! uniform calling convention that differs from the RNG/stats surface. It is
//! surfaced to callers as `gsl.sf`.
//!
//! ## Shape of GSL's special functions
//!
//! Every GSL special function comes in two flavors:
//!
//!   - a *natural* form `gsl_sf_<name>(args...)` that returns a bare `double`
//!     (and reports errors only through the global error handler), and
//!   - an *error-form* `gsl_sf_<name>_e(args..., gsl_sf_result *r)` that returns
//!     an `int` status code and writes both the value and an error estimate into
//!     `r` (`{ val, err }`).
//!
//! This binding is built on the **error-form**, which is GSL's actual workhorse
//! (the natural form is itself implemented on top of it):
//!
//!   - Every wrapper returns **`Error!f64`** — idiomatic Zig error handling. On
//!     success you get the value; a GSL domain/overflow/etc. status becomes a
//!     Zig `Error`. Wrappers are grouped into namespaces tracking GSL's headers
//!     (`erf`, `gamma`, `bessel`, `zeta`, ...), so a call reads
//!     `const g = try sf.gamma.gamma(5.0);`.
//!   - The `err` *estimate* (the second field of `gsl_sf_result`) is not exposed
//!     per-function yet. A first-class `result` namespace for it is **reserved**
//!     (see `result`); until then reach it generically with `evaluate`.
//!   - Sequence routines that fill a run of orders at once are bound as the
//!     `*Array` functions, writing into a caller-owned contiguous `[]f64`.
//!
//! If you want the bare `f64` (no error union, e.g. a hot loop that accepts
//! GSL's abort/NaN behavior), call the raw `c.gsl_sf_*` symbol directly.
//!
//! ## Error convention (and the auto-installed handler)
//!
//! GSL reports errors by invoking a process-global handler whose default action
//! is `abort()`. That handler fires even from the `_e` error-forms *before* they
//! return their status code — so for the `Error!f64` contract to actually return
//! an error instead of killing the process, the non-aborting handler must be
//! installed. This module therefore **installs it automatically** (lazily, once)
//! the first time any binding here is called. This is a process-global change to
//! *failure* behavior only — it never affects numerical results — and it also
//! benefits `gsl.zig`/`gsl_fft`, which recommend the same. Call
//! `disableDefaultErrorHandler()` yourself if you want it installed earlier.
//!
//! ## Mode-carrying functions
//!
//! A few families (Airy, elliptic integrals) take a `gsl_mode_t` precision
//! selector, surfaced as the `Mode` enum; pass `.double` for the usual
//! full-precision result.
//!
//! ## Omitted from GSL (reach through `c` for these)
//!
//! This pass covers the scalar-valued functions plus the 1-D sequence fillers.
//! Deferred (they need a different shape — multi-axis packing, workspaces,
//! multi-output, or extended range):
//!
//!   - Associated-Legendre packed `(l, m)` arrays and their workspace/norm API
//!     (`gsl_sf_alf.h`) — inherently multi-axis.
//!   - Mathieu functions (`gsl_sf_mathieu_*`) — require an allocated workspace.
//!   - Continuum Coulomb wave functions (`gsl_sf_coulomb_wave_*`) — multi-output
//!     arrays with overflow bookkeeping; only the bound-state hydrogenic radial
//!     functions are wrapped.
//!   - Complex-valued forms and `gsl_sf_result_e10` (extended-exponent) forms.
//!   - The `_err` forms that propagate an input error estimate.

const std = @import("std");
const testing = std.testing;

/// The raw C API. Use it directly for anything not wrapped here (see the
/// module-level "Omitted from GSL" list), for the bare `f64` natural forms, and
/// to pass `*_e` symbols to `evaluate`.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_mode.h");
    // Umbrella header pulling in every gsl_sf_*.h special-function header.
    @cInclude("gsl/gsl_sf.h");
});

// One-time, lazy install of the non-aborting error handler. Installing the
// no-op handler is idempotent (it just stores a handler pointer), so the guard
// races benignly: at worst a few threads each install it before the flag flips.
var handler_installed = std.atomic.Value(bool).init(false);

inline fn ensureHandler() void {
    if (!handler_installed.load(.monotonic)) {
        _ = c.gsl_set_error_handler_off();
        handler_installed.store(true, .monotonic);
    }
}

/// Replace GSL's default (aborting) error handler with the no-op handler, so
/// that fallible routines return status codes instead of terminating the
/// process. This module installs it automatically on first use; call this
/// yourself only if you want it installed earlier. Toggles the same
/// process-wide handler as `gsl.zig`.
pub fn disableDefaultErrorHandler() void {
    _ = c.gsl_set_error_handler_off();
    handler_installed.store(true, .monotonic);
}

/// A special-function value together with GSL's estimate of its absolute error,
/// mirroring `gsl_sf_result`.
pub const Result = struct {
    val: f64,
    err: f64,
};

/// Zig error set covering the status codes GSL's special functions raise. Codes
/// without a dedicated variant map to `Unspecified`; the raw `c_int` is always
/// available from the `_e` symbol if you need the exact value.
pub const Error = error{
    /// `GSL_EDOM` — input outside the function's domain.
    Domain,
    /// `GSL_ERANGE` — output outside the representable range.
    Range,
    /// `GSL_EOVRFLW` — the result overflowed.
    Overflow,
    /// `GSL_EUNDRFLW` — the result underflowed to zero.
    Underflow,
    /// `GSL_ELOSS` — loss of accuracy (e.g. catastrophic cancellation).
    LossOfAccuracy,
    /// `GSL_EROUND` — failed to converge because of roundoff.
    Roundoff,
    /// `GSL_EMAXITER` — hit the internal iteration limit.
    MaxIterations,
    /// `GSL_EBADLEN`, or a caller-supplied output buffer of the wrong length.
    BadLength,
    /// Any other nonzero GSL status code.
    Unspecified,
};

/// Translate a GSL status code into `Error!void` (`GSL_SUCCESS` -> `{}`).
pub fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EDOM => Error.Domain,
        c.GSL_ERANGE => Error.Range,
        c.GSL_EOVRFLW => Error.Overflow,
        c.GSL_EUNDRFLW => Error.Underflow,
        c.GSL_ELOSS => Error.LossOfAccuracy,
        c.GSL_EROUND => Error.Roundoff,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_EBADLEN => Error.BadLength,
        else => Error.Unspecified,
    };
}

/// Shared driver for the scalar wrappers: install the handler, call the `_e`
/// error-form with the caller's leading `args` plus a `gsl_sf_result *` output,
/// map the status, and return the value.
fn call(comptime ef: anytype, args: anytype) Error!f64 {
    ensureHandler();
    var r: c.gsl_sf_result = undefined;
    try check(@call(.auto, ef, args ++ .{&r}));
    return r.val;
}

/// Call a `gsl_sf_*_e` error-form symbol and return its value **plus error
/// estimate** as a `Result`, or a Zig `Error` on a nonzero status. This is the
/// interim way to obtain the error estimate until the `result` namespace lands.
///
/// `args` is a tuple of the leading arguments *before* the trailing
/// `gsl_sf_result *` output pointer, which this helper supplies:
///
/// ```
/// const r = try sf.evaluate(sf.c.gsl_sf_bessel_Jnu_e, .{ 2.5, 1.0 });
/// // r.val, r.err
/// ```
pub fn evaluate(comptime ef: anytype, args: anytype) Error!Result {
    ensureHandler();
    var r: c.gsl_sf_result = undefined;
    try check(@call(.auto, ef, args ++ .{&r}));
    return .{ .val = r.val, .err = r.err };
}

/// Reserved: a namespace of `Error!Result` (value + error-estimate) variants
/// mirroring the scalar functions, modeled on `stats.weighted`. Not yet
/// implemented — use `evaluate` with the raw `c.gsl_sf_*_e` symbol to obtain the
/// error estimate for now.
pub const result = @compileError(
    "gsl.sf.result (the value+error-estimate tier) is reserved and not yet " ++
        "implemented; use sf.evaluate(c.gsl_sf_<name>_e, .{args}) to get a " ++
        "Result{ val, err } in the meantime.",
);

/// Precision selector for the `gsl_mode_t`-carrying families (Airy functions,
/// elliptic integrals). `.double` requests full double precision and is the
/// right default; `.single`/`.approx` trade accuracy for speed.
pub const Mode = enum(c.gsl_mode_t) {
    double = c.GSL_PREC_DOUBLE,
    single = c.GSL_PREC_SINGLE,
    approx = c.GSL_PREC_APPROX,
};

inline fn modeInt(mode: Mode) c.gsl_mode_t {
    return @intFromEnum(mode);
}

// ===== Elementary operations (gsl_sf_elementary) =============================

/// `x * y` with a GSL error estimate available via the `_e` form. Rarely needed
/// directly; useful mostly for propagating error bars.
pub fn multiply(x: f64, y: f64) Error!f64 {
    return call(c.gsl_sf_multiply_e, .{ x, y });
}

// ===== Powers (gsl_sf_pow_int) ===============================================

/// `x^n` for integer `n`, evaluated by repeated multiplication.
pub fn powInt(x: f64, n: c_int) Error!f64 {
    return call(c.gsl_sf_pow_int_e, .{ x, n });
}

// ===== Trigonometric functions (gsl_sf_trig, gsl_sf_sincos_pi) ===============

/// Trigonometric functions carrying GSL error estimates, plus a few helpers not
/// in libm (`sinc`, `lnsinh`, `lncosh`, argument reduction).
pub const trig = struct {
    /// `sin(x)`.
    pub fn sin(x: f64) Error!f64 {
        return call(c.gsl_sf_sin_e, .{x});
    }
    /// `cos(x)`.
    pub fn cos(x: f64) Error!f64 {
        return call(c.gsl_sf_cos_e, .{x});
    }
    /// `hypot(x, y) = sqrt(x^2 + y^2)`, avoiding overflow.
    pub fn hypot(x: f64, y: f64) Error!f64 {
        return call(c.gsl_sf_hypot_e, .{ x, y });
    }
    /// `sinc(x) = sin(pi x) / (pi x)`.
    pub fn sinc(x: f64) Error!f64 {
        return call(c.gsl_sf_sinc_e, .{x});
    }
    /// `sin(pi x)`, accurate for large/half-integer `x`.
    pub fn sinPi(x: f64) Error!f64 {
        return call(c.gsl_sf_sin_pi_e, .{x});
    }
    /// `cos(pi x)`, accurate for large/half-integer `x`.
    pub fn cosPi(x: f64) Error!f64 {
        return call(c.gsl_sf_cos_pi_e, .{x});
    }
    /// `log(sinh(x))` for `x > 0`.
    pub fn lnSinh(x: f64) Error!f64 {
        return call(c.gsl_sf_lnsinh_e, .{x});
    }
    /// `log(cosh(x))`.
    pub fn lnCosh(x: f64) Error!f64 {
        return call(c.gsl_sf_lncosh_e, .{x});
    }
    /// Reduce an angle to `(-pi, pi]`.
    pub fn angleRestrictSymm(theta: f64) Error!f64 {
        return call(c.gsl_sf_angle_restrict_symm_err_e, .{theta});
    }
    /// Reduce an angle to `[0, 2pi)`.
    pub fn angleRestrictPos(theta: f64) Error!f64 {
        return call(c.gsl_sf_angle_restrict_pos_err_e, .{theta});
    }
};

// ===== Exponential and logarithm (gsl_sf_exp, gsl_sf_log) ====================

/// Exponential-family functions, including the numerically careful relatives
/// `expm1` and `exprel` used to avoid cancellation near zero.
pub const exp = struct {
    /// `exp(x)` with GSL over/underflow reporting.
    pub fn exp(x: f64) Error!f64 {
        return call(c.gsl_sf_exp_e, .{x});
    }
    /// `y * exp(x)`, guarded against intermediate overflow.
    pub fn expMult(x: f64, y: f64) Error!f64 {
        return call(c.gsl_sf_exp_mult_e, .{ x, y });
    }
    /// `exp(x) - 1`, accurate for small `x`.
    pub fn expm1(x: f64) Error!f64 {
        return call(c.gsl_sf_expm1_e, .{x});
    }
    /// `(exp(x) - 1) / x`.
    pub fn exprel(x: f64) Error!f64 {
        return call(c.gsl_sf_exprel_e, .{x});
    }
    /// `2 (exp(x) - 1 - x) / x^2`.
    pub fn exprel2(x: f64) Error!f64 {
        return call(c.gsl_sf_exprel_2_e, .{x});
    }
    /// `n`-th order relative exponential, `= 1F1(1, 1+n, x)`.
    pub fn exprelN(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_exprel_n_e, .{ n, x });
    }
};

/// Logarithm-family functions, including the cancellation-safe `log1plusx`.
pub const log = struct {
    /// `log(x)` with GSL domain reporting.
    pub fn log(x: f64) Error!f64 {
        return call(c.gsl_sf_log_e, .{x});
    }
    /// `log(|x|)`.
    pub fn logAbs(x: f64) Error!f64 {
        return call(c.gsl_sf_log_abs_e, .{x});
    }
    /// `log(1 + x)`, accurate for small `x`.
    pub fn log1plusx(x: f64) Error!f64 {
        return call(c.gsl_sf_log_1plusx_e, .{x});
    }
    /// `log(1 + x) - x`, accurate for small `x`.
    pub fn log1plusxMx(x: f64) Error!f64 {
        return call(c.gsl_sf_log_1plusx_mx_e, .{x});
    }
};

// ===== Error functions (gsl_sf_erf) ==========================================

/// The error function and its relatives, including the Gaussian probability
/// helpers `Z` (density) and `Q` (upper tail) and the hazard function.
pub const erf = struct {
    /// `erf(x) = (2/sqrt(pi)) integral_0^x exp(-t^2) dt`.
    pub fn erf(x: f64) Error!f64 {
        return call(c.gsl_sf_erf_e, .{x});
    }
    /// `erfc(x) = 1 - erf(x)`.
    pub fn erfc(x: f64) Error!f64 {
        return call(c.gsl_sf_erfc_e, .{x});
    }
    /// `log(erfc(x))`, accurate in the far tail.
    pub fn logErfc(x: f64) Error!f64 {
        return call(c.gsl_sf_log_erfc_e, .{x});
    }
    /// Gaussian density `Z(x) = (1/sqrt(2pi)) exp(-x^2/2)`.
    pub fn z(x: f64) Error!f64 {
        return call(c.gsl_sf_erf_Z_e, .{x});
    }
    /// Gaussian upper-tail probability `Q(x)`.
    pub fn q(x: f64) Error!f64 {
        return call(c.gsl_sf_erf_Q_e, .{x});
    }
    /// Hazard function (inverse Mills ratio) `H(x) = Z(x)/Q(x)`.
    pub fn hazard(x: f64) Error!f64 {
        return call(c.gsl_sf_hazard_e, .{x});
    }
};

// ===== Gamma and beta (gsl_sf_gamma) =========================================

/// The gamma function and its relatives: log-gamma, reciprocal gamma,
/// factorials, binomial coefficients, Pochhammer symbols, incomplete gamma, and
/// the beta functions.
pub const gamma = struct {
    /// `Gamma(x)`.
    pub fn gamma(x: f64) Error!f64 {
        return call(c.gsl_sf_gamma_e, .{x});
    }
    /// `log|Gamma(x)|`.
    pub fn lnGamma(x: f64) Error!f64 {
        return call(c.gsl_sf_lngamma_e, .{x});
    }
    /// Regulated gamma `Gamma*(x) = Gamma(x) / (sqrt(2pi) x^(x-1/2) e^-x)`.
    pub fn gammaStar(x: f64) Error!f64 {
        return call(c.gsl_sf_gammastar_e, .{x});
    }
    /// `1 / Gamma(x)`.
    pub fn gammaInv(x: f64) Error!f64 {
        return call(c.gsl_sf_gammainv_e, .{x});
    }
    /// `x^n / n!`.
    pub fn taylorCoeff(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_taylorcoeff_e, .{ n, x });
    }
    /// `n!`.
    pub fn fact(n: c_uint) Error!f64 {
        return call(c.gsl_sf_fact_e, .{n});
    }
    /// `n!! = n(n-2)(n-4)...`.
    pub fn doubleFact(n: c_uint) Error!f64 {
        return call(c.gsl_sf_doublefact_e, .{n});
    }
    /// `log(n!)`.
    pub fn lnFact(n: c_uint) Error!f64 {
        return call(c.gsl_sf_lnfact_e, .{n});
    }
    /// `log(n!!)`.
    pub fn lnDoubleFact(n: c_uint) Error!f64 {
        return call(c.gsl_sf_lndoublefact_e, .{n});
    }
    /// `C(n, m) = n! / (m! (n-m)!)`.
    pub fn choose(n: c_uint, m: c_uint) Error!f64 {
        return call(c.gsl_sf_choose_e, .{ n, m });
    }
    /// `log C(n, m)`.
    pub fn lnChoose(n: c_uint, m: c_uint) Error!f64 {
        return call(c.gsl_sf_lnchoose_e, .{ n, m });
    }
    /// Pochhammer symbol `(a)_x = Gamma(a+x)/Gamma(a)`.
    pub fn poch(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_poch_e, .{ a, x });
    }
    /// `log|(a)_x|`.
    pub fn lnPoch(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_lnpoch_e, .{ a, x });
    }
    /// Relative Pochhammer `((a)_x - 1)/x`.
    pub fn pochRel(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_pochrel_e, .{ a, x });
    }
    /// Normalized upper incomplete gamma `Q(a, x)`.
    pub fn incQ(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gamma_inc_Q_e, .{ a, x });
    }
    /// Normalized lower incomplete gamma `P(a, x)`.
    pub fn incP(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gamma_inc_P_e, .{ a, x });
    }
    /// Non-normalized upper incomplete gamma `Gamma(a, x)`.
    pub fn inc(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gamma_inc_e, .{ a, x });
    }
    /// Beta function `B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)`.
    pub fn beta(a: f64, b: f64) Error!f64 {
        return call(c.gsl_sf_beta_e, .{ a, b });
    }
    /// `log B(a, b)`.
    pub fn lnBeta(a: f64, b: f64) Error!f64 {
        return call(c.gsl_sf_lnbeta_e, .{ a, b });
    }
    /// Normalized incomplete beta `I_x(a, b) = B_x(a,b)/B(a,b)`.
    pub fn incBeta(a: f64, b: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_beta_inc_e, .{ a, b, x });
    }
};

// ===== Psi / polygamma (gsl_sf_psi) ==========================================

/// The digamma function `psi = d/dx log Gamma`, the trigamma `psi^(1)`, and the
/// general polygamma `psi^(n)`.
pub const psi = struct {
    /// Digamma `psi(x)`.
    pub fn psi(x: f64) Error!f64 {
        return call(c.gsl_sf_psi_e, .{x});
    }
    /// Digamma at a positive integer, `psi(n)`.
    pub fn psiInt(n: c_int) Error!f64 {
        return call(c.gsl_sf_psi_int_e, .{n});
    }
    /// `Re[psi(1 + iy)]`.
    pub fn psi1piy(y: f64) Error!f64 {
        return call(c.gsl_sf_psi_1piy_e, .{y});
    }
    /// Trigamma `psi^(1)(x)`.
    pub fn psi1(x: f64) Error!f64 {
        return call(c.gsl_sf_psi_1_e, .{x});
    }
    /// Trigamma at a positive integer, `psi^(1)(n)`.
    pub fn psi1Int(n: c_int) Error!f64 {
        return call(c.gsl_sf_psi_1_int_e, .{n});
    }
    /// Polygamma `psi^(n)(x)` for `n >= 0`, `x > 0`.
    pub fn psiN(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_psi_n_e, .{ n, x });
    }
};

// ===== Zeta, eta (gsl_sf_zeta) ===============================================

/// The Riemann and Hurwitz zeta functions plus the Dirichlet eta function, with
/// `...m1` (zeta - 1) variants for accuracy at large argument.
pub const zeta = struct {
    /// Riemann zeta `zeta(s)`, `s != 1`.
    pub fn zeta(s: f64) Error!f64 {
        return call(c.gsl_sf_zeta_e, .{s});
    }
    /// Riemann zeta at an integer, `zeta(n)`, `n != 1`.
    pub fn zetaInt(n: c_int) Error!f64 {
        return call(c.gsl_sf_zeta_int_e, .{n});
    }
    /// `zeta(s) - 1`.
    pub fn zetaM1(s: f64) Error!f64 {
        return call(c.gsl_sf_zetam1_e, .{s});
    }
    /// `zeta(n) - 1` at an integer.
    pub fn zetaM1Int(n: c_int) Error!f64 {
        return call(c.gsl_sf_zetam1_int_e, .{n});
    }
    /// Hurwitz zeta `zeta(s, q) = sum_{k>=0} (k+q)^-s`.
    pub fn hurwitz(s: f64, qq: f64) Error!f64 {
        return call(c.gsl_sf_hzeta_e, .{ s, qq });
    }
    /// Dirichlet eta `eta(s) = (1 - 2^(1-s)) zeta(s)`.
    pub fn eta(s: f64) Error!f64 {
        return call(c.gsl_sf_eta_e, .{s});
    }
    /// Dirichlet eta at an integer, `eta(n)`.
    pub fn etaInt(n: c_int) Error!f64 {
        return call(c.gsl_sf_eta_int_e, .{n});
    }
};

// ===== Dawson, Clausen, Dilog (misc single-argument) =========================

/// Dawson's integral `D(x) = exp(-x^2) integral_0^x exp(t^2) dt`.
pub fn dawson(x: f64) Error!f64 {
    return call(c.gsl_sf_dawson_e, .{x});
}

/// Clausen function `Cl_2(x) = -integral_0^x log|2 sin(t/2)| dt`.
pub fn clausen(x: f64) Error!f64 {
    return call(c.gsl_sf_clausen_e, .{x});
}

/// Dilogarithm `Li_2(x)` (real argument).
pub fn dilog(x: f64) Error!f64 {
    return call(c.gsl_sf_dilog_e, .{x});
}

// ===== Debye functions (gsl_sf_debye) ========================================

/// The Debye functions `D_n(x) = (n/x^n) integral_0^x t^n/(e^t - 1) dt`, for
/// `n = 1..6`.
pub const debye = struct {
    pub fn d1(x: f64) Error!f64 {
        return call(c.gsl_sf_debye_1_e, .{x});
    }
    pub fn d2(x: f64) Error!f64 {
        return call(c.gsl_sf_debye_2_e, .{x});
    }
    pub fn d3(x: f64) Error!f64 {
        return call(c.gsl_sf_debye_3_e, .{x});
    }
    pub fn d4(x: f64) Error!f64 {
        return call(c.gsl_sf_debye_4_e, .{x});
    }
    pub fn d5(x: f64) Error!f64 {
        return call(c.gsl_sf_debye_5_e, .{x});
    }
    pub fn d6(x: f64) Error!f64 {
        return call(c.gsl_sf_debye_6_e, .{x});
    }
};

// ===== Lambert W (gsl_sf_lambert) ============================================

/// Branches of the Lambert W function, the inverse of `w -> w e^w`.
pub const lambert = struct {
    /// Principal branch `W_0(x)`, real for `x >= -1/e`.
    pub fn w0(x: f64) Error!f64 {
        return call(c.gsl_sf_lambert_W0_e, .{x});
    }
    /// Secondary real branch `W_{-1}(x)`, real for `-1/e <= x < 0`.
    pub fn wm1(x: f64) Error!f64 {
        return call(c.gsl_sf_lambert_Wm1_e, .{x});
    }
};

// ===== Synchrotron, Transport (gsl_sf_synchrotron, gsl_sf_transport) =========

/// The synchrotron radiation functions.
pub const synchrotron = struct {
    /// First synchrotron function `x integral_x^inf K_{5/3}(t) dt`.
    pub fn s1(x: f64) Error!f64 {
        return call(c.gsl_sf_synchrotron_1_e, .{x});
    }
    /// Second synchrotron function `x K_{2/3}(x)`.
    pub fn s2(x: f64) Error!f64 {
        return call(c.gsl_sf_synchrotron_2_e, .{x});
    }
};

/// The transport functions `J(n, x) = integral_0^x t^n e^t / (e^t - 1)^2 dt`,
/// for `n = 2..5`.
pub const transport = struct {
    pub fn j2(x: f64) Error!f64 {
        return call(c.gsl_sf_transport_2_e, .{x});
    }
    pub fn j3(x: f64) Error!f64 {
        return call(c.gsl_sf_transport_3_e, .{x});
    }
    pub fn j4(x: f64) Error!f64 {
        return call(c.gsl_sf_transport_4_e, .{x});
    }
    pub fn j5(x: f64) Error!f64 {
        return call(c.gsl_sf_transport_5_e, .{x});
    }
};

// ===== Exponential integrals (gsl_sf_expint) =================================

/// The exponential, hyperbolic, and trigonometric integrals.
pub const expint = struct {
    /// `E_1(x) = integral_1^inf e^{-xt}/t dt`.
    pub fn e1(x: f64) Error!f64 {
        return call(c.gsl_sf_expint_E1_e, .{x});
    }
    /// `E_2(x)`.
    pub fn e2(x: f64) Error!f64 {
        return call(c.gsl_sf_expint_E2_e, .{x});
    }
    /// `E_n(x)`.
    pub fn en(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_expint_En_e, .{ n, x });
    }
    /// Exponential integral `Ei(x)` (principal value).
    pub fn ei(x: f64) Error!f64 {
        return call(c.gsl_sf_expint_Ei_e, .{x});
    }
    /// Hyperbolic sine integral `Shi(x)`.
    pub fn shi(x: f64) Error!f64 {
        return call(c.gsl_sf_Shi_e, .{x});
    }
    /// Hyperbolic cosine integral `Chi(x)`.
    pub fn chi(x: f64) Error!f64 {
        return call(c.gsl_sf_Chi_e, .{x});
    }
    /// `Ei_3(x) = integral_0^x exp(-t^3) dt`.
    pub fn exp3(x: f64) Error!f64 {
        return call(c.gsl_sf_expint_3_e, .{x});
    }
    /// Sine integral `Si(x) = integral_0^x sin(t)/t dt`.
    pub fn si(x: f64) Error!f64 {
        return call(c.gsl_sf_Si_e, .{x});
    }
    /// Cosine integral `Ci(x)`.
    pub fn ci(x: f64) Error!f64 {
        return call(c.gsl_sf_Ci_e, .{x});
    }
    /// Arctangent integral `integral_0^x arctan(t)/t dt`.
    pub fn atanInt(x: f64) Error!f64 {
        return call(c.gsl_sf_atanint_e, .{x});
    }
};

// ===== Fermi-Dirac integrals (gsl_sf_fermi_dirac) ============================

/// The complete Fermi-Dirac integrals `F_j(x)`, for the tabulated orders GSL
/// provides plus the incomplete order-0 form.
pub const fermiDirac = struct {
    /// `F_{-1}(x) = e^x / (1 + e^x)`.
    pub fn m1(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_m1_e, .{x});
    }
    /// `F_0(x) = log(1 + e^x)`.
    pub fn f0(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_0_e, .{x});
    }
    /// `F_1(x)`.
    pub fn f1(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_1_e, .{x});
    }
    /// `F_2(x)`.
    pub fn f2(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_2_e, .{x});
    }
    /// Integer order `F_j(x)`.
    pub fn int(j: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_int_e, .{ j, x });
    }
    /// `F_{-1/2}(x)`.
    pub fn mhalf(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_mhalf_e, .{x});
    }
    /// `F_{1/2}(x)`.
    pub fn half(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_half_e, .{x});
    }
    /// `F_{3/2}(x)`.
    pub fn threeHalf(x: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_3half_e, .{x});
    }
    /// Incomplete order-0 Fermi-Dirac integral with lower limit `b`.
    pub fn inc0(x: f64, b: f64) Error!f64 {
        return call(c.gsl_sf_fermi_dirac_inc_0_e, .{ x, b });
    }
};

// ===== Bessel functions (gsl_sf_bessel) ======================================

/// Cylindrical, spherical, and irregular Bessel functions of integer and
/// fractional order, their exponentially scaled variants, low-order zeros, and
/// the `*Array` sequence fillers.
///
/// Naming follows GSL: uppercase `J/Y/I/K` are cylindrical, lowercase `j/y/i/k`
/// are spherical, `nu` suffixes take a real (fractional) order, and `Scaled`
/// suffixes return the exponentially rescaled function that stays finite for
/// large argument.
///
/// The `*Array` functions fill a caller-owned contiguous `[]f64` (GSL exposes no
/// stride here): the `Jn`-style ones write orders `nmin..=nmax` (so
/// `out.len == nmax - nmin + 1`), the spherical/`l`-indexed ones write orders
/// `0..=lmax` (so `out.len == lmax + 1`). A wrong length is `error.BadLength`.
pub const bessel = struct {
    // --- Cylindrical, regular: J ---
    pub fn J0(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_J0_e, .{x});
    }
    pub fn J1(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_J1_e, .{x});
    }
    /// Integer order `J_n(x)`.
    pub fn Jn(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Jn_e, .{ n, x });
    }
    /// Fractional order `J_nu(x)`.
    pub fn Jnu(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Jnu_e, .{ nu, x });
    }
    /// Fill `J_{nmin..=nmax}(x)`.
    pub fn JnArray(nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
        return fillN(c.gsl_sf_bessel_Jn_array, nmin, nmax, x, out);
    }

    // --- Cylindrical, irregular: Y ---
    pub fn Y0(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Y0_e, .{x});
    }
    pub fn Y1(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Y1_e, .{x});
    }
    pub fn Yn(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Yn_e, .{ n, x });
    }
    pub fn Ynu(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Ynu_e, .{ nu, x });
    }
    /// Fill `Y_{nmin..=nmax}(x)`.
    pub fn YnArray(nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
        return fillN(c.gsl_sf_bessel_Yn_array, nmin, nmax, x, out);
    }

    // --- Modified, regular: I (and e^{-|x|}-scaled) ---
    pub fn I0(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_I0_e, .{x});
    }
    pub fn I1(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_I1_e, .{x});
    }
    pub fn In(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_In_e, .{ n, x });
    }
    pub fn Inu(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Inu_e, .{ nu, x });
    }
    pub fn I0Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_I0_scaled_e, .{x});
    }
    pub fn I1Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_I1_scaled_e, .{x});
    }
    pub fn InScaled(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_In_scaled_e, .{ n, x });
    }
    pub fn InuScaled(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Inu_scaled_e, .{ nu, x });
    }
    /// Fill `I_{nmin..=nmax}(x)`.
    pub fn InArray(nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
        return fillN(c.gsl_sf_bessel_In_array, nmin, nmax, x, out);
    }
    /// Fill scaled `I_{nmin..=nmax}(x)`.
    pub fn InScaledArray(nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
        return fillN(c.gsl_sf_bessel_In_scaled_array, nmin, nmax, x, out);
    }

    // --- Modified, irregular: K (and e^{|x|}-scaled) ---
    pub fn K0(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_K0_e, .{x});
    }
    pub fn K1(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_K1_e, .{x});
    }
    pub fn Kn(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Kn_e, .{ n, x });
    }
    pub fn Knu(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Knu_e, .{ nu, x });
    }
    pub fn lnKnu(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_lnKnu_e, .{ nu, x });
    }
    pub fn K0Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_K0_scaled_e, .{x});
    }
    pub fn K1Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_K1_scaled_e, .{x});
    }
    pub fn KnScaled(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Kn_scaled_e, .{ n, x });
    }
    pub fn KnuScaled(nu: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_Knu_scaled_e, .{ nu, x });
    }
    /// Fill `K_{nmin..=nmax}(x)`.
    pub fn KnArray(nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
        return fillN(c.gsl_sf_bessel_Kn_array, nmin, nmax, x, out);
    }
    /// Fill scaled `K_{nmin..=nmax}(x)`.
    pub fn KnScaledArray(nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
        return fillN(c.gsl_sf_bessel_Kn_scaled_array, nmin, nmax, x, out);
    }

    // --- Spherical, regular: j ---
    pub fn j0(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_j0_e, .{x});
    }
    pub fn j1(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_j1_e, .{x});
    }
    pub fn j2(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_j2_e, .{x});
    }
    /// Order-`l` spherical `j_l(x)`.
    pub fn jl(l: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_jl_e, .{ l, x });
    }
    /// Fill spherical `j_{0..=lmax}(x)`.
    pub fn jlArray(lmax: c_int, x: f64, out: []f64) Error!void {
        return fillL(c.gsl_sf_bessel_jl_array, lmax, x, out);
    }

    // --- Spherical, irregular: y ---
    pub fn y0(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_y0_e, .{x});
    }
    pub fn y1(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_y1_e, .{x});
    }
    pub fn y2(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_y2_e, .{x});
    }
    pub fn yl(l: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_yl_e, .{ l, x });
    }
    /// Fill spherical `y_{0..=lmax}(x)`.
    pub fn ylArray(lmax: c_int, x: f64, out: []f64) Error!void {
        return fillL(c.gsl_sf_bessel_yl_array, lmax, x, out);
    }

    // --- Spherical modified, scaled: i / k ---
    pub fn i0Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_i0_scaled_e, .{x});
    }
    pub fn i1Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_i1_scaled_e, .{x});
    }
    pub fn i2Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_i2_scaled_e, .{x});
    }
    pub fn ilScaled(l: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_il_scaled_e, .{ l, x });
    }
    /// Fill scaled spherical `i_{0..=lmax}(x)`.
    pub fn ilScaledArray(lmax: c_int, x: f64, out: []f64) Error!void {
        return fillL(c.gsl_sf_bessel_il_scaled_array, lmax, x, out);
    }
    pub fn k0Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_k0_scaled_e, .{x});
    }
    pub fn k1Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_k1_scaled_e, .{x});
    }
    pub fn k2Scaled(x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_k2_scaled_e, .{x});
    }
    pub fn klScaled(l: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_bessel_kl_scaled_e, .{ l, x });
    }
    /// Fill scaled spherical `k_{0..=lmax}(x)`.
    pub fn klScaledArray(lmax: c_int, x: f64, out: []f64) Error!void {
        return fillL(c.gsl_sf_bessel_kl_scaled_array, lmax, x, out);
    }

    // --- Positive zeros ---
    /// `s`-th positive zero of `J_0`.
    pub fn zeroJ0(s: c_uint) Error!f64 {
        return call(c.gsl_sf_bessel_zero_J0_e, .{s});
    }
    /// `s`-th positive zero of `J_1`.
    pub fn zeroJ1(s: c_uint) Error!f64 {
        return call(c.gsl_sf_bessel_zero_J1_e, .{s});
    }
    /// `s`-th positive zero of `J_nu`.
    pub fn zeroJnu(nu: f64, s: c_uint) Error!f64 {
        return call(c.gsl_sf_bessel_zero_Jnu_e, .{ nu, s });
    }
};

// ===== Airy functions (gsl_sf_airy) ==========================================

/// The Airy functions `Ai`/`Bi`, their derivatives, exponentially scaled
/// variants, and zeros. The evaluated (non-zero) functions take a `Mode`.
pub const airy = struct {
    pub fn Ai(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Ai_e, .{ x, modeInt(mode) });
    }
    pub fn Bi(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Bi_e, .{ x, modeInt(mode) });
    }
    pub fn AiScaled(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Ai_scaled_e, .{ x, modeInt(mode) });
    }
    pub fn BiScaled(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Bi_scaled_e, .{ x, modeInt(mode) });
    }
    pub fn AiDeriv(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Ai_deriv_e, .{ x, modeInt(mode) });
    }
    pub fn BiDeriv(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Bi_deriv_e, .{ x, modeInt(mode) });
    }
    pub fn AiDerivScaled(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Ai_deriv_scaled_e, .{ x, modeInt(mode) });
    }
    pub fn BiDerivScaled(x: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_airy_Bi_deriv_scaled_e, .{ x, modeInt(mode) });
    }
    /// `s`-th zero of `Ai`.
    pub fn zeroAi(s: c_uint) Error!f64 {
        return call(c.gsl_sf_airy_zero_Ai_e, .{s});
    }
    /// `s`-th zero of `Bi`.
    pub fn zeroBi(s: c_uint) Error!f64 {
        return call(c.gsl_sf_airy_zero_Bi_e, .{s});
    }
    /// `s`-th zero of `Ai'`.
    pub fn zeroAiDeriv(s: c_uint) Error!f64 {
        return call(c.gsl_sf_airy_zero_Ai_deriv_e, .{s});
    }
    /// `s`-th zero of `Bi'`.
    pub fn zeroBiDeriv(s: c_uint) Error!f64 {
        return call(c.gsl_sf_airy_zero_Bi_deriv_e, .{s});
    }
};

// ===== Elliptic integrals (gsl_sf_ellint, gsl_sf_elljac) =====================

/// Legendre and Carlson forms of the elliptic integrals. All take a `Mode`.
/// Legendre forms use the modulus `k` (not the parameter `m = k^2`) and an
/// amplitude `phi` for the incomplete cases, matching GSL's convention.
pub const ellint = struct {
    /// Complete `K(k)`.
    pub fn kComp(k: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_Kcomp_e, .{ k, modeInt(mode) });
    }
    /// Complete `E(k)`.
    pub fn eComp(k: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_Ecomp_e, .{ k, modeInt(mode) });
    }
    /// Complete `Pi(k, n)`.
    pub fn pComp(k: f64, n: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_Pcomp_e, .{ k, n, modeInt(mode) });
    }
    /// Complete `D(k)`.
    pub fn dComp(k: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_Dcomp_e, .{ k, modeInt(mode) });
    }
    /// Incomplete `F(phi, k)`.
    pub fn f(phi: f64, k: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_F_e, .{ phi, k, modeInt(mode) });
    }
    /// Incomplete `E(phi, k)`.
    pub fn e(phi: f64, k: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_E_e, .{ phi, k, modeInt(mode) });
    }
    /// Incomplete `Pi(phi, k, n)`.
    pub fn p(phi: f64, k: f64, n: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_P_e, .{ phi, k, n, modeInt(mode) });
    }
    /// Incomplete `D(phi, k)`.
    pub fn d(phi: f64, k: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_D_e, .{ phi, k, modeInt(mode) });
    }
    /// Carlson `R_C(x, y)`.
    pub fn rc(x: f64, y: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_RC_e, .{ x, y, modeInt(mode) });
    }
    /// Carlson `R_D(x, y, z)`.
    pub fn rd(x: f64, y: f64, zz: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_RD_e, .{ x, y, zz, modeInt(mode) });
    }
    /// Carlson `R_F(x, y, z)`.
    pub fn rf(x: f64, y: f64, zz: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_RF_e, .{ x, y, zz, modeInt(mode) });
    }
    /// Carlson `R_J(x, y, z, p)`.
    pub fn rj(x: f64, y: f64, zz: f64, pp: f64, mode: Mode) Error!f64 {
        return call(c.gsl_sf_ellint_RJ_e, .{ x, y, zz, pp, modeInt(mode) });
    }
};

/// The three Jacobian elliptic functions `sn`, `cn`, `dn` of argument `u` and
/// parameter `m`. GSL only provides the error-form, so this returns them
/// together (or an `Error`).
pub fn elljac(u: f64, m: f64) Error!struct { sn: f64, cn: f64, dn: f64 } {
    ensureHandler();
    var sn: f64 = undefined;
    var cn: f64 = undefined;
    var dn: f64 = undefined;
    try check(c.gsl_sf_elljac_e(u, m, &sn, &cn, &dn));
    return .{ .sn = sn, .cn = cn, .dn = dn };
}

// ===== Legendre functions (gsl_sf_legendre) ==================================

/// Legendre polynomials `P_l`, second-kind `Q_l`, associated `P_l^m` (with the
/// spherical-harmonic normalized form), and the conical / `H3d` families.
///
/// Only the scalar entry points (plus the 1-D `plArray`) are wrapped here; the
/// packed-array associated-Legendre API (`gsl_sf_alf.h`, a `(l, m)` triangle) is
/// left to the raw `c` API for now.
pub const legendre = struct {
    /// Legendre polynomial `P_l(x)`.
    pub fn pl(l: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_legendre_Pl_e, .{ l, x });
    }
    /// Fill `P_{0..=lmax}(x)`.
    pub fn plArray(lmax: c_int, x: f64, out: []f64) Error!void {
        ensureHandler();
        if (lmax < 0) return Error.BadLength;
        if (out.len != @as(usize, @intCast(lmax + 1))) return Error.BadLength;
        try check(c.gsl_sf_legendre_Pl_array(lmax, x, out.ptr));
    }
    /// Second-kind Legendre `Q_0(x)`.
    pub fn q0(x: f64) Error!f64 {
        return call(c.gsl_sf_legendre_Q0_e, .{x});
    }
    /// Second-kind Legendre `Q_1(x)`.
    pub fn q1(x: f64) Error!f64 {
        return call(c.gsl_sf_legendre_Q1_e, .{x});
    }
    /// Second-kind Legendre `Q_l(x)`.
    pub fn ql(l: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_legendre_Ql_e, .{ l, x });
    }
    /// Associated Legendre `P_l^m(x)` (GSL's default normalization).
    pub fn plm(l: c_int, m: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_legendre_Plm_e, .{ l, m, x });
    }
    /// Spherical-harmonic normalized associated Legendre function.
    pub fn sphPlm(l: c_int, m: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_legendre_sphPlm_e, .{ l, m, x });
    }
    /// Conical (Mehler) function `P^{-1/2}_{-1/2+i lambda}(x)`.
    pub fn conicalHalf(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_conicalP_half_e, .{ lambda, x });
    }
    /// Conical function of order `-1/2`.
    pub fn conicalMhalf(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_conicalP_mhalf_e, .{ lambda, x });
    }
    /// Conical function `P^0`.
    pub fn conical0(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_conicalP_0_e, .{ lambda, x });
    }
    /// Conical function `P^1`.
    pub fn conical1(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_conicalP_1_e, .{ lambda, x });
    }
    /// Regular spherical conical function.
    pub fn conicalSphReg(l: c_int, lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_conicalP_sph_reg_e, .{ l, lambda, x });
    }
    /// Regular cylindrical conical function.
    pub fn conicalCylReg(m: c_int, lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_conicalP_cyl_reg_e, .{ m, lambda, x });
    }
    /// `l = 0` radial eigenfunction on the 3d hyperbolic space.
    pub fn h3d0(lambda: f64, eta: f64) Error!f64 {
        return call(c.gsl_sf_legendre_H3d_0_e, .{ lambda, eta });
    }
    /// `l = 1` radial eigenfunction on the 3d hyperbolic space.
    pub fn h3d1(lambda: f64, eta: f64) Error!f64 {
        return call(c.gsl_sf_legendre_H3d_1_e, .{ lambda, eta });
    }
    /// Order-`l` radial eigenfunction on the 3d hyperbolic space.
    pub fn h3d(l: c_int, lambda: f64, eta: f64) Error!f64 {
        return call(c.gsl_sf_legendre_H3d_e, .{ l, lambda, eta });
    }
};

// ===== Laguerre, Gegenbauer, Hermite (orthogonal polynomials) ================

/// Generalized Laguerre polynomials `L_n^a(x)`.
pub const laguerre = struct {
    pub fn l1(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_laguerre_1_e, .{ a, x });
    }
    pub fn l2(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_laguerre_2_e, .{ a, x });
    }
    pub fn l3(a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_laguerre_3_e, .{ a, x });
    }
    /// Order-`n` generalized Laguerre `L_n^a(x)`.
    pub fn ln(n: c_int, a: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_laguerre_n_e, .{ n, a, x });
    }
};

/// Gegenbauer (ultraspherical) polynomials `C_n^lambda(x)`.
pub const gegenbauer = struct {
    pub fn c1(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gegenpoly_1_e, .{ lambda, x });
    }
    pub fn c2(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gegenpoly_2_e, .{ lambda, x });
    }
    pub fn c3(lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gegenpoly_3_e, .{ lambda, x });
    }
    /// Order-`n` Gegenbauer `C_n^lambda(x)`.
    pub fn cn(n: c_int, lambda: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_gegenpoly_n_e, .{ n, lambda, x });
    }
    /// Fill `C_{0..=nmax}^lambda(x)`.
    pub fn array(nmax: c_int, lambda: f64, x: f64, out: []f64) Error!void {
        ensureHandler();
        if (nmax < 0) return Error.BadLength;
        if (out.len != @as(usize, @intCast(nmax + 1))) return Error.BadLength;
        try check(c.gsl_sf_gegenpoly_array(nmax, lambda, x, out.ptr));
    }
};

/// Hermite polynomials and functions. `physicist` is the classical `H_n`,
/// `probabilist` is `He_n`, and `func` is the Hermite function
/// `psi_n(x) = (n! 2^n sqrt(pi))^{-1/2} e^{-x^2/2} H_n(x)`.
pub const hermite = struct {
    /// Physicists' Hermite polynomial `H_n(x)`.
    pub fn physicist(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hermite_e, .{ n, x });
    }
    /// Probabilists' Hermite polynomial `He_n(x)`.
    pub fn probabilist(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hermite_prob_e, .{ n, x });
    }
    /// Hermite function `psi_n(x)`.
    pub fn func(n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hermite_func_e, .{ n, x });
    }
    /// `m`-th derivative of `H_n(x)`.
    pub fn physicistDeriv(m: c_int, n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hermite_deriv_e, .{ m, n, x });
    }
    /// `m`-th derivative of `He_n(x)`.
    pub fn probabilistDeriv(m: c_int, n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hermite_prob_deriv_e, .{ m, n, x });
    }
    /// `s`-th zero of `H_n`.
    pub fn physicistZero(n: c_int, s: c_int) Error!f64 {
        return call(c.gsl_sf_hermite_zero_e, .{ n, s });
    }
    /// `s`-th zero of `He_n`.
    pub fn probabilistZero(n: c_int, s: c_int) Error!f64 {
        return call(c.gsl_sf_hermite_prob_zero_e, .{ n, s });
    }
    /// `s`-th zero of the Hermite function `psi_n`.
    pub fn funcZero(n: c_int, s: c_int) Error!f64 {
        return call(c.gsl_sf_hermite_func_zero_e, .{ n, s });
    }
};

// ===== Coulomb (gsl_sf_coulomb) ==============================================

/// The bound-state hydrogenic radial wave functions. The continuum Coulomb wave
/// functions (`gsl_sf_coulomb_wave_*`, array-valued) are left to the raw `c`
/// API.
pub const coulomb = struct {
    /// Lowest hydrogenic radial function `R_1(Z, r)`.
    pub fn hydrogenicR1(zz: f64, r: f64) Error!f64 {
        return call(c.gsl_sf_hydrogenicR_1_e, .{ zz, r });
    }
    /// Hydrogenic radial function `R_{n,l}(Z, r)`.
    pub fn hydrogenicR(n: c_int, l: c_int, zz: f64, r: f64) Error!f64 {
        return call(c.gsl_sf_hydrogenicR_e, .{ n, l, zz, r });
    }
};

// ===== Coupling coefficients (gsl_sf_coupling) ===============================

/// Wigner 3j/6j/9j angular-momentum coupling coefficients. Every argument is
/// *twice* the corresponding angular momentum (so half-integers are exactly
/// representable), matching GSL's `two_j*` convention.
pub const coupling = struct {
    /// Wigner 3j symbol.
    pub fn threeJ(two_ja: c_int, two_jb: c_int, two_jc: c_int, two_ma: c_int, two_mb: c_int, two_mc: c_int) Error!f64 {
        return call(c.gsl_sf_coupling_3j_e, .{ two_ja, two_jb, two_jc, two_ma, two_mb, two_mc });
    }
    /// Wigner 6j symbol.
    pub fn sixJ(two_ja: c_int, two_jb: c_int, two_jc: c_int, two_jd: c_int, two_je: c_int, two_jf: c_int) Error!f64 {
        return call(c.gsl_sf_coupling_6j_e, .{ two_ja, two_jb, two_jc, two_jd, two_je, two_jf });
    }
    /// Wigner 9j symbol.
    pub fn nineJ(
        two_ja: c_int,
        two_jb: c_int,
        two_jc: c_int,
        two_jd: c_int,
        two_je: c_int,
        two_jf: c_int,
        two_jg: c_int,
        two_jh: c_int,
        two_ji: c_int,
    ) Error!f64 {
        return call(c.gsl_sf_coupling_9j_e, .{ two_ja, two_jb, two_jc, two_jd, two_je, two_jf, two_jg, two_jh, two_ji });
    }
};

// ===== Hypergeometric functions (gsl_sf_hyperg) ==============================

/// Confluent and Gauss hypergeometric functions. Only the scalar real-argument
/// entry points are wrapped.
pub const hyperg = struct {
    /// `_0F_1(c; x)`.
    pub fn f0F1(cc: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_0F1_e, .{ cc, x });
    }
    /// Confluent `_1F_1(m; n; x)` for integer parameters.
    pub fn f1F1Int(m: c_int, n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_1F1_int_e, .{ m, n, x });
    }
    /// Confluent `_1F_1(a; b; x)`.
    pub fn f1F1(a: f64, b: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_1F1_e, .{ a, b, x });
    }
    /// Confluent `U(m, n, x)` for integer parameters.
    pub fn uInt(m: c_int, n: c_int, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_U_int_e, .{ m, n, x });
    }
    /// Confluent `U(a, b, x)`.
    pub fn u(a: f64, b: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_U_e, .{ a, b, x });
    }
    /// Gauss `_2F_1(a, b; c; x)`.
    pub fn f2F1(a: f64, b: f64, cc: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_2F1_e, .{ a, b, cc, x });
    }
    /// Gauss `_2F_1` with complex conjugate upper parameters `a = aR ± i aI`.
    pub fn f2F1Conj(aR: f64, aI: f64, cc: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_2F1_conj_e, .{ aR, aI, cc, x });
    }
    /// Renormalized Gauss `_2F_1 / Gamma(c)`.
    pub fn f2F1Renorm(a: f64, b: f64, cc: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_2F1_renorm_e, .{ a, b, cc, x });
    }
    /// `_2F_0(a, b; x)`.
    pub fn f2F0(a: f64, b: f64, x: f64) Error!f64 {
        return call(c.gsl_sf_hyperg_2F0_e, .{ a, b, x });
    }
};

// ===== Array-fill helpers ====================================================

/// Driver for the `(nmin, nmax, x, out[])` sequence fillers.
fn fillN(comptime af: anytype, nmin: c_int, nmax: c_int, x: f64, out: []f64) Error!void {
    ensureHandler();
    if (nmax < nmin) return Error.BadLength;
    if (out.len != @as(usize, @intCast(nmax - nmin + 1))) return Error.BadLength;
    try check(af(nmin, nmax, x, out.ptr));
}

/// Driver for the `(lmax, x, out[])` sequence fillers (orders `0..=lmax`).
fn fillL(comptime af: anytype, lmax: c_int, x: f64, out: []f64) Error!void {
    ensureHandler();
    if (lmax < 0) return Error.BadLength;
    if (out.len != @as(usize, @intCast(lmax + 1))) return Error.BadLength;
    try check(af(lmax, x, out.ptr));
}

// ===== Tests =================================================================

const eps = 1e-10;

test "sf: gamma and factorials hit known integer values" {
    try testing.expectApproxEqAbs(@as(f64, 24.0), try gamma.gamma(5.0), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 24.0), try gamma.fact(4), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 720.0), try gamma.gamma(7.0), 1e-6);
    // ln(Gamma(1)) = ln(Gamma(2)) = 0.
    try testing.expectApproxEqAbs(@as(f64, 0.0), try gamma.lnGamma(1.0), eps);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try gamma.lnGamma(2.0), eps);
    // C(6, 2) = 15.
    try testing.expectApproxEqAbs(@as(f64, 15.0), try gamma.choose(6, 2), 1e-9);
    // Beta(2, 3) = 1/12.
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 12.0), try gamma.beta(2.0, 3.0), eps);
}

test "sf: error function reference values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try erf.erf(0.0), eps);
    // erf + erfc == 1.
    try testing.expectApproxEqAbs(@as(f64, 1.0), (try erf.erf(0.7)) + (try erf.erfc(0.7)), eps);
    // erf(1) is a well-known constant.
    try testing.expectApproxEqAbs(@as(f64, 0.8427007929497149), try erf.erf(1.0), eps);
}

test "sf: zeta, eta and dilog match closed forms" {
    const pi = std.math.pi;
    // zeta(2) = pi^2 / 6.
    try testing.expectApproxEqAbs(pi * pi / 6.0, try zeta.zeta(2.0), eps);
    try testing.expectApproxEqAbs(pi * pi / 6.0, try zeta.zetaInt(2), eps);
    // Li_2(1) = pi^2 / 6.
    try testing.expectApproxEqAbs(pi * pi / 6.0, try dilog(1.0), eps);
    // eta(1) = ln 2.
    try testing.expectApproxEqAbs(@as(f64, std.math.ln2), try zeta.eta(1.0), eps);
}

test "sf: exp/log careful variants agree with libm on tame inputs" {
    try testing.expectApproxEqAbs(std.math.e, try exp.exp(1.0), eps);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try log.log(1.0), eps);
    // expm1 and log1plusx are inverses near zero.
    const x = 1e-6;
    try testing.expectApproxEqAbs(x, try log.log1plusx(try exp.expm1(x)), 1e-14);
}

test "sf: bessel identities and a tabulated zero" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try bessel.J0(0.0), eps);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try bessel.J1(0.0), eps);
    // Spherical j0(x) = sin(x)/x -> 1 as x -> 0.
    try testing.expectApproxEqAbs(@as(f64, 1.0), try bessel.j0(1e-8), 1e-12);
    // First positive zero of J0 ~ 2.404825557695773.
    try testing.expectApproxEqAbs(@as(f64, 2.404825557695773), try bessel.zeroJ0(1), 1e-9);
    // Jn(2, x) should equal Jnu(2.0, x).
    try testing.expectApproxEqAbs(try bessel.Jn(2, 3.0), try bessel.Jnu(2.0, 3.0), eps);
}

test "sf: bessel array fillers match the scalar functions" {
    var buf: [5]f64 = undefined;
    // J_0..J_4 at x = 3.
    try bessel.JnArray(0, 4, 3.0, &buf);
    inline for (0..5) |n| {
        try testing.expectApproxEqAbs(try bessel.Jn(@intCast(n), 3.0), buf[n], eps);
    }
    // Spherical j_0..j_3 at x = 2.
    var sph: [4]f64 = undefined;
    try bessel.jlArray(3, 2.0, &sph);
    inline for (0..4) |l| {
        try testing.expectApproxEqAbs(try bessel.jl(@intCast(l), 2.0), sph[l], eps);
    }
    // Wrong-length buffers are rejected before calling GSL.
    var short: [2]f64 = undefined;
    try testing.expectError(Error.BadLength, bessel.JnArray(0, 4, 3.0, &short));
}

test "sf: airy and elliptic integrals with an explicit mode" {
    const pi = std.math.pi;
    // Complete K(0) = E(0) = pi/2.
    try testing.expectApproxEqAbs(pi / 2.0, try ellint.kComp(0.0, .double), eps);
    try testing.expectApproxEqAbs(pi / 2.0, try ellint.eComp(0.0, .double), eps);
    // Airy Ai(0) = 3^(-2/3)/Gamma(2/3).
    const ai0 = 1.0 / (std.math.pow(f64, 3.0, 2.0 / 3.0) * try gamma.gamma(2.0 / 3.0));
    try testing.expectApproxEqAbs(ai0, try airy.Ai(0.0, .double), eps);
}

test "sf: jacobian elliptic functions reduce to trig at m=0" {
    const r = try elljac(0.5, 0.0);
    // At m = 0: sn = sin(u), cn = cos(u), dn = 1.
    try testing.expectApproxEqAbs(@sin(@as(f64, 0.5)), r.sn, eps);
    try testing.expectApproxEqAbs(@cos(@as(f64, 0.5)), r.cn, eps);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.dn, eps);
}

test "sf: orthogonal polynomials at low order" {
    // Legendre P_2(x) = (3x^2 - 1)/2.
    try testing.expectApproxEqAbs(@as(f64, (3.0 * 0.25 - 1.0) / 2.0), try legendre.pl(2, 0.5), eps);
    // Physicists' H_3(x) = 8x^3 - 12x; at x=1 that is -4.
    try testing.expectApproxEqAbs(@as(f64, -4.0), try hermite.physicist(3, 1.0), eps);
    // Gegenbauer C_1^lambda(x) = 2 lambda x.
    try testing.expectApproxEqAbs(@as(f64, 2.0 * 1.5 * 0.3), try gegenbauer.c1(1.5, 0.3), eps);
    // Laguerre L_1^a(x) = 1 + a - x.
    try testing.expectApproxEqAbs(@as(f64, 1.0 + 2.0 - 0.5), try laguerre.l1(2.0, 0.5), eps);
}

test "sf: coupling 3j symbol reference value" {
    // (1 1 0 / 0 0 0) = -1/sqrt(3).
    const v = try coupling.threeJ(2, 2, 0, 0, 0, 0);
    try testing.expectApproxEqAbs(@as(f64, -1.0 / @sqrt(3.0)), v, eps);
}

test "sf: evaluate returns value plus a small error estimate" {
    const r = try evaluate(c.gsl_sf_gamma_e, .{5.0});
    try testing.expectApproxEqAbs(@as(f64, 24.0), r.val, 1e-9);
    try testing.expect(r.err >= 0.0);
    try testing.expect(r.err < 1e-6);
    // The generic path must agree with the primary wrapper.
    try testing.expectEqual(try gamma.gamma(5.0), r.val);

    // A two-argument error-form symbol also works through evaluate.
    const rb = try evaluate(c.gsl_sf_bessel_Jnu_e, .{ 2.0, 3.0 });
    try testing.expectApproxEqAbs(try bessel.Jn(2, 3.0), rb.val, eps);
}

test "sf: domain errors surface as Zig errors instead of aborting" {
    // No manual handler setup: the module auto-installs the non-aborting handler
    // on first use, so this returns an error rather than calling abort().
    try testing.expectError(Error.Domain, log.log(-1.0));
    try testing.expectError(Error.Domain, evaluate(c.gsl_sf_log_e, .{-1.0}));
}

test "sf: every wrapped function is invoked (symbol + arity coverage)" {
    // Exhaustively call every binding with a benign valid input, discarding both
    // value and any domain error. Purpose: force every extern `gsl_sf_*` symbol
    // to link and its argument order/types to compile. Correctness is checked by
    // the closed-form tests above.
    const run = struct {
        fn call(eu: anytype) void {
            if (eu) |_| {} else |_| {}
        }
    }.call;

    run(multiply(2.0, 3.0));
    run(powInt(2.0, 3));
    run(dawson(1.0));
    run(clausen(1.0));
    run(dilog(0.5));

    inline for (.{ trig.sin, trig.cos, trig.sinc, trig.sinPi, trig.cosPi, trig.lnSinh, trig.lnCosh, trig.angleRestrictSymm, trig.angleRestrictPos }) |f| run(f(1.0));
    run(trig.hypot(3.0, 4.0));

    run(exp.exp(1.0));
    run(exp.expMult(1.0, 2.0));
    inline for (.{ exp.expm1, exp.exprel, exp.exprel2 }) |f| run(f(0.5));
    run(exp.exprelN(3, 0.5));
    inline for (.{ log.log, log.logAbs, log.log1plusx, log.log1plusxMx }) |f| run(f(0.5));

    inline for (.{ erf.erf, erf.erfc, erf.logErfc, erf.z, erf.q, erf.hazard }) |f| run(f(0.7));

    inline for (.{ gamma.gamma, gamma.lnGamma, gamma.gammaStar, gamma.gammaInv }) |f| run(f(1.5));
    run(gamma.taylorCoeff(3, 1.0));
    inline for (.{ gamma.fact, gamma.doubleFact, gamma.lnFact, gamma.lnDoubleFact }) |f| run(f(5));
    run(gamma.choose(6, 2));
    run(gamma.lnChoose(6, 2));
    inline for (.{ gamma.poch, gamma.lnPoch, gamma.pochRel, gamma.incQ, gamma.incP, gamma.inc, gamma.beta, gamma.lnBeta }) |f| run(f(2.0, 1.0));
    run(gamma.incBeta(2.0, 3.0, 0.5));

    run(psi.psi(1.5));
    run(psi.psiInt(3));
    run(psi.psi1piy(1.0));
    run(psi.psi1(1.5));
    run(psi.psi1Int(3));
    run(psi.psiN(2, 1.5));

    inline for (.{ zeta.zeta, zeta.zetaM1, zeta.eta }) |f| run(f(2.0));
    inline for (.{ zeta.zetaInt, zeta.zetaM1Int, zeta.etaInt }) |f| run(f(2));
    run(zeta.hurwitz(2.0, 1.0));

    inline for (.{ debye.d1, debye.d2, debye.d3, debye.d4, debye.d5, debye.d6 }) |f| run(f(1.0));
    run(lambert.w0(1.0));
    run(lambert.wm1(-0.2));
    inline for (.{ synchrotron.s1, synchrotron.s2 }) |f| run(f(1.0));
    inline for (.{ transport.j2, transport.j3, transport.j4, transport.j5 }) |f| run(f(1.0));

    inline for (.{ expint.e1, expint.e2, expint.ei, expint.shi, expint.chi, expint.exp3, expint.si, expint.ci, expint.atanInt }) |f| run(f(1.0));
    run(expint.en(3, 1.0));

    inline for (.{ fermiDirac.m1, fermiDirac.f0, fermiDirac.f1, fermiDirac.f2, fermiDirac.mhalf, fermiDirac.half, fermiDirac.threeHalf }) |f| run(f(0.5));
    run(fermiDirac.int(3, 0.5));
    run(fermiDirac.inc0(1.0, 0.5));

    inline for (.{ bessel.J0, bessel.J1, bessel.Y0, bessel.Y1, bessel.I0, bessel.I1, bessel.I0Scaled, bessel.I1Scaled, bessel.K0, bessel.K1, bessel.K0Scaled, bessel.K1Scaled, bessel.j0, bessel.j1, bessel.j2, bessel.y0, bessel.y1, bessel.y2, bessel.i0Scaled, bessel.i1Scaled, bessel.i2Scaled, bessel.k0Scaled, bessel.k1Scaled, bessel.k2Scaled }) |f| run(f(1.5));
    inline for (.{ bessel.Jn, bessel.Yn, bessel.In, bessel.InScaled, bessel.Kn, bessel.KnScaled, bessel.jl, bessel.yl, bessel.ilScaled, bessel.klScaled }) |f| run(f(2, 1.5));
    inline for (.{ bessel.Jnu, bessel.Ynu, bessel.Inu, bessel.InuScaled, bessel.Knu, bessel.KnuScaled, bessel.lnKnu }) |f| run(f(2.5, 1.5));
    inline for (.{ bessel.zeroJ0, bessel.zeroJ1 }) |f| run(f(1));
    run(bessel.zeroJnu(2.5, 1));

    var abuf: [4]f64 = undefined;
    inline for (.{ bessel.JnArray, bessel.YnArray, bessel.InArray, bessel.InScaledArray, bessel.KnArray, bessel.KnScaledArray }) |f| run(f(0, 3, 1.5, &abuf));
    inline for (.{ bessel.jlArray, bessel.ylArray, bessel.ilScaledArray, bessel.klScaledArray, legendre.plArray }) |f| run(f(3, 1.5, &abuf));
    run(gegenbauer.array(3, 1.5, 0.3, &abuf));

    inline for (.{ airy.Ai, airy.Bi, airy.AiScaled, airy.BiScaled, airy.AiDeriv, airy.BiDeriv, airy.AiDerivScaled, airy.BiDerivScaled }) |f| run(f(0.5, .double));
    inline for (.{ airy.zeroAi, airy.zeroBi, airy.zeroAiDeriv, airy.zeroBiDeriv }) |f| run(f(1));

    inline for (.{ ellint.kComp, ellint.eComp, ellint.dComp }) |f| run(f(0.5, .double));
    run(ellint.pComp(0.5, 0.3, .double));
    inline for (.{ ellint.f, ellint.e, ellint.d, ellint.rc }) |f| run(f(0.5, 0.5, .double));
    run(ellint.p(0.5, 0.5, 0.3, .double));
    run(ellint.rd(1.0, 2.0, 3.0, .double));
    run(ellint.rf(1.0, 2.0, 3.0, .double));
    run(ellint.rj(1.0, 2.0, 3.0, 4.0, .double));
    run(elljac(0.5, 0.5));

    run(legendre.pl(3, 0.5));
    inline for (.{ legendre.q0, legendre.q1 }) |f| run(f(2.0));
    run(legendre.ql(2, 2.0));
    inline for (.{ legendre.plm, legendre.sphPlm }) |f| run(f(3, 1, 0.5));
    inline for (.{ legendre.conicalHalf, legendre.conicalMhalf, legendre.conical0, legendre.conical1 }) |f| run(f(1.0, 2.0));
    inline for (.{ legendre.conicalSphReg, legendre.conicalCylReg }) |f| run(f(2, 1.0, 2.0));
    inline for (.{ legendre.h3d0, legendre.h3d1 }) |f| run(f(1.0, 1.0));
    run(legendre.h3d(2, 1.0, 1.0));

    inline for (.{ laguerre.l1, laguerre.l2, laguerre.l3 }) |f| run(f(1.0, 0.5));
    run(laguerre.ln(4, 1.0, 0.5));
    inline for (.{ gegenbauer.c1, gegenbauer.c2, gegenbauer.c3 }) |f| run(f(1.5, 0.3));
    run(gegenbauer.cn(4, 1.5, 0.3));
    inline for (.{ hermite.physicist, hermite.probabilist, hermite.func }) |f| run(f(3, 0.5));
    inline for (.{ hermite.physicistDeriv, hermite.probabilistDeriv }) |f| run(f(1, 3, 0.5));
    inline for (.{ hermite.physicistZero, hermite.probabilistZero, hermite.funcZero }) |f| run(f(4, 1));

    run(coulomb.hydrogenicR1(1.0, 1.0));
    run(coulomb.hydrogenicR(3, 1, 1.0, 1.0));
    run(coupling.threeJ(2, 2, 0, 0, 0, 0));
    run(coupling.sixJ(2, 2, 2, 2, 2, 2));
    run(coupling.nineJ(0, 0, 0, 0, 0, 0, 0, 0, 0));

    run(hyperg.f0F1(2.0, 0.5));
    run(hyperg.f1F1Int(1, 2, 0.5));
    run(hyperg.f1F1(1.0, 2.0, 0.5));
    run(hyperg.uInt(1, 2, 0.5));
    run(hyperg.u(1.0, 2.0, 0.5));
    inline for (.{ hyperg.f2F1, hyperg.f2F1Conj, hyperg.f2F1Renorm }) |f| run(f(1.0, 1.0, 2.0, 0.5));
    run(hyperg.f2F0(0.5, 0.5, -0.1));
}
