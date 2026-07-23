//! Idiomatic Zig bindings for the GNU Scientific Library's Chebyshev
//! approximation module (`gsl_chebyshev`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the Chebyshev-series
//! chapter. It reuses that module's process-global error-handler switch but
//! keeps the `gsl_chebyshev` C API behind its own `c`. It is reached as
//! `gsl.cheb`.
//!
//! ## Shape of the surface
//!
//! A `Chebyshev` holds a truncated Chebyshev expansion of a function over an
//! interval `[a, b]`. You allocate one of a chosen `order`, `fit` it to a
//! function once (which samples the function and computes the coefficients),
//! and thereafter evaluate the smooth approximation cheaply — no further
//! callbacks:
//!
//! ```zig
//! var cs = try gsl.cheb.Chebyshev.init(40);
//! defer cs.deinit();
//! try cs.fit(.initFn(f), 0, 1);          // sample f over [0, 1]
//! const y = cs.eval(0.5);                // approximate f(0.5)
//! const r = cs.evalErr(0.5);             // value + error estimate
//! ```
//!
//! Because `fit` samples the function synchronously and stores only
//! coefficients, the `Callback` is transient — it need not outlive the `fit`
//! call (unlike the stateful `roots`/`min` solvers).
//!
//! From a fitted series you can derive new series for its derivative and its
//! definite integral (fixed to zero at `a`); each is a fresh `Chebyshev` the
//! caller owns:
//!
//! ```zig
//! var d = try cs.deriv();   // series for f'
//! defer d.deinit();
//! var i = try cs.integ();   // series for ∫ₐˣ f
//! defer i.deinit();
//! ```
//!
//! ## Omissions
//!
//!   - The `gsl_mode_t` "eval_mode" variants (`gsl_cheb_eval_mode*`) are not
//!     wrapped; they exist for single-precision tuning and are documented by
//!     GSL as not meant for casual use. The full-precision `eval`/`evalN`
//!     forms cover the common cases.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_math.h");
    @cInclude("gsl/gsl_chebyshev.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the Chebyshev routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_cheb_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument.
    Invalid,
    /// `GSL_ENOMEM` — allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// An approximation value and its estimated absolute error, returned by the
/// `*Err` evaluators.
pub const Result = struct {
    /// The evaluated approximation.
    value: f64,
    /// GSL's estimate of the absolute error in `value`.
    abserr: f64,
};

/// A `gsl_function`-shaped callback value for `fit`. Construct with
/// `Callback.initFn(f)` (a plain function) or `Callback.initCtx(&ctx)`.
pub const Callback = callback.Function(c.gsl_function);

/// A Chebyshev series approximating a function over an interval `[a, b]`. Owns
/// its GSL allocation; call `deinit` to free.
pub const Chebyshev = struct {
    ptr: *c.gsl_cheb_series,

    /// Allocate a series capable of holding an expansion up to `max_order` (so
    /// `max_order + 1` coefficients). A higher order approximates a smooth
    /// function more accurately at the cost of more work in `fit`.
    pub fn init(max_order: usize) Error!Chebyshev {
        gsl.ensureHandler();
        const p = c.gsl_cheb_alloc(max_order) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: *Chebyshev) void {
        c.gsl_cheb_free(self.ptr);
    }

    /// Sample `cb` over `[a, b]` and compute the Chebyshev coefficients. May be
    /// called again to re-fit the same allocation to a new function/interval.
    /// The callback is used only for the duration of this call.
    pub fn fit(self: *Chebyshev, cb: Callback, a: f64, b: f64) Error!void {
        // gsl_cheb_init takes a const gsl_function*; a local copy keeps the
        // transient callback struct alive for the duration of the call.
        var f = cb;
        try check(c.gsl_cheb_init(self.ptr, &f.gf, a, b));
    }

    /// Evaluate the fitted approximation at `x` (expected within `[a, b]`).
    pub fn eval(self: *const Chebyshev, x: f64) f64 {
        return c.gsl_cheb_eval(self.ptr, x);
    }

    /// Evaluate at `x`, also returning GSL's absolute-error estimate.
    pub fn evalErr(self: *const Chebyshev, x: f64) Result {
        var result: f64 = undefined;
        var abserr: f64 = undefined;
        // gsl_cheb_eval_err cannot fail for a fitted series (per GSL); the
        // returned status is always success, so it is not surfaced.
        _ = c.gsl_cheb_eval_err(self.ptr, x, &result, &abserr);
        return .{ .value = result, .abserr = abserr };
    }

    /// Evaluate at `x` using only the terms up to `n` (a coarser, cheaper
    /// approximation). `n` is clamped to the series' order.
    pub fn evalN(self: *const Chebyshev, n: usize, x: f64) f64 {
        return c.gsl_cheb_eval_n(self.ptr, n, x);
    }

    /// Evaluate to order `n` at `x`, also returning the absolute-error estimate.
    pub fn evalNErr(self: *const Chebyshev, n: usize, x: f64) Result {
        var result: f64 = undefined;
        var abserr: f64 = undefined;
        _ = c.gsl_cheb_eval_n_err(self.ptr, n, x, &result, &abserr);
        return .{ .value = result, .abserr = abserr };
    }

    /// The order of the expansion (the series holds `order + 1` coefficients).
    pub fn order(self: *const Chebyshev) usize {
        return c.gsl_cheb_order(self.ptr);
    }

    /// A borrowed view of the `order + 1` Chebyshev coefficients. Valid until
    /// the series is re-fitted or freed.
    pub fn coeffs(self: *const Chebyshev) []const f64 {
        const n = c.gsl_cheb_size(self.ptr);
        return c.gsl_cheb_coeffs(self.ptr)[0..n];
    }

    /// Build a new series (same order and interval) approximating the
    /// derivative `f'` of this one. The returned `Chebyshev` is owned by the
    /// caller; `deinit` it.
    pub fn deriv(self: *const Chebyshev) Error!Chebyshev {
        var out = try Chebyshev.init(self.order());
        errdefer out.deinit();
        try check(c.gsl_cheb_calc_deriv(out.ptr, self.ptr));
        return out;
    }

    /// Build a new series (same order and interval) approximating the definite
    /// integral `∫ₐˣ f` of this one (zero at the left endpoint `a`). The
    /// returned `Chebyshev` is owned by the caller; `deinit` it.
    pub fn integ(self: *const Chebyshev) Error!Chebyshev {
        var out = try Chebyshev.init(self.order());
        errdefer out.deinit();
        try check(c.gsl_cheb_calc_integ(out.ptr, self.ptr));
        return out;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn expFn(x: f64) f64 {
    return @exp(x);
}

test "cheb: a fitted series approximates exp on [0, 1]" {
    var cs = try Chebyshev.init(40);
    defer cs.deinit();
    try cs.fit(.initFn(expFn), 0.0, 1.0);

    try testing.expectEqual(@as(usize, 40), cs.order());
    try testing.expectEqual(@as(usize, 41), cs.coeffs().len);

    // The high-order series matches exp to near machine precision.
    var x: f64 = 0.0;
    while (x <= 1.0) : (x += 0.05) {
        try testing.expectApproxEqAbs(@exp(x), cs.eval(x), 1e-12);
    }
}

test "cheb: evalErr reports a value with a small error bound" {
    var cs = try Chebyshev.init(40);
    defer cs.deinit();
    try cs.fit(.initFn(expFn), 0.0, 1.0);

    const r = cs.evalErr(0.5);
    try testing.expectApproxEqAbs(@exp(0.5), r.value, 1e-12);
    try testing.expect(r.abserr >= 0.0 and r.abserr < 1e-6);
}

test "cheb: a lower-order evaluation is coarser than the full series" {
    var cs = try Chebyshev.init(40);
    defer cs.deinit();
    try cs.fit(.initFn(expFn), 0.0, 1.0);

    const want = @exp(0.5);
    const full_err = @abs(cs.eval(0.5) - want);
    const coarse_err = @abs(cs.evalN(4, 0.5) - want);
    // Truncating to 4 terms is still decent but strictly worse than the full 40.
    try testing.expect(coarse_err > full_err);
    try testing.expect(coarse_err < 1e-3);

    // evalNErr agrees with evalN on the value.
    const rn = cs.evalNErr(4, 0.5);
    try testing.expectEqual(cs.evalN(4, 0.5), rn.value);
}

test "cheb: the derivative series approximates f'" {
    // d/dx exp = exp, so the derivative series should again match exp.
    var cs = try Chebyshev.init(40);
    defer cs.deinit();
    try cs.fit(.initFn(expFn), 0.0, 1.0);

    var d = try cs.deriv();
    defer d.deinit();
    try testing.expectEqual(cs.order(), d.order());

    var x: f64 = 0.1;
    while (x <= 0.9) : (x += 0.1) {
        try testing.expectApproxEqAbs(@exp(x), d.eval(x), 1e-10);
    }
}

test "cheb: the integral series approximates ∫ₐˣ f" {
    // ∫₀ˣ exp = exp(x) − 1 (fixed to zero at the left endpoint a = 0).
    var cs = try Chebyshev.init(40);
    defer cs.deinit();
    try cs.fit(.initFn(expFn), 0.0, 1.0);

    var i = try cs.integ();
    defer i.deinit();

    try testing.expectApproxEqAbs(@as(f64, 0.0), i.eval(0.0), 1e-12);
    var x: f64 = 0.1;
    while (x <= 1.0) : (x += 0.1) {
        try testing.expectApproxEqAbs(@exp(x) - 1.0, i.eval(x), 1e-10);
    }
}

test "cheb: a context struct captures a parameter" {
    // f(x) = sin(k·x); check the fit reproduces it.
    const Wave = struct {
        k: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            return @sin(self.k * x);
        }
    };
    var w = Wave{ .k = 3.0 };
    var cs = try Chebyshev.init(50);
    defer cs.deinit();
    try cs.fit(.initCtx(&w), 0.0, std.math.pi);

    var x: f64 = 0.0;
    while (x <= std.math.pi) : (x += 0.2) {
        try testing.expectApproxEqAbs(@sin(w.k * x), cs.eval(x), 1e-9);
    }
}

test "cheb: a series can be re-fitted to a new function" {
    var cs = try Chebyshev.init(30);
    defer cs.deinit();

    try cs.fit(.initFn(expFn), 0.0, 1.0);
    try testing.expectApproxEqAbs(@exp(0.5), cs.eval(0.5), 1e-10);

    // Reuse the same allocation for a different function/interval.
    const cosFn = struct {
        fn f(x: f64) f64 {
            return @cos(x);
        }
    }.f;
    try cs.fit(.initFn(cosFn), -1.0, 1.0);
    try testing.expectApproxEqAbs(@cos(0.3), cs.eval(0.3), 1e-10);
}
