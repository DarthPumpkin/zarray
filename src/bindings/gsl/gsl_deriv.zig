//! Idiomatic Zig bindings for the GNU Scientific Library's numerical
//! differentiation module (`gsl_deriv`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the numerical
//! differentiation chapter — the smallest of the callback-based chapters, and
//! the first consumer of the shared callback bridge (`gsl.callback`). It reuses
//! `gsl.zig`'s process-global error-handler switch but keeps the `gsl_deriv` C
//! API behind its own `c`. It is reached as `gsl.deriv`.
//!
//! ## Shape of the surface
//!
//! Each routine estimates the derivative of a caller-supplied function at a
//! point `x` using a step size `h`, returning both the value and an absolute
//! error estimate (`Result`). The function is passed as a `Callback` value,
//! constructed with either factory method (see `gsl.callback`):
//!
//!   - `.initFn(f)`   — a plain `*const fn(f64) f64` (a bare function coerces in).
//!   - `.initCtx(ctx)` — a pointer to a struct with `pub fn eval(self, x) f64`,
//!     for callbacks that capture parameters (no allocation).
//!
//! The three routines differ only in stencil:
//!
//!   - `central`  — central difference (most accurate; needs the function on
//!                  both sides of `x`).
//!   - `forward`  — forward difference (uses points at and to the right of `x`).
//!   - `backward` — backward difference (points at and to the left of `x`).
//!
//! ## Omissions
//!
//!   - None. `gsl_deriv` is exactly these three routines; all are wrapped.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_math.h");
    @cInclude("gsl/gsl_deriv.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the differentiation routines. The raw `c_int` status is
/// always available from the underlying `c.gsl_deriv_*` symbol if you need the
/// exact code.
pub const Error = error{
    /// `GSL_EDOM` — a domain error inside the evaluation.
    Domain,
    /// `GSL_EINVAL` — an invalid argument.
    Invalid,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EDOM => Error.Domain,
        c.GSL_EINVAL => Error.Invalid,
        else => Error.Unspecified,
    };
}

/// A derivative estimate and its absolute error bound.
pub const Result = struct {
    /// The estimated derivative `f'(x)`.
    value: f64,
    /// GSL's estimate of the absolute error in `value`.
    abserr: f64,
};

/// A `gsl_function`-shaped callback value for this chapter. Construct with
/// `Callback.initFn(f)` (a plain function) or `Callback.initCtx(&ctx)` (a
/// context struct); at a call site the leading dot suffices, e.g.
/// `central(.initFn(myFn), x, h)`.
pub const Callback = callback.Function(c.gsl_function);

/// Central-difference estimate of `f'(x)` with step `h` (the most accurate of
/// the three; evaluates the function on both sides of `x`).
///
/// Example:
/// ```
/// fn f(x: f64) f64 { return x * x; }
/// const d = try gsl.deriv.central(.initFn(f), 2.0, 1e-8); // d.value ≈ 4
/// ```
pub fn central(cb: Callback, x: f64, h: f64) Error!Result {
    return finish(c.gsl_deriv_central, cb, x, h);
}

/// Forward-difference estimate of `f'(x)` with step `h` (uses points at and to
/// the right of `x`; for one-sided domains).
pub fn forward(cb: Callback, x: f64, h: f64) Error!Result {
    return finish(c.gsl_deriv_forward, cb, x, h);
}

/// Backward-difference estimate of `f'(x)` with step `h` (uses points at and to
/// the left of `x`).
pub fn backward(cb: Callback, x: f64, h: f64) Error!Result {
    return finish(c.gsl_deriv_backward, cb, x, h);
}

const Routine = fn ([*c]const c.gsl_function, f64, f64, [*c]f64, [*c]f64) callconv(.c) c_int;

fn finish(comptime routine: Routine, cb: Callback, x: f64, h: f64) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    try check(routine(&cb.gf, x, h, &result, &abserr));
    return .{ .value = result, .abserr = abserr };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn square(x: f64) f64 {
    return x * x;
}

test "deriv: central difference of x^2 is 2x" {
    // d/dx x^2 = 2x; at x = 3 that is 6.
    const d = try central(.initFn(square), 3.0, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 6.0), d.value, 1e-6);
    try testing.expect(d.abserr >= 0.0 and d.abserr < 1e-3);
}

test "deriv: forward and backward agree with central on a smooth function" {
    const s = struct {
        fn f(x: f64) f64 {
            return @sin(x);
        }
    }.f;
    // d/dx sin(x) = cos(x); at x = 1.
    const want = @cos(1.0);
    const cen = try central(.initFn(s), 1.0, 1e-6);
    const fwd = try forward(.initFn(s), 1.0, 1e-6);
    const bwd = try backward(.initFn(s), 1.0, 1e-6);
    try testing.expectApproxEqAbs(want, cen.value, 1e-6);
    try testing.expectApproxEqAbs(want, fwd.value, 1e-4);
    try testing.expectApproxEqAbs(want, bwd.value, 1e-4);
}

test "deriv: accepts a context struct capturing a parameter" {
    // f(x) = sin(k*x); f'(x) = k*cos(k*x). Capture k in a context.
    const Wave = struct {
        k: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            return @sin(self.k * x);
        }
    };
    var w = Wave{ .k = 3.0 };
    const d = try central(.initCtx(&w), 0.5, 1e-6);
    const want = w.k * @cos(w.k * 0.5);
    try testing.expectApproxEqAbs(want, d.value, 1e-6);
}

test "deriv: a Callback value can be built once and reused" {
    const cb: Callback = .initFn(square);
    const d1 = try central(cb, 3.0, 1e-6);
    const d2 = try forward(cb, 3.0, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 6.0), d1.value, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 6.0), d2.value, 1e-4);
}
