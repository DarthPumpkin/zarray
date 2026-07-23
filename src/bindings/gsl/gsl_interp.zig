//! Idiomatic Zig bindings for the GNU Scientific Library's 1-D interpolation
//! modules (`gsl_interp`, `gsl_spline`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the interpolation
//! chapter. It reuses that module's process-global error-handler switch (so a
//! GSL domain error surfaces as a Zig `Error` instead of aborting), but keeps
//! the interpolation-specific C API behind its own `c` (the `gsl_interp`/
//! `gsl_spline` headers are not pulled in by `gsl.zig`). It is reached as
//! `gsl.interp`.
//!
//! ## Shape of the surface
//!
//! GSL splits interpolation into two layers: the low-level `gsl_interp` (which
//! re-takes the `x`/`y` arrays on every call) and the higher-level `gsl_spline`
//! (which stores its own copy of the data, so evaluation needs only `x`). These
//! bindings wrap **`gsl_spline`** — strictly the more ergonomic of the two.
//!
//!   - `Spline` owns a `gsl_spline` (allocation + a private copy of the data).
//!     Build it with `Spline.init(type, xs, ys)`; the `xs` must be strictly
//!     increasing and at least `type.minSize()` long. It evaluates the value
//!     (`eval`), first/second derivative (`evalDeriv`/`evalDeriv2`), and the
//!     definite integral over `[a, b]` (`evalInteg`).
//!   - `Accel` is an optional O(1) lookup accelerator (`gsl_interp_accel`).
//!     Because a `Spline` is immutable during evaluation and only the `Accel`
//!     mutates, one `Spline` can be shared read-only across threads as long as
//!     each thread brings its **own** `Accel`. The eval methods therefore take
//!     `accel: ?Accel` explicitly — pass a per-thread `Accel` for speed, or
//!     `null` for a one-off lookup (GSL falls back to a binary search).
//!   - `Type` selects the interpolation method (mirrors `gsl_interp_*` types).
//!
//! For the bare `f64` (non-erroring) forms, or the low-level `gsl_interp` API,
//! reach the raw `c.gsl_spline_*` / `c.gsl_interp_*` symbols directly.
//!
//! ## Omissions
//!
//!   - The low-level `gsl_interp` layer (which re-takes the `x`/`y` arrays on
//!     every call) is not wrapped — only the higher-level `gsl_spline`, which
//!     stores its own copy of the data. Reach `c.gsl_interp_*` directly if you
//!     need it.
//!   - The bare, non-`_e` evaluation forms (`gsl_spline_eval` returning a plain
//!     `f64`) are not re-exposed; the wrappers use the `_e` forms so a
//!     domain error surfaces as a Zig `Error`. Use `c.gsl_spline_eval` for the
//!     non-erroring form.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for the bare `f64` `gsl_spline_eval` forms or
/// the low-level `gsl_interp` layer not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_interp.h");
    @cInclude("gsl/gsl_spline.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the interpolation routines. The raw `c_int` status is
/// always available from the `c.gsl_spline_*_e` symbols if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EDOM` — the evaluation point is outside the interpolation range
    /// `[x[0], x[n-1]]`, or an integration endpoint is.
    Domain,
    /// `GSL_EINVAL` — invalid input, e.g. `x` values that are not strictly
    /// increasing.
    Invalid,
    /// `GSL_EBADLEN`, or `xs.len != ys.len`, or fewer points than the chosen
    /// interpolation type requires (`Type.minSize`).
    BadLength,
    /// `GSL_ENOMEM` — allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EDOM => Error.Domain,
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADLEN => Error.BadLength,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// The interpolation method, mirroring GSL's `gsl_interp_*` types.
pub const Type = enum {
    /// Piecewise linear. Fast, no smoothness; `minSize` 2.
    linear,
    /// Polynomial through all points (Neville's algorithm). Fine for a handful
    /// of points; oscillates for many.
    polynomial,
    /// Natural cubic spline (C² continuous). The usual default; `minSize` 3.
    cspline,
    /// Periodic cubic spline (matches value/derivatives at the ends).
    cspline_periodic,
    /// Akima spline — cubic, less prone to overshoot near outliers; `minSize` 5.
    akima,
    /// Periodic Akima spline.
    akima_periodic,
    /// Steffen's monotonic cubic spline — preserves monotonicity of the data
    /// (no spurious local extrema between points); `minSize` 3.
    steffen,

    fn typePtr(self: Type) [*c]const c.gsl_interp_type {
        return switch (self) {
            .linear => c.gsl_interp_linear,
            .polynomial => c.gsl_interp_polynomial,
            .cspline => c.gsl_interp_cspline,
            .cspline_periodic => c.gsl_interp_cspline_periodic,
            .akima => c.gsl_interp_akima,
            .akima_periodic => c.gsl_interp_akima_periodic,
            .steffen => c.gsl_interp_steffen,
        };
    }

    /// Minimum number of data points this interpolation type requires.
    pub fn minSize(self: Type) u32 {
        return c.gsl_interp_type_min_size(self.typePtr());
    }
};

/// An O(1) lookup accelerator (`gsl_interp_accel`) that caches the last bracket
/// found, speeding up evaluations that move smoothly through the data. It is
/// *mutable* and **not** thread-safe: give each thread its own `Accel`. Owns its
/// GSL allocation; call `deinit` to free it.
pub const Accel = struct {
    ptr: *c.gsl_interp_accel,

    /// Allocate an accelerator. Fails only if the underlying allocation fails.
    pub fn init() error{OutOfMemory}!Accel {
        const p = c.gsl_interp_accel_alloc() orelse return error.OutOfMemory;
        return .{ .ptr = p };
    }
    pub fn deinit(self: Accel) void {
        c.gsl_interp_accel_free(self.ptr);
    }
    /// Clear the cached bracket (e.g. before evaluating a new, unrelated
    /// sequence of points with the same accelerator).
    pub fn reset(self: Accel) void {
        _ = c.gsl_interp_accel_reset(self.ptr);
    }
};

/// A 1-D interpolant over a fixed set of `(x, y)` points (`gsl_spline`). Owns
/// its GSL allocation and a private copy of the data (so the caller's `xs`/`ys`
/// may be freed after `init`); call `deinit` to free it.
///
/// Example:
/// ```
/// const xs = [_]f64{ 0, 1, 2, 3 };
/// const ys = [_]f64{ 0, 1, 4, 9 };
/// var s = try gsl.interp.Spline.init(.cspline, &xs, &ys);
/// defer s.deinit();
/// var acc = try gsl.interp.Accel.init();
/// defer acc.deinit();
/// const y = try s.eval(1.5, acc);       // interpolated value
/// const dy = try s.evalDeriv(1.5, acc); // first derivative
/// ```
pub const Spline = struct {
    ptr: *c.gsl_spline,

    /// Build an interpolant of the given `Type` over `(xs[i], ys[i])`. Requires
    /// `xs.len == ys.len`, `xs` strictly increasing, and at least
    /// `t.minSize()` points. The data is copied into the spline.
    pub fn init(t: Type, xs: []const f64, ys: []const f64) Error!Spline {
        if (xs.len != ys.len) return Error.BadLength;
        // gsl_spline_alloc returns NULL (indistinguishable from OOM) when given
        // fewer points than the type needs, so reject that up front for a clear
        // error.
        if (xs.len < t.minSize()) return Error.BadLength;
        gsl.ensureHandler();
        const p = c.gsl_spline_alloc(t.typePtr(), xs.len) orelse return Error.OutOfMemory;
        errdefer c.gsl_spline_free(p);
        try check(c.gsl_spline_init(p, xs.ptr, ys.ptr, xs.len));
        return .{ .ptr = p };
    }

    pub fn deinit(self: Spline) void {
        c.gsl_spline_free(self.ptr);
    }

    /// Re-fit this spline on new data *of the same length* it was allocated for,
    /// reusing the allocation (no realloc). `xs.len` must match the original
    /// size; otherwise use a fresh `init`.
    pub fn reinit(self: Spline, xs: []const f64, ys: []const f64) Error!void {
        if (xs.len != ys.len) return Error.BadLength;
        try check(c.gsl_spline_init(self.ptr, xs.ptr, ys.ptr, xs.len));
    }

    /// The interpolation type's name (e.g. `"cspline"`).
    pub fn name(self: Spline) [:0]const u8 {
        return std.mem.span(c.gsl_spline_name(self.ptr));
    }

    /// Interpolated value at `x`. `x` must lie in `[x[0], x[n-1]]`.
    pub fn eval(self: Spline, x: f64, accel: ?Accel) Error!f64 {
        var y: f64 = undefined;
        try check(c.gsl_spline_eval_e(self.ptr, x, accelPtr(accel), &y));
        return y;
    }
    /// First derivative of the interpolant at `x`.
    pub fn evalDeriv(self: Spline, x: f64, accel: ?Accel) Error!f64 {
        var d: f64 = undefined;
        try check(c.gsl_spline_eval_deriv_e(self.ptr, x, accelPtr(accel), &d));
        return d;
    }
    /// Second derivative of the interpolant at `x`.
    pub fn evalDeriv2(self: Spline, x: f64, accel: ?Accel) Error!f64 {
        var d2: f64 = undefined;
        try check(c.gsl_spline_eval_deriv2_e(self.ptr, x, accelPtr(accel), &d2));
        return d2;
    }
    /// Definite integral of the interpolant over `[a, b]`.
    pub fn evalInteg(self: Spline, a: f64, b: f64, accel: ?Accel) Error!f64 {
        var r: f64 = undefined;
        try check(c.gsl_spline_eval_integ_e(self.ptr, a, b, accelPtr(accel), &r));
        return r;
    }
};

inline fn accelPtr(accel: ?Accel) ?*c.gsl_interp_accel {
    return if (accel) |a| a.ptr else null;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "interp: linear interpolation reproduces a straight line exactly" {
    const xs = [_]f64{ 0, 1, 2, 3, 4 };
    // y = 3x + 1
    const ys = [_]f64{ 1, 4, 7, 10, 13 };
    var s = try Spline.init(.linear, &xs, &ys);
    defer s.deinit();

    try testing.expectEqualStrings("linear", s.name());

    // At the nodes and between them, a linear interpolant of a line is exact.
    try testing.expectApproxEqAbs(@as(f64, 4.0), try s.eval(1.0, null), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.5), try s.eval(1.5, null), 1e-12);
    // Derivative is the slope; the definite integral matches the analytic area.
    try testing.expectApproxEqAbs(@as(f64, 3.0), try s.evalDeriv(2.5, null), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), try s.evalInteg(0.0, 2.0, null), 1e-12);
}

test "interp: cubic spline approximates a smooth function and its derivative" {
    var xs: [21]f64 = undefined;
    var ys: [21]f64 = undefined;
    for (&xs, &ys, 0..) |*x, *y, i| {
        x.* = @as(f64, @floatFromInt(i)) * (std.math.pi / 20.0); // 0..pi
        y.* = @sin(x.*);
    }
    var s = try Spline.init(.cspline, &xs, &ys);
    defer s.deinit();
    var acc = try Accel.init();
    defer acc.deinit();

    // Between the nodes the cubic spline tracks sin and its derivative (cos).
    const x = 0.6123;
    try testing.expectApproxEqAbs(@sin(x), try s.eval(x, acc), 1e-4);
    try testing.expectApproxEqAbs(@cos(x), try s.evalDeriv(x, acc), 1e-3);
    // Integral of sin over [0, pi] is 2.
    try testing.expectApproxEqAbs(@as(f64, 2.0), try s.evalInteg(0.0, std.math.pi, acc), 1e-3);
}

test "interp: out-of-range evaluation is a domain error, not an abort" {
    const xs = [_]f64{ 0, 1, 2 };
    const ys = [_]f64{ 0, 1, 2 };
    var s = try Spline.init(.linear, &xs, &ys);
    defer s.deinit();
    try testing.expectError(Error.Domain, s.eval(5.0, null));
    try testing.expectError(Error.Domain, s.eval(-1.0, null));
}

test "interp: mismatched or too-short inputs are rejected" {
    const xs = [_]f64{ 0, 1, 2 };
    const ys_short = [_]f64{ 0, 1 };
    try testing.expectError(Error.BadLength, Spline.init(.linear, &xs, &ys_short));

    // cspline needs at least 3 points; too few is rejected as BadLength
    // (gsl_spline_alloc returns NULL below the type's minimum size).
    const two_x = [_]f64{ 0, 1 };
    const two_y = [_]f64{ 0, 1 };
    try testing.expectError(Error.BadLength, Spline.init(.cspline, &two_x, &two_y));
}

test "interp: reinit reuses the allocation for new same-length data" {
    const xs = [_]f64{ 0, 1, 2, 3 };
    const ys = [_]f64{ 0, 0, 0, 0 };
    var s = try Spline.init(.linear, &xs, &ys);
    defer s.deinit();
    try testing.expectApproxEqAbs(@as(f64, 0.0), try s.eval(1.5, null), 1e-12);

    const ys2 = [_]f64{ 0, 2, 4, 6 }; // y = 2x
    try s.reinit(&xs, &ys2);
    try testing.expectApproxEqAbs(@as(f64, 3.0), try s.eval(1.5, null), 1e-12);
}

test "interp: passing an accel agrees with the null (bsearch) path" {
    const xs = [_]f64{ 0, 1, 2, 3, 4, 5 };
    const ys = [_]f64{ 0, 1, 8, 27, 64, 125 }; // y = x^3 samples
    var s = try Spline.init(.cspline, &xs, &ys);
    defer s.deinit();
    var acc = try Accel.init();
    defer acc.deinit();

    var x: f64 = 0.25;
    while (x < 5.0) : (x += 0.5) {
        try testing.expectEqual(try s.eval(x, null), try s.eval(x, acc));
    }
    acc.reset(); // exercise reset
}

test "interp: every type instantiates, reports its size, and evaluates" {
    // Enough strictly-increasing points for even the akima family (minSize 5).
    var xs: [8]f64 = undefined;
    var ys: [8]f64 = undefined;
    for (&xs, &ys, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i);
        y.* = @as(f64, @floatFromInt(i)) * 0.5 + 1.0; // monotone (ok for steffen)
    }
    inline for (comptime std.enums.values(Type)) |t| {
        try testing.expect(t.minSize() >= 2);
        var s = try Spline.init(t, &xs, &ys);
        defer s.deinit();
        // Midpoint eval succeeds and the value/derivative/integral all link.
        _ = try s.eval(3.5, null);
        _ = try s.evalDeriv(3.5, null);
        _ = try s.evalDeriv2(3.5, null);
        _ = try s.evalInteg(1.0, 6.0, null);
    }
}
