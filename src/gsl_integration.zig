//! Idiomatic Zig bindings for the GNU Scientific Library's numerical
//! integration module (`gsl_integration`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the numerical
//! quadrature chapter — the second consumer of the shared callback bridge
//! (`gsl.callback`), and the first that pairs a callback with a reusable
//! workspace. It reuses `gsl.zig`'s process-global error-handler switch but
//! keeps the `gsl_integration` C API behind its own `c`. It is reached as
//! `gsl.integration`.
//!
//! ## Shape of the surface
//!
//! Each routine integrates a caller-supplied function over an interval,
//! returning both the estimate and an absolute error bound (`Result`). The
//! integrand is passed as a `Callback` value, constructed with either factory
//! method (see `gsl.callback`):
//!
//!   - `.initFn(f)`   — a plain `*const fn(f64) f64` (a bare function coerces in).
//!   - `.initCtx(ctx)` — a pointer to a struct with `pub fn eval(self, x) f64`,
//!     for integrands that capture parameters (no allocation).
//!
//! The routines split into a non-adaptive front door and the adaptive family:
//!
//!   - `qng`   — non-adaptive Gauss-Kronrod on `[a, b]`. Needs *no* workspace;
//!               the simplest way to integrate a smooth function.
//!   - `qag`   — adaptive Gauss-Kronrod on `[a, b]` with a chosen rule `Key`.
//!   - `qags`  — adaptive with extrapolation; handles integrable singularities.
//!   - `qagi`  — adaptive over the whole line `(-inf, +inf)`.
//!   - `qagiu` — adaptive over `[a, +inf)`.
//!   - `qagil` — adaptive over `(-inf, b]`.
//!
//! The adaptive routines subdivide the interval and store the subintervals in a
//! `Workspace`; allocate one (sized by the maximum number of subintervals) and
//! reuse it across calls.
//!
//! Every routine takes its error target as a single `Tol{ .abs, .rel }` value
//! (rather than two adjacent `f64` tolerances) so the two cannot be transposed;
//! it defaults to a purely relative target (`.{ .rel = 1e-9 }`).
//!
//! ## Omissions
//!
//!   - The weighted/oscillatory and singular-point families (`qawo`, `qawc`,
//!     `qaws`, `qawf`, `qagp`), the CQUAD (`gsl_integration_cquad`), Romberg
//!     (`gsl_integration_romberg`), fixed-point (`gsl_integration_fixed`), and
//!     Gauss-Legendre table (`gsl_integration_glfixed`) routines are not yet
//!     wrapped; the raw C API remains available through `c` for them.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_math.h");
    @cInclude("gsl/gsl_integration.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the integration routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_integration_*` symbol if you need the
/// exact code.
pub const Error = error{
    /// `GSL_EDOM` — a domain error inside the evaluation (e.g. bad tolerances).
    Domain,
    /// `GSL_EINVAL` — an invalid argument (e.g. a workspace too small).
    Invalid,
    /// `GSL_ETOL` — the requested tolerance could not be reached.
    Tolerance,
    /// `GSL_EMAXITER` — the maximum number of subdivisions was exceeded.
    MaxIterations,
    /// `GSL_EROUND` — round-off error prevents the tolerance from being reached.
    RoundOff,
    /// `GSL_ESING` — a non-integrable singularity or other bad integrand
    /// behavior was detected.
    Singularity,
    /// `GSL_EDIVERGE` — the integral is divergent, or slowly convergent.
    Diverge,
    /// `GSL_ENOMEM` — workspace allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EDOM => Error.Domain,
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_ETOL => Error.Tolerance,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_EROUND => Error.RoundOff,
        c.GSL_ESING => Error.Singularity,
        c.GSL_EDIVERGE => Error.Diverge,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// An integral estimate and its absolute error bound.
pub const Result = struct {
    /// The estimated value of the integral.
    value: f64,
    /// GSL's estimate of the absolute error in `value`.
    abserr: f64,
};

/// The requested error target for a routine. A routine stops once *either*
/// criterion is met, so a purely relative target is the common case (leave
/// `abs` at 0). Grouping the two tolerances into one named-field value keeps
/// them from being transposed at a call site (both are `f64`).
pub const Tol = struct {
    /// Absolute error target (`epsabs`). 0 disables the absolute criterion.
    abs: f64 = 0,
    /// Relative error target (`epsrel`).
    rel: f64 = 1e-9,
};

/// The Gauss-Kronrod rule used by `qag`. Higher-order rules cost more per
/// subinterval but converge faster for smooth integrands; lower-order rules are
/// more robust for functions with local difficulties.
pub const Key = enum {
    /// 15-point Gauss-Kronrod rule.
    gauss15,
    /// 21-point Gauss-Kronrod rule.
    gauss21,
    /// 31-point Gauss-Kronrod rule.
    gauss31,
    /// 41-point Gauss-Kronrod rule.
    gauss41,
    /// 51-point Gauss-Kronrod rule.
    gauss51,
    /// 61-point Gauss-Kronrod rule.
    gauss61,

    fn toC(self: Key) c_int {
        return switch (self) {
            .gauss15 => c.GSL_INTEG_GAUSS15,
            .gauss21 => c.GSL_INTEG_GAUSS21,
            .gauss31 => c.GSL_INTEG_GAUSS31,
            .gauss41 => c.GSL_INTEG_GAUSS41,
            .gauss51 => c.GSL_INTEG_GAUSS51,
            .gauss61 => c.GSL_INTEG_GAUSS61,
        };
    }
};

/// A `gsl_function`-shaped callback value for this chapter. Construct with
/// `Callback.initFn(f)` (a plain function) or `Callback.initCtx(&ctx)` (a
/// context struct); at a call site the leading dot suffices, e.g.
/// `qng(.initFn(myFn), 0, 1, 1e-8, 1e-8)`.
pub const Callback = callback.Function(c.gsl_function);

/// A reusable workspace for the adaptive routines. It holds the subinterval
/// bookkeeping for up to `limit` subdivisions. Owns its GSL allocation; call
/// `deinit` to free.
///
/// Example:
/// ```
/// var ws = try gsl.integration.Workspace.init(1000);
/// defer ws.deinit();
/// const r = try gsl.integration.qags(.initFn(f), 0, 1, 0, 1e-9, &ws);
/// ```
pub const Workspace = struct {
    ptr: *c.gsl_integration_workspace,

    /// Allocate a workspace holding up to `max_intervals` subintervals, which
    /// must be at least 1; a few hundred to a few thousand is typical for the
    /// adaptive routines.
    pub fn init(max_intervals: usize) Error!Workspace {
        if (max_intervals == 0) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_integration_workspace_alloc(max_intervals) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Workspace) void {
        c.gsl_integration_workspace_free(self.ptr);
    }

    /// The maximum number of subintervals this workspace was allocated for. The
    /// adaptive routines use this as their subdivision limit.
    pub fn limit(self: Workspace) usize {
        return self.ptr.*.limit;
    }
};

/// Non-adaptive Gauss-Kronrod integration of `f` over `[a, b]`. Requires no
/// workspace; a good first choice for smooth integrands. Stops once `tol` is
/// met (see `Tol`).
///
/// Example:
/// ```
/// fn f(x: f64) f64 { return @sin(x); }
/// const r = try gsl.integration.qng(.initFn(f), 0, std.math.pi, .{ .rel = 1e-9 });
/// // r.value ≈ 2
/// ```
pub fn qng(cb: Callback, a: f64, b: f64, tol: Tol) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    var neval: usize = undefined;
    try check(c.gsl_integration_qng(&cb.gf, a, b, tol.abs, tol.rel, &result, &abserr, &neval));
    return .{ .value = result, .abserr = abserr };
}

/// Adaptive Gauss-Kronrod integration of `f` over `[a, b]` using rule `key`.
/// Subdivides where the integrand is difficult, up to `ws.limit()` subintervals.
pub fn qag(cb: Callback, a: f64, b: f64, tol: Tol, key: Key, ws: *Workspace) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    try check(c.gsl_integration_qag(&cb.gf, a, b, tol.abs, tol.rel, ws.ptr.*.limit, key.toC(), ws.ptr, &result, &abserr));
    return .{ .value = result, .abserr = abserr };
}

/// Adaptive integration of `f` over `[a, b]` with extrapolation, suitable for
/// integrands with integrable endpoint or interior singularities.
pub fn qags(cb: Callback, a: f64, b: f64, tol: Tol, ws: *Workspace) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    try check(c.gsl_integration_qags(&cb.gf, a, b, tol.abs, tol.rel, ws.ptr.*.limit, ws.ptr, &result, &abserr));
    return .{ .value = result, .abserr = abserr };
}

/// Adaptive integration of `f` over the whole real line `(-inf, +inf)`. The
/// infinite range is mapped onto a finite interval internally.
pub fn qagi(cb: Callback, tol: Tol, ws: *Workspace) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    // GSL declares `f` non-const here (unlike qng/qag/qags) purely because the
    // infinite-range transform stashes the pointer in a non-const internal
    // field; the callback is only read, never mutated, so the cast is safe.
    try check(c.gsl_integration_qagi(@constCast(&cb.gf), tol.abs, tol.rel, ws.ptr.*.limit, ws.ptr, &result, &abserr));
    return .{ .value = result, .abserr = abserr };
}

/// Adaptive integration of `f` over the half-infinite range `[a, +inf)`.
pub fn qagiu(cb: Callback, a: f64, tol: Tol, ws: *Workspace) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    // Non-const `f` is a GSL API artifact; see `qagi`. The callback is only read.
    try check(c.gsl_integration_qagiu(@constCast(&cb.gf), a, tol.abs, tol.rel, ws.ptr.*.limit, ws.ptr, &result, &abserr));
    return .{ .value = result, .abserr = abserr };
}

/// Adaptive integration of `f` over the half-infinite range `(-inf, b]`.
pub fn qagil(cb: Callback, b: f64, tol: Tol, ws: *Workspace) Error!Result {
    gsl.ensureHandler();
    var result: f64 = undefined;
    var abserr: f64 = undefined;
    // Non-const `f` is a GSL API artifact; see `qagi`. The callback is only read.
    try check(c.gsl_integration_qagil(@constCast(&cb.gf), b, tol.abs, tol.rel, ws.ptr.*.limit, ws.ptr, &result, &abserr));
    return .{ .value = result, .abserr = abserr };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn sinFn(x: f64) f64 {
    return @sin(x);
}

fn oneOverSqrt(x: f64) f64 {
    return 1.0 / @sqrt(x);
}

fn gaussian(x: f64) f64 {
    return @exp(-x * x);
}

test "integration: qng of sin over [0, pi] is 2" {
    const r = try qng(.initFn(sinFn), 0.0, std.math.pi, .{ .rel = 1e-9 });
    try testing.expectApproxEqAbs(@as(f64, 2.0), r.value, 1e-9);
    try testing.expect(r.abserr >= 0.0);
}

test "integration: qag of sin over [0, pi] is 2 with every rule" {
    var ws = try Workspace.init(100);
    defer ws.deinit();
    inline for (comptime std.enums.values(Key)) |key| {
        const r = try qag(.initFn(sinFn), 0.0, std.math.pi, .{ .rel = 1e-9 }, key, &ws);
        try testing.expectApproxEqAbs(@as(f64, 2.0), r.value, 1e-8);
    }
}

test "integration: qags handles an endpoint singularity (∫₀¹ 1/√x = 2)" {
    var ws = try Workspace.init(1000);
    defer ws.deinit();
    const r = try qags(.initFn(oneOverSqrt), 0.0, 1.0, .{ .rel = 1e-9 }, &ws);
    try testing.expectApproxEqAbs(@as(f64, 2.0), r.value, 1e-7);
}

test "integration: qagi of a Gaussian over the whole line is √π" {
    var ws = try Workspace.init(1000);
    defer ws.deinit();
    const r = try qagi(.initFn(gaussian), .{ .rel = 1e-9 }, &ws);
    try testing.expectApproxEqAbs(@sqrt(std.math.pi), r.value, 1e-8);
}

test "integration: qagiu and qagil split the Gaussian into equal halves" {
    var ws = try Workspace.init(1000);
    defer ws.deinit();
    const upper = try qagiu(.initFn(gaussian), 0.0, .{ .rel = 1e-9 }, &ws);
    const lower = try qagil(.initFn(gaussian), 0.0, .{ .rel = 1e-9 }, &ws);
    const half = @sqrt(std.math.pi) / 2.0;
    try testing.expectApproxEqAbs(half, upper.value, 1e-8);
    try testing.expectApproxEqAbs(half, lower.value, 1e-8);
}

test "integration: a context struct captures an integrand parameter" {
    // ∫₀^π sin(k·x) dx = (1 − cos(kπ)) / k.
    const Wave = struct {
        k: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            return @sin(self.k * x);
        }
    };
    var w = Wave{ .k = 3.0 };
    const r = try qng(.initCtx(&w), 0.0, std.math.pi, .{ .rel = 1e-9 });
    const want = (1.0 - @cos(w.k * std.math.pi)) / w.k;
    try testing.expectApproxEqAbs(want, r.value, 1e-9);
}

test "integration: a Callback and Workspace can both be reused" {
    const cb: Callback = .initFn(sinFn);
    var ws = try Workspace.init(200);
    defer ws.deinit();
    const a = try qag(cb, 0.0, std.math.pi, .{ .rel = 1e-9 }, .gauss21, &ws);
    const b = try qag(cb, 0.0, std.math.pi / 2.0, .{ .rel = 1e-9 }, .gauss21, &ws);
    try testing.expectApproxEqAbs(@as(f64, 2.0), a.value, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.0), b.value, 1e-9);
}

test "integration: Tol defaults to a purely relative target" {
    // With no fields set, `abs = 0` and `rel = 1e-9`, so a smooth integrand
    // still converges to its closed form.
    const r = try qng(.initFn(sinFn), 0.0, std.math.pi, .{});
    try testing.expectApproxEqAbs(@as(f64, 2.0), r.value, 1e-8);
}

test "integration: workspace exposes its allocated limit and rejects zero" {
    var ws = try Workspace.init(512);
    defer ws.deinit();
    try testing.expectEqual(@as(usize, 512), ws.limit());
    try testing.expectError(Error.Invalid, Workspace.init(0));
}

test "integration: an unreachable tolerance surfaces as an Error, not an abort" {
    // Demand an absurd tolerance on a genuinely hard (oscillatory-ish) case so
    // GSL exhausts its subdivisions and returns a nonzero status.
    var ws = try Workspace.init(10);
    defer ws.deinit();
    const s = struct {
        fn f(x: f64) f64 {
            return @sin(1.0 / x);
        }
    }.f;
    try testing.expectError(Error.MaxIterations, qag(.initFn(s), 1e-6, 1.0, .{ .rel = 1e-12 }, .gauss15, &ws));
}
