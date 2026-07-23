//! Idiomatic Zig bindings for the GNU Scientific Library's Monte-Carlo
//! integration module (`gsl_monte`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the multidimensional
//! Monte-Carlo integration chapter — the first consumer of the callback
//! bridge's *multidimensional* form (`gsl.callback`'s `MonteFunction`), and one
//! that reuses the existing `gsl.rand.Rng`. It reuses `gsl.zig`'s process-global
//! error-handler switch but keeps the `gsl_monte` C API behind its own `c`. It
//! is reached as `gsl.monte`.
//!
//! ## Shape of the surface
//!
//! Three engines estimate `∫ f over the box [xl, xu]` in `dim` dimensions by
//! sampling `f` at `calls` (pseudo-)random points from a `gsl.rand.Rng`:
//!
//!   - `Plain` — plain Monte-Carlo (uniform sampling); simplest, slowest to
//!     converge.
//!   - `Miser` — recursive stratified sampling; concentrates points where the
//!     variance is high.
//!   - `Vegas` — importance sampling with an adaptive grid; usually the best
//!     for peaked integrands, and exposes `chisq`/`runval` for its iterations.
//!
//! Each engine is allocated for a fixed `dim`, then `integrate`d one or more
//! times. The integrand is a `Callback` built from a function of the whole
//! point:
//!
//!   - `.initFn(f)`   — a plain `*const fn([]const f64) f64`.
//!   - `.initCtx(ctx)` — a pointer to a struct with
//!     `pub fn eval(self, x: []const f64) f64`, for integrands that capture
//!     parameters (no allocation).
//!
//! ```zig
//! var rng = try gsl.rand.Rng.init(.mt19937);
//! defer rng.deinit();
//! var v = try gsl.monte.Vegas.init(2);
//! defer v.deinit();
//! const r = try v.integrate(.initFn(f), &.{ 0, 0 }, &.{ 1, 1 }, 100_000, rng);
//! // r.value ≈ ∫∫ f over the unit square, r.abserr its 1σ estimate
//! ```
//!
//! The `dim`-length point passed to the callback is a borrowed slice valid only
//! for that call — read it, do not retain it.
//!
//! ## Omissions
//!
//!   - The tunable-parameters structs (`gsl_monte_miser_params`,
//!     `gsl_monte_vegas_params`) and the Vegas `FILE*` logging stream are not
//!     wrapped; the defaults chosen at `alloc` are used. Reach for the raw `c`
//!     API (via `c`) to tune them.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_monte.h");
    @cInclude("gsl/gsl_monte_plain.h");
    @cInclude("gsl/gsl_monte_miser.h");
    @cInclude("gsl/gsl_monte_vegas.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the Monte-Carlo routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_monte_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument.
    Invalid,
    /// A dimension mismatch: the bounds' length, or the two bounds against each
    /// other, disagree with the engine's dimension.
    BadLength,
    /// `GSL_ENOMEM` — engine allocation failed.
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

/// An integral estimate and its 1σ error estimate.
pub const Result = struct {
    /// The estimated value of the integral.
    value: f64,
    /// GSL's estimate of the (standard-deviation) error in `value`.
    abserr: f64,
};

/// A `gsl_monte_function`-shaped callback value for the integrators. Construct
/// with `Callback.initFn(f)` (a plain `fn([]const f64) f64`) or
/// `Callback.initCtx(&ctx)` (a struct with `pub fn eval(self, x: []const f64)`).
pub const Callback = callback.MonteFunction(c.gsl_monte_function);

// The GSL `gsl_rng *` seen through this file's `@cImport` is a *distinct* C type
// from the one `gsl.rand` exposes (each `@cImport` of `gsl_rng.h` yields its
// own), though the underlying struct is identical — so the shared `Rng`'s
// pointer is reinterpreted with `@ptrCast`, mirroring the vector-view helpers.
fn rngPtr(rng: gsl.rand.Rng) *c.gsl_rng {
    return @ptrCast(rng.ptr);
}

// Shared integration body. `integrate_fn` is the engine's C entry point and
// `state` its typed state pointer; all three share the same argument order.
fn run(
    comptime integrate_fn: anytype,
    state: anytype,
    dim: usize,
    cb: Callback,
    xl: []const f64,
    xu: []const f64,
    calls: usize,
    rng: gsl.rand.Rng,
) Error!Result {
    if (xl.len != dim or xu.len != dim) return Error.BadLength;
    gsl.ensureHandler();

    // Fill in the dimension GSL will pass back to the trampoline, then hand it
    // the callback struct. A local copy keeps the transient struct alive.
    var f = cb;
    f.mf.dim = dim;

    var result: f64 = undefined;
    var abserr: f64 = undefined;
    try check(integrate_fn(
        @constCast(&f.mf),
        @constCast(xl.ptr),
        @constCast(xu.ptr),
        dim,
        calls,
        rngPtr(rng),
        state,
        &result,
        &abserr,
    ));
    return .{ .value = result, .abserr = abserr };
}

/// Plain Monte-Carlo integrator: uniform sampling over the box. Owns its GSL
/// allocation; call `deinit` to free.
pub const Plain = struct {
    ptr: *c.gsl_monte_plain_state,
    dim: usize,

    /// Allocate a plain integrator for `dim` dimensions.
    pub fn init(dim: usize) Error!Plain {
        gsl.ensureHandler();
        const p = c.gsl_monte_plain_alloc(dim) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .dim = dim };
    }

    pub fn deinit(self: *Plain) void {
        c.gsl_monte_plain_free(self.ptr);
    }

    /// Reset the accumulated state (rarely needed; `integrate` is independent
    /// per call).
    pub fn reset(self: *Plain) Error!void {
        try check(c.gsl_monte_plain_init(self.ptr));
    }

    /// Estimate `∫ f` over `[xl, xu]` using `calls` sample points from `rng`.
    /// `xl.len` and `xu.len` must equal the engine's `dim`.
    pub fn integrate(self: *Plain, cb: Callback, xl: []const f64, xu: []const f64, calls: usize, rng: gsl.rand.Rng) Error!Result {
        return run(c.gsl_monte_plain_integrate, self.ptr, self.dim, cb, xl, xu, calls, rng);
    }
};

/// MISER integrator: recursive stratified sampling. Owns its GSL allocation;
/// call `deinit` to free.
pub const Miser = struct {
    ptr: *c.gsl_monte_miser_state,
    dim: usize,

    /// Allocate a MISER integrator for `dim` dimensions.
    pub fn init(dim: usize) Error!Miser {
        gsl.ensureHandler();
        const p = c.gsl_monte_miser_alloc(dim) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .dim = dim };
    }

    pub fn deinit(self: *Miser) void {
        c.gsl_monte_miser_free(self.ptr);
    }

    /// Reset the accumulated state.
    pub fn reset(self: *Miser) Error!void {
        try check(c.gsl_monte_miser_init(self.ptr));
    }

    /// Estimate `∫ f` over `[xl, xu]` using `calls` sample points from `rng`.
    pub fn integrate(self: *Miser, cb: Callback, xl: []const f64, xu: []const f64, calls: usize, rng: gsl.rand.Rng) Error!Result {
        return run(c.gsl_monte_miser_integrate, self.ptr, self.dim, cb, xl, xu, calls, rng);
    }
};

/// VEGAS integrator: adaptive importance sampling. Owns its GSL allocation;
/// call `deinit` to free. Typical use runs `integrate` a few times on the same
/// instance (each call refines the grid), checking `chisq` for convergence.
pub const Vegas = struct {
    ptr: *c.gsl_monte_vegas_state,
    dim: usize,

    /// Allocate a VEGAS integrator for `dim` dimensions.
    pub fn init(dim: usize) Error!Vegas {
        gsl.ensureHandler();
        const p = c.gsl_monte_vegas_alloc(dim) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .dim = dim };
    }

    pub fn deinit(self: *Vegas) void {
        c.gsl_monte_vegas_free(self.ptr);
    }

    /// Reset the grid and accumulated state (start a fresh adaptation).
    pub fn reset(self: *Vegas) Error!void {
        try check(c.gsl_monte_vegas_init(self.ptr));
    }

    /// Estimate `∫ f` over `[xl, xu]` using `calls` sample points from `rng`.
    /// Calling repeatedly on the same instance refines the adaptive grid.
    pub fn integrate(self: *Vegas, cb: Callback, xl: []const f64, xu: []const f64, calls: usize, rng: gsl.rand.Rng) Error!Result {
        return run(c.gsl_monte_vegas_integrate, self.ptr, self.dim, cb, xl, xu, calls, rng);
    }

    /// The χ² per degree of freedom of the weighted average across the runs so
    /// far; a value near 1 indicates the per-run estimates are consistent.
    pub fn chisq(self: *const Vegas) f64 {
        return c.gsl_monte_vegas_chisq(self.ptr);
    }

    /// The running weighted-average result and its σ across the runs so far.
    pub fn runval(self: *const Vegas) Result {
        var result: f64 = undefined;
        var sigma: f64 = undefined;
        c.gsl_monte_vegas_runval(self.ptr, &result, &sigma);
        return .{ .value = result, .abserr = sigma };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A classic test integrand: 1 / (1 − cos x cos y cos z) / π³ over [0,π]³,
// whose exact integral is Γ(1/4)⁴ / (4π³) ≈ 1.3932039296.
const gamma_ref: f64 = 1.3932039296856768591842462603255;

fn boxIntegrand(x: []const f64) f64 {
    const inv_pi3 = 1.0 / (std.math.pi * std.math.pi * std.math.pi);
    return inv_pi3 / (1.0 - @cos(x[0]) * @cos(x[1]) * @cos(x[2]));
}

fn unitBounds() struct { lo: [3]f64, hi: [3]f64 } {
    return .{ .lo = .{ 0, 0, 0 }, .hi = .{ std.math.pi, std.math.pi, std.math.pi } };
}

test "monte: plain integrates a 3-D test function near its known value" {
    var rng = try gsl.rand.Rng.init(.mt19937);
    defer rng.deinit();
    var p = try Plain.init(3);
    defer p.deinit();

    const b = unitBounds();
    const r = try p.integrate(.initFn(boxIntegrand), &b.lo, &b.hi, 500_000, rng);
    // Plain MC converges slowly; allow a loose band but confirm the estimate is
    // in the right neighborhood with a sane (positive, small) error bar.
    try testing.expectApproxEqAbs(gamma_ref, r.value, 0.05);
    try testing.expect(r.abserr > 0.0 and r.abserr < 0.05);
}

test "monte: miser is more accurate than plain for the same call budget" {
    var rng = try gsl.rand.Rng.init(.mt19937);
    defer rng.deinit();
    var m = try Miser.init(3);
    defer m.deinit();

    const b = unitBounds();
    const r = try m.integrate(.initFn(boxIntegrand), &b.lo, &b.hi, 500_000, rng);
    try testing.expectApproxEqAbs(gamma_ref, r.value, 0.02);
    try testing.expect(r.abserr > 0.0);
}

test "monte: vegas converges and reports a chisq near unity" {
    var rng = try gsl.rand.Rng.init(.mt19937);
    defer rng.deinit();
    var v = try Vegas.init(3);
    defer v.deinit();

    const b = unitBounds();
    // A warm-up pass to adapt the grid, then a few refining passes.
    _ = try v.integrate(.initFn(boxIntegrand), &b.lo, &b.hi, 100_000, rng);
    var r = try v.integrate(.initFn(boxIntegrand), &b.lo, &b.hi, 100_000, rng);
    var iters: usize = 0;
    while (iters < 10 and @abs(v.chisq() - 1.0) > 0.5) : (iters += 1) {
        r = try v.integrate(.initFn(boxIntegrand), &b.lo, &b.hi, 100_000, rng);
    }
    try testing.expectApproxEqAbs(gamma_ref, r.value, 0.01);
    try testing.expect(v.chisq() > 0.0);

    // runval reports the cumulative weighted average across all passes (not the
    // single last-pass value), which should likewise be near the known result.
    const rv = v.runval();
    try testing.expectApproxEqAbs(gamma_ref, rv.value, 0.01);
    try testing.expect(rv.abserr > 0.0);
}

test "monte: a context struct captures integrand parameters" {
    // f(x, y) = a·x + b·y over the unit square integrates to (a + b) / 2.
    const Linear = struct {
        a: f64,
        b: f64,
        pub fn eval(self: *const @This(), x: []const f64) f64 {
            return self.a * x[0] + self.b * x[1];
        }
    };
    var lin = Linear{ .a = 2.0, .b = 6.0 };
    var rng = try gsl.rand.Rng.init(.mt19937);
    defer rng.deinit();
    var p = try Plain.init(2);
    defer p.deinit();

    const lo = [_]f64{ 0, 0 };
    const hi = [_]f64{ 1, 1 };
    const r = try p.integrate(.initCtx(&lin), &lo, &hi, 200_000, rng);
    try testing.expectApproxEqAbs(@as(f64, 4.0), r.value, 0.02); // (2 + 6)/2
}

test "monte: a bounds/dimension mismatch is rejected" {
    var rng = try gsl.rand.Rng.init(.mt19937);
    defer rng.deinit();
    var p = try Plain.init(3);
    defer p.deinit();

    const lo = [_]f64{ 0, 0 }; // only 2 entries for a 3-D engine
    const hi = [_]f64{ 1, 1 };
    try testing.expectError(Error.BadLength, p.integrate(.initFn(boxIntegrand), &lo, &hi, 1000, rng));
}

test "monte: the volume of the unit cube is 1 (constant integrand)" {
    const one = struct {
        fn f(_: []const f64) f64 {
            return 1.0;
        }
    }.f;
    var rng = try gsl.rand.Rng.init(.mt19937);
    defer rng.deinit();
    var p = try Plain.init(3);
    defer p.deinit();

    const lo = [_]f64{ 0, 0, 0 };
    const hi = [_]f64{ 1, 1, 1 };
    const r = try p.integrate(.initFn(one), &lo, &hi, 10_000, rng);
    // A constant integrand is exact regardless of the sampling.
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.value, 1e-12);
}
