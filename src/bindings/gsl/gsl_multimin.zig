//! Idiomatic Zig bindings for the GNU Scientific Library's multidimensional
//! minimization module (`gsl_multimin`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the multidimensional
//! function-minimization chapter. It is the vector-valued analogue of `gsl.min`:
//! a stateful solver you `init`, `set`, and `iterate` to drive a point toward a
//! local minimum of a scalar objective `f: R^n -> R`. It reuses `gsl.zig`'s
//! process-global error-handler switch but keeps the `gsl_multimin` C API behind
//! its own `c`. It is reached as `gsl.multimin`.
//!
//! ## Two solver families
//!
//! GSL offers two kinds of minimizer, and this binding wraps both:
//!
//!   - `Minimizer` — *derivative-free* (`gsl_multimin_fminimizer`). You supply
//!     only the objective `f` and a per-coordinate initial step; the Nelder-Mead
//!     simplex algorithms explore with function evaluations alone. Convergence is
//!     tested on the simplex `size` (`testSize`).
//!   - `GradientMinimizer` — *gradient-based* (`gsl_multimin_fdfminimizer`). You
//!     supply `f` and its gradient `∇f` (conjugate-gradient, BFGS, or steepest
//!     descent). Convergence is tested on the gradient norm (`testGradient`).
//!
//! ## Shape of the surface (derivative-free)
//!
//! ```zig
//! // Minimize f(x, y) = (x - 1)² + (y - 2)²; the minimum sits at (1, 2).
//! const Paraboloid = struct {
//!     pub fn eval(_: *const @This(), x: []const f64) f64 {
//!         return (x[0] - 1) * (x[0] - 1) + (x[1] - 2) * (x[1] - 2);
//!     }
//! };
//! var p = Paraboloid{};
//! var func = gsl.multimin.Function.initCtx(&p, 2);
//! var m = try gsl.multimin.Minimizer.init(.nmsimplex2, 2);
//! defer m.deinit();
//! try m.set(&func, &.{ 5.0, 5.0 }, &.{ 1.0, 1.0 });
//! while (true) {
//!     try m.iterate();
//!     if (try m.testSize(1e-4)) break;
//! }
//! var xmin: [2]f64 = undefined;
//! m.xInto(&xmin);          // ≈ { 1, 2 }
//! const fmin = m.minimum(); // ≈ 0
//! ```
//!
//! The gradient family is identical except `set` takes a scalar `step_size` and
//! `tol`, the context also declares `pub fn gradient(self, x, g: []f64)` (and may
//! declare a fused `pub fn evalGradient(self, x, f: *f64, g: []f64)`), and you
//! test with `testGradient`.
//!
//! ## Lifetime
//!
//! Like `gsl.min`, both solvers retain a *pointer* to the callback bundle across
//! iterations. The bundle lives in a caller-owned `Function`/`FunctionFdf`; keep
//! it — and the context it wraps — alive and unmoved between `set` and the final
//! `iterate`. The solver itself must also not be moved across that span.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_multimin.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the minimization routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_multimin_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument (e.g. a length mismatch between the
    /// initial point and the problem dimension, or a nonsensical tolerance).
    Invalid,
    /// `GSL_EBADFUNC` — the objective or gradient produced a singular (Inf/NaN)
    /// value.
    BadFunction,
    /// `GSL_ENOPROG` — the iteration cannot improve the current estimate (the
    /// simplex or line search has stalled).
    NoProgress,
    /// `GSL_EMAXITER` — the iteration failed to make progress.
    MaxIterations,
    /// `GSL_ENOMEM` — solver allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADFUNC => Error.BadFunction,
        c.GSL_ENOPROG => Error.NoProgress,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// The derivative-free minimization algorithm used by `Minimizer`, mirroring
/// GSL's `gsl_multimin_fminimizer_*` types. All are Nelder-Mead simplex variants.
pub const Method = enum {
    /// The improved (O(n) per step) Nelder-Mead simplex — the usual default.
    nmsimplex2,
    /// The original O(n²) Nelder-Mead simplex.
    nmsimplex,
    /// `nmsimplex2` with a randomized initial simplex orientation.
    nmsimplex2rand,

    fn typePtr(self: Method) *const c.gsl_multimin_fminimizer_type {
        return switch (self) {
            .nmsimplex2 => c.gsl_multimin_fminimizer_nmsimplex2,
            .nmsimplex => c.gsl_multimin_fminimizer_nmsimplex,
            .nmsimplex2rand => c.gsl_multimin_fminimizer_nmsimplex2rand,
        };
    }
};

/// The gradient-based minimization algorithm used by `GradientMinimizer`,
/// mirroring GSL's `gsl_multimin_fdfminimizer_*` types.
pub const GradientMethod = enum {
    /// Fletcher-Reeves conjugate gradient.
    conjugate_fr,
    /// Polak-Ribiere conjugate gradient.
    conjugate_pr,
    /// Broyden-Fletcher-Goldfarb-Shanno (BFGS).
    vector_bfgs,
    /// The improved, more efficient BFGS variant — the usual default.
    vector_bfgs2,
    /// Steepest descent (mostly a reference/testing baseline).
    steepest_descent,

    fn typePtr(self: GradientMethod) *const c.gsl_multimin_fdfminimizer_type {
        return switch (self) {
            .conjugate_fr => c.gsl_multimin_fdfminimizer_conjugate_fr,
            .conjugate_pr => c.gsl_multimin_fdfminimizer_conjugate_pr,
            .vector_bfgs => c.gsl_multimin_fdfminimizer_vector_bfgs,
            .vector_bfgs2 => c.gsl_multimin_fdfminimizer_vector_bfgs2,
            .steepest_descent => c.gsl_multimin_fdfminimizer_steepest_descent,
        };
    }
};

/// A `gsl_multimin_function`-shaped objective callback bundle for the
/// derivative-free `Minimizer`. Construct from a context pointer with `initCtx`;
/// the context must declare `pub fn eval(self, x: []const f64) f64`.
pub const Function = struct {
    raw: c.gsl_multimin_function,

    /// Wrap a context declaring `pub fn eval(self, x) f64` over an `n`-dimensional
    /// domain.
    pub fn initCtx(ctx: anytype, n: usize) Function {
        return .{ .raw = callback.multiminF(c.gsl_multimin_function, c.gsl_vector, n, ctx) };
    }
};

/// A `gsl_multimin_function_fdf`-shaped objective+gradient callback bundle for
/// the `GradientMinimizer`. Construct from a context pointer with `initCtx`; the
/// context must declare `pub fn eval(self, x) f64` and `pub fn gradient(self, x,
/// g: []f64)`, and may declare a fused `pub fn evalGradient(self, x, f: *f64, g:
/// []f64)`.
pub const FunctionFdf = struct {
    raw: c.gsl_multimin_function_fdf,

    /// Wrap a context declaring `eval` + `gradient` over an `n`-dimensional domain.
    pub fn initCtx(ctx: anytype, n: usize) FunctionFdf {
        return .{ .raw = callback.multiminFdf(c.gsl_multimin_function_fdf, c.gsl_vector, n, ctx) };
    }
};

/// A stateful derivative-free multidimensional minimizer (Nelder-Mead simplex).
/// Owns its GSL allocation (`deinit` frees it). Once `set`, GSL holds a pointer
/// to the caller's `Function`, so keep it alive and do not move this minimizer
/// between `set` and the last `iterate`.
pub const Minimizer = struct {
    ptr: *c.gsl_multimin_fminimizer,
    n: usize,

    /// Allocate an `n`-dimensional minimizer using algorithm `method`.
    pub fn init(method: Method, n: usize) Error!Minimizer {
        gsl.ensureHandler();
        const p = c.gsl_multimin_fminimizer_alloc(method.typePtr(), n) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .n = n };
    }

    pub fn deinit(self: *Minimizer) void {
        c.gsl_multimin_fminimizer_free(self.ptr);
    }

    /// Provide the objective `func`, an initial point `x0`, and the initial
    /// per-coordinate simplex step sizes (both length `n`). May be called again
    /// to restart the minimizer on a new starting point.
    pub fn set(self: *Minimizer, func: *Function, x0: []const f64, step: []const f64) Error!void {
        if (x0.len != self.n or step.len != self.n) return Error.Invalid;
        gsl.ensureHandler();
        var xv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(x0));
        var sv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(step));
        try check(c.gsl_multimin_fminimizer_set(self.ptr, &func.raw, &xv, &sv));
    }

    /// Perform one simplex iteration.
    pub fn iterate(self: *Minimizer) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_multimin_fminimizer_iterate(self.ptr));
    }

    /// Copy the current best estimate of the minimizer's *location* (length `n`)
    /// into `out`.
    pub fn xInto(self: *const Minimizer, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multimin_fminimizer_x(self.ptr));
    }

    /// The objective value at the current estimate.
    pub fn minimum(self: *const Minimizer) f64 {
        return c.gsl_multimin_fminimizer_minimum(self.ptr);
    }

    /// A characteristic size of the current simplex — the value fed to
    /// `testSize`. Shrinks toward zero as the simplex collapses onto the minimum.
    pub fn size(self: *const Minimizer) f64 {
        return c.gsl_multimin_fminimizer_size(self.ptr);
    }

    /// The standardized name of the algorithm (e.g. `"nmsimplex2"`).
    pub fn name(self: *const Minimizer) [:0]const u8 {
        return std.mem.span(c.gsl_multimin_fminimizer_name(self.ptr));
    }

    /// Test whether the simplex has collapsed to within `epsabs` (via
    /// `gsl_multimin_test_size` on the current `size`). Returns `true` once
    /// converged, `false` to keep iterating, and an `Error` for a bad argument.
    pub fn testSize(self: *const Minimizer, epsabs: f64) Error!bool {
        return testStatus(c.gsl_multimin_test_size(self.size(), epsabs));
    }
};

/// A stateful gradient-based multidimensional minimizer (conjugate gradient,
/// BFGS, or steepest descent). Owns its GSL allocation (`deinit` frees it). Once
/// `set`, GSL holds a pointer to the caller's `FunctionFdf`; keep it alive and do
/// not move this minimizer between `set` and the last `iterate`.
pub const GradientMinimizer = struct {
    ptr: *c.gsl_multimin_fdfminimizer,
    n: usize,

    /// Allocate an `n`-dimensional gradient minimizer using algorithm `method`.
    pub fn init(method: GradientMethod, n: usize) Error!GradientMinimizer {
        gsl.ensureHandler();
        const p = c.gsl_multimin_fdfminimizer_alloc(method.typePtr(), n) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .n = n };
    }

    pub fn deinit(self: *GradientMinimizer) void {
        c.gsl_multimin_fdfminimizer_free(self.ptr);
    }

    /// Provide the objective+gradient `func`, an initial point `x0` (length `n`),
    /// the size of the first trial step (`step_size`), and the line-search
    /// accuracy `tol` (roughly `|g·g'|/|g||g'|`; ~0.1 is typical, smaller is more
    /// accurate but costlier). May be called again to restart on a new point.
    pub fn set(self: *GradientMinimizer, func: *FunctionFdf, x0: []const f64, step_size: f64, tol: f64) Error!void {
        if (x0.len != self.n) return Error.Invalid;
        gsl.ensureHandler();
        var xv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(x0));
        try check(c.gsl_multimin_fdfminimizer_set(self.ptr, &func.raw, &xv, step_size, tol));
    }

    /// Perform one descent iteration.
    pub fn iterate(self: *GradientMinimizer) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_multimin_fdfminimizer_iterate(self.ptr));
    }

    /// Copy the current best estimate of the minimizer's *location* (length `n`)
    /// into `out`.
    pub fn xInto(self: *const GradientMinimizer, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multimin_fdfminimizer_x(self.ptr));
    }

    /// Copy the current gradient `∇f` (length `n`) into `out`.
    pub fn gradientInto(self: *const GradientMinimizer, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multimin_fdfminimizer_gradient(self.ptr));
    }

    /// The objective value at the current estimate.
    pub fn minimum(self: *const GradientMinimizer) f64 {
        return c.gsl_multimin_fdfminimizer_minimum(self.ptr);
    }

    /// Reset the minimizer to use the current point as a new starting point,
    /// discarding accumulated conjugacy/curvature information. Useful when
    /// progress has stalled.
    pub fn restart(self: *GradientMinimizer) Error!void {
        try check(c.gsl_multimin_fdfminimizer_restart(self.ptr));
    }

    /// The standardized name of the algorithm (e.g. `"vector_bfgs2"`).
    pub fn name(self: *const GradientMinimizer) [:0]const u8 {
        return std.mem.span(c.gsl_multimin_fdfminimizer_name(self.ptr));
    }

    /// Test whether the current gradient norm is below `epsabs` (via
    /// `gsl_multimin_test_gradient`). Returns `true` once converged, `false` to
    /// keep iterating, and an `Error` for a bad argument.
    pub fn testGradient(self: *const GradientMinimizer, epsabs: f64) Error!bool {
        const g = c.gsl_multimin_fdfminimizer_gradient(self.ptr);
        return testStatus(c.gsl_multimin_test_gradient(g, epsabs));
    }
};

/// Map a GSL convergence-test status to a bool: `GSL_SUCCESS`->true (converged),
/// `GSL_CONTINUE`->false (keep going), otherwise an `Error`.
fn testStatus(status: c_int) Error!bool {
    return switch (status) {
        c.GSL_SUCCESS => true,
        c.GSL_CONTINUE => false,
        else => {
            try check(status);
            unreachable; // a non-success/continue status always maps to an error
        },
    };
}

/// Copy a GSL vector into `out`, honoring its stride.
fn copyVec(out: []f64, v: [*c]const c.gsl_vector) void {
    var i: usize = 0;
    while (i < out.len) : (i += 1) out[i] = v.*.data[i * v.*.stride];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// f(x, y) = (x - a)² + (y - b)²; a convex paraboloid whose unique minimum is at
// (a, b) with value 0. Its gradient is (2(x - a), 2(y - b)).
const Paraboloid = struct {
    a: f64,
    b: f64,
    pub fn eval(self: *const Paraboloid, x: []const f64) f64 {
        const dx = x[0] - self.a;
        const dy = x[1] - self.b;
        return dx * dx + dy * dy;
    }
    pub fn gradient(self: *const Paraboloid, x: []const f64, g: []f64) void {
        g[0] = 2.0 * (x[0] - self.a);
        g[1] = 2.0 * (x[1] - self.b);
    }
};

test "multimin: every derivative-free method finds the paraboloid minimum" {
    var p = Paraboloid{ .a = 1.0, .b = 2.0 };
    inline for (comptime std.enums.values(Method)) |method| {
        var func = Function.initCtx(&p, 2);
        var m = try Minimizer.init(method, 2);
        defer m.deinit();
        try m.set(&func, &.{ -3.0, 5.0 }, &.{ 1.0, 1.0 });

        var iters: usize = 0;
        while (iters < 500) : (iters += 1) {
            try m.iterate();
            if (try m.testSize(1e-4)) break;
        }
        try testing.expect(iters < 500);
        var xmin: [2]f64 = undefined;
        m.xInto(&xmin);
        try testing.expectApproxEqAbs(@as(f64, 1.0), xmin[0], 1e-2);
        try testing.expectApproxEqAbs(@as(f64, 2.0), xmin[1], 1e-2);
        try testing.expectApproxEqAbs(@as(f64, 0.0), m.minimum(), 1e-3);
    }
}

test "multimin: every gradient method finds the paraboloid minimum" {
    var p = Paraboloid{ .a = -2.0, .b = 3.5 };
    inline for (comptime std.enums.values(GradientMethod)) |method| {
        var func = FunctionFdf.initCtx(&p, 2);
        var m = try GradientMinimizer.init(method, 2);
        defer m.deinit();
        try m.set(&func, &.{ 4.0, 4.0 }, 0.1, 1e-4);

        var iters: usize = 0;
        while (iters < 500) : (iters += 1) {
            try m.iterate();
            if (try m.testGradient(1e-6)) break;
        }
        try testing.expect(iters < 500);
        var xmin: [2]f64 = undefined;
        m.xInto(&xmin);
        try testing.expectApproxEqAbs(@as(f64, -2.0), xmin[0], 1e-4);
        try testing.expectApproxEqAbs(@as(f64, 3.5), xmin[1], 1e-4);
        try testing.expectApproxEqAbs(@as(f64, 0.0), m.minimum(), 1e-6);

        // The gradient at the minimum is (near) zero.
        var grad: [2]f64 = undefined;
        m.gradientInto(&grad);
        try testing.expectApproxEqAbs(@as(f64, 0.0), grad[0], 1e-3);
        try testing.expectApproxEqAbs(@as(f64, 0.0), grad[1], 1e-3);
    }
}

test "multimin: fused evalGradient is used when the context declares it" {
    // A context with a fused value+gradient method; a flag records that GSL's
    // combined fdf callback routed through it.
    const Fused = struct {
        used_fused: bool = false,
        pub fn eval(_: *const @This(), x: []const f64) f64 {
            return x[0] * x[0] + x[1] * x[1];
        }
        pub fn gradient(_: *const @This(), x: []const f64, g: []f64) void {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
        }
        pub fn evalGradient(self: *@This(), x: []const f64, f: *f64, g: []f64) void {
            self.used_fused = true;
            f.* = x[0] * x[0] + x[1] * x[1];
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
        }
    };
    var ctx = Fused{};
    var func = FunctionFdf.initCtx(&ctx, 2);
    var m = try GradientMinimizer.init(.vector_bfgs2, 2);
    defer m.deinit();
    try m.set(&func, &.{ 1.0, 1.0 }, 0.1, 1e-4);
    for (0..200) |_| {
        try m.iterate();
        if (try m.testGradient(1e-6)) break;
    }
    try testing.expect(ctx.used_fused);
}

test "multimin: restart and name are available" {
    var p = Paraboloid{ .a = 0.0, .b = 0.0 };
    var func = FunctionFdf.initCtx(&p, 2);
    var m = try GradientMinimizer.init(.conjugate_fr, 2);
    defer m.deinit();
    try m.set(&func, &.{ 3.0, -3.0 }, 0.1, 1e-4);
    try m.iterate();
    try m.restart(); // re-seed from the current point; must not error
    try testing.expectEqualStrings("conjugate_fr", m.name());
}

test "multimin: length validation" {
    var p = Paraboloid{ .a = 1.0, .b = 2.0 };
    var func = Function.initCtx(&p, 2);
    var m = try Minimizer.init(.nmsimplex2, 2);
    defer m.deinit();
    try testing.expectError(Error.Invalid, m.set(&func, &.{1.0}, &.{ 1.0, 1.0 }));
    try testing.expectError(Error.Invalid, m.set(&func, &.{ 1.0, 2.0 }, &.{1.0}));
}
