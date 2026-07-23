//! Idiomatic Zig bindings for the GNU Scientific Library's multidimensional
//! root-finding module (`gsl_multiroots`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the multidimensional
//! root-finding chapter: solving a system of `n` nonlinear equations `F(x) = 0`
//! for `x ∈ R^n`. It is the vector-valued analogue of `gsl.roots`, and a close
//! structural twin of `gsl.multimin`: a stateful solver you `init`, `set`, and
//! `iterate`. It reuses `gsl.zig`'s process-global error-handler switch but keeps
//! the `gsl_multiroots` C API behind its own `c`. It is reached as
//! `gsl.multiroots`.
//!
//! ## Two solver families
//!
//!   - `Solver` — *derivative-free* (`gsl_multiroot_fsolver`). You supply only
//!     the system `F`; the algorithm approximates the Jacobian internally.
//!   - `DerivSolver` — *Jacobian-based* (`gsl_multiroot_fdfsolver`). You supply
//!     `F` and its Jacobian `J` (`J[i][j] = ∂f_i/∂x_j`).
//!
//! Both converge either on a small residual (`testResidual` — `|f| < epsabs`) or
//! a small step (`testDelta` — `|dx| < epsabs + epsrel·|x|`).
//!
//! ## Shape of the surface (derivative-free)
//!
//! ```zig
//! // Solve the 2x2 system  x₀² + x₁² = 1,  x₀ = x₁  (a point on the unit
//! // circle on the diagonal, e.g. (√½, √½)).
//! const System = struct {
//!     pub fn equations(_: *const @This(), x: []const f64, f: gsl.StridedMut(f64)) void {
//!         f.set(0, x[0] * x[0] + x[1] * x[1] - 1.0);
//!         f.set(1, x[0] - x[1]);
//!     }
//! };
//! var s = System{};
//! var sys = gsl.multiroots.System.initCtx(&s, 2);
//! var solver = try gsl.multiroots.Solver.init(.hybrids, 2);
//! defer solver.deinit();
//! try solver.set(&sys, &.{ 0.5, 0.5 });
//! while (true) {
//!     try solver.iterate();
//!     if (try solver.testResidual(1e-8)) break;
//! }
//! var root: [2]f64 = undefined;
//! solver.rootInto(&root);
//! ```
//!
//! The Jacobian family is identical except the context also declares
//! `pub fn jacobian(self, x, J: gsl.MatrixMut(f64))`.
//!
//! ## Lifetime
//!
//! Like `gsl.multimin`, both solvers retain a *pointer* to the callback bundle
//! across iterations. The bundle lives in a caller-owned `System`/`DerivSystem`;
//! keep it — and the context it wraps — alive and unmoved between `set` and the
//! final `iterate`, and do not move the solver across that span.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_multiroots.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the root-finding routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_multiroot_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument (e.g. a length mismatch between the
    /// initial point and the problem dimension, or a nonsensical tolerance).
    Invalid,
    /// `GSL_EBADFUNC` — the system or Jacobian produced a singular (Inf/NaN)
    /// value.
    BadFunction,
    /// `GSL_ENOPROG` — the iteration cannot improve the current estimate (the
    /// step became too small to make progress).
    NoProgress,
    /// `GSL_ENOPROGJ` — repeated Jacobian evaluations are not improving the
    /// solution (Jacobian-based solvers only).
    NoProgressJacobian,
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
        c.GSL_ENOPROGJ => Error.NoProgressJacobian,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// The derivative-free root-finding algorithm used by `Solver`, mirroring GSL's
/// `gsl_multiroot_fsolver_*` types.
pub const Method = enum {
    /// Modified Powell hybrid with an internal scaling of the variables — the
    /// usual default; robust and efficient.
    hybrids,
    /// Powell hybrid without internal scaling.
    hybrid,
    /// Discrete Newton (finite-difference Jacobian); fast near a root but not
    /// globally convergent.
    dnewton,
    /// Broyden's quasi-Newton method; cheapest per step, least robust.
    broyden,

    fn typePtr(self: Method) *const c.gsl_multiroot_fsolver_type {
        return switch (self) {
            .hybrids => c.gsl_multiroot_fsolver_hybrids,
            .hybrid => c.gsl_multiroot_fsolver_hybrid,
            .dnewton => c.gsl_multiroot_fsolver_dnewton,
            .broyden => c.gsl_multiroot_fsolver_broyden,
        };
    }
};

/// The Jacobian-based root-finding algorithm used by `DerivSolver`, mirroring
/// GSL's `gsl_multiroot_fdfsolver_*` types.
pub const DerivMethod = enum {
    /// Modified Powell hybrid with internal scaling, using the analytic Jacobian
    /// — the usual default.
    hybridsj,
    /// Powell hybrid without internal scaling, using the analytic Jacobian.
    hybridj,
    /// Newton's method.
    newton,
    /// Globally convergent (line-searched) Newton's method.
    gnewton,

    fn typePtr(self: DerivMethod) *const c.gsl_multiroot_fdfsolver_type {
        return switch (self) {
            .hybridsj => c.gsl_multiroot_fdfsolver_hybridsj,
            .hybridj => c.gsl_multiroot_fdfsolver_hybridj,
            .newton => c.gsl_multiroot_fdfsolver_newton,
            .gnewton => c.gsl_multiroot_fdfsolver_gnewton,
        };
    }
};

/// A `gsl_multiroot_function`-shaped system callback bundle for the
/// derivative-free `Solver`. Construct from a context pointer with `initCtx`; the
/// context must declare `pub fn equations(self, x: []const f64, f:
/// gsl.StridedMut(f64))`.
pub const System = struct {
    raw: c.gsl_multiroot_function,

    /// Wrap a context declaring `pub fn equations(self, x, f)` for an `n`-equation
    /// system.
    pub fn initCtx(ctx: anytype, n: usize) System {
        return .{ .raw = callback.multirootF(c.gsl_multiroot_function, c.gsl_vector, n, ctx) };
    }
};

/// A `gsl_multiroot_function_fdf`-shaped system+Jacobian callback bundle for the
/// `DerivSolver`. Construct from a context pointer with `initCtx`; the context
/// must declare `pub fn equations(self, x, f: gsl.StridedMut(f64))` and `pub fn
/// jacobian(self, x, J: gsl.MatrixMut(f64))`.
pub const DerivSystem = struct {
    raw: c.gsl_multiroot_function_fdf,

    /// Wrap a context declaring `equations` + `jacobian` for an `n`-equation
    /// system.
    pub fn initCtx(ctx: anytype, n: usize) DerivSystem {
        return .{ .raw = callback.multirootFdf(c.gsl_multiroot_function_fdf, c.gsl_vector, c.gsl_matrix, n, ctx) };
    }
};

/// A stateful derivative-free multidimensional root finder. Owns its GSL
/// allocation (`deinit` frees it). Once `set`, GSL holds a pointer to the
/// caller's `System`, so keep it alive and do not move this solver between `set`
/// and the last `iterate`.
pub const Solver = struct {
    ptr: *c.gsl_multiroot_fsolver,
    n: usize,

    /// Allocate an `n`-dimensional solver using algorithm `method`.
    pub fn init(method: Method, n: usize) Error!Solver {
        gsl.ensureHandler();
        const p = c.gsl_multiroot_fsolver_alloc(method.typePtr(), n) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .n = n };
    }

    pub fn deinit(self: *Solver) void {
        c.gsl_multiroot_fsolver_free(self.ptr);
    }

    /// Provide the system `sys` and an initial guess `x0` (length `n`). May be
    /// called again to restart the solver on a new starting point.
    pub fn set(self: *Solver, sys: *System, x0: []const f64) Error!void {
        if (x0.len != self.n) return Error.Invalid;
        gsl.ensureHandler();
        var xv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(x0));
        try check(c.gsl_multiroot_fsolver_set(self.ptr, &sys.raw, &xv));
    }

    /// Perform one iteration toward the root.
    pub fn iterate(self: *Solver) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_multiroot_fsolver_iterate(self.ptr));
    }

    /// Copy the current estimate of the root (length `n`) into `out`.
    pub fn rootInto(self: *const Solver, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multiroot_fsolver_root(self.ptr));
    }

    /// Copy the current residual `F(x)` (length `n`) into `out`.
    pub fn fInto(self: *const Solver, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multiroot_fsolver_f(self.ptr));
    }

    /// Copy the last step `dx` (length `n`) into `out`.
    pub fn dxInto(self: *const Solver, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multiroot_fsolver_dx(self.ptr));
    }

    /// The standardized name of the algorithm (e.g. `"hybrids"`).
    pub fn name(self: *const Solver) [:0]const u8 {
        return std.mem.span(c.gsl_multiroot_fsolver_name(self.ptr));
    }

    /// Test whether the current residual is below `epsabs` (via
    /// `gsl_multiroot_test_residual`). Returns `true` once converged, `false` to
    /// keep iterating, and an `Error` for a bad argument.
    pub fn testResidual(self: *const Solver, epsabs: f64) Error!bool {
        return testStatus(c.gsl_multiroot_test_residual(c.gsl_multiroot_fsolver_f(self.ptr), epsabs));
    }

    /// Test whether the last step `dx` is small relative to `x` (via
    /// `gsl_multiroot_test_delta`, `|dx| < epsabs + epsrel·|x|`). Returns `true`
    /// once converged, `false` to keep iterating, and an `Error` for a bad
    /// argument.
    pub fn testDelta(self: *const Solver, epsabs: f64, epsrel: f64) Error!bool {
        return testStatus(c.gsl_multiroot_test_delta(
            c.gsl_multiroot_fsolver_dx(self.ptr),
            c.gsl_multiroot_fsolver_root(self.ptr),
            epsabs,
            epsrel,
        ));
    }
};

/// A stateful Jacobian-based multidimensional root finder. Owns its GSL
/// allocation (`deinit` frees it). Once `set`, GSL holds a pointer to the
/// caller's `DerivSystem`, so keep it alive and do not move this solver between
/// `set` and the last `iterate`.
pub const DerivSolver = struct {
    ptr: *c.gsl_multiroot_fdfsolver,
    n: usize,

    /// Allocate an `n`-dimensional Jacobian-based solver using algorithm `method`.
    pub fn init(method: DerivMethod, n: usize) Error!DerivSolver {
        gsl.ensureHandler();
        const p = c.gsl_multiroot_fdfsolver_alloc(method.typePtr(), n) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .n = n };
    }

    pub fn deinit(self: *DerivSolver) void {
        c.gsl_multiroot_fdfsolver_free(self.ptr);
    }

    /// Provide the system+Jacobian `sys` and an initial guess `x0` (length `n`).
    /// May be called again to restart the solver on a new starting point.
    pub fn set(self: *DerivSolver, sys: *DerivSystem, x0: []const f64) Error!void {
        if (x0.len != self.n) return Error.Invalid;
        gsl.ensureHandler();
        var xv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(x0));
        try check(c.gsl_multiroot_fdfsolver_set(self.ptr, &sys.raw, &xv));
    }

    /// Perform one iteration toward the root.
    pub fn iterate(self: *DerivSolver) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_multiroot_fdfsolver_iterate(self.ptr));
    }

    /// Copy the current estimate of the root (length `n`) into `out`.
    pub fn rootInto(self: *const DerivSolver, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multiroot_fdfsolver_root(self.ptr));
    }

    /// Copy the current residual `F(x)` (length `n`) into `out`.
    pub fn fInto(self: *const DerivSolver, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multiroot_fdfsolver_f(self.ptr));
    }

    /// Copy the last step `dx` (length `n`) into `out`.
    pub fn dxInto(self: *const DerivSolver, out: []f64) void {
        std.debug.assert(out.len == self.n);
        copyVec(out, c.gsl_multiroot_fdfsolver_dx(self.ptr));
    }

    /// The standardized name of the algorithm (e.g. `"hybridsj"`).
    pub fn name(self: *const DerivSolver) [:0]const u8 {
        return std.mem.span(c.gsl_multiroot_fdfsolver_name(self.ptr));
    }

    /// Test whether the current residual is below `epsabs`. See `Solver.testResidual`.
    pub fn testResidual(self: *const DerivSolver, epsabs: f64) Error!bool {
        return testStatus(c.gsl_multiroot_test_residual(c.gsl_multiroot_fdfsolver_f(self.ptr), epsabs));
    }

    /// Test whether the last step `dx` is small relative to `x`. See `Solver.testDelta`.
    pub fn testDelta(self: *const DerivSolver, epsabs: f64, epsrel: f64) Error!bool {
        return testStatus(c.gsl_multiroot_test_delta(
            c.gsl_multiroot_fdfsolver_dx(self.ptr),
            c.gsl_multiroot_fdfsolver_root(self.ptr),
            epsabs,
            epsrel,
        ));
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

// The 2x2 system
//   f₀ = x₀² + x₁² − 1   (on the unit circle)
//   f₁ = x₀ − x₁         (on the diagonal)
// whose root in the first quadrant is (√½, √½) ≈ (0.7071, 0.7071).
const Circle = struct {
    pub fn equations(_: *const Circle, x: []const f64, f: gsl.StridedMut(f64)) void {
        f.set(0, x[0] * x[0] + x[1] * x[1] - 1.0);
        f.set(1, x[0] - x[1]);
    }
    pub fn jacobian(_: *const Circle, x: []const f64, j: gsl.MatrixMut(f64)) void {
        j.set(0, 0, 2.0 * x[0]);
        j.set(0, 1, 2.0 * x[1]);
        j.set(1, 0, 1.0);
        j.set(1, 1, -1.0);
    }
};

const root_half = std.math.sqrt1_2; // √½ ≈ 0.70710678

test "multiroots: every derivative-free method finds the root" {
    var ctx = Circle{};
    inline for (comptime std.enums.values(Method)) |method| {
        var sys = System.initCtx(&ctx, 2);
        var solver = try Solver.init(method, 2);
        defer solver.deinit();
        try solver.set(&sys, &.{ 0.5, 0.5 });

        var iters: usize = 0;
        while (iters < 200) : (iters += 1) {
            try solver.iterate();
            if (try solver.testResidual(1e-10)) break;
        }
        try testing.expect(iters < 200);
        var root: [2]f64 = undefined;
        solver.rootInto(&root);
        try testing.expectApproxEqAbs(root_half, root[0], 1e-7);
        try testing.expectApproxEqAbs(root_half, root[1], 1e-7);
    }
}

test "multiroots: every Jacobian method finds the root" {
    var ctx = Circle{};
    inline for (comptime std.enums.values(DerivMethod)) |method| {
        var sys = DerivSystem.initCtx(&ctx, 2);
        var solver = try DerivSolver.init(method, 2);
        defer solver.deinit();
        try solver.set(&sys, &.{ 0.5, 0.5 });

        var iters: usize = 0;
        while (iters < 200) : (iters += 1) {
            try solver.iterate();
            if (try solver.testResidual(1e-10)) break;
        }
        try testing.expect(iters < 200);
        var root: [2]f64 = undefined;
        solver.rootInto(&root);
        try testing.expectApproxEqAbs(root_half, root[0], 1e-8);
        try testing.expectApproxEqAbs(root_half, root[1], 1e-8);

        // The residual there is (near) zero.
        var f: [2]f64 = undefined;
        solver.fInto(&f);
        try testing.expectApproxEqAbs(@as(f64, 0.0), f[0], 1e-8);
        try testing.expectApproxEqAbs(@as(f64, 0.0), f[1], 1e-8);
    }
}

test "multiroots: testDelta converges and dx/name are available" {
    var ctx = Circle{};
    var sys = System.initCtx(&ctx, 2);
    var solver = try Solver.init(.hybrids, 2);
    defer solver.deinit();
    try solver.set(&sys, &.{ 0.5, 0.5 });

    var iters: usize = 0;
    while (iters < 200) : (iters += 1) {
        try solver.iterate();
        if (try solver.testDelta(1e-10, 0.0)) break;
    }
    try testing.expect(iters < 200);

    var dx: [2]f64 = undefined;
    solver.dxInto(&dx); // the converged step is tiny
    try testing.expect(@abs(dx[0]) < 1e-6 and @abs(dx[1]) < 1e-6);
    try testing.expectEqualStrings("hybrids", solver.name());
}

test "multiroots: length validation" {
    var ctx = Circle{};
    var sys = System.initCtx(&ctx, 2);
    var solver = try Solver.init(.hybrids, 2);
    defer solver.deinit();
    try testing.expectError(Error.Invalid, solver.set(&sys, &.{1.0}));
}
