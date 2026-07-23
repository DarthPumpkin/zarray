//! Idiomatic Zig bindings for the GNU Scientific Library's nonlinear
//! least-squares module (`gsl_multifit_nlinear`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with trust-region
//! nonlinear least-squares fitting: given a residual vector `f(x)` of `n`
//! functions in `p` parameters, it finds `x` minimizing `||f(x)||^2`. It reuses
//! `gsl.zig`'s process-global error handler and the `gsl_callback.zig` bridge,
//! and is reached as `gsl.nlinear`.
//!
//! ## Shape of the surface
//!
//!   1. Build a `Problem` from a context pointer:
//!      - `Problem.initCtx(&ctx, n, p)` for residual-only models (GSL computes
//!        the Jacobian by finite differences). `ctx` declares
//!        `pub fn residual(self, x: []const f64, f: gsl.StridedMut(f64)) void|c_int`.
//!      - `Problem.initCtxWithJacobian(&ctx, n, p)` when an analytic Jacobian is
//!        available; `ctx` additionally declares
//!        `pub fn jacobian(self, x: []const f64, J: gsl.MatrixMut(f64)) void|c_int`
//!        (`J` is the `n`x`p` Jacobian; fill with `J.set(i, j, ...)`).
//!   2. Allocate a `Workspace` for a `Type` (currently `.trust`) and tunable
//!      `Parameters` (trust-region subproblem, scaling, linear solver, finite
//!      difference type).
//!   3. Seed with `initSolution(x0)`, then either run the whole optimization
//!      with `driver(max_iter, .{})`, or drive it manually with
//!      `iterate` + `testConvergence`.
//!   4. Read results: `solutionInto`, `residualInto`, `niter`, `rcond`, and
//!      `covariance` (parameter covariance from the final Jacobian).
//!
//! ```zig
//! // Fit y = a * exp(b * t) to data by minimizing residuals r_i = model_i - y_i.
//! const Model = struct {
//!     t: []const f64,
//!     y: []const f64,
//!     pub fn residual(self: *const @This(), x: []const f64, r: gsl.StridedMut(f64)) void {
//!         for (self.t, self.y, 0..) |ti, yi, i| r.set(i, x[0] * @exp(x[1] * ti) - yi);
//!     }
//! };
//!
//! var model = Model{ .t = &t, .y = &y };
//! var prob = gsl.nlinear.Problem.initCtx(&model, t.len, 2);
//! var ws = try gsl.nlinear.Workspace.init(&prob, .trust, .{});
//! defer ws.deinit();
//!
//! try ws.initSolution(&.{ 1.0, 0.0 });
//! _ = try ws.driver(100, .{});
//!
//! var params: [2]f64 = undefined;
//! try ws.solutionInto(&params);
//! ```
//!
//! ## The residual is presented as a strided view
//!
//! The residual output `f` is a `gsl.StridedMut(f64)`, not a plain slice,
//! because GSL's finite-difference Jacobian evaluates the residual directly into
//! a strided *column* of the Jacobian matrix. Write elements with `f.set(i, v)`
//! and read its length via `f.len`. The parameter vector `x` is always
//! contiguous, so it stays a `[]const f64`.
//!
//! ## Lifetime
//!
//! `Workspace.init` hands GSL a pointer to the caller-owned `Problem` (GSL keeps
//! it for the workspace's lifetime), so keep the `Problem` alive and unmoved
//! until `deinit`.
//!
//! ## Omissions
//!
//!   - Geodesic acceleration (`fvv`) is not wired; the `df`/finite-difference
//!     paths cover the common case.
//!   - The lower-level per-iteration `avratio`/scale/solver internals beyond the
//!     tunables in `Parameters` are reachable through the raw `c` API.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_multifit_nlinear.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the nonlinear least-squares routines. The raw `c_int`
/// status is always available from the underlying `c.gsl_multifit_nlinear_*`
/// symbol if you need the exact code.
pub const Error = error{
    /// `GSL_EINVAL` — invalid argument (e.g. `n < p`, or a zero-sized problem).
    Invalid,
    /// A supplied slice length does not match the problem's `n` or `p`.
    BadLength,
    /// `GSL_EBADFUNC` — a callback returned a bad/unsupported function status.
    BadFunction,
    /// `GSL_ENOPROG` — the iteration is not making progress toward a solution.
    NoProgress,
    /// `GSL_EFAILED` / `GSL_FAILURE` — generic internal failure.
    Failed,
    /// `GSL_EMAXITER` — the configured maximum number of iterations was reached
    /// before the convergence test passed.
    MaxIterations,
    /// `GSL_ENOMEM` — allocation failed.
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
        c.GSL_EFAILED, c.GSL_FAILURE => Error.Failed,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// Top-level algorithm family (`gsl_multifit_nlinear_type`). GSL currently ships
/// exactly one: the trust-region method.
pub const Type = enum {
    trust,

    fn typePtr(self: Type) *const c.gsl_multifit_nlinear_type {
        return switch (self) {
            .trust => c.gsl_multifit_nlinear_trust,
        };
    }
};

/// Trust-region subproblem strategy (`gsl_multifit_nlinear_trs`).
pub const Trs = enum {
    /// Levenberg-Marquardt.
    lm,
    /// Levenberg-Marquardt with geodesic acceleration.
    lmaccel,
    /// Powell's dogleg.
    dogleg,
    /// Double dogleg.
    ddogleg,
    /// Two-dimensional subspace.
    subspace2d,

    fn ptr(self: Trs) *const c.gsl_multifit_nlinear_trs {
        return switch (self) {
            .lm => c.gsl_multifit_nlinear_trs_lm,
            .lmaccel => c.gsl_multifit_nlinear_trs_lmaccel,
            .dogleg => c.gsl_multifit_nlinear_trs_dogleg,
            .ddogleg => c.gsl_multifit_nlinear_trs_ddogleg,
            .subspace2d => c.gsl_multifit_nlinear_trs_subspace2D,
        };
    }
};

/// Scaling strategy for the trust-region (`gsl_multifit_nlinear_scale`).
pub const Scale = enum {
    /// Levenberg's original (identity) scaling.
    levenberg,
    /// Marquardt's scaling.
    marquardt,
    /// More's scaling (the GSL default; good general-purpose choice).
    more,

    fn ptr(self: Scale) *const c.gsl_multifit_nlinear_scale {
        return switch (self) {
            .levenberg => c.gsl_multifit_nlinear_scale_levenberg,
            .marquardt => c.gsl_multifit_nlinear_scale_marquardt,
            .more => c.gsl_multifit_nlinear_scale_more,
        };
    }
};

/// Linear least-squares solver used at each iteration
/// (`gsl_multifit_nlinear_solver`).
pub const Solver = enum {
    /// Cholesky factorization of the normal equations (fastest, least robust).
    cholesky,
    /// Modified Cholesky (handles near-singular normal equations).
    mcholesky,
    /// QR factorization of the Jacobian (the GSL default; robust).
    qr,
    /// SVD of the Jacobian (most robust, slowest).
    svd,

    fn ptr(self: Solver) *const c.gsl_multifit_nlinear_solver {
        return switch (self) {
            .cholesky => c.gsl_multifit_nlinear_solver_cholesky,
            .mcholesky => c.gsl_multifit_nlinear_solver_mcholesky,
            .qr => c.gsl_multifit_nlinear_solver_qr,
            .svd => c.gsl_multifit_nlinear_solver_svd,
        };
    }
};

/// Finite-difference scheme used when no analytic Jacobian is supplied.
pub const FdType = enum {
    /// Forward difference (one extra residual evaluation per column).
    forward,
    /// Central difference (two evaluations per column; more accurate).
    central,

    fn toC(self: FdType) c.gsl_multifit_nlinear_fdtype {
        return switch (self) {
            .forward => c.GSL_MULTIFIT_NLINEAR_FWDIFF,
            .central => c.GSL_MULTIFIT_NLINEAR_CTRDIFF,
        };
    }
};

/// Tunable solver parameters. Fields default to GSL's recommended choices
/// (`gsl_multifit_nlinear_default_parameters`); override only what you need.
pub const Parameters = struct {
    trs: Trs = .lm,
    scale: Scale = .more,
    solver: Solver = .qr,
    fdtype: FdType = .forward,

    fn toC(self: Parameters) c.gsl_multifit_nlinear_parameters {
        var p = c.gsl_multifit_nlinear_default_parameters();
        p.trs = self.trs.ptr();
        p.scale = self.scale.ptr();
        p.solver = self.solver.ptr();
        p.fdtype = self.fdtype.toC();
        return p;
    }
};

/// Reason the convergence test was satisfied (`info` out-parameter of
/// `gsl_multifit_nlinear_test`).
pub const Info = enum(c_int) {
    /// Converged because the step size `|dx|` became small relative to `|x|`.
    small_step = 1,
    /// Converged because the gradient `|g|` became small.
    small_gradient = 2,
    _,
};

/// Convergence tolerances for `driver`/`testConvergence`.
pub const Convergence = struct {
    /// Step-size tolerance (`|dx| <= xtol * (|x| + xtol)`).
    xtol: f64 = 1e-8,
    /// Gradient tolerance (`|g|_inf <= gtol`).
    gtol: f64 = 1e-8,
    /// Residual tolerance (relative decrease in `||f||`); 0 disables the test.
    ftol: f64 = 0.0,
};

/// Outcome of a full `driver` run.
pub const DriverResult = struct {
    /// Which criterion triggered convergence.
    converged_by: Info,
    /// Number of iterations performed.
    iterations: usize,
};

/// The residual (+ optional Jacobian) callback bundle
/// (`gsl_multifit_nlinear_fdf`). Construct from a context pointer with either
/// `initCtx` (residual only; finite-difference Jacobian) or
/// `initCtxWithJacobian` (analytic Jacobian). See the file header for the
/// required context methods.
pub const Problem = struct {
    fdf: c.gsl_multifit_nlinear_fdf,

    /// Residual-only model (`n` functions, `p` parameters); GSL approximates the
    /// Jacobian by finite differences.
    pub fn initCtx(ctx: anytype, num_residuals: usize, num_params: usize) Problem {
        return .{ .fdf = callback.multifitFdf(c.gsl_multifit_nlinear_fdf, c.gsl_vector, num_residuals, num_params, ctx) };
    }

    /// Model with an analytic Jacobian (`n` functions, `p` parameters).
    pub fn initCtxWithJacobian(ctx: anytype, num_residuals: usize, num_params: usize) Problem {
        return .{ .fdf = callback.multifitFdfWithJacobian(c.gsl_multifit_nlinear_fdf, c.gsl_vector, c.gsl_matrix, num_residuals, num_params, ctx) };
    }

    /// Number of residual functions (`n`).
    pub fn n(self: *const Problem) usize {
        return self.fdf.n;
    }
    /// Number of parameters (`p`).
    pub fn p(self: *const Problem) usize {
        return self.fdf.p;
    }
};

/// High-level nonlinear least-squares solver
/// (`gsl_multifit_nlinear_workspace`). Owns its GSL allocation; call `deinit` to
/// free.
///
/// Lifetime note: this stores a pointer to the caller-owned `Problem`; keep that
/// `Problem` alive and unmoved for the life of the workspace.
pub const Workspace = struct {
    ptr: *c.gsl_multifit_nlinear_workspace,
    problem: *Problem,

    /// Allocate a workspace for `problem` using algorithm `ty` and `params`.
    pub fn init(problem: *Problem, ty: Type, params: Parameters) Error!Workspace {
        if (problem.fdf.n == 0 or problem.fdf.p == 0 or problem.fdf.n < problem.fdf.p) return Error.Invalid;
        gsl.ensureHandler();
        var cp = params.toC();
        const w = c.gsl_multifit_nlinear_alloc(ty.typePtr(), &cp, problem.fdf.n, problem.fdf.p) orelse return Error.OutOfMemory;
        return .{ .ptr = w, .problem = problem };
    }

    pub fn deinit(self: *Workspace) void {
        c.gsl_multifit_nlinear_free(self.ptr);
    }

    /// Seed the solver with an initial parameter guess `x0` (length `p`).
    pub fn initSolution(self: *Workspace, x0: []const f64) Error!void {
        if (x0.len != self.problem.fdf.p) return Error.BadLength;
        gsl.ensureHandler();
        var xv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(x0));
        try check(c.gsl_multifit_nlinear_init(&xv, &self.problem.fdf, self.ptr));
    }

    /// Seed the solver with an initial guess `x0` and per-residual `weights`
    /// (length `n`), minimizing the weighted sum of squares.
    pub fn initSolutionWeighted(self: *Workspace, x0: []const f64, weights: []const f64) Error!void {
        if (x0.len != self.problem.fdf.p) return Error.BadLength;
        if (weights.len != self.problem.fdf.n) return Error.BadLength;
        gsl.ensureHandler();
        var xv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(x0));
        var wv = gsl.constVectorViewOf(c.gsl_vector, gsl.Strided(f64).fromSlice(weights));
        try check(c.gsl_multifit_nlinear_winit(&xv, &wv, &self.problem.fdf, self.ptr));
    }

    /// Run the optimization to convergence (or `max_iter` iterations). Returns
    /// which criterion converged and the iteration count; yields
    /// `error.MaxIterations` if `max_iter` was hit first.
    pub fn driver(self: *Workspace, max_iter: usize, conv: Convergence) Error!DriverResult {
        gsl.ensureHandler();
        var info: c_int = 0;
        try check(c.gsl_multifit_nlinear_driver(
            max_iter,
            conv.xtol,
            conv.gtol,
            conv.ftol,
            null,
            null,
            &info,
            self.ptr,
        ));
        return .{ .converged_by = @enumFromInt(info), .iterations = self.niter() };
    }

    /// Perform a single trust-region iteration.
    pub fn iterate(self: *Workspace) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_multifit_nlinear_iterate(self.ptr));
    }

    /// Test the current state against `conv`. Returns the triggering `Info` if
    /// converged, or `null` to keep iterating.
    pub fn testConvergence(self: *Workspace, conv: Convergence) Error!?Info {
        var info: c_int = 0;
        const st = c.gsl_multifit_nlinear_test(conv.xtol, conv.gtol, conv.ftol, &info, self.ptr);
        if (st == c.GSL_SUCCESS) return @as(Info, @enumFromInt(info));
        if (st == c.GSL_CONTINUE) return null;
        try check(st);
        unreachable;
    }

    /// Number of iterations performed so far.
    pub fn niter(self: *const Workspace) usize {
        return c.gsl_multifit_nlinear_niter(self.ptr);
    }

    /// Name of the selected algorithm (e.g. `"trust-region"`).
    pub fn name(self: *const Workspace) []const u8 {
        return std.mem.span(c.gsl_multifit_nlinear_name(self.ptr));
    }

    /// Name of the active trust-region subproblem method (e.g. `"levenberg-marquardt"`).
    pub fn trsName(self: *const Workspace) []const u8 {
        return std.mem.span(c.gsl_multifit_nlinear_trs_name(self.ptr));
    }

    /// Copy the current parameter estimate (length `p`) into `out`.
    pub fn solutionInto(self: *Workspace, out: []f64) Error!void {
        if (out.len != self.problem.fdf.p) return Error.BadLength;
        copyVec(out, c.gsl_multifit_nlinear_position(self.ptr));
    }

    /// Copy the current residual vector (length `n`) into `out`.
    pub fn residualInto(self: *Workspace, out: []f64) Error!void {
        if (out.len != self.problem.fdf.n) return Error.BadLength;
        copyVec(out, c.gsl_multifit_nlinear_residual(self.ptr));
    }

    /// Reciprocal condition number of the final Jacobian (a rough measure of
    /// how well-determined the fit is).
    pub fn rcond(self: *Workspace) Error!f64 {
        var value: f64 = 0;
        try check(c.gsl_multifit_nlinear_rcond(&value, self.ptr));
        return value;
    }

    /// Compute the `p`x`p` parameter covariance matrix from the final Jacobian,
    /// writing it row-major into `out` (length `p*p`). `epsrel` drops linearly
    /// dependent columns whose relative magnitude is below it.
    pub fn covariance(self: *Workspace, epsrel: f64, out: []f64) Error!void {
        const p = self.problem.fdf.p;
        if (out.len != p * p) return Error.BadLength;
        gsl.ensureHandler();
        const jac = c.gsl_multifit_nlinear_jac(self.ptr);
        var cm = gsl.mutMatrixViewOf(c.gsl_matrix, gsl.MatrixMut(f64).fromSlice(out, p, p));
        try check(c.gsl_multifit_nlinear_covar(jac, epsrel, &cm));
    }
};

/// Copy a (contiguous) GSL vector into `out`, honoring its stride.
fn copyVec(out: []f64, v: [*c]const c.gsl_vector) void {
    var i: usize = 0;
    while (i < out.len) : (i += 1) out[i] = v.*.data[i * v.*.stride];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A straight-line model r_i = a*t_i + b - y_i. Linear, so the least-squares
// solution is exact; it exercises the whole pipeline deterministically.
const Line = struct {
    t: []const f64,
    y: []const f64,
    pub fn residual(self: *const Line, x: []const f64, r: gsl.StridedMut(f64)) void {
        for (self.t, self.y, 0..) |ti, yi, i| r.set(i, x[0] * ti + x[1] - yi);
    }
    pub fn jacobian(self: *const Line, x: []const f64, j: gsl.MatrixMut(f64)) void {
        _ = x;
        for (self.t, 0..) |ti, i| {
            j.set(i, 0, ti); // d r_i / d a
            j.set(i, 1, 1.0); // d r_i / d b
        }
    }
};

test "nlinear: finite-difference driver recovers a linear fit" {
    // Data generated from a=2, b=-1 exactly.
    var t = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f64{ -1.0, 1.0, 3.0, 5.0, 7.0 };
    var model = Line{ .t = &t, .y = &y };

    var prob = Problem.initCtx(&model, t.len, 2);
    var ws = try Workspace.init(&prob, .trust, .{});
    defer ws.deinit();

    try ws.initSolution(&.{ 0.0, 0.0 });
    const res = try ws.driver(100, .{});
    _ = res;

    var params: [2]f64 = undefined;
    try ws.solutionInto(&params);
    try testing.expectApproxEqAbs(@as(f64, 2.0), params[0], 1e-8);
    try testing.expectApproxEqAbs(@as(f64, -1.0), params[1], 1e-8);
}

test "nlinear: analytic Jacobian recovers the same fit" {
    var t = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f64{ -1.0, 1.0, 3.0, 5.0, 7.0 };
    var model = Line{ .t = &t, .y = &y };

    var prob = Problem.initCtxWithJacobian(&model, t.len, 2);
    var ws = try Workspace.init(&prob, .trust, .{ .solver = .qr });
    defer ws.deinit();

    try ws.initSolution(&.{ 10.0, 10.0 });
    _ = try ws.driver(100, .{});

    var params: [2]f64 = undefined;
    try ws.solutionInto(&params);
    try testing.expectApproxEqAbs(@as(f64, 2.0), params[0], 1e-8);
    try testing.expectApproxEqAbs(@as(f64, -1.0), params[1], 1e-8);

    // Residuals are essentially zero for an exact linear fit.
    var resid: [5]f64 = undefined;
    try ws.residualInto(&resid);
    for (resid) |ri| try testing.expectApproxEqAbs(@as(f64, 0.0), ri, 1e-7);
}

test "nlinear: manual iterate/testConvergence loop converges" {
    var t = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f64{ 1.0, 3.0, 5.0, 7.0 }; // a=2, b=-1
    var model = Line{ .t = &t, .y = &y };

    var prob = Problem.initCtxWithJacobian(&model, t.len, 2);
    var ws = try Workspace.init(&prob, .trust, .{});
    defer ws.deinit();

    try ws.initSolution(&.{ 0.0, 0.0 });

    var iter: usize = 0;
    const converged = while (iter < 100) : (iter += 1) {
        try ws.iterate();
        if (try ws.testConvergence(.{})) |_| break true;
    } else false;

    try testing.expect(converged);
    var params: [2]f64 = undefined;
    try ws.solutionInto(&params);
    try testing.expectApproxEqAbs(@as(f64, 2.0), params[0], 1e-8);
    try testing.expectApproxEqAbs(@as(f64, -1.0), params[1], 1e-8);
}

test "nlinear: covariance and rcond are available after a fit" {
    var t = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    var y = [_]f64{ 1.0, 2.0, 3.0, 4.0 }; // a=1, b=1
    var model = Line{ .t = &t, .y = &y };

    var prob = Problem.initCtxWithJacobian(&model, t.len, 2);
    var ws = try Workspace.init(&prob, .trust, .{});
    defer ws.deinit();

    try ws.initSolution(&.{ 0.0, 0.0 });
    _ = try ws.driver(100, .{});

    var cov: [4]f64 = undefined;
    try ws.covariance(0.0, &cov);
    // Covariance is symmetric positive semi-definite.
    try testing.expect(cov[0] >= 0.0 and cov[3] >= 0.0);
    try testing.expectApproxEqAbs(cov[1], cov[2], 1e-12);

    const rc = try ws.rcond();
    try testing.expect(rc >= 0.0);
}

test "nlinear: length and shape validation" {
    var t = [_]f64{ 1.0, 2.0, 3.0 };
    var y = [_]f64{ 1.0, 2.0, 3.0 };
    var model = Line{ .t = &t, .y = &y };

    // n < p is rejected at allocation.
    var bad = Problem.initCtx(&model, 1, 2);
    try testing.expectError(Error.Invalid, Workspace.init(&bad, .trust, .{}));

    var prob = Problem.initCtx(&model, 3, 2);
    var ws = try Workspace.init(&prob, .trust, .{});
    defer ws.deinit();

    try testing.expectError(Error.BadLength, ws.initSolution(&.{0.0}));
    try ws.initSolution(&.{ 0.0, 0.0 });

    var too_small: [1]f64 = undefined;
    try testing.expectError(Error.BadLength, ws.solutionInto(&too_small));
    var wrong_cov: [3]f64 = undefined;
    try testing.expectError(Error.BadLength, ws.covariance(0.0, &wrong_cov));
}
