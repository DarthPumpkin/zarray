//! Idiomatic Zig bindings for the GNU Scientific Library's one-dimensional
//! root-finding module (`gsl_roots`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the one-dim root
//! chapter — the first consumer of the callback bridge's *derivative-bearing*
//! form (`gsl.callback`'s `FunctionFdf`). It reuses `gsl.zig`'s process-global
//! error-handler switch but keeps the `gsl_roots` C API behind its own `c`. It
//! is reached as `gsl.roots`.
//!
//! ## Shape of the surface
//!
//! GSL offers two solver families, both iterative and stateful:
//!
//!   - `Solver` — *bracketing* solvers (`bisection`/`brent`/`falsepos`). You
//!     supply a function and a bracket `[lower, upper]` that straddles a sign
//!     change; each `iterate` shrinks the bracket. Needs only `f` (a `Callback`).
//!   - `PolishSolver` — *derivative* solvers (`newton`/`secant`/`steffenson`).
//!     You supply `f` **and** `f'` (a `CallbackFdf` context with `eval`/`deriv`)
//!     plus a single starting guess; each `iterate` refines the estimate.
//!
//! The usage pattern for both is identical: `init` a solver, `set` it with the
//! function and starting condition, then `iterate` in a loop until a convergence
//! test passes:
//!
//! ```zig
//! var s = try gsl.roots.Solver.init(.brent);
//! defer s.deinit();
//! try s.set(.initFn(f), 0, 2);
//! while (true) {
//!     try s.iterate();
//!     const iv = s.interval();
//!     if (try gsl.roots.testInterval(iv.lower, iv.upper, .{ .abs = 1e-9 })) break;
//! }
//! const root = s.root();
//! ```
//!
//! The convergence helpers `testInterval`/`testDelta`/`testResidual` return
//! `true` when the tolerance is met and `false` to keep iterating (GSL's
//! `GSL_CONTINUE`), surfacing any genuine error otherwise.
//!
//! ## Lifetime
//!
//! GSL's solvers retain a *pointer* to the function across iterations, so each
//! solver stores the `Callback` it was `set` with **by value**. Consequently,
//! do not move (copy to a new location) a solver between `set` and the final
//! `iterate`, and — for context callbacks (`.initCtx`) — keep the context alive
//! for that whole span (the usual bridge rule).
//!
//! ## Omissions
//!
//!   - None of the module's public routines are omitted; all three bracketing
//!     and all three polishing algorithms, plus the three convergence tests,
//!     are wrapped.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_math.h");
    @cInclude("gsl/gsl_roots.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the root-finding routines. The raw `c_int` status is
/// always available from the underlying `c.gsl_root_*` symbol if you need the
/// exact code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument; most often the bracket `[lower,
    /// upper]` given to a `Solver` does not straddle a sign change.
    Invalid,
    /// `GSL_EBADTOL` — a nonsensical tolerance (e.g. negative) in a test.
    BadTolerance,
    /// `GSL_EZERODIV` — a derivative solver hit a zero derivative.
    ZeroDivide,
    /// `GSL_EMAXITER` — the iteration failed to make progress.
    MaxIterations,
    /// `GSL_EBADFUNC` — the function produced a singular (Inf/NaN) value.
    BadFunction,
    /// `GSL_ENOMEM` — solver allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADTOL => Error.BadTolerance,
        c.GSL_EZERODIV => Error.ZeroDivide,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_EBADFUNC => Error.BadFunction,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// A convergence tolerance for the test helpers. Grouping the two tolerances
/// into one named-field value keeps them from being transposed at a call site
/// (both are `f64`); it defaults to a purely absolute target.
pub const Tol = struct {
    /// Absolute tolerance (`epsabs`).
    abs: f64 = 1e-9,
    /// Relative tolerance (`epsrel`). 0 disables the relative criterion.
    rel: f64 = 0,
};

/// A closed interval known to bracket a root, as reported by `Solver.interval`.
pub const Interval = struct {
    lower: f64,
    upper: f64,
};

/// A `gsl_function`-shaped callback value for the bracketing solvers. Construct
/// with `Callback.initFn(f)` (a plain function) or `Callback.initCtx(&ctx)`.
pub const Callback = callback.Function(c.gsl_function);

/// A `gsl_function_fdf`-shaped callback value for the derivative solvers.
/// Construct with `CallbackFdf.initCtx(&ctx)`, where `ctx` exposes
/// `pub fn eval(self, x) f64` and `pub fn deriv(self, x) f64` (and, optionally,
/// a fused `pub fn evalDeriv(self, x, *f64, *f64)`; see `gsl.callback`).
pub const CallbackFdf = callback.FunctionFdf(c.gsl_function_fdf);

/// The bracketing algorithm used by `Solver`, mirroring GSL's
/// `gsl_root_fsolver_*` types.
pub const Bracket = enum {
    /// Bisection — always converges, linearly; the most robust.
    bisection,
    /// Brent-Dekker — combines bisection with interpolation; fast and robust.
    brent,
    /// False position (regula falsi) — interpolating, keeps a bracket.
    falsepos,

    fn typePtr(self: Bracket) *const c.gsl_root_fsolver_type {
        return switch (self) {
            .bisection => c.gsl_root_fsolver_bisection,
            .brent => c.gsl_root_fsolver_brent,
            .falsepos => c.gsl_root_fsolver_falsepos,
        };
    }
};

/// The derivative-based algorithm used by `PolishSolver`, mirroring GSL's
/// `gsl_root_fdfsolver_*` types.
pub const Polish = enum {
    /// Newton's method — quadratic convergence near a simple root.
    newton,
    /// Secant method — uses the derivative only for the first step.
    secant,
    /// Steffenson's method — Newton with Aitken acceleration; the fastest.
    steffenson,

    fn typePtr(self: Polish) *const c.gsl_root_fdfsolver_type {
        return switch (self) {
            .newton => c.gsl_root_fdfsolver_newton,
            .secant => c.gsl_root_fdfsolver_secant,
            .steffenson => c.gsl_root_fdfsolver_steffenson,
        };
    }
};

/// A stateful bracketing root solver. Owns its GSL allocation (`deinit` frees
/// it) and, once `set`, an inline copy of the `Callback` GSL points at across
/// iterations — so do not move it between `set` and the last `iterate`.
pub const Solver = struct {
    ptr: *c.gsl_root_fsolver,
    cb: Callback = undefined,

    /// Allocate a solver using algorithm `bracket`.
    pub fn init(bracket: Bracket) Error!Solver {
        gsl.ensureHandler();
        const p = c.gsl_root_fsolver_alloc(bracket.typePtr()) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: *Solver) void {
        c.gsl_root_fsolver_free(self.ptr);
    }

    /// Provide the function and a bracket `[lower, upper]` straddling a sign
    /// change. Returns `Error.Invalid` if the endpoints do not bracket a root.
    /// May be called again to restart the solver on a new problem.
    pub fn set(self: *Solver, cb: Callback, lower: f64, upper: f64) Error!void {
        self.cb = cb;
        try check(c.gsl_root_fsolver_set(self.ptr, &self.cb.gf, lower, upper));
    }

    /// Perform one iteration, shrinking the bracket toward the root.
    pub fn iterate(self: *Solver) Error!void {
        try check(c.gsl_root_fsolver_iterate(self.ptr));
    }

    /// The current best estimate of the root.
    pub fn root(self: *const Solver) f64 {
        return c.gsl_root_fsolver_root(self.ptr);
    }

    /// The current bracket `[lower, upper]`.
    pub fn interval(self: *const Solver) Interval {
        return .{
            .lower = c.gsl_root_fsolver_x_lower(self.ptr),
            .upper = c.gsl_root_fsolver_x_upper(self.ptr),
        };
    }

    /// The standardized name of the algorithm (e.g. "brent").
    pub fn name(self: *const Solver) [:0]const u8 {
        return std.mem.span(c.gsl_root_fsolver_name(self.ptr));
    }
};

/// A stateful derivative-based root solver. Owns its GSL allocation (`deinit`
/// frees it) and, once `set`, an inline copy of the `CallbackFdf` GSL points at
/// across iterations — so do not move it between `set` and the last `iterate`.
pub const PolishSolver = struct {
    ptr: *c.gsl_root_fdfsolver,
    cb: CallbackFdf = undefined,

    /// Allocate a solver using algorithm `polish`.
    pub fn init(polish: Polish) Error!PolishSolver {
        gsl.ensureHandler();
        const p = c.gsl_root_fdfsolver_alloc(polish.typePtr()) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: *PolishSolver) void {
        c.gsl_root_fdfsolver_free(self.ptr);
    }

    /// Provide the function (with its derivative) and a starting guess. May be
    /// called again to restart the solver on a new problem.
    pub fn set(self: *PolishSolver, cb: CallbackFdf, guess: f64) Error!void {
        self.cb = cb;
        try check(c.gsl_root_fdfsolver_set(self.ptr, &self.cb.fdf, guess));
    }

    /// Perform one iteration, refining the root estimate.
    pub fn iterate(self: *PolishSolver) Error!void {
        try check(c.gsl_root_fdfsolver_iterate(self.ptr));
    }

    /// The current best estimate of the root.
    pub fn root(self: *const PolishSolver) f64 {
        return c.gsl_root_fdfsolver_root(self.ptr);
    }

    /// The standardized name of the algorithm (e.g. "newton").
    pub fn name(self: *const PolishSolver) [:0]const u8 {
        return std.mem.span(c.gsl_root_fdfsolver_name(self.ptr));
    }
};

// --- Convergence tests -----------------------------------------------------
//
// Each returns `true` once the tolerance is met, `false` to keep iterating
// (GSL's `GSL_CONTINUE`), and an `Error` for a genuinely bad argument.

fn converged(status: c_int) Error!bool {
    return switch (status) {
        c.GSL_SUCCESS => true,
        c.GSL_CONTINUE => false,
        else => {
            try check(status);
            unreachable; // a non-success/continue status always maps to an error
        },
    };
}

/// Test whether a bracket `[lower, upper]` is small enough to satisfy `tol`
/// (`|upper - lower| < abs + rel·min(|lower|, |upper|)`). Pair with `Solver`.
pub fn testInterval(lower: f64, upper: f64, tol: Tol) Error!bool {
    return converged(c.gsl_root_test_interval(lower, upper, tol.abs, tol.rel));
}

/// Test whether two successive iterates `x1`, `x0` differ by less than `tol`
/// (`|x1 - x0| < abs + rel·|x1|`). Pair with `PolishSolver`.
pub fn testDelta(x1: f64, x0: f64, tol: Tol) Error!bool {
    return converged(c.gsl_root_test_delta(x1, x0, tol.abs, tol.rel));
}

/// Test whether a residual `f` (a function value at the current estimate) is
/// within absolute tolerance `epsabs` of zero (`|f| < epsabs`).
pub fn testResidual(f: f64, epsabs: f64) Error!bool {
    return converged(c.gsl_root_test_residual(f, epsabs));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const sqrt2 = std.math.sqrt2;

// f(x) = x^2 - 2, with root √2 on [0, 2].
fn quadratic(x: f64) f64 {
    return x * x - 2.0;
}

// A context carrying f, f', and a fused evalDeriv for the polishing solvers.
const Quad = struct {
    pub fn eval(_: *const @This(), x: f64) f64 {
        return x * x - 2.0;
    }
    pub fn deriv(_: *const @This(), x: f64) f64 {
        return 2.0 * x;
    }
};

test "roots: every bracketing solver finds √2 of x²−2 on [0, 2]" {
    inline for (comptime std.enums.values(Bracket)) |bracket| {
        var s = try Solver.init(bracket);
        defer s.deinit();
        try s.set(.initFn(quadratic), 0.0, 2.0);

        var iters: usize = 0;
        while (iters < 100) : (iters += 1) {
            try s.iterate();
            const iv = s.interval();
            if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-10 })) break;
        }
        try testing.expect(iters < 100);
        try testing.expectApproxEqAbs(sqrt2, s.root(), 1e-9);
    }
}

test "roots: a bracket that does not straddle a root is rejected" {
    var s = try Solver.init(.brent);
    defer s.deinit();
    // f > 0 on all of [2, 3], so there is no sign change to bracket.
    try testing.expectError(Error.Invalid, s.set(.initFn(quadratic), 2.0, 3.0));
}

test "roots: every polishing solver refines √2 from a nearby guess" {
    inline for (comptime std.enums.values(Polish)) |polish| {
        var s = try PolishSolver.init(polish);
        defer s.deinit();
        var q = Quad{};
        try s.set(.initCtx(&q), 1.5);

        var iters: usize = 0;
        while (iters < 100) : (iters += 1) {
            const x0 = s.root();
            try s.iterate();
            const x1 = s.root();
            if (try testDelta(x1, x0, .{ .abs = 1e-12 })) break;
        }
        try testing.expect(iters < 100);
        try testing.expectApproxEqAbs(sqrt2, s.root(), 1e-9);
    }
}

test "roots: a bracketing solver captures a parameter via a context" {
    // f(x) = x² − k; with k = 3 the positive root is √3.
    const Param = struct {
        k: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            return x * x - self.k;
        }
    };
    var p = Param{ .k = 3.0 };
    var s = try Solver.init(.brent);
    defer s.deinit();
    try s.set(.initCtx(&p), 0.0, 3.0);

    var iters: usize = 0;
    while (iters < 100) : (iters += 1) {
        try s.iterate();
        const iv = s.interval();
        if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-10 })) break;
    }
    try testing.expectApproxEqAbs(@sqrt(3.0), s.root(), 1e-9);
}

test "roots: solvers report their algorithm name" {
    var s = try Solver.init(.bisection);
    defer s.deinit();
    try testing.expectEqualStrings("bisection", s.name());

    var p = try PolishSolver.init(.steffenson);
    defer p.deinit();
    try testing.expectEqualStrings("steffenson", p.name());
}

test "roots: convergence helpers report continue then success" {
    // A wide interval is not converged; a tight one is.
    try testing.expect(!try testInterval(0.0, 1.0, .{ .abs = 1e-6 }));
    try testing.expect(try testInterval(1.0, 1.0 + 1e-9, .{ .abs = 1e-6 }));

    // testDelta on successive iterates.
    try testing.expect(!try testDelta(1.0, 2.0, .{ .abs = 1e-6 }));
    try testing.expect(try testDelta(1.0, 1.0 + 1e-9, .{ .abs = 1e-6 }));

    // testResidual on a function value.
    try testing.expect(!try testResidual(0.1, 1e-6));
    try testing.expect(try testResidual(1e-9, 1e-6));

    // A negative tolerance is a bad argument, not "not converged".
    try testing.expectError(Error.BadTolerance, testInterval(0.0, 1.0, .{ .abs = -1.0 }));
}

test "roots: a solver can be reset onto a new problem" {
    var s = try Solver.init(.brent);
    defer s.deinit();

    // First: root of x² − 2 on [0, 2].
    try s.set(.initFn(quadratic), 0.0, 2.0);
    for (0..100) |_| {
        try s.iterate();
        const iv = s.interval();
        if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-10 })) break;
    }
    try testing.expectApproxEqAbs(sqrt2, s.root(), 1e-9);

    // Reuse the same allocation for cos(x) = 0 on [1, 2] → π/2.
    const cosFn = struct {
        fn f(x: f64) f64 {
            return @cos(x);
        }
    }.f;
    try s.set(.initFn(cosFn), 1.0, 2.0);
    for (0..100) |_| {
        try s.iterate();
        const iv = s.interval();
        if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-10 })) break;
    }
    try testing.expectApproxEqAbs(std.math.pi / 2.0, s.root(), 1e-9);
}
