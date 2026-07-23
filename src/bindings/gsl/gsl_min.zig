//! Idiomatic Zig bindings for the GNU Scientific Library's one-dimensional
//! minimization module (`gsl_min`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the one-dim function
//! minimization chapter. Its shape is a close twin of `gsl.roots` — a stateful
//! solver you `init`, `set`, and `iterate` — differing only in that it brackets
//! a *minimum* rather than a root. It reuses `gsl.zig`'s process-global
//! error-handler switch but keeps the `gsl_min` C API behind its own `c`. It is
//! reached as `gsl.min`.
//!
//! ## Shape of the surface
//!
//! `Minimizer` wraps GSL's `gsl_min_fminimizer`. You supply a function, a
//! bracketing triple `lower < guess < upper` where the function at `guess` is
//! *lower* than at both endpoints (so a minimum is trapped inside), and then
//! iterate to tighten the bracket:
//!
//! ```zig
//! var m = try gsl.min.Minimizer.init(.brent);
//! defer m.deinit();
//! try m.set(.initFn(f), 2.0, 0.0, 6.0); // guess=2 inside [0, 6]
//! while (true) {
//!     try m.iterate();
//!     const iv = m.interval();
//!     if (try gsl.min.testInterval(iv.lower, iv.upper, .{ .abs = 1e-6 })) break;
//! }
//! const x = m.minimum();   // location of the minimum
//! const fx = m.fMinimum(); // function value there
//! ```
//!
//! `testInterval` returns `true` when the bracket satisfies the tolerance and
//! `false` to keep iterating (GSL's `GSL_CONTINUE`), surfacing any genuine
//! error otherwise.
//!
//! ## Lifetime
//!
//! Like `gsl.roots`, the minimizer retains a *pointer* to the function across
//! iterations, so it stores the `Callback` it was `set` with **by value**. Do
//! not move the minimizer between `set` and the final `iterate`, and — for
//! context callbacks (`.initCtx`) — keep the context alive for that span.
//!
//! ## Omissions
//!
//!   - `gsl_min_fminimizer_set_with_values` (pass precomputed function values)
//!     is not wrapped; `set` recomputes them, which is what nearly all callers
//!     want. The individual `f_lower`/`f_upper` accessors and the experimental
//!     `gsl_min_find_bracket` are likewise left to the raw `c` API.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_math.h");
    @cInclude("gsl/gsl_min.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the minimization routines. The raw `c_int` status is
/// always available from the underlying `c.gsl_min_*` symbol if you need the
/// exact code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument; most often the triple `lower < guess
    /// < upper` does not trap a minimum (i.e. `f(guess)` is not below both
    /// endpoints), or a nonsensical tolerance in a test.
    Invalid,
    /// `GSL_EBADTOL` — a nonsensical tolerance (e.g. negative) in a test.
    BadTolerance,
    /// `GSL_EMAXITER` — the iteration failed to make progress.
    MaxIterations,
    /// `GSL_EBADFUNC` — the function produced a singular (Inf/NaN) value.
    BadFunction,
    /// `GSL_EFAILED`/`GSL_FAILURE` — a general failure inside an iteration
    /// (e.g. a solver that cannot shrink the bracket further on a flat region).
    Failed,
    /// `GSL_ENOMEM` — minimizer allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADTOL => Error.BadTolerance,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_EBADFUNC => Error.BadFunction,
        c.GSL_FAILURE, c.GSL_EFAILED => Error.Failed,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// A convergence tolerance for `testInterval`. Grouping the two tolerances into
/// one named-field value keeps them from being transposed at a call site (both
/// are `f64`); it defaults to a purely absolute target.
pub const Tol = struct {
    /// Absolute tolerance (`epsabs`).
    abs: f64 = 1e-9,
    /// Relative tolerance (`epsrel`). 0 disables the relative criterion.
    rel: f64 = 0,
};

/// The bracket `[lower, upper]` currently trapping the minimum, as reported by
/// `Minimizer.interval`.
pub const Interval = struct {
    lower: f64,
    upper: f64,
};

/// A `gsl_function`-shaped callback value for the minimizer. Construct with
/// `Callback.initFn(f)` (a plain function) or `Callback.initCtx(&ctx)`.
pub const Callback = callback.Function(c.gsl_function);

/// The minimization algorithm used by `Minimizer`, mirroring GSL's
/// `gsl_min_fminimizer_*` types.
pub const Method = enum {
    /// Golden-section search — robust, linear convergence.
    goldensection,
    /// Brent's method — parabolic interpolation with a golden-section
    /// fallback; the usual default.
    brent,
    /// A safeguarded quadratic/golden hybrid (Gill & Murray).
    quad_golden,

    fn typePtr(self: Method) *const c.gsl_min_fminimizer_type {
        return switch (self) {
            .goldensection => c.gsl_min_fminimizer_goldensection,
            .brent => c.gsl_min_fminimizer_brent,
            .quad_golden => c.gsl_min_fminimizer_quad_golden,
        };
    }
};

/// A stateful one-dimensional function minimizer. Owns its GSL allocation
/// (`deinit` frees it) and, once `set`, an inline copy of the `Callback` GSL
/// points at across iterations — so do not move it between `set` and the last
/// `iterate`.
pub const Minimizer = struct {
    ptr: *c.gsl_min_fminimizer,
    cb: Callback = undefined,

    /// Allocate a minimizer using algorithm `method`.
    pub fn init(method: Method) Error!Minimizer {
        gsl.ensureHandler();
        const p = c.gsl_min_fminimizer_alloc(method.typePtr()) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: *Minimizer) void {
        c.gsl_min_fminimizer_free(self.ptr);
    }

    /// Provide the function and a bracketing triple: `lower < guess < upper`
    /// with `f(guess)` strictly below both `f(lower)` and `f(upper)`, so a
    /// minimum is trapped in `[lower, upper]`. Returns `Error.Invalid` if the
    /// triple does not satisfy that condition. May be called again to restart
    /// the minimizer on a new problem.
    pub fn set(self: *Minimizer, cb: Callback, guess: f64, lower: f64, upper: f64) Error!void {
        self.cb = cb;
        try check(c.gsl_min_fminimizer_set(self.ptr, &self.cb.gf, guess, lower, upper));
    }

    /// Perform one iteration, tightening the bracket around the minimum.
    pub fn iterate(self: *Minimizer) Error!void {
        try check(c.gsl_min_fminimizer_iterate(self.ptr));
    }

    /// The current best estimate of the minimum's *location* (`x_minimum`).
    pub fn minimum(self: *const Minimizer) f64 {
        return c.gsl_min_fminimizer_x_minimum(self.ptr);
    }

    /// The function value at the current estimated minimum (`f_minimum`).
    pub fn fMinimum(self: *const Minimizer) f64 {
        return c.gsl_min_fminimizer_f_minimum(self.ptr);
    }

    /// The current bracket `[lower, upper]` trapping the minimum.
    pub fn interval(self: *const Minimizer) Interval {
        return .{
            .lower = c.gsl_min_fminimizer_x_lower(self.ptr),
            .upper = c.gsl_min_fminimizer_x_upper(self.ptr),
        };
    }

    /// The standardized name of the algorithm (e.g. "brent").
    pub fn name(self: *const Minimizer) [:0]const u8 {
        return std.mem.span(c.gsl_min_fminimizer_name(self.ptr));
    }
};

/// Test whether a bracket `[lower, upper]` is small enough to satisfy `tol`
/// (`|upper - lower| < abs + rel·min(|lower|, |upper|)`). Returns `true` once
/// converged, `false` to keep iterating (GSL's `GSL_CONTINUE`), and an `Error`
/// for a genuinely bad argument.
pub fn testInterval(lower: f64, upper: f64, tol: Tol) Error!bool {
    return switch (c.gsl_min_test_interval(lower, upper, tol.abs, tol.rel)) {
        c.GSL_SUCCESS => true,
        c.GSL_CONTINUE => false,
        else => |status| {
            try check(status);
            unreachable; // a non-success/continue status always maps to an error
        },
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// f(x) = (x - 2)^2, with its minimum at x = 2 (value 0).
fn parabola(x: f64) f64 {
    return (x - 2.0) * (x - 2.0);
}

test "min: every method locates the minimum of cos on [0, 6]" {
    // cos has its minimum at x = π (value −1) on [0, 6]. A transcendental
    // target keeps all three methods contracting the bracket normally — an
    // exactly-quadratic target lets the interpolating methods jump to the
    // vertex and then stall the bracket. Note the tolerance: locating a 1-D
    // minimum is only accurate to ~√eps, so an over-tight interval target is
    // physically unreachable (golden section bottoms out around 1e-7 and
    // reports `GSL_FAILURE`).
    const cosFn = struct {
        fn f(x: f64) f64 {
            return @cos(x);
        }
    }.f;
    inline for (comptime std.enums.values(Method)) |method| {
        var m = try Minimizer.init(method);
        defer m.deinit();
        try m.set(.initFn(cosFn), 3.0, 0.0, 6.0);

        var iters: usize = 0;
        while (iters < 200) : (iters += 1) {
            try m.iterate();
            const iv = m.interval();
            if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-5 })) break;
        }
        try testing.expect(iters < 200);
        try testing.expectApproxEqAbs(std.math.pi, m.minimum(), 1e-5);
        try testing.expectApproxEqAbs(@as(f64, -1.0), m.fMinimum(), 1e-8);
    }
}

test "min: a triple that does not trap a minimum is rejected" {
    var m = try Minimizer.init(.brent);
    defer m.deinit();
    // f(guess) must be below both endpoints; here f is monotone on [3, 5], so
    // f(4) is not a low point between f(3) and f(5).
    try testing.expectError(Error.Invalid, m.set(.initFn(parabola), 4.0, 3.0, 5.0));
}

test "min: a context struct captures a parameter" {
    // f(x) = (x - c)²; the minimum sits at x = c.
    const Shifted = struct {
        c: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            const d = x - self.c;
            return d * d;
        }
    };
    var s = Shifted{ .c = 3.5 };
    var m = try Minimizer.init(.brent);
    defer m.deinit();
    try m.set(.initCtx(&s), 3.0, 0.0, 6.0);

    var iters: usize = 0;
    while (iters < 200) : (iters += 1) {
        try m.iterate();
        const iv = m.interval();
        if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-5 })) break;
    }
    try testing.expectApproxEqAbs(@as(f64, 3.5), m.minimum(), 1e-5);
}

test "min: minimizer reports its algorithm name" {
    var m = try Minimizer.init(.goldensection);
    defer m.deinit();
    try testing.expectEqualStrings("goldensection", m.name());
}

test "min: a minimizer can be reset onto a new problem" {
    var m = try Minimizer.init(.brent);
    defer m.deinit();

    // First: minimum of (x − 2)² on [0, 5] at x = 2.
    try m.set(.initFn(parabola), 1.0, 0.0, 5.0);
    for (0..200) |_| {
        try m.iterate();
        const iv = m.interval();
        if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-5 })) break;
    }
    try testing.expectApproxEqAbs(@as(f64, 2.0), m.minimum(), 1e-5);

    // Reuse the same allocation for cos(x) on [0, 6], whose minimum is at π.
    const cosFn = struct {
        fn f(x: f64) f64 {
            return @cos(x);
        }
    }.f;
    try m.set(.initFn(cosFn), 3.0, 0.0, 6.0);
    for (0..200) |_| {
        try m.iterate();
        const iv = m.interval();
        if (try testInterval(iv.lower, iv.upper, .{ .abs = 1e-5 })) break;
    }
    try testing.expectApproxEqAbs(std.math.pi, m.minimum(), 1e-5);
    try testing.expectApproxEqAbs(@as(f64, -1.0), m.fMinimum(), 1e-8);
}

test "min: testInterval reports continue then success, and rejects bad tolerance" {
    try testing.expect(!try testInterval(0.0, 1.0, .{ .abs = 1e-6 }));
    try testing.expect(try testInterval(1.0, 1.0 + 1e-9, .{ .abs = 1e-6 }));
    try testing.expectError(Error.BadTolerance, testInterval(0.0, 1.0, .{ .abs = -1.0 }));
}
