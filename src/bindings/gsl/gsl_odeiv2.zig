//! Idiomatic Zig bindings for the GNU Scientific Library's initial-value ODE
//! module (`gsl_odeiv2`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with ODE integration of
//! first-order systems `y'(t) = f(t, y)`. It reuses `gsl.zig`'s process-global
//! error-handler switch but keeps the `gsl_odeiv2` C API behind its own `c`.
//! It is reached as `gsl.ode`.
//!
//! ## Shape of the surface
//!
//! The wrapped high-level handle is `Driver`, which owns a
//! `gsl_odeiv2_driver` and integrates one system over one mutable state vector:
//!
//!   1. Build a `System` from a context pointer:
//!      - `System.initCtx(&ctx, dim)` for RHS-only systems (`rhs` callback).
//!      - `System.initCtxWithJacobian(&ctx, dim)` when an explicit Jacobian is
//!        available (`rhs` + `jacobian` callbacks).
//!   2. Allocate a `Driver` with a stepper `Step`, starting step `h_start`, and
//!      tolerance `Tol{ .abs, .rel }` via `Driver.initY` (or `initYp`).
//!   3. Repeatedly call `apply` (adaptive to target time) or
//!      `applyFixedStep` (exactly `n` fixed-size steps).
//!
//! ```zig
//! const Decay = struct {
//!     k: f64,
//!     pub fn rhs(self: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
//!         _ = t;
//!         dydt[0] = -self.k * y[0];
//!     }
//! };
//!
//! var model = Decay{ .k = 2.0 };
//! var sys = gsl.ode.System.initCtx(&model, 1);
//! var drv = try gsl.ode.Driver.initY(&sys, .rk8pd, 1e-6, .{ .abs = 1e-10, .rel = 1e-10 });
//! defer drv.deinit();
//!
//! var t: f64 = 0;
//! var y = [_]f64{1.0};
//! try drv.apply(&t, 1.0, &y);
//! // y[0] ≈ exp(-2)
//! ```
//!
//! ## Lifetime
//!
//! `gsl_odeiv2_driver` stores a pointer to the supplied `gsl_odeiv2_system`, so
//! `Driver.initY`/`initYp` take `*System` and the caller must keep that `System`
//! alive and unmoved until `deinit`.
//!
//! ## Omissions
//!
//!   - The lower-level `step`/`control`/`evolve` objects are not wrapped yet.
//!   - The `alloc_standard_new` / `alloc_scaled_new` driver constructors are not
//!     wrapped; use the raw `c` API for custom control heuristics.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const callback = @import("gsl_callback.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_odeiv2.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the ODE routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_odeiv2_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EINVAL` — invalid argument.
    Invalid,
    /// The state slice length does not match the system dimension.
    BadLength,
    /// `GSL_EBADFUNC` — callback returned a bad/unsupported function status.
    BadFunction,
    /// `GSL_ENOPROG` / `GSL_ENOPROGJ` — stepper is not making progress.
    NoProgress,
    /// `GSL_EFAILED` / `GSL_FAILURE` — generic internal failure.
    Failed,
    /// `GSL_EMAXITER` — configured maximum number of internal steps reached.
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
        c.GSL_ENOPROG, c.GSL_ENOPROGJ => Error.NoProgress,
        c.GSL_EFAILED, c.GSL_FAILURE => Error.Failed,
        c.GSL_EMAXITER => Error.MaxIterations,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// Absolute/relative error targets for adaptive stepping.
pub const Tol = struct {
    /// Absolute tolerance (`epsabs`).
    abs: f64 = 1e-9,
    /// Relative tolerance (`epsrel`).
    rel: f64 = 1e-9,
};

/// Available steppers (`gsl_odeiv2_step_*`).
pub const Step = enum {
    rk2,
    rk4,
    rkf45,
    rkck,
    rk8pd,
    rk2imp,
    rk4imp,
    bsimp,
    rk1imp,
    msadams,
    msbdf,

    fn typePtr(self: Step) *const c.gsl_odeiv2_step_type {
        return switch (self) {
            .rk2 => c.gsl_odeiv2_step_rk2,
            .rk4 => c.gsl_odeiv2_step_rk4,
            .rkf45 => c.gsl_odeiv2_step_rkf45,
            .rkck => c.gsl_odeiv2_step_rkck,
            .rk8pd => c.gsl_odeiv2_step_rk8pd,
            .rk2imp => c.gsl_odeiv2_step_rk2imp,
            .rk4imp => c.gsl_odeiv2_step_rk4imp,
            .bsimp => c.gsl_odeiv2_step_bsimp,
            .rk1imp => c.gsl_odeiv2_step_rk1imp,
            .msadams => c.gsl_odeiv2_step_msadams,
            .msbdf => c.gsl_odeiv2_step_msbdf,
        };
    }
};

/// ODE system callback bundle (`gsl_odeiv2_system`). Construct from a context
/// with either:
///
///   - `initCtx(&ctx, dim)` where `ctx` declares
///     `pub fn rhs(self, t: f64, y: [*c]const f64, dydt: [*c]f64) void|c_int`
///   - `initCtxWithJacobian(&ctx, dim)` where `ctx` additionally declares
///     `pub fn jacobian(self, t: f64, y: [*c]const f64, dfdy: [*c]f64, dfdt: [*c]f64) void|c_int`
///
/// The pointer arguments match GSL's callback signature; read/write the first
/// `dim` entries (`dfdy` is row-major, length `dim*dim`).
pub const System = struct {
    sys: c.gsl_odeiv2_system,

    pub fn initCtx(ctx: anytype, dim: usize) System {
        return .{ .sys = callback.odeSystem(c.gsl_odeiv2_system, dim, ctx) };
    }

    pub fn initCtxWithJacobian(ctx: anytype, dim: usize) System {
        return .{ .sys = callback.odeSystemWithJacobian(c.gsl_odeiv2_system, dim, ctx) };
    }

    pub fn dimension(self: *const System) usize {
        return self.sys.dimension;
    }
};

/// High-level adaptive ODE integrator (`gsl_odeiv2_driver`). Owns its GSL
/// allocation; call `deinit` to free.
///
/// Lifetime note: this stores a pointer to a caller-owned `System`; keep that
/// `System` alive and unmoved for the life of the driver.
pub const Driver = struct {
    ptr: *c.gsl_odeiv2_driver,
    system: *System,

    /// Allocate a driver using the standard `y` control heuristic.
    pub fn initY(system: *System, step: Step, h_start: f64, tol: Tol) Error!Driver {
        if (system.sys.dimension == 0) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_odeiv2_driver_alloc_y_new(&system.sys, step.typePtr(), h_start, tol.abs, tol.rel) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .system = system };
    }

    /// Allocate a driver using the `yp` control heuristic.
    pub fn initYp(system: *System, step: Step, h_start: f64, tol: Tol) Error!Driver {
        if (system.sys.dimension == 0) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_odeiv2_driver_alloc_yp_new(&system.sys, step.typePtr(), h_start, tol.abs, tol.rel) orelse return Error.OutOfMemory;
        return .{ .ptr = p, .system = system };
    }

    pub fn deinit(self: *Driver) void {
        c.gsl_odeiv2_driver_free(self.ptr);
    }

    /// Integrate from `t.*` to `t1`, updating `t.*` and `y` in place.
    pub fn apply(self: *Driver, t: *f64, t1: f64, y: []f64) Error!void {
        if (y.len != self.system.sys.dimension) return Error.BadLength;
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_apply(self.ptr, t, t1, y.ptr));
    }

    /// Advance exactly `n` fixed-size steps of length `h`.
    pub fn applyFixedStep(self: *Driver, t: *f64, h: f64, n: usize, y: []f64) Error!void {
        if (y.len != self.system.sys.dimension) return Error.BadLength;
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_apply_fixed_step(self.ptr, t, h, @as(c_ulong, @intCast(n)), y.ptr));
    }

    /// Reset the internal step/control/evolve state.
    pub fn reset(self: *Driver) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_reset(self.ptr));
    }

    /// Reset and set a new starting step size.
    pub fn resetHStart(self: *Driver, h_start: f64) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_reset_hstart(self.ptr, h_start));
    }

    /// Set a minimum internal step size.
    pub fn setHMin(self: *Driver, h_min: f64) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_set_hmin(self.ptr, h_min));
    }

    /// Set a maximum internal step size.
    pub fn setHMax(self: *Driver, h_max: f64) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_set_hmax(self.ptr, h_max));
    }

    /// Set the maximum number of internal steps per `apply` call.
    pub fn setNMax(self: *Driver, n_max: usize) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_odeiv2_driver_set_nmax(self.ptr, @as(c_ulong, @intCast(n_max))));
    }

    /// Number of internal steps taken so far.
    pub fn stepCount(self: *const Driver) usize {
        return @intCast(self.ptr.*.n);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "ode: adaptive explicit solver integrates dy/dt = -k y" {
    const Decay = struct {
        k: f64,
        pub fn rhs(self: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
            _ = t;
            dydt[0] = -self.k * y[0];
        }
    };

    var model = Decay{ .k = 2.0 };
    var sys = System.initCtx(&model, 1);
    var d = try Driver.initY(&sys, .rk8pd, 1e-6, .{ .abs = 1e-10, .rel = 1e-10 });
    defer d.deinit();

    var t: f64 = 0.0;
    var y = [_]f64{1.0};
    try d.apply(&t, 1.0, &y);

    try testing.expectApproxEqAbs(@as(f64, 1.0), t, 1e-12);
    try testing.expectApproxEqAbs(@exp(@as(f64, -2.0)), y[0], 1e-9);
}

test "ode: implicit bsimp uses a supplied jacobian" {
    const Stiff = struct {
        k: f64,
        pub fn rhs(self: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
            _ = t;
            dydt[0] = -self.k * y[0];
        }
        pub fn jacobian(self: *const @This(), t: f64, y: [*c]const f64, dfdy: [*c]f64, dfdt: [*c]f64) void {
            _ = t;
            _ = y;
            dfdy[0] = -self.k;
            dfdt[0] = 0.0;
        }
    };

    var model = Stiff{ .k = 25.0 };
    var sys = System.initCtxWithJacobian(&model, 1);
    var d = try Driver.initY(&sys, .bsimp, 1e-6, .{ .abs = 1e-10, .rel = 1e-10 });
    defer d.deinit();

    var t: f64 = 0.0;
    var y = [_]f64{1.0};
    try d.apply(&t, 0.1, &y);

    try testing.expectApproxEqAbs(@exp(@as(f64, -2.5)), y[0], 1e-8);
}

test "ode: fixed-step integration and reset helpers" {
    const Decay = struct {
        pub fn rhs(_: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
            _ = t;
            dydt[0] = -y[0];
        }
    };

    var model = Decay{};
    var sys = System.initCtx(&model, 1);
    var d = try Driver.initY(&sys, .rk4, 1e-3, .{ .abs = 1e-10, .rel = 1e-10 });
    defer d.deinit();

    try d.setHMin(1e-12);
    try d.setHMax(1e-1);
    try d.setNMax(200_000);

    var t: f64 = 0.0;
    var y = [_]f64{1.0};
    try d.applyFixedStep(&t, 1e-3, 1000, &y);

    try testing.expectApproxEqAbs(@as(f64, 1.0), t, 1e-12);
    try testing.expectApproxEqAbs(@exp(@as(f64, -1.0)), y[0], 1e-6);

    try d.reset();
    try d.resetHStart(5e-4);
    try testing.expect(d.stepCount() > 0);
}

test "ode: apply rejects a state vector whose length mismatches the system" {
    const Pair = struct {
        pub fn rhs(_: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
            _ = t;
            dydt[0] = y[1];
            dydt[1] = -y[0];
        }
    };

    var model = Pair{};
    var sys = System.initCtx(&model, 2);
    var d = try Driver.initY(&sys, .rkf45, 1e-3, .{});
    defer d.deinit();

    var t: f64 = 0.0;
    var y = [_]f64{1.0};
    try testing.expectError(Error.BadLength, d.apply(&t, 1.0, &y));
}

test "ode: callback-returned EBADFUNC surfaces as Error.BadFunction" {
    const Bad = struct {
        pub fn rhs(_: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) c_int {
            _ = t;
            _ = y;
            _ = dydt;
            return c.GSL_EBADFUNC;
        }
    };

    var bad = Bad{};
    var sys = System.initCtx(&bad, 1);
    var d = try Driver.initY(&sys, .rk4, 1e-3, .{});
    defer d.deinit();

    var t: f64 = 0.0;
    var y = [_]f64{1.0};
    try testing.expectError(Error.BadFunction, d.apply(&t, 0.1, &y));
}

test "ode: a zero-dimension system is rejected by driver init" {
    const Empty = struct {
        pub fn rhs(_: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
            _ = t;
            _ = y;
            _ = dydt;
        }
    };

    var e = Empty{};
    var sys = System.initCtx(&e, 0);
    try testing.expectError(Error.Invalid, Driver.initY(&sys, .rk4, 1e-3, .{}));
}
