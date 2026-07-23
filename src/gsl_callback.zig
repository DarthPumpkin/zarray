//! The callback bridge: turn an idiomatic Zig callable into the C
//! function-pointer-plus-`void *params` structs GSL's callback chapters expect
//! (`gsl_function`, `gsl_function_fdf`, `gsl_monte_function`, and
//! `gsl_odeiv2_system`).
//!
//! This file deliberately has **no `@cImport`**. Every builder is generic over
//! the *target* C struct type, which the calling chapter supplies from its own
//! `@cImport` (e.g. `gsl.deriv` passes its `c.gsl_function`). Two `@cImport`
//! blocks that both pull `gsl_math.h` produce *distinct* `gsl_function` types,
//! so a bridge with its own copy would be incompatible with every consumer —
//! exactly the problem the generic `constVectorViewOf`/`mutVectorViewOf` helpers
//! solve for `gsl_vector`. Being `@cImport`-free, this bridge is pure comptime
//! glue that constructs whatever struct type it's handed.
//!
//! ## Two distinct forms (ratified: D-cb2)
//!
//! The plain-function and stateful-context cases are constructed through
//! *named factory methods* on a per-chapter callback value, so the caller is
//! never in doubt about what to supply:
//!
//!   - `Callback.initFn(f)` — `f` is a `*const fn(f64) f64`. A bare top-level
//!     function coerces to this pointer type. Nothing is captured.
//!   - `Callback.initCtx(ctx)` — `ctx` is a pointer to a struct exposing
//!     `pub fn eval(self, x: f64) f64`. The pointer is carried through `params`,
//!     capturing state with no allocation. The method must be `pub` (the
//!     generated trampoline lives in this file and calls it across the module
//!     boundary); a comptime check reports a clear error if it is missing.
//!
//! Each chapter declares its own callback value type by instantiating the
//! generic `Function(GF)` (or `FunctionFdf(GF)`) over its own `c.gsl_function`
//! type — distinct `@cImport` blocks produce distinct C struct types, so the
//! value cannot be a single shared type. A routine takes that value and reads
//! `&value.gf` for the synchronous GSL call:
//!
//! ```zig
//! // in a chapter (e.g. gsl_deriv.zig):
//! pub const Callback = callback.Function(c.gsl_function);
//! pub fn central(cb: Callback, x: f64, h: f64) Error!Result { ... &cb.gf ... }
//!
//! // at the call site:
//! const d1 = try gsl.deriv.central(.initFn(myFn), 2.0, 1e-8);
//! const d2 = try gsl.deriv.central(.initCtx(&wave), 0.5, 1e-6);
//! ```
//!
//! The low-level builders `function`/`context`/`functionFdf` (which construct
//! the raw C struct directly) remain available for callers that want them.
//!
//! `functionFdf`/`FunctionFdf` are the derivative-bearing analog for root
//! polishing: `ctx` exposes `eval` and `deriv`, and optionally `evalDeriv(x,
//! *f64, *f64)` for the combined callback (otherwise it is synthesized from
//! `eval` + `deriv`; ratified D-cb3).
//!
//! `MonteFunction` is the *multidimensional* analog for Monte-Carlo integration
//! (`gsl_monte_function`, shaped `f(x[], dim, params)`). The plain-function
//! form takes a `*const fn([]const f64) f64` and the context form a `*struct`
//! with `pub fn eval(self, x: []const f64) f64`; the trampoline reconstructs the
//! `dim`-length slice from GSL's raw pointer before calling. The struct's `dim`
//! field is left 0 at construction and filled in by the consuming chapter
//! immediately before the call (it equals the integration dimension), so the
//! caller never repeats the dimension when building the callback.
//!
//! `odeSystem` and `odeSystemWithJacobian` build the ODE callback struct
//! (`gsl_odeiv2_system`). Both require a context pointer and explicit
//! `dimension`; `odeSystem` needs `pub fn rhs(self, t, y, dydt)`, while
//! `odeSystemWithJacobian` additionally needs
//! `pub fn jacobian(self, t, y, dfdy, dfdt)`. The array arguments are passed in
//! raw C-pointer form (`[*c]const f64` / `[*c]f64`) matching GSL's callback
//! signature; routines read/write the first `dimension` entries. Either method
//! may return `void` (treated as success) or a `c_int` GSL status code.
//!
//! ## Conventions
//!
//!   - **Error propagation (D-cb1):** callbacks must return a finite `f64` and
//!     are assumed infallible. Fallible user code should use the *context* form
//!     with a `caught: *?anyerror` field it sets on failure (returning `NaN`);
//!     the caller inspects that variable after the GSL routine returns. The
//!     bridge itself is unaware of `caught` — it is a pure caller-side pattern.
//!   - **Lifetime:** the returned struct borrows the context via `params`. Build
//!     it immediately before the GSL call and keep the context alive across the
//!     call; do not let the struct outlive the context or escape the scope.
//!   - **Re-entrancy:** trampolines are pure forwarders with no global state, so
//!     they are reentrant/thread-safe as long as each thread builds its own
//!     struct over its own context.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// Build an `Fn` (`gsl_function`-shaped) from a plain function pointer. A bare
/// top-level `fn(f64) f64` coerces to `*const fn(f64) f64`, so it may be passed
/// directly; nothing is captured.
pub fn function(comptime Fn: type, f: *const fn (f64) f64) Fn {
    const Tr = struct {
        fn call(x: f64, params: ?*anyopaque) callconv(.c) f64 {
            const fp: *const fn (f64) f64 = @ptrCast(@alignCast(params.?));
            return fp(x);
        }
    };
    return .{ .function = &Tr.call, .params = @constCast(f) };
}

/// Build an `Fn` (`gsl_function`-shaped) from a pointer to a context struct that
/// exposes `pub fn eval(self, x: f64) f64`. The pointer is carried through
/// `params`, capturing state with no allocation.
pub fn context(comptime Fn: type, ctx: anytype) Fn {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{"eval"});
    const Tr = struct {
        fn call(x: f64, params: ?*anyopaque) callconv(.c) f64 {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            return self.eval(x);
        }
    };
    return .{ .function = &Tr.call, .params = @constCast(ctx) };
}

/// A ready-to-pass `gsl_function`-shaped callback value for a chapter's API.
/// Instantiate once per chapter over its own `c.gsl_function` type
/// (`pub const Callback = callback.Function(c.gsl_function);`), then construct
/// values with `initFn` (a plain function) or `initCtx` (a context struct). A
/// routine receives the value and passes `&value.gf` to GSL.
pub fn Function(comptime GF: type) type {
    return struct {
        gf: GF,
        const Self = @This();

        /// From a plain `*const fn(f64) f64` (a bare function coerces in).
        pub fn initFn(f: *const fn (f64) f64) Self {
            return .{ .gf = function(GF, f) };
        }

        /// From a pointer to a context struct with `pub fn eval(self, x: f64) f64`.
        pub fn initCtx(ctx: anytype) Self {
            return .{ .gf = context(GF, ctx) };
        }
    };
}

/// A ready-to-pass `gsl_function_fdf`-shaped callback value for the
/// derivative-based routines (e.g. root polishing). Instantiate once per
/// chapter over its own `c.gsl_function_fdf` type; construct with `initCtx`
/// (a context with `eval` and `deriv`).
pub fn FunctionFdf(comptime GF: type) type {
    return struct {
        fdf: GF,
        const Self = @This();

        /// From a pointer to a context struct with `pub fn eval` and
        /// `pub fn deriv` (and optionally `pub fn evalDeriv`, per D-cb3).
        pub fn initCtx(ctx: anytype) Self {
            return .{ .fdf = functionFdf(GF, ctx) };
        }
    };
}

/// Build an `Fdf` (`gsl_function_fdf`-shaped) from a pointer to a context struct
/// exposing `pub fn eval(self, x: f64) f64` and `pub fn deriv(self, x: f64) f64`.
/// The combined `fdf` callback uses the context's own
/// `pub fn evalDeriv(self, x: f64, f: *f64, df: *f64) void` when it declares one,
/// otherwise it is synthesized from `eval` + `deriv` (D-cb3).
pub fn functionFdf(comptime Fdf: type, ctx: anytype) Fdf {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{ "eval", "deriv" });
    const Child = @typeInfo(Ptr).pointer.child;
    const Tr = struct {
        fn f(x: f64, params: ?*anyopaque) callconv(.c) f64 {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            return self.eval(x);
        }
        fn df(x: f64, params: ?*anyopaque) callconv(.c) f64 {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            return self.deriv(x);
        }
        fn fdf(x: f64, params: ?*anyopaque, out_f: [*c]f64, out_df: [*c]f64) callconv(.c) void {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            if (@hasDecl(Child, "evalDeriv")) {
                var fv: f64 = undefined;
                var dv: f64 = undefined;
                self.evalDeriv(x, &fv, &dv);
                out_f.* = fv;
                out_df.* = dv;
            } else {
                out_f.* = self.eval(x);
                out_df.* = self.deriv(x);
            }
        }
    };
    return .{ .f = &Tr.f, .df = &Tr.df, .fdf = &Tr.fdf, .params = @constCast(ctx) };
}

/// A ready-to-pass `gsl_monte_function`-shaped callback value for the
/// multidimensional Monte-Carlo routines. Instantiate once per chapter over its
/// own `c.gsl_monte_function` type; construct with `initFn` (a plain
/// `*const fn([]const f64) f64`) or `initCtx` (a `*struct` with
/// `pub fn eval(self, x: []const f64) f64`). The `dim` field is set to 0 at
/// construction; the consuming routine fills it in (from the integration
/// bounds) before handing `&value.mf` to GSL.
pub fn MonteFunction(comptime MF: type) type {
    return struct {
        mf: MF,
        const Self = @This();

        /// From a plain `*const fn([]const f64) f64` (a bare function coerces in).
        pub fn initFn(f: *const fn ([]const f64) f64) Self {
            return .{ .mf = monteFunction(MF, f) };
        }

        /// From a pointer to a context struct with
        /// `pub fn eval(self, x: []const f64) f64`.
        pub fn initCtx(ctx: anytype) Self {
            return .{ .mf = monteContext(MF, ctx) };
        }
    };
}

/// Build an `MF` (`gsl_monte_function`-shaped) from a plain function pointer
/// taking the point as a slice. `dim` is left 0 for the consumer to fill in.
pub fn monteFunction(comptime MF: type, f: *const fn ([]const f64) f64) MF {
    const Tr = struct {
        fn call(x: [*c]f64, dim: usize, params: ?*anyopaque) callconv(.c) f64 {
            const fp: *const fn ([]const f64) f64 = @ptrCast(@alignCast(params.?));
            return fp(x[0..dim]);
        }
    };
    return .{ .f = &Tr.call, .dim = 0, .params = @constCast(f) };
}

/// Build an `MF` (`gsl_monte_function`-shaped) from a pointer to a context
/// struct exposing `pub fn eval(self, x: []const f64) f64`. `dim` is left 0 for
/// the consumer to fill in.
pub fn monteContext(comptime MF: type, ctx: anytype) MF {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{"eval"});
    const Tr = struct {
        fn call(x: [*c]f64, dim: usize, params: ?*anyopaque) callconv(.c) f64 {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            return self.eval(x[0..dim]);
        }
    };
    return .{ .f = &Tr.call, .dim = 0, .params = @constCast(ctx) };
}

/// Build a `Sys` (`gsl_odeiv2_system`-shaped) from a pointer to a context
/// struct exposing `pub fn rhs(self, t, y, dydt)`, where `y`/`dydt` are raw
/// C pointers (`[*c]const f64` / `[*c]f64`) matching GSL's callback signature.
/// `rhs` may return `void` (treated as `GSL_SUCCESS`) or a `c_int` status code.
pub fn odeSystem(comptime Sys: type, dimension: usize, ctx: anytype) Sys {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{"rhs"});
    const Tr = struct {
        fn f(t: f64, y: [*c]const f64, dydt: [*c]f64, params: ?*anyopaque) callconv(.c) c_int {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            const ret = self.rhs(t, y, dydt);
            return odeStatus(ret);
        }
    };
    return .{ .function = &Tr.f, .jacobian = null, .dimension = dimension, .params = @constCast(ctx) };
}

/// Build a `Sys` (`gsl_odeiv2_system`-shaped) from a pointer to a context
/// struct exposing both `pub fn rhs(self, t, y, dydt)` and
/// `pub fn jacobian(self, t, y, dfdy, dfdt)`, where all array arguments are raw
/// C pointers matching GSL's callback signature. Both methods may return `void`
/// (treated as `GSL_SUCCESS`) or a `c_int` status code.
pub fn odeSystemWithJacobian(comptime Sys: type, dimension: usize, ctx: anytype) Sys {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{ "rhs", "jacobian" });
    const Tr = struct {
        fn f(t: f64, y: [*c]const f64, dydt: [*c]f64, params: ?*anyopaque) callconv(.c) c_int {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            const ret = self.rhs(t, y, dydt);
            return odeStatus(ret);
        }
        fn jac(t: f64, y: [*c]const f64, dfdy: [*c]f64, dfdt: [*c]f64, params: ?*anyopaque) callconv(.c) c_int {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            const ret = self.jacobian(t, y, dfdy, dfdt);
            return odeStatus(ret);
        }
    };
    return .{ .function = &Tr.f, .jacobian = &Tr.jac, .dimension = dimension, .params = @constCast(ctx) };
}

fn odeStatus(ret: anytype) c_int {
    const Ret = @TypeOf(ret);
    if (Ret == void) return 0;
    if (Ret == c_int) return ret;
    @compileError("ODE callback methods must return void or c_int; got " ++ @typeName(Ret));
}

// ---------------------------------------------------------------------------
// gsl_multifit_nlinear — residual (+ optional Jacobian) callback bundle
// ---------------------------------------------------------------------------
//
// The `gsl_multifit_nlinear_fdf` callback struct is shaped
//   f  (const gsl_vector *x, void *params, gsl_vector *f)   // residual
//   df (const gsl_vector *x, void *params, gsl_matrix *J)   // Jacobian (optional)
//   fvv(...)                                                // geodesic accel (unused)
// Unlike the ODE callbacks (raw `double[]`), these pass `gsl_vector *`/
// `gsl_matrix *`, so the trampolines are generic over the chapter's `@cImport`
// `Vec`/`Mat` types.
//
// The parameter vector `x` is always contiguous (a plain workspace/allocation),
// so it is presented as a `[]const f64` slice. The residual output `f`, however,
// is *not* always contiguous: when GSL approximates the Jacobian by finite
// differences it evaluates the residual directly into a strided *column view* of
// the Jacobian (`stride == tda`). It is therefore presented as a
// `gsl.StridedMut(f64)` (write with `f.set(i, v)`), and the Jacobian as a
// row-major `gsl.MatrixMut(f64)` (`J.set(i, j, v)`). Both are zero-copy.

/// Build a `Fdf` (`gsl_multifit_nlinear_fdf`-shaped) from a context struct
/// exposing `pub fn residual(self, x: []const f64, f: gsl.StridedMut(f64))`. The
/// Jacobian pointer is left `null`, so GSL approximates it by finite
/// differences. `residual` may return `void` (treated as `GSL_SUCCESS`) or a
/// `c_int` status.
pub fn multifitFdf(comptime Fdf: type, comptime Vec: type, n: usize, p: usize, ctx: anytype) Fdf {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{"residual"});
    const Tr = struct {
        fn f(x: [*c]const Vec, params: ?*anyopaque, out: [*c]Vec) callconv(.c) c_int {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            return multifitStatus(self.residual(paramSlice(x), residualView(out)));
        }
    };
    return multifitStruct(Fdf, &Tr.f, null, n, p, ctx);
}

/// Build a `Fdf` (`gsl_multifit_nlinear_fdf`-shaped) from a context struct
/// exposing both `pub fn residual(self, x: []const f64, f: gsl.StridedMut(f64))`
/// and `pub fn jacobian(self, x: []const f64, J: gsl.MatrixMut(f64))`. Both
/// methods may return `void` (treated as `GSL_SUCCESS`) or a `c_int` status.
pub fn multifitFdfWithJacobian(comptime Fdf: type, comptime Vec: type, comptime Mat: type, n: usize, p: usize, ctx: anytype) Fdf {
    const Ptr = @TypeOf(ctx);
    comptime requireMethods(Ptr, &.{ "residual", "jacobian" });
    const Tr = struct {
        fn f(x: [*c]const Vec, params: ?*anyopaque, out: [*c]Vec) callconv(.c) c_int {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            return multifitStatus(self.residual(paramSlice(x), residualView(out)));
        }
        fn df(x: [*c]const Vec, params: ?*anyopaque, jac: [*c]Mat) callconv(.c) c_int {
            const self: Ptr = @ptrCast(@alignCast(params.?));
            const J = gsl.MatrixMut(f64).init(@ptrCast(jac.*.data), jac.*.size1, jac.*.size2, jac.*.tda);
            return multifitStatus(self.jacobian(paramSlice(x), J));
        }
    };
    return multifitStruct(Fdf, &Tr.f, &Tr.df, n, p, ctx);
}

/// Assemble the `gsl_multifit_nlinear_fdf` struct literal, zeroing the
/// evaluation counters GSL maintains.
fn multifitStruct(comptime Fdf: type, f: anytype, df: anytype, n: usize, p: usize, ctx: anytype) Fdf {
    return .{
        .f = f,
        .df = df,
        .fvv = null,
        .n = n,
        .p = p,
        .params = @constCast(ctx),
        .nevalf = 0,
        .nevaldf = 0,
        .nevalfvv = 0,
    };
}

/// Present a contiguous GSL parameter vector as a `[]const f64` (debug-asserts
/// stride 1, which always holds for `gsl_multifit_nlinear`'s `x`).
fn paramSlice(v: anytype) []const f64 {
    std.debug.assert(v.*.stride == 1);
    return v.*.data[0..v.*.size];
}

/// Present a (possibly strided) GSL residual vector as a `gsl.StridedMut(f64)`.
fn residualView(v: anytype) gsl.StridedMut(f64) {
    return gsl.StridedMut(f64).init(@ptrCast(v.*.data), v.*.stride, v.*.size);
}

fn multifitStatus(ret: anytype) c_int {
    const Ret = @TypeOf(ret);
    if (Ret == void) return 0;
    if (Ret == c_int) return ret;
    @compileError("multifit callback methods must return void or c_int; got " ++ @typeName(Ret));
}

/// Comptime guard: `Ptr` must be a single-item pointer to a struct declaring
/// each named (public) method. Produces a targeted error otherwise.
fn requireMethods(comptime Ptr: type, comptime methods: []const []const u8) void {
    const info = @typeInfo(Ptr);
    if (info != .pointer or info.pointer.size != .one or
        @typeInfo(info.pointer.child) != .@"struct")
    {
        @compileError("callback context must be a pointer to a struct; got " ++ @typeName(Ptr));
    }
    const Child = info.pointer.child;
    inline for (methods) |m| {
        if (!@hasDecl(Child, m)) {
            @compileError(@typeName(Child) ++ " must declare `pub fn " ++ m ++ "` to be used as a callback context");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — validated against mock structs matching the C layout, so the bridge
// is exercised in isolation (no GSL dependency here; end-to-end validation
// lives in the consuming chapters, e.g. gsl_deriv.zig).
// ---------------------------------------------------------------------------

const MockFn = extern struct {
    function: ?*const fn (f64, ?*anyopaque) callconv(.c) f64,
    params: ?*anyopaque,
};

const MockFdf = extern struct {
    f: ?*const fn (f64, ?*anyopaque) callconv(.c) f64,
    df: ?*const fn (f64, ?*anyopaque) callconv(.c) f64,
    fdf: ?*const fn (f64, ?*anyopaque, [*c]f64, [*c]f64) callconv(.c) void,
    params: ?*anyopaque,
};

const MockMonte = extern struct {
    f: ?*const fn ([*c]f64, usize, ?*anyopaque) callconv(.c) f64,
    dim: usize,
    params: ?*anyopaque,
};

const MockOde = extern struct {
    function: ?*const fn (f64, [*c]const f64, [*c]f64, ?*anyopaque) callconv(.c) c_int,
    jacobian: ?*const fn (f64, [*c]const f64, [*c]f64, [*c]f64, ?*anyopaque) callconv(.c) c_int,
    dimension: usize,
    params: ?*anyopaque,
};

// Mock `gsl_vector`/`gsl_matrix`/`gsl_multifit_nlinear_fdf` matching the C
// layouts so the multifit bridge can be exercised without linking GSL.
const MockVec = extern struct {
    size: usize,
    stride: usize,
    data: [*c]f64,
    block: ?*anyopaque,
    owner: c_int,
};

const MockMat = extern struct {
    size1: usize,
    size2: usize,
    tda: usize,
    data: [*c]f64,
    block: ?*anyopaque,
    owner: c_int,
};

const MockMultifit = extern struct {
    f: ?*const fn ([*c]const MockVec, ?*anyopaque, [*c]MockVec) callconv(.c) c_int,
    df: ?*const fn ([*c]const MockVec, ?*anyopaque, [*c]MockMat) callconv(.c) c_int,
    fvv: ?*anyopaque,
    n: usize,
    p: usize,
    params: ?*anyopaque,
    nevalf: usize,
    nevaldf: usize,
    nevalfvv: usize,
};

fn square(x: f64) f64 {
    return x * x;
}

test "callback: function accepts a bare fn (coerced to a pointer)" {
    const m = function(MockFn, square);
    try testing.expect(m.params != null);
    try testing.expectEqual(@as(f64, 9.0), m.function.?(3.0, m.params));
}

test "callback: function accepts an explicit fn pointer" {
    const fp: *const fn (f64) f64 = &square;
    const m = function(MockFn, fp);
    try testing.expectEqual(@as(f64, 16.0), m.function.?(4.0, m.params));
}

test "callback: context captures state read through the live pointer" {
    const Wave = struct {
        k: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            return self.k * x;
        }
    };
    var w = Wave{ .k = 2.5 };
    const m = context(MockFn, &w);
    try testing.expectEqual(@as(f64, 10.0), m.function.?(4.0, m.params));
    w.k = 3.0;
    try testing.expectEqual(@as(f64, 12.0), m.function.?(4.0, m.params));
}

test "callback: Function value builds from initFn and initCtx" {
    const CB = Function(MockFn);
    const Ctx = struct {
        k: f64,
        pub fn eval(self: *const @This(), x: f64) f64 {
            return self.k + x;
        }
    };

    const a: CB = .initFn(square); // decl-literal resolves against the param type
    try testing.expectEqual(@as(f64, 9.0), a.gf.function.?(3.0, a.gf.params));

    var ctx = Ctx{ .k = 1.0 };
    const b: CB = .initCtx(&ctx);
    try testing.expectEqual(@as(f64, 5.0), b.gf.function.?(4.0, b.gf.params));
}

test "callback: functionFdf synthesizes fdf from eval + deriv" {
    const Poly = struct {
        pub fn eval(_: *const @This(), x: f64) f64 {
            return x * x - 2;
        }
        pub fn deriv(_: *const @This(), x: f64) f64 {
            return 2 * x;
        }
    };
    var p = Poly{};
    const m = functionFdf(MockFdf, &p);
    try testing.expectEqual(@as(f64, 2.0), m.f.?(2.0, m.params)); // 4 - 2
    try testing.expectEqual(@as(f64, 4.0), m.df.?(2.0, m.params)); // 2*2
    var fv: f64 = undefined;
    var dv: f64 = undefined;
    m.fdf.?(2.0, m.params, &fv, &dv);
    try testing.expectEqual(@as(f64, 2.0), fv);
    try testing.expectEqual(@as(f64, 4.0), dv);
}

test "callback: functionFdf prefers a context-supplied evalDeriv" {
    const Fused = struct {
        calls: *usize,
        pub fn eval(_: *const @This(), x: f64) f64 {
            return x;
        }
        pub fn deriv(_: *const @This(), x: f64) f64 {
            _ = x;
            return 1;
        }
        pub fn evalDeriv(self: *const @This(), x: f64, f: *f64, df: *f64) void {
            self.calls.* += 1;
            f.* = x * 10;
            df.* = 100;
        }
    };
    var n: usize = 0;
    var fused = Fused{ .calls = &n };
    const m = functionFdf(MockFdf, &fused);
    var fv: f64 = undefined;
    var dv: f64 = undefined;
    m.fdf.?(3.0, m.params, &fv, &dv);
    try testing.expectEqual(@as(usize, 1), n); // fused path taken
    try testing.expectEqual(@as(f64, 30.0), fv);
    try testing.expectEqual(@as(f64, 100.0), dv);
}

test "callback: MonteFunction reconstructs the point slice for a plain fn" {
    const dot = struct {
        fn f(x: []const f64) f64 {
            var s: f64 = 0;
            for (x) |xi| s += xi * xi;
            return s;
        }
    }.f;
    const CB = MonteFunction(MockMonte);
    var cb: CB = .initFn(dot);
    cb.mf.dim = 3; // the consuming chapter fills this in before the call
    var pt = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(@as(f64, 14.0), cb.mf.f.?(&pt, cb.mf.dim, cb.mf.params));
}

test "callback: MonteFunction context captures state and reads the slice" {
    const Weighted = struct {
        w: []const f64,
        pub fn eval(self: *const @This(), x: []const f64) f64 {
            var s: f64 = 0;
            for (x, self.w) |xi, wi| s += wi * xi;
            return s;
        }
    };
    var wv = [_]f64{ 10.0, 100.0 };
    var ctx = Weighted{ .w = &wv };
    const CB = MonteFunction(MockMonte);
    var cb: CB = .initCtx(&ctx);
    cb.mf.dim = 2;
    var pt = [_]f64{ 3.0, 4.0 };
    try testing.expectEqual(@as(f64, 430.0), cb.mf.f.?(&pt, cb.mf.dim, cb.mf.params));
}

test "callback: odeSystem builds rhs-only gsl_odeiv2_system" {
    const Decay = struct {
        k: f64,
        pub fn rhs(self: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) void {
            _ = t;
            dydt[0] = -self.k * y[0];
        }
    };

    var d = Decay{ .k = 2.0 };
    const s = odeSystem(MockOde, 1, &d);
    try testing.expect(s.jacobian == null);
    try testing.expectEqual(@as(usize, 1), s.dimension);

    const y = [_]f64{3.0};
    var dydt = [_]f64{0.0};
    try testing.expectEqual(@as(c_int, 0), s.function.?(0.0, &y, &dydt, s.params));
    try testing.expectEqual(@as(f64, -6.0), dydt[0]);
}

test "callback: odeSystemWithJacobian wires both callbacks" {
    const Linear = struct {
        a: f64,
        pub fn rhs(self: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) c_int {
            _ = t;
            dydt[0] = self.a * y[0];
            return 0;
        }
        pub fn jacobian(self: *const @This(), t: f64, y: [*c]const f64, dfdy: [*c]f64, dfdt: [*c]f64) void {
            _ = t;
            _ = y;
            dfdy[0] = self.a;
            dfdt[0] = 0.0;
        }
    };

    var lin = Linear{ .a = -4.0 };
    const s = odeSystemWithJacobian(MockOde, 1, &lin);
    try testing.expect(s.jacobian != null);

    const y = [_]f64{1.5};
    var dydt = [_]f64{0.0};
    var dfdy = [_]f64{0.0};
    var dfdt = [_]f64{1.0};

    try testing.expectEqual(@as(c_int, 0), s.function.?(2.0, &y, &dydt, s.params));
    try testing.expectEqual(@as(c_int, 0), s.jacobian.?(2.0, &y, &dfdy, &dfdt, s.params));
    try testing.expectEqual(@as(f64, -6.0), dydt[0]);
    try testing.expectEqual(@as(f64, -4.0), dfdy[0]);
    try testing.expectEqual(@as(f64, 0.0), dfdt[0]);
}

test "callback: odeSystem forwards a nonzero c_int status" {
    const Fail = struct {
        pub fn rhs(_: *const @This(), t: f64, y: [*c]const f64, dydt: [*c]f64) c_int {
            _ = t;
            _ = y;
            _ = dydt;
            return 1234;
        }
    };

    var f = Fail{};
    const s = odeSystem(MockOde, 1, &f);
    const y = [_]f64{0.0};
    var dydt = [_]f64{0.0};
    try testing.expectEqual(@as(c_int, 1234), s.function.?(0.0, &y, &dydt, s.params));
}

test "callback: multifitFdf wires residual only, leaving df null" {
    // Model r_i(a) = a*t_i - y_i for two points; residual only.
    const Fit = struct {
        t: []const f64,
        y: []const f64,
        pub fn residual(self: *const @This(), x: []const f64, r: gsl.StridedMut(f64)) void {
            for (self.t, self.y, 0..) |ti, yi, i| r.set(i, x[0] * ti + x[1] - yi);
        }
    };
    var t = [_]f64{ 1.0, 2.0 };
    var y = [_]f64{ 3.0, 5.0 };
    var fit = Fit{ .t = &t, .y = &y };

    const fdf = multifitFdf(MockMultifit, MockVec, 2, 2, &fit);
    try testing.expect(fdf.df == null);
    try testing.expectEqual(@as(usize, 2), fdf.n);
    try testing.expectEqual(@as(usize, 2), fdf.p);

    var xdata = [_]f64{ 2.0, 1.0 }; // a=2, b=1
    var xvec = MockVec{ .size = 2, .stride = 1, .data = &xdata, .block = null, .owner = 0 };
    var rdata = [_]f64{ 0.0, 0.0 };
    var rvec = MockVec{ .size = 2, .stride = 1, .data = &rdata, .block = null, .owner = 0 };

    try testing.expectEqual(@as(c_int, 0), fdf.f.?(&xvec, fdf.params, &rvec));
    try testing.expectEqual(@as(f64, 0.0), rdata[0]); // 2*1+1-3
    try testing.expectEqual(@as(f64, 0.0), rdata[1]); // 2*2+1-5
}

test "callback: multifitFdfWithJacobian fills a tda-aware Jacobian" {
    // r_i = a*t_i + b - y_i ; dr_i/da = t_i, dr_i/db = 1.
    const Fit = struct {
        t: []const f64,
        y: []const f64,
        pub fn residual(self: *const @This(), x: []const f64, r: gsl.StridedMut(f64)) c_int {
            for (self.t, self.y, 0..) |ti, yi, i| r.set(i, x[0] * ti + x[1] - yi);
            return 0;
        }
        pub fn jacobian(self: *const @This(), x: []const f64, J: gsl.MatrixMut(f64)) void {
            _ = x;
            for (self.t, 0..) |ti, i| {
                J.set(i, 0, ti);
                J.set(i, 1, 1.0);
            }
        }
    };
    var t = [_]f64{ 1.0, 2.0, 3.0 };
    var y = [_]f64{ 0.0, 0.0, 0.0 };
    var fit = Fit{ .t = &t, .y = &y };

    const fdf = multifitFdfWithJacobian(MockMultifit, MockVec, MockMat, 3, 2, &fit);
    try testing.expect(fdf.df != null);

    var xdata = [_]f64{ 0.0, 0.0 };
    var xvec = MockVec{ .size = 2, .stride = 1, .data = &xdata, .block = null, .owner = 0 };

    // Give the Jacobian a padded leading dimension (tda=4 > cols=2) to prove
    // the row stride is honoured.
    var jdata = [_]f64{-1.0} ** 12; // 3 rows x tda 4
    var jmat = MockMat{ .size1 = 3, .size2 = 2, .tda = 4, .data = &jdata, .block = null, .owner = 0 };

    try testing.expectEqual(@as(c_int, 0), fdf.df.?(&xvec, fdf.params, &jmat));
    // Row i at jdata[i*4 + j].
    try testing.expectEqual(@as(f64, 1.0), jdata[0]); // (0,0)=t0
    try testing.expectEqual(@as(f64, 1.0), jdata[1]); // (0,1)=1
    try testing.expectEqual(@as(f64, 2.0), jdata[4]); // (1,0)=t1
    try testing.expectEqual(@as(f64, 1.0), jdata[5]); // (1,1)=1
    try testing.expectEqual(@as(f64, 3.0), jdata[8]); // (2,0)=t2
    try testing.expectEqual(@as(f64, 1.0), jdata[9]); // (2,1)=1
    // Padding column untouched.
    try testing.expectEqual(@as(f64, -1.0), jdata[2]);
}

test "callback: multifit residual forwards a nonzero c_int status" {
    const Fail = struct {
        pub fn residual(_: *const @This(), x: []const f64, r: gsl.StridedMut(f64)) c_int {
            _ = x;
            _ = r;
            return 7;
        }
    };
    var fail = Fail{};
    const fdf = multifitFdf(MockMultifit, MockVec, 1, 1, &fail);
    var xdata = [_]f64{0.0};
    var xvec = MockVec{ .size = 1, .stride = 1, .data = &xdata, .block = null, .owner = 0 };
    var rdata = [_]f64{0.0};
    var rvec = MockVec{ .size = 1, .stride = 1, .data = &rdata, .block = null, .owner = 0 };
    try testing.expectEqual(@as(c_int, 7), fdf.f.?(&xvec, fdf.params, &rvec));
}
