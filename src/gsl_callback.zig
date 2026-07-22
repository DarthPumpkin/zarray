//! The callback bridge: turn an idiomatic Zig callable into the C
//! function-pointer-plus-`void *params` structs GSL's callback chapters expect
//! (`gsl_function`, `gsl_function_fdf`, and — later — the monte/ODE variants).
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
