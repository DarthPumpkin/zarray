//! Accelerate-specific BLAS extension entry points.
//!
//! These symbols are intentionally kept out of `bindings.blas` so the core BLAS
//! surface remains portable across CBLAS implementations.
//!
//! Notes:
//! - `csrot`/`zdrot` are exposed as a higher-level `rot_complex` operation over
//!   `NamedArray` views.
//! - `crotg`/`zrotg` are intentionally not surfaced here: they generate a
//!   complex-parameter Givens rotation, but this module intentionally does not
//!   provide the corresponding complex-`s` apply routine.

const std = @import("std");
const assert = std.debug.assert;
const Complex = std.math.Complex;

const named_array = @import("../../named_array.zig");
const NamedArray = named_array.NamedArray;
const core = @import("../blas.zig");

const acc = @cImport({
    @cInclude("vecLib/cblas.h");
});

/// `cblas_errprn` in CBLAS (ATLAS/Accelerate extension).
pub fn errprn(ierr: c_int, info: c_int, form: [*:0]const u8) c_int {
    return acc.cblas_errprn(ierr, info, @constCast(form));
}

/// `cblas_xerbla` in CBLAS (ATLAS/Accelerate extension).
pub fn xerbla(p: c_int, rout: [*:0]const u8, form: [*:0]const u8) void {
    acc.cblas_xerbla(p, @constCast(rout), @constCast(form));
}

/// `cblas_csrot` / `cblas_zdrot` (ATLAS/Accelerate extensions), exposed as a
/// `NamedArray` operation.
pub fn rot_complex(
    comptime RealScalar: type,
    rot: core.GivensRotationReal(RealScalar),
    points: NamedArray(core.IJ, Complex(RealScalar)),
) void {
    const f = switch (RealScalar) {
        f32 => acc.cblas_csrot,
        f64 => acc.cblas_zdrot,
        else => @compileError("rot_complex is incompatible with given RealScalar type."),
    };

    assert(points.idx.shape.j == 2);
    const n_usize: usize = points.idx.shape.i;
    if (n_usize == 0) return;

    const n: c_int = @intCast(n_usize);
    const inc: c_int = @intCast(points.idx.strides.i);
    const x = points.at(.{ .i = 0, .j = 0 });
    const y = points.at(.{ .i = 0, .j = 1 });

    f(n, @ptrCast(x), inc, @ptrCast(y), inc, rot.c, rot.s);
}

test "rot_complex 90-degree swap" {
    const T = f64;
    const rot = core.GivensRotationReal(T){ .c = 0.0, .s = 1.0 };

    var points_buf = [_]Complex(T){
        .{ .re = 2.0, .im = -3.0 }, .{ .re = -5.0, .im = 7.0 },
        .{ .re = 1.0, .im = 4.0 },  .{ .re = 6.0, .im = -8.0 },
    };
    const points = NamedArray(core.IJ, Complex(T)).init(.initContiguous(.{ .i = 2, .j = 2 }), &points_buf);

    rot_complex(T, rot, points);

    const eps: T = 1e-12;
    try std.testing.expectApproxEqAbs(@as(T, -5.0), points_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(T, 7.0), points_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(T, -2.0), points_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(T, 3.0), points_buf[1].im, eps);

    try std.testing.expectApproxEqAbs(@as(T, 6.0), points_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(T, -8.0), points_buf[2].im, eps);
    try std.testing.expectApproxEqAbs(@as(T, -1.0), points_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(T, -4.0), points_buf[3].im, eps);
}

test "accelerate blas_ext" {
    std.testing.refAllDecls(@This());
}
