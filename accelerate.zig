const std = @import("std");
const math = std.math;
const meta = std.meta;
const assert = std.debug.assert;
const Complex = math.complex.Complex;

const named_index = @import("named_index.zig");
const named_array = @import("named_array.zig");
const NamedIndex = named_index.NamedIndex;
const NamedArray = named_array.NamedArray;
const NamedArrayConst = named_array.NamedArrayConst;

const acc = @cImport(@cInclude("Accelerate/Accelerate.h"));

pub const blas = struct {
    // TODO: sdsdot, dsdot (internally double precision)
    pub fn dot(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArrayConst(Axis, Scalar),
    ) Scalar {
        const cblas_dot = switch (Scalar) {
            f32 => acc.cblas_sdot,
            f64 => acc.cblas_ddot,
            else => @compileError("dot is incompatible with given Scalar type."),
        };

        const x_blas = Blas1d(Scalar).init(Axis, x);
        const y_blas = Blas1d(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        return cblas_dot(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
    }

    pub fn dotu(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArrayConst(Axis, Scalar),
    ) Scalar {
        const cblas_dotu_sub = switch (Scalar) {
            Complex(f32) => acc.cblas_cdotu_sub,
            Complex(f64) => acc.cblas_zdotu_sub,
            else => @compileError("dotu is incompatible with given Scalar type."),
        };

        const x_blas = Blas1d(Scalar).init(Axis, x);
        const y_blas = Blas1d(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        var result: Scalar = .{ .re = 0, .im = 0 };
        cblas_dotu_sub(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc, &result);
        return result;
    }

    pub fn dotc(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArrayConst(Axis, Scalar),
    ) Scalar {
        const cblas_dotc_sub = switch (Scalar) {
            Complex(f32) => acc.cblas_cdotc_sub,
            Complex(f64) => acc.cblas_zdotc_sub,
            else => @compileError("dotc is incompatible with given Scalar type."),
        };

        const x_blas = Blas1d(Scalar).init(Axis, x);
        const y_blas = Blas1d(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        var result: Scalar = .{ .re = 0, .im = 0 };
        cblas_dotc_sub(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc, &result);
        return result;
    }

    fn Blas1d(comptime Scalar: type) type {
        return struct {
            len: c_int,
            ptr: *const Scalar,
            inc: c_int,

            fn init(comptime Axis: type, arr: anytype) @This() {
                const axis_name = comptime blk: {
                    const fields = meta.fields(Axis);
                    assert(fields.len == 1);
                    break :blk fields[0].name;
                };

                const len = @field(arr.idx.shape, axis_name);
                const inc = @field(arr.idx.strides, axis_name);
                // The pointer is expected to be to the scalar that comes first in virtual memory.
                // For negative strides, this corresponds to the logically last scalar.
                const ptr: *const Scalar = if (inc >= 0) arr.at(@bitCast([_]usize{0})) else arr.at(@bitCast([_]usize{len - 1}));

                return .{
                    .len = @intCast(len),
                    .ptr = ptr,
                    .inc = @intCast(inc),
                };
            }
        };
    }
};

test "dot" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArrayConst(I, T);

    var x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{ 2.0, 3.0, 5.0 },
    };
    x.idx = x.idx.stride(.{ .i = -1 });
    const y = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{ 1.0, 10.0, 100.0 },
    };

    const expected: T = 235.0;
    const actual = blas.dot(I, T, x, y);
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(T, expected),
    );
}

test "dotu" {
    const I = enum { i };
    const T = Complex(f32);
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 1.0, .im = 2.0 },
            .{ .re = 3.0, .im = 4.0 },
        },
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 5.0, .im = 6.0 },
            .{ .re = 7.0, .im = 8.0 },
        },
    };

    const expected: T = .{ .re = -18.0, .im = 68.0 };
    const actual = blas.dotu(I, T, x, y);
    try std.testing.expectApproxEqAbs(
        expected.re,
        actual.re,
        math.floatEpsAt(f32, expected.re),
    );
    try std.testing.expectApproxEqAbs(
        expected.im,
        actual.im,
        math.floatEpsAt(f32, expected.im),
    );
}

test "dotc" {
    const I = enum { i };
    const T = Complex(f32);
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 1.0, .im = 2.0 },
            .{ .re = 3.0, .im = 4.0 },
        },
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 5.0, .im = 6.0 },
            .{ .re = 7.0, .im = 8.0 },
        },
    };

    const expected: T = .{ .re = 70.0, .im = -8.0 };
    const actual = blas.dotc(I, T, x, y);
    try std.testing.expectApproxEqAbs(
        expected.re,
        actual.re,
        math.floatEpsAt(f32, expected.re),
    );
    try std.testing.expectApproxEqAbs(
        expected.im,
        actual.im,
        math.floatEpsAt(f32, expected.im),
    );
}
