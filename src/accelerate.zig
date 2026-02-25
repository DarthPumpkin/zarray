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
    // TODO: all the rot* functions
    pub fn dot(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArrayConst(Axis, Scalar),
        comptime config: struct { internal_double_precision: bool = false },
    ) Scalar {
        const cblas_dot = switch (Scalar) {
            f32 => if (config.internal_double_precision) acc.cblas_sdsdot else acc.cblas_sdot,
            f64 => if (config.internal_double_precision) acc.cblas_dsdot else acc.cblas_ddot,
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

    pub fn nrm2(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
    ) switch (Scalar) {
        f32 => f32,
        f64 => f64,
        Complex(f32) => f32,
        Complex(f64) => f64,
        else => @compileError("nrm2 is incompatible with given Scalar type."),
    } {
        const x_blas = Blas1d(Scalar).init(Axis, x);
        const f = switch (Scalar) {
            f32 => acc.cblas_snrm2,
            f64 => acc.cblas_dnrm2,
            Complex(f32) => acc.cblas_scnrm2,
            Complex(f64) => acc.cblas_dznrm2,
            else => unreachable,
        };
        return f(x_blas.len, x_blas.ptr, x_blas.inc);
    }

    pub fn asum(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
    ) switch (Scalar) {
        f32 => f32,
        f64 => f64,
        Complex(f32) => f32,
        Complex(f64) => f64,
        else => @compileError("asum is incompatible with given Scalar type."),
    } {
        const x_blas = Blas1d(Scalar).init(Axis, x);
        const f = switch (Scalar) {
            f32 => acc.cblas_sasum,
            f64 => acc.cblas_dasum,
            Complex(f32) => acc.cblas_scasum,
            Complex(f64) => acc.cblas_dzasum,
            else => unreachable,
        };
        return f(x_blas.len, x_blas.ptr, x_blas.inc);
    }

    pub fn i_amax(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
    ) usize {
        const x_blas = Blas1d(Scalar).init(Axis, x);
        const f = switch (Scalar) {
            f32 => acc.cblas_isamax,
            f64 => acc.cblas_idamax,
            Complex(f32) => acc.cblas_icamax,
            Complex(f64) => acc.cblas_izamax,
            else => @compileError("i_amax is incompatible with given Scalar type."),
        };
        const idx: c_int = f(x_blas.len, x_blas.ptr, x_blas.inc);
        return @intCast(idx);
    }

    pub fn swap(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArray(Axis, Scalar),
        y: NamedArray(Axis, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_sswap,
            f64 => acc.cblas_dswap,
            Complex(f32) => acc.cblas_cswap,
            Complex(f64) => acc.cblas_zswap,
            else => @compileError("swap is incompatible with given Scalar type."),
        };

        const x_blas = Blas1dMut(Scalar).init(Axis, x);
        const y_blas = Blas1dMut(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        f(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
    }

    pub fn copy(
        comptime Axis: type,
        comptime Scalar: type,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArray(Axis, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_scopy,
            f64 => acc.cblas_dcopy,
            Complex(f32) => acc.cblas_ccopy,
            Complex(f64) => acc.cblas_zcopy,
            else => @compileError("copy is incompatible with given Scalar type."),
        };

        const x_blas = Blas1d(Scalar).init(Axis, x);
        const y_blas = Blas1dMut(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        f(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
    }

    pub fn axpy(
        comptime Axis: type,
        comptime Scalar: type,
        alpha: Scalar,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArray(Axis, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_saxpy,
            f64 => acc.cblas_daxpy,
            Complex(f32) => acc.cblas_caxpy,
            Complex(f64) => acc.cblas_zaxpy,
            else => @compileError("axpy is incompatible with given Scalar type."),
        };

        const x_blas = Blas1d(Scalar).init(Axis, x);
        const y_blas = Blas1dMut(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        const alpha_blas = if (Scalar == Complex(f32) or Scalar == Complex(f64)) &alpha else alpha;
        f(x_blas.len, alpha_blas, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
    }

    pub fn scal(
        comptime Axis: type,
        comptime VecScalar: type,
        comptime AlphaScalar: type,
        alpha: AlphaScalar,
        x: NamedArray(Axis, VecScalar),
    ) void {
        const f = switch (VecScalar) {
            f32 => switch (AlphaScalar) {
                f32 => acc.cblas_sscal,
                else => @compileError("scal: alpha type must be f32 when vector is f32."),
            },
            f64 => switch (AlphaScalar) {
                f64 => acc.cblas_dscal,
                else => @compileError("scal: alpha type must be f64 when vector is f64."),
            },
            Complex(f32) => switch (AlphaScalar) {
                // Complex vector scaled by real scalar
                f32 => acc.cblas_csscal,
                // Complex vector scaled by complex scalar
                Complex(f32) => acc.cblas_cscal,
                else => @compileError("scal: alpha type must be f32 or Complex(f32) when vector is Complex(f32)."),
            },
            Complex(f64) => switch (AlphaScalar) {
                f64 => acc.cblas_zdscal,
                Complex(f64) => acc.cblas_zscal,
                else => @compileError("scal: alpha type must be f64 or Complex(f64) when vector is Complex(f64)."),
            },
            else => @compileError("scal is incompatible with given vector Scalar type."),
        };

        const x_blas = Blas1dMut(VecScalar).init(Axis, x);
        const alpha_blas = switch (AlphaScalar) {
            f32, f64 => alpha,
            Complex(f32), Complex(f64) => &alpha,
            else => @compileError("scal: unsupported alpha type."),
        };
        f(x_blas.len, alpha_blas, x_blas.ptr, x_blas.inc);
    }

    /// `srotg` and `drotg` in BLAS.
    /// See `rotg_complex` for the complex versions.
    pub fn rotg_real(
        comptime Scalar: type,
        a: *Scalar,
        b: *Scalar,
    ) GivensRotationReal(Scalar) {
        const f = switch (Scalar) {
            f32 => acc.cblas_srotg,
            f64 => acc.cblas_drotg,
            else => @compileError("rotg_real is incompatible with given Scalar type."),
        };
        var rotation: GivensRotationReal(Scalar) = undefined;
        f(a, b, &rotation.c, &rotation.s);
        return rotation;
    }

    /// `crotg` and `zrotg` in BLAS.
    /// See `rotg_real` for the real versions.
    pub fn rotg_complex(
        comptime RealScalar: type,
        a: *Complex(RealScalar),
        b: *Complex(RealScalar),
    ) GivensRotationComplex(RealScalar) {
        const f = switch (RealScalar) {
            f32 => acc.cblas_crotg,
            f64 => acc.cblas_zrotg,
            else => @compileError("rotg_complex is incompatible with given RealScalar type."),
        };
        var rotation: GivensRotationComplex(RealScalar) = undefined;
        f(a, b, &rotation.c, &rotation.s);
        return rotation;
    }

    /// `srot` and `drot` in BLAS. See `rot_complex` for the complex versions.
    pub fn rot_real(
        comptime Scalar: type,
        rot: GivensRotationReal(Scalar),
        points: NamedArray(enum { i, j }, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_srot,
            f64 => acc.cblas_drot,
            else => @compileError("rot_real is incompatible with given Scalar type."),
        };
        const I = enum { i };
        assert(points.idx.shape.j == 2);
        const x_na = NamedArray(I, Scalar).init(
            points.idx.sliceAxis(.j, 0, 1).conformAxes(I),
            points.buf,
        );
        const y_na = NamedArray(I, Scalar).init(
            points.idx.sliceAxis(.j, 1, 2).conformAxes(I),
            points.buf,
        );
        const x_blas = Blas1dMut(Scalar).init(I, x_na);
        const y_blas = Blas1dMut(Scalar).init(I, y_na);
        f(
            x_blas.len,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            &rot.c,
            &rot.s,
        );
    }

    /// `crot` and `zrot` in BLAS. See `rot_real` for the real versions.
    pub fn rot_complex(
        comptime RealScalar: type,
        rot: GivensRotationComplex(RealScalar),
        points: NamedArray(enum { i, j }, Complex(RealScalar)),
    ) void {
        const f = switch (RealScalar) {
            f32 => acc.cblas_crot,
            f64 => acc.cblas_zrot,
            else => @compileError("rot_complex is incompatible with given RealScalar type."),
        };
        const I = enum { i };
        assert(points.idx.shape.j == 2);
        const x_na = NamedArray(I, Complex(RealScalar)).init(
            points.idx.sliceAxis(.j, 0, 1).conformAxes(I),
            points.buf,
        );
        const y_na = NamedArray(I, Complex(RealScalar)).init(
            points.idx.sliceAxis(.j, 1, 2).conformAxes(I),
            points.buf,
        );
        const x_blas = Blas1dMut(Complex(RealScalar)).init(I, x_na);
        const y_blas = Blas1dMut(Complex(RealScalar)).init(I, y_na);
        f(
            x_blas.len,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            &rot.c,
            &rot.s,
        );
    }

    pub fn rotmg(
        comptime Scalar: type,
        d1: *Scalar,
        d2: *Scalar,
        a: *Scalar,
        b: Scalar,
    ) ModifiedGivensRotation(Scalar) {
        const f = switch (Scalar) {
            f32 => acc.cblas_srotmg,
            f64 => acc.cblas_drotmg,
            else => @compileError("rotmg is incompatible with given Scalar type."),
        };
        var rotation: ModifiedGivensRotation(Scalar) = undefined;
        f(d1, d2, a, b, &rotation.data);
        return rotation;
    }

    pub fn rotm(
        comptime Scalar: type,
        rot: ModifiedGivensRotation(Scalar),
        points: NamedArray(enum { i, j }, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_srotm,
            f64 => acc.cblas_drotm,
            else => @compileError("rotm is incompatible with given Scalar type."),
        };
        const I = enum { i };
        assert(points.idx.shape.j == 2);
        const x_na = NamedArray(I, Scalar).init(
            points.idx.sliceAxis(.j, 0, 1).conformAxes(I),
            points.buf,
        );
        const y_na = NamedArray(I, Scalar).init(
            points.idx.sliceAxis(.j, 1, 2).conformAxes(I),
            points.buf,
        );
        const x_blas = Blas1dMut(Scalar).init(I, x_na);
        const y_blas = Blas1dMut(Scalar).init(I, y_na);
        f(
            x_blas.len,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            &rot.data,
        );
    }

    pub fn GivensRotationReal(comptime Scalar: type) type {
        return struct {
            c: Scalar,
            s: Scalar,
        };
    }

    pub fn GivensRotationComplex(comptime RealScalar: type) type {
        return struct {
            c: RealScalar,
            s: Complex(RealScalar),
        };
    }

    pub fn ModifiedGivensRotation(comptime Scalar: type) type {
        return struct {
            data: [5]Scalar,

            pub fn flag(self: @This()) MGRFlag {
                return switch (self.data[0]) {
                    -1.0 => MGRFlag.Full,
                    0.0 => MGRFlag.OffDiagonal,
                    1.0 => MGRFlag.Diagonal,
                    2.0 => MGRFlag.Identity,
                    else => @compileError("Invalid flag value in ModifiedGivensRotation data."),
                };
            }

            pub fn fromFlag(flag_: MGRFlag) @This() {
                var data: [5]Scalar = undefined;
                data[0] = switch (flag_) {
                    MGRFlag.Full => -1.0,
                    MGRFlag.OffDiagonal => 0.0,
                    MGRFlag.Diagonal => 1.0,
                    MGRFlag.Identity => 2.0,
                };
                return .{ .data = data };
            }
        };
    }

    pub const MGRFlag = enum { Full, OffDiagonal, Diagonal, Identity };

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

    fn Blas1dMut(comptime Scalar: type) type {
        return struct {
            len: c_int,
            ptr: *Scalar,
            inc: c_int,

            fn init(comptime Axis: type, arr: anytype) @This() {
                const axis_name = comptime blk: {
                    const fields = meta.fields(Axis);
                    assert(fields.len == 1);
                    break :blk fields[0].name;
                };

                const len = @field(arr.idx.shape, axis_name);
                const inc = @field(arr.idx.strides, axis_name);
                const ptr: *Scalar = if (inc >= 0) arr.at(@bitCast([_]usize{0})) else arr.at(@bitCast([_]usize{len - 1}));

                return .{
                    .len = @intCast(len),
                    .ptr = ptr,
                    .inc = @intCast(inc),
                };
            }
        };
    }
};

pub const lapack = struct { @compileError("To do: Implement LAPACK interface") };

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
    const actual = blas.dot(I, T, x, y, .{});
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

test "nrm2 real" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{ 3.0, 4.0 },
    };

    const expected: T = 5.0;
    const actual = blas.nrm2(I, T, x);
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(T, expected),
    );
}

test "nrm2 complex" {
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

    const expected: f32 = math.sqrt(@as(f32, 30.0));
    const actual = blas.nrm2(I, T, x);
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(f32, expected),
    );
}

test "asum real" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{ 2.0, -3.0, 5.0 },
    };

    const expected: T = 10.0;
    const actual = blas.asum(I, T, x);
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(T, expected),
    );
}

test "asum complex" {
    const I = enum { i };
    const T = Complex(f32);
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 1.0, .im = 2.0 },
            .{ .re = -3.0, .im = 4.0 },
        },
    };

    const expected: f32 = 10.0; // |1|+|2| + |−3|+|4|
    const actual = blas.asum(I, T, x);
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(f32, expected),
    );
}

test "i_amax real" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{ 2.0, -3.0, 5.0 },
    };

    const actual = blas.i_amax(I, T, x);
    try std.testing.expectEqual(@as(usize, 2), actual);
}

test "i_amax complex" {
    const I = enum { i };
    const T = Complex(f32);
    const Arr = NamedArrayConst(I, T);

    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{
            .{ .re = 1.0, .im = 2.0 }, // |.| ≈ 2.236
            .{ .re = 3.0, .im = 1.0 }, // |.| ≈ 3.162
            .{ .re = -3.0, .im = 4.0 }, // |.| = 5
        },
    };

    const actual = blas.i_amax(I, T, x);
    try std.testing.expectEqual(@as(usize, 2), actual);
}

test "swap real" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArray(I, T);

    var x_buf = [_]T{ 1.0, 2.0, 3.0 };
    var y_buf = [_]T{ 4.0, 5.0, 6.0 };
    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &x_buf,
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &y_buf,
    };

    blas.swap(I, T, x, y);
    try std.testing.expectEqualSlices(T, &[_]T{ 4.0, 5.0, 6.0 }, x.buf);
    try std.testing.expectEqualSlices(T, &[_]T{ 1.0, 2.0, 3.0 }, y.buf);
}

test "copy complex" {
    const I = enum { i };
    const T = Complex(f32);
    const ArrC = NamedArrayConst(I, T);
    const Arr = NamedArray(I, T);

    var y_buf = [_]T{
        .{ .re = 0.0, .im = 0.0 },
        .{ .re = 0.0, .im = 0.0 },
    };
    const x = ArrC{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 1.0, .im = -2.0 },
            .{ .re = 3.5, .im = 4.0 },
        },
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &y_buf,
    };

    blas.copy(I, T, x, y);
    try std.testing.expectEqualDeep(x.buf[0], y.buf[0]);
    try std.testing.expectEqualDeep(x.buf[1], y.buf[1]);
}

test "axpy real" {
    const I = enum { i };
    const T = f32;
    const ArrC = NamedArrayConst(I, T);
    const Arr = NamedArray(I, T);

    const x_buf = [_]T{ 1.0, -2.0, 3.0 };
    var y_buf = [_]T{ 10.0, 20.0, 30.0 };
    const x = ArrC{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &x_buf,
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &y_buf,
    };

    const alpha: T = 2.0;
    blas.axpy(I, T, alpha, x, y);
    try std.testing.expectEqualSlices(T, &[_]T{ 12.0, 16.0, 36.0 }, y.buf);
}

test "axpy complex" {
    const I = enum { i };
    const T = Complex(f32);
    const ArrC = NamedArrayConst(I, T);
    const Arr = NamedArray(I, T);

    var y_buf = [_]T{
        .{ .re = 5.0, .im = 6.0 },
        .{ .re = 7.0, .im = 8.0 },
    };
    const x = ArrC{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &[_]T{
            .{ .re = 1.0, .im = 2.0 },
            .{ .re = -3.0, .im = 4.0 },
        },
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 2 }),
        .buf = &y_buf,
    };

    const alpha: T = .{ .re = 2.0, .im = -1.0 };
    blas.axpy(I, T, alpha, x, y);
    // Manually compute expected:
    // y0 + alpha*x0 = (5+6i) + (2-i)*(1+2i) = (5+6i) + (2+4i - i -2i^2) = (5+6i) + (4 + 3i) = (9 + 9i)
    // y1 + alpha*x1 = (7+8i) + (2-i)*(-3+4i) = (7+8i) + (-6+8i +3i -4i^2) = (7+8i) + (-2 +11i) = (5 + 19i)
    try std.testing.expectApproxEqAbs(9.0, y.buf[0].re, math.floatEpsAt(f32, 9.0));
    try std.testing.expectApproxEqAbs(9.0, y.buf[0].im, math.floatEpsAt(f32, 9.0));
    try std.testing.expectApproxEqAbs(5.0, y.buf[1].re, math.floatEpsAt(f32, 5.0));
    try std.testing.expectApproxEqAbs(19.0, y.buf[1].im, math.floatEpsAt(f32, 19.0));
}

test "scal real" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArray(I, T);

    var buf_x: [4]T = .{ 1.0, -2.0, 3.0, -4.0 };
    const x = Arr{
        .idx = .initContiguous(.{ .i = 4 }),
        .buf = &buf_x,
    };

    const alpha: T = 2.5;
    blas.scal(I, T, T, alpha, x);

    const expected: [4]T = .{ 2.5, -5.0, 7.5, -10.0 };
    try std.testing.expectEqualSlices(T, expected[0..], x.buf);
}

test "scal complex with real alpha" {
    const I = enum { i };
    const T = Complex(f32);
    const Arr = NamedArray(I, T);

    var buf_x: [3]T = .{
        .{ .re = 1.0, .im = 2.0 },
        .{ .re = -3.0, .im = 4.0 },
        .{ .re = 0.5, .im = -1.5 },
    };
    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &buf_x,
    };

    const alpha: f32 = 2.0;
    // Use csscal path: complex vector scaled by real alpha
    blas.scal(I, T, f32, alpha, x);

    // Expected: element-wise 2 * x
    try std.testing.expectApproxEqAbs(2.0, x.buf[0].re, math.floatEpsAt(f32, 2.0));
    try std.testing.expectApproxEqAbs(4.0, x.buf[0].im, math.floatEpsAt(f32, 4.0));
    try std.testing.expectApproxEqAbs(-6.0, x.buf[1].re, math.floatEpsAt(f32, -6.0));
    try std.testing.expectApproxEqAbs(8.0, x.buf[1].im, math.floatEpsAt(f32, 8.0));
    try std.testing.expectApproxEqAbs(1.0, x.buf[2].re, math.floatEpsAt(f32, 1.0));
    try std.testing.expectApproxEqAbs(-3.0, x.buf[2].im, math.floatEpsAt(f32, -3.0));
}

test "scal complex" {
    const I = enum { i };
    const T = Complex(f32);
    const Arr = NamedArray(I, T);

    var buf_x: [3]T = .{
        .{ .re = 1.0, .im = 2.0 },
        .{ .re = -3.0, .im = 4.0 },
        .{ .re = 0.5, .im = -1.5 },
    };
    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &buf_x,
    };

    const alpha: T = .{ .re = 2.0, .im = -1.0 };
    blas.scal(I, T, T, alpha, x);

    // Expected: element-wise (2 - i) * x
    // e0: (2 - i)*(1 + 2i) = 2 + 4i - i - 2i^2 = (4 + 3i)
    // e1: (2 - i)*(-3 + 4i) = -6 + 8i + 3i - 4i^2 = (-2 + 11i)
    // e2: (2 - i)*(0.5 - 1.5i) = 1 - 3i - 0.5i + 1.5i^2 = ( -0.5 - 3.5i )
    try std.testing.expectApproxEqAbs(4.0, x.buf[0].re, math.floatEpsAt(f32, 4.0));
    try std.testing.expectApproxEqAbs(3.0, x.buf[0].im, math.floatEpsAt(f32, 3.0));
    try std.testing.expectApproxEqAbs(-2.0, x.buf[1].re, math.floatEpsAt(f32, -2.0));
    try std.testing.expectApproxEqAbs(11.0, x.buf[1].im, math.floatEpsAt(f32, 11.0));
    try std.testing.expectApproxEqAbs(-0.5, x.buf[2].re, math.floatEpsAt(f32, -0.5));
    try std.testing.expectApproxEqAbs(-3.5, x.buf[2].im, math.floatEpsAt(f32, -3.5));
}
