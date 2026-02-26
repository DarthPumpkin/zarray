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
    /// In contrast with BLAS, returns the rotation as a struct instead of using output arguments.
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
    /// The points to be rotated are given as an `ij` NamedArray,
    /// where `i` is the index of the point and `j` the index of the dimension.
    /// The `j` axis must have length 2.
    pub fn rot_real(
        comptime Scalar: type,
        rot: GivensRotationReal(Scalar),
        points: NamedArray(IJ, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_srot,
            f64 => acc.cblas_drot,
            else => @compileError("rot_real is incompatible with given Scalar type."),
        };
        assert(points.idx.shape.j == 2);
        const x_na = NamedArray(I, Scalar){
            .idx = points.idx.sliceAxis(.j, 0, 1).conformAxes(I),
            .buf = points.buf,
        };
        const y_na = NamedArray(I, Scalar){
            .idx = points.idx.sliceAxis(.j, 1, 2).conformAxes(I),
            .buf = points.buf,
        };
        const x_blas = Blas1dMut(Scalar).init(I, x_na);
        const y_blas = Blas1dMut(Scalar).init(I, y_na);
        f(
            x_blas.len,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            rot.c,
            rot.s,
        );
    }

    /// **TODO: THIS IS CURRENTLY BROKEN.**
    /// Accelerate's `crot_` and `zrot_` don't seem to behave in the same way as the BLAS counterparts.
    /// I haven't been able to figure this out yet.
    ///
    /// `crot` and `zrot` in BLAS/LAPACK. See `rot_real` for the real versions.
    /// The points to be rotated are given as an `ij` NamedArray,
    /// where `i` is the index of the point and `j` the index of the dimension.
    /// The `j` axis must have length 2.
    pub fn rot_complex(
        comptime RealScalar: type,
        rot: GivensRotationComplex(RealScalar),
        points: NamedArray(IJ, Complex(RealScalar)),
    ) void {
        const f = switch (RealScalar) {
            f32 => acc.crot_,
            f64 => acc.zrot_,
            else => @compileError("rot_complex is incompatible with given RealScalar type."),
        };
        assert(points.idx.shape.j == 2);
        const x_na = NamedArray(I, Complex(RealScalar)){
            .idx = points.idx.sliceAxis(.j, 0, 1).conformAxes(I),
            .buf = points.buf,
        };
        const y_na = NamedArray(I, Complex(RealScalar)){
            .idx = points.idx.sliceAxis(.j, 1, 2).conformAxes(I),
            .buf = points.buf,
        };
        const x_blas = Blas1dMut(Complex(RealScalar)).init(I, x_na);
        const y_blas = Blas1dMut(Complex(RealScalar)).init(I, y_na);
        _ = f(
            x_blas.len,
            @ptrCast(x_blas.ptr),
            x_blas.inc,
            @ptrCast(y_blas.ptr),
            y_blas.inc,
            @ptrCast(@constCast(&rot.c)),
            @ptrCast(@constCast(&rot.s)),
        );
    }

    /// `srotmg` and `drotmg` in BLAS.
    /// In contrast with BLAS, returns the rotation as a struct instead of using an output argument.
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

    /// `srotm` and `drotm` in BLAS.
    /// The points to be rotated are given as an `ij` NamedArray,
    /// where `i` is the index of the point and `j` the index of the dimension.
    /// The `j` axis must have length 2.
    pub fn rotm(
        comptime Scalar: type,
        rot: ModifiedGivensRotation(Scalar),
        points: NamedArray(IJ, Scalar),
    ) void {
        const f = switch (Scalar) {
            f32 => acc.cblas_srotm,
            f64 => acc.cblas_drotm,
            else => @compileError("rotm is incompatible with given Scalar type."),
        };
        assert(points.idx.shape.j == 2);
        const x_na = NamedArray(I, Scalar){
            .idx = points.idx.sliceAxis(.j, 0, 1).conformAxes(I),
            .buf = points.buf,
        };
        const y_na = NamedArray(I, Scalar){
            .idx = points.idx.sliceAxis(.j, 1, 2).conformAxes(I),
            .buf = points.buf,
        };
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

    /// `sgemv`, `dgemv`, `cgemv` and `zgemv` in BLAS.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    /// The major dimension of `A` is inferred from the axis names of `A` and `x`, which must share one axis name.
    pub fn gemv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisXY: type,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArrayConst(AxisXY, Scalar),
        y: NamedArray(AxisXY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const a_names = comptime meta.fieldNames(AxisA);
        const shared_axis = comptime blk: {
            assert(meta.fields(AxisA).len == 2);
            assert(meta.fields(AxisXY).len == 1);
            const xy_name = meta.fields(AxisXY)[0].name;
            assert(std.mem.eql(u8, a_names[0], xy_name) or std.mem.eql(u8, a_names[1], xy_name));
            break :blk if (std.mem.eql(u8, a_names[0], xy_name)) @as(usize, 0) else @as(usize, 1);
        };
        const f = switch (Scalar) {
            f32 => acc.cblas_sgemv,
            f64 => acc.cblas_dgemv,
            Complex(f32) => acc.cblas_cgemv,
            Complex(f64) => acc.cblas_zgemv,
            else => @compileError("gemv is incompatible with given Scalar type."),
        };
        const AlphaType = if (Scalar == Complex(f32) or Scalar == Complex(f64)) *const Scalar else Scalar;

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("gemv: dimension mismatch");

        const a_ij_idx = A.idx.rename(IJ, &.{
            .{ .old = a_names[shared_axis], .new = "j" },
            .{ .old = a_names[1 - shared_axis], .new = "i" },
        });
        const A_ij: NamedArrayConst(IJ, Scalar) = .{ .idx = a_ij_idx, .buf = A.buf };
        const A_blas = Blas2d(Scalar).init(A_ij);
        const x_blas = Blas1d(Scalar).init(AxisXY, x);
        const y_blas = Blas1dMut(Scalar).init(AxisXY, y);
        const alpha_blas: AlphaType = if (Scalar == Complex(f32) or Scalar == Complex(f64)) &scalars.alpha else scalars.alpha;
        const beta_blas: AlphaType = if (Scalar == Complex(f32) or Scalar == Complex(f64)) &scalars.beta else scalars.beta;
        f(
            A_blas.layout,
            acc.CblasNoTrans,
            A_blas.rows,
            A_blas.cols,
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            x_blas.ptr,
            x_blas.inc,
            beta_blas,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `chemv` and `zhemv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is a Hermitian matrix.
    /// Only one triangle of `A` is read, selected by `uplo`.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    /// `A` must have one axis in common with `x` and `y`.
    pub fn hemv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisXY: type,
        uplo: Uplo,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArrayConst(AxisXY, Scalar),
        y: NamedArray(AxisXY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const a_names = comptime meta.fieldNames(AxisA);
        comptime {
            assert(meta.fields(AxisA).len == 2);
            assert(meta.fields(AxisXY).len == 1);
            const xy_name = meta.fields(AxisXY)[0].name;
            assert(std.mem.eql(u8, a_names[0], xy_name) or std.mem.eql(u8, a_names[1], xy_name));
        }
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_chemv,
            Complex(f64) => acc.cblas_zhemv,
            else => @compileError("hemv requires Complex(f32) or Complex(f64)."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("hemv: dimension mismatch");

        const a_ij_idx = A.idx.rename(IJ, &.{
            .{ .old = a_names[0], .new = "i" },
            .{ .old = a_names[1], .new = "j" },
        });
        const A_ij: NamedArrayConst(IJ, Scalar) = .{ .idx = a_ij_idx, .buf = A.buf };
        const A_blas = Blas2d(Scalar).init(A_ij);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisXY, x);
        const y_blas = Blas1dMut(Scalar).init(AxisXY, y);

        const uplo_blas: acc.CBLAS_UPLO = switch (uplo) {
            .upper => @intCast(acc.CblasUpper),
            .lower => @intCast(acc.CblasLower),
        };

        f(
            A_blas.layout,
            uplo_blas,
            A_blas.rows, // N
            &scalars.alpha,
            A_blas.ptr,
            A_blas.leading,
            x_blas.ptr,
            x_blas.inc,
            &scalars.beta,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `ssymv` and `dsymv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is a real symmetric matrix.
    /// Only one triangle of `A` is read, selected by `uplo`.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn symv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisXY: type,
        uplo: Uplo,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArrayConst(AxisXY, Scalar),
        y: NamedArray(AxisXY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const a_names = comptime meta.fieldNames(AxisA);
        comptime {
            assert(meta.fields(AxisA).len == 2);
            assert(meta.fields(AxisXY).len == 1);
            const xy_name = meta.fields(AxisXY)[0].name;
            assert(std.mem.eql(u8, a_names[0], xy_name) or std.mem.eql(u8, a_names[1], xy_name));
        }
        const f = switch (Scalar) {
            f32 => acc.cblas_ssymv,
            f64 => acc.cblas_dsymv,
            else => @compileError("symv requires f32 or f64."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("symv: dimension mismatch");

        const a_ij_idx = A.idx.rename(IJ, &.{
            .{ .old = a_names[0], .new = "i" },
            .{ .old = a_names[1], .new = "j" },
        });
        const A_ij: NamedArrayConst(IJ, Scalar) = .{ .idx = a_ij_idx, .buf = A.buf };
        const A_blas = Blas2d(Scalar).init(A_ij);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisXY, x);
        const y_blas = Blas1dMut(Scalar).init(AxisXY, y);

        const uplo_blas: acc.CBLAS_UPLO = switch (uplo) {
            .upper => @intCast(acc.CblasUpper),
            .lower => @intCast(acc.CblasLower),
        };

        f(
            A_blas.layout,
            uplo_blas,
            A_blas.rows, // N
            scalars.alpha,
            A_blas.ptr,
            A_blas.leading,
            x_blas.ptr,
            x_blas.inc,
            scalars.beta,
            y_blas.ptr,
            y_blas.inc,
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
                    else => @panic("Invalid flag value in ModifiedGivensRotation data."),
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

    pub const Uplo = enum { upper, lower };

    pub const IJ = enum { i, j };
    const I = enum { i };

    fn one(comptime T: type) T {
        return switch (T) {
            f32, f64 => 1.0,
            Complex(f32), Complex(f64) => .{ .re = 1.0, .im = 0.0 },
            else => @compileError("one: T must be f32, f64 or Complex(...)"),
        };
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

    fn Blas2d(comptime Scalar: type) type {
        return struct {
            // trans: acc.CBLAS_TRANSPOSE,
            layout: acc.CBLAS_ORDER,
            rows: c_int,
            cols: c_int,
            leading: c_int,
            ptr: *const Scalar,

            fn init(arr: NamedArrayConst(IJ, Scalar)) @This() {
                assert(arr.idx.isContiguous());
                assert(arr.idx.strides.i > 0);
                assert(arr.idx.strides.j > 0);
                const order = arr.idx.axisOrder();
                const layout: acc.CBLAS_ORDER = switch (order[0]) {
                    .i => @intCast(acc.CblasRowMajor),
                    .j => @intCast(acc.CblasColMajor),
                };
                const leading = switch (order[0]) {
                    .i => arr.idx.shape.j,
                    .j => arr.idx.shape.i,
                };
                return .{
                    .layout = layout,
                    .rows = @intCast(arr.idx.shape.i),
                    .cols = @intCast(arr.idx.shape.j),
                    .leading = @intCast(leading),
                    .ptr = @ptrCast(arr.buf.ptr),
                };
            }
        };
    }

    // const Blas2dLayout = enum(bool) { RowMajor, ColMajor };
    // const Blas2dTrans = enum(u8) { NoTrans = 'N', Trans = 'T', ConjTrans = 'C' };
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

test "rotg_real" {
    const T = f32;

    var a: T = 3.0;
    var b: T = 4.0;

    const rotation = blas.rotg_real(T, &a, &b);

    // sqrt(3^2 + 4^2) = 5, so c = 3/5 and s = 4/5.
    try std.testing.expectApproxEqAbs(0.6, rotation.c, math.floatEpsAt(T, 0.6));
    try std.testing.expectApproxEqAbs(0.8, rotation.s, math.floatEpsAt(T, 0.8));
    try std.testing.expectApproxEqAbs(5.0, a, math.floatEpsAt(T, 5.0));
    // b is overwritten with some value that Apple's documentation claims to be zero.
    // That is incorrect.
}

test "rotg_complex" {
    const T = f64;

    var a: Complex(T) = .{ .re = 3.0, .im = 4.0 };
    var b: Complex(T) = .{ .re = 1.0, .im = -2.0 };
    const rotation = blas.rotg_complex(T, &a, &b);

    var a_cblas: Complex(T) = .{ .re = 3.0, .im = 4.0 };
    var b_cblas: Complex(T) = .{ .re = 1.0, .im = -2.0 };
    var c_cblas: T = undefined;
    var s_cblas: Complex(T) = undefined;
    acc.cblas_zrotg(@ptrCast(&a_cblas), @ptrCast(&b_cblas), &c_cblas, @ptrCast(&s_cblas));

    // The expected values are taken from the output of cblas_crotg in Accelerate.
    try std.testing.expectApproxEqAbs(c_cblas, rotation.c, math.floatEpsAt(T, c_cblas));
    try std.testing.expectApproxEqAbs(s_cblas.re, rotation.s.re, math.floatEpsAt(T, s_cblas.re));
    try std.testing.expectApproxEqAbs(s_cblas.im, rotation.s.im, math.floatEpsAt(T, s_cblas.im));
    try std.testing.expectApproxEqAbs(a_cblas.re, a.re, math.floatEpsAt(T, a_cblas.re));
    try std.testing.expectApproxEqAbs(a_cblas.im, a.im, math.floatEpsAt(T, a_cblas.im));
}

test "rot_real" {
    const T = f32;

    const theta: T = math.pi / 4.0; // 45 degrees
    const rot = blas.GivensRotationReal(T){
        .c = math.cos(theta),
        .s = math.sin(theta),
    };
    var points_buf = [_]T{
        -1.0,           0.0,
        math.sqrt(2.0), math.sqrt(2.0),
    };
    const points = NamedArray(blas.IJ, T){
        .idx = .initContiguous(.{ .i = 2, .j = 2 }),
        .buf = &points_buf,
    };
    const expected_points = &[_]T{
        -math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0,
        2.0,                   0.0,
    };

    blas.rot_real(T, rot, points);

    for (points_buf, 0..) |p, i| {
        try std.testing.expectApproxEqAbs(
            expected_points[i],
            p,
            1e-6,
        );
    }
}

test "rotmg" {
    const T = f32;

    var d1: T = 1.0;
    var d2: T = 2.0;
    var a: T = 3.0;
    const b: T = 4.0;

    var d1_cblas = d1;
    var d2_cblas = d2;
    var a_cblas = a;
    const b_cblas = b;
    var p_cblas: [5]T = undefined;
    acc.cblas_srotmg(&d1_cblas, &d2_cblas, &a_cblas, b_cblas, &p_cblas);

    const rotation = blas.rotmg(T, &d1, &d2, &a, b);

    try std.testing.expectApproxEqAbs(d1_cblas, d1, math.floatEpsAt(T, d1_cblas));
    try std.testing.expectApproxEqAbs(d2_cblas, d2, math.floatEpsAt(T, d2_cblas));
    try std.testing.expectApproxEqAbs(a_cblas, a, math.floatEpsAt(T, a_cblas));
    try std.testing.expectApproxEqAbs(b_cblas, b, math.floatEpsAt(T, b_cblas));
    for (rotation.data, p_cblas) |r, e| {
        try std.testing.expectApproxEqAbs(e, r, math.floatEpsAt(T, e));
    }
}

test "rotm" {
    const T = f64;

    const x_buf = [_]T{ -1.0, math.sqrt(2.0) };
    const y_buf = [_]T{ 0.0, -math.sqrt(2.0) };
    // reversed order to test negative strides
    const points_buf = [_]T{
        x_buf[1], y_buf[1],
        x_buf[0], y_buf[0],
    };

    const p1 = [_]T{ -2.0, 0.5, 0.5, 1.3948, -0.7071 };
    const p2 = [_]T{ -1.0, 0.5, 0.5, 1.3948, -0.7071 };
    const p3 = [_]T{ 0.0, 0.5, 0.5, 1.3948, -0.7071 };
    const p4 = [_]T{ 1.0, 0.5, 0.5, 1.3948, -0.7071 };

    const rot1 = blas.ModifiedGivensRotation(T){ .data = p1 };
    const rot2 = blas.ModifiedGivensRotation(T){ .data = p2 };
    const rot3 = blas.ModifiedGivensRotation(T){ .data = p3 };
    const rot4 = blas.ModifiedGivensRotation(T){ .data = p4 };

    var x_buf1 = x_buf;
    var x_buf2 = x_buf;
    var x_buf3 = x_buf;
    var x_buf4 = x_buf;
    var y_buf1 = y_buf;
    var y_buf2 = y_buf;
    var y_buf3 = y_buf;
    var y_buf4 = y_buf;

    acc.cblas_drotm(2, @ptrCast(&x_buf1), 1, @ptrCast(&y_buf1), 1, &p1);
    acc.cblas_drotm(2, @ptrCast(&x_buf2), 1, @ptrCast(&y_buf2), 1, &p2);
    acc.cblas_drotm(2, @ptrCast(&x_buf3), 1, @ptrCast(&y_buf3), 1, &p3);
    acc.cblas_drotm(2, @ptrCast(&x_buf4), 1, @ptrCast(&y_buf4), 1, &p4);

    var points_buf1 = points_buf;
    var points_buf2 = points_buf;
    var points_buf3 = points_buf;
    var points_buf4 = points_buf;

    const idx = NamedIndex(blas.IJ).initContiguous(.{ .i = 2, .j = 2 }).stride(.{ .i = -1 });
    const points1 = NamedArray(blas.IJ, T){ .idx = idx, .buf = &points_buf1 };
    const points2 = NamedArray(blas.IJ, T){ .idx = idx, .buf = &points_buf2 };
    const points3 = NamedArray(blas.IJ, T){ .idx = idx, .buf = &points_buf3 };
    const points4 = NamedArray(blas.IJ, T){ .idx = idx, .buf = &points_buf4 };

    blas.rotm(T, rot1, points1);
    blas.rotm(T, rot2, points2);
    blas.rotm(T, rot3, points3);
    blas.rotm(T, rot4, points4);

    for (0..2) |i| {
        for (0..2) |j| {
            const expected1 = if (j == 0) x_buf1[i] else y_buf1[i];
            const expected2 = if (j == 0) x_buf2[i] else y_buf2[i];
            const expected3 = if (j == 0) x_buf3[i] else y_buf3[i];
            const expected4 = if (j == 0) x_buf4[i] else y_buf4[i];
            const actual1 = points1.at(.{ .i = i, .j = j }).*;
            const actual2 = points2.at(.{ .i = i, .j = j }).*;
            const actual3 = points3.at(.{ .i = i, .j = j }).*;
            const actual4 = points4.at(.{ .i = i, .j = j }).*;
            try std.testing.expectApproxEqAbs(
                expected1,
                actual1,
                math.floatEpsAt(T, expected1),
            );
            try std.testing.expectApproxEqAbs(
                expected2,
                actual2,
                math.floatEpsAt(T, expected2),
            );
            try std.testing.expectApproxEqAbs(
                expected3,
                actual3,
                math.floatEpsAt(T, expected3),
            );
            try std.testing.expectApproxEqAbs(
                expected4,
                actual4,
                math.floatEpsAt(T, expected4),
            );
        }
    }
}

test "gemv real" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] (3x3 row-major)
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // y = 1 * A * x + 0 * y = A * x = [14, 32, 50]
    blas.gemv(T, MK, K, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 14.0, 32.0, 50.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemv real nontrivial scalars and strides" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // x_buf physical = [1, 2, 3]; stride -1 → logical x = [3, 2, 1]
    const x_buf = [_]T{ 1, 2, 3 };
    var x_idx = NamedIndex(K).initContiguous(.{ .k = 3 });
    x_idx = x_idx.stride(.{ .k = -1 });
    const x = NamedArrayConst(K, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // y with stride 2: positions 0, 2, 4 in buffer
    var y_buf = [_]T{ 10, 99, 20, 99, 30 };
    const y_idx: NamedIndex(K) = .{
        .shape = .{ .k = 3 },
        .strides = .{ .k = 2 },
        .offset = 0,
    };
    const y = NamedArray(K, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 2.0, beta = -1.0
    // A * x_logical = A * [3,2,1]:
    //   [1*3+2*2+3*1, 4*3+5*2+6*1, 7*3+8*2+9*1] = [10, 28, 46]
    // y = 2 * [10, 28, 46] + (-1) * [10, 20, 30] = [10, 36, 62]
    blas.gemv(T, MK, K, A, x, y, .{ .alpha = 2.0, .beta = -1.0 });

    const expected = [_]T{ 10.0, 36.0, 62.0 };
    try std.testing.expectApproxEqAbs(expected[0], y_buf[0], math.floatEpsAt(T, expected[0]));
    try std.testing.expectApproxEqAbs(expected[1], y_buf[2], math.floatEpsAt(T, expected[1]));
    try std.testing.expectApproxEqAbs(expected[2], y_buf[4], math.floatEpsAt(T, expected[2]));
    // sentinel values untouched
    try std.testing.expectEqual(@as(T, 99.0), y_buf[1]);
    try std.testing.expectEqual(@as(T, 99.0), y_buf[3]);
}

test "gemv real column-major matrix" {
    // Use enum { i, j } with column-major strides to exercise CblasColMajor path
    const IJ = enum { i, j };
    const J = enum { j };
    const T = f32;

    // A = [[1, 3], [2, 4]] stored column-major: columns contiguous
    // Column-major buffer for 2x2: col0=[1,2], col1=[3,4]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 }, // column-major: i-stride=1, j-stride=rows
        },
        .buf = &a_buf,
    };

    // x = [5, 6], y = [0, 0]
    // Shared axis is j (second name in IJ matches J).
    // After rename: j→j (cols), i→i (rows) → NoTrans
    // y = A*x = [1*5+3*6, 2*5+4*6] = [23, 34]
    const x_buf = [_]T{ 5, 6 };
    const x = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0 };
    const y = NamedArray(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &y_buf,
    };

    blas.gemv(T, IJ, J, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23.0, 34.0 };
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemv complex" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    // A (3x3):
    //   [[1+i, 2+0i, 0+i],
    //    [0+i, 1+0i, 2+i],
    //    [1+0i, 0+i, 1+i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 1 },
        .{ .re = 0, .im = 1 }, .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 1 },
        .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 }, .{ .re = 1, .im = 1 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // x = [1+0i, 0+1i, 1+i]
    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = 1 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // alpha = 2+i, beta = 0
    //
    // A*x:
    //   row0: (1+i)*1 + 2*(i) + (i)*(1+i) = 1+i + 2i + i-1 = 0+4i
    //   row1: (i)*1 + 1*(i) + (2+i)*(1+i) = i + i + 2+3i-1 = 1+5i
    //   row2: 1*1 + (i)*(i) + (1+i)*(1+i) = 1 - 1 + 2i = 0+2i
    //
    // y = (2+i)*[4i, 1+5i, 2i]:
    //   (2+i)(4i)   = 8i+4i² = -4+8i
    //   (2+i)(1+5i) = 2+10i+i+5i² = -3+11i
    //   (2+i)(2i)   = 4i+2i² = -2+4i
    blas.gemv(T, MK, K, A, x, y, .{
        .alpha = .{ .re = 2, .im = 1 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = -4, .im = 8 },
        .{ .re = -3, .im = 11 },
        .{ .re = -2, .im = 4 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "gemv complex nontrivial scalars and strides" {
    const AB = enum { a, b };
    const B = enum { b };
    const T = Complex(f32);

    // A (2x2): [[1+i, 2-i], [3+0i, 0+2i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = -1 },
        .{ .re = 3, .im = 0 }, .{ .re = 0, .im = 2 },
    };
    const A = NamedArrayConst(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1+0i, 0+1i]; stride -1 → logical x = [0+1i, 1+0i]
    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    var x_idx = NamedIndex(B).initContiguous(.{ .b = 2 });
    x_idx = x_idx.stride(.{ .b = -1 });
    const x = NamedArrayConst(B, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // y with stride 2; y_init = [1+0i, 2+0i]
    var y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 99, .im = 99 }, // sentinel
        .{ .re = 2, .im = 0 },
    };
    const y_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const y = NamedArray(B, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 1+i, beta = 2+0i
    //
    // A * x_logical = A * [i, 1]:
    //   row0: (1+i)(i) + (2-i)(1) = i+i² + 2-i = -1+i+2-i = 1+0i
    //   row1: (3)(i) + (2i)(1) = 3i + 2i = 0+5i
    //
    // y = (1+i)*[1, 5i] + 2*[1, 2]:
    //   (1+i)(1) + 2  = 1+i+2 = 3+i
    //   (1+i)(5i) + 4 = 5i+5i²+4 = -5+5i+4 = -1+5i
    blas.gemv(T, AB, B, A, x, y, .{
        .alpha = .{ .re = 1, .im = 1 },
        .beta = .{ .re = 2, .im = 0 },
    });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 3), y_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 1), y_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, -1), y_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 5), y_buf[2].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99), y_buf[1].re);
    try std.testing.expectEqual(@as(f32, 99), y_buf[1].im);
}

test "hemv upper" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    // Hermitian 3x3 matrix (upper triangle stored):
    //   [[2,      1-i,   3+2i],
    //    [1+i,    5,     2-i ],
    //    [3-2i,   2+i,   4   ]]
    // Buffer is row-major; BLAS reads only upper triangle (col >= row).
    const a_buf = [_]T{
        .{ .re = 2, .im = 0 },   .{ .re = 1, .im = -1 },  .{ .re = 3, .im = 2 },
        .{ .re = 99, .im = 99 }, .{ .re = 5, .im = 0 },   .{ .re = 2, .im = -1 },
        .{ .re = 99, .im = 99 }, .{ .re = 99, .im = 99 }, .{ .re = 4, .im = 0 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // y = A * x:
    //   y[0] = 2*(1) + (1-i)*(i) + (3+2i)*(1) = 2 + i+1 + 3+2i = 6+3i
    //   y[1] = (1+i)*(1) + 5*(i) + (2-i)*(1) = 1+i + 5i + 2-i = 3+5i
    //   y[2] = (3-2i)*(1) + (2+i)*(i) + 4*(1) = 3-2i + 2i-1 + 4 = 6+0i
    blas.hemv(T, MK, K, .upper, A, x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 6, .im = 3 },
        .{ .re = 3, .im = 5 },
        .{ .re = 6, .im = 0 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "hemv lower" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    // Same Hermitian matrix, but now the *lower* triangle is stored.
    // Upper triangle positions hold sentinels that BLAS must ignore.
    //   [[2,      _,      _     ],
    //    [1+i,    5,      _     ],
    //    [3-2i,   2+i,    4     ]]
    const a_buf = [_]T{
        .{ .re = 2, .im = 0 },  .{ .re = 99, .im = 99 }, .{ .re = 99, .im = 99 },
        .{ .re = 1, .im = 1 },  .{ .re = 5, .im = 0 },   .{ .re = 99, .im = 99 },
        .{ .re = 3, .im = -2 }, .{ .re = 2, .im = 1 },   .{ .re = 4, .im = 0 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // Same result as the upper test: y = [6+3i, 3+5i, 6+0i]
    blas.hemv(T, MK, K, .lower, A, x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 6, .im = 3 },
        .{ .re = 3, .im = 5 },
        .{ .re = 6, .im = 0 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "hemv nontrivial scalars and strides" {
    const AB = enum { a, b };
    const B = enum { b };
    const T = Complex(f32);

    // Hermitian 2x2: [[3, 1+2i], [1-2i, 5]]
    // Upper triangle stored; lower positions hold sentinels.
    const a_buf = [_]T{
        .{ .re = 3, .im = 0 },   .{ .re = 1, .im = 2 },
        .{ .re = 99, .im = 99 }, .{ .re = 5, .im = 0 },
    };
    const A = NamedArrayConst(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1+i, 2+0i]; stride -1 → logical x = [2+0i, 1+i]
    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    var x_idx = NamedIndex(B).initContiguous(.{ .b = 2 });
    x_idx = x_idx.stride(.{ .b = -1 });
    const x = NamedArrayConst(B, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // y with stride 2; y_init = [1+0i, 2+i]
    var y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 99, .im = 99 }, // sentinel
        .{ .re = 2, .im = 1 },
    };
    const y_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const y = NamedArray(B, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 1+i, beta = 2
    //
    // A * x_logical = A * [2, 1+i]:
    //   row0: 3*(2) + (1+2i)*(1+i) = 6 + 1+i+2i+2i² = 6 + -1+3i = 5+3i
    //   row1: (1-2i)*(2) + 5*(1+i) = 2-4i + 5+5i = 7+i
    //
    // y = (1+i)*[5+3i, 7+i] + 2*[1, 2+i]:
    //   (1+i)(5+3i) = 5+3i+5i+3i² = 2+8i
    //   (1+i)(7+i)  = 7+i+7i+i²   = 6+8i
    //   y[0] = 2+8i + 2   = 4+8i
    //   y[1] = 6+8i + 4+2i = 10+10i
    blas.hemv(T, AB, B, .upper, A, x, y, .{
        .alpha = .{ .re = 1, .im = 1 },
        .beta = .{ .re = 2, .im = 0 },
    });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 4), y_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 8), y_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 10), y_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 10), y_buf[2].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99), y_buf[1].re);
    try std.testing.expectEqual(@as(f32, 99), y_buf[1].im);
}

test "symv upper" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Symmetric 3x3 matrix (upper triangle stored):
    //   [[2, 3, 5],
    //    [3, 7, 11],
    //    [5, 11, 13]]
    // Lower triangle positions hold sentinels that BLAS must ignore.
    const a_buf = [_]T{
        2,  3,  5,
        99, 7,  11,
        99, 99, 13,
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // y = A * x:
    //   y[0] = 2*1 + 3*2 + 5*3  = 2 + 6 + 15  = 23
    //   y[1] = 3*1 + 7*2 + 11*3 = 3 + 14 + 33 = 50
    //   y[2] = 5*1 + 11*2 + 13*3 = 5 + 22 + 39 = 66
    blas.symv(T, MK, K, .upper, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23.0, 50.0, 66.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symv lower" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Same symmetric matrix, but now the *lower* triangle is stored.
    // Upper triangle positions hold sentinels that BLAS must ignore.
    //   [[2,  _,  _ ],
    //    [3,  7,  _ ],
    //    [5,  11, 13]]
    const a_buf = [_]T{
        2, 99, 99,
        3, 7,  99,
        5, 11, 13,
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // Same result as the upper test: y = [23, 50, 66]
    blas.symv(T, MK, K, .lower, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23.0, 50.0, 66.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symv nontrivial scalars and strides" {
    const AB = enum { a, b };
    const B = enum { b };
    const T = f32;

    // Symmetric 2x2: [[4, 3], [3, 7]]
    // Upper triangle stored; lower position holds sentinel.
    const a_buf = [_]T{ 4, 3, 99, 7 };
    const A = NamedArrayConst(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1, 2]; stride -1 → logical x = [2, 1]
    const x_buf = [_]T{ 1, 2 };
    var x_idx = NamedIndex(B).initContiguous(.{ .b = 2 });
    x_idx = x_idx.stride(.{ .b = -1 });
    const x = NamedArrayConst(B, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // y with stride 2; y_init = [10, 20]
    var y_buf = [_]T{ 10, 99, 20 };
    const y_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const y = NamedArray(B, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 2.5, beta = -1
    //
    // A * x_logical = A * [2, 1]:
    //   row0: 4*2 + 3*1 = 11
    //   row1: 3*2 + 7*1 = 13
    //
    // y = 2.5 * [11, 13] + (-1) * [10, 20] = [27.5-10, 32.5-20] = [17.5, 12.5]
    blas.symv(T, AB, B, .upper, A, x, y, .{ .alpha = 2.5, .beta = -1.0 });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 17.5), y_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), y_buf[2], eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99.0), y_buf[1]);
}

// TODO: Figure this out. See blas.rot_complex above.
// test "rot_complex" {
//     const T = f64;

//     var x_buf: [4]Complex(T) = .{
//         .{ .re = -1.0, .im = 0.0 },
//         .{ .re = math.sqrt(2.0), .im = math.sqrt(2.0) },
//         .{ .re = 0.0, .im = 0.0 },
//         .{ .re = 29834.346, .im = -0.000000007583 },
//     };
//     var y_buf: [4]Complex(T) = .{
//         .{ .re = 29853.7, .im = 0.0000000000001 },
//         .{ .re = 0.0, .im = 0.0 },
//         .{ .re = -348.1, .im = 0.294857 },
//         .{ .re = -0.0000002, .im = 29857.7 },
//     };
//     var points_buf = [_]Complex(T){
//         x_buf[0], y_buf[0],
//         x_buf[1], y_buf[1],
//         x_buf[2], y_buf[2],
//         x_buf[3], y_buf[3],
//     };
//     const points = NamedArray(blas.IJ, Complex(T)){
//         .idx = .initContiguous(.{ .i = 4, .j = 2 }),
//         .buf = &points_buf,
//     };
//     var c: T = math.cos(math.pi / 3.0);
//     var s: Complex(T) = .{ .re = 2.824, .im = -0.00000000001 };
//     const rot = blas.GivensRotationComplex(T){
//         .c = c,
//         .s = s,
//     };

//     const n: c_int = 4;
//     _ = acc.zrot_(@ptrCast(@alignCast(@constCast(&n))), @ptrCast(&x_buf), 1, @ptrCast(&y_buf), 1, @ptrCast(&c), @ptrCast(&s));

//     blas.rot_complex(T, rot, points);

//     for (0..4) |i| {
//         for (0..2) |j| {
//             const expected = if (j == 0) x_buf[i] else y_buf[i];
//             const actual = points.at(.{ .i = i, .j = j }).*;
//             try std.testing.expectApproxEqAbs(
//                 expected.re,
//                 actual.re,
//                 math.floatEpsAt(T, expected.re),
//             );
//             try std.testing.expectApproxEqAbs(
//                 expected.im,
//                 actual.im,
//                 math.floatEpsAt(T, expected.im),
//             );
//         }
//     }
// }
