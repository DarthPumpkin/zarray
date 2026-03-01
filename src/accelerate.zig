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
    /// `sdot` and `ddot` in BLAS (`sdsdot` with `internal_double_precision`).
    /// Computes the dot product `xᵀy = Σ x_i * y_i` for real scalars.
    /// `x` and `y` must have the same axis and length.
    /// If `internal_double_precision` is `true`, the accumulation is performed in double precision.
    /// This option is only supported for `f32` scalars (maps to `cblas_sdsdot`).
    pub fn dot(
        comptime Scalar: type,
        comptime Axis: type,
        x: NamedArrayConst(Axis, Scalar),
        y: NamedArrayConst(Axis, Scalar),
        comptime config: struct { internal_double_precision: bool = false },
    ) Scalar {
        const x_blas = Blas1d(Scalar).init(Axis, x);
        const y_blas = Blas1d(Scalar).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        if (config.internal_double_precision) {
            // cblas_sdsdot computes the dot product in double precision and returns
            // `alpha + xᵀy` as a float. We pass alpha = 0 to get the pure dot product.
            const cblas_dot = switch (Scalar) {
                f32 => acc.cblas_sdsdot,
                else => @compileError("internal_double_precision is only supported for f32 scalars."),
            };
            return cblas_dot(x_blas.len, 0.0, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
        } else {
            const cblas_dot = switch (Scalar) {
                f32 => acc.cblas_sdot,
                f64 => acc.cblas_ddot,
                else => @compileError("dot is incompatible with given Scalar type."),
            };
            return cblas_dot(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
        }
    }

    /// `dsdot` in BLAS.
    /// Computes the dot product `xᵀy = Σ x_i * y_i` with `f32` inputs,
    /// accumulating and returning the result in `f64` (double precision).
    pub fn dsdot(
        comptime Axis: type,
        x: NamedArrayConst(Axis, f32),
        y: NamedArrayConst(Axis, f32),
    ) f64 {
        const x_blas = Blas1d(f32).init(Axis, x);
        const y_blas = Blas1d(f32).init(Axis, y);
        assert(x_blas.len == y_blas.len);

        return acc.cblas_dsdot(x_blas.len, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
    }

    /// `cdotu` and `zdotu` in BLAS.
    /// Computes the unconjugated dot product `xᵀy = Σ x_i * y_i` for complex scalars.
    /// `x` and `y` must have the same axis and length.
    pub fn dotu(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `cdotc` and `zdotc` in BLAS.
    /// Computes the conjugated dot product `xᴴy = Σ conj(x_i) * y_i` for complex scalars.
    /// `x` and `y` must have the same axis and length.
    pub fn dotc(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `snrm2`, `dnrm2`, `scnrm2` and `dznrm2` in BLAS.
    /// Computes the Euclidean norm `‖x‖₂ = √(Σ |x_i|²)`.
    /// For complex vectors, returns a real scalar.
    pub fn nrm2(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `sasum`, `dasum`, `scasum` and `dzasum` in BLAS.
    /// Computes the sum of absolute values `Σ |x_i|`.
    /// For complex vectors, computes `Σ (|Re(x_i)| + |Im(x_i)|)` and returns a real scalar.
    pub fn asum(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `isamax`, `idamax`, `icamax` and `izamax` in BLAS.
    /// Returns the index of the element with the largest absolute value.
    /// For complex vectors, uses `|Re(x_i)| + |Im(x_i)|` as the magnitude.
    pub fn i_amax(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `sswap`, `dswap`, `cswap` and `zswap` in BLAS.
    /// Swaps the elements of `x` and `y`.
    /// `x` and `y` must have the same axis and length.
    pub fn swap(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `scopy`, `dcopy`, `ccopy` and `zcopy` in BLAS.
    /// Copies the elements of `x` into `y`.
    /// `x` and `y` must have the same axis and length.
    pub fn copy(
        comptime Scalar: type,
        comptime Axis: type,
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

    /// `saxpy`, `daxpy`, `caxpy` and `zaxpy` in BLAS.
    /// Computes `y := alpha * x + y`.
    /// `x` and `y` must have the same axis and length.
    pub fn axpy(
        comptime Scalar: type,
        comptime Axis: type,
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

        const alpha_blas = if (comptime isComplex(Scalar)) &alpha else alpha;
        f(x_blas.len, alpha_blas, x_blas.ptr, x_blas.inc, y_blas.ptr, y_blas.inc);
    }

    /// `sscal`, `dscal`, `cscal`, `zscal`, `csscal` and `zdscal` in BLAS.
    /// Computes `x := alpha * x`, scaling every element of `x` in-place.
    /// `AlphaScalar` may be the real component type of `VecScalar`,
    /// allowing a complex vector to be scaled by a real scalar.
    pub fn scal(
        comptime VecScalar: type,
        comptime AlphaScalar: type,
        comptime Axis: type,
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
    /// Computes `y = alpha * A * x + beta * y`.
    /// `x`'s axis must match one axis of `A` (the contracted dimension)
    /// and `y`'s axis must match the other axis of `A` (the output dimension).
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn gemv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const x_axis_idx = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            f32 => acc.cblas_sgemv,
            f64 => acc.cblas_dgemv,
            Complex(f32) => acc.cblas_cgemv,
            Complex(f64) => acc.cblas_zgemv,
            else => @compileError("gemv is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("gemv: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, 1 - x_axis_idx);
        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisY, y);
        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;
        const beta_blas = if (comptime isComplex(Scalar)) &scalars.beta else scalars.beta;
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
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn hemv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        triangle: AxisA,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_chemv,
            Complex(f64) => acc.cblas_zhemv,
            else => @compileError("hemv requires Complex(f32) or Complex(f64)."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("hemv: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisY, y);

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
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
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn symv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        triangle: AxisA,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssymv,
            f64 => acc.cblas_dsymv,
            else => @compileError("symv requires f32 or f64."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("symv: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisY, y);

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
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

    /// `strmv`, `dtrmv`, `ctrmv` and `ztrmv` in BLAS.
    /// Computes `x = A * x` in-place where `A` is a triangular matrix.
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    pub fn trmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        diag: Diag,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArray(AxisX, Scalar),
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_strmv,
            f64 => acc.cblas_dtrmv,
            Complex(f32) => acc.cblas_ctrmv,
            Complex(f64) => acc.cblas_ztrmv,
            else => @compileError("trmv is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape }) catch
            @panic("trmv: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1dMut(Scalar).init(AxisX, x);

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
            @intCast(acc.CblasNoTrans),
            diag_blas,
            A_blas.rows, // N
            A_blas.ptr,
            A_blas.leading,
            x_blas.ptr,
            x_blas.inc,
        );
    }

    /// `strsv`, `dtrsv`, `ctrsv` and `ztrsv` in BLAS.
    /// Solves `A * x_new = x_old` in-place, i.e. computes `x := A⁻¹ * x`,
    /// where `A` is a triangular matrix.
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    pub fn trsv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        diag: Diag,
        A: NamedArrayConst(AxisA, Scalar),
        x: NamedArray(AxisX, Scalar),
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_strsv,
            f64 => acc.cblas_dtrsv,
            Complex(f32) => acc.cblas_ctrsv,
            Complex(f64) => acc.cblas_ztrsv,
            else => @compileError("trsv is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape }) catch
            @panic("trsv: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1dMut(Scalar).init(AxisX, x);

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
            @intCast(acc.CblasNoTrans),
            diag_blas,
            A_blas.rows, // N
            A_blas.ptr,
            A_blas.leading,
            x_blas.ptr,
            x_blas.inc,
        );
    }

    /// `sgbmv`, `dgbmv`, `cgbmv` and `zgbmv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is an M×N general band matrix
    /// with `kl` sub-diagonals and `ku` super-diagonals, stored in BLAS band format.
    /// `A` is a 2D array with a band axis (size `kl + ku + 1`) and a vector axis
    /// matching `x`'s axis (size N). `y`'s axis provides M.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    ///
    /// **Storage requirement**: The band axis of `A` must be contiguous (stride 1)
    /// and the vector axis stride must be at least `kl + ku + 1`. This is the
    /// standard BLAS band storage layout where each column's band entries are
    /// adjacent in memory.
    pub fn gbmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: BlasBand(Scalar, AxisA, AxisX),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
        comptime band: struct { kl: usize, ku: usize },
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        comptime assert(meta.fields(AxisY).len == 1);
        const f = switch (Scalar) {
            f32 => acc.cblas_sgbmv,
            f64 => acc.cblas_dgbmv,
            Complex(f32) => acc.cblas_cgbmv,
            Complex(f64) => acc.cblas_zgbmv,
            else => @compileError("gbmv is incompatible with given Scalar type."),
        };

        assert(A.band_size == band.kl + band.ku + 1);

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisY, y);

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;
        const beta_blas = if (comptime isComplex(Scalar)) &scalars.beta else scalars.beta;

        f(
            @intCast(acc.CblasColMajor),
            @intCast(acc.CblasNoTrans),
            y_blas.len, // M
            x_blas.len, // N
            @intCast(band.kl),
            @intCast(band.ku),
            alpha_blas,
            A.ptr,
            A.leading, // LDA
            x_blas.ptr,
            x_blas.inc,
            beta_blas,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `ssbmv` and `dsbmv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is an N×N real symmetric band matrix
    /// with bandwidth `K`, stored in BLAS band format.
    /// `A` is a 2D array with a band axis (size `K + 1`) and a vector axis
    /// matching `x`'s axis (size N). `K` is inferred from the band axis size.
    /// `triangle` selects the stored triangle (second axis → upper, first axis → lower).
    /// The scalars `alpha` and `beta` are optional and default to 1.
    ///
    /// **Storage requirement**: The band axis of `A` must be contiguous (stride 1)
    /// and the vector axis stride must be at least `K + 1`. This is the standard
    /// BLAS band storage layout where each column's band entries are adjacent in memory.
    pub fn sbmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        A: BlasBand(Scalar, AxisA, AxisX),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisX, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssbmv,
            f64 => acc.cblas_dsbmv,
            else => @compileError("sbmv requires f32 or f64."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisX, y);

        f(
            @intCast(acc.CblasColMajor),
            uploBlas(AxisA, triangle),
            A.n, // N
            A.k(),
            scalars.alpha,
            A.ptr,
            A.leading, // LDA
            x_blas.ptr,
            x_blas.inc,
            scalars.beta,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `chbmv` and `zhbmv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is an N×N Hermitian band matrix
    /// with bandwidth `K`, stored in BLAS band format.
    /// `A` is a 2D array with a band axis (size `K + 1`) and a vector axis
    /// matching `x`'s axis (size N). `K` is inferred from the band axis size.
    /// `triangle` selects the stored triangle (second axis → upper, first axis → lower).
    /// The scalars `alpha` and `beta` are optional and default to 1.
    ///
    /// **Storage requirement**: The band axis of `A` must be contiguous (stride 1)
    /// and the vector axis stride must be at least `K + 1`. This is the standard
    /// BLAS band storage layout where each column's band entries are adjacent in memory.
    pub fn hbmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        A: BlasBand(Scalar, AxisA, AxisX),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisX, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_chbmv,
            Complex(f64) => acc.cblas_zhbmv,
            else => @compileError("hbmv requires Complex(f32) or Complex(f64)."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisX, y);

        f(
            @intCast(acc.CblasColMajor),
            uploBlas(AxisA, triangle),
            A.n, // N
            A.k(),
            &scalars.alpha,
            A.ptr,
            A.leading, // LDA
            x_blas.ptr,
            x_blas.inc,
            &scalars.beta,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `stbmv`, `dtbmv`, `ctbmv` and `ztbmv` in BLAS.
    /// Computes `x = A * x` in-place where `A` is a triangular band matrix
    /// with bandwidth `K`, stored in BLAS band format.
    /// `A` is a 2D array with a band axis (size `K + 1`) and a vector axis
    /// matching `x`'s axis (size N). `K` is inferred from the band axis size.
    /// `triangle` selects the stored triangle (second axis → upper, first axis → lower).
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    ///
    /// **Storage requirement**: The band axis of `A` must be contiguous (stride 1)
    /// and the vector axis stride must be at least `K + 1`. This is the standard
    /// BLAS band storage layout where each column's band entries are adjacent in memory.
    pub fn tbmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        diag: Diag,
        A: BlasBand(Scalar, AxisA, AxisX),
        x: NamedArray(AxisX, Scalar),
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_stbmv,
            f64 => acc.cblas_dtbmv,
            Complex(f32) => acc.cblas_ctbmv,
            Complex(f64) => acc.cblas_ztbmv,
            else => @compileError("tbmv is incompatible with given Scalar type."),
        };

        const x_blas = Blas1dMut(Scalar).init(AxisX, x);

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        f(
            @intCast(acc.CblasColMajor),
            uploBlas(AxisA, triangle),
            @intCast(acc.CblasNoTrans),
            diag_blas,
            A.n, // N
            A.k(),
            A.ptr,
            A.leading, // LDA
            x_blas.ptr,
            x_blas.inc,
        );
    }

    /// `stbsv`, `dtbsv`, `ctbsv` and `ztbsv` in BLAS.
    /// Solves `A * x_new = x_old` in-place, i.e. computes `x := A⁻¹ * x`,
    /// where `A` is a triangular band matrix with bandwidth `K`, stored in BLAS band format.
    /// `A` is a 2D array with a band axis (size `K + 1`) and a vector axis
    /// matching `x`'s axis (size N). `K` is inferred from the band axis size.
    /// `triangle` selects the stored triangle (second axis → upper, first axis → lower).
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    ///
    /// **Storage requirement**: The band axis of `A` must be contiguous (stride 1)
    /// and the vector axis stride must be at least `K + 1`. This is the standard
    /// BLAS band storage layout where each column's band entries are adjacent in memory.
    pub fn tbsv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        diag: Diag,
        A: BlasBand(Scalar, AxisA, AxisX),
        x: NamedArray(AxisX, Scalar),
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_stbsv,
            f64 => acc.cblas_dtbsv,
            Complex(f32) => acc.cblas_ctbsv,
            Complex(f64) => acc.cblas_ztbsv,
            else => @compileError("tbsv is incompatible with given Scalar type."),
        };

        const x_blas = Blas1dMut(Scalar).init(AxisX, x);

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        f(
            @intCast(acc.CblasColMajor),
            uploBlas(AxisA, triangle),
            @intCast(acc.CblasNoTrans),
            diag_blas,
            A.n, // N
            A.k(),
            A.ptr,
            A.leading, // LDA
            x_blas.ptr,
            x_blas.inc,
        );
    }

    /// `sspmv` and `dspmv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is an N×N real symmetric matrix
    /// stored in packed format.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `symv`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA` and `y`'s axis the other.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    ///
    /// **Storage requirement**: `AP` must be a contiguous slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn spmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: BlasPacked(Scalar, AxisA),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            f32 => acc.cblas_sspmv,
            f64 => acc.cblas_dspmv,
            else => @compileError("spmv requires f32 or f64."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisY, y);

        assert(x_blas.len == y_blas.len); // Square matrix: x and y dimensions must match
        A.validate(@intCast(x_blas.len));

        f(
            A.layout(),
            A.uplo(),
            x_blas.len, // N
            scalars.alpha,
            A.ptr(),
            x_blas.ptr,
            x_blas.inc,
            scalars.beta,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `chpmv` and `zhpmv` in BLAS.
    /// Computes `y = alpha * A * x + beta * y` where `A` is an N×N Hermitian matrix
    /// stored in packed format.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `hemv`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA` and `y`'s axis the other.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    ///
    /// **Storage requirement**: `AP` must be a contiguous slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn hpmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: BlasPacked(Scalar, AxisA),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArray(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_chpmv,
            Complex(f64) => acc.cblas_zhpmv,
            else => @compileError("hpmv requires Complex(f32) or Complex(f64)."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1dMut(Scalar).init(AxisY, y);

        assert(x_blas.len == y_blas.len); // Square matrix: x and y dimensions must match
        A.validate(@intCast(x_blas.len));

        f(
            A.layout(),
            A.uplo(),
            x_blas.len, // N
            &scalars.alpha,
            A.ptr(),
            x_blas.ptr,
            x_blas.inc,
            &scalars.beta,
            y_blas.ptr,
            y_blas.inc,
        );
    }

    /// `stpmv`, `dtpmv`, `ctpmv` and `ztpmv` in BLAS.
    /// Computes `x = A * x` in-place where `A` is an N×N triangular matrix
    /// stored in packed format.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `trmv`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA`.
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    ///
    /// **Storage requirement**: `AP` must be a contiguous slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn tpmv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        A: BlasPacked(Scalar, AxisA),
        diag: Diag,
        x: NamedArray(AxisX, Scalar),
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_stpmv,
            f64 => acc.cblas_dtpmv,
            Complex(f32) => acc.cblas_ctpmv,
            Complex(f64) => acc.cblas_ztpmv,
            else => @compileError("tpmv is incompatible with given Scalar type."),
        };

        const x_blas = Blas1dMut(Scalar).init(AxisX, x);
        A.validate(@intCast(x_blas.len));

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        f(
            A.layout(),
            A.uplo(),
            @intCast(acc.CblasNoTrans),
            diag_blas,
            x_blas.len, // N
            A.ptr(),
            x_blas.ptr,
            x_blas.inc,
        );
    }

    /// `stpsv`, `dtpsv`, `ctpsv` and `ztpsv` in BLAS.
    /// Solves `A * x_new = x_old` in-place, i.e. computes `x := A⁻¹ * x`,
    /// where `A` is an N×N triangular matrix stored in packed format.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `trsv`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA`.
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    ///
    /// **Storage requirement**: `AP` must be a contiguous slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn tpsv(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        A: BlasPacked(Scalar, AxisA),
        diag: Diag,
        x: NamedArray(AxisX, Scalar),
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_stpsv,
            f64 => acc.cblas_dtpsv,
            Complex(f32) => acc.cblas_ctpsv,
            Complex(f64) => acc.cblas_ztpsv,
            else => @compileError("tpsv is incompatible with given Scalar type."),
        };

        const x_blas = Blas1dMut(Scalar).init(AxisX, x);
        A.validate(@intCast(x_blas.len));

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        f(
            A.layout(),
            A.uplo(),
            @intCast(acc.CblasNoTrans),
            diag_blas,
            x_blas.len, // N
            A.ptr(),
            x_blas.ptr,
            x_blas.inc,
        );
    }

    /// `sspr` and `dspr` in BLAS.
    /// Computes `A := alpha * x * xᵀ + A` (symmetric packed rank-1 update) for real scalars.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `syr`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA`.
    /// The scalar `alpha` is optional and defaults to 1.
    ///
    /// **Storage requirement**: `AP` must be a contiguous mutable slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn spr(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        A: BlasPackedMut(Scalar, AxisA),
        x: NamedArrayConst(AxisX, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_sspr,
            f64 => acc.cblas_dspr,
            else => @compileError("spr requires f32 or f64."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        A.validate(@intCast(x_blas.len));

        f(
            A.layout(),
            A.uplo(),
            x_blas.len, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            A.ptr(),
        );
    }

    /// `chpr` and `zhpr` in BLAS.
    /// Computes `A := alpha * x * xᴴ + A` (Hermitian packed rank-1 update) for complex scalars.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `her`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA`.
    /// The scalar `alpha` is **real** and optional, defaulting to 1.
    ///
    /// **Storage requirement**: `AP` must be a contiguous mutable slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn hpr(
        comptime RealScalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        A: BlasPackedMut(Complex(RealScalar), AxisA),
        x: NamedArrayConst(AxisX, Complex(RealScalar)),
        scalars: struct { alpha: RealScalar = 1.0 },
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (RealScalar) {
            f32 => acc.cblas_chpr,
            f64 => acc.cblas_zhpr,
            else => @compileError("hpr requires f32 or f64 as RealScalar."),
        };

        const Scalar = Complex(RealScalar);
        const x_blas = Blas1d(Scalar).init(AxisX, x);
        A.validate(@intCast(x_blas.len));

        f(
            A.layout(),
            A.uplo(),
            x_blas.len, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            A.ptr(),
        );
    }

    /// `sspr2` and `dspr2` in BLAS.
    /// Computes `A := alpha * x * yᵀ + alpha * y * xᵀ + A` (symmetric packed rank-2 update)
    /// for real scalars.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `syr2`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA` and `y`'s axis the other.
    /// The scalar `alpha` is optional and defaults to 1.
    ///
    /// **Storage requirement**: `AP` must be a contiguous mutable slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn spr2(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: BlasPackedMut(Scalar, AxisA),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            f32 => acc.cblas_sspr2,
            f64 => acc.cblas_dspr2,
            else => @compileError("spr2 requires f32 or f64."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1d(Scalar).init(AxisY, y);

        assert(x_blas.len == y_blas.len); // Square matrix: x and y dimensions must match
        A.validate(@intCast(x_blas.len));

        f(
            A.layout(),
            A.uplo(),
            x_blas.len, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            A.ptr(),
        );
    }

    /// `chpr2` and `zhpr2` in BLAS.
    /// Computes `A := alpha * x * yᴴ + conj(alpha) * y * xᴴ + A` (Hermitian packed rank-2 update)
    /// for complex scalars.
    /// `AxisA` is the 2D conceptual axis type of `A`; `triangle` selects which triangle
    /// is stored (second axis → upper, first axis → lower), using the same convention
    /// as `her2`. `major` selects the packing axis: first axis → row-major,
    /// second axis → column-major.
    /// `x`'s axis must match one axis of `AxisA` and `y`'s axis the other.
    /// The scalar `alpha` is complex and optional, defaulting to 1.
    ///
    /// **Storage requirement**: `AP` must be a contiguous mutable slice of exactly
    /// `N * (N + 1) / 2` elements in the packed order specified by `major`.
    pub fn hpr2(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: BlasPackedMut(Scalar, AxisA),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_chpr2,
            Complex(f64) => acc.cblas_zhpr2,
            else => @compileError("hpr2 requires Complex(f32) or Complex(f64)."),
        };

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1d(Scalar).init(AxisY, y);

        assert(x_blas.len == y_blas.len); // Square matrix: x and y dimensions must match
        A.validate(@intCast(x_blas.len));

        f(
            A.layout(),
            A.uplo(),
            x_blas.len, // N
            &scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            A.ptr(),
        );
    }

    /// `sger` and `dger` in BLAS.
    /// Computes `A := alpha * x * yᵀ + A` (rank-1 update) for real scalars.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn ger(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        const x_axis_idx = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            f32 => acc.cblas_sger,
            f64 => acc.cblas_dger,
            else => @compileError("ger requires f32 or f64."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("ger: dimension mismatch");

        const A_blas = Blas2dMut(Scalar).fromNamedArray(AxisA, A, x_axis_idx);
        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1d(Scalar).init(AxisY, y);

        f(
            A_blas.layout,
            A_blas.rows, // M
            A_blas.cols, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            A_blas.ptr,
            A_blas.leading,
        );
    }

    /// `cgeru` and `zgeru` in BLAS.
    /// Computes `A := alpha * x * yᵀ + A` (rank-1 update, unconjugated) for complex scalars.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn geru(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        geruc(Scalar, AxisA, AxisX, AxisY, false, A, x, y, scalars.alpha);
    }

    /// `cgerc` and `zgerc` in BLAS.
    /// Computes `A := alpha * x * yᴴ + A` (rank-1 update, conjugated) for complex scalars.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn gerc(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        geruc(Scalar, AxisA, AxisX, AxisY, true, A, x, y, scalars.alpha);
    }

    fn geruc(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        comptime conjugated: bool,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        alpha: Scalar,
    ) void {
        const x_axis_idx = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            Complex(f32) => if (conjugated) acc.cblas_cgerc else acc.cblas_cgeru,
            Complex(f64) => if (conjugated) acc.cblas_zgerc else acc.cblas_zgeru,
            else => @compileError("geru/gerc requires Complex(f32) or Complex(f64)."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic(if (conjugated) "gerc: dimension mismatch" else "geru: dimension mismatch");

        const A_blas = Blas2dMut(Scalar).fromNamedArray(AxisA, A, x_axis_idx);
        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1d(Scalar).init(AxisY, y);

        f(
            A_blas.layout,
            A_blas.rows, // M
            A_blas.cols, // N
            &alpha,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            A_blas.ptr,
            A_blas.leading,
        );
    }

    /// `ssyr` and `dsyr` in BLAS.
    /// Computes `A := alpha * x * xᵀ + A` (symmetric rank-1 update) for real scalars.
    /// Only the triangle of `A` where `triangle >= the other axis` is read and written.
    /// `x`'s axis must match one axis of `A`.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn syr(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssyr,
            f64 => acc.cblas_dsyr,
            else => @compileError("syr requires f32 or f64."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape }) catch
            @panic("syr: dimension mismatch");

        const A_blas = Blas2dMut(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisX, x);

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
            A_blas.rows, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            A_blas.ptr,
            A_blas.leading,
        );
    }

    /// `cher` and `zher` in BLAS.
    /// Computes `A := alpha * x * xᴴ + A` (Hermitian rank-1 update) for complex scalars.
    /// Only the triangle of `A` where `triangle >= the other axis` is read and written.
    /// `x`'s axis must match one axis of `A`.
    /// The scalar `alpha` is real and optional, defaulting to 1.
    pub fn her(
        comptime RealScalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        triangle: AxisA,
        A: NamedArray(AxisA, Complex(RealScalar)),
        x: NamedArrayConst(AxisX, Complex(RealScalar)),
        scalars: struct { alpha: RealScalar = 1.0 },
    ) void {
        const Scalar = Complex(RealScalar);
        comptime assertMatchingAxis(AxisA, AxisX);
        const f = switch (RealScalar) {
            f32 => acc.cblas_cher,
            f64 => acc.cblas_zher,
            else => @compileError("her requires f32 or f64 as RealScalar."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape }) catch
            @panic("her: dimension mismatch");

        const A_blas = Blas2dMut(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisX, x);

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
            A_blas.rows, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            A_blas.ptr,
            A_blas.leading,
        );
    }

    /// `ssyr2` and `dsyr2` in BLAS.
    /// Computes `A := alpha * x * yᵀ + alpha * y * xᵀ + A` (symmetric rank-2 update) for real scalars.
    /// Only the triangle of `A` where `triangle >= the other axis` is read and written.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn syr2(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        triangle: AxisA,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssyr2,
            f64 => acc.cblas_dsyr2,
            else => @compileError("syr2 requires f32 or f64."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("syr2: dimension mismatch");

        const A_blas = Blas2dMut(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1d(Scalar).init(AxisY, y);

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
            A_blas.rows, // N
            scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            A_blas.ptr,
            A_blas.leading,
        );
    }

    /// `cher2` and `zher2` in BLAS.
    /// Computes `A := alpha * x * yᴴ + conj(alpha) * y * xᴴ + A` (Hermitian rank-2 update) for complex scalars.
    /// Only the triangle of `A` where `triangle >= the other axis` is read and written.
    /// `x`'s axis must match one axis of `A` and `y`'s axis must match the other.
    /// The scalar `alpha` is complex and optional, defaulting to 1.
    pub fn her2(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisX: type,
        comptime AxisY: type,
        triangle: AxisA,
        A: NamedArray(AxisA, Scalar),
        x: NamedArrayConst(AxisX, Scalar),
        y: NamedArrayConst(AxisY, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        _ = comptime matchingAxisIdx(AxisA, AxisX, AxisY);
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_cher2,
            Complex(f64) => acc.cblas_zher2,
            else => @compileError("her2 requires Complex(f32) or Complex(f64)."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, x.idx.shape, y.idx.shape }) catch
            @panic("her2: dimension mismatch");

        const A_blas = Blas2dMut(Scalar).fromNamedArray(AxisA, A, 0);
        assert(A_blas.rows == A_blas.cols);

        const x_blas = Blas1d(Scalar).init(AxisX, x);
        const y_blas = Blas1d(Scalar).init(AxisY, y);

        f(
            A_blas.layout,
            uploBlas(AxisA, triangle),
            A_blas.rows, // N
            &scalars.alpha,
            x_blas.ptr,
            x_blas.inc,
            y_blas.ptr,
            y_blas.inc,
            A_blas.ptr,
            A_blas.leading,
        );
    }

    // ─────────────────────────────────────────────────────────────
    //  Level 3 BLAS — matrix-matrix operations
    // ─────────────────────────────────────────────────────────────

    /// `sgemm`, `dgemm`, `cgemm` and `zgemm` in BLAS.
    /// Computes `C := alpha * A * B + beta * C` (general matrix-matrix multiply).
    /// The contraction axis is the axis name shared between `A` and `B` but absent from `C`.
    /// `C`'s two axes must match the remaining (non-contraction) axes of `A` and `B`.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn gemm(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        comptime AxisC: type,
        A: NamedArrayConst(AxisA, Scalar),
        B: NamedArrayConst(AxisB, Scalar),
        C: NamedArray(AxisC, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const axes = comptime gemmAxisNames(AxisA, AxisB, AxisC);
        const f = switch (Scalar) {
            f32 => acc.cblas_sgemm,
            f64 => acc.cblas_dgemm,
            Complex(f32) => acc.cblas_cgemm,
            Complex(f64) => acc.cblas_zgemm,
            else => @compileError("gemm is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, B.idx.shape, C.idx.shape }) catch
            @panic("gemm: dimension mismatch");

        // C: i_name → rows (M), j_name → cols (N)
        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C, comptime nameIdx(AxisC, axes.i_name));
        // A: i_name → rows (M), k_name → cols (K)
        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, comptime nameIdx(AxisA, axes.i_name));
        // B: k_name → rows (K), j_name → cols (N)
        const B_blas = Blas2d(Scalar).fromNamedArray(AxisB, B, comptime nameIdx(AxisB, axes.k_name));

        // Determine transpose flags based on layout comparison with C
        const transA: acc.CBLAS_TRANSPOSE = if (A_blas.layout == C_blas.layout)
            @intCast(acc.CblasNoTrans)
        else
            @intCast(acc.CblasTrans);
        const transB: acc.CBLAS_TRANSPOSE = if (B_blas.layout == C_blas.layout)
            @intCast(acc.CblasNoTrans)
        else
            @intCast(acc.CblasTrans);

        const M = C_blas.rows;
        const N = C_blas.cols;
        const K = A_blas.cols; // K dimension

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;
        const beta_blas = if (comptime isComplex(Scalar)) &scalars.beta else scalars.beta;

        f(
            C_blas.layout,
            transA,
            transB,
            M,
            N,
            K,
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
            beta_blas,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `ssymm`, `dsymm`, `csymm` and `zsymm` in BLAS.
    /// Computes `C := alpha * A * B + beta * C` where `A` is a symmetric matrix.
    /// Axis matching follows the `gemm` convention (one shared contraction axis between `A` and `B`,
    /// remaining axes match `C`). `A` must be square (asserted at runtime).
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// `B` and `C` must have the same physical layout.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn symm(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        comptime AxisC: type,
        A: BlasTriangular(Scalar, AxisA),
        B: NamedArrayConst(AxisB, Scalar),
        C: NamedArray(AxisC, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const axes = comptime gemmAxisNames(AxisA, AxisB, AxisC);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssymm,
            f64 => acc.cblas_dsymm,
            Complex(f32) => acc.cblas_csymm,
            Complex(f64) => acc.cblas_zsymm,
            else => @compileError("symm is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.data.idx.shape, B.idx.shape, C.idx.shape }) catch
            @panic("symm: dimension mismatch");

        // A: preserve axis order for uploBlas correctness
        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A.data, 0);
        assert(A_blas.rows == A_blas.cols); // square

        // B: contraction axis → rows, j-axis → cols
        const B_blas = Blas2d(Scalar).fromNamedArray(AxisB, B, comptime nameIdx(AxisB, axes.k_name));
        // C: i-axis → rows (M), j-axis → cols (N)
        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C, comptime nameIdx(AxisC, axes.i_name));

        assert(B_blas.layout == C_blas.layout); // B and C must have same layout

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;
        const beta_blas = if (comptime isComplex(Scalar)) &scalars.beta else scalars.beta;

        f(
            C_blas.layout,
            @intCast(acc.CblasLeft), // always left via axis renaming
            A.uplo(),
            C_blas.rows, // M
            C_blas.cols, // N
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
            beta_blas,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `chemm` and `zhemm` in BLAS.
    /// Computes `C := alpha * A * B + beta * C` where `A` is a Hermitian matrix.
    /// Axis matching follows the `gemm` convention. `A` must be square (asserted at runtime).
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// `B` and `C` must have the same physical layout.
    /// The scalars `alpha` and `beta` are complex, optional and default to 1.
    pub fn hemm(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        comptime AxisC: type,
        A: BlasTriangular(Scalar, AxisA),
        B: NamedArrayConst(AxisB, Scalar),
        C: NamedArray(AxisC, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const axes = comptime gemmAxisNames(AxisA, AxisB, AxisC);
        const f = switch (Scalar) {
            Complex(f32) => acc.cblas_chemm,
            Complex(f64) => acc.cblas_zhemm,
            else => @compileError("hemm requires Complex(f32) or Complex(f64)."),
        };

        _ = named_index.resolveDimensions(.{ A.data.idx.shape, B.idx.shape, C.idx.shape }) catch
            @panic("hemm: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A.data, 0);
        assert(A_blas.rows == A_blas.cols);

        const B_blas = Blas2d(Scalar).fromNamedArray(AxisB, B, comptime nameIdx(AxisB, axes.k_name));
        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C, comptime nameIdx(AxisC, axes.i_name));

        assert(B_blas.layout == C_blas.layout);

        f(
            C_blas.layout,
            @intCast(acc.CblasLeft),
            A.uplo(),
            C_blas.rows,
            C_blas.cols,
            &scalars.alpha,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
            &scalars.beta,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `ssyrk`, `dsyrk`, `csyrk` and `zsyrk` in BLAS.
    /// Computes `C := alpha * A * Aᵀ + beta * C` (symmetric rank-k update).
    /// `A` has one axis matching an axis of `C` (the N dimension) and one that does not (the K dimension).
    /// `C` must be square (asserted at runtime).
    /// Only the triangle of `C` where `triangle >= the other axis` is read and written.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn syrk(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisC: type,
        A: NamedArrayConst(AxisA, Scalar),
        C: BlasTriangularMut(Scalar, AxisC),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const axes = comptime rankKAxisNames(AxisA, AxisC);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssyrk,
            f64 => acc.cblas_dsyrk,
            Complex(f32) => acc.cblas_csyrk,
            Complex(f64) => acc.cblas_zsyrk,
            else => @compileError("syrk is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, C.data.idx.shape }) catch
            @panic("syrk: dimension mismatch");

        // C: preserve axis order for uploBlas correctness
        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C.data, 0);
        assert(C_blas.rows == C_blas.cols); // square

        // A: N-axis → rows, K-axis → cols
        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, comptime nameIdx(AxisA, axes.n_name));

        const trans: acc.CBLAS_TRANSPOSE = if (A_blas.layout == C_blas.layout)
            @intCast(acc.CblasNoTrans)
        else
            @intCast(acc.CblasTrans);

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;
        const beta_blas = if (comptime isComplex(Scalar)) &scalars.beta else scalars.beta;

        f(
            C_blas.layout,
            C.uplo(),
            trans,
            C_blas.rows, // N
            A_blas.cols, // K
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            beta_blas,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `cherk` and `zherk` in BLAS.
    /// Computes `C := alpha * A * Aᴴ + beta * C` (Hermitian rank-k update) for complex scalars.
    /// `A` has one axis matching an axis of `C` (the N dimension) and one that does not (the K dimension).
    /// `C` must be square (asserted at runtime).
    /// Only the triangle of `C` where `triangle >= the other axis` is read and written.
    /// Both `alpha` and `beta` are real scalars, optional and default to 1.
    pub fn herk(
        comptime RealScalar: type,
        comptime AxisA: type,
        comptime AxisC: type,
        A: NamedArrayConst(AxisA, Complex(RealScalar)),
        C: BlasTriangularMut(Complex(RealScalar), AxisC),
        scalars: struct { alpha: RealScalar = 1.0, beta: RealScalar = 1.0 },
    ) void {
        const Scalar = Complex(RealScalar);
        const axes = comptime rankKAxisNames(AxisA, AxisC);
        const f = switch (RealScalar) {
            f32 => acc.cblas_cherk,
            f64 => acc.cblas_zherk,
            else => @compileError("herk requires f32 or f64 as RealScalar."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, C.data.idx.shape }) catch
            @panic("herk: dimension mismatch");

        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C.data, 0);
        assert(C_blas.rows == C_blas.cols);

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, comptime nameIdx(AxisA, axes.n_name));

        const trans: acc.CBLAS_TRANSPOSE = if (A_blas.layout == C_blas.layout)
            @intCast(acc.CblasNoTrans)
        else
            @intCast(acc.CblasTrans);

        f(
            C_blas.layout,
            C.uplo(),
            trans,
            C_blas.rows, // N
            A_blas.cols, // K
            scalars.alpha,
            A_blas.ptr,
            A_blas.leading,
            scalars.beta,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `ssyr2k`, `dsyr2k`, `csyr2k` and `zsyr2k` in BLAS.
    /// Computes `C := alpha * A * Bᵀ + alpha * B * Aᵀ + beta * C` (symmetric rank-2k update).
    /// Axis matching follows the `gemm` convention (one shared contraction axis between `A` and `B`,
    /// remaining axes match `C`). `C` must be square (asserted at runtime).
    /// Only the triangle of `C` where `triangle >= the other axis` is read and written.
    /// `A` and `B` must have the same physical layout.
    /// The scalars `alpha` and `beta` are optional and default to 1.
    pub fn syr2k(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        comptime AxisC: type,
        A: NamedArrayConst(AxisA, Scalar),
        B: NamedArrayConst(AxisB, Scalar),
        C: BlasTriangularMut(Scalar, AxisC),
        scalars: struct { alpha: Scalar = one(Scalar), beta: Scalar = one(Scalar) },
    ) void {
        const axes = comptime gemmAxisNames(AxisA, AxisB, AxisC);
        const f = switch (Scalar) {
            f32 => acc.cblas_ssyr2k,
            f64 => acc.cblas_dsyr2k,
            Complex(f32) => acc.cblas_csyr2k,
            Complex(f64) => acc.cblas_zsyr2k,
            else => @compileError("syr2k is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, B.idx.shape, C.data.idx.shape }) catch
            @panic("syr2k: dimension mismatch");

        // C: preserve axis order for uploBlas correctness
        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C.data, 0);
        assert(C_blas.rows == C_blas.cols);

        // A: axis matching C (i_name) → rows (N), contraction axis → cols (K)
        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, comptime nameIdx(AxisA, axes.i_name));
        // B: axis matching C (j_name) → rows (N), contraction axis → cols (K)
        const B_blas = Blas2d(Scalar).fromNamedArray(AxisB, B, comptime nameIdx(AxisB, axes.j_name));

        assert(A_blas.layout == B_blas.layout); // A and B must have same layout

        const trans: acc.CBLAS_TRANSPOSE = if (A_blas.layout == C_blas.layout)
            @intCast(acc.CblasNoTrans)
        else
            @intCast(acc.CblasTrans);

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;
        const beta_blas = if (comptime isComplex(Scalar)) &scalars.beta else scalars.beta;

        f(
            C_blas.layout,
            C.uplo(),
            trans,
            C_blas.rows, // N
            A_blas.cols, // K
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
            beta_blas,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `cher2k` and `zher2k` in BLAS.
    /// Computes `C := alpha * A * Bᴴ + conj(alpha) * B * Aᴴ + beta * C` (Hermitian rank-2k update).
    /// Axis matching follows the `gemm` convention. `C` must be square (asserted at runtime).
    /// Only the triangle of `C` where `triangle >= the other axis` is read and written.
    /// `A` and `B` must have the same physical layout.
    /// `alpha` is complex, `beta` is real; both are optional and default to 1.
    pub fn her2k(
        comptime RealScalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        comptime AxisC: type,
        A: NamedArrayConst(AxisA, Complex(RealScalar)),
        B: NamedArrayConst(AxisB, Complex(RealScalar)),
        C: BlasTriangularMut(Complex(RealScalar), AxisC),
        scalars: struct { alpha: Complex(RealScalar) = .{ .re = 1.0, .im = 0.0 }, beta: RealScalar = 1.0 },
    ) void {
        const Scalar = Complex(RealScalar);
        const axes = comptime gemmAxisNames(AxisA, AxisB, AxisC);
        const f = switch (RealScalar) {
            f32 => acc.cblas_cher2k,
            f64 => acc.cblas_zher2k,
            else => @compileError("her2k requires f32 or f64 as RealScalar."),
        };

        _ = named_index.resolveDimensions(.{ A.idx.shape, B.idx.shape, C.data.idx.shape }) catch
            @panic("her2k: dimension mismatch");

        const C_blas = Blas2dMut(Scalar).fromNamedArray(AxisC, C.data, 0);
        assert(C_blas.rows == C_blas.cols);

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A, comptime nameIdx(AxisA, axes.i_name));
        const B_blas = Blas2d(Scalar).fromNamedArray(AxisB, B, comptime nameIdx(AxisB, axes.j_name));

        assert(A_blas.layout == B_blas.layout);

        const trans: acc.CBLAS_TRANSPOSE = if (A_blas.layout == C_blas.layout)
            @intCast(acc.CblasNoTrans)
        else
            @intCast(acc.CblasTrans);

        f(
            C_blas.layout,
            C.uplo(),
            trans,
            C_blas.rows, // N
            A_blas.cols, // K
            &scalars.alpha,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
            scalars.beta,
            C_blas.ptr,
            C_blas.leading,
        );
    }

    /// `strmm`, `dtrmm`, `ctrmm` and `ztrmm` in BLAS.
    /// Computes `B := alpha * A * B` in-place where `A` is a triangular matrix.
    /// Exactly one axis of `A` must match one axis of `B`; this determines the side of multiplication.
    /// `A` must be square (asserted at runtime).
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn trmm(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        A: BlasTriangular(Scalar, AxisA),
        diag: Diag,
        B: NamedArray(AxisB, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        const match = comptime trmmAxisMatch(AxisA, AxisB);
        const f = switch (Scalar) {
            f32 => acc.cblas_strmm,
            f64 => acc.cblas_dtrmm,
            Complex(f32) => acc.cblas_ctrmm,
            Complex(f64) => acc.cblas_ztrmm,
            else => @compileError("trmm is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.data.idx.shape, B.idx.shape }) catch
            @panic("trmm: dimension mismatch");

        // Preserve axis order for both A and B
        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A.data, 0);
        assert(A_blas.rows == A_blas.cols); // square

        const B_blas = Blas2dMut(Scalar).fromNamedArray(AxisB, B, 0);

        // Determine side: which axis of B does A share with?
        const side: acc.CBLAS_SIDE = @intCast(if (match.b_match_idx == 0)
            acc.CblasLeft
        else
            acc.CblasRight);

        // Determine transA: if A's layout differs from B's, we need Trans + uplo swap
        const a_needs_trans = (A_blas.layout != B_blas.layout);
        const transA: acc.CBLAS_TRANSPOSE = @intCast(if (a_needs_trans)
            acc.CblasTrans
        else
            acc.CblasNoTrans);
        const uplo: acc.CBLAS_UPLO = blk: {
            const base = A.uplo();
            if (a_needs_trans) {
                // Swap upper↔lower to compensate for the transpose
                break :blk if (base == @as(acc.CBLAS_UPLO, @intCast(acc.CblasUpper)))
                    @as(acc.CBLAS_UPLO, @intCast(acc.CblasLower))
                else
                    @as(acc.CBLAS_UPLO, @intCast(acc.CblasUpper));
            }
            break :blk base;
        };

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;

        f(
            B_blas.layout,
            side,
            uplo,
            transA,
            diag_blas,
            B_blas.rows, // M
            B_blas.cols, // N
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
        );
    }

    /// `strsm`, `dtrsm`, `ctrsm` and `ztrsm` in BLAS.
    /// Solves `A * X = alpha * B` in-place (i.e. computes `B := alpha * A⁻¹ * B`) where `A` is triangular.
    /// Exactly one axis of `A` must match one axis of `B`; this determines the side of the solve.
    /// `A` must be square (asserted at runtime).
    /// Only the triangle of `A` where `triangle >= the other axis` is read.
    /// If `diag` is `.unit`, the diagonal of `A` is assumed to be all ones and is not read.
    /// The scalar `alpha` is optional and defaults to 1.
    pub fn trsm(
        comptime Scalar: type,
        comptime AxisA: type,
        comptime AxisB: type,
        A: BlasTriangular(Scalar, AxisA),
        diag: Diag,
        B: NamedArray(AxisB, Scalar),
        scalars: struct { alpha: Scalar = one(Scalar) },
    ) void {
        const match = comptime trmmAxisMatch(AxisA, AxisB);
        const f = switch (Scalar) {
            f32 => acc.cblas_strsm,
            f64 => acc.cblas_dtrsm,
            Complex(f32) => acc.cblas_ctrsm,
            Complex(f64) => acc.cblas_ztrsm,
            else => @compileError("trsm is incompatible with given Scalar type."),
        };

        _ = named_index.resolveDimensions(.{ A.data.idx.shape, B.idx.shape }) catch
            @panic("trsm: dimension mismatch");

        const A_blas = Blas2d(Scalar).fromNamedArray(AxisA, A.data, 0);
        assert(A_blas.rows == A_blas.cols);

        const B_blas = Blas2dMut(Scalar).fromNamedArray(AxisB, B, 0);

        const side: acc.CBLAS_SIDE = @intCast(if (match.b_match_idx == 0)
            acc.CblasLeft
        else
            acc.CblasRight);

        const a_needs_trans = (A_blas.layout != B_blas.layout);
        const transA: acc.CBLAS_TRANSPOSE = @intCast(if (a_needs_trans)
            acc.CblasTrans
        else
            acc.CblasNoTrans);
        const uplo: acc.CBLAS_UPLO = blk: {
            const base = A.uplo();
            if (a_needs_trans) {
                break :blk if (base == @as(acc.CBLAS_UPLO, @intCast(acc.CblasUpper)))
                    @as(acc.CBLAS_UPLO, @intCast(acc.CblasLower))
                else
                    @as(acc.CBLAS_UPLO, @intCast(acc.CblasUpper));
            }
            break :blk base;
        };

        const diag_blas: acc.CBLAS_DIAG = switch (diag) {
            .unit => @intCast(acc.CblasUnit),
            .non_unit => @intCast(acc.CblasNonUnit),
        };

        const alpha_blas = if (comptime isComplex(Scalar)) &scalars.alpha else scalars.alpha;

        f(
            B_blas.layout,
            side,
            uplo,
            transA,
            diag_blas,
            B_blas.rows, // M
            B_blas.cols, // N
            alpha_blas,
            A_blas.ptr,
            A_blas.leading,
            B_blas.ptr,
            B_blas.leading,
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

    pub const Diag = enum { unit, non_unit };

    pub const IJ = enum { i, j };
    const I = enum { i };

    fn one(comptime T: type) T {
        return switch (T) {
            f32, f64 => 1.0,
            Complex(f32), Complex(f64) => .{ .re = 1.0, .im = 0.0 },
            else => @compileError("one: T must be f32, f64 or Complex(...)"),
        };
    }

    fn isComplex(comptime T: type) bool {
        return T == Complex(f32) or T == Complex(f64);
    }

    /// Maps a triangle axis value to the corresponding CBLAS UPLO constant.
    /// Second axis (enum ordinal 1) → Upper; first axis (ordinal 0) → Lower.
    fn uploBlas(comptime AxisA: type, triangle: AxisA) acc.CBLAS_UPLO {
        return if (@intFromEnum(triangle) == 1)
            @intCast(acc.CblasUpper)
        else
            @intCast(acc.CblasLower);
    }

    /// Maps a major-axis value to the corresponding CBLAS layout constant.
    /// First axis (ordinal 0) as major → row-major; second axis (ordinal 1) → column-major.
    fn layoutBlas(comptime AxisA: type, major: AxisA) acc.CBLAS_ORDER {
        return if (@intFromEnum(major) == 0)
            @intCast(acc.CblasRowMajor)
        else
            @intCast(acc.CblasColMajor);
    }

    /// Validates that AxisA is 2D, AxisX and AxisY are each 1D with distinct names,
    /// and each matches a different axis of AxisA.
    /// Returns the index (0 or 1) of the AxisA field that matches AxisX.
    fn matchingAxisIdx(comptime AxisA: type, comptime AxisX: type, comptime AxisY: type) usize {
        const a_names = meta.fieldNames(AxisA);
        assert(a_names.len == 2);
        assert(meta.fields(AxisX).len == 1);
        assert(meta.fields(AxisY).len == 1);
        const x_name = meta.fields(AxisX)[0].name;
        const y_name = meta.fields(AxisY)[0].name;
        assert(!std.mem.eql(u8, x_name, y_name));
        assert(std.mem.eql(u8, x_name, a_names[0]) or std.mem.eql(u8, x_name, a_names[1]));
        assert(std.mem.eql(u8, y_name, a_names[0]) or std.mem.eql(u8, y_name, a_names[1]));
        return if (std.mem.eql(u8, x_name, a_names[0])) 0 else 1;
    }

    /// Validates that AxisA is 2D, AxisX is 1D, and AxisX's name matches one axis of AxisA.
    fn assertMatchingAxis(comptime AxisA: type, comptime AxisX: type) void {
        const a_names = meta.fieldNames(AxisA);
        assert(a_names.len == 2);
        assert(meta.fields(AxisX).len == 1);
        const x_name = meta.fields(AxisX)[0].name;
        assert(std.mem.eql(u8, x_name, a_names[0]) or std.mem.eql(u8, x_name, a_names[1]));
    }

    /// For gemm-family operations: validates that AxisA, AxisB, AxisC are all 2D.
    /// Finds the contraction axis (shared between A and B, absent from C),
    /// the i-axis (shared between A and C, absent from B → M dimension),
    /// and the j-axis (shared between B and C, absent from A → N dimension).
    fn gemmAxisNames(comptime AxisA: type, comptime AxisB: type, comptime AxisC: type) struct {
        i_name: [:0]const u8, // A ∩ C \ B (M dim)
        j_name: [:0]const u8, // B ∩ C \ A (N dim)
        k_name: [:0]const u8, // A ∩ B \ C (contraction)
    } {
        const a_names = meta.fieldNames(AxisA);
        const b_names = meta.fieldNames(AxisB);
        const c_names = meta.fieldNames(AxisC);
        assert(a_names.len == 2);
        assert(b_names.len == 2);
        assert(c_names.len == 2);

        var i_name: ?[:0]const u8 = null;
        var j_name: ?[:0]const u8 = null;
        var k_name: ?[:0]const u8 = null;

        // Find contraction axis: in A and B, not in C
        for (a_names) |an| {
            const in_b = nameIn(an, b_names);
            const in_c = nameIn(an, c_names);
            if (in_b and !in_c) {
                assert(k_name == null); // at most one contraction axis
                k_name = an;
            } else if (!in_b and in_c) {
                assert(i_name == null);
                i_name = an;
            } else {
                assert(false); // axis must be in exactly one of {B, C}
            }
        }
        // Find j_name: in B and C, not in A
        for (b_names) |bn| {
            const in_a = nameIn(bn, a_names);
            const in_c = nameIn(bn, c_names);
            if (!in_a and in_c) {
                assert(j_name == null);
                j_name = bn;
            } else if (in_a and !in_c) {
                // This is the contraction axis, already found
                assert(std.mem.eql(u8, bn, k_name.?));
            } else {
                assert(false);
            }
        }

        assert(i_name != null and j_name != null and k_name != null);
        assert(!std.mem.eql(u8, i_name.?, j_name.?));
        return .{ .i_name = i_name.?, .j_name = j_name.?, .k_name = k_name.? };
    }

    /// For rank-k operations (syrk, herk): validates that AxisA is 2D and AxisC is 2D.
    /// Finds the N-axis (shared between A and C) and the K-axis (in A but not C).
    fn rankKAxisNames(comptime AxisA: type, comptime AxisC: type) struct {
        n_name: [:0]const u8,
        k_name: [:0]const u8,
    } {
        const a_names = meta.fieldNames(AxisA);
        const c_names = meta.fieldNames(AxisC);
        assert(a_names.len == 2);
        assert(c_names.len == 2);

        var n_name: ?[:0]const u8 = null;
        var k_name: ?[:0]const u8 = null;

        for (a_names) |an| {
            if (nameIn(an, c_names)) {
                assert(n_name == null);
                n_name = an;
            } else {
                assert(k_name == null);
                k_name = an;
            }
        }
        assert(n_name != null and k_name != null);
        return .{ .n_name = n_name.?, .k_name = k_name.? };
    }

    /// For trmm/trsm: validates that AxisA is 2D and AxisB is 2D.
    /// Finds which axis of A matches which axis of B.
    /// Returns the index (0 or 1) of the matching axis in both A and B.
    fn trmmAxisMatch(comptime AxisA: type, comptime AxisB: type) struct {
        a_match_idx: usize,
        b_match_idx: usize,
    } {
        const a_names = meta.fieldNames(AxisA);
        const b_names = meta.fieldNames(AxisB);
        assert(a_names.len == 2);
        assert(b_names.len == 2);

        var a_idx: ?usize = null;
        var b_idx: ?usize = null;
        for (a_names, 0..) |an, ai| {
            for (b_names, 0..) |bn, bi| {
                if (std.mem.eql(u8, an, bn)) {
                    assert(a_idx == null); // at most one match
                    a_idx = ai;
                    b_idx = bi;
                }
            }
        }
        assert(a_idx != null and b_idx != null);
        return .{ .a_match_idx = a_idx.?, .b_match_idx = b_idx.? };
    }

    /// Checks whether a name appears in a list of names (comptime).
    fn nameIn(name: [:0]const u8, names: []const [:0]const u8) bool {
        for (names) |n| {
            if (std.mem.eql(u8, name, n)) return true;
        }
        return false;
    }

    /// Returns the ordinal index (0 or 1) of the given axis name within a 2D axis type.
    fn nameIdx(comptime Axis: type, comptime name: [:0]const u8) usize {
        const names = meta.fieldNames(Axis);
        assert(names.len == 2);
        if (std.mem.eql(u8, name, names[0])) return 0;
        if (std.mem.eql(u8, name, names[1])) return 1;
        unreachable;
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
                assert(arr.idx.strides.i > 0);
                assert(arr.idx.strides.j > 0);
                const si: usize = @intCast(arr.idx.strides.i);
                const sj: usize = @intCast(arr.idx.strides.j);

                // The minor axis must have stride 1 (contiguous).
                // The major axis stride is the leading dimension (lda),
                // which may exceed the minor axis shape (submatrix view / padding).
                if (sj == 1 and si >= arr.idx.shape.j) {
                    // Row-major: j is contiguous, lda = stride along i
                    return .{
                        .layout = @intCast(acc.CblasRowMajor),
                        .rows = @intCast(arr.idx.shape.i),
                        .cols = @intCast(arr.idx.shape.j),
                        .leading = @intCast(si),
                        .ptr = @ptrCast(arr.buf.ptr),
                    };
                } else if (si == 1 and sj >= arr.idx.shape.i) {
                    // Col-major: i is contiguous, lda = stride along j
                    return .{
                        .layout = @intCast(acc.CblasColMajor),
                        .rows = @intCast(arr.idx.shape.i),
                        .cols = @intCast(arr.idx.shape.j),
                        .leading = @intCast(sj),
                        .ptr = @ptrCast(arr.buf.ptr),
                    };
                } else {
                    @panic("Blas2d: minor axis must have stride 1, and major axis stride must be >= minor axis shape");
                }
            }

            /// Creates a Blas2d from an arbitrary 2D NamedArrayConst by renaming axes to IJ.
            /// `i_axis` selects which AxisA field (0 or 1) maps to i; the other maps to j.
            fn fromNamedArray(comptime AxisA: type, arr: NamedArrayConst(AxisA, Scalar), comptime i_axis: usize) @This() {
                const rename_pairs = comptime [_]named_index.AxisRenamePair{
                    .{ .old = meta.fieldNames(AxisA)[i_axis], .new = "i" },
                    .{ .old = meta.fieldNames(AxisA)[1 - i_axis], .new = "j" },
                };
                const a_ij_idx = arr.idx.rename(IJ, &rename_pairs);
                const arr_ij: NamedArrayConst(IJ, Scalar) = .{ .idx = a_ij_idx, .buf = arr.buf };
                return init(arr_ij);
            }
        };
    }

    fn Blas2dMut(comptime Scalar: type) type {
        return struct {
            layout: acc.CBLAS_ORDER,
            rows: c_int,
            cols: c_int,
            leading: c_int,
            ptr: *Scalar,

            fn init(arr: NamedArray(IJ, Scalar)) @This() {
                assert(arr.idx.strides.i > 0);
                assert(arr.idx.strides.j > 0);
                const si: usize = @intCast(arr.idx.strides.i);
                const sj: usize = @intCast(arr.idx.strides.j);

                if (sj == 1 and si >= arr.idx.shape.j) {
                    return .{
                        .layout = @intCast(acc.CblasRowMajor),
                        .rows = @intCast(arr.idx.shape.i),
                        .cols = @intCast(arr.idx.shape.j),
                        .leading = @intCast(si),
                        .ptr = @ptrCast(arr.buf.ptr),
                    };
                } else if (si == 1 and sj >= arr.idx.shape.i) {
                    return .{
                        .layout = @intCast(acc.CblasColMajor),
                        .rows = @intCast(arr.idx.shape.i),
                        .cols = @intCast(arr.idx.shape.j),
                        .leading = @intCast(sj),
                        .ptr = @ptrCast(arr.buf.ptr),
                    };
                } else {
                    @panic("Blas2dMut: minor axis must have stride 1, and major axis stride must be >= minor axis shape");
                }
            }

            /// Creates a Blas2dMut from an arbitrary 2D NamedArray by renaming axes to IJ.
            /// `i_axis` selects which AxisA field (0 or 1) maps to i; the other maps to j.
            fn fromNamedArray(comptime AxisA: type, arr: NamedArray(AxisA, Scalar), comptime i_axis: usize) @This() {
                const rename_pairs = comptime [_]named_index.AxisRenamePair{
                    .{ .old = meta.fieldNames(AxisA)[i_axis], .new = "i" },
                    .{ .old = meta.fieldNames(AxisA)[1 - i_axis], .new = "j" },
                };
                const a_ij_idx = arr.idx.rename(IJ, &rename_pairs);
                const arr_ij: NamedArray(IJ, Scalar) = .{ .idx = a_ij_idx, .buf = arr.buf };
                return init(arr_ij);
            }
        };
    }

    /// Wrapper for a matrix stored in BLAS packed format (read-only).
    /// Bundles the triangle selector, packing order, and data slice into a single
    /// value, making the storage assumptions explicit at the call site.
    ///
    /// Construct via struct literal:
    /// ```
    /// .{ .triangle = .k, .major = .k, .data = &ap }
    /// ```
    pub fn BlasPacked(comptime Scalar: type, comptime AxisA: type) type {
        return struct {
            triangle: AxisA,
            major: AxisA,
            data: []const Scalar,

            pub fn uplo(self: @This()) acc.CBLAS_UPLO {
                return uploBlas(AxisA, self.triangle);
            }

            pub fn layout(self: @This()) acc.CBLAS_ORDER {
                return layoutBlas(AxisA, self.major);
            }

            pub fn ptr(self: @This()) [*]const Scalar {
                return self.data.ptr;
            }

            /// Assert that the packed slice has exactly `n * (n + 1) / 2` elements.
            pub fn validate(self: @This(), n: usize) void {
                assert(self.data.len == n * (n + 1) / 2);
            }
        };
    }

    /// Wrapper for a matrix stored in BLAS packed format (mutable).
    /// Same semantics as `BlasPacked` but the data slice is mutable.
    pub fn BlasPackedMut(comptime Scalar: type, comptime AxisA: type) type {
        return struct {
            triangle: AxisA,
            major: AxisA,
            data: []Scalar,

            pub fn uplo(self: @This()) acc.CBLAS_UPLO {
                return uploBlas(AxisA, self.triangle);
            }

            pub fn layout(self: @This()) acc.CBLAS_ORDER {
                return layoutBlas(AxisA, self.major);
            }

            pub fn ptr(self: @This()) [*]Scalar {
                return self.data.ptr;
            }

            /// Assert that the packed slice has exactly `n * (n + 1) / 2` elements.
            pub fn validate(self: @This(), n: usize) void {
                assert(self.data.len == n * (n + 1) / 2);
            }
        };
    }

    /// Wrapper for a dense triangular / symmetric / Hermitian matrix (read-only).
    /// Bundles the triangle selector with the underlying `NamedArrayConst`,
    /// following the same convention as `BlasPacked`.
    ///
    /// Construct via struct literal:
    /// ```
    /// .{ .triangle = .k, .data = A }
    /// ```
    pub fn BlasTriangular(comptime Scalar: type, comptime AxisA: type) type {
        return struct {
            triangle: AxisA,
            data: NamedArrayConst(AxisA, Scalar),

            pub fn uplo(self: @This()) acc.CBLAS_UPLO {
                return uploBlas(AxisA, self.triangle);
            }
        };
    }

    /// Wrapper for a dense triangular / symmetric / Hermitian matrix (mutable).
    /// Same semantics as `BlasTriangular` but the data is mutable.
    pub fn BlasTriangularMut(comptime Scalar: type, comptime AxisA: type) type {
        return struct {
            triangle: AxisA,
            data: NamedArray(AxisA, Scalar),

            pub fn uplo(self: @This()) acc.CBLAS_UPLO {
                return uploBlas(AxisA, self.triangle);
            }
        };
    }

    /// Wrapper for a matrix stored in BLAS band format (read-only).
    /// Constructed from a 2D `NamedArrayConst` whose axes are a band axis
    /// (contiguous, stride 1) and a vector axis matching `AxisX`.
    /// Validates the band-storage layout constraints at construction time.
    pub fn BlasBand(comptime Scalar: type, comptime AxisA: type, comptime AxisX: type) type {
        return struct {
            n: c_int,
            band_size: c_int,
            leading: c_int,
            ptr: [*]const Scalar,

            const x_axis_idx = blk: {
                const a_names = meta.fieldNames(AxisA);
                const x_name = meta.fields(AxisX)[0].name;
                break :blk if (std.mem.eql(u8, x_name, a_names[0])) @as(usize, 0) else @as(usize, 1);
            };

            pub fn init(A: NamedArrayConst(AxisA, Scalar)) @This() {
                const a_names = comptime meta.fieldNames(AxisA);
                const band_name = a_names[1 - x_axis_idx];
                const n_name = a_names[x_axis_idx];
                const band_stride = @field(A.idx.strides, band_name);
                const n_stride = @field(A.idx.strides, n_name);
                const band_sz: usize = @field(A.idx.shape, band_name);
                const n_sz: usize = @field(A.idx.shape, n_name);

                assert(band_stride == 1); // Band axis must be contiguous (stride 1)
                assert(n_stride >= band_sz); // LDA >= band size

                return .{
                    .n = @intCast(n_sz),
                    .band_size = @intCast(band_sz),
                    .leading = @intCast(n_stride),
                    .ptr = @ptrCast(A.buf.ptr),
                };
            }

            /// Bandwidth `k` for symmetric / triangular band matrices (`band_size - 1`).
            pub fn k(self: @This()) c_int {
                return self.band_size - 1;
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
    const actual = blas.dot(T, I, x, y, .{});
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(T, expected),
    );
}

test "dot internal_double_precision" {
    const I = enum { i };
    const T = f32;
    const Arr = NamedArrayConst(I, T);

    // Use values where double-precision accumulation matters:
    // large * large + small can lose the small contribution in f32.
    const x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{ 1.0, 2.0, 3.0 },
    };
    const y = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]T{ 4.0, 5.0, 6.0 },
    };

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    const expected: T = 32.0;
    const actual = blas.dot(T, I, x, y, .{ .internal_double_precision = true });
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(T, expected),
    );
}

test "dsdot" {
    const I = enum { i };
    const Arr = NamedArrayConst(I, f32);

    var x = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]f32{ 2.0, 3.0, 5.0 },
    };
    x.idx = x.idx.stride(.{ .i = -1 });
    const y = Arr{
        .idx = .initContiguous(.{ .i = 3 }),
        .buf = &[_]f32{ 1.0, 10.0, 100.0 },
    };

    // With reversed x: [5, 3, 2] · [1, 10, 100] = 5 + 30 + 200 = 235
    const expected: f64 = 235.0;
    const actual: f64 = blas.dsdot(I, x, y);
    try std.testing.expectApproxEqAbs(
        expected,
        actual,
        math.floatEpsAt(f64, expected),
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
    const actual = blas.dotu(T, I, x, y);
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
    const actual = blas.dotc(T, I, x, y);
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
    const actual = blas.nrm2(T, I, x);
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
    const actual = blas.nrm2(T, I, x);
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
    const actual = blas.asum(T, I, x);
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
    const actual = blas.asum(T, I, x);
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

    const actual = blas.i_amax(T, I, x);
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

    const actual = blas.i_amax(T, I, x);
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

    blas.swap(T, I, x, y);
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

    blas.copy(T, I, x, y);
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
    blas.axpy(T, I, alpha, x, y);
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
    blas.axpy(T, I, alpha, x, y);
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
    blas.scal(T, T, I, alpha, x);

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
    blas.scal(T, f32, I, alpha, x);

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
    blas.scal(T, T, I, alpha, x);

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

test "gemv real padded row-major" {
    // 2×3 submatrix of a wider 2×5 row-major buffer.
    // A = [1 2 3 . .]    x = [1]    y = A*x = [ 14]
    //     [4 5 6 . .]        [2]               [ 32]
    //                         [3]
    // lda = 5 (stride of the m axis), shape.k = 3.
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    const a_buf = [_]T{
        1, 2, 3, 99, 99,
        4, 5, 6, 99, 99,
    };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 3 }, .strides = .{ .m = 5, .k = 1 } },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &y_buf,
    };

    blas.gemv(T, MK, K, M, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 14.0), y_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 32.0), y_buf[1], 1e-10);
}

test "gemv real padded column-major" {
    // 2×3 submatrix in a taller 4×3 column-major buffer.
    // A = [1 3 5]    x = [1]    y = A*x = [ 22]
    //     [2 4 6]        [2]               [ 28]
    //  (padding row)     [3]
    //  (padding row)
    // lda = 4 (stride of the k axis), shape.m = 2.
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    const a_buf = [_]T{
        1, 2, 99, 99,
        3, 4, 99, 99,
        5, 6, 99, 99,
    };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 3 }, .strides = .{ .m = 1, .k = 4 } },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &y_buf,
    };

    blas.gemv(T, MK, K, M, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 22.0), y_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 28.0), y_buf[1], 1e-10);
}

test "symv padded row-major" {
    // 3×3 symmetric matrix in a 3×5 row-major buffer (lda=5).
    // A = [1 2 3 . .]    x = [1]    y = A*x = [14]
    //     [2 5 6 . .]        [2]               [30]
    //     [3 6 9 . .]        [3]               [42]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    const a_buf = [_]T{
        1, 2, 3, 99, 99,
        2, 5, 6, 99, 99,
        3, 6, 9, 99, 99,
    };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 3, .k = 3 }, .strides = .{ .m = 5, .k = 1 } },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .k (second axis) → Upper
    blas.symv(T, MK, K, M, .k, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 14.0, 30.0, 42.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "ger padded row-major" {
    // Rank-1 update on a 2×3 matrix in a 2×5 row-major buffer (lda=5).
    // A_init = [1 2 3 . .]    x = [1]    y = [4 5 6]
    //          [4 5 6 . .]        [2]
    //
    // A_new = A_init + x*y^T = [1+4  2+5   3+6  . .] = [5  7   9  . .]
    //                           [4+8  5+10  6+12 . .]   [12 15  18 . .]
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    var a_buf = [_]T{
        1, 2, 3, 99, 99,
        4, 5, 6, 99, 99,
    };
    const A = NamedArray(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 3 }, .strides = .{ .m = 5, .k = 1 } },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2 };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 4, 5, 6 };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    blas.ger(T, MK, M, K, A, x, y, .{});

    // Check result values (padding must be untouched)
    try std.testing.expectApproxEqAbs(@as(T, 5.0), a_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 7.0), a_buf[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 9.0), a_buf[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 99.0), a_buf[3], 1e-10); // padding untouched
    try std.testing.expectApproxEqAbs(@as(T, 99.0), a_buf[4], 1e-10); // padding untouched
    try std.testing.expectApproxEqAbs(@as(T, 12.0), a_buf[5], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 15.0), a_buf[6], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 18.0), a_buf[7], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 99.0), a_buf[8], 1e-10); // padding untouched
    try std.testing.expectApproxEqAbs(@as(T, 99.0), a_buf[9], 1e-10); // padding untouched
}

test "gemv real" {
    const MK = enum { m, k };
    const M = enum { m };
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // y = 1 * A * x + 0 * y = A * x = [14, 32, 50]
    blas.gemv(T, MK, K, M, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 14.0, 32.0, 50.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemv real rectangular" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    // A = [[1, 2], [3, 4], [5, 6]] (3x2, m=3, k=2)
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 2 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // y = A * x = [1*1+2*2, 3*1+4*2, 5*1+6*2] = [5, 11, 17]
    blas.gemv(T, MK, K, M, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 5.0, 11.0, 17.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemv real nontrivial scalars and strides" {
    const MK = enum { m, k };
    const M = enum { m };
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
    const y_idx: NamedIndex(M) = .{
        .shape = .{ .m = 3 },
        .strides = .{ .m = 2 },
        .offset = 0,
    };
    const y = NamedArray(M, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 2.0, beta = -1.0
    // A * x_logical = A * [3,2,1]:
    //   [1*3+2*2+3*1, 4*3+5*2+6*1, 7*3+8*2+9*1] = [10, 28, 46]
    // y = 2 * [10, 28, 46] + (-1) * [10, 20, 30] = [10, 36, 62]
    blas.gemv(T, MK, K, M, A, x, y, .{ .alpha = 2.0, .beta = -1.0 });

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
    const I = enum { i };
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
    // x axis is j (contracted), y axis is i (output).
    // After rename: j→j (cols), i→i (rows) → NoTrans
    // y = A*x = [1*5+3*6, 2*5+4*6] = [23, 34]
    const x_buf = [_]T{ 5, 6 };
    const x = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0 };
    const y = NamedArray(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &y_buf,
    };

    blas.gemv(T, IJ, J, I, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23.0, 34.0 };
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemv real 1x1 matrix" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    // A = [[7]] (1x1 matrix)
    const a_buf = [_]T{7};
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 1, .k = 1 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{3};
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 1 }),
        .buf = &x_buf,
    };

    // y starts at 10, beta = 2, alpha = 3
    // y = alpha * A * x + beta * y = 3 * 7 * 3 + 2 * 10 = 63 + 20 = 83
    var y_buf = [_]T{10};
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 1 }),
        .buf = &y_buf,
    };

    blas.gemv(T, MK, K, M, A, x, y, .{ .alpha = 3.0, .beta = 2.0 });

    try std.testing.expectApproxEqAbs(@as(T, 83.0), y_buf[0], math.floatEpsAt(T, 83.0));
}

test "gemv complex" {
    const MK = enum { m, k };
    const M = enum { m };
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
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
    blas.gemv(T, MK, K, M, A, x, y, .{
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
    const A_ = enum { a };
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
    const y_idx: NamedIndex(A_) = .{
        .shape = .{ .a = 2 },
        .strides = .{ .a = 2 },
        .offset = 0,
    };
    const y = NamedArray(A_, T){
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
    blas.gemv(T, AB, B, A_, A, x, y, .{
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

test "hemv upper (triangle = second axis)" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    // Hermitian 3x3 matrix (upper triangle stored, triangle = .k, i.e. data where k >= m):
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // y = A * x:
    //   y[0] = 2*(1) + (1-i)*(i) + (3+2i)*(1) = 2 + i+1 + 3+2i = 6+3i
    //   y[1] = (1+i)*(1) + 5*(i) + (2-i)*(1) = 1+i + 5i + 2-i = 3+5i
    //   y[2] = (3-2i)*(1) + (2+i)*(i) + 4*(1) = 3-2i + 2i-1 + 4 = 6+0i
    blas.hemv(T, MK, K, M, .k, A, x, y, .{
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

test "hemv lower (triangle = first axis)" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    // Same Hermitian matrix, but now the *lower* triangle is stored (triangle = .m, i.e. data where m >= k).
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // Same result as the upper test: y = [6+3i, 3+5i, 6+0i]
    blas.hemv(T, MK, K, M, .m, A, x, y, .{
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
    const A_ = enum { a };
    const B = enum { b };
    const T = Complex(f32);

    // Hermitian 2x2: [[3, 1+2i], [1-2i, 5]]
    // Upper triangle stored (triangle = .b, i.e. data where b >= a); lower positions hold sentinels.
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
    const y_idx: NamedIndex(A_) = .{
        .shape = .{ .a = 2 },
        .strides = .{ .a = 2 },
        .offset = 0,
    };
    const y = NamedArray(A_, T){
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
    blas.hemv(T, AB, B, A_, .b, A, x, y, .{
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

test "hemv column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = Complex(f64);

    // Hermitian 2x2: [[2, 1-i], [1+i, 3]]
    // Upper triangle stored (triangle = .j, data where j >= i).
    // Column-major buffer: col0=[2+0i, sentinel], col1=[1-i, 3+0i]
    const a_buf = [_]T{
        .{ .re = 2, .im = 0 },  .{ .re = 99, .im = 99 },
        .{ .re = 1, .im = -1 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArrayConst(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const x = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const y = NamedArray(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &y_buf,
    };

    // y = A*x:
    //   y[0] = 2*1 + (1-i)*(i) = 2 + i+1 = 3+i
    //   y[1] = (1+i)*1 + 3*(i) = 1+i+3i = 1+4i
    blas.hemv(T, IJ, J, I, .j, A, x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), y_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), y_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), y_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), y_buf[1].im, eps);
}

test "symv upper (triangle = second axis)" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    // Symmetric 3x3 matrix (upper triangle stored, triangle = .k, i.e. data where k >= m):
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // y = A * x:
    //   y[0] = 2*1 + 3*2 + 5*3  = 2 + 6 + 15  = 23
    //   y[1] = 3*1 + 7*2 + 11*3 = 3 + 14 + 33 = 50
    //   y[2] = 5*1 + 11*2 + 13*3 = 5 + 22 + 39 = 66
    blas.symv(T, MK, K, M, .k, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23.0, 50.0, 66.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symv lower (triangle = first axis)" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    // Same symmetric matrix, but now the *lower* triangle is stored (triangle = .m, i.e. data where m >= k).
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // Same result as the upper test: y = [23, 50, 66]
    blas.symv(T, MK, K, M, .m, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23.0, 50.0, 66.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symv nontrivial scalars and strides" {
    const AB = enum { a, b };
    const A_ = enum { a };
    const B = enum { b };
    const T = f32;

    // Symmetric 2x2: [[4, 3], [3, 7]]
    // Upper triangle stored (triangle = .b, i.e. data where b >= a); lower position holds sentinel.
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
    const y_idx: NamedIndex(A_) = .{
        .shape = .{ .a = 2 },
        .strides = .{ .a = 2 },
        .offset = 0,
    };
    const y = NamedArray(A_, T){
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
    blas.symv(T, AB, B, A_, .b, A, x, y, .{ .alpha = 2.5, .beta = -1.0 });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 17.5), y_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), y_buf[2], eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99.0), y_buf[1]);
}

test "symv column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = f64;

    // Symmetric 2x2: [[4, 3], [3, 7]]
    // Upper triangle stored (triangle = .j, data where j >= i).
    // Column-major buffer: col0=[4, sentinel], col1=[3, 7]
    const a_buf = [_]T{ 4, 99, 3, 7 };
    const A = NamedArrayConst(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2 };
    const x = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0 };
    const y = NamedArray(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &y_buf,
    };

    // y = A*x = [4*1+3*2, 3*1+7*2] = [10, 17]
    blas.symv(T, IJ, J, I, .j, A, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 10.0, 17.0 };
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trmv real" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Upper triangular 3x3 (triangle = .k, i.e. data where k >= m).
    // Lower triangle positions hold sentinels.
    //   [[2, 3, 5],
    //    [_, 7, 11],
    //    [_, _, 13]]
    const a_buf = [_]T{
        2,  3,  5,
        99, 7,  11,
        99, 99, 13,
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    var x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // x := A * x:
    //   x[0] = 2*1 + 3*2 + 5*3  = 23
    //   x[1] = 0*1 + 7*2 + 11*3 = 47
    //   x[2] = 0*1 + 0*2 + 13*3 = 39
    blas.trmv(T, MK, K, .k, .non_unit, A, x);

    const expected = [_]T{ 23.0, 47.0, 39.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trmv real unit diagonal" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Lower triangular 3x3 with unit diagonal (triangle = .m, i.e. data where m >= k).
    // Diagonal and upper positions hold sentinels that BLAS must ignore.
    //   [[1, _, _],       effective matrix (diag = 1)
    //    [4, 1, _],
    //    [5, 6, 1]]
    const a_buf = [_]T{
        99, 99, 99,
        4,  99, 99,
        5,  6,  99,
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    var x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // x := A * x (with unit diagonal):
    //   x[0] = 1*1 + 0*2 + 0*3 = 1
    //   x[1] = 4*1 + 1*2 + 0*3 = 6
    //   x[2] = 5*1 + 6*2 + 1*3 = 20
    blas.trmv(T, MK, K, .m, .unit, A, x);

    const expected = [_]T{ 1.0, 6.0, 20.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trmv complex" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    // Upper triangular 2x2 (triangle = .k):
    //   [[1+i,  2+3i],
    //    [ _,   4-i ]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 },   .{ .re = 2, .im = 3 },
        .{ .re = 99, .im = 99 }, .{ .re = 4, .im = -1 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    var x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &x_buf,
    };

    // x := A * x:
    //   x[0] = (1+i)(1) + (2+3i)(i) = 1+i + 2i+3i² = 1+i+2i-3 = -2+3i
    //   x[1] = 0*(1) + (4-i)(i) = 4i-i² = 1+4i
    blas.trmv(T, MK, K, .k, .non_unit, A, x);

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, -2), x_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), x_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), x_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), x_buf[1].im, eps);
}

test "trmv nontrivial strides" {
    const AB = enum { a, b };
    const B = enum { b };
    const T = Complex(f32);

    // Lower triangular 2x2 (triangle = .a, i.e. data where a >= b):
    //   [[3+0i,    _    ],
    //    [1+2i,  5+0i   ]]
    const a_buf = [_]T{
        .{ .re = 3, .im = 0 }, .{ .re = 99, .im = 99 },
        .{ .re = 1, .im = 2 }, .{ .re = 5, .im = 0 },
    };
    const A = NamedArrayConst(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x with stride 2: positions 0, 2 in buffer; logical x = [1+i, 2+0i]
    var x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 99, .im = 99 }, // sentinel
        .{ .re = 2, .im = 0 },
    };
    const x_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const x = NamedArray(B, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // x := A * x:
    //   x[0] = (3)(1+i) + 0*(2) = 3+3i
    //   x[1] = (1+2i)(1+i) + (5)(2) = 1+i+2i+2i² + 10 = 9+3i
    blas.trmv(T, AB, B, .a, .non_unit, A, x);

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 3), x_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 3), x_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 9), x_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 3), x_buf[2].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99), x_buf[1].re);
    try std.testing.expectEqual(@as(f32, 99), x_buf[1].im);
}

test "trmv column-major matrix" {
    const IJ = enum { i, j };
    const J = enum { j };
    const T = f64;

    // Upper triangular 2x2 (triangle = .j, data where j >= i):
    //   [[2, 3],
    //    [_, 5]]
    // Column-major buffer: col0=[2, sentinel], col1=[3, 5]
    const a_buf = [_]T{ 2, 99, 3, 5 };
    const A = NamedArrayConst(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    // x maps to j-axis; x = [1, 2]
    // x := A * x:
    //   x[0] = 2*1 + 3*2 = 8
    //   x[1] = 0*1 + 5*2 = 10
    var x_buf = [_]T{ 1, 2 };
    const x = NamedArray(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    blas.trmv(T, IJ, J, .j, .non_unit, A, x);

    const expected = [_]T{ 8.0, 10.0 };
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trsv real" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Upper triangular 3x3 (triangle = .k, i.e. data where k >= m).
    // Lower triangle positions hold sentinels.
    //   [[2, 3, 5],
    //    [_, 7, 11],
    //    [_, _, 13]]
    const a_buf = [_]T{
        2,  3,  5,
        99, 7,  11,
        99, 99, 13,
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // x_init = A * [1,2,3] = [23, 47, 39]  (from trmv test)
    // After trsv: x should be [1, 2, 3].
    var x_buf = [_]T{ 23, 47, 39 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.trsv(T, MK, K, .k, .non_unit, A, x);

    const expected = [_]T{ 1.0, 2.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trsv real unit diagonal" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Lower triangular 3x3 with unit diagonal (triangle = .m, i.e. data where m >= k).
    // Diagonal and upper positions hold sentinels that BLAS must ignore.
    //   [[1, _, _],       effective matrix (diag = 1)
    //    [4, 1, _],
    //    [5, 6, 1]]
    const a_buf = [_]T{
        99, 99, 99,
        4,  99, 99,
        5,  6,  99,
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // x_init = A * [1,2,3] = [1, 6, 20]  (from trmv unit diagonal test)
    // After trsv: x should be [1, 2, 3].
    var x_buf = [_]T{ 1, 6, 20 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.trsv(T, MK, K, .m, .unit, A, x);

    const expected = [_]T{ 1.0, 2.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trsv complex" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    // Upper triangular 2x2 (triangle = .k):
    //   [[1+i,  2+3i],
    //    [ _,   4-i ]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 },   .{ .re = 2, .im = 3 },
        .{ .re = 99, .im = 99 }, .{ .re = 4, .im = -1 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // x_init = A * [1, i] = [-2+3i, 1+4i]  (from trmv complex test)
    // After trsv: x should be [1, i].
    var x_buf = [_]T{
        .{ .re = -2, .im = 3 },
        .{ .re = 1, .im = 4 },
    };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &x_buf,
    };

    blas.trsv(T, MK, K, .k, .non_unit, A, x);

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 1), x_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), x_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), x_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), x_buf[1].im, eps);
}

test "trsv nontrivial strides" {
    const AB = enum { a, b };
    const B = enum { b };
    const T = Complex(f32);

    // Lower triangular 2x2 (triangle = .a, i.e. data where a >= b):
    //   [[3+0i,    _    ],
    //    [1+2i,  5+0i   ]]
    const a_buf = [_]T{
        .{ .re = 3, .im = 0 }, .{ .re = 99, .im = 99 },
        .{ .re = 1, .im = 2 }, .{ .re = 5, .im = 0 },
    };
    const A = NamedArrayConst(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x_init = A * [1+i, 2] = [3+3i, 9+3i]  (from trmv nontrivial strides test, but stride=1 semantics)
    // x with stride 2: positions 0, 2 in buffer
    // After trsv: x should be [1+i, 2+0i].
    var x_buf = [_]T{
        .{ .re = 3, .im = 3 },
        .{ .re = 99, .im = 99 }, // sentinel
        .{ .re = 9, .im = 3 },
    };
    const x_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const x = NamedArray(B, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    blas.trsv(T, AB, B, .a, .non_unit, A, x);

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 1), x_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 1), x_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 2), x_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), x_buf[2].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99), x_buf[1].re);
    try std.testing.expectEqual(@as(f32, 99), x_buf[1].im);
}

test "trsv column-major matrix" {
    const IJ = enum { i, j };
    const J = enum { j };
    const T = f64;

    // Same upper triangular 2x2 as trmv column-major test:
    //   [[2, 3],
    //    [_, 5]]
    // Column-major buffer: col0=[2, sentinel], col1=[3, 5]
    const a_buf = [_]T{ 2, 99, 3, 5 };
    const A = NamedArrayConst(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    // x_init = A * [1,2] = [8, 10]  (from trmv column-major test)
    // After trsv: x should be [1, 2].
    var x_buf = [_]T{ 8, 10 };
    const x = NamedArray(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    blas.trsv(T, IJ, J, .j, .non_unit, A, x);

    const expected = [_]T{ 1.0, 2.0 };
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "ger real" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    // A = zeros(3, 2), x = [1, 2, 3] (m-axis), y = [4, 5] (k-axis)
    // A := 1 * x * y^T + A = [[4, 5], [8, 10], [12, 15]]
    var a_buf = [_]T{ 0, 0, 0, 0, 0, 0 };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 2 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 4, 5 };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &y_buf,
    };

    blas.ger(T, MK, M, K, A, x, y, .{});

    const expected = [_]T{ 4, 5, 8, 10, 12, 15 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], a_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "ger real nontrivial strides" {
    const AB = enum { a, b };
    const A_ = enum { a };
    const B = enum { b };
    const T = f32;

    // A_init = [[1, 2], [3, 4]]
    var a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArray(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1, 2]; stride -1 → logical x = [2, 1]
    const x_buf = [_]T{ 1, 2 };
    var x_idx = NamedIndex(A_).initContiguous(.{ .a = 2 });
    x_idx = x_idx.stride(.{ .a = -1 });
    const x = NamedArrayConst(A_, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // y with stride 2: positions 0, 2 in buffer; logical y = [1, 2]
    const y_buf = [_]T{ 1, 99, 2 };
    const y_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const y = NamedArrayConst(B, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 3
    // A_new = 3 * outer([2,1], [1,2]) + [[1,2],[3,4]]
    //       = 3 * [[2,4],[1,2]] + [[1,2],[3,4]]
    //       = [[6,12],[3,6]] + [[1,2],[3,4]]
    //       = [[7, 14], [6, 10]]
    blas.ger(T, AB, A_, B, A, x, y, .{ .alpha = 3.0 });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 7), a_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 14), a_buf[1], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 6), a_buf[2], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 10), a_buf[3], eps);
}

test "ger column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = f64;

    // A = zeros(2, 3), column-major
    // Column-major buffer (2 rows, 3 cols): col0=[0,0], col1=[0,0], col2=[0,0]
    var a_buf = [_]T{ 0, 0, 0, 0, 0, 0 };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 3 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2 };
    const x = NamedArrayConst(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 3, 4, 5 };
    const y = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 3 }),
        .buf = &y_buf,
    };

    // A := x * y^T = [[3,4,5],[6,8,10]]
    // Column-major buffer: col0=[3,6], col1=[4,8], col2=[5,10]
    blas.ger(T, IJ, I, J, A, x, y, .{});

    const expected = [_]T{ 3, 6, 4, 8, 5, 10 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], a_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "geru complex" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    // A = zeros(2, 3), x = [1+i, 2] (m-axis), y = [1, i, 1+i] (k-axis)
    // A := x * y^T (unconjugated):
    //   [0,0] = (1+i)(1)   = 1+i
    //   [0,1] = (1+i)(i)   = -1+i
    //   [0,2] = (1+i)(1+i) = 2i
    //   [1,0] = (2)(1)     = 2
    //   [1,1] = (2)(i)     = 2i
    //   [1,2] = (2)(1+i)   = 2+2i
    var a_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = 1 },
    };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    blas.geru(T, MK, M, K, A, x, y, .{});

    const expected = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = -1, .im = 1 }, .{ .re = 0, .im = 2 },
        .{ .re = 2, .im = 0 }, .{ .re = 0, .im = 2 },  .{ .re = 2, .im = 2 },
    };
    const eps = 1e-10;
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, a_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, a_buf[i].im, eps);
    }
}

test "geru column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = Complex(f64);

    // A = zeros(2, 2), column-major
    var a_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const y = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &y_buf,
    };

    // A := x * y^T (unconjugated):
    //   (0,0) = (1+i)*1   = 1+i
    //   (1,0) = 2*1       = 2
    //   (0,1) = (1+i)*i   = -1+i
    //   (1,1) = 2*i       = 2i
    // Column-major buffer: col0=[1+i, 2], col1=[-1+i, 2i]
    blas.geru(T, IJ, I, J, A, x, y, .{});

    const expected = [_]T{
        .{ .re = 1, .im = 1 },  .{ .re = 2, .im = 0 },
        .{ .re = -1, .im = 1 }, .{ .re = 0, .im = 2 },
    };
    const eps = 1e-10;
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, a_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, a_buf[i].im, eps);
    }
}

test "gerc complex" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    // Same x and y as geru test, but conjugated:
    // A := x * y^H = x * conj(y)^T
    // conj(y) = [1, -i, 1-i]
    //   [0,0] = (1+i)(1)   = 1+i
    //   [0,1] = (1+i)(-i)  = 1-i
    //   [0,2] = (1+i)(1-i) = 2
    //   [1,0] = (2)(1)     = 2
    //   [1,1] = (2)(-i)    = -2i
    //   [1,2] = (2)(1-i)   = 2-2i
    var a_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = 1 },
    };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    blas.gerc(T, MK, M, K, A, x, y, .{});

    const expected = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 1, .im = -1 }, .{ .re = 2, .im = 0 },
        .{ .re = 2, .im = 0 }, .{ .re = 0, .im = -2 }, .{ .re = 2, .im = -2 },
    };
    const eps = 1e-10;
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, a_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, a_buf[i].im, eps);
    }
}

test "gerc nontrivial strides" {
    const AB = enum { a, b };
    const A_ = enum { a };
    const B = enum { b };
    const T = Complex(f32);

    // A_init = [[1+0i, 2+i], [0+i, 3+0i]]
    var a_buf = [_]T{
        .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 1 },
        .{ .re = 0, .im = 1 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArray(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1+i, 2+0i]; stride -1 → logical x = [2, 1+i]
    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    var x_idx = NamedIndex(A_).initContiguous(.{ .a = 2 });
    x_idx = x_idx.stride(.{ .a = -1 });
    const x = NamedArrayConst(A_, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // y with stride 2: physical [1+0i, 99, 0+i], logical y = [1, i]
    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 99, .im = 99 },
        .{ .re = 0, .im = 1 },
    };
    const y_idx: NamedIndex(B) = .{
        .shape = .{ .b = 2 },
        .strides = .{ .b = 2 },
        .offset = 0,
    };
    const y = NamedArrayConst(B, T){
        .idx = y_idx,
        .buf = &y_buf,
    };

    // alpha = 1+i
    // conj(y) = [1, -i]
    // outer(x_logical, conj(y)) = outer([2, 1+i], [1, -i])
    //   = [[2, -2i], [1+i, 1-i]]
    // (1+i) * [[2, -2i], [1+i, 1-i]]
    //   = [[2+2i, 2-2i], [2i, 2]]
    // A_new = [[2+2i, 2-2i], [2i, 2]] + [[1, 2+i], [i, 3]]
    //       = [[3+2i, 4-i], [3i, 5]]
    blas.gerc(T, AB, A_, B, A, x, y, .{
        .alpha = .{ .re = 1, .im = 1 },
    });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 3), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 2), a_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 4), a_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, -1), a_buf[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 3), a_buf[2].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 5), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a_buf[3].im, eps);
}

test "gerc column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = Complex(f64);

    // A = zeros(2, 2), column-major
    var a_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const y = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &y_buf,
    };

    // A := x * y^H = x * conj(y)^T:
    //   conj(y) = [1, -i]
    //   (0,0) = (1+i)*1   = 1+i
    //   (1,0) = 2*1       = 2
    //   (0,1) = (1+i)*(-i) = -i-i^2 = 1-i
    //   (1,1) = 2*(-i)    = -2i
    // Column-major buffer: col0=[1+i, 2], col1=[1-i, -2i]
    blas.gerc(T, IJ, I, J, A, x, y, .{});

    const expected = [_]T{
        .{ .re = 1, .im = 1 },  .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = -1 }, .{ .re = 0, .im = -2 },
    };
    const eps = 1e-10;
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, a_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, a_buf[i].im, eps);
    }
}

test "syr real (triangle = second axis)" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    // Symmetric 3x3, upper triangle stored (triangle = .k, data where k >= m).
    // Lower triangle positions hold sentinels.
    //   [[1, 2, 3],
    //    [_, 5, 6],
    //    [_, _, 9]]
    var a_buf = [_]T{
        1,  2,  3,
        99, 5,  6,
        99, 99, 9,
    };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // A_new upper = 1 * x*x^T upper + A_init upper
    // x*x^T = [[1,2,3],[2,4,6],[3,6,9]]
    //   [0,0] = 1+1=2,  [0,1] = 2+2=4,  [0,2] = 3+3=6
    //                    [1,1] = 4+5=9,   [1,2] = 6+6=12
    //                                     [2,2] = 9+9=18
    blas.syr(T, MK, K, .k, A, x, .{});

    const eps = 1e-10;
    // Row 0
    try std.testing.expectApproxEqAbs(@as(T, 2), a_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(T, 4), a_buf[1], eps);
    try std.testing.expectApproxEqAbs(@as(T, 6), a_buf[2], eps);
    // Row 1: sentinel, then diagonal and upper
    try std.testing.expectEqual(@as(T, 99), a_buf[3]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(T, 9), a_buf[4], eps);
    try std.testing.expectApproxEqAbs(@as(T, 12), a_buf[5], eps);
    // Row 2: sentinels, then diagonal
    try std.testing.expectEqual(@as(T, 99), a_buf[6]); // sentinel untouched
    try std.testing.expectEqual(@as(T, 99), a_buf[7]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(T, 18), a_buf[8], eps);
}

test "syr real nontrivial strides" {
    const AB = enum { a, b };
    const B = enum { b };
    const T = f32;

    // Symmetric 2x2, lower triangle stored (triangle = .a, data where a >= b).
    // Upper off-diagonal holds sentinel.
    //   [[3, _ ],
    //    [1, 7 ]]
    var a_buf = [_]T{ 3, 99, 1, 7 };
    const A = NamedArray(AB, T){
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

    // alpha = 2
    // x*x^T = [[4, 2], [2, 1]]
    // A_new lower = 2 * [[4,_],[2,1]] + [[3,_],[1,7]]
    //            = [[8,_],[4,2]] + [[3,_],[1,7]]
    //            = [[11,_],[5,9]]
    blas.syr(T, AB, B, .a, A, x, .{ .alpha = 2.0 });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 11), a_buf[0], eps);
    try std.testing.expectEqual(@as(f32, 99), a_buf[1]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(f32, 5), a_buf[2], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 9), a_buf[3], eps);
}

test "syr column-major matrix" {
    const IJ = enum { i, j };
    const J = enum { j };
    const T = f64;

    // Symmetric 2x2, lower triangle stored (triangle = .i, data where i >= j).
    // A_init = [[1, _], [2, 3]]
    // Column-major buffer: col0=[1, 2], col1=[sentinel, 3]
    var a_buf = [_]T{ 1, 2, 99, 3 };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2 };
    const x = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    // x*x^T = [[1, 2], [2, 4]]
    // A_new lower = [[1+1, _], [2+2, 3+4]] = [[2, _], [4, 7]]
    // Column-major buffer: [2, 4, sentinel, 7]
    blas.syr(T, IJ, J, .i, A, x, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(T, 2), a_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(T, 4), a_buf[1], eps);
    try std.testing.expectEqual(@as(T, 99), a_buf[2]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(T, 7), a_buf[3], eps);
}

test "her complex (triangle = second axis)" {
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    // Hermitian 2x2, upper triangle stored (triangle = .k, data where k >= m).
    // Lower off-diagonal holds sentinel.
    //   [[1+0i,    2+i ],
    //    [  _,     3+0i]]
    var a_buf = [_]T{
        .{ .re = 1, .im = 0 },   .{ .re = 2, .im = 1 },
        .{ .re = 99, .im = 99 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &x_buf,
    };

    // alpha = 1 (real)
    // x*x^H = [[|1+i|^2, (1+i)*conj(2)], [(2)*conj(1+i), |2|^2]]
    //       = [[2, 2+2i], [2-2i, 4]]
    // Upper part: [[2, 2+2i], [_, 4]]
    // A_new upper = [[2, 2+2i], [_, 4]] + [[1, 2+i], [_, 3]]
    //            = [[3+0i, 4+3i], [_, 7+0i]]
    blas.her(f64, MK, K, .k, A, x, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), a_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), a_buf[1].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].re);
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].im);
    try std.testing.expectApproxEqAbs(@as(f64, 7), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[3].im, eps);
}

test "her complex nontrivial strides" {
    const AB = enum { a, b };
    const A_ = enum { a };
    const T = Complex(f32);

    // Hermitian 2x2, lower triangle stored (triangle = .a, data where a >= b).
    // Upper off-diagonal holds sentinel.
    //   [[5+0i,       _     ],
    //    [1-i,     3+0i     ]]
    var a_buf = [_]T{
        .{ .re = 5, .im = 0 },  .{ .re = 99, .im = 99 },
        .{ .re = 1, .im = -1 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArray(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1+i, i]; stride -1 → logical x = [i, 1+i]
    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 0, .im = 1 },
    };
    var x_idx = NamedIndex(A_).initContiguous(.{ .a = 2 });
    x_idx = x_idx.stride(.{ .a = -1 });
    const x = NamedArrayConst(A_, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    // alpha = 2 (real)
    // x*x^H with logical x = [i, 1+i]:
    //   [0,0] = |i|^2 = 1
    //   [1,0] = (1+i)*conj(i) = (1+i)*(-i) = -i-i^2 = 1-i
    //   [1,1] = |1+i|^2 = 2
    // Lower part: [[1, _], [1-i, 2]]
    // A_new lower = 2*[[1,_],[1-i,2]] + [[5,_],[1-i,3]]
    //            = [[2,_],[2-2i,4]] + [[5,_],[1-i,3]]
    //            = [[7+0i, _], [3-3i, 7+0i]]
    blas.her(f32, AB, A_, .a, A, x, .{ .alpha = 2.0 });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 7), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a_buf[0].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99), a_buf[1].re);
    try std.testing.expectEqual(@as(f32, 99), a_buf[1].im);
    try std.testing.expectApproxEqAbs(@as(f32, 3), a_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, -3), a_buf[2].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 7), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a_buf[3].im, eps);
}

test "her column-major matrix" {
    const IJ = enum { i, j };
    const J = enum { j };
    const T = Complex(f64);

    // Hermitian 2x2, lower triangle stored (triangle = .i, data where i >= j).
    // A_init = [[3+0i, _], [1+i, 5+0i]]
    // Column-major buffer: col0=[3+0i, 1+i], col1=[sentinel, 5+0i]
    var a_buf = [_]T{
        .{ .re = 3, .im = 0 },   .{ .re = 1, .im = 1 },
        .{ .re = 99, .im = 99 }, .{ .re = 5, .im = 0 },
    };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 1 },
    };
    const x = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &x_buf,
    };

    // alpha = 1 (real, default)
    // x*x^H = [[|1|^2, 1*conj(1+i)], [(1+i)*conj(1), |1+i|^2]]
    //       = [[1, 1-i], [1+i, 2]]
    // Lower part: [[1, _], [1+i, 2]]
    // A_new lower = [[3+1, _], [1+i+1+i, 5+2]] = [[4+0i, _], [2+2i, 7+0i]]
    // Column-major buffer: [4+0i, 2+2i, sentinel, 7+0i]
    blas.her(f64, IJ, J, .i, A, x, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 4), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), a_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), a_buf[1].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].re);
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].im);
    try std.testing.expectApproxEqAbs(@as(f64, 7), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[3].im, eps);
}

test "syr2 real (triangle = second axis)" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = f64;

    // Symmetric 3x3, upper triangle stored (triangle = .k, data where k >= m).
    // Lower triangle positions hold sentinels.
    //   [[1, 2, 3],
    //    [_, 5, 6],
    //    [_, _, 9]]
    var a_buf = [_]T{
        1,  2,  3,
        99, 5,  6,
        99, 99, 9,
    };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 4, 5, 6 };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &y_buf,
    };

    // alpha = 1 (default)
    // x*y^T = [[4,5,6],[8,10,12],[12,15,18]]
    // y*x^T = [[4,8,12],[5,10,15],[6,12,18]]
    // x*y^T + y*x^T = [[8,13,18],[13,20,27],[18,27,36]]
    // A_new upper = [[8,13,18],[_,20,27],[_,_,36]] + [[1,2,3],[_,5,6],[_,_,9]]
    //            = [[9,15,21],[_,25,33],[_,_,45]]
    blas.syr2(T, MK, M, K, .k, A, x, y, .{});

    const eps = 1e-10;
    // Row 0
    try std.testing.expectApproxEqAbs(@as(T, 9), a_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(T, 15), a_buf[1], eps);
    try std.testing.expectApproxEqAbs(@as(T, 21), a_buf[2], eps);
    // Row 1: sentinel, then diagonal and upper
    try std.testing.expectEqual(@as(T, 99), a_buf[3]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(T, 25), a_buf[4], eps);
    try std.testing.expectApproxEqAbs(@as(T, 33), a_buf[5], eps);
    // Row 2: sentinels, then diagonal
    try std.testing.expectEqual(@as(T, 99), a_buf[6]); // sentinel untouched
    try std.testing.expectEqual(@as(T, 99), a_buf[7]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(T, 45), a_buf[8], eps);
}

test "syr2 real nontrivial strides" {
    const AB = enum { a, b };
    const A_ = enum { a };
    const B = enum { b };
    const T = f32;

    // Symmetric 2x2, lower triangle stored (triangle = .a, data where a >= b).
    // Upper off-diagonal holds sentinel.
    //   [[3, _ ],
    //    [1, 7 ]]
    var a_buf = [_]T{ 3, 99, 1, 7 };
    const A = NamedArray(AB, T){
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

    const y_buf = [_]T{ 3, 4 };
    const y = NamedArrayConst(A_, T){
        .idx = NamedIndex(A_).initContiguous(.{ .a = 2 }),
        .buf = &y_buf,
    };

    // alpha = 2
    // logical x = [2, 1], y = [3, 4]
    // x*y^T = [[6, 8], [3, 4]]
    // y*x^T = [[6, 3], [8, 4]]
    // x*y^T + y*x^T = [[12, 11], [11, 8]]
    // A_new lower = 2 * [[12, _], [11, 8]] + [[3, _], [1, 7]]
    //            = [[24, _], [22, 16]] + [[3, _], [1, 7]]
    //            = [[27, _], [23, 23]]
    blas.syr2(T, AB, B, A_, .a, A, x, y, .{ .alpha = 2.0 });

    const eps: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 27), a_buf[0], eps);
    try std.testing.expectEqual(@as(f32, 99), a_buf[1]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(f32, 23), a_buf[2], eps);
    try std.testing.expectApproxEqAbs(@as(f32, 23), a_buf[3], eps);
}

test "syr2 column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = f64;

    // Symmetric 2x2, lower triangle stored (triangle = .i, data where i >= j).
    // A_init = [[1, _], [2, 5]]
    // Column-major buffer: col0=[1, 2], col1=[sentinel, 5]
    var a_buf = [_]T{ 1, 2, 99, 5 };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{ 1, 2 };
    const x = NamedArrayConst(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 3, 4 };
    const y = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &y_buf,
    };

    // alpha = 1 (default)
    // x*y^T = [[3,4],[6,8]], y*x^T = [[3,6],[4,8]]
    // x*y^T + y*x^T = [[6,10],[10,16]]
    // Lower part: [[6,_],[10,16]]
    // A_new lower = [[1+6,_],[2+10,5+16]] = [[7,_],[12,21]]
    // Column-major buffer: [7, 12, sentinel, 21]
    blas.syr2(T, IJ, I, J, .i, A, x, y, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(T, 7), a_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(T, 12), a_buf[1], eps);
    try std.testing.expectEqual(@as(T, 99), a_buf[2]); // sentinel untouched
    try std.testing.expectApproxEqAbs(@as(T, 21), a_buf[3], eps);
}

test "her2 complex (triangle = second axis)" {
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    // Hermitian 2x2, upper triangle stored (triangle = .k, data where k >= m).
    // Lower off-diagonal holds sentinel.
    //   [[1+0i,    2+i ],
    //    [  _,     3+0i]]
    var a_buf = [_]T{
        .{ .re = 1, .im = 0 },   .{ .re = 2, .im = 1 },
        .{ .re = 99, .im = 99 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArray(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = -1 },
    };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &y_buf,
    };

    // alpha = 1+0i (default)
    // x = [1+i, 2], y = [i, 1-i]
    // x*y^H:
    //   (1+i)*conj(i) = (1+i)*(-i) = -i-i^2 = 1-i
    //   (1+i)*conj(1-i) = (1+i)*(1+i) = 1+2i+i^2 = 2i
    //   2*conj(i) = 2*(-i) = -2i
    //   2*conj(1-i) = 2*(1+i) = 2+2i
    // x*y^H = [[1-i, 2i], [-2i, 2+2i]]
    //
    // conj(alpha)*y*x^H = y*x^H (since alpha=1):
    //   i*conj(1+i) = i*(1-i) = i-i^2 = 1+i
    //   i*conj(2) = 2i
    //   (1-i)*conj(1+i) = (1-i)*(1-i) = 1-2i+i^2 = -2i
    //   (1-i)*conj(2) = 2-2i
    // y*x^H = [[1+i, 2i], [-2i, 2-2i]]
    //
    // x*y^H + y*x^H = [[2, 4i], [-4i, 4]]
    // Upper part: [[2, 4i], [_, 4]]
    // A_new upper = [[2, 4i], [_, 4]] + [[1, 2+i], [_, 3]]
    //            = [[3+0i, 2+5i], [_, 7+0i]]
    blas.her2(T, MK, M, K, .k, A, x, y, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), a_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 5), a_buf[1].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].re);
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].im);
    try std.testing.expectApproxEqAbs(@as(f64, 7), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[3].im, eps);
}

test "her2 complex nontrivial strides" {
    const AB = enum { a, b };
    const A_ = enum { a };
    const B = enum { b };
    const T = Complex(f32);

    // Hermitian 2x2, lower triangle stored (triangle = .a, data where a >= b).
    // Upper off-diagonal holds sentinel.
    //   [[5+0i,       _     ],
    //    [1-i,     3+0i     ]]
    var a_buf = [_]T{
        .{ .re = 5, .im = 0 },  .{ .re = 99, .im = 99 },
        .{ .re = 1, .im = -1 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArray(AB, T){
        .idx = NamedIndex(AB).initContiguous(.{ .a = 2, .b = 2 }),
        .buf = &a_buf,
    };

    // x physical = [1+i, i]; stride -1 → logical x = [i, 1+i]
    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 0, .im = 1 },
    };
    var x_idx = NamedIndex(B).initContiguous(.{ .b = 2 });
    x_idx = x_idx.stride(.{ .b = -1 });
    const x = NamedArrayConst(B, T){
        .idx = x_idx,
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const y = NamedArrayConst(A_, T){
        .idx = NamedIndex(A_).initContiguous(.{ .a = 2 }),
        .buf = &y_buf,
    };

    // alpha = 2+i
    // logical x = [i, 1+i], y = [1, i]
    //
    // alpha * x * y^H:
    //   x*y^H:
    //     i*conj(1) = i,  i*conj(i) = i*(-i) = 1
    //     (1+i)*conj(1) = 1+i,  (1+i)*conj(i) = (1+i)*(-i) = -i-i^2 = 1-i
    //   x*y^H = [[i, 1], [1+i, 1-i]]
    //   alpha * x*y^H = (2+i)*[[i, 1], [1+i, 1-i]]
    //     [0,0] = (2+i)*i = 2i+i^2 = -1+2i
    //     [0,1] = (2+i)*1 = 2+i
    //     [1,0] = (2+i)*(1+i) = 2+2i+i+i^2 = 1+3i
    //     [1,1] = (2+i)*(1-i) = 2-2i+i-i^2 = 3-i
    //
    // conj(alpha) * y * x^H:
    //   y*x^H:
    //     1*conj(i) = -i,  1*conj(1+i) = 1-i
    //     i*conj(i) = i*(-i) = 1,  i*conj(1+i) = i*(1-i) = i-i^2 = 1+i
    //   y*x^H = [[-i, 1-i], [1, 1+i]]
    //   conj(alpha) * y*x^H = (2-i)*[[-i, 1-i], [1, 1+i]]
    //     [0,0] = (2-i)*(-i) = -2i+i^2 = -1-2i
    //     [0,1] = (2-i)*(1-i) = 2-2i-i+i^2 = 1-3i
    //     [1,0] = (2-i)*1 = 2-i
    //     [1,1] = (2-i)*(1+i) = 2+2i-i-i^2 = 3+i
    //
    // sum = alpha*x*y^H + conj(alpha)*y*x^H:
    //   [0,0] = (-1+2i) + (-1-2i) = -2+0i
    //   [0,1] = (2+i) + (1-3i) = 3-2i
    //   [1,0] = (1+3i) + (2-i) = 3+2i
    //   [1,1] = (3-i) + (3+i) = 6+0i
    //
    // Lower part: [[-2, _], [3+2i, 6]]
    // A_new lower = [[-2,_],[3+2i,6]] + [[5,_],[1-i,3]]
    //            = [[3+0i, _], [4+i, 9+0i]]
    blas.her2(T, AB, B, A_, .a, A, x, y, .{ .alpha = .{ .re = 2, .im = 1 } });

    const eps: f32 = 1e-4;
    try std.testing.expectApproxEqAbs(@as(f32, 3), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a_buf[0].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f32, 99), a_buf[1].re);
    try std.testing.expectEqual(@as(f32, 99), a_buf[1].im);
    try std.testing.expectApproxEqAbs(@as(f32, 4), a_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 1), a_buf[2].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 9), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a_buf[3].im, eps);
}

// ============================================================
// Band matrix operation tests
// ============================================================

test "gbmv real" {
    // 4×4 tridiagonal matrix (KL=1, KU=1):
    // A = [2 1 0 0]    x = [1]    y = A*x = [ 4]
    //     [1 2 1 0]        [2]               [ 8]
    //     [0 1 2 1]        [3]               [12]
    //     [0 0 1 2]        [4]               [11]
    const BK = enum { band, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    // Band storage with band axis contiguous (stride 1), as required by BLAS.
    // Each group of 3 is one column: [superdiag, diag, subdiag]
    // Col 0: [0, 2, 1], Col 1: [1, 2, 1], Col 2: [1, 2, 1], Col 3: [1, 2, 0]
    const ab_buf = [_]T{
        0, 2, 1,
        1, 2, 1,
        1, 2, 1,
        1, 2, 0,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 3, .k = 4 },
            .strides = .{ .band = 1, .k = 3 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 4 }),
        .buf = &y_buf,
    };

    blas.gbmv(T, BK, K, M, blas.BlasBand(T, BK, K).init(AB), x, y, .{ .alpha = 1.0, .beta = 0.0 }, .{ .kl = 1, .ku = 1 });

    const expected = [_]T{ 4.0, 8.0, 12.0, 11.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gbmv real rectangular" {
    // 3×4 matrix (M=3, N=4, KL=1, KU=1):
    // A = [2 1 0 0]    x = [1]    y = A*x = [ 4]
    //     [1 2 1 0]        [2]               [ 8]
    //     [0 1 2 1]        [3]               [12]
    //                      [4]
    const BK = enum { band, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    // Band storage with band axis contiguous (stride 1), same data as square case but M=3
    // Col 0: [0, 2, 1], Col 1: [1, 2, 1], Col 2: [1, 2, 1], Col 3: [1, 2, 0]
    const ab_buf = [_]T{
        0, 2, 1,
        1, 2, 1,
        1, 2, 1,
        1, 2, 0,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 3, .k = 4 },
            .strides = .{ .band = 1, .k = 3 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    blas.gbmv(T, BK, K, M, blas.BlasBand(T, BK, K).init(AB), x, y, .{ .alpha = 1.0, .beta = 0.0 }, .{ .kl = 1, .ku = 1 });

    const expected = [_]T{ 4.0, 8.0, 12.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gbmv column-major matrix" {
    // Same 4×4 tridiagonal matrix, column-major band storage
    const BK = enum { band, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    // Column-major band storage (3 band rows × 4 cols):
    // Col 0: [0, 2, 1], Col 1: [1, 2, 1], Col 2: [1, 2, 1], Col 3: [1, 2, 0]
    const ab_buf = [_]T{
        0, 2, 1,
        1, 2, 1,
        1, 2, 1,
        1, 2, 0,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 3, .k = 4 },
            .strides = .{ .band = 1, .k = 3 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 4 }),
        .buf = &y_buf,
    };

    blas.gbmv(T, BK, K, M, blas.BlasBand(T, BK, K).init(AB), x, y, .{ .alpha = 1.0, .beta = 0.0 }, .{ .kl = 1, .ku = 1 });

    const expected = [_]T{ 4.0, 8.0, 12.0, 11.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "sbmv real upper" {
    // 4×4 symmetric tridiagonal (K=1), upper band storage:
    // A = [2 1 0 0]    x = [1]    y = A*x = [ 4]
    //     [1 2 1 0]        [2]               [ 8]
    //     [0 1 2 1]        [3]               [12]
    //     [0 0 1 2]        [4]               [11]
    const BK = enum { band, k };
    const K = enum { k };
    const T = f64;

    // Upper band storage (K+1=2 band rows × N=4 cols), band axis contiguous:
    // Col 0: [0, 2], Col 1: [1, 2], Col 2: [1, 2], Col 3: [1, 2]
    const ab_buf = [_]T{
        0, 2,
        1, 2,
        1, 2,
        1, 2,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0, 0 };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &y_buf,
    };

    // triangle = .k (second axis, ordinal 1) → Upper
    blas.sbmv(T, BK, K, .k, blas.BlasBand(T, BK, K).init(AB), x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 4.0, 8.0, 12.0, 11.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "sbmv column-major matrix" {
    // Same symmetric tridiagonal, column-major upper band storage
    const BK = enum { band, k };
    const K = enum { k };
    const T = f64;

    // Column-major: Col 0: [0, 2], Col 1: [1, 2], Col 2: [1, 2], Col 3: [1, 2]
    const ab_buf = [_]T{
        0, 2,
        1, 2,
        1, 2,
        1, 2,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0, 0 };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &y_buf,
    };

    blas.sbmv(T, BK, K, .k, blas.BlasBand(T, BK, K).init(AB), x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 4.0, 8.0, 12.0, 11.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "hbmv complex upper" {
    // 4×4 Hermitian tridiagonal (K=1), upper band storage:
    // A = [2    1+i  0    0   ]    x = [1]    y = [3+i ]
    //     [1-i  3    2+i  0   ]        [1]        [6   ]
    //     [0    2-i  4    1+i ]        [1]        [7   ]
    //     [0    0    1-i  5   ]        [1]        [6-i ]
    const BK = enum { band, k };
    const K = enum { k };
    const T = Complex(f64);

    // Upper band storage (K+1=2 band rows × N=4 cols), band axis contiguous:
    // Col 0: [0, 2], Col 1: [1+i, 3], Col 2: [2+i, 4], Col 3: [1+i, 5]
    const ab_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = 1 }, .{ .re = 3, .im = 0 },
        .{ .re = 2, .im = 1 }, .{ .re = 4, .im = 0 },
        .{ .re = 1, .im = 1 }, .{ .re = 5, .im = 0 },
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &y_buf,
    };

    blas.hbmv(T, BK, K, .k, blas.BlasBand(T, BK, K).init(AB), x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 3, .im = 1 },
        .{ .re = 6, .im = 0 },
        .{ .re = 7, .im = 0 },
        .{ .re = 6, .im = -1 },
    };
    const eps = 1e-10;
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "hbmv column-major matrix" {
    // Same Hermitian tridiagonal, column-major upper band storage
    const BK = enum { band, k };
    const K = enum { k };
    const T = Complex(f64);

    // Column-major: Col 0: [0, 2], Col 1: [1+i, 3], Col 2: [2+i, 4], Col 3: [1+i, 5]
    const ab_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = 1 }, .{ .re = 3, .im = 0 },
        .{ .re = 2, .im = 1 }, .{ .re = 4, .im = 0 },
        .{ .re = 1, .im = 1 }, .{ .re = 5, .im = 0 },
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const y = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &y_buf,
    };

    blas.hbmv(T, BK, K, .k, blas.BlasBand(T, BK, K).init(AB), x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 3, .im = 1 },
        .{ .re = 6, .im = 0 },
        .{ .re = 7, .im = 0 },
        .{ .re = 6, .im = -1 },
    };
    const eps = 1e-10;
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "tbmv real upper" {
    // 4×4 upper triangular bidiagonal (K=1):
    // A = [2 1 0 0]    x = [1]    x' = A*x = [ 4]
    //     [0 3 1 0]        [2]                [ 9]
    //     [0 0 4 1]        [3]                [16]
    //     [0 0 0 5]        [4]                [20]
    const BK = enum { band, k };
    const K = enum { k };
    const T = f64;

    // Upper band storage (K+1=2 band rows × N=4 cols), band axis contiguous:
    // Col 0: [0, 2], Col 1: [1, 3], Col 2: [1, 4], Col 3: [1, 5]
    const ab_buf = [_]T{
        0, 2,
        1, 3,
        1, 4,
        1, 5,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    var x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    // triangle = .k (second axis) → Upper
    blas.tbmv(T, BK, K, .k, .non_unit, blas.BlasBand(T, BK, K).init(AB), x);

    const expected = [_]T{ 4.0, 9.0, 16.0, 20.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tbmv column-major matrix" {
    // Same upper triangular bidiagonal, column-major band storage
    const BK = enum { band, k };
    const K = enum { k };
    const T = f64;

    // Column-major: Col 0: [0, 2], Col 1: [1, 3], Col 2: [1, 4], Col 3: [1, 5]
    const ab_buf = [_]T{
        0, 2,
        1, 3,
        1, 4,
        1, 5,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    var x_buf = [_]T{ 1, 2, 3, 4 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    blas.tbmv(T, BK, K, .k, .non_unit, blas.BlasBand(T, BK, K).init(AB), x);

    const expected = [_]T{ 4.0, 9.0, 16.0, 20.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tbsv real upper" {
    // Solve A*x = b where A is the same 4×4 upper triangular bidiagonal:
    // A = [2 1 0 0]    b = [4]    x = A⁻¹*b = [1]
    //     [0 3 1 0]        [9]                  [2]
    //     [0 0 4 1]        [16]                 [3]
    //     [0 0 0 5]        [20]                 [4]
    const BK = enum { band, k };
    const K = enum { k };
    const T = f64;

    // Upper band storage, band axis contiguous:
    // Col 0: [0, 2], Col 1: [1, 3], Col 2: [1, 4], Col 3: [1, 5]
    const ab_buf = [_]T{
        0, 2,
        1, 3,
        1, 4,
        1, 5,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    var x_buf = [_]T{ 4, 9, 16, 20 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    blas.tbsv(T, BK, K, .k, .non_unit, blas.BlasBand(T, BK, K).init(AB), x);

    const expected = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tbsv column-major matrix" {
    // Same solve, column-major band storage
    const BK = enum { band, k };
    const K = enum { k };
    const T = f64;

    const ab_buf = [_]T{
        0, 2,
        1, 3,
        1, 4,
        1, 5,
    };
    const AB = NamedArrayConst(BK, T){
        .idx = .{
            .shape = .{ .band = 2, .k = 4 },
            .strides = .{ .band = 1, .k = 2 },
        },
        .buf = &ab_buf,
    };

    var x_buf = [_]T{ 4, 9, 16, 20 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 4 }),
        .buf = &x_buf,
    };

    blas.tbsv(T, BK, K, .k, .non_unit, blas.BlasBand(T, BK, K).init(AB), x);

    const expected = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "her2 column-major matrix" {
    const IJ = enum { i, j };
    const I = enum { i };
    const J = enum { j };
    const T = Complex(f64);

    // Hermitian 2x2, lower triangle stored (triangle = .i, data where i >= j).
    // A_init = [[2+0i, _], [1-i, 4+0i]]
    // Column-major buffer: col0=[2+0i, 1-i], col1=[sentinel, 4+0i]
    var a_buf = [_]T{
        .{ .re = 2, .im = 0 },   .{ .re = 1, .im = -1 },
        .{ .re = 99, .im = 99 }, .{ .re = 4, .im = 0 },
    };
    const A = NamedArray(IJ, T){
        .idx = .{
            .shape = .{ .i = 2, .j = 2 },
            .strides = .{ .i = 1, .j = 2 },
        },
        .buf = &a_buf,
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const x = NamedArrayConst(I, T){
        .idx = NamedIndex(I).initContiguous(.{ .i = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 1 },
    };
    const y = NamedArrayConst(J, T){
        .idx = NamedIndex(J).initContiguous(.{ .j = 2 }),
        .buf = &y_buf,
    };

    // alpha = 1+0i (default)
    // x = [1, i], y = [1, 1+i]
    // x*y^H:
    //   (0,0) = 1*conj(1) = 1
    //   (1,0) = i*conj(1) = i
    //   (0,1) = 1*conj(1+i) = 1-i
    //   (1,1) = i*conj(1+i) = i*(1-i) = i+1 = 1+i
    // conj(alpha)*y*x^H = y*x^H:
    //   (0,0) = 1*conj(1) = 1
    //   (1,0) = (1+i)*conj(1) = 1+i
    //   (0,1) = 1*conj(i) = -i
    //   (1,1) = (1+i)*conj(i) = (1+i)*(-i) = -i+1 = 1-i
    // x*y^H + y*x^H:
    //   (0,0) = 1+1 = 2       (real, as expected for Hermitian diagonal)
    //   (1,0) = i + 1+i = 1+2i
    //   (0,1) = 1-i + (-i) = 1-2i  (conjugate of (1,0) ✓)
    //   (1,1) = 1+i + 1-i = 2      (real ✓)
    // Lower part: [[2, _], [1+2i, 2]]
    // A_new lower = [[2+2, _], [1-i+1+2i, 4+2]] = [[4+0i, _], [2+i, 6+0i]]
    // Column-major buffer: [4+0i, 2+i, sentinel, 6+0i]
    blas.her2(T, IJ, I, J, .i, A, x, y, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 4), a_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), a_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), a_buf[1].im, eps);
    // sentinel untouched
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].re);
    try std.testing.expectEqual(@as(f64, 99), a_buf[2].im);
    try std.testing.expectApproxEqAbs(@as(f64, 6), a_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), a_buf[3].im, eps);
}

test "spmv real upper" {
    // 3×3 symmetric matrix, upper packed storage:
    // A = [1 2 3]    x = [1]    y = A*x = [14]
    //     [2 5 6]        [2]               [30]
    //     [3 6 9]        [3]               [42]
    //
    // Upper packed (col-major): col0=[1], col1=[2,5], col2=[3,6,9]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    const ap = [_]T{ 1, 2, 5, 3, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .k (second axis) → Upper
    blas.spmv(T, MK, K, M, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 14.0, 30.0, 42.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spmv real lower" {
    // Same 3×3 symmetric matrix, lower packed storage:
    // Lower packed (col-major): col0=[1,2,3], col1=[5,6], col2=[9]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    const ap = [_]T{ 1, 2, 3, 5, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.spmv(T, MK, K, M, .{ .triangle = .m, .major = .k, .data = &ap }, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 14.0, 30.0, 42.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spmv real upper row-major" {
    // Same 3×3 symmetric matrix as "spmv real upper", but packed in row-major order.
    // A = [1 2 3]    x = [1]    y = A*x = [14]
    //     [2 5 6]        [2]               [30]
    //     [3 6 9]        [3]               [42]
    //
    // Upper packed (row-major): row0=[1,2,3], row1=[5,6], row2=[9]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    const ap = [_]T{ 1, 2, 3, 5, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    var y_buf = [_]T{ 0, 0, 0 };
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .k (upper), major = .m (first axis → row-major)
    blas.spmv(T, MK, K, M, .{ .triangle = .k, .major = .m, .data = &ap }, x, y, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 14.0, 30.0, 42.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spmv real nontrivial scalars and strides" {
    // Same matrix, alpha=2, beta=1, x stride=2, y stride=2
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    const ap = [_]T{ 1, 2, 5, 3, 6, 9 };

    // x = [1, 2, 3] stored with stride 2
    const x_buf = [_]T{ 1, 0, 2, 0, 3 };
    const x = NamedArrayConst(K, T){
        .idx = .{ .shape = .{ .k = 3 }, .strides = .{ .k = 2 } },
        .buf = &x_buf,
    };

    // y = [1, 1, 1] stored with stride 2, expect y = 2*A*x + y = [29, 61, 85]
    var y_buf = [_]T{ 1, 0, 1, 0, 1 };
    const y = NamedArray(M, T){
        .idx = .{ .shape = .{ .m = 3 }, .strides = .{ .m = 2 } },
        .buf = &y_buf,
    };

    blas.spmv(T, MK, K, M, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{ .alpha = 2.0, .beta = 1.0 });

    const expected = [_]T{ 29.0, 61.0, 85.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], y_buf[i * 2], math.floatEpsAt(T, expected[i]));
    }
}

test "hpmv complex upper" {
    // 3×3 Hermitian matrix, upper packed storage:
    // A = [2      1+i    3    ]    x = [1]    y = A*x = [6+i ]
    //     [1-i    4      2+i  ]        [1]               [7   ]
    //     [3      2-i    6    ]        [1]               [11-i]
    //
    // Upper packed (col-major): col0=[2], col1=[1+i, 4], col2=[3, 2+i, 6]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = Complex(f64);

    const ap = [_]T{
        .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = 1 },
        .{ .re = 4, .im = 0 },
        .{ .re = 3, .im = 0 },
        .{ .re = 2, .im = 1 },
        .{ .re = 6, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    blas.hpmv(T, MK, K, M, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 6, .im = 1 },
        .{ .re = 7, .im = 0 },
        .{ .re = 11, .im = -1 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "hpmv complex lower" {
    // Same Hermitian matrix, lower packed storage:
    // Lower packed (col-major): col0=[2, 1-i, 3], col1=[4, 2-i], col2=[6]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = Complex(f64);

    const ap = [_]T{
        .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = -1 },
        .{ .re = 3, .im = 0 },
        .{ .re = 4, .im = 0 },
        .{ .re = 2, .im = -1 },
        .{ .re = 6, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.hpmv(T, MK, K, M, .{ .triangle = .m, .major = .k, .data = &ap }, x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 6, .im = 1 },
        .{ .re = 7, .im = 0 },
        .{ .re = 11, .im = -1 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "hpmv complex upper row-major" {
    // Same 3×3 Hermitian matrix as "hpmv complex upper", but packed in row-major order.
    // A = [2      1+i    3    ]    x = [1]    y = A*x = [6+i ]
    //     [1-i    4      2+i  ]        [1]               [7   ]
    //     [3      2-i    6    ]        [1]               [11-i]
    //
    // Upper packed (row-major): row0=[2, 1+i, 3], row1=[4, 2+i], row2=[6]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = Complex(f64);

    const ap = [_]T{
        .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = 1 },
        .{ .re = 3, .im = 0 },
        .{ .re = 4, .im = 0 },
        .{ .re = 2, .im = 1 },
        .{ .re = 6, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
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
    const y = NamedArray(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .k (upper), major = .m (first axis → row-major)
    blas.hpmv(T, MK, K, M, .{ .triangle = .k, .major = .m, .data = &ap }, x, y, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 6, .im = 1 },
        .{ .re = 7, .im = 0 },
        .{ .re = 11, .im = -1 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, y_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, y_buf[i].im, eps);
    }
}

test "tpmv real upper" {
    // 3×3 upper triangular, packed storage:
    // A = [2 1 3]    x = [1]    A*x = [13]
    //     [0 4 2]        [2]          [14]
    //     [0 0 5]        [3]          [15]
    //
    // Upper packed (col-major): col0=[2], col1=[1,4], col2=[3,2,5]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    const ap = [_]T{ 2, 1, 4, 3, 2, 5 };

    var x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // triangle = .k (second axis) → Upper
    blas.tpmv(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, .non_unit, x);

    const expected = [_]T{ 13.0, 14.0, 15.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tpmv real lower" {
    // 3×3 lower triangular, packed storage:
    // A = [2 0 0]    x = [1]    A*x = [ 2]
    //     [1 4 0]        [2]          [ 9]
    //     [3 2 5]        [3]          [22]
    //
    // Lower packed (col-major): col0=[2,1,3], col1=[4,2], col2=[5]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    const ap = [_]T{ 2, 1, 3, 4, 2, 5 };

    var x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.tpmv(T, MK, K, .{ .triangle = .m, .major = .k, .data = &ap }, .non_unit, x);

    const expected = [_]T{ 2.0, 9.0, 22.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tpmv real unit diagonal" {
    // 3×3 upper triangular with unit diagonal:
    // A = [1 2 3]    x = [1]    A*x = [14]
    //     [0 1 1]        [2]          [ 5]
    //     [0 0 1]        [3]          [ 3]
    //
    // Upper packed: [*, 2, *, 3, 1, *] — diagonal entries are not read.
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    const ap = [_]T{ 0, 2, 0, 3, 1, 0 }; // diagonal positions contain 0 (ignored)

    var x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.tpmv(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, .unit, x);

    const expected = [_]T{ 14.0, 5.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tpmv complex upper" {
    // 3×3 upper triangular complex:
    // A = [2    1+i  3  ]    x = [1]    A*x = [2+2+2i+9]   = [13+2i]
    //     [0    4    2-i]        [2]          [0+8+6-3i]     [14-3i]
    //     [0    0    5  ]        [3]          [0+0+15]       [15   ]
    //
    // Upper packed: [2, 1+i, 4, 3, 2-i, 5]
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    const ap = [_]T{
        .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = 1 },
        .{ .re = 4, .im = 0 },
        .{ .re = 3, .im = 0 },
        .{ .re = 2, .im = -1 },
        .{ .re = 5, .im = 0 },
    };

    var x_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 2, .im = 0 },
        .{ .re = 3, .im = 0 },
    };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.tpmv(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, .non_unit, x);

    const expected = [_]T{
        .{ .re = 13, .im = 2 },
        .{ .re = 14, .im = -3 },
        .{ .re = 15, .im = 0 },
    };
    const eps = 1e-10;
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, x_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, x_buf[i].im, eps);
    }
}

test "tpmv nontrivial strides" {
    // Upper triangular, x with stride 2
    const AB = enum { a, b };
    const B = enum { b };
    const T = f64;

    const ap = [_]T{ 2, 1, 4, 3, 2, 5 };

    // x = [1, 2, 3] stored with stride 2
    var x_buf = [_]T{ 1, 0, 2, 0, 3 };
    const x = NamedArray(B, T){
        .idx = .{ .shape = .{ .b = 3 }, .strides = .{ .b = 2 } },
        .buf = &x_buf,
    };

    blas.tpmv(T, AB, B, .{ .triangle = .b, .major = .b, .data = &ap }, .non_unit, x);

    const expected = [_]T{ 13.0, 14.0, 15.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i * 2], math.floatEpsAt(T, expected[i]));
    }
}

test "tpsv real upper" {
    // Solve A*x = b where A is 3×3 upper triangular:
    // A = [2 1 3]    b = [13]    x = A⁻¹*b = [1]
    //     [0 4 2]        [14]                 [2]
    //     [0 0 5]        [15]                 [3]
    //
    // Upper packed: [2, 1, 4, 3, 2, 5]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    const ap = [_]T{ 2, 1, 4, 3, 2, 5 };

    var x_buf = [_]T{ 13, 14, 15 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.tpsv(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, .non_unit, x);

    const expected = [_]T{ 1.0, 2.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tpsv real lower" {
    // Solve A*x = b where A is 3×3 lower triangular:
    // A = [2 0 0]    b = [ 2]    x = A⁻¹*b = [1]
    //     [1 4 0]        [ 9]                 [2]
    //     [3 2 5]        [22]                 [3]
    //
    // Lower packed: [2, 1, 3, 4, 2, 5]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    const ap = [_]T{ 2, 1, 3, 4, 2, 5 };

    var x_buf = [_]T{ 2, 9, 22 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.tpsv(T, MK, K, .{ .triangle = .m, .major = .k, .data = &ap }, .non_unit, x);

    const expected = [_]T{ 1.0, 2.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tpsv real unit diagonal" {
    // Solve A*x = b where A is 3×3 upper triangular with unit diagonal:
    // A = [1 2 3]    b = [14]    x = A⁻¹*b = [1]
    //     [0 1 1]        [ 5]                 [2]
    //     [0 0 1]        [ 3]                 [3]
    //
    // Upper packed: [*, 2, *, 3, 1, *]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    const ap = [_]T{ 0, 2, 0, 3, 1, 0 };

    var x_buf = [_]T{ 14, 5, 3 };
    const x = NamedArray(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.tpsv(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, .unit, x);

    const expected = [_]T{ 1.0, 2.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "tpsv nontrivial strides" {
    // Upper triangular solve, x with stride 2
    const AB = enum { a, b };
    const B = enum { b };
    const T = f64;

    const ap = [_]T{ 2, 1, 4, 3, 2, 5 };

    // b = [13, 14, 15] stored with stride 2
    var x_buf = [_]T{ 13, 0, 14, 0, 15 };
    const x = NamedArray(B, T){
        .idx = .{ .shape = .{ .b = 3 }, .strides = .{ .b = 2 } },
        .buf = &x_buf,
    };

    blas.tpsv(T, AB, B, .{ .triangle = .b, .major = .b, .data = &ap }, .non_unit, x);

    const expected = [_]T{ 1.0, 2.0, 3.0 };
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(expected[i], x_buf[i * 2], math.floatEpsAt(T, expected[i]));
    }
}

test "spr real upper" {
    // 3×3 symmetric matrix, upper packed storage:
    // A_init = [1 2 3]    x = [1]
    //          [2 5 6]        [2]
    //          [3 6 9]        [3]
    //
    // x*x^T = [[1,2,3],[2,4,6],[3,6,9]]
    // A_new = A_init + x*x^T:
    //   upper: [0,0]=2  [0,1]=4  [0,2]=6  [1,1]=9  [1,2]=12  [2,2]=18
    //
    // Upper packed (col-major): col0=[1], col1=[2,5], col2=[3,6,9]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    var ap = [_]T{ 1, 2, 5, 3, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    blas.spr(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, x, .{});

    const expected = [_]T{ 2, 4, 9, 6, 12, 18 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spr real lower" {
    // Same 3×3 symmetric matrix, lower packed storage:
    // Lower packed (col-major): col0=[1,2,3], col1=[5,6], col2=[9]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    var ap = [_]T{ 1, 2, 3, 5, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.spr(T, MK, K, .{ .triangle = .m, .major = .k, .data = &ap }, x, .{});

    const expected = [_]T{ 2, 4, 6, 9, 12, 18 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spr real upper row-major" {
    // Same 3×3 symmetric matrix as "spr real upper", but packed in row-major order.
    // A_init = [1 2 3]    x = [1]
    //          [2 5 6]        [2]
    //          [3 6 9]        [3]
    //
    // x*x^T = [[1,2,3],[2,4,6],[3,6,9]]
    // A_new upper: [0,0]=2  [0,1]=4  [0,2]=6  [1,1]=9  [1,2]=12  [2,2]=18
    //
    // Upper packed (row-major): row0=[1,2,3], row1=[5,6], row2=[9]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    var ap = [_]T{ 1, 2, 3, 5, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    // triangle = .k (upper), major = .m (first axis → row-major)
    blas.spr(T, MK, K, .{ .triangle = .k, .major = .m, .data = &ap }, x, .{});

    // Row-major upper: [A00, A01, A02, A11, A12, A22]
    const expected = [_]T{ 2, 4, 6, 9, 12, 18 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spr nontrivial alpha and strides" {
    // Same matrix, alpha=2, x stored with stride 2
    // A_new = 2 * x*x^T + A_init
    // 2*x*x^T = [[2,4,6],[4,8,12],[6,12,18]]
    // Upper packed: [1+2, 2+4, 5+8, 3+6, 6+12, 9+18] = [3, 6, 13, 9, 18, 27]
    const MK = enum { m, k };
    const K = enum { k };
    const T = f64;

    var ap = [_]T{ 1, 2, 5, 3, 6, 9 };

    const x_buf = [_]T{ 1, 0, 2, 0, 3 };
    const x = NamedArrayConst(K, T){
        .idx = .{ .shape = .{ .k = 3 }, .strides = .{ .k = 2 } },
        .buf = &x_buf,
    };

    blas.spr(T, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, x, .{ .alpha = 2.0 });

    const expected = [_]T{ 3, 6, 13, 9, 18, 27 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "hpr complex upper" {
    // 2×2 Hermitian, upper packed storage:
    // A_init = [[1+0i,  2+i ],
    //           [2-i,   3+0i]]
    //
    // x = [1+i, 2]
    // x*x^H:
    //   (1+i)*conj(1+i) = (1+i)*(1-i) = 2
    //   (1+i)*conj(2)   = (1+i)*2     = 2+2i
    //   2*conj(1+i)     = 2*(1-i)     = 2-2i
    //   2*conj(2)       = 4
    // x*x^H = [[2, 2+2i], [2-2i, 4]]
    //
    // A_new upper = [[1+2, 2+i+2+2i], [_, 3+4]] = [[3+0i, 4+3i], [_, 7+0i]]
    // Upper packed (col-major): col0=[1+0i], col1=[2+i, 3+0i]
    // After: [3+0i, 4+3i, 7+0i]
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    var ap = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 2, .im = 1 },
        .{ .re = 3, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &x_buf,
    };

    blas.hpr(f64, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, x, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), ap[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), ap[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), ap[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 7), ap[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[2].im, eps);
}

test "hpr complex lower" {
    // Same 2×2 Hermitian, lower packed storage:
    // Lower packed (col-major): col0=[1+0i, 2-i], col1=[3+0i]
    // x*x^H = [[2, 2+2i], [2-2i, 4]]
    // A_new lower: [[1+2, _], [2-i+2-2i, 3+4]] = [[3+0i, _], [4-3i, 7+0i]]
    // After: [3+0i, 4-3i, 7+0i]
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f64);

    var ap = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 2, .im = -1 },
        .{ .re = 3, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &x_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.hpr(f64, MK, K, .{ .triangle = .m, .major = .k, .data = &ap }, x, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), ap[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), ap[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, -3), ap[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 7), ap[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[2].im, eps);
}

test "hpr nontrivial alpha and strides" {
    // 2×2 Hermitian upper, alpha=2, x with stride 2
    // A_init upper packed: [5+0i, 1+i, 3+0i]
    // x physical = [1+i, 0, 2], stride=2 → logical x = [1+i, 2]
    // x*x^H = [[2, 2+2i], [2-2i, 4]]
    // 2*x*x^H = [[4, 4+4i], [4-4i, 8]]
    // A_new upper: [5+4, 1+i+4+4i, 3+8] = [9+0i, 5+5i, 11+0i]
    const MK = enum { m, k };
    const K = enum { k };
    const T = Complex(f32);

    var ap = [_]T{
        .{ .re = 5, .im = 0 },
        .{ .re = 1, .im = 1 },
        .{ .re = 3, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 0, .im = 0 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(K, T){
        .idx = .{ .shape = .{ .k = 2 }, .strides = .{ .k = 2 } },
        .buf = &x_buf,
    };

    blas.hpr(f32, MK, K, .{ .triangle = .k, .major = .k, .data = &ap }, x, .{ .alpha = 2.0 });

    const eps: f32 = 1e-4;
    try std.testing.expectApproxEqAbs(@as(f32, 9), ap[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ap[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 5), ap[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 5), ap[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 11), ap[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ap[2].im, eps);
}

test "spr2 real upper" {
    // 3×3 symmetric, upper packed storage:
    // A_init = [1 2 3]    x = [1]    y = [1]
    //          [2 5 6]        [2]        [0]
    //          [3 6 9]        [3]        [1]
    //
    // x*y^T = [[1,0,1],[2,0,2],[3,0,3]]
    // y*x^T = [[1,2,3],[0,0,0],[1,2,3]]
    // x*y^T + y*x^T = [[2,2,4],[2,0,2],[4,2,6]]
    // A_new upper: [1+2, 2+2, 5+0, 3+4, 6+2, 9+6] = [3, 4, 5, 7, 8, 15]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    var ap = [_]T{ 1, 2, 5, 3, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 1, 0, 1 };
    const y = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    blas.spr2(T, MK, K, M, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{});

    const expected = [_]T{ 3, 4, 5, 7, 8, 15 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spr2 real lower" {
    // Same matrix, lower packed storage:
    // Lower packed (col-major): col0=[1,2,3], col1=[5,6], col2=[9]
    // x*y^T + y*x^T = [[2,2,4],[2,0,2],[4,2,6]]
    // A_new lower: [1+2, 2+2, 3+4, 5+0, 6+2, 9+6] = [3, 4, 7, 5, 8, 15]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    var ap = [_]T{ 1, 2, 3, 5, 6, 9 };

    const x_buf = [_]T{ 1, 2, 3 };
    const x = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 3 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 1, 0, 1 };
    const y = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 3 }),
        .buf = &y_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.spr2(T, MK, K, M, .{ .triangle = .m, .major = .k, .data = &ap }, x, y, .{});

    const expected = [_]T{ 3, 4, 7, 5, 8, 15 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "spr2 nontrivial alpha and strides" {
    // Upper packed, alpha=2, x stride=2, y stride=2
    // x = [1,2,3], y = [1,0,1]
    // 2*(x*y^T + y*x^T) = 2*[[2,2,4],[2,0,2],[4,2,6]] = [[4,4,8],[4,0,4],[8,4,12]]
    // Upper packed: [1+4, 2+4, 5+0, 3+8, 6+4, 9+12] = [5, 6, 5, 11, 10, 21]
    const MK = enum { m, k };
    const K = enum { k };
    const M = enum { m };
    const T = f64;

    var ap = [_]T{ 1, 2, 5, 3, 6, 9 };

    const x_buf = [_]T{ 1, 0, 2, 0, 3 };
    const x = NamedArrayConst(K, T){
        .idx = .{ .shape = .{ .k = 3 }, .strides = .{ .k = 2 } },
        .buf = &x_buf,
    };

    const y_buf = [_]T{ 1, 0, 0, 0, 1 };
    const y = NamedArrayConst(M, T){
        .idx = .{ .shape = .{ .m = 3 }, .strides = .{ .m = 2 } },
        .buf = &y_buf,
    };

    blas.spr2(T, MK, K, M, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{ .alpha = 2.0 });

    const expected = [_]T{ 5, 6, 5, 11, 10, 21 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], ap[i], math.floatEpsAt(T, expected[i]));
    }
}

test "hpr2 complex upper" {
    // 2×2 Hermitian, upper packed storage:
    // A_init upper packed: [1+0i, 2+i, 3+0i]
    // A_init = [[1+0i, 2+i], [2-i, 3+0i]]
    //
    // x = [1+i, 2], y = [i, 1-i], alpha = 1+0i (default)
    //
    // x*y^H:
    //   (1+i)*conj(i) = (1+i)*(-i) = -i-i^2 = 1-i
    //   (1+i)*conj(1-i) = (1+i)*(1+i) = 1+2i-1 = 2i
    //   2*conj(i) = -2i
    //   2*conj(1-i) = 2*(1+i) = 2+2i
    // x*y^H = [[1-i, 2i], [-2i, 2+2i]]
    //
    // y*x^H:
    //   i*conj(1+i) = i*(1-i) = i+1 = 1+i
    //   i*conj(2) = 2i
    //   (1-i)*conj(1+i) = (1-i)*(1-i) = 1-2i-1 = -2i
    //   (1-i)*conj(2) = 2-2i
    // y*x^H = [[1+i, 2i], [-2i, 2-2i]]
    //
    // x*y^H + y*x^H = [[2+0i, 4i], [-4i, 4+0i]]
    //
    // A_new upper packed: [1+2, 2+i+4i, 3+4] = [3+0i, 2+5i, 7+0i]
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    var ap = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 2, .im = 1 },
        .{ .re = 3, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = -1 },
    };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &y_buf,
    };

    blas.hpr2(T, MK, M, K, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), ap[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), ap[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 5), ap[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 7), ap[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[2].im, eps);
}

test "hpr2 complex lower" {
    // Same 2×2 Hermitian, lower packed storage:
    // A_init lower packed: [1+0i, 2-i, 3+0i]
    //
    // x*y^H + y*x^H = [[2+0i, 4i], [-4i, 4+0i]]  (same as above)
    //
    // A_new lower packed: [1+2, 2-i-4i, 3+4] = [3+0i, 2-5i, 7+0i]
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f64);

    var ap = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 2, .im = -1 },
        .{ .re = 3, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const x = NamedArrayConst(M, T){
        .idx = NamedIndex(M).initContiguous(.{ .m = 2 }),
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 0, .im = 1 },
        .{ .re = 1, .im = -1 },
    };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &y_buf,
    };

    // triangle = .m (first axis) → Lower
    blas.hpr2(T, MK, M, K, .{ .triangle = .m, .major = .k, .data = &ap }, x, y, .{});

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), ap[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), ap[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, -5), ap[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 7), ap[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ap[2].im, eps);
}

test "hpr2 nontrivial alpha and strides" {
    // 2×2 Hermitian upper, alpha = 2+i, x with stride 2
    // A_init upper packed: [5+0i, 1+i, 3+0i]
    //
    // x physical = [i, 0, 1+i], stride=2 → logical x = [i, 1+i]
    // y = [1, i]
    //
    // alpha * x * y^H:
    //   x*y^H:
    //     i*conj(1) = i,  i*conj(i) = i*(-i) = 1
    //     (1+i)*conj(1) = 1+i,  (1+i)*conj(i) = (1+i)*(-i) = -i+1 = 1-i
    //   x*y^H = [[i, 1], [1+i, 1-i]]
    //   alpha*x*y^H = (2+i)*[[i, 1], [1+i, 1-i]]
    //     [0,0] = (2+i)*i = -1+2i
    //     [0,1] = (2+i)*1 = 2+i
    //     [1,0] = (2+i)*(1+i) = 2+2i+i-1 = 1+3i
    //     [1,1] = (2+i)*(1-i) = 2-2i+i+1 = 3-i
    //
    // conj(alpha) * y * x^H:
    //   y*x^H:
    //     1*conj(i) = -i,  1*conj(1+i) = 1-i
    //     i*conj(i) = 1,  i*conj(1+i) = i-i^2 = 1+i
    //   y*x^H = [[-i, 1-i], [1, 1+i]]
    //   conj(alpha)*y*x^H = (2-i)*[[-i, 1-i], [1, 1+i]]
    //     [0,0] = (2-i)*(-i) = -2i+i^2 = -1-2i
    //     [0,1] = (2-i)*(1-i) = 2-2i-i+i^2 = 1-3i
    //     [1,0] = (2-i)*1 = 2-i
    //     [1,1] = (2-i)*(1+i) = 2+2i-i+1 = 3+i
    //
    // sum = alpha*x*y^H + conj(alpha)*y*x^H:
    //   [0,0] = -1+2i + -1-2i = -2+0i
    //   [0,1] = 2+i + 1-3i = 3-2i
    //   [1,0] = 1+3i + 2-i = 3+2i
    //   [1,1] = 3-i + 3+i = 6+0i
    //
    // A_new upper packed: [5-2, 1+i+3-2i, 3+6] = [3+0i, 4-i, 9+0i]
    const MK = enum { m, k };
    const M = enum { m };
    const K = enum { k };
    const T = Complex(f32);

    var ap = [_]T{
        .{ .re = 5, .im = 0 },
        .{ .re = 1, .im = 1 },
        .{ .re = 3, .im = 0 },
    };

    const x_buf = [_]T{
        .{ .re = 0, .im = 1 },
        .{ .re = 0, .im = 0 },
        .{ .re = 1, .im = 1 },
    };
    const x = NamedArrayConst(M, T){
        .idx = .{ .shape = .{ .m = 2 }, .strides = .{ .m = 2 } },
        .buf = &x_buf,
    };

    const y_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const y = NamedArrayConst(K, T){
        .idx = NamedIndex(K).initContiguous(.{ .k = 2 }),
        .buf = &y_buf,
    };

    blas.hpr2(T, MK, M, K, .{ .triangle = .k, .major = .k, .data = &ap }, x, y, .{ .alpha = .{ .re = 2, .im = 1 } });

    const eps: f32 = 1e-4;
    try std.testing.expectApproxEqAbs(@as(f32, 3), ap[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ap[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 4), ap[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, -1), ap[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 9), ap[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ap[2].im, eps);
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

// ─────────────────────────────────────────────────────────────
//  Level 3 BLAS tests
// ─────────────────────────────────────────────────────────────

test "gemm real 3x3" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (identity)
    const b_buf = [_]T{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 3, .n = 3 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 3, .n = 3 }),
        .buf = &c_buf,
    };

    // C = 1*A*I + 0*C = A
    blas.gemm(T, MK, KN, MN, A, B, C, .{ .alpha = 1.0, .beta = 0.0 });

    for (0..9) |i| {
        try std.testing.expectApproxEqAbs(a_buf[i], c_buf[i], math.floatEpsAt(T, a_buf[i]));
    }
}

test "gemm real rectangular" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x3) = [[1, 2, 3], [4, 5, 6]]
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 3 }),
        .buf = &a_buf,
    };

    // B (3x2) = [[7, 8], [9, 10], [11, 12]]
    const b_buf = [_]T{ 7, 8, 9, 10, 11, 12 };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 3, .n = 2 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // C = A*B = [[58, 64], [139, 154]]
    blas.gemm(T, MK, KN, MN, A, B, C, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 58, 64, 139, 154 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemm real nontrivial scalars" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) = [[1, 2], [3, 4]]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x2) = [[5, 6], [7, 8]]
    const b_buf = [_]T{ 5, 6, 7, 8 };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 2 }),
        .buf = &b_buf,
    };

    // C (2x2) = [[1, 1], [1, 1]]
    var c_buf = [_]T{ 1, 1, 1, 1 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // C = 2*A*B + 3*C
    // A*B = [[19, 22], [43, 50]]
    // 2*A*B = [[38, 44], [86, 100]]
    // 3*C = [[3, 3], [3, 3]]
    // result = [[41, 47], [89, 103]]
    blas.gemm(T, MK, KN, MN, A, B, C, .{ .alpha = 2.0, .beta = 3.0 });

    const expected = [_]T{ 41, 47, 89, 103 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemm real column-major C" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) row-major = [[1, 2], [3, 4]]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x2) row-major = [[5, 6], [7, 8]]
    const b_buf = [_]T{ 5, 6, 7, 8 };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 2 }),
        .buf = &b_buf,
    };

    // C column-major: col0=[0,0], col1=[0,0]
    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = .{
            .shape = .{ .m = 2, .n = 2 },
            .strides = .{ .m = 1, .n = 2 },
        },
        .buf = &c_buf,
    };

    // C = A*B = [[19, 22], [43, 50]]
    // Column-major storage: [19, 43, 22, 50]
    blas.gemm(T, MK, KN, MN, A, B, C, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 19, 43, 22, 50 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "gemm complex" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x2): [[1+i, 2+0i], [0+i, 1+0i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 },
        .{ .re = 0, .im = 1 }, .{ .re = 1, .im = 0 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x2): [[1+0i, 0+i], [0+0i, 1+i]]
    const b_buf = [_]T{
        .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 },
        .{ .re = 0, .im = 0 }, .{ .re = 1, .im = 1 },
    };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 2 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // C = A*B:
    // C[0,0] = (1+i)*1 + 2*0 = 1+i
    // C[0,1] = (1+i)*(i) + 2*(1+i) = i+i² + 2+2i = -1+i + 2+2i = 1+3i
    // C[1,0] = (i)*1 + 1*0 = i
    // C[1,1] = (i)*(i) + 1*(1+i) = -1 + 1+i = i
    blas.gemm(T, MK, KN, MN, A, B, C, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const expected = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 1, .im = 3 },
        .{ .re = 0, .im = 1 },
        .{ .re = 0, .im = 1 },
    };
    const eps = 1e-10;
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i].re, c_buf[i].re, eps);
        try std.testing.expectApproxEqAbs(expected[i].im, c_buf[i].im, eps);
    }
}

test "gemm real 1x1" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    const a_buf = [_]T{3.0};
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 1, .k = 1 }),
        .buf = &a_buf,
    };
    const b_buf = [_]T{4.0};
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 1, .n = 1 }),
        .buf = &b_buf,
    };
    var c_buf = [_]T{1.0};
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 1, .n = 1 }),
        .buf = &c_buf,
    };

    // C = 2*3*4 + 5*1 = 24 + 5 = 29
    blas.gemm(T, MK, KN, MN, A, B, C, .{ .alpha = 2.0, .beta = 5.0 });
    try std.testing.expectApproxEqAbs(@as(T, 29.0), c_buf[0], 1e-10);
}

test "gemm all column-major" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) col-major: [[1, 3], [2, 4]] stored as [1, 2, 3, 4]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 2 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x2) col-major: [[5, 7], [6, 8]] stored as [5, 6, 7, 8]
    const b_buf = [_]T{ 5, 6, 7, 8 };
    const B = NamedArrayConst(KN, T){
        .idx = .{ .shape = .{ .k = 2, .n = 2 }, .strides = .{ .k = 1, .n = 2 } },
        .buf = &b_buf,
    };

    // C (2x2) col-major
    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &c_buf,
    };

    // A = [[1, 3], [2, 4]], B = [[5, 7], [6, 8]]
    // A*B = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]] = [[23, 31], [34, 46]]
    // Col-major storage: [23, 34, 31, 46]
    blas.gemm(T, MK, KN, MN, A, B, C, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 23, 34, 31, 46 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symm real left (triangle = second axis)" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (3x3) symmetric upper: [[1, 2, 3], [2, 5, 6], [3, 6, 9]]
    // Only upper triangle is read: 1, 2, 3, *, 5, 6, *, *, 9
    const a_buf = [_]T{ 1, 2, 3, 2, 5, 6, 3, 6, 9 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // B (3x2) = [[1, 0], [0, 1], [1, 1]]
    const b_buf = [_]T{ 1, 0, 0, 1, 1, 1 };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 3, .n = 2 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{ 0, 0, 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 3, .n = 2 }),
        .buf = &c_buf,
    };

    // C = A*B:
    // row0: 1*1+2*0+3*1=4, 1*0+2*1+3*1=5
    // row1: 2*1+5*0+6*1=8, 2*0+5*1+6*1=11
    // row2: 3*1+6*0+9*1=12, 3*0+6*1+9*1=15
    blas.symm(T, MK, KN, MN, .{ .triangle = .k, .data = A }, B, C, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 4, 5, 8, 11, 12, 15 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symm real nontrivial scalars" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) symmetric: [[2, 1], [1, 3]]
    const a_buf = [_]T{ 2, 1, 1, 3 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x2) = [[1, 2], [3, 4]]
    const b_buf = [_]T{ 1, 2, 3, 4 };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 2 }),
        .buf = &b_buf,
    };

    // C (2x2) = [[10, 10], [10, 10]]
    var c_buf = [_]T{ 10, 10, 10, 10 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // A*B = [[2+3, 4+4], [1+9, 2+12]] = [[5, 8], [10, 14]]
    // C = 2*A*B + 3*C = [[10+30, 16+30], [20+30, 28+30]] = [[40, 46], [50, 58]]
    blas.symm(T, MK, KN, MN, .{ .triangle = .k, .data = A }, B, C, .{ .alpha = 2.0, .beta = 3.0 });

    const expected = [_]T{ 40, 46, 50, 58 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "symm real column-major" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) symmetric col-major: [[2, 1], [1, 3]] stored as [2, 1, 1, 3]
    const a_buf = [_]T{ 2, 1, 1, 3 };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 2 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x2) col-major: [[1, 3], [2, 4]] stored as [1, 2, 3, 4]
    const b_buf = [_]T{ 1, 2, 3, 4 };
    const B = NamedArrayConst(KN, T){
        .idx = .{ .shape = .{ .k = 2, .n = 2 }, .strides = .{ .k = 1, .n = 2 } },
        .buf = &b_buf,
    };

    // C (2x2) col-major
    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &c_buf,
    };

    // A = [[2, 1], [1, 3]], B = [[1, 3], [2, 4]]
    // A*B = [[2+2, 6+4], [1+6, 3+12]] = [[4, 10], [7, 15]]
    // Col-major storage: [4, 7, 10, 15]
    blas.symm(T, MK, KN, MN, .{ .triangle = .k, .data = A }, B, C, .{ .alpha = 1.0, .beta = 0.0 });

    const expected = [_]T{ 4, 7, 10, 15 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], c_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "hemm complex left (triangle = second axis)" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x2) Hermitian: [[2+0i, 1-i], [1+i, 3+0i]]
    // Upper triangle: 2, 1-i, *, 3
    const a_buf = [_]T{
        .{ .re = 2, .im = 0 }, .{ .re = 1, .im = -1 },
        .{ .re = 1, .im = 1 }, .{ .re = 3, .im = 0 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[1+0i], [0+i]]
    const b_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const B = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 1 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &c_buf,
    };

    // C = A*B:
    // row0: 2*1 + (1-i)*(i) = 2 + i-i² = 2+i+1 = 3+i
    // row1: (1+i)*1 + 3*(i) = 1+i+3i = 1+4i
    blas.hemm(T, MK, KN, MN, .{ .triangle = .k, .data = A }, B, C, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 3), c_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), c_buf[1].im, eps);
}

test "hemm complex column-major" {
    const MK = enum { m, k };
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x2) Hermitian col-major: [[2+0i, 1-i], [1+i, 3+0i]]
    // stored col-major: [2, 1+i, 1-i, 3]
    const a_buf = [_]T{
        .{ .re = 2, .im = 0 },
        .{ .re = 1, .im = 1 },
        .{ .re = 1, .im = -1 },
        .{ .re = 3, .im = 0 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 2 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x2) col-major: [[1, 0], [0, 1]] stored as [1, 0, 0, 1]
    const b_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 1, .im = 0 },
    };
    const B = NamedArrayConst(KN, T){
        .idx = .{ .shape = .{ .k = 2, .n = 2 }, .strides = .{ .k = 1, .n = 2 } },
        .buf = &b_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &c_buf,
    };

    // C = A * I = A (col-major storage)
    // A = [[2, 1-i], [1+i, 3]]
    // Col-major: [2, 1+i, 1-i, 3]
    blas.hemm(T, MK, KN, MN, .{ .triangle = .k, .data = A }, B, C, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = .{ .re = 0, .im = 0 },
    });

    const eps = 1e-10;
    // Col-major: col0=[2, 1+i], col1=[1-i, 3]
    try std.testing.expectApproxEqAbs(@as(f64, 2), c_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, -1), c_buf[2].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), c_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].im, eps);
}

test "syrk real upper (triangle = second axis)" {
    const NK = enum { n, k };
    const NN = enum { n, n2 };
    const T = f64;

    // A (2x3) = [[1, 2, 3], [4, 5, 6]]
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6 };
    const A = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 3 }),
        .buf = &a_buf,
    };

    // C (2x2) = [[0, 0], [0, 0]]
    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(NN, T){
        .idx = NamedIndex(NN).initContiguous(.{ .n = 2, .n2 = 2 }),
        .buf = &c_buf,
    };

    // C = A * Aᵀ:
    // C[0,0] = 1+4+9 = 14
    // C[0,1] = 4+10+18 = 32
    // C[1,0] = (not read/written in upper)
    // C[1,1] = 16+25+36 = 77
    blas.syrk(T, NK, NN, A, .{ .triangle = .n2, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 14), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 32), c_buf[1], 1e-10);
    // c_buf[2] is in the lower triangle (not written with beta=0, but CBLAS may set it)
    try std.testing.expectApproxEqAbs(@as(T, 77), c_buf[3], 1e-10);
}

test "syrk real lower (triangle = first axis)" {
    const NK = enum { n, k };
    const NN = enum { n, n2 };
    const T = f64;

    // A (2x3) = [[1, 2, 3], [4, 5, 6]]
    const a_buf = [_]T{ 1, 2, 3, 4, 5, 6 };
    const A = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 3 }),
        .buf = &a_buf,
    };

    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(NN, T){
        .idx = NamedIndex(NN).initContiguous(.{ .n = 2, .n2 = 2 }),
        .buf = &c_buf,
    };

    // C = A * Aᵀ (lower triangle):
    // C[0,0] = 14, C[1,0] = 32, C[1,1] = 77
    blas.syrk(T, NK, NN, A, .{ .triangle = .n, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 14), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 32), c_buf[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 77), c_buf[3], 1e-10);
}

test "syrk real nontrivial scalars" {
    const NK = enum { n, k };
    const NN = enum { n, n2 };
    const T = f64;

    // A (2x2) = [[1, 2], [3, 4]]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // C (2x2) = [[1, 1], [1, 1]]
    var c_buf = [_]T{ 1, 1, 1, 1 };
    const C = NamedArray(NN, T){
        .idx = NamedIndex(NN).initContiguous(.{ .n = 2, .n2 = 2 }),
        .buf = &c_buf,
    };

    // A*Aᵀ = [[1+4, 3+8], [3+8, 9+16]] = [[5, 11], [11, 25]]
    // C = 2*A*Aᵀ + 3*C = [[10+3, 22+3], [22+3, 50+3]] = [[13, 25], [25, 53]]
    blas.syrk(T, NK, NN, A, .{ .triangle = .n2, .data = C }, .{ .alpha = 2.0, .beta = 3.0 });

    try std.testing.expectApproxEqAbs(@as(T, 13), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 25), c_buf[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 53), c_buf[3], 1e-10);
}

test "syrk column-major" {
    const NK = enum { n, k };
    const NN = enum { n, n2 };
    const T = f64;

    // A (2x2) col-major: [[1, 3], [2, 4]] stored as [1, 2, 3, 4]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(NK, T){
        .idx = .{ .shape = .{ .n = 2, .k = 2 }, .strides = .{ .n = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // C (2x2) col-major
    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(NN, T){
        .idx = .{ .shape = .{ .n = 2, .n2 = 2 }, .strides = .{ .n = 1, .n2 = 2 } },
        .buf = &c_buf,
    };

    // A = [[1, 3], [2, 4]]
    // A*Aᵀ = [[1+9, 2+12], [2+12, 4+16]] = [[10, 14], [14, 20]]
    // Use lower triangle (.n = first axis, ordinal 0) so that C[0,0], C[1,0], C[1,1] are written.
    // Col-major storage: c_buf[0]=C[0,0]=10, c_buf[1]=C[1,0]=14, c_buf[3]=C[1,1]=20
    blas.syrk(T, NK, NN, A, .{ .triangle = .n, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 10), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 14), c_buf[1], 1e-10);
    // c_buf[2] = C[0,1] is upper triangle, not written
    try std.testing.expectApproxEqAbs(@as(T, 20), c_buf[3], 1e-10);
}

test "herk complex upper" {
    const NK = enum { n, k };
    const NN = enum { n, n2 };
    const T = Complex(f64);

    // A (2x2) = [[1+i, 2+0i], [0+i, 1-i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 },
        .{ .re = 0, .im = 1 }, .{ .re = 1, .im = -1 },
    };
    const A = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 2 }),
        .buf = &a_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(NN, T){
        .idx = NamedIndex(NN).initContiguous(.{ .n = 2, .n2 = 2 }),
        .buf = &c_buf,
    };

    // C = A * Aᴴ:
    // C[0,0] = (1+i)(1-i) + 2*2 = 1+1 + 4 = 6 (real, since diagonal of Hermitian)
    // C[0,1] = (1+i)(0-i) + 2*(1+i) = (i-i²+0) + 2+2i = 1-i + 2+2i ... wait
    // Let me recalculate:
    // C[0,1] = (1+i)*conj(0+i) + (2+0i)*conj(1-i)
    //        = (1+i)*(-i) + 2*(1+i)
    //        = -i-i² + 2+2i = -i+1 + 2+2i = 3+i
    // C[1,1] = (0+i)*conj(0+i) + (1-i)*conj(1-i)
    //        = i*(-i) + (1-i)*(1+i)
    //        = 1 + 1+1 = 1 + 2 = 3 (real)
    blas.herk(f64, NK, NN, A, .{ .triangle = .n2, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 6), c_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), c_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), c_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].im, eps);
}

test "herk complex lower" {
    const NK = enum { n, k };
    const NN = enum { n, n2 };
    const T = Complex(f64);

    // A (2x1) = [[1+i], [2+0i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const A = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 1 }),
        .buf = &a_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(NN, T){
        .idx = NamedIndex(NN).initContiguous(.{ .n = 2, .n2 = 2 }),
        .buf = &c_buf,
    };

    // C = A * Aᴴ (lower triangle):
    // C[0,0] = (1+i)(1-i) = 2
    // C[1,0] = 2*(1-i) = 2-2i
    // C[1,1] = 2*2 = 4
    blas.herk(f64, NK, NN, A, .{ .triangle = .n, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 2), c_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 2), c_buf[2].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, -2), c_buf[2].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 4), c_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].im, eps);
}

test "syr2k real upper (triangle = second axis)" {
    const MK = enum { m, k };
    const NK = enum { n, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) = [[1, 2], [3, 4]]
    const a_buf = [_]T{ 1, 2, 3, 4 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x2) = [[5, 6], [7, 8]]
    const b_buf = [_]T{ 5, 6, 7, 8 };
    const B = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 2 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // C = A*Bᵀ + B*Aᵀ:
    // A*Bᵀ = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17, 23], [39, 53]]
    // B*Aᵀ = [[5*1+6*2, 5*3+6*4], [7*1+8*2, 7*3+8*4]] = [[17, 39], [23, 53]]
    // C = [[34, 62], [62, 106]]
    blas.syr2k(T, MK, NK, MN, A, B, .{ .triangle = .n, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 34), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 62), c_buf[1], 1e-10);
    // c_buf[2] is lower triangle
    try std.testing.expectApproxEqAbs(@as(T, 106), c_buf[3], 1e-10);
}

test "syr2k real nontrivial scalars" {
    const MK = enum { m, k };
    const NK = enum { n, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x1) = [[1], [2]]
    const a_buf = [_]T{ 1, 2 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 1 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[3], [4]]
    const b_buf = [_]T{ 3, 4 };
    const B = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 1 }),
        .buf = &b_buf,
    };

    // C (2x2) = [[1, 1], [1, 1]]
    var c_buf = [_]T{ 1, 1, 1, 1 };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // A*Bᵀ = [[3], [6]] * [[3, 4]] ... wait no
    // A is 2x1, Bᵀ is 1x2: A*Bᵀ = [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
    // B*Aᵀ = [[3*1, 3*2], [4*1, 4*2]] = [[3, 6], [4, 8]]
    // A*Bᵀ + B*Aᵀ = [[6, 10], [10, 16]]
    // C = 2*(sum) + 3*C = [[12+3, 20+3], [20+3, 32+3]] = [[15, 23], [23, 35]]
    blas.syr2k(T, MK, NK, MN, A, B, .{ .triangle = .n, .data = C }, .{ .alpha = 2.0, .beta = 3.0 });

    try std.testing.expectApproxEqAbs(@as(T, 15), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 23), c_buf[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 35), c_buf[3], 1e-10);
}

test "syr2k column-major" {
    const MK = enum { m, k };
    const NK = enum { n, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x1) col-major = [[1], [2]] stored as [1, 2]
    const a_buf = [_]T{ 1, 2 };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 1 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x1) col-major = [[3], [4]] stored as [3, 4]
    const b_buf = [_]T{ 3, 4 };
    const B = NamedArrayConst(NK, T){
        .idx = .{ .shape = .{ .n = 2, .k = 1 }, .strides = .{ .n = 1, .k = 2 } },
        .buf = &b_buf,
    };

    var c_buf = [_]T{ 0, 0, 0, 0 };
    const C = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &c_buf,
    };

    // A*Bᵀ + B*Aᵀ = [[6, 10], [10, 16]]
    // Use lower triangle (.m = first axis, ordinal 0) so C[0,0], C[1,0], C[1,1] are written.
    // Col-major: c_buf[0]=C[0,0]=6, c_buf[1]=C[1,0]=10, c_buf[3]=C[1,1]=16
    blas.syr2k(T, MK, NK, MN, A, B, .{ .triangle = .m, .data = C }, .{ .alpha = 1.0, .beta = 0.0 });

    try std.testing.expectApproxEqAbs(@as(T, 6), c_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 10), c_buf[1], 1e-10);
    // c_buf[2] = C[0,1] is upper triangle, not written
    try std.testing.expectApproxEqAbs(@as(T, 16), c_buf[3], 1e-10);
}

test "her2k complex upper" {
    const MK = enum { m, k };
    const NK = enum { n, k };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x1) = [[1+i], [2+0i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = 0 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 1 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[1+0i], [0+i]]
    const b_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const B = NamedArrayConst(NK, T){
        .idx = NamedIndex(NK).initContiguous(.{ .n = 2, .k = 1 }),
        .buf = &b_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &c_buf,
    };

    // C = alpha*A*Bᴴ + conj(alpha)*B*Aᴴ, alpha = 1+0i:
    // A*Bᴴ = [[(1+i)*1, (1+i)*(-i)], [2*1, 2*(-i)]]
    //       = [[1+i, -i+1], [2, -2i]]
    //       = [[1+i, 1-i], [2, -2i]]
    //
    // B*Aᴴ = [[1*(1-i), 1*2], [i*(1-i), i*2]]
    //       = [[1-i, 2], [1+i, 2i]]
    //
    // C = A*Bᴴ + B*Aᴴ = [[2, 3-i], [3+i, 0]]
    // Upper triangle (.n = second axis): C[0,0]=2, C[0,1]=3-i, C[1,1]=0
    // Row-major: c_buf[0]=C[0,0], c_buf[1]=C[0,1] (upper), c_buf[3]=C[1,1]
    blas.her2k(f64, MK, NK, MN, A, B, .{ .triangle = .n, .data = C }, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = 0.0,
    });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 2), c_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), c_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, -1), c_buf[1].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].im, eps);
}

test "her2k column-major" {
    const MK = enum { m, k };
    const NK = enum { n, k };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x1) col-major
    const a_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 1 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x1) col-major
    const b_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 1, .im = 0 },
    };
    const B = NamedArrayConst(NK, T){
        .idx = .{ .shape = .{ .n = 2, .k = 1 }, .strides = .{ .n = 1, .k = 2 } },
        .buf = &b_buf,
    };

    var c_buf = [_]T{
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 },
    };
    const C = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &c_buf,
    };

    // A = [[1], [i]], B = [[1], [1]]
    // A*Bᴴ = [[1, 1], [i, i]]
    // B*Aᴴ = [[1, -i], [1, -i]]
    // C = A*Bᴴ + B*Aᴴ = [[2, 1-i], [1+i, 0]]
    // Use lower triangle (.m = first axis) so C[0,0], C[1,0], C[1,1] are written.
    // Col-major: c_buf[0]=C[0,0]=2, c_buf[1]=C[1,0]=1+i, c_buf[3]=C[1,1]=0
    blas.her2k(f64, MK, NK, MN, A, B, .{ .triangle = .m, .data = C }, .{
        .alpha = .{ .re = 1, .im = 0 },
        .beta = 0.0,
    });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 2), c_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c_buf[1].im, eps);
    // c_buf[2] = C[0,1] is upper triangle, not written
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c_buf[3].im, eps);
}

test "trmm real left (triangle = second axis)" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (3x3) upper triangular: [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
    const a_buf = [_]T{ 1, 2, 3, 0, 4, 5, 0, 0, 6 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 3, .k = 3 }),
        .buf = &a_buf,
    };

    // B (3x2) = [[1, 0], [0, 1], [1, 1]]
    var b_buf = [_]T{ 1, 0, 0, 1, 1, 1 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 3, .n = 2 }),
        .buf = &b_buf,
    };

    // B := A * B (left side):
    // row0: 1*1+2*0+3*1=4, 1*0+2*1+3*1=5
    // row1: 0+4*0+5*1=5, 0+4*1+5*1=9
    // row2: 0+0+6*1=6, 0+0+6*1=6
    blas.trmm(T, MK, MN, .{ .triangle = .k, .data = A }, .non_unit, B, .{ .alpha = 1.0 });

    const expected = [_]T{ 4, 5, 5, 9, 6, 6 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], b_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trmm real right" {
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) upper triangular: [[1, 2], [0, 3]], triangle = second axis = n
    const a_buf = [_]T{ 1, 2, 0, 3 };
    const A = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 2 }),
        .buf = &a_buf,
    };

    // B (3x2) = [[1, 0], [0, 1], [2, 3]]
    var b_buf = [_]T{ 1, 0, 0, 1, 2, 3 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 3, .n = 2 }),
        .buf = &b_buf,
    };

    // B := B * A (right side, A shares 'n' with B's second axis):
    // col0 of result: B * A col0 = B * [1, 0]ᵀ = B[:,0] = [1, 0, 2]
    // col1 of result: B * A col1 = B * [2, 3]ᵀ = [1*2+0*3, 0*2+1*3, 2*2+3*3] = [2, 3, 13]
    // Result: [[1, 2], [0, 3], [2, 13]]
    blas.trmm(T, KN, MN, .{ .triangle = .n, .data = A }, .non_unit, B, .{ .alpha = 1.0 });

    const expected = [_]T{ 1, 2, 0, 3, 2, 13 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], b_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trmm real unit diagonal" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) upper triangular, unit diagonal: [[1, 3], [0, 1]] (diag assumed 1)
    const a_buf = [_]T{ 99, 3, 0, 99 }; // diagonal values should be ignored
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x2) = [[1, 2], [3, 4]]
    var b_buf = [_]T{ 1, 2, 3, 4 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &b_buf,
    };

    // B := A * B with unit diagonal → A = [[1, 3], [0, 1]]
    // row0: 1*1+3*3=10, 1*2+3*4=14
    // row1: 0+1*3=3, 0+1*4=4
    blas.trmm(T, MK, MN, .{ .triangle = .k, .data = A }, .unit, B, .{ .alpha = 1.0 });

    const expected = [_]T{ 10, 14, 3, 4 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], b_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trmm complex" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x2) upper triangular: [[1+i, 2], [0, 1-i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 1, .im = -1 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[1+0i], [0+i]]
    var b_buf = [_]T{
        .{ .re = 1, .im = 0 },
        .{ .re = 0, .im = 1 },
    };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &b_buf,
    };

    // B := A * B:
    // row0: (1+i)*1 + 2*(i) = 1+i+2i = 1+3i
    // row1: 0 + (1-i)*(i) = i-i² = 1+i
    blas.trmm(T, MK, MN, .{ .triangle = .k, .data = A }, .non_unit, B, .{
        .alpha = .{ .re = 1, .im = 0 },
    });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 1), b_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 3), b_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), b_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), b_buf[1].im, eps);
}

test "trmm column-major" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) lower triangular col-major: [[2, 0], [3, 4]]
    // Col-major storage: col0=[2, 3], col1=[0, 4] → [2, 3, 0, 4]
    const a_buf = [_]T{ 2, 3, 0, 4 };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 2 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x2) col-major: [[1, 5], [2, 6]] stored as [1, 2, 5, 6]
    var b_buf = [_]T{ 1, 2, 5, 6 };
    const B = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &b_buf,
    };

    // B := A * B, A = [[2, 0], [3, 4]], B = [[1, 5], [2, 6]]
    // row0: 2*1+0*2=2, 2*5+0*6=10
    // row1: 3*1+4*2=11, 3*5+4*6=39
    // Result: [[2, 10], [11, 39]]
    // Col-major: [2, 11, 10, 39]
    blas.trmm(T, MK, MN, .{ .triangle = .m, .data = A }, .non_unit, B, .{ .alpha = 1.0 });

    const expected = [_]T{ 2, 11, 10, 39 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], b_buf[i], math.floatEpsAt(T, expected[i]));
    }
}

test "trsm real left (triangle = second axis)" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) upper triangular: [[2, 1], [0, 3]]
    const a_buf = [_]T{ 2, 1, 0, 3 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[5], [9]]
    var b_buf = [_]T{ 5, 9 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &b_buf,
    };

    // Solve A * X = B:
    // [[2, 1], [0, 3]] * X = [[5], [9]]
    // x1 = 9/3 = 3, x0 = (5 - 1*3)/2 = 1
    // X = [[1], [3]]
    blas.trsm(T, MK, MN, .{ .triangle = .k, .data = A }, .non_unit, B, .{ .alpha = 1.0 });

    try std.testing.expectApproxEqAbs(@as(T, 1), b_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 3), b_buf[1], 1e-10);
}

test "trsm real right" {
    const KN = enum { k, n };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) upper triangular: [[2, 1], [0, 3]]
    const a_buf = [_]T{ 2, 1, 0, 3 };
    const A = NamedArrayConst(KN, T){
        .idx = NamedIndex(KN).initContiguous(.{ .k = 2, .n = 2 }),
        .buf = &a_buf,
    };

    // B (2x2) = [[4, 3], [8, 6]]
    var b_buf = [_]T{ 4, 3, 8, 6 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 2 }),
        .buf = &b_buf,
    };

    // Solve X * A = B (right side, A shares 'n' with B):
    // X * [[2, 1], [0, 3]] = [[4, 3], [8, 6]]
    // For row0 of X: [x0, x1] * [[2, 1], [0, 3]] = [4, 3]
    //   2*x0 = 4 → x0 = 2
    //   x0 + 3*x1 = 3 → 2 + 3*x1 = 3 → x1 = 1/3
    // For row1: [x0, x1] * [[2, 1], [0, 3]] = [8, 6]
    //   2*x0 = 8 → x0 = 4
    //   x0 + 3*x1 = 6 → 4 + 3*x1 = 6 → x1 = 2/3
    blas.trsm(T, KN, MN, .{ .triangle = .n, .data = A }, .non_unit, B, .{ .alpha = 1.0 });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(T, 2), b_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(T, 1.0 / 3.0), b_buf[1], eps);
    try std.testing.expectApproxEqAbs(@as(T, 4), b_buf[2], eps);
    try std.testing.expectApproxEqAbs(@as(T, 2.0 / 3.0), b_buf[3], eps);
}

test "trsm real unit diagonal" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) upper triangular, unit diagonal: [[1, 4], [0, 1]]
    const a_buf = [_]T{ 99, 4, 0, 99 }; // diagonal ignored
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[7], [2]]
    var b_buf = [_]T{ 7, 2 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &b_buf,
    };

    // Solve [[1, 4], [0, 1]] * X = [[7], [2]]
    // x1 = 2, x0 = 7 - 4*2 = -1
    blas.trsm(T, MK, MN, .{ .triangle = .k, .data = A }, .unit, B, .{ .alpha = 1.0 });

    try std.testing.expectApproxEqAbs(@as(T, -1), b_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 2), b_buf[1], 1e-10);
}

test "trsm complex" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = Complex(f64);

    // A (2x2) upper triangular: [[1+i, 2], [0, 1-i]]
    const a_buf = [_]T{
        .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 },
        .{ .re = 0, .im = 0 }, .{ .re = 1, .im = -1 },
    };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // Compute A * known_X, then solve to recover X.
    // X = [[1+0i], [0+i]]
    // A*X = [[(1+i)*1+2*(i)], [0+(1-i)*(i)]]
    //     = [[1+i+2i], [i-i²]] = [[1+3i], [1+i]]
    var b_buf = [_]T{
        .{ .re = 1, .im = 3 },
        .{ .re = 1, .im = 1 },
    };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &b_buf,
    };

    blas.trsm(T, MK, MN, .{ .triangle = .k, .data = A }, .non_unit, B, .{
        .alpha = .{ .re = 1, .im = 0 },
    });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(f64, 1), b_buf[0].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), b_buf[0].im, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 0), b_buf[1].re, eps);
    try std.testing.expectApproxEqAbs(@as(f64, 1), b_buf[1].im, eps);
}

test "trsm column-major" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) lower triangular col-major: [[3, 0], [1, 2]]
    // Col-major: [3, 1, 0, 2]
    const a_buf = [_]T{ 3, 1, 0, 2 };
    const A = NamedArrayConst(MK, T){
        .idx = .{ .shape = .{ .m = 2, .k = 2 }, .strides = .{ .m = 1, .k = 2 } },
        .buf = &a_buf,
    };

    // B (2x2) col-major: [[6, 3], [8, 6]] stored as [6, 8, 3, 6]
    var b_buf = [_]T{ 6, 8, 3, 6 };
    const B = NamedArray(MN, T){
        .idx = .{ .shape = .{ .m = 2, .n = 2 }, .strides = .{ .m = 1, .n = 2 } },
        .buf = &b_buf,
    };

    // Solve A * X = B, A = [[3, 0], [1, 2]]
    // For col0: [[3,0],[1,2]] * x = [6, 8]
    //   3*x0 = 6 → x0 = 2
    //   x0 + 2*x1 = 8 → 2 + 2*x1 = 8 → x1 = 3
    // For col1: [[3,0],[1,2]] * x = [3, 6]
    //   3*x0 = 3 → x0 = 1
    //   x0 + 2*x1 = 6 → 1 + 2*x1 = 6 → x1 = 5/2
    // X = [[2, 1], [3, 5/2]], col-major: [2, 3, 1, 2.5]
    blas.trsm(T, MK, MN, .{ .triangle = .m, .data = A }, .non_unit, B, .{ .alpha = 1.0 });

    const eps = 1e-10;
    try std.testing.expectApproxEqAbs(@as(T, 2), b_buf[0], eps);
    try std.testing.expectApproxEqAbs(@as(T, 3), b_buf[1], eps);
    try std.testing.expectApproxEqAbs(@as(T, 1), b_buf[2], eps);
    try std.testing.expectApproxEqAbs(@as(T, 2.5), b_buf[3], eps);
}

test "trmm with alpha" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) lower triangular: [[2, 0], [1, 3]]
    const a_buf = [_]T{ 2, 0, 1, 3 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[1], [2]]
    var b_buf = [_]T{ 1, 2 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &b_buf,
    };

    // B := alpha * A * B, alpha = 3
    // A * B = [[2*1+0], [1*1+3*2]] = [[2], [7]]
    // 3 * [[2], [7]] = [[6], [21]]
    blas.trmm(T, MK, MN, .{ .triangle = .m, .data = A }, .non_unit, B, .{ .alpha = 3.0 });

    try std.testing.expectApproxEqAbs(@as(T, 6), b_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 21), b_buf[1], 1e-10);
}

test "trsm with alpha" {
    const MK = enum { m, k };
    const MN = enum { m, n };
    const T = f64;

    // A (2x2) lower triangular: [[2, 0], [0, 4]]
    const a_buf = [_]T{ 2, 0, 0, 4 };
    const A = NamedArrayConst(MK, T){
        .idx = NamedIndex(MK).initContiguous(.{ .m = 2, .k = 2 }),
        .buf = &a_buf,
    };

    // B (2x1) = [[6], [12]]
    var b_buf = [_]T{ 6, 12 };
    const B = NamedArray(MN, T){
        .idx = NamedIndex(MN).initContiguous(.{ .m = 2, .n = 1 }),
        .buf = &b_buf,
    };

    // Solve A * X = alpha * B, alpha = 2
    // A * X = 2*B = [[12], [24]]
    // X = A⁻¹ * [[12], [24]] = [[12/2], [24/4]] = [[6], [6]]
    blas.trsm(T, MK, MN, .{ .triangle = .m, .data = A }, .non_unit, B, .{ .alpha = 2.0 });

    try std.testing.expectApproxEqAbs(@as(T, 6), b_buf[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(T, 6), b_buf[1], 1e-10);
}
