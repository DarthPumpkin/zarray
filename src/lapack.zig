//! Idiomatic Zig bindings for the LAPACK "workhorse" routines exposed by
//! Accelerate's *new* LAPACK interface (`ACCELERATE_NEW_LAPACK`, the standard
//! reference Fortran ABI — the same signatures reference LAPACK / OpenBLAS use,
//! so this binding is portable off macOS later).
//!
//! Everything operates on `NamedArray` views. The design mirrors the decisions
//! we settled on for the BLAS binding, with a few LAPACK-specific twists:
//!
//!   - **Storage order is absorbed for free where the routine allows.** LAPACK
//!     is column-major, but a row-major matrix is bit-identical to the
//!     column-major transpose, so — exactly like `blas.gemm` uses transpose
//!     flags — the square LU family (`solve`/`lu`/`luSolve`/`detInplace`) reads
//!     A's actual layout and sets the `getrs` `trans` flag accordingly. No copy.
//!     `inv` and the value-only spectral routines (`eig`/`eigSym`/`svd`) are
//!     layout-transparent because inversion and eigen/singular *values* are
//!     invariant under the transpose. `cholesky`/`choleskySolve`/`eigSym`
//!     absorb row-major storage of a symmetric matrix by flipping `uplo`.
//!     `qr`/`lstsq` accept any positive-stride layout: their `*Inplace` forms
//!     absorb it cheaply — `lstsqInplace` reinterprets a row-major real `a` as
//!     its transpose via the `gels` `trans` flag (copying only for complex `a`),
//!     and `qrInplace` packs a column-major copy only when `a` isn't already
//!     column-major. The input-preserving `qr`/`lstsq` copy `a` first regardless.
//!
//!   - **Error taxonomy follows "recoverable → error, unrecoverable → panic".**
//!     Layout incompatibility (a doubly-strided view, or a multi-column
//!     row-major right-hand side) is a *recoverable* condition — the caller can
//!     pack into a contiguous/column-major buffer and retry — so it is returned
//!     as an error, never a panic. Numerical outcomes (singularity, non-SPD,
//!     non-convergence) are likewise errors. Genuine programmer bugs
//!     (non-square A, mismatched extents, too-short scratch) `@panic`, and
//!     structural mistakes (wrong axis count, no shared axis) are
//!     `@compileError`.
//!
//!   - **No hidden allocation in the strict routines.** `solve`/`lu`/`luSolve`/
//!     `detInplace`/`cholesky`/`choleskySolve` never allocate; the caller
//!     supplies any scratch (e.g. the `ipiv` pivot array). Routines that
//!     fundamentally need LAPACK workspace whose size is a runtime query (`inv`,
//!     `lstsq`, `qr`, `eig`, `eigSym`, `svd`, and their `*Inplace` forms) take
//!     an `allocator`.
//!
//!   - **Input-preserving default, `*Inplace` opt-out.** The decomposition and
//!     determinant routines come in pairs. The plain name (`det`/`lstsq`/`qr`/
//!     `eig`/`eigSym`/`svd`/`eigVectors`/`svdVectors`) is the default: it leaves
//!     the caller's `a` untouched by factoring a private contiguous copy. The
//!     `*Inplace` variant (`detInplace`/`lstsqInplace`/…) reuses `a` directly as
//!     scratch (overwriting it) to skip that copy. Both forms take an
//!     `allocator` for LAPACK workspace, so the allocator is *not* what tells
//!     them apart — the name is. (`eigSymVectors` is inherently copy-based and
//!     has no in-place form.)
//!
//!   - **Vector-returning spectral variants.** `eigSymVectors`/`eigVectors`/
//!     `svdVectors` also return eigen/singular *vectors*. Vectors are *not*
//!     transpose-invariant, so each first gets A into LAPACK's column-major
//!     orientation with the cheapest correct trick: `eigSymVectors` makes a
//!     column-major copy (symmetric ⇒ `uplo` picks the triangle) and leaves A
//!     intact; `eigVectorsInplace` transposes A in place (packing a copy only
//!     for a padded row-major view); `svdVectorsInplace` relabels the U↔V output
//!     swap of the transposed problem. `eigVectorsInplace`/`svdVectorsInplace`
//!     use A as scratch (overwritten); passing a column-major A avoids the
//!     internal packing. The default `eigVectors`/`svdVectors` copy A first, so
//!     it is preserved regardless of layout. Returned vector matrices carry
//!     synthesized axis names (like `qr`).
//!
//!   - **Generalized & rank-revealing routines.** `eigSymGen`/`eigSymGenVectors`
//!     solve the symmetric/Hermitian-definite generalized eigenproblem
//!     `A·x = λ·B·x` (`sygv`/`hegv`; `GenEigProblem` picks the `itype` form), with
//!     B required positive definite (else `error.NotPositiveDefinite`).
//!     `lstsqSvd` solves possibly rank-deficient least squares via
//!     divide-and-conquer SVD (`gelsd`), returning the minimum-norm solution, the
//!     effective numerical `rank`, and the singular values, thresholded by an
//!     `rcond` cutoff (negative ⇒ machine precision). Both follow the
//!     input-preserving default / `*Inplace` convention.
//!
//! Supported scalars: `f32`/`f64` and `Complex(f32)`/`Complex(f64)` across the
//! whole surface. Complex decompositions use the Hermitian/unitary analogs
//! (`heev` for `eigSym`, `ungqr` for `qr`) and route through a header-verified C
//! shim (`src/lapack_shim.{c,h}`) that checks every complex prototype against
//! Apple's `<vecLib/lapack.h>`. Value-typed results generalize to `RealOf(T)`
//! (real eigenvalues/singular values for a complex `A`); for real `T`,
//! `RealOf(T) == T`, so real callers are unaffected.
//!
//! The raw C API is reachable via `c` for anything not wrapped here.

const std = @import("std");
const math = std.math;
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Complex = math.complex.Complex;

const named_array = @import("named_array.zig");
const axis_meta = @import("axis_meta.zig");
const NamedArray = named_array.NamedArray;
const NamedArrayConst = named_array.NamedArrayConst;
const NamedIndex = @import("named_index.zig").NamedIndex;
const KeyEnum = axis_meta.KeyEnum;

pub const c = @cImport({
    @cDefine("ACCELERATE_NEW_LAPACK", "1");
    // Standalone LAPACK header (not the Accelerate umbrella, which pulls in
    // vImage that Zig's translate-c cannot resolve). `__LAPACK_int` is `c_int`
    // in the default (LP64) build.
    @cInclude("vecLib/lapack.h");
});

// translate-c cannot model C `double _Complex` / `float _Complex`, so the
// complex LAPACK entry points come back from the cImport as compile errors. All
// complex symbols are therefore reached through the header-verified C shim
// (`src/lapack_shim.{c,h}`): the shim's `zarray_*` forwarders speak only
// primitive pointer types, so this Zig<->C boundary carries no complex
// ambiguity, and every complex prototype is checked against `<vecLib/lapack.h>`
// when the shim TU is compiled. A complex array is passed as `[*]f32`/`[*]f64`
// (interleaved re/im), obtained by `@ptrCast` from `[*]Complex(T)`.

// LU family.
extern fn zarray_cgetrf(m: *const c_int, n: *const c_int, a: [*]f32, lda: *const c_int, ipiv: [*]c_int, info: *c_int) void;
extern fn zarray_zgetrf(m: *const c_int, n: *const c_int, a: [*]f64, lda: *const c_int, ipiv: [*]c_int, info: *c_int) void;
extern fn zarray_cgetrs(trans: *const u8, n: *const c_int, nrhs: *const c_int, a: [*]f32, lda: *const c_int, ipiv: [*]const c_int, b: [*]f32, ldb: *const c_int, info: *c_int) void;
extern fn zarray_zgetrs(trans: *const u8, n: *const c_int, nrhs: *const c_int, a: [*]f64, lda: *const c_int, ipiv: [*]const c_int, b: [*]f64, ldb: *const c_int, info: *c_int) void;
extern fn zarray_cgetri(n: *const c_int, a: [*]f32, lda: *const c_int, ipiv: [*]const c_int, work: [*]f32, lwork: *const c_int, info: *c_int) void;
extern fn zarray_zgetri(n: *const c_int, a: [*]f64, lda: *const c_int, ipiv: [*]const c_int, work: [*]f64, lwork: *const c_int, info: *c_int) void;
// Cholesky family.
extern fn zarray_cpotrf(uplo: *const u8, n: *const c_int, a: [*]f32, lda: *const c_int, info: *c_int) void;
extern fn zarray_zpotrf(uplo: *const u8, n: *const c_int, a: [*]f64, lda: *const c_int, info: *c_int) void;
extern fn zarray_cpotrs(uplo: *const u8, n: *const c_int, nrhs: *const c_int, a: [*]f32, lda: *const c_int, b: [*]f32, ldb: *const c_int, info: *c_int) void;
extern fn zarray_zpotrs(uplo: *const u8, n: *const c_int, nrhs: *const c_int, a: [*]f64, lda: *const c_int, b: [*]f64, ldb: *const c_int, info: *c_int) void;
// Hermitian eigenvalues (heev): w/rwork stay real.
extern fn zarray_cheev(jobz: *const u8, uplo: *const u8, n: *const c_int, a: [*]f32, lda: *const c_int, w: [*]f32, work: [*]f32, lwork: *const c_int, rwork: [*]f32, info: *c_int) void;
extern fn zarray_zheev(jobz: *const u8, uplo: *const u8, n: *const c_int, a: [*]f64, lda: *const c_int, w: [*]f64, work: [*]f64, lwork: *const c_int, rwork: [*]f64, info: *c_int) void;
// General eigenvalues (geev): single complex w, real rwork.
extern fn zarray_cgeev(jobvl: *const u8, jobvr: *const u8, n: *const c_int, a: [*]f32, lda: *const c_int, w: [*]f32, vl: [*]f32, ldvl: *const c_int, vr: [*]f32, ldvr: *const c_int, work: [*]f32, lwork: *const c_int, rwork: [*]f32, info: *c_int) void;
extern fn zarray_zgeev(jobvl: *const u8, jobvr: *const u8, n: *const c_int, a: [*]f64, lda: *const c_int, w: [*]f64, vl: [*]f64, ldvl: *const c_int, vr: [*]f64, ldvr: *const c_int, work: [*]f64, lwork: *const c_int, rwork: [*]f64, info: *c_int) void;
// SVD (gesdd): s/rwork stay real, u/vt complex.
extern fn zarray_cgesdd(jobz: *const u8, m: *const c_int, n: *const c_int, a: [*]f32, lda: *const c_int, s: [*]f32, u: [*]f32, ldu: *const c_int, vt: [*]f32, ldvt: *const c_int, work: [*]f32, lwork: *const c_int, rwork: [*]f32, iwork: [*]c_int, info: *c_int) void;
extern fn zarray_zgesdd(jobz: *const u8, m: *const c_int, n: *const c_int, a: [*]f64, lda: *const c_int, s: [*]f64, u: [*]f64, ldu: *const c_int, vt: [*]f64, ldvt: *const c_int, work: [*]f64, lwork: *const c_int, rwork: [*]f64, iwork: [*]c_int, info: *c_int) void;
// Least squares (gels): all complex.
extern fn zarray_cgels(trans: *const u8, m: *const c_int, n: *const c_int, nrhs: *const c_int, a: [*]f32, lda: *const c_int, b: [*]f32, ldb: *const c_int, work: [*]f32, lwork: *const c_int, info: *c_int) void;
extern fn zarray_zgels(trans: *const u8, m: *const c_int, n: *const c_int, nrhs: *const c_int, a: [*]f64, lda: *const c_int, b: [*]f64, ldb: *const c_int, work: [*]f64, lwork: *const c_int, info: *c_int) void;
// QR factor (geqrf) + unitary Q assembly (ungqr).
extern fn zarray_cgeqrf(m: *const c_int, n: *const c_int, a: [*]f32, lda: *const c_int, tau: [*]f32, work: [*]f32, lwork: *const c_int, info: *c_int) void;
extern fn zarray_zgeqrf(m: *const c_int, n: *const c_int, a: [*]f64, lda: *const c_int, tau: [*]f64, work: [*]f64, lwork: *const c_int, info: *c_int) void;
extern fn zarray_cungqr(m: *const c_int, n: *const c_int, k: *const c_int, a: [*]f32, lda: *const c_int, tau: [*]f32, work: [*]f32, lwork: *const c_int, info: *c_int) void;
extern fn zarray_zungqr(m: *const c_int, n: *const c_int, k: *const c_int, a: [*]f64, lda: *const c_int, tau: [*]f64, work: [*]f64, lwork: *const c_int, info: *c_int) void;
// Generalized Hermitian-definite eigenproblem (hegv): w/rwork stay real.
extern fn zarray_chegv(itype: *const c_int, jobz: *const u8, uplo: *const u8, n: *const c_int, a: [*]f32, lda: *const c_int, b: [*]f32, ldb: *const c_int, w: [*]f32, work: [*]f32, lwork: *const c_int, rwork: [*]f32, info: *c_int) void;
extern fn zarray_zhegv(itype: *const c_int, jobz: *const u8, uplo: *const u8, n: *const c_int, a: [*]f64, lda: *const c_int, b: [*]f64, ldb: *const c_int, w: [*]f64, work: [*]f64, lwork: *const c_int, rwork: [*]f64, info: *c_int) void;
// Rank-deficient least squares via SVD (gelsd): s/rcond/rwork stay real.
extern fn zarray_cgelsd(m: *const c_int, n: *const c_int, nrhs: *const c_int, a: [*]f32, lda: *const c_int, b: [*]f32, ldb: *const c_int, s: [*]f32, rcond: *const f32, rank: *c_int, work: [*]f32, lwork: *const c_int, rwork: [*]f32, iwork: [*]c_int, info: *c_int) void;
extern fn zarray_zgelsd(m: *const c_int, n: *const c_int, nrhs: *const c_int, a: [*]f64, lda: *const c_int, b: [*]f64, ldb: *const c_int, s: [*]f64, rcond: *const f64, rank: *c_int, work: [*]f64, lwork: *const c_int, rwork: [*]f64, iwork: [*]c_int, info: *c_int) void;

// ===== Public error set / small enums ========================================

pub const LapackError = error{
    /// The view has no unit-stride axis (doubly strided) or uses a negative
    /// stride; LAPACK needs a single leading dimension over positive strides.
    /// Copy to a contiguous buffer and retry.
    NotContiguous,
    /// The routine requires a column-major matrix and got a different layout.
    /// Copy to column-major and retry.
    NotColumnMajor,
    /// A multi-column right-hand side is row-major; `getrs`/`potrs` have no
    /// transpose knob for B, so it must be column-major. Copy B and retry.
    RhsNotColumnMajor,
    /// The factored matrix is exactly singular (a zero pivot): no solution.
    Singular,
    /// The matrix is not (numerically) positive definite; Cholesky failed.
    NotPositiveDefinite,
    /// An iterative kernel (eigen/SVD) failed to converge.
    ConvergenceFailure,
};

/// Which triangle of a symmetric/Hermitian matrix is referenced.
pub const Triangle = enum {
    upper,
    lower,
    fn flip(self: Triangle) Triangle {
        return switch (self) {
            .upper => .lower,
            .lower => .upper,
        };
    }
    fn uploChar(self: Triangle) u8 {
        return switch (self) {
            .upper => 'U',
            .lower => 'L',
        };
    }
};

const Layout = enum { col_major, row_major };

// ===== Scalar helpers ========================================================

fn RealOf(comptime T: type) type {
    return switch (T) {
        f32, Complex(f32) => f32,
        f64, Complex(f64) => f64,
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    };
}

fn isComplex(comptime T: type) bool {
    return T == Complex(f32) or T == Complex(f64);
}

// ===== Q5: scratch-sizing helpers ============================================
// Pure, allocation-free functions exposing LAPACK-specific scratch sizes an
// allocator cannot infer on its own:
//
//   * `pivotLen` sizes the caller-supplied `ipiv` for the non-allocating LU
//     family (`solve`/`lu`/`luSolve`/`detInplace`).
//   * `eigSymRworkLen`/`eigRworkLen`/`svdRworkLen`/`svdIworkLen` are the complex
//     `rwork`/`iwork` lengths LAPACK's workspace query does *not* return, so the
//     internal routines call these same helpers — the numbers live in one place
//     and are exercised transitively by the complex decomposition tests.
//
// The `lwork` scratch is NOT exposed here: every allocator-taking routine issues
// the `lwork = -1` query and allocates the optimal size itself, so a caller has
// nothing to pre-size. (If a BYO-buffer `*Into` API is ever added — Q5 option C
// — the matching documented-minimum helpers can come back with it.)

/// Length of the pivot array (`ipiv`) for the LU family (`solve`/`lu`/`det`/
/// `luSolve`/`inv`): one entry per row.
pub fn pivotLen(n: usize) usize {
    return @max(n, 1);
}

/// Length of `gesdd`'s integer workspace (`iwork`) for an m×n matrix.
pub fn svdIworkLen(m: usize, n: usize) usize {
    return @max(8 * @min(m, n), 1);
}

/// Length of the real workspace (`rwork`) for the Hermitian eigensolver
/// (`cheev`/`zheev`): `max(1, 3n−2)`. Zero for real `syev` (no `rwork`).
pub fn eigSymRworkLen(n: usize) usize {
    return if (n >= 1) @max(3 * n - 2, 1) else 1;
}

/// Length of the real workspace (`rwork`) for the general complex eigensolver
/// (`cgeev`/`zgeev`): `2n`. Zero for real `geev` (uses `wr`/`wi` instead).
pub fn eigRworkLen(n: usize) usize {
    return @max(2 * n, 1);
}

/// Length of the real workspace (`rwork`) for the complex SVD (`cgesdd`/
/// `zgesdd`). `want_vectors` selects the (larger) `jobz∈{'S','A'}` bound versus
/// the `jobz='N'` (values-only) bound. Symmetric in m/n, so layout/transpose
/// does not change it. Returns a safe documented upper bound; real `gesdd` has
/// no `rwork`.
pub fn svdRworkLen(m: usize, n: usize, want_vectors: bool) usize {
    const mn = @min(m, n);
    const mx = @max(m, n);
    if (!want_vectors) return @max(7 * mn, 1); // jobz = 'N'
    // jobz = 'S'/'A': max over LAPACK's documented cases.
    const a = 5 * mn * mn + 5 * mn;
    const b = 2 * mx * mn + 2 * mn * mn + mn;
    return @max(@max(a, b), 1);
}

// ===== Matrix descriptor (the "adapter") =====================================

/// A described matrix parameterized by its element-pointer type. `Ptr` is
/// `[*]T` for a mutable input (`describe`) and `[*]const T` for a read-only one
/// (`describeConst`). `const`-ness thus rides the descriptor all the way to the
/// FFI boundary, where LAPACK's non-`const` C ABI forces a `@constCast`.
fn DescriptorOf(comptime Ptr: type) type {
    return struct {
        layout: Layout,
        m: c_int, // logical rows (extent of rows_name)
        n: c_int, // logical cols (extent of cols_name)
        lda: c_int, // leading dimension
        ptr: Ptr, // element (0,0), accounting for the view's offset
    };
}

fn Descriptor(comptime T: type) type {
    return DescriptorOf([*]T);
}

fn DescriptorConst(comptime T: type) type {
    return DescriptorOf([*]const T);
}

const DescGeom = struct { layout: Layout, m: c_int, n: c_int, lda: c_int };

/// Compute a matrix's LAPACK geometry (layout, dimensions, leading dimension)
/// from its index alone — the part of describing that doesn't touch the buffer
/// and so is independent of the element pointer's `const`-ness.
fn describeGeom(
    idx: anytype,
    comptime rows_name: [:0]const u8,
    comptime cols_name: [:0]const u8,
) LapackError!DescGeom {
    const nr = @field(idx.shape, rows_name);
    const nc = @field(idx.shape, cols_name);
    if (nr == 0 or nc == 0) return error.NotContiguous;
    const sr = @field(idx.strides, rows_name);
    const sc = @field(idx.strides, cols_name);
    const nr_i: isize = @intCast(nr);
    const nc_i: isize = @intCast(nc);

    // Column-major: rows contiguous. (A single column never steps by lda, so
    // clamp lda up to the row count to satisfy LAPACK's lda >= max(1,m).)
    if (sr == 1 and (nc == 1 or sc >= nr_i)) {
        const lda: c_int = if (nc == 1) @intCast(@max(nr, 1)) else @intCast(sc);
        return .{ .layout = .col_major, .m = @intCast(nr), .n = @intCast(nc), .lda = lda };
    }
    // Row-major: cols contiguous. LAPACK, reading this column-major, sees Aᵀ.
    if (sc == 1 and (nr == 1 or sr >= nc_i)) {
        const lda: c_int = if (nr == 1) @intCast(@max(nc, 1)) else @intCast(sr);
        return .{ .layout = .row_major, .m = @intCast(nr), .n = @intCast(nc), .lda = lda };
    }
    return error.NotContiguous;
}

/// Describe a 2-axis `NamedArray` as a mutable LAPACK matrix. `rows_name` is the
/// Fortran first index; `cols_name` the second. Returns `error.NotContiguous`
/// for any view that can't be expressed with a single positive leading dimension.
fn describe(
    comptime T: type,
    comptime Axis: type,
    arr: NamedArray(Axis, T),
    comptime rows_name: [:0]const u8,
    comptime cols_name: [:0]const u8,
) LapackError!Descriptor(T) {
    const g = try describeGeom(arr.idx, rows_name, cols_name);
    const Axes = @TypeOf(arr.idx.shape);
    const base: [*]T = @ptrCast(arr.at(std.mem.zeroes(Axes)));
    return .{ .layout = g.layout, .m = g.m, .n = g.n, .lda = g.lda, .ptr = base };
}

/// Read-only counterpart of `describe`: describes a `NamedArrayConst`, yielding
/// a descriptor whose `ptr` is `[*]const T`. Routines that only *read* the
/// matrix (via `readElem`) use it directly; routines that pass it to LAPACK
/// `@constCast` the pointer at the call site, where the non-`const` C ABI
/// demands it.
fn describeConst(
    comptime T: type,
    comptime Axis: type,
    arr: NamedArrayConst(Axis, T),
    comptime rows_name: [:0]const u8,
    comptime cols_name: [:0]const u8,
) LapackError!DescriptorConst(T) {
    const g = try describeGeom(arr.idx, rows_name, cols_name);
    const Axes = @TypeOf(arr.idx.shape);
    const base: [*]const T = @ptrCast(arr.at(std.mem.zeroes(Axes)));
    return .{ .layout = g.layout, .m = g.m, .n = g.n, .lda = g.lda, .ptr = base };
}

// ===== Comptime axis helpers =================================================

fn assertTwoAxes(comptime Axis: type) void {
    if (meta.fields(Axis).len != 2)
        @compileError("lapack: expected a 2-axis (matrix) enum, got " ++ @typeName(Axis));
}

/// The single axis name shared by two 2-axis enums. `@compileError` unless
/// exactly one name is shared.
fn sharedAxisName(comptime AxisA: type, comptime AxisB: type) [:0]const u8 {
    const a = meta.fieldNames(AxisA);
    const b = meta.fieldNames(AxisB);
    var found: ?[:0]const u8 = null;
    for (a) |an| {
        for (b) |bn| {
            if (std.mem.eql(u8, an, bn)) {
                if (found != null)
                    @compileError("lapack: A and B share more than one axis name");
                found = an;
            }
        }
    }
    return found orelse @compileError("lapack: A and B share no axis name (need one common 'row' axis)");
}

/// The other of a 2-axis enum's names.
fn otherAxis(comptime Axis: type, comptime name: [:0]const u8) [:0]const u8 {
    const names = meta.fieldNames(Axis);
    if (std.mem.eql(u8, names[0], name)) return names[1];
    return names[0];
}

/// Synthesized 2-axis enum {cols-of-A, rhs-of-B} naming the solution X of AX=B.
fn SolutionAxis(comptime MatAxis: type, comptime RhsAxis: type) type {
    const row = sharedAxisName(MatAxis, RhsAxis);
    const col = otherAxis(MatAxis, row);
    const rhs = otherAxis(RhsAxis, row);
    return KeyEnum(&.{ col, rhs });
}

/// The solution X of A·X = B. This is a *view aliasing the RHS buffer* `b`, not
/// a new allocation — there is nothing to free, and writes through it mutate `b`.
pub fn Solution(comptime MatAxis: type, comptime RhsAxis: type, comptime T: type) type {
    return NamedArray(SolutionAxis(MatAxis, RhsAxis), T);
}

// Synthesized inner-axis names for the factorization results. A decomposition
// introduces one genuinely new axis (rank / eigen / singular index) that maps to
// no input axis; each factor's *component* axes instead reuse A's row/col labels
// (see the result-type factories below). The two factors that contract over the
// inner axis share the *same* synthesized name, so they compose by name.
const qr_inner = "qr_rank";
const eig_inner = "eig";
const svd_inner = "sv";

/// `@compileError` if a synthesized inner-axis name collides with one of the
/// caller's input axis names. The inner axis is deliberately not caller-supplied,
/// so a collision is the caller's to resolve by renaming their own axis.
fn assertInnerFree(comptime Axis: type, comptime inner: [:0]const u8) void {
    inline for (comptime meta.fieldNames(Axis)) |name| {
        if (comptime std.mem.eql(u8, name, inner))
            @compileError("lapack: input axis '" ++ name ++ "' collides with the synthesized inner axis '" ++ inner ++ "'; rename your input axis");
    }
}

// ===== Raw LAPACK dispatch wrappers ==========================================
// Thin comptime switches over the scalar precision. Complex pointers are
// @ptrCast because our Complex(T) is layout-compatible with C `_Complex`.

fn xgetrf(comptime T: type, m: *c_int, n: *c_int, a: [*]T, lda: *c_int, ipiv: [*]c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgetrf_(m, n, a, lda, ipiv, info),
        f64 => c.dgetrf_(m, n, a, lda, ipiv, info),
        Complex(f32) => zarray_cgetrf(m, n, @ptrCast(a), lda, ipiv, info),
        Complex(f64) => zarray_zgetrf(m, n, @ptrCast(a), lda, ipiv, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xgetrs(comptime T: type, trans: *const u8, n: *c_int, nrhs: *c_int, a: [*]T, lda: *c_int, ipiv: [*]c_int, b: [*]T, ldb: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info),
        f64 => c.dgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info),
        Complex(f32) => zarray_cgetrs(trans, n, nrhs, @ptrCast(a), lda, ipiv, @ptrCast(b), ldb, info),
        Complex(f64) => zarray_zgetrs(trans, n, nrhs, @ptrCast(a), lda, ipiv, @ptrCast(b), ldb, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xgetri(comptime T: type, n: *c_int, a: [*]T, lda: *c_int, ipiv: [*]c_int, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgetri_(n, a, lda, ipiv, work, lwork, info),
        f64 => c.dgetri_(n, a, lda, ipiv, work, lwork, info),
        Complex(f32) => zarray_cgetri(n, @ptrCast(a), lda, ipiv, @ptrCast(work), lwork, info),
        Complex(f64) => zarray_zgetri(n, @ptrCast(a), lda, ipiv, @ptrCast(work), lwork, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xpotrf(comptime T: type, uplo: *const u8, n: *c_int, a: [*]T, lda: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.spotrf_(uplo, n, a, lda, info),
        f64 => c.dpotrf_(uplo, n, a, lda, info),
        Complex(f32) => zarray_cpotrf(uplo, n, @ptrCast(a), lda, info),
        Complex(f64) => zarray_zpotrf(uplo, n, @ptrCast(a), lda, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xpotrs(comptime T: type, uplo: *const u8, n: *c_int, nrhs: *c_int, a: [*]T, lda: *c_int, b: [*]T, ldb: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.spotrs_(uplo, n, nrhs, a, lda, b, ldb, info),
        f64 => c.dpotrs_(uplo, n, nrhs, a, lda, b, ldb, info),
        Complex(f32) => zarray_cpotrs(uplo, n, nrhs, @ptrCast(a), lda, @ptrCast(b), ldb, info),
        Complex(f64) => zarray_zpotrs(uplo, n, nrhs, @ptrCast(a), lda, @ptrCast(b), ldb, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xgels(comptime T: type, trans: *const u8, m: *c_int, n: *c_int, nrhs: *c_int, a: [*]T, lda: *c_int, b: [*]T, ldb: *c_int, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info),
        f64 => c.dgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info),
        Complex(f32) => zarray_cgels(trans, m, n, nrhs, @ptrCast(a), lda, @ptrCast(b), ldb, @ptrCast(work), lwork, info),
        Complex(f64) => zarray_zgels(trans, m, n, nrhs, @ptrCast(a), lda, @ptrCast(b), ldb, @ptrCast(work), lwork, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xgeqrf(comptime T: type, m: *c_int, n: *c_int, a: [*]T, lda: *c_int, tau: [*]T, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgeqrf_(m, n, a, lda, tau, work, lwork, info),
        f64 => c.dgeqrf_(m, n, a, lda, tau, work, lwork, info),
        Complex(f32) => zarray_cgeqrf(m, n, @ptrCast(a), lda, @ptrCast(tau), @ptrCast(work), lwork, info),
        Complex(f64) => zarray_zgeqrf(m, n, @ptrCast(a), lda, @ptrCast(tau), @ptrCast(work), lwork, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

/// Assembles the leading columns of Q. Real matrices use `orgqr` (orthogonal);
/// complex use `ungqr` (unitary) — the complex analog with the same argument
/// shape, routed through the shim.
fn xorgqr(comptime T: type, m: *c_int, n: *c_int, k: *c_int, a: [*]T, lda: *c_int, tau: [*]T, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sorgqr_(m, n, k, a, lda, tau, work, lwork, info),
        f64 => c.dorgqr_(m, n, k, a, lda, tau, work, lwork, info),
        Complex(f32) => zarray_cungqr(m, n, k, @ptrCast(a), lda, @ptrCast(tau), @ptrCast(work), lwork, info),
        Complex(f64) => zarray_zungqr(m, n, k, @ptrCast(a), lda, @ptrCast(tau), @ptrCast(work), lwork, info),
        else => @compileError("lapack: unsupported scalar " ++ @typeName(T)),
    }
}

fn xsyev(comptime T: type, jobz: *const u8, uplo: *const u8, n: *c_int, a: [*]T, lda: *c_int, w: [*]T, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info),
        f64 => c.dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info),
        else => @compileError("lapack: xsyev is real-only; complex uses xheev (" ++ @typeName(T) ++ ")"),
    }
}

/// Hermitian eigensolve (complex analog of `syev`). Eigenvalues `w` and `rwork`
/// stay real (`RealOf(T)`); `a`/`work` are complex. Complex scalars only.
fn xheev(comptime T: type, jobz: *const u8, uplo: *const u8, n: *c_int, a: [*]T, lda: *c_int, w: [*]RealOf(T), work: [*]T, lwork: *c_int, rwork: [*]RealOf(T), info: *c_int) void {
    switch (T) {
        Complex(f32) => zarray_cheev(jobz, uplo, n, @ptrCast(a), lda, w, @ptrCast(work), lwork, rwork, info),
        Complex(f64) => zarray_zheev(jobz, uplo, n, @ptrCast(a), lda, w, @ptrCast(work), lwork, rwork, info),
        else => @compileError("lapack: xheev is complex-only (" ++ @typeName(T) ++ ")"),
    }
}

fn xgeev(comptime T: type, jobvl: *const u8, jobvr: *const u8, n: *c_int, a: [*]T, lda: *c_int, wr: [*]T, wi: [*]T, vl: [*]T, ldvl: *c_int, vr: [*]T, ldvr: *c_int, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info),
        f64 => c.dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info),
        else => @compileError("lapack: xgeev is real-only; complex uses xgeevc (" ++ @typeName(T) ++ ")"),
    }
}

/// General eigensolve (complex analog of `geev`). A single complex `w` replaces
/// the real `wr`/`wi` split; `rwork` stays real. Complex scalars only.
fn xgeevc(comptime T: type, jobvl: *const u8, jobvr: *const u8, n: *c_int, a: [*]T, lda: *c_int, w: [*]T, vl: [*]T, ldvl: *c_int, vr: [*]T, ldvr: *c_int, work: [*]T, lwork: *c_int, rwork: [*]RealOf(T), info: *c_int) void {
    switch (T) {
        Complex(f32) => zarray_cgeev(jobvl, jobvr, n, @ptrCast(a), lda, @ptrCast(w), @ptrCast(vl), ldvl, @ptrCast(vr), ldvr, @ptrCast(work), lwork, rwork, info),
        Complex(f64) => zarray_zgeev(jobvl, jobvr, n, @ptrCast(a), lda, @ptrCast(w), @ptrCast(vl), ldvl, @ptrCast(vr), ldvr, @ptrCast(work), lwork, rwork, info),
        else => @compileError("lapack: xgeevc is complex-only (" ++ @typeName(T) ++ ")"),
    }
}

fn xgesdd(comptime T: type, jobz: *const u8, m: *c_int, n: *c_int, a: [*]T, lda: *c_int, s: [*]T, u: [*]T, ldu: *c_int, vt: [*]T, ldvt: *c_int, work: [*]T, lwork: *c_int, iwork: [*]c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info),
        f64 => c.dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info),
        else => @compileError("lapack: xgesdd is real-only; complex uses xgesddc (" ++ @typeName(T) ++ ")"),
    }
}

/// SVD (complex analog of `gesdd`). Singular values `s` and `rwork` stay real;
/// `u`/`vt` are complex. Complex scalars only.
fn xgesddc(comptime T: type, jobz: *const u8, m: *c_int, n: *c_int, a: [*]T, lda: *c_int, s: [*]RealOf(T), u: [*]T, ldu: *c_int, vt: [*]T, ldvt: *c_int, work: [*]T, lwork: *c_int, rwork: [*]RealOf(T), iwork: [*]c_int, info: *c_int) void {
    switch (T) {
        Complex(f32) => zarray_cgesdd(jobz, m, n, @ptrCast(a), lda, s, @ptrCast(u), ldu, @ptrCast(vt), ldvt, @ptrCast(work), lwork, rwork, iwork, info),
        Complex(f64) => zarray_zgesdd(jobz, m, n, @ptrCast(a), lda, s, @ptrCast(u), ldu, @ptrCast(vt), ldvt, @ptrCast(work), lwork, rwork, iwork, info),
        else => @compileError("lapack: xgesddc is complex-only (" ++ @typeName(T) ++ ")"),
    }
}

fn xsygv(comptime T: type, itype: *c_int, jobz: *const u8, uplo: *const u8, n: *c_int, a: [*]T, lda: *c_int, b: [*]T, ldb: *c_int, w: [*]T, work: [*]T, lwork: *c_int, info: *c_int) void {
    switch (T) {
        f32 => c.ssygv_(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info),
        f64 => c.dsygv_(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info),
        else => @compileError("lapack: xsygv is real-only; complex uses xhegv (" ++ @typeName(T) ++ ")"),
    }
}

/// Generalized Hermitian-definite eigensolve (complex analog of `sygv`).
/// Eigenvalues `w` and `rwork` stay real (`RealOf(T)`); `a`/`b`/`work` are
/// complex. Complex scalars only.
fn xhegv(comptime T: type, itype: *c_int, jobz: *const u8, uplo: *const u8, n: *c_int, a: [*]T, lda: *c_int, b: [*]T, ldb: *c_int, w: [*]RealOf(T), work: [*]T, lwork: *c_int, rwork: [*]RealOf(T), info: *c_int) void {
    switch (T) {
        Complex(f32) => zarray_chegv(itype, jobz, uplo, n, @ptrCast(a), lda, @ptrCast(b), ldb, w, @ptrCast(work), lwork, rwork, info),
        Complex(f64) => zarray_zhegv(itype, jobz, uplo, n, @ptrCast(a), lda, @ptrCast(b), ldb, w, @ptrCast(work), lwork, rwork, info),
        else => @compileError("lapack: xhegv is complex-only (" ++ @typeName(T) ++ ")"),
    }
}

fn xgelsd(comptime T: type, m: *c_int, n: *c_int, nrhs: *c_int, a: [*]T, lda: *c_int, b: [*]T, ldb: *c_int, s: [*]T, rcond: *const T, rank: *c_int, work: [*]T, lwork: *c_int, iwork: [*]c_int, info: *c_int) void {
    switch (T) {
        f32 => c.sgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info),
        f64 => c.dgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info),
        else => @compileError("lapack: xgelsd is real-only; complex uses xgelsdc (" ++ @typeName(T) ++ ")"),
    }
}

/// Rank-deficient least squares via SVD (complex analog of `gelsd`). Singular
/// values `s`, `rcond`, and `rwork` stay real (`RealOf(T)`); `a`/`b`/`work` are
/// complex. Complex scalars only.
fn xgelsdc(comptime T: type, m: *c_int, n: *c_int, nrhs: *c_int, a: [*]T, lda: *c_int, b: [*]T, ldb: *c_int, s: [*]RealOf(T), rcond: *const RealOf(T), rank: *c_int, work: [*]T, lwork: *c_int, rwork: [*]RealOf(T), iwork: [*]c_int, info: *c_int) void {
    switch (T) {
        Complex(f32) => zarray_cgelsd(m, n, nrhs, @ptrCast(a), lda, @ptrCast(b), ldb, s, rcond, rank, @ptrCast(work), lwork, rwork, iwork, info),
        Complex(f64) => zarray_zgelsd(m, n, nrhs, @ptrCast(a), lda, @ptrCast(b), ldb, s, rcond, rank, @ptrCast(work), lwork, rwork, iwork, info),
        else => @compileError("lapack: xgelsdc is complex-only (" ++ @typeName(T) ++ ")"),
    }
}

// ===== LU family =============================================================

/// Solve A·X = B in place (LAPACK `getrf` + `getrs`). A's storage order is
/// absorbed via the `getrs` transpose flag (zero copy). Returns X as a view
/// that *aliases B's buffer*, with B's row axis relabeled to A's column axis.
///
/// Input: overwrites `a` with LU factors and `b` with X. To keep `a`, factor a
/// copy with `lu` and call `luSolve`.
/// Ownership: the result shares `b`'s memory — nothing to free.
///
/// - `a`: square n×n, any (positive-stride) layout. Overwritten with LU factors.
/// - `b`: n×nrhs; column-major, or a single RHS (layout-agnostic). Overwritten
///        with X.
/// - `ipiv`: caller-owned scratch, length >= n.
///
/// Panics on an unrecoverable contract violation (non-square A, mismatched row
/// extent, short `ipiv`). Returns an error for recoverable conditions.
pub fn solve(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    a: NamedArray(MatAxis, T),
    b: NamedArray(RhsAxis, T),
    ipiv: []c_int,
) LapackError!Solution(MatAxis, RhsAxis, T) {
    comptime assertTwoAxes(MatAxis);
    comptime assertTwoAxes(RhsAxis);
    const row = comptime sharedAxisName(MatAxis, RhsAxis);
    const col = comptime otherAxis(MatAxis, row);

    const n = @field(a.idx.shape, row);
    if (@field(a.idx.shape, col) != n) @panic("solve: A must be square");
    if (@field(b.idx.shape, row) != n) @panic("solve: A and B row extents differ");
    assert(ipiv.len >= n);

    const am = try describe(T, MatAxis, a, row, col);
    const bm = try describe(T, RhsAxis, b, row, comptime otherAxis(RhsAxis, row));
    if (bm.layout == .row_major and bm.n > 1) return error.RhsNotColumnMajor;

    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var ldb: c_int = bm.lda;
    var nrhs: c_int = bm.n;
    var info: c_int = undefined;

    xgetrf(T, &n_i, &n_i, am.ptr, &lda, ipiv.ptr, &info);
    if (info < 0) @panic("solve: illegal argument to getrf (binding bug)");
    if (info > 0) return error.Singular;

    // Row-major A reinterpreted column-major is Aᵀ, so 'T' recovers A·X=B.
    // Plain transpose (never 'C') even for complex: reinterpretation transposes
    // but does not conjugate.
    var trans: u8 = if (am.layout == .col_major) 'N' else 'T';
    xgetrs(T, &trans, &n_i, &nrhs, am.ptr, &lda, ipiv.ptr, bm.ptr, &ldb, &info);
    if (info < 0) @panic("solve: illegal argument to getrs (binding bug)");

    return b.renameAxes(SolutionAxis(MatAxis, RhsAxis), &.{.{ .old = row, .new = col }});
}

/// LU-factor a square matrix in place (LAPACK `getrf`). On return `a` holds the
/// L (unit-diagonal, below) and U (on/above) factors and `ipiv` the pivots.
/// Feed both to `luSolve`. Returns `error.Singular` for a zero pivot.
///
/// Input: overwrites `a` with the L/U factors — this is the deliverable (feed to
/// `luSolve`), not an incidental side effect.
/// Ownership: returns void; `a` and `ipiv` are caller-owned — nothing to free.
pub fn lu(
    comptime T: type,
    comptime Axis: type,
    a: NamedArray(Axis, T),
    ipiv: []c_int,
) LapackError!void {
    comptime assertTwoAxes(Axis);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("lu: matrix must be square");
    assert(ipiv.len >= n);

    const am = try describe(T, Axis, a, rows, cols);
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var info: c_int = undefined;
    xgetrf(T, &n_i, &n_i, am.ptr, &lda, ipiv.ptr, &info);
    if (info < 0) @panic("lu: illegal argument to getrf (binding bug)");
    if (info > 0) return error.Singular;
}

/// Solve A·X = B using factors from `lu`. `a_lu`/`ipiv` come from `lu` on A;
/// `a_lu` must carry the same index (layout) A had. Returns X aliasing B.
///
/// Input: reads `a_lu` (unmodified); overwrites `b` with X.
/// Ownership: the result shares `b`'s memory — nothing to free.
pub fn luSolve(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    a_lu: NamedArrayConst(MatAxis, T),
    ipiv: []const c_int,
    b: NamedArray(RhsAxis, T),
) LapackError!Solution(MatAxis, RhsAxis, T) {
    comptime assertTwoAxes(MatAxis);
    comptime assertTwoAxes(RhsAxis);
    const row = comptime sharedAxisName(MatAxis, RhsAxis);
    const col = comptime otherAxis(MatAxis, row);

    const n = @field(a_lu.idx.shape, row);
    if (@field(a_lu.idx.shape, col) != n) @panic("luSolve: A must be square");
    if (@field(b.idx.shape, row) != n) @panic("luSolve: A and B row extents differ");
    assert(ipiv.len >= n);

    const am = try describeConst(T, MatAxis, a_lu, row, col);
    const bm = try describe(T, RhsAxis, b, row, comptime otherAxis(RhsAxis, row));
    if (bm.layout == .row_major and bm.n > 1) return error.RhsNotColumnMajor;

    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var ldb: c_int = bm.lda;
    var nrhs: c_int = bm.n;
    var info: c_int = undefined;
    var trans: u8 = if (am.layout == .col_major) 'N' else 'T';
    // getrs only reads A and ipiv, but its C ABI types both as non-const;
    // cast away const at the boundary for both.
    const a_ptr: [*]T = @constCast(am.ptr);
    const ipiv_ptr: [*]c_int = @constCast(ipiv.ptr);
    xgetrs(T, &trans, &n_i, &nrhs, a_ptr, &lda, ipiv_ptr, bm.ptr, &ldb, &info);
    if (info < 0) @panic("luSolve: illegal argument to getrs (binding bug)");

    return b.renameAxes(SolutionAxis(MatAxis, RhsAxis), &.{.{ .old = row, .new = col }});
}

/// Determinant of a square matrix. Overwrites `a` with its LU factors and uses
/// `ipiv` as scratch (length >= n). Layout-transparent. A singular matrix
/// yields exactly 0 (not an error).
///
/// Input: overwrites `a` (scratch). Use `det` to preserve `a`.
/// Ownership: returns a scalar by value — nothing to free.
pub fn detInplace(
    comptime T: type,
    comptime Axis: type,
    a: NamedArray(Axis, T),
    ipiv: []c_int,
) T {
    comptime assertTwoAxes(Axis);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("det: matrix must be square");
    assert(ipiv.len >= n);

    const am = describe(T, Axis, a, rows, cols) catch @panic("det: matrix view is not contiguous");
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var info: c_int = undefined;
    xgetrf(T, &n_i, &n_i, am.ptr, &lda, ipiv.ptr, &info);
    if (info < 0) @panic("det: illegal argument to getrf (binding bug)");
    if (info > 0) return if (comptime isComplex(T)) T.init(0, 0) else 0;

    // det = (sign from row swaps) * product of U's diagonal. The diagonal sits
    // at buffer positions base + i*(lda+1) in the column-major interpretation
    // (identical for the row-major/transpose case: det(Aᵀ) = det(A)).
    const lda_us: usize = @intCast(am.lda);
    var product: T = if (comptime isComplex(T)) T.init(1, 0) else 1;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const diag = am.ptr[i * (lda_us + 1)];
        product = if (comptime isComplex(T)) product.mul(diag) else product * diag;
    }
    var sign: RealOf(T) = 1;
    i = 0;
    while (i < n) : (i += 1) {
        if (ipiv[i] != @as(c_int, @intCast(i + 1))) sign = -sign;
    }
    return if (comptime isComplex(T)) T.init(product.re * sign, product.im * sign) else product * sign;
}

/// Input-preserving (default) variant of `detInplace`: factors a private
/// contiguous copy of `a` and allocates its own pivots, leaving the caller's
/// `a` untouched.
///
/// Input: `a` is left unmodified. Ownership: returns a scalar by value — nothing
/// to free; the internal copy and pivots are freed before returning.
pub fn det(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
) Allocator.Error!T {
    comptime assertTwoAxes(Axis);
    const n = @field(a.idx.shape, meta.fieldNames(Axis)[0]);
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    const ipiv = try allocator.alloc(c_int, @max(n, 1));
    defer allocator.free(ipiv);
    return detInplace(T, Axis, copy, ipiv);
}

/// Invert a square matrix in place (LAPACK `getrf` + `getri`). Layout-
/// transparent (inv(Aᵀ) = inv(A)ᵀ). Allocates the LAPACK workspace and pivots.
/// Returns `error.Singular` if A is singular.
///
/// Input: overwrites `a` with its inverse — this is the deliverable.
/// Ownership: returns void; the inverse overwrites `a`. All internal scratch is
/// freed before returning — nothing for the caller to free.
pub fn inv(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
) (LapackError || Allocator.Error)!void {
    comptime assertTwoAxes(Axis);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("inv: matrix must be square");

    const am = try describe(T, Axis, a, rows, cols);
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var info: c_int = undefined;

    const ipiv = try allocator.alloc(c_int, @max(n, 1));
    defer allocator.free(ipiv);

    xgetrf(T, &n_i, &n_i, am.ptr, &lda, ipiv.ptr, &info);
    if (info < 0) @panic("inv: illegal argument to getrf (binding bug)");
    if (info > 0) return error.Singular;

    // Workspace size query (lwork = -1 returns the optimum in work[0]).
    var lwork: c_int = -1;
    var work_query: [1]T = undefined;
    xgetri(T, &n_i, am.ptr, &lda, ipiv.ptr, &work_query, &lwork, &info);
    if (info < 0) @panic("inv: illegal argument to getri query (binding bug)");
    lwork = lworkFrom(T, work_query[0]);
    const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
    defer allocator.free(work);

    xgetri(T, &n_i, am.ptr, &lda, ipiv.ptr, work.ptr, &lwork, &info);
    if (info < 0) @panic("inv: illegal argument to getri (binding bug)");
    if (info > 0) return error.Singular;
}

// ===== Cholesky family =======================================================

/// Cholesky-factor a symmetric/Hermitian positive-definite matrix in place
/// (LAPACK `potrf`). `tri` selects which triangle holds the input (and receives
/// the factor). Row-major storage is absorbed by flipping the effective
/// triangle. Returns `error.NotPositiveDefinite` if the factorization fails.
///
/// Input: overwrites `a` with its Cholesky factor — this is the deliverable.
/// Ownership: returns void; the factor overwrites `a` — nothing to free.
pub fn cholesky(
    comptime T: type,
    comptime Axis: type,
    a: NamedArray(Axis, T),
    tri: Triangle,
) LapackError!void {
    comptime assertTwoAxes(Axis);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("cholesky: matrix must be square");

    const am = try describe(T, Axis, a, rows, cols);
    const eff = if (am.layout == .col_major) tri else tri.flip();
    var uplo: u8 = eff.uploChar();
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var info: c_int = undefined;
    xpotrf(T, &uplo, &n_i, am.ptr, &lda, &info);
    if (info < 0) @panic("cholesky: illegal argument to potrf (binding bug)");
    if (info > 0) return error.NotPositiveDefinite;
}

/// Solve A·X = B given a Cholesky factor from `cholesky` (LAPACK `potrs`).
/// Pass the same `tri` used to factor. Returns X aliasing B.
///
/// Input: reads `a_chol` (unmodified); overwrites `b` with X.
/// Ownership: the result shares `b`'s memory — nothing to free.
pub fn choleskySolve(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    a_chol: NamedArrayConst(MatAxis, T),
    b: NamedArray(RhsAxis, T),
    tri: Triangle,
) LapackError!Solution(MatAxis, RhsAxis, T) {
    comptime assertTwoAxes(MatAxis);
    comptime assertTwoAxes(RhsAxis);
    const row = comptime sharedAxisName(MatAxis, RhsAxis);
    const col = comptime otherAxis(MatAxis, row);

    const n = @field(a_chol.idx.shape, row);
    if (@field(a_chol.idx.shape, col) != n) @panic("choleskySolve: A must be square");
    if (@field(b.idx.shape, row) != n) @panic("choleskySolve: A and B row extents differ");

    const am = try describeConst(T, MatAxis, a_chol, row, col);
    const bm = try describe(T, RhsAxis, b, row, comptime otherAxis(RhsAxis, row));
    if (bm.layout == .row_major and bm.n > 1) return error.RhsNotColumnMajor;

    const eff = if (am.layout == .col_major) tri else tri.flip();
    var uplo: u8 = eff.uploChar();
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var ldb: c_int = bm.lda;
    var nrhs: c_int = bm.n;
    var info: c_int = undefined;
    // potrs only reads the Cholesky factor, but its C ABI types A as non-const.
    const a_ptr: [*]T = @constCast(am.ptr);
    xpotrs(T, &uplo, &n_i, &nrhs, a_ptr, &lda, bm.ptr, &ldb, &info);
    if (info < 0) @panic("choleskySolve: illegal argument to potrs (binding bug)");

    return b.renameAxes(SolutionAxis(MatAxis, RhsAxis), &.{.{ .old = row, .new = col }});
}

// ===== Least squares =========================================================

/// Minimum-norm least-squares solution of an overdetermined (m >= n) system
/// A·X ≈ B (LAPACK `gels`). `a` is used as scratch (overwritten with QR/LQ
/// factors) and `b` (m×nrhs) is overwritten, its first n rows holding X.
/// Any (positive-stride) layout of `a` is accepted with **no copy**: a row-major
/// `a` is bit-identical to Aᵀ in column-major, which `gels` absorbs via its
/// `trans` flag (the same reinterpretation `solve` uses). `b` must be
/// column-major (or a single RHS).
///
/// Returns the solution X = {C, B.rhs} (C = A's column axis) as a *view of `b`'s
/// first n rows* — the same axis vocabulary `solve`/`choleskySolve` return.
///
/// Input: overwrites `a` (QR/LQ factors) and `b` (holds X). Use `lstsq` to
/// preserve `a` (`b` still receives X either way). A complex row-major `a` is
/// left unmodified (factored from a packed column-major copy).
/// Ownership: the returned view aliases `b`'s buffer — nothing to free, and
/// writes through it mutate `b`. Internal workspace is freed before returning.
pub fn lstsqInplace(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    allocator: Allocator,
    a: NamedArray(MatAxis, T),
    b: NamedArray(RhsAxis, T),
) (LapackError || Allocator.Error)!Solution(MatAxis, RhsAxis, T) {
    comptime assertTwoAxes(MatAxis);
    comptime assertTwoAxes(RhsAxis);
    const row = comptime sharedAxisName(MatAxis, RhsAxis);
    const col = comptime otherAxis(MatAxis, row);

    const m = @field(a.idx.shape, row);
    const nn = @field(a.idx.shape, otherAxis(MatAxis, row));
    if (@field(b.idx.shape, row) != m) @panic("lstsq: A and B row extents differ");
    if (m < nn) @panic("lstsq: only overdetermined systems (m >= n) are supported for now");

    const am = try describe(T, MatAxis, a, row, comptime otherAxis(MatAxis, row));
    const bm = try describe(T, RhsAxis, b, row, comptime otherAxis(RhsAxis, row));
    if (bm.layout != .col_major and bm.n > 1) return error.NotColumnMajor;

    var a_ptr = am.ptr;
    var a_lda: c_int = am.lda;
    var trans: u8 = undefined;
    var m_i: c_int = undefined;
    var n_i: c_int = undefined;
    var packed_a: ?[]T = null;
    if (comptime isComplex(T)) {
        // `gels` complex offers only trans ∈ {'N','C'} — no plain 'T' — so the
        // real row-major = Aᵀ reinterpretation trick is unavailable (it would
        // need an unconjugated transpose). Pack a column-major copy of a
        // row-major A and solve with trans='N'; a column-major A is native.
        trans = 'N';
        m_i = @intCast(m);
        n_i = @intCast(nn);
        if (am.layout != .col_major) {
            const buf = try allocator.alloc(T, @max(m * nn, 1));
            var j: usize = 0;
            while (j < nn) : (j += 1) {
                var i: usize = 0;
                while (i < m) : (i += 1) buf[i + j * m] = readElem(T, am, i, j);
            }
            packed_a = buf;
            a_ptr = buf.ptr;
            a_lda = @intCast(@max(m, 1));
        }
    } else {
        // A row-major A is Aᵀ in column-major; `gels` recovers A·X≈B by setting
        // trans='T' and swapping the logical m/n it is told (zero copy).
        trans = if (am.layout == .col_major) 'N' else 'T';
        m_i = if (am.layout == .col_major) @intCast(m) else @intCast(nn);
        n_i = if (am.layout == .col_major) @intCast(nn) else @intCast(m);
    }
    defer if (packed_a) |buf| allocator.free(buf);

    var nrhs: c_int = bm.n;
    var ldb: c_int = bm.lda;
    var info: c_int = undefined;

    var lwork: c_int = -1;
    var work_query: [1]T = undefined;
    xgels(T, &trans, &m_i, &n_i, &nrhs, a_ptr, &a_lda, bm.ptr, &ldb, &work_query, &lwork, &info);
    if (info < 0) @panic("lstsq: illegal argument to gels query (binding bug)");
    lwork = lworkFrom(T, work_query[0]);
    const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
    defer allocator.free(work);

    xgels(T, &trans, &m_i, &n_i, &nrhs, a_ptr, &a_lda, bm.ptr, &ldb, work.ptr, &lwork, &info);
    if (info < 0) @panic("lstsq: illegal argument to gels (binding bug)");
    if (info > 0) return error.Singular; // A is rank-deficient

    // X occupies the first n rows of b (column-major). Slice b's row axis to
    // [0, n) and rename that axis row→C, yielding X = {C, B.rhs} as a view.
    const sliced = b.idx.sliceAxis(@field(RhsAxis, row), 0, nn);
    const view = NamedArray(RhsAxis, T).init(sliced, b.buf);
    return view.renameAxes(SolutionAxis(MatAxis, RhsAxis), &.{.{ .old = row, .new = col }});
}

/// Input-preserving (default) variant of `lstsqInplace`: factors a private
/// contiguous copy of `a`, leaving the caller's `a` untouched. `b` is still
/// overwritten with X (it is the result buffer the returned view aliases).
///
/// Input: `a` is left unmodified; `b` receives X. Ownership: the returned view
/// aliases `b`'s buffer — nothing to free; the internal copy of `a` is freed
/// before returning.
pub fn lstsq(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    allocator: Allocator,
    a: NamedArrayConst(MatAxis, T),
    b: NamedArray(RhsAxis, T),
) (LapackError || Allocator.Error)!Solution(MatAxis, RhsAxis, T) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return lstsqInplace(T, MatAxis, RhsAxis, allocator, copy, b);
}

// ===== Rank-deficient least squares (divide-and-conquer SVD) =================

pub fn LstsqSvdResult(comptime MatAxis: type, comptime RhsAxis: type, comptime T: type) type {
    return struct {
        /// Minimum-norm least-squares solution X = {C, B.rhs} (C = A's column
        /// axis), a *view aliasing `b`'s first n rows* — writes through it mutate
        /// `b`; there is nothing to free for this field.
        x: Solution(MatAxis, RhsAxis, T),
        /// Effective numerical rank of A: the number of singular values above
        /// the `rcond` cutoff.
        rank: usize,
        /// Singular values of A in descending order, length min(m, n). Owned.
        singular_values: []RealOf(T),
        /// Free the owned `singular_values` (the `x` view aliases `b`). Call
        /// once, with the allocator passed to `lstsqSvd`.
        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.singular_values);
        }
    };
}

/// Minimum-norm least-squares solution of A·X ≈ B for a possibly **rank-
/// deficient** overdetermined (m >= n) system, via divide-and-conquer SVD
/// (LAPACK `gelsd`). Unlike `lstsq` (`gels`, which assumes full rank), this
/// tolerates rank deficiency and returns the effective numerical `rank` and the
/// singular values. Singular values `s` below `rcond * s[0]` are treated as
/// zero; a negative `rcond` uses machine precision.
///
/// `gelsd` has no transpose flag, so `a` must be column-major — a non-column-
/// major `a` is packed into a column-major copy (leaving the caller's `a`
/// unmodified); a column-major `a` is overwritten (scratch). `b` (m×nrhs) is
/// overwritten, its first n rows holding X; it must be column-major (or a single
/// RHS).
///
/// Input: a column-major `a` is overwritten; a non-column-major `a` is preserved.
/// `b` is overwritten (holds X). Use `lstsqSvd` to always preserve `a`.
/// Ownership: the returned `x` aliases `b`; free `singular_values` with
/// `res.deinit(allocator)`.
pub fn lstsqSvdInplace(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    allocator: Allocator,
    a: NamedArray(MatAxis, T),
    b: NamedArray(RhsAxis, T),
    rcond: RealOf(T),
) (LapackError || Allocator.Error)!LstsqSvdResult(MatAxis, RhsAxis, T) {
    comptime assertTwoAxes(MatAxis);
    comptime assertTwoAxes(RhsAxis);
    const Re = RealOf(T);
    const row = comptime sharedAxisName(MatAxis, RhsAxis);
    const col = comptime otherAxis(MatAxis, row);

    const m = @field(a.idx.shape, row);
    const nn = @field(a.idx.shape, otherAxis(MatAxis, row));
    if (@field(b.idx.shape, row) != m) @panic("lstsqSvd: A and B row extents differ");
    if (m < nn) @panic("lstsqSvd: only overdetermined systems (m >= n) are supported for now");

    const am = try describe(T, MatAxis, a, row, comptime otherAxis(MatAxis, row));
    const bm = try describe(T, RhsAxis, b, row, comptime otherAxis(RhsAxis, row));
    if (bm.layout != .col_major and bm.n > 1) return error.NotColumnMajor;

    // gelsd has no transpose flag, so A must be column-major. A column-major A is
    // factored in place; a non-column-major A is packed into a column-major copy
    // (the caller's A is then left unmodified).
    var a_ptr = am.ptr;
    var a_lda: c_int = am.lda;
    var packed_a: ?[]T = null;
    if (am.layout != .col_major) {
        const buf = try allocator.alloc(T, @max(m * nn, 1));
        var j: usize = 0;
        while (j < nn) : (j += 1) {
            var i: usize = 0;
            while (i < m) : (i += 1) buf[i + j * m] = readElem(T, am, i, j);
        }
        packed_a = buf;
        a_ptr = buf.ptr;
        a_lda = @intCast(@max(m, 1));
    }
    defer if (packed_a) |buf| allocator.free(buf);

    var m_i: c_int = @intCast(m);
    var n_i: c_int = @intCast(nn);
    var nrhs: c_int = bm.n;
    var ldb: c_int = bm.lda;
    var rc: Re = rcond;
    var rank_i: c_int = undefined;
    var info: c_int = undefined;

    const minmn = @min(m, nn);
    const s = try allocator.alloc(Re, @max(minmn, 1));
    errdefer allocator.free(s);

    // Workspace query. LAPACK returns the optimal `lwork` in work[0], the minimum
    // `liwork` in iwork[0], and (complex) the minimum `lrwork` in rwork[0]. The
    // query scratch is initialized to 0 so a non-writing implementation can't
    // hand us garbage; each size is floored at 1.
    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    var iwq: [1]c_int = .{0};
    if (comptime isComplex(T)) {
        var rwq: [1]Re = .{0};
        xgelsdc(T, &m_i, &n_i, &nrhs, a_ptr, &a_lda, bm.ptr, &ldb, s.ptr, &rc, &rank_i, &wq, &lwork, &rwq, &iwq, &info);
        if (info < 0) @panic("lstsqSvd: illegal argument to gelsd query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const lrwork: usize = @intCast(@max(@as(c_int, @intFromFloat(@round(rwq[0]))), 1));
        const liwork: usize = @intCast(@max(iwq[0], 1));
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        const rwork = try allocator.alloc(Re, lrwork);
        defer allocator.free(rwork);
        const iwork = try allocator.alloc(c_int, liwork);
        defer allocator.free(iwork);
        xgelsdc(T, &m_i, &n_i, &nrhs, a_ptr, &a_lda, bm.ptr, &ldb, s.ptr, &rc, &rank_i, work.ptr, &lwork, rwork.ptr, iwork.ptr, &info);
        if (info < 0) @panic("lstsqSvd: illegal argument to gelsd (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    } else {
        xgelsd(T, &m_i, &n_i, &nrhs, a_ptr, &a_lda, bm.ptr, &ldb, s.ptr, &rc, &rank_i, &wq, &lwork, &iwq, &info);
        if (info < 0) @panic("lstsqSvd: illegal argument to gelsd query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const liwork: usize = @intCast(@max(iwq[0], 1));
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        const iwork = try allocator.alloc(c_int, liwork);
        defer allocator.free(iwork);
        xgelsd(T, &m_i, &n_i, &nrhs, a_ptr, &a_lda, bm.ptr, &ldb, s.ptr, &rc, &rank_i, work.ptr, &lwork, iwork.ptr, &info);
        if (info < 0) @panic("lstsqSvd: illegal argument to gelsd (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    }

    // X occupies the first n rows of b (column-major); slice + rename row→C.
    const sliced = b.idx.sliceAxis(@field(RhsAxis, row), 0, nn);
    const view = NamedArray(RhsAxis, T).init(sliced, b.buf);
    const x = view.renameAxes(SolutionAxis(MatAxis, RhsAxis), &.{.{ .old = row, .new = col }});
    return .{ .x = x, .rank = @intCast(rank_i), .singular_values = s };
}

/// Input-preserving (default) variant of `lstsqSvdInplace`: factors a private
/// contiguous copy of `a`, so the caller's `a` is always left unmodified. `b` is
/// still overwritten with X (the result buffer the returned view aliases).
///
/// Input: `a` is left unmodified; `b` receives X. Ownership: the returned `x`
/// aliases `b`; free `singular_values` with `res.deinit(allocator)`. The
/// internal copy of `a` is freed before returning.
pub fn lstsqSvd(
    comptime T: type,
    comptime MatAxis: type,
    comptime RhsAxis: type,
    allocator: Allocator,
    a: NamedArrayConst(MatAxis, T),
    b: NamedArray(RhsAxis, T),
    rcond: RealOf(T),
) (LapackError || Allocator.Error)!LstsqSvdResult(MatAxis, RhsAxis, T) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return lstsqSvdInplace(T, MatAxis, RhsAxis, allocator, copy, b, rcond);
}

// ===== QR ====================================================================

pub fn QrResult(comptime Axis: type, comptime T: type) type {
    comptime assertTwoAxes(Axis);
    comptime assertInnerFree(Axis, qr_inner);
    const R = comptime meta.fieldNames(Axis)[0];
    const C = comptime meta.fieldNames(Axis)[1];
    const QAxis = KeyEnum(&.{ R, qr_inner });
    const RAxis = KeyEnum(&.{ qr_inner, C });
    return struct {
        /// Q with orthonormal columns, shape {R, qr_rank} = m×min(m,n) — R is A's
        /// row axis; `qr_rank` is the synthesized inner (rank) axis.
        q: NamedArray(QAxis, T),
        /// Upper-triangular R, shape {qr_rank, C} = min(m,n)×n — C is A's column
        /// axis. `q` and `r` share the `qr_rank` axis, so they compose by name.
        r: NamedArray(RAxis, T),
        /// Free both `q` and `r`. Call exactly once, with the same allocator
        /// passed to `qr`.
        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.q.buf);
            allocator.free(self.r.buf);
        }
    };
}

/// Thin/reduced QR factorization A = Q·R (LAPACK `geqrf` + `orgqr`), m >= n.
/// Any (positive-stride) layout of `a` is accepted. A column-major `a` is
/// factored in place (overwritten); QR is *not* transpose-invariant and `geqrf`
/// has no transpose flag, so a non-column-major `a` cannot be absorbed for free
/// (unlike `solve`/`lstsq`) — it is copied into a column-major work buffer (one
/// m×n allocation) and the caller's `a` is left unmodified. Q (m×n) and R
/// (n×n upper) are freshly allocated column-major arrays. Real matrices use
/// `orgqr` to form Q; complex use `ungqr` (unitary Q, `Qᴴ Q = I`).
///
/// Input: a column-major `a` is overwritten (factored in place); a
/// non-column-major `a` is left unmodified (packed copy). Use `qr` to
/// always preserve `a` regardless of layout.
/// Ownership: the caller owns the result; free it with `res.deinit(allocator)`
/// (releases both `q` and `r`).
pub fn qrInplace(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
) (LapackError || Allocator.Error)!QrResult(Axis, T) {
    comptime assertTwoAxes(Axis);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const m = @field(a.idx.shape, rows);
    const nn = @field(a.idx.shape, cols);
    if (m < nn) @panic("qr: only tall/square matrices (m >= n) are supported for now");

    const am = try describe(T, Axis, a, rows, cols);

    // QR is not transpose-invariant and geqrf has no transpose flag, so a
    // non-column-major A can't be absorbed for free (the only zero-copy route
    // is an LQ factorization of Aᵀ via gelqf/orglq, which we don't bind). Pack
    // a column-major copy; the caller's A is then left unmodified. A
    // column-major A is factored in place with no copy.
    var work_ptr = am.ptr;
    var work_lda = am.lda;
    var packed_a: ?[]T = null;
    if (am.layout != .col_major) {
        const w = try allocator.alloc(T, @max(m * nn, 1));
        var jc: usize = 0;
        while (jc < nn) : (jc += 1) {
            var ir: usize = 0;
            while (ir < m) : (ir += 1) w[ir + jc * m] = readElem(T, am, ir, jc);
        }
        packed_a = w;
        work_ptr = w.ptr;
        work_lda = @intCast(@max(m, 1));
    }
    defer if (packed_a) |w| allocator.free(w);

    const k = nn; // min(m, n) since m >= n
    var m_i: c_int = @intCast(m);
    var n_i: c_int = @intCast(nn);
    var k_i: c_int = @intCast(k);
    var lda: c_int = work_lda;
    var info: c_int = undefined;

    const tau = try allocator.alloc(T, @max(k, 1));
    defer allocator.free(tau);

    // Factor.
    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    xgeqrf(T, &m_i, &n_i, work_ptr, &lda, tau.ptr, &wq, &lwork, &info);
    if (info < 0) @panic("qr: illegal argument to geqrf query (binding bug)");
    lwork = lworkFrom(T, wq[0]);
    const work1 = try allocator.alloc(T, @intCast(@max(lwork, 1)));
    defer allocator.free(work1);
    xgeqrf(T, &m_i, &n_i, work_ptr, &lda, tau.ptr, work1.ptr, &lwork, &info);
    if (info < 0) @panic("qr: illegal argument to geqrf (binding bug)");

    // Extract R (k×n upper-triangular) into a fresh column-major buffer before
    // orgqr overwrites `a` with Q. Axes reuse A's labels: R = {qr_rank, C}.
    const RAxis = KeyEnum(&.{ qr_inner, cols });
    const zero = if (comptime isComplex(T)) T.init(0, 0) else 0;
    const rbuf = try allocator.alloc(T, @max(k * nn, 1));
    errdefer allocator.free(rbuf);
    @memset(rbuf, zero);
    const lda_us: usize = @intCast(work_lda);
    {
        var jc: usize = 0;
        while (jc < nn) : (jc += 1) {
            var ir: usize = 0;
            const rmax = @min(jc + 1, k);
            while (ir < rmax) : (ir += 1) rbuf[ir + jc * k] = work_ptr[jc * lda_us + ir];
        }
    }

    // Form Q (m×k) in place, then copy into a fresh column-major buffer.
    lwork = -1;
    xorgqr(T, &m_i, &k_i, &k_i, work_ptr, &lda, tau.ptr, &wq, &lwork, &info);
    if (info < 0) @panic("qr: illegal argument to orgqr query (binding bug)");
    lwork = lworkFrom(T, wq[0]);
    const work2 = try allocator.alloc(T, @intCast(@max(lwork, 1)));
    defer allocator.free(work2);
    xorgqr(T, &m_i, &k_i, &k_i, work_ptr, &lda, tau.ptr, work2.ptr, &lwork, &info);
    if (info < 0) @panic("qr: illegal argument to orgqr (binding bug)");

    // Q = {R, qr_rank}, sharing `qr_rank` with R above.
    const QAxis = KeyEnum(&.{ rows, qr_inner });
    const qbuf = try allocator.alloc(T, @max(m * k, 1));
    errdefer allocator.free(qbuf);
    {
        var jc: usize = 0;
        while (jc < k) : (jc += 1) {
            var ir: usize = 0;
            while (ir < m) : (ir += 1) qbuf[ir + jc * m] = work_ptr[jc * lda_us + ir];
        }
    }

    return .{
        .q = wrapMat(QAxis, T, qbuf, m, k, 1, @intCast(@max(m, 1))),
        .r = wrapMat(RAxis, T, rbuf, k, nn, 1, @intCast(@max(k, 1))),
    };
}

/// Input-preserving (default) variant of `qrInplace`: factors a private
/// contiguous copy of `a`, so the caller's `a` is always left unmodified
/// (`qrInplace` overwrites a column-major `a` in place).
///
/// Input: `a` is left unmodified. Ownership: the caller owns the result; free it
/// with `res.deinit(allocator)`. The internal copy of `a` is freed before
/// returning.
pub fn qr(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
) (LapackError || Allocator.Error)!QrResult(Axis, T) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return qrInplace(T, Axis, allocator, copy);
}

// ===== Spectral (values only) ================================================

/// Eigenvalues of a symmetric/Hermitian matrix, ascending (LAPACK `syev` for
/// real, `heev` for complex; `jobz='N'`). `tri` selects the referenced
/// triangle; row-major storage is absorbed by flipping it. `a` is overwritten
/// (scratch). For complex `T` the matrix is treated as **Hermitian** (the
/// imaginary part of the diagonal is assumed zero); eigenvalues are real, so the
/// result element type is `RealOf(T)`.
///
/// Input: overwrites `a` (scratch). Use `eigSym` to preserve `a`.
///
/// Ownership: the caller owns the returned slice; free it with
/// `allocator.free(slice)`.
pub fn eigSymInplace(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
    tri: Triangle,
) (LapackError || Allocator.Error)![]RealOf(T) {
    comptime assertTwoAxes(Axis);
    const R = RealOf(T);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("eigSym: matrix must be square");

    const am = try describe(T, Axis, a, rows, cols);
    // A row-major Hermitian view stored in the given triangle is the transpose
    // (= conjugate of the other triangle); flipping the triangle absorbs it, and
    // heev conjugates internally, so eigenvalues are unaffected.
    const eff = if (am.layout == .col_major) tri else tri.flip();
    var jobz: u8 = 'N';
    var uplo: u8 = eff.uploChar();
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var info: c_int = undefined;

    const w = try allocator.alloc(R, @max(n, 1));
    errdefer allocator.free(w);

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    if (comptime isComplex(T)) {
        const rwork = try allocator.alloc(R, eigSymRworkLen(n));
        defer allocator.free(rwork);
        xheev(T, &jobz, &uplo, &n_i, am.ptr, &lda, w.ptr, &wq, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigSym: illegal argument to heev query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xheev(T, &jobz, &uplo, &n_i, am.ptr, &lda, w.ptr, work.ptr, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigSym: illegal argument to heev (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    } else {
        xsyev(T, &jobz, &uplo, &n_i, am.ptr, &lda, w.ptr, &wq, &lwork, &info);
        if (info < 0) @panic("eigSym: illegal argument to syev query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xsyev(T, &jobz, &uplo, &n_i, am.ptr, &lda, w.ptr, work.ptr, &lwork, &info);
        if (info < 0) @panic("eigSym: illegal argument to syev (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    }
    return w;
}

/// Input-preserving (default) variant of `eigSymInplace`: computes on a private
/// contiguous copy of `a`, leaving the caller's `a` untouched.
///
/// Input: `a` is left unmodified. Ownership: the caller owns the returned slice;
/// free it with `allocator.free(slice)`. The internal copy of `a` is freed
/// before returning.
pub fn eigSym(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
    tri: Triangle,
) (LapackError || Allocator.Error)![]RealOf(T) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return eigSymInplace(T, Axis, allocator, copy, tri);
}

/// Eigenvalues of a general square matrix (LAPACK `geev`, no vectors).
/// Layout-transparent (spectrum of Aᵀ equals that of A). `a` is overwritten.
/// Returns complex eigenvalues.
///
/// Input: overwrites `a` (scratch). Use `eig` to preserve `a`.
///
/// Ownership: the caller owns the returned slice; free it with
/// `allocator.free(slice)`.
pub fn eigInplace(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
) (LapackError || Allocator.Error)![]Complex(RealOf(T)) {
    comptime assertTwoAxes(Axis);
    const Cx = Complex(RealOf(T));
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("eig: matrix must be square");

    const am = try describe(T, Axis, a, rows, cols);
    var jobvl: u8 = 'N';
    var jobvr: u8 = 'N';
    var n_i: c_int = @intCast(n);
    var lda: c_int = am.lda;
    var one_i: c_int = 1;
    var info: c_int = undefined;
    var lwork: c_int = -1;
    var wq: [1]T = undefined;

    if (comptime isComplex(T)) {
        // Complex geev returns a single complex `w` directly (no wr/wi split);
        // T == Complex(RealOf(T)), so `w` *is* the result buffer.
        const w = try allocator.alloc(T, @max(n, 1));
        errdefer allocator.free(w);
        const rwork = try allocator.alloc(RealOf(T), eigRworkLen(n));
        defer allocator.free(rwork);
        var dummy: [1]T = undefined;
        xgeevc(T, &jobvl, &jobvr, &n_i, am.ptr, &lda, w.ptr, &dummy, &one_i, &dummy, &one_i, &wq, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eig: illegal argument to geev query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xgeevc(T, &jobvl, &jobvr, &n_i, am.ptr, &lda, w.ptr, &dummy, &one_i, &dummy, &one_i, work.ptr, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eig: illegal argument to geev (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
        return w;
    }

    const wr = try allocator.alloc(T, @max(n, 1));
    defer allocator.free(wr);
    const wi = try allocator.alloc(T, @max(n, 1));
    defer allocator.free(wi);
    var dummy: [1]T = undefined;

    xgeev(T, &jobvl, &jobvr, &n_i, am.ptr, &lda, wr.ptr, wi.ptr, &dummy, &one_i, &dummy, &one_i, &wq, &lwork, &info);
    if (info < 0) @panic("eig: illegal argument to geev query (binding bug)");
    lwork = lworkFrom(T, wq[0]);
    const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
    defer allocator.free(work);
    xgeev(T, &jobvl, &jobvr, &n_i, am.ptr, &lda, wr.ptr, wi.ptr, &dummy, &one_i, &dummy, &one_i, work.ptr, &lwork, &info);
    if (info < 0) @panic("eig: illegal argument to geev (binding bug)");
    if (info > 0) return error.ConvergenceFailure;

    const out = try allocator.alloc(Cx, @max(n, 1));
    var i: usize = 0;
    while (i < n) : (i += 1) out[i] = Cx.init(wr[i], wi[i]);
    return out;
}

/// Input-preserving (default) variant of `eigInplace`: computes on a private
/// contiguous copy of `a`, leaving the caller's `a` untouched.
///
/// Input: `a` is left unmodified. Ownership: the caller owns the returned slice;
/// free it with `allocator.free(slice)`. The internal copy of `a` is freed
/// before returning.
pub fn eig(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
) (LapackError || Allocator.Error)![]Complex(RealOf(T)) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return eigInplace(T, Axis, allocator, copy);
}

/// Singular values of a general m×n matrix, descending (LAPACK `gesdd`,
/// `jobz='N'`). Layout-transparent. `a` is overwritten. Singular values are
/// real, so the result element type is `RealOf(T)`.
///
/// Input: overwrites `a` (scratch). Use `svd` to preserve `a`.
///
/// Ownership: the caller owns the returned slice; free it with
/// `allocator.free(slice)`.
pub fn svdInplace(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
) (LapackError || Allocator.Error)![]RealOf(T) {
    comptime assertTwoAxes(Axis);
    const R = RealOf(T);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const m = @field(a.idx.shape, rows);
    const nn = @field(a.idx.shape, cols);

    const am = try describe(T, Axis, a, rows, cols);
    // gesdd wants column-major; a row-major view is the transpose, whose
    // singular values are identical — so hand LAPACK the transposed dims. (For
    // complex, the transpose is unconjugated, but |singular values| are
    // invariant under transpose *and* conjugation, so this is still valid.)
    var m_i: c_int = if (am.layout == .col_major) @intCast(m) else @intCast(nn);
    var n_i: c_int = if (am.layout == .col_major) @intCast(nn) else @intCast(m);
    var jobz: u8 = 'N';
    var lda: c_int = am.lda;
    var ldu: c_int = 1;
    var ldvt: c_int = 1;
    var info: c_int = undefined;
    const mind = @min(m, nn);

    const s = try allocator.alloc(R, @max(mind, 1));
    errdefer allocator.free(s);
    var dummy: [1]T = undefined;
    const iwork = try allocator.alloc(c_int, svdIworkLen(m, nn));
    defer allocator.free(iwork);

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    if (comptime isComplex(T)) {
        const rwork = try allocator.alloc(R, svdRworkLen(m, nn, false));
        defer allocator.free(rwork);
        xgesddc(T, &jobz, &m_i, &n_i, am.ptr, &lda, s.ptr, &dummy, &ldu, &dummy, &ldvt, &wq, &lwork, rwork.ptr, iwork.ptr, &info);
        if (info < 0) @panic("svd: illegal argument to gesdd query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xgesddc(T, &jobz, &m_i, &n_i, am.ptr, &lda, s.ptr, &dummy, &ldu, &dummy, &ldvt, work.ptr, &lwork, rwork.ptr, iwork.ptr, &info);
        if (info < 0) @panic("svd: illegal argument to gesdd (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    } else {
        xgesdd(T, &jobz, &m_i, &n_i, am.ptr, &lda, s.ptr, &dummy, &ldu, &dummy, &ldvt, &wq, &lwork, iwork.ptr, &info);
        if (info < 0) @panic("svd: illegal argument to gesdd query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xgesdd(T, &jobz, &m_i, &n_i, am.ptr, &lda, s.ptr, &dummy, &ldu, &dummy, &ldvt, work.ptr, &lwork, iwork.ptr, &info);
        if (info < 0) @panic("svd: illegal argument to gesdd (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    }
    return s;
}

/// Input-preserving (default) variant of `svdInplace`: computes on a private
/// contiguous copy of `a`, leaving the caller's `a` untouched.
///
/// Input: `a` is left unmodified. Ownership: the caller owns the returned slice;
/// free it with `allocator.free(slice)`. The internal copy of `a` is freed
/// before returning.
pub fn svd(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
) (LapackError || Allocator.Error)![]RealOf(T) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return svdInplace(T, Axis, allocator, copy);
}

// ===== Spectral (with vectors) ===============================================

// Vector-result axes reuse the caller's A row/col labels for each factor's
// component axes, and add a single shared synthesized inner axis (`eig`/`sv`)
// for the eigenvalue/singular index. The per-result axis enums are built inside
// each `*Result` factory below from the caller's `Axis`.

/// Read logical element (i, j) from a described matrix, honoring its layout.
/// Accepts either a mutable (`Descriptor`) or read-only (`DescriptorConst`)
/// descriptor — it only reads.
fn readElem(comptime T: type, am: anytype, i: usize, j: usize) T {
    const lda_us: usize = @intCast(am.lda);
    return switch (am.layout) {
        .col_major => am.ptr[i + j * lda_us],
        .row_major => am.ptr[i * lda_us + j],
    };
}

/// Build a 2-axis `NamedArray` view over `buf` with explicit shape and strides.
fn wrapMat(
    comptime Axis: type,
    comptime S: type,
    buf: []S,
    rows: usize,
    cols: usize,
    srow: isize,
    scol: isize,
) NamedArray(Axis, S) {
    const names = comptime meta.fieldNames(Axis);
    var idx: NamedIndex(Axis) = undefined;
    idx.offset = 0;
    @field(idx.shape, names[0]) = rows;
    @field(idx.shape, names[1]) = cols;
    @field(idx.strides, names[0]) = srow;
    @field(idx.strides, names[1]) = scol;
    return NamedArray(Axis, S).init(idx, buf);
}

fn ColMajorSquare(comptime T: type) type {
    return struct { ptr: [*]T, lda: c_int, owned: ?[]T };
}

/// Normalize a described square matrix to column-major orientation for LAPACK.
/// Column-major is used in place; a dense row-major block is transposed in
/// place (no allocation); a padded row-major view is packed into an owned copy.
fn toColMajorSquare(
    comptime T: type,
    allocator: Allocator,
    am: Descriptor(T),
    n: usize,
) Allocator.Error!ColMajorSquare(T) {
    if (am.layout == .col_major) return .{ .ptr = am.ptr, .lda = am.lda, .owned = null };
    const lda_us: usize = @intCast(am.lda);
    if (lda_us == n) {
        // Dense row-major n×n block ⇒ transpose in place ⇒ column-major A.
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = i + 1;
            while (j < n) : (j += 1) {
                const t = am.ptr[i * n + j];
                am.ptr[i * n + j] = am.ptr[j * n + i];
                am.ptr[j * n + i] = t;
            }
        }
        return .{ .ptr = am.ptr, .lda = am.lda, .owned = null };
    }
    // Padded row-major: pack a dense column-major copy (leaves the caller's A).
    const buf = try allocator.alloc(T, @max(n * n, 1));
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var i: usize = 0;
        while (i < n) : (i += 1) buf[i + j * n] = am.ptr[i * lda_us + j];
    }
    return .{ .ptr = buf.ptr, .lda = @intCast(@max(n, 1)), .owned = buf };
}

/// Assemble a complex n×n eigenvector matrix (column-major) from `geev`'s packed
/// real storage: a real eigenvalue gives one real column; a complex-conjugate
/// pair (wi[j] > 0, wi[j+1] < 0) gives columns `re ± i·im`.
fn assembleEigvecs(
    comptime T: type,
    allocator: Allocator,
    n: usize,
    src: [*]const T,
    wi: []const T,
) Allocator.Error![]Complex(RealOf(T)) {
    const C = Complex(RealOf(T));
    const out = try allocator.alloc(C, @max(n * n, 1));
    errdefer allocator.free(out);
    var j: usize = 0;
    while (j < n) {
        if (wi[j] == 0) {
            var i: usize = 0;
            while (i < n) : (i += 1) out[i + j * n] = C.init(src[i + j * n], 0);
            j += 1;
        } else {
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const re = src[i + j * n];
                const im = src[i + (j + 1) * n];
                out[i + j * n] = C.init(re, im);
                out[i + (j + 1) * n] = C.init(re, -im);
            }
            j += 2;
        }
    }
    return out;
}

pub fn EighResult(comptime Axis: type, comptime T: type) type {
    comptime assertTwoAxes(Axis);
    comptime assertInnerFree(Axis, eig_inner);
    const C = comptime meta.fieldNames(Axis)[1];
    const VecAxis = KeyEnum(&.{ C, eig_inner });
    return struct {
        /// Eigenvalues in ascending order. Real (`RealOf(T)`) even for a complex
        /// Hermitian `A`.
        values: []RealOf(T),
        /// Eigenvectors as columns: column j is the unit eigenvector for
        /// `values[j]`. Shape {C, eig} = n×n, column-major — C is A's column axis
        /// (A·v contracts A's columns), `eig` the synthesized eigenvalue index.
        vectors: NamedArray(VecAxis, T),
        /// Free `values` and `vectors`. Call exactly once, with the same
        /// allocator passed to `eigSymVectors`.
        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.values);
            allocator.free(self.vectors.buf);
        }
    };
}

/// Eigenvalues (ascending) **and** eigenvectors of a symmetric/Hermitian matrix
/// (LAPACK `syev`/`heev`, `jobz='V'`). `tri` selects the referenced triangle.
/// `a` is copied internally and left unmodified. For complex `T` the matrix is
/// treated as **Hermitian**; eigenvalues are real (`RealOf(T)`) and the
/// eigenvectors are complex.
///
/// Input: `a` is left unmodified (copied internally) — inherently
/// input-preserving, so there is no in-place `eigSymVectorsInplace` variant.
/// Ownership: the caller owns the result; free it with `res.deinit(allocator)`
/// (releases both `values` and `vectors`).
pub fn eigSymVectors(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
    tri: Triangle,
) (LapackError || Allocator.Error)!EighResult(Axis, T) {
    comptime assertTwoAxes(Axis);
    const R = RealOf(T);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("eigSymVectors: matrix must be square");

    // `a` is read-only — described purely to copy it into `vbuf` below (never
    // written through), so a const descriptor suffices.
    const am = try describeConst(T, Axis, a, rows, cols);

    // Faithful column-major copy of A (n×n). `syev`/`heev` overwrites it with the
    // eigenvectors, so it doubles as the owned output buffer; because we copy A
    // element-by-element there is no transpose to absorb, so `tri` is used
    // as-is.
    const vbuf = try allocator.alloc(T, @max(n * n, 1));
    errdefer allocator.free(vbuf);
    {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var i: usize = 0;
            while (i < n) : (i += 1) vbuf[i + j * n] = readElem(T, am, i, j);
        }
    }

    const w = try allocator.alloc(R, @max(n, 1));
    errdefer allocator.free(w);

    var jobz: u8 = 'V';
    var uplo: u8 = tri.uploChar();
    var n_i: c_int = @intCast(n);
    var lda: c_int = @intCast(@max(n, 1));
    var info: c_int = undefined;

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    if (comptime isComplex(T)) {
        const rwork = try allocator.alloc(R, eigSymRworkLen(n));
        defer allocator.free(rwork);
        xheev(T, &jobz, &uplo, &n_i, vbuf.ptr, &lda, w.ptr, &wq, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigSymVectors: illegal argument to heev query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xheev(T, &jobz, &uplo, &n_i, vbuf.ptr, &lda, w.ptr, work.ptr, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigSymVectors: illegal argument to heev (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    } else {
        xsyev(T, &jobz, &uplo, &n_i, vbuf.ptr, &lda, w.ptr, &wq, &lwork, &info);
        if (info < 0) @panic("eigSymVectors: illegal argument to syev query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xsyev(T, &jobz, &uplo, &n_i, vbuf.ptr, &lda, w.ptr, work.ptr, &lwork, &info);
        if (info < 0) @panic("eigSymVectors: illegal argument to syev (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    }

    const VecAxis = KeyEnum(&.{ cols, eig_inner });
    return .{
        .values = w,
        .vectors = wrapMat(VecAxis, T, vbuf, n, n, 1, @intCast(@max(n, 1))),
    };
}

// ===== Generalized symmetric/Hermitian-definite eigenproblem ================

/// Which generalized eigenproblem `sygv`/`hegv` solves (LAPACK's `itype`). All
/// three assume A symmetric/Hermitian and B symmetric/Hermitian **positive
/// definite**.
pub const GenEigProblem = enum(c_int) {
    /// A·x = λ·B·x (itype 1) — the standard form. Eigenvectors are B-orthonormal
    /// (Zᴴ·B·Z = I).
    a_bx = 1,
    /// A·B·x = λ·x (itype 2).
    ab_x = 2,
    /// B·A·x = λ·x (itype 3).
    ba_x = 3,
};

/// Interpret the `info` return of `sygv`/`hegv`. `info < 0` is a binding bug
/// (panic). `1 <= info <= n` is an eigenvalue convergence failure. `info > n`
/// means the leading minor of order `info - n` of B is not positive definite
/// (B is not PD, so the problem is not definite).
fn gvInfo(info: c_int, n: c_int) LapackError!void {
    if (info < 0) @panic("eigSymGen: illegal argument to sygv/hegv (binding bug)");
    if (info == 0) return;
    if (info <= n) return error.ConvergenceFailure;
    return error.NotPositiveDefinite;
}

/// Eigenvalues of the symmetric/Hermitian-definite generalized eigenproblem
/// `A·x = λ·B·x` (or the `itype` 2/3 forms), ascending (LAPACK `sygv` for real,
/// `hegv` for complex; `jobz='N'`). B must be positive definite. `tri` selects
/// the referenced triangle of **both** A and B. Both `a` and `b` are used as
/// scratch (overwritten — `a` with intermediate data, `b` with its Cholesky
/// factor). For complex `T`, A and B are treated as **Hermitian**; eigenvalues
/// are real (`RealOf(T)`).
///
/// Input: overwrites both `a` and `b`. Use `eigSymGen` to preserve them.
/// Ownership: the caller owns the returned slice; free it with
/// `allocator.free(slice)`.
pub fn eigSymGenInplace(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
    b: NamedArray(Axis, T),
    problem: GenEigProblem,
    tri: Triangle,
) (LapackError || Allocator.Error)![]RealOf(T) {
    comptime assertTwoAxes(Axis);
    const Re = RealOf(T);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("eigSymGen: A must be square");
    if (@field(b.idx.shape, rows) != n or @field(b.idx.shape, cols) != n) @panic("eigSymGen: B must be square and match A");

    // Normalize both A and B to column-major (transposing a dense row-major block
    // in place gives genuine column-major logical data, so `tri` is used as-is
    // for both). A padded view is packed into an owned copy.
    const am = try describe(T, Axis, a, rows, cols);
    const bm = try describe(T, Axis, b, rows, cols);
    const acm = try toColMajorSquare(T, allocator, am, n);
    defer if (acm.owned) |buf| allocator.free(buf);
    const bcm = try toColMajorSquare(T, allocator, bm, n);
    defer if (bcm.owned) |buf| allocator.free(buf);

    var itype: c_int = @intFromEnum(problem);
    var jobz: u8 = 'N';
    var uplo: u8 = tri.uploChar();
    var n_i: c_int = @intCast(n);
    var lda: c_int = acm.lda;
    var ldb: c_int = bcm.lda;
    var info: c_int = undefined;

    const w = try allocator.alloc(Re, @max(n, 1));
    errdefer allocator.free(w);

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    if (comptime isComplex(T)) {
        const rwork = try allocator.alloc(Re, eigSymRworkLen(n));
        defer allocator.free(rwork);
        xhegv(T, &itype, &jobz, &uplo, &n_i, acm.ptr, &lda, bcm.ptr, &ldb, w.ptr, &wq, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigSymGen: illegal argument to hegv query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xhegv(T, &itype, &jobz, &uplo, &n_i, acm.ptr, &lda, bcm.ptr, &ldb, w.ptr, work.ptr, &lwork, rwork.ptr, &info);
        try gvInfo(info, n_i);
    } else {
        xsygv(T, &itype, &jobz, &uplo, &n_i, acm.ptr, &lda, bcm.ptr, &ldb, w.ptr, &wq, &lwork, &info);
        if (info < 0) @panic("eigSymGen: illegal argument to sygv query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xsygv(T, &itype, &jobz, &uplo, &n_i, acm.ptr, &lda, bcm.ptr, &ldb, w.ptr, work.ptr, &lwork, &info);
        try gvInfo(info, n_i);
    }
    return w;
}

/// Input-preserving (default) variant of `eigSymGenInplace`: computes on private
/// contiguous copies of `a` and `b`, leaving both untouched.
///
/// Input: `a` and `b` are left unmodified. Ownership: the caller owns the
/// returned slice; free it with `allocator.free(slice)`.
pub fn eigSymGen(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
    b: NamedArrayConst(Axis, T),
    problem: GenEigProblem,
    tri: Triangle,
) (LapackError || Allocator.Error)![]RealOf(T) {
    const acopy = try a.toContiguous(allocator);
    defer allocator.free(acopy.buf);
    const bcopy = try b.toContiguous(allocator);
    defer allocator.free(bcopy.buf);
    return eigSymGenInplace(T, Axis, allocator, acopy, bcopy, problem, tri);
}

/// Eigenvalues (ascending) **and** eigenvectors of the symmetric/Hermitian-
/// definite generalized eigenproblem `A·x = λ·B·x` (LAPACK `sygv`/`hegv`,
/// `jobz='V'`). B must be positive definite. `tri` selects the referenced
/// triangle of both A and B. `a` and `b` are copied internally and left
/// unmodified. For `problem = .a_bx` (itype 1) the eigenvectors are
/// B-orthonormal (Zᴴ·B·Z = I); vectors reuse A's column axis: {C, eig}.
///
/// Input: `a` and `b` are left unmodified (copied internally) — inherently
/// input-preserving, so there is no in-place variant (like `eigSymVectors`).
/// Ownership: the caller owns the result; free it with `res.deinit(allocator)`.
pub fn eigSymGenVectors(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
    b: NamedArrayConst(Axis, T),
    problem: GenEigProblem,
    tri: Triangle,
) (LapackError || Allocator.Error)!EighResult(Axis, T) {
    comptime assertTwoAxes(Axis);
    const Re = RealOf(T);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("eigSymGenVectors: A must be square");
    if (@field(b.idx.shape, rows) != n or @field(b.idx.shape, cols) != n) @panic("eigSymGenVectors: B must be square and match A");

    // `a`/`b` are read-only — described purely to copy them below (never written
    // through), so const descriptors suffice.
    const am = try describeConst(T, Axis, a, rows, cols);
    const bm = try describeConst(T, Axis, b, rows, cols);

    // Faithful column-major copies (read logical (i,j), so `tri` is used as-is):
    // A's copy doubles as the eigenvector output; B's copy is overwritten with
    // its Cholesky factor.
    const vbuf = try allocator.alloc(T, @max(n * n, 1));
    errdefer allocator.free(vbuf);
    const bbuf = try allocator.alloc(T, @max(n * n, 1));
    defer allocator.free(bbuf);
    {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var i: usize = 0;
            while (i < n) : (i += 1) {
                vbuf[i + j * n] = readElem(T, am, i, j);
                bbuf[i + j * n] = readElem(T, bm, i, j);
            }
        }
    }

    const w = try allocator.alloc(Re, @max(n, 1));
    errdefer allocator.free(w);

    var itype: c_int = @intFromEnum(problem);
    var jobz: u8 = 'V';
    var uplo: u8 = tri.uploChar();
    var n_i: c_int = @intCast(n);
    var lda: c_int = @intCast(@max(n, 1));
    var ldb: c_int = @intCast(@max(n, 1));
    var info: c_int = undefined;

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    if (comptime isComplex(T)) {
        const rwork = try allocator.alloc(Re, eigSymRworkLen(n));
        defer allocator.free(rwork);
        xhegv(T, &itype, &jobz, &uplo, &n_i, vbuf.ptr, &lda, bbuf.ptr, &ldb, w.ptr, &wq, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigSymGenVectors: illegal argument to hegv query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xhegv(T, &itype, &jobz, &uplo, &n_i, vbuf.ptr, &lda, bbuf.ptr, &ldb, w.ptr, work.ptr, &lwork, rwork.ptr, &info);
        try gvInfo(info, n_i);
    } else {
        xsygv(T, &itype, &jobz, &uplo, &n_i, vbuf.ptr, &lda, bbuf.ptr, &ldb, w.ptr, &wq, &lwork, &info);
        if (info < 0) @panic("eigSymGenVectors: illegal argument to sygv query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xsygv(T, &itype, &jobz, &uplo, &n_i, vbuf.ptr, &lda, bbuf.ptr, &ldb, w.ptr, work.ptr, &lwork, &info);
        try gvInfo(info, n_i);
    }

    const VecAxis = KeyEnum(&.{ cols, eig_inner });
    return .{
        .values = w,
        .vectors = wrapMat(VecAxis, T, vbuf, n, n, 1, @intCast(@max(n, 1))),
    };
}

pub const EigSides = enum { right, left, both };

pub fn EigResult(comptime Axis: type, comptime T: type, comptime sides: EigSides) type {
    comptime assertTwoAxes(Axis);
    comptime assertInnerFree(Axis, eig_inner);
    const Cx = Complex(RealOf(T));
    const Rname = comptime meta.fieldNames(Axis)[0];
    const Cname = comptime meta.fieldNames(Axis)[1];
    // A·vr contracts A's columns ⇒ right vector indexes C; uᴴ·A contracts A's
    // rows ⇒ left vector indexes R. Both share the synthesized `eig` axis.
    const RightAxis = KeyEnum(&.{ Cname, eig_inner });
    const LeftAxis = KeyEnum(&.{ Rname, eig_inner });
    return switch (sides) {
        .right => struct {
            values: []Cx,
            /// Right eigenvectors as columns: A·v_j = values[j]·v_j. Shape {C, eig}.
            right: NamedArray(RightAxis, Cx),
            /// Free `values` and `right`. Call exactly once, with the same
            /// allocator passed to `eigVectors`.
            pub fn deinit(self: @This(), allocator: Allocator) void {
                allocator.free(self.values);
                allocator.free(self.right.buf);
            }
        },
        .left => struct {
            values: []Cx,
            /// Left eigenvectors as columns: u_jᴴ·A = values[j]·u_jᴴ. Shape {R, eig}.
            left: NamedArray(LeftAxis, Cx),
            /// Free `values` and `left`. Call exactly once, with the same
            /// allocator passed to `eigVectors`.
            pub fn deinit(self: @This(), allocator: Allocator) void {
                allocator.free(self.values);
                allocator.free(self.left.buf);
            }
        },
        .both => struct {
            values: []Cx,
            right: NamedArray(RightAxis, Cx),
            left: NamedArray(LeftAxis, Cx),
            /// Free `values`, `right`, and `left`. Call exactly once, with the
            /// same allocator passed to `eigVectors`.
            pub fn deinit(self: @This(), allocator: Allocator) void {
                allocator.free(self.values);
                allocator.free(self.right.buf);
                allocator.free(self.left.buf);
            }
        },
    };
}

/// Eigenvalues **and** eigenvectors of a general square matrix (LAPACK `geev`).
/// `sides` (comptime) selects which of the right/left eigenvectors are returned
/// and shapes the result type accordingly. A is brought to column-major
/// orientation (in-place transpose when needed) and used as scratch; the
/// returned vectors are complex n×n matrices with columns matching `values`.
/// Real scalars only for now.
///
/// Input: overwrites `a` (scratch). Use `eigVectors` to preserve `a`.
/// Ownership: the caller owns the result; free it with `res.deinit(allocator)`
/// (releases `values` and the requested eigenvector matrix/matrices).
pub fn eigVectorsInplace(
    comptime T: type,
    comptime Axis: type,
    comptime sides: EigSides,
    allocator: Allocator,
    a: NamedArray(Axis, T),
) (LapackError || Allocator.Error)!EigResult(Axis, T, sides) {
    comptime assertTwoAxes(Axis);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const n = @field(a.idx.shape, rows);
    if (@field(a.idx.shape, cols) != n) @panic("eigVectors: matrix must be square");

    const am = try describe(T, Axis, a, rows, cols);
    const cm = try toColMajorSquare(T, allocator, am, n);
    defer if (cm.owned) |o| allocator.free(o);

    const C = Complex(RealOf(T));
    const want_r = sides == .right or sides == .both;
    const want_l = sides == .left or sides == .both;

    var jobvl: u8 = if (want_l) 'V' else 'N';
    var jobvr: u8 = if (want_r) 'V' else 'N';
    var n_i: c_int = @intCast(n);
    var lda: c_int = cm.lda;
    var info: c_int = undefined;
    const stride_n: isize = @intCast(@max(n, 1));
    const RightAxis = KeyEnum(&.{ cols, eig_inner });
    const LeftAxis = KeyEnum(&.{ rows, eig_inner });

    if (comptime isComplex(T)) {
        // Complex geev delivers a single complex `w` and complex eigenvector
        // columns directly — no wr/wi split and no real-storage packing (so
        // `assembleEigvecs` is real-only). `C == T` here.
        const values = try allocator.alloc(T, @max(n, 1));
        errdefer allocator.free(values);
        const rwork = try allocator.alloc(RealOf(T), eigRworkLen(n));
        defer allocator.free(rwork);

        // Owned output buffer for each wanted side; a 1-element stub otherwise
        // (LAPACK ignores it but still requires ldv >= 1).
        const vr_buf = try allocator.alloc(T, if (want_r) @max(n * n, 1) else 1);
        errdefer allocator.free(vr_buf);
        const vl_buf = try allocator.alloc(T, if (want_l) @max(n * n, 1) else 1);
        errdefer allocator.free(vl_buf);
        var ldvr: c_int = if (want_r) @intCast(@max(n, 1)) else 1;
        var ldvl: c_int = if (want_l) @intCast(@max(n, 1)) else 1;

        var lwork: c_int = -1;
        var wq: [1]T = undefined;
        xgeevc(T, &jobvl, &jobvr, &n_i, cm.ptr, &lda, values.ptr, vl_buf.ptr, &ldvl, vr_buf.ptr, &ldvr, &wq, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigVectors: illegal argument to geev query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xgeevc(T, &jobvl, &jobvr, &n_i, cm.ptr, &lda, values.ptr, vl_buf.ptr, &ldvl, vr_buf.ptr, &ldvr, work.ptr, &lwork, rwork.ptr, &info);
        if (info < 0) @panic("eigVectors: illegal argument to geev (binding bug)");
        if (info > 0) return error.ConvergenceFailure;

        switch (sides) {
            .right => {
                allocator.free(vl_buf);
                return .{ .values = values, .right = wrapMat(RightAxis, T, vr_buf, n, n, 1, stride_n) };
            },
            .left => {
                allocator.free(vr_buf);
                return .{ .values = values, .left = wrapMat(LeftAxis, T, vl_buf, n, n, 1, stride_n) };
            },
            .both => return .{
                .values = values,
                .right = wrapMat(RightAxis, T, vr_buf, n, n, 1, stride_n),
                .left = wrapMat(LeftAxis, T, vl_buf, n, n, 1, stride_n),
            },
        }
    }

    const wr = try allocator.alloc(T, @max(n, 1));
    defer allocator.free(wr);
    const wi = try allocator.alloc(T, @max(n, 1));
    defer allocator.free(wi);

    // When a side isn't wanted, `jobv?='N'` and LAPACK ignores the buffer, but
    // the leading dimension must still be >= 1, so allocate a 1-element stub.
    const vr_buf = try allocator.alloc(T, if (want_r) @max(n * n, 1) else 1);
    defer allocator.free(vr_buf);
    const vl_buf = try allocator.alloc(T, if (want_l) @max(n * n, 1) else 1);
    defer allocator.free(vl_buf);
    var ldvr: c_int = if (want_r) @intCast(@max(n, 1)) else 1;
    var ldvl: c_int = if (want_l) @intCast(@max(n, 1)) else 1;

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    xgeev(T, &jobvl, &jobvr, &n_i, cm.ptr, &lda, wr.ptr, wi.ptr, vl_buf.ptr, &ldvl, vr_buf.ptr, &ldvr, &wq, &lwork, &info);
    if (info < 0) @panic("eigVectors: illegal argument to geev query (binding bug)");
    lwork = lworkFrom(T, wq[0]);
    const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
    defer allocator.free(work);
    xgeev(T, &jobvl, &jobvr, &n_i, cm.ptr, &lda, wr.ptr, wi.ptr, vl_buf.ptr, &ldvl, vr_buf.ptr, &ldvr, work.ptr, &lwork, &info);
    if (info < 0) @panic("eigVectors: illegal argument to geev (binding bug)");
    if (info > 0) return error.ConvergenceFailure;

    const values = try allocator.alloc(C, @max(n, 1));
    errdefer allocator.free(values);
    {
        var i: usize = 0;
        while (i < n) : (i += 1) values[i] = C.init(wr[i], wi[i]);
    }

    switch (sides) {
        .right => {
            const rc = try assembleEigvecs(T, allocator, n, vr_buf.ptr, wi);
            return .{ .values = values, .right = wrapMat(RightAxis, C, rc, n, n, 1, stride_n) };
        },
        .left => {
            const lc = try assembleEigvecs(T, allocator, n, vl_buf.ptr, wi);
            return .{ .values = values, .left = wrapMat(LeftAxis, C, lc, n, n, 1, stride_n) };
        },
        .both => {
            const rc = try assembleEigvecs(T, allocator, n, vr_buf.ptr, wi);
            errdefer allocator.free(rc);
            const lc = try assembleEigvecs(T, allocator, n, vl_buf.ptr, wi);
            return .{
                .values = values,
                .right = wrapMat(RightAxis, C, rc, n, n, 1, stride_n),
                .left = wrapMat(LeftAxis, C, lc, n, n, 1, stride_n),
            };
        },
    }
}

/// Input-preserving (default) variant of `eigVectorsInplace`: computes on a
/// private contiguous copy of `a`, leaving the caller's `a` untouched.
///
/// Input: `a` is left unmodified. Ownership: the caller owns the result; free it
/// with `res.deinit(allocator)`. The internal copy of `a` is freed before
/// returning.
pub fn eigVectors(
    comptime T: type,
    comptime Axis: type,
    comptime sides: EigSides,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
) (LapackError || Allocator.Error)!EigResult(Axis, T, sides) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return eigVectorsInplace(T, Axis, sides, allocator, copy);
}

pub const SvdMode = enum { thin, full };

pub fn SvdResult(comptime Axis: type, comptime T: type) type {
    comptime assertTwoAxes(Axis);
    comptime assertInnerFree(Axis, svd_inner);
    const R = comptime meta.fieldNames(Axis)[0];
    const C = comptime meta.fieldNames(Axis)[1];
    // Reconstruction A(i,j)=Σₗ U(i,l)·s(l)·Vt(l,j): free row i ⇒ U's row = R;
    // free col j ⇒ Vt's col = C; the summed `sv` is the shared inner axis.
    const UAxis = KeyEnum(&.{ R, svd_inner });
    const VtAxis = KeyEnum(&.{ svd_inner, C });
    return struct {
        /// Singular values in descending order. Length min(m, n). Real
        /// (`RealOf(T)`) even for a complex `A`.
        s: []RealOf(T),
        /// Left singular vectors as columns. {R, sv} = m×(k or m) — R is A's row
        /// axis, `sv` the synthesized singular index.
        u: NamedArray(UAxis, T),
        /// Right singular vectors as rows (Vᴴ for complex, Vᵀ for real).
        /// {sv, C} = (k or n)×n — C is A's column axis. `u` and `vt` share the
        /// `sv` axis, so they compose by name.
        vt: NamedArray(VtAxis, T),
        /// Free `s`, `u`, and `vt`. Call exactly once, with the same allocator
        /// passed to `svdVectors`.
        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.s);
            allocator.free(self.u.buf);
            allocator.free(self.vt.buf);
        }
    };
}

/// Singular values **and** vectors of a general m×n matrix (LAPACK `gesdd`).
/// `mode` picks `.thin` (k = min(m,n) columns/rows) or `.full`. A column-major A
/// is factored natively; a **real** row-major A is factored as its transpose and
/// the resulting U↔V swap is undone by relabeling (no data copy). Singular
/// values are real (`RealOf(T)`); `u`/`vt` carry A's scalar type.
///
/// For **complex** A the row-major reinterpretation is an *unconjugated*
/// transpose, so the pure relabel identities would deliver the SVD of the
/// conjugate. Rather than conjugate the outputs, a complex row-major A is packed
/// into a column-major copy and factored natively (one m×n allocation); a
/// complex column-major A is still factored in place. `a` is used as scratch
/// (overwritten) except for the packed complex-row-major case.
///
/// Input: overwrites `a` (scratch; a complex row-major `a` is left unmodified,
/// factored from a packed copy). Use `svdVectors` to always preserve `a`.
/// Ownership: the caller owns the result; free it with `res.deinit(allocator)`
/// (releases `s`, `u`, and `vt`).
pub fn svdVectorsInplace(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArray(Axis, T),
    mode: SvdMode,
) (LapackError || Allocator.Error)!SvdResult(Axis, T) {
    comptime assertTwoAxes(Axis);
    const R = RealOf(T);
    const rows = comptime meta.fieldNames(Axis)[0];
    const cols = comptime meta.fieldNames(Axis)[1];
    const m = @field(a.idx.shape, rows);
    const nn = @field(a.idx.shape, cols);
    const k = @min(m, nn);

    var am = try describe(T, Axis, a, rows, cols);
    // Complex row-major A: pack a column-major copy so we can factor natively
    // (the relabel trick's transpose is unconjugated — wrong for complex).
    var packed_a: ?[]T = null;
    if (comptime isComplex(T)) {
        if (am.layout != .col_major) {
            const buf = try allocator.alloc(T, @max(m * nn, 1));
            var j: usize = 0;
            while (j < nn) : (j += 1) {
                var i: usize = 0;
                while (i < m) : (i += 1) buf[i + j * m] = readElem(T, am, i, j);
            }
            packed_a = buf;
            am = .{ .layout = .col_major, .m = @intCast(m), .n = @intCast(nn), .lda = @intCast(@max(m, 1)), .ptr = buf.ptr };
        }
    }
    defer if (packed_a) |b| allocator.free(b);

    const transposed = am.layout != .col_major;
    const full = mode == .full;

    // Dimensions of the problem handed to gesdd: A when column-major, else Aᵀ.
    const pm: usize = if (transposed) nn else m;
    const pn: usize = if (transposed) m else nn;
    const u_cols_p: usize = if (full) pm else k; // columns of U(P)
    const vt_rows_p: usize = if (full) pn else k; // rows of Vt(P)

    var jobz: u8 = if (full) 'A' else 'S';
    var pm_i: c_int = @intCast(@max(pm, 1));
    var pn_i: c_int = @intCast(@max(pn, 1));
    var lda: c_int = am.lda;
    var ldu: c_int = @intCast(@max(pm, 1));
    var ldvt: c_int = @intCast(@max(vt_rows_p, 1));
    var info: c_int = undefined;

    const s = try allocator.alloc(R, @max(k, 1));
    errdefer allocator.free(s);
    const ubuf = try allocator.alloc(T, @max(pm * u_cols_p, 1));
    errdefer allocator.free(ubuf);
    const vtbuf = try allocator.alloc(T, @max(vt_rows_p * pn, 1));
    errdefer allocator.free(vtbuf);
    const iwork = try allocator.alloc(c_int, svdIworkLen(m, nn));
    defer allocator.free(iwork);

    var lwork: c_int = -1;
    var wq: [1]T = undefined;
    if (comptime isComplex(T)) {
        const rwork = try allocator.alloc(R, svdRworkLen(m, nn, true));
        defer allocator.free(rwork);
        xgesddc(T, &jobz, &pm_i, &pn_i, am.ptr, &lda, s.ptr, ubuf.ptr, &ldu, vtbuf.ptr, &ldvt, &wq, &lwork, rwork.ptr, iwork.ptr, &info);
        if (info < 0) @panic("svdVectors: illegal argument to gesdd query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xgesddc(T, &jobz, &pm_i, &pn_i, am.ptr, &lda, s.ptr, ubuf.ptr, &ldu, vtbuf.ptr, &ldvt, work.ptr, &lwork, rwork.ptr, iwork.ptr, &info);
        if (info < 0) @panic("svdVectors: illegal argument to gesdd (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    } else {
        xgesdd(T, &jobz, &pm_i, &pn_i, am.ptr, &lda, s.ptr, ubuf.ptr, &ldu, vtbuf.ptr, &ldvt, &wq, &lwork, iwork.ptr, &info);
        if (info < 0) @panic("svdVectors: illegal argument to gesdd query (binding bug)");
        lwork = lworkFrom(T, wq[0]);
        const work = try allocator.alloc(T, @intCast(@max(lwork, 1)));
        defer allocator.free(work);
        xgesdd(T, &jobz, &pm_i, &pn_i, am.ptr, &lda, s.ptr, ubuf.ptr, &ldu, vtbuf.ptr, &ldvt, work.ptr, &lwork, iwork.ptr, &info);
        if (info < 0) @panic("svdVectors: illegal argument to gesdd (binding bug)");
        if (info > 0) return error.ConvergenceFailure;
    }

    const ldu_s: isize = @intCast(@max(pm, 1));
    const ldvt_s: isize = @intCast(@max(vt_rows_p, 1));
    const UAxis = KeyEnum(&.{ rows, svd_inner });
    const VtAxis = KeyEnum(&.{ svd_inner, cols });
    if (!transposed) {
        // Native: U(A)=ubuf (m×u_cols_p), Vt(A)=vtbuf (vt_rows_p×n), column-major.
        return .{
            .s = s,
            .u = wrapMat(UAxis, T, ubuf, m, u_cols_p, 1, ldu_s),
            .vt = wrapMat(VtAxis, T, vtbuf, vt_rows_p, nn, 1, ldvt_s),
        };
    }
    // Transposed problem: U(A) = (Vt(Aᵀ))ᵀ = vtbuf relabeled; Vt(A) =
    // (U(Aᵀ))ᵀ = ubuf relabeled. Both are pure stride swaps (no data move):
    //   u(i,l)  = vtbuf[l + i·ldvt] ⇒ shape (m, vt_rows_p), strides (ldvt, 1)
    //   vt(l,j) = ubuf[j + l·ldu]   ⇒ shape (u_cols_p, nn), strides (ldu, 1)
    return .{
        .s = s,
        .u = wrapMat(UAxis, T, vtbuf, m, vt_rows_p, ldvt_s, 1),
        .vt = wrapMat(VtAxis, T, ubuf, u_cols_p, nn, ldu_s, 1),
    };
}

/// Input-preserving (default) variant of `svdVectorsInplace`: computes on a
/// private contiguous copy of `a`, leaving the caller's `a` untouched.
///
/// Input: `a` is left unmodified. Ownership: the caller owns the result; free it
/// with `res.deinit(allocator)`. The internal copy of `a` is freed before
/// returning.
pub fn svdVectors(
    comptime T: type,
    comptime Axis: type,
    allocator: Allocator,
    a: NamedArrayConst(Axis, T),
    mode: SvdMode,
) (LapackError || Allocator.Error)!SvdResult(Axis, T) {
    const copy = try a.toContiguous(allocator);
    defer allocator.free(copy.buf);
    return svdVectorsInplace(T, Axis, allocator, copy, mode);
}

// ===== misc ==================================================================

/// Convert an lwork query result (returned as a real scalar in work[0]) to an
/// integer length.
fn lworkFrom(comptime T: type, v: T) c_int {
    const r = if (comptime isComplex(T)) v.re else v;
    return @intFromFloat(@round(r));
}

test {
    std.testing.refAllDecls(@This());
}

// ===== Tests =================================================================

const testing = std.testing;
const IJ = enum { i, j };
const IK = enum { i, k };
const JK = enum { j, k };

fn rowMajor2x2(buf: *[4]f64) NamedArray(IJ, f64) {
    const idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 2 });
    return NamedArray(IJ, f64).init(idx, buf);
}

fn colMajor(comptime Axis: type, buf: []f64, rows: usize, cols: usize) NamedArray(Axis, f64) {
    const names = comptime std.meta.fieldNames(Axis);
    var idx: NamedIndex(Axis) = undefined;
    idx.offset = 0;
    @field(idx.shape, names[0]) = rows;
    @field(idx.shape, names[1]) = cols;
    @field(idx.strides, names[0]) = 1;
    @field(idx.strides, names[1]) = @intCast(rows);
    return NamedArray(Axis, f64).init(idx, buf);
}

test "solve: row-major A, vector RHS" {
    // A = [[1,2],[3,4]], b = [5,6] -> x = [-4, 4.5]
    var abuf = [_]f64{ 1, 2, 3, 4 };
    const a = rowMajor2x2(&abuf);
    var bbuf = [_]f64{ 5, 6 };
    const b = NamedArray(IK, f64).init(NamedIndex(IK).initContiguous(.{ .i = 2, .k = 1 }), &bbuf);
    var ipiv: [2]c_int = undefined;
    const x = try solve(f64, IJ, IK, a, b, &ipiv);
    try testing.expectApproxEqAbs(@as(f64, -4.0), x.at(.{ .j = 0, .k = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.5), x.at(.{ .j = 1, .k = 0 }).*, 1e-12);
}

test "solve: column-major A matches row-major" {
    // Same A = [[1,2],[3,4]] stored column-major: [1,3,2,4].
    var abuf = [_]f64{ 1, 3, 2, 4 };
    const a = colMajor(IJ, &abuf, 2, 2);
    var bbuf = [_]f64{ 5, 6 };
    const b = NamedArray(IK, f64).init(NamedIndex(IK).initContiguous(.{ .i = 2, .k = 1 }), &bbuf);
    var ipiv: [2]c_int = undefined;
    const x = try solve(f64, IJ, IK, a, b, &ipiv);
    try testing.expectApproxEqAbs(@as(f64, -4.0), x.at(.{ .j = 0, .k = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.5), x.at(.{ .j = 1, .k = 0 }).*, 1e-12);
}

test "solve: multi-column column-major RHS" {
    // A = [[2,1],[1,3]], B columns = [1,2] and [0,5].
    var abuf = [_]f64{ 2, 1, 1, 3 }; // symmetric, row==col major
    const a = rowMajor2x2(&abuf);
    // B column-major 2x2: col0=[1,2], col1=[0,5] -> buf [1,2,0,5]
    var bbuf = [_]f64{ 1, 2, 0, 5 };
    const b = colMajor(IK, &bbuf, 2, 2);
    var ipiv: [2]c_int = undefined;
    const x = try solve(f64, IJ, IK, a, b, &ipiv);
    // A x0 = [1,2] -> x0 = [0.2, 0.6]; A x1 = [0,5] -> x1 = [-1, 2]
    try testing.expectApproxEqAbs(@as(f64, 0.2), x.at(.{ .j = 0, .k = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.6), x.at(.{ .j = 1, .k = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -1.0), x.at(.{ .j = 0, .k = 1 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 1 }).*, 1e-12);
}

test "solve: singular A returns error" {
    var abuf = [_]f64{ 1, 2, 2, 4 }; // rank 1
    const a = rowMajor2x2(&abuf);
    var bbuf = [_]f64{ 1, 1 };
    const b = NamedArray(IK, f64).init(NamedIndex(IK).initContiguous(.{ .i = 2, .k = 1 }), &bbuf);
    var ipiv: [2]c_int = undefined;
    try testing.expectError(error.Singular, solve(f64, IJ, IK, a, b, &ipiv));
}

test "lu + luSolve reproduce solve" {
    var abuf = [_]f64{ 1, 2, 3, 4 };
    const a = rowMajor2x2(&abuf);
    var ipiv: [2]c_int = undefined;
    try lu(f64, IJ, a, &ipiv);
    var bbuf = [_]f64{ 5, 6 };
    const b = NamedArray(IK, f64).init(NamedIndex(IK).initContiguous(.{ .i = 2, .k = 1 }), &bbuf);
    const x = try luSolve(f64, IJ, IK, a.asConst(), &ipiv, b);
    try testing.expectApproxEqAbs(@as(f64, -4.0), x.at(.{ .j = 0, .k = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.5), x.at(.{ .j = 1, .k = 0 }).*, 1e-12);
}

test "detInplace: 2x2 and 3x3" {
    var abuf = [_]f64{ 1, 2, 3, 4 };
    const a = rowMajor2x2(&abuf);
    var ipiv: [3]c_int = undefined;
    try testing.expectApproxEqAbs(@as(f64, -2.0), detInplace(f64, IJ, a, &ipiv), 1e-12);

    // 3x3 with det = 1*(5*9-6*8) - 2*(4*9-6*7) + 3*(4*8-5*7) = -3+12-9 = 0? use invertible.
    var b3 = [_]f64{ 2, 0, 1, 0, 3, 0, 1, 0, 4 }; // det = 2*(12) -0 +1*(0-3) = 24-3 = 21
    const I3 = enum { i, j };
    const a3 = NamedArray(I3, f64).init(NamedIndex(I3).initContiguous(.{ .i = 3, .j = 3 }), &b3);
    try testing.expectApproxEqAbs(@as(f64, 21.0), detInplace(f64, I3, a3, &ipiv), 1e-10);
}

test "inv: in place, row-major" {
    var abuf = [_]f64{ 1, 2, 3, 4 };
    const a = rowMajor2x2(&abuf);
    try inv(f64, IJ, testing.allocator, a);
    // inv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]
    try testing.expectApproxEqAbs(@as(f64, -2.0), a.at(.{ .i = 0, .j = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), a.at(.{ .i = 0, .j = 1 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.5), a.at(.{ .i = 1, .j = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.5), a.at(.{ .i = 1, .j = 1 }).*, 1e-12);
}

test "cholesky + choleskySolve" {
    // A = [[4,2],[2,3]] SPD, b = [1,1] -> x = [0.125, 0.25]
    var abuf = [_]f64{ 4, 2, 2, 3 };
    const a = rowMajor2x2(&abuf);
    try cholesky(f64, IJ, a, .lower);
    var bbuf = [_]f64{ 1, 1 };
    const b = NamedArray(IK, f64).init(NamedIndex(IK).initContiguous(.{ .i = 2, .k = 1 }), &bbuf);
    const x = try choleskySolve(f64, IJ, IK, a.asConst(), b, .lower);
    try testing.expectApproxEqAbs(@as(f64, 0.125), x.at(.{ .j = 0, .k = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.25), x.at(.{ .j = 1, .k = 0 }).*, 1e-12);
}

test "cholesky: not positive definite" {
    var abuf = [_]f64{ 1, 2, 2, 1 }; // indefinite
    const a = rowMajor2x2(&abuf);
    try testing.expectError(error.NotPositiveDefinite, cholesky(f64, IJ, a, .lower));
}

test "eigSymInplace: symmetric eigenvalues ascending" {
    // [[2,1],[1,2]] -> eigenvalues 1, 3
    var abuf = [_]f64{ 2, 1, 1, 2 };
    const a = rowMajor2x2(&abuf);
    const w = try eigSymInplace(f64, IJ, testing.allocator, a, .upper);
    defer testing.allocator.free(w);
    try testing.expectApproxEqAbs(@as(f64, 1.0), w[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), w[1], 1e-12);
}

test "svdInplace: singular values descending" {
    // diag(2,3) -> singular values 3, 2
    var abuf = [_]f64{ 2, 0, 0, 3 };
    const a = rowMajor2x2(&abuf);
    const s = try svdInplace(f64, IJ, testing.allocator, a);
    defer testing.allocator.free(s);
    try testing.expectApproxEqAbs(@as(f64, 3.0), s[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), s[1], 1e-12);
}

test "eigInplace: complex eigenvalues of a rotation" {
    // [[0,-1],[1,0]] -> eigenvalues +-i
    var abuf = [_]f64{ 0, -1, 1, 0 };
    const a = rowMajor2x2(&abuf);
    const w = try eigInplace(f64, IJ, testing.allocator, a);
    defer testing.allocator.free(w);
    // Both eigenvalues have zero real part and imaginary parts +-1.
    try testing.expectApproxEqAbs(@as(f64, 0.0), w[0].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), w[1].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), @abs(w[0].im), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), @abs(w[1].im), 1e-12);
    try testing.expect(w[0].im * w[1].im < 0); // conjugate pair
}

test "qrInplace: reconstructs A and Q is orthonormal" {
    // A = [[1,2],[3,4]] column-major.
    var abuf = [_]f64{ 1, 3, 2, 4 };
    const a = colMajor(IJ, &abuf, 2, 2);
    const res = try qrInplace(f64, IJ, testing.allocator, a);
    defer res.deinit(testing.allocator);
    // Reconstruct A[i,j] = sum_k Q[i,k] R[k,j].
    const orig = [_][2]f64{ .{ 1, 2 }, .{ 3, 4 } };
    var ii: usize = 0;
    while (ii < 2) : (ii += 1) {
        var jj: usize = 0;
        while (jj < 2) : (jj += 1) {
            var acc: f64 = 0;
            var kk: usize = 0;
            while (kk < 2) : (kk += 1) {
                acc += res.q.at(.{ .i = ii, .qr_rank = kk }).* * res.r.at(.{ .qr_rank = kk, .j = jj }).*;
            }
            try testing.expectApproxEqAbs(orig[ii][jj], acc, 1e-10);
        }
    }
    // Q^T Q = I.
    var c0: f64 = 0;
    var c1: f64 = 0;
    var cross: f64 = 0;
    var ii2: usize = 0;
    while (ii2 < 2) : (ii2 += 1) {
        c0 += std.math.pow(f64, res.q.at(.{ .i = ii2, .qr_rank = 0 }).*, 2);
        c1 += std.math.pow(f64, res.q.at(.{ .i = ii2, .qr_rank = 1 }).*, 2);
        cross += res.q.at(.{ .i = ii2, .qr_rank = 0 }).* * res.q.at(.{ .i = ii2, .qr_rank = 1 }).*;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), c0, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), c1, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cross, 1e-10);
}

test "lstsqInplace: overdetermined exact fit" {
    // A = [[1,0],[0,1],[1,1]] (3x2) column-major; b = [1,2,3] -> x = [1,2].
    var abuf = [_]f64{ 1, 0, 1, 0, 1, 1 }; // col0=[1,0,1], col1=[0,1,1]
    const a = colMajor(IJ, &abuf, 3, 2);
    var bbuf = [_]f64{ 1, 2, 3 };
    const b = colMajor(IK, &bbuf, 3, 1);
    const x = try lstsqInplace(f64, IJ, IK, testing.allocator, a, b);
    // The returned view X = {j, k} aliases b's first n rows.
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 0, .k = 0 }).*, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 0 }).*, 1e-10);
}

test "solve: complex system" {
    // A = [[1+0i, 0],[0, 2i]], b = [2, 4i] -> x = [2, 2].
    const C = Complex(f64);
    var abuf = [_]C{ C.init(1, 0), C.init(0, 0), C.init(0, 0), C.init(0, 2) };
    const a = NamedArray(IJ, C).init(NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 2 }), &abuf);
    var bbuf = [_]C{ C.init(2, 0), C.init(0, 4) };
    const b = NamedArray(IK, C).init(NamedIndex(IK).initContiguous(.{ .i = 2, .k = 1 }), &bbuf);
    var ipiv: [2]c_int = undefined;
    const x = try solve(C, IJ, IK, a, b, &ipiv);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 0, .k = 0 }).re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), x.at(.{ .j = 0, .k = 0 }).im, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 0 }).re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), x.at(.{ .j = 1, .k = 0 }).im, 1e-12);
}

test "eigSymVectors: A v = lambda v, unit norm, A preserved" {
    // [[2,1],[1,2]] -> eigenvalues 1, 3.
    var abuf = [_]f64{ 2, 1, 1, 2 };
    const a = rowMajor2x2(&abuf);
    const res = try eigSymVectors(f64, IJ, testing.allocator, a.asConst(), .upper);
    defer res.deinit(testing.allocator);
    try testing.expectApproxEqAbs(@as(f64, 1.0), res.values[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), res.values[1], 1e-12);
    // A is copied internally, so the caller's buffer is untouched.
    try testing.expectEqual(@as(f64, 2.0), abuf[0]);
    const A = [_][2]f64{ .{ 2, 1 }, .{ 1, 2 } };
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const v0 = res.vectors.at(.{ .j = 0, .eig = jc }).*;
        const v1 = res.vectors.at(.{ .j = 1, .eig = jc }).*;
        const lam = res.values[jc];
        try testing.expectApproxEqAbs(lam * v0, A[0][0] * v0 + A[0][1] * v1, 1e-10);
        try testing.expectApproxEqAbs(lam * v1, A[1][0] * v0 + A[1][1] * v1, 1e-10);
        try testing.expectApproxEqAbs(@as(f64, 1.0), v0 * v0 + v1 * v1, 1e-10);
    }
}

test "eigVectorsInplace: right eigenvectors satisfy A v = lambda v" {
    // A = [[2,0],[1,3]] (row-major) -> eigenvalues 2, 3 (real).
    var abuf = [_]f64{ 2, 0, 1, 3 };
    const a = rowMajor2x2(&abuf);
    const res = try eigVectorsInplace(f64, IJ, .right, testing.allocator, a);
    defer res.deinit(testing.allocator);
    const A = [_][2]f64{ .{ 2, 0 }, .{ 1, 3 } };
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const v0 = res.right.at(.{ .j = 0, .eig = jc }).*;
        const v1 = res.right.at(.{ .j = 1, .eig = jc }).*;
        const lam = res.values[jc];
        try testing.expectApproxEqAbs(@as(f64, 0.0), lam.im, 1e-12);
        try testing.expectApproxEqAbs(lam.re * v0.re, A[0][0] * v0.re + A[0][1] * v1.re, 1e-10);
        try testing.expectApproxEqAbs(lam.re * v1.re, A[1][0] * v0.re + A[1][1] * v1.re, 1e-10);
    }
}

test "eigVectorsInplace: complex-conjugate eigenvectors of a rotation" {
    // [[0,-1],[1,0]] -> eigenvalues +-i, complex eigenvectors.
    var abuf = [_]f64{ 0, -1, 1, 0 };
    const a = rowMajor2x2(&abuf);
    const res = try eigVectorsInplace(f64, IJ, .right, testing.allocator, a);
    defer res.deinit(testing.allocator);
    const C = Complex(f64);
    const A = [_][2]C{
        .{ C.init(0, 0), C.init(-1, 0) },
        .{ C.init(1, 0), C.init(0, 0) },
    };
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const v0 = res.right.at(.{ .j = 0, .eig = jc }).*;
        const v1 = res.right.at(.{ .j = 1, .eig = jc }).*;
        const lam = res.values[jc];
        const av0 = A[0][0].mul(v0).add(A[0][1].mul(v1));
        const av1 = A[1][0].mul(v0).add(A[1][1].mul(v1));
        const lv0 = lam.mul(v0);
        const lv1 = lam.mul(v1);
        try testing.expectApproxEqAbs(lv0.re, av0.re, 1e-10);
        try testing.expectApproxEqAbs(lv0.im, av0.im, 1e-10);
        try testing.expectApproxEqAbs(lv1.re, av1.re, 1e-10);
        try testing.expectApproxEqAbs(lv1.im, av1.im, 1e-10);
    }
}

test "eigVectorsInplace: both sides, left eigenvectors satisfy A^T u = lambda u" {
    // A = [[2,0],[1,3]], real spectrum 2, 3; left eigenvectors solve Aᵀu = λu.
    var abuf = [_]f64{ 2, 0, 1, 3 };
    const a = rowMajor2x2(&abuf);
    const res = try eigVectorsInplace(f64, IJ, .both, testing.allocator, a);
    defer res.deinit(testing.allocator);
    const AT = [_][2]f64{ .{ 2, 1 }, .{ 0, 3 } }; // transpose of [[2,0],[1,3]]
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const uu0 = res.left.at(.{ .i = 0, .eig = jc }).*;
        const uu1 = res.left.at(.{ .i = 1, .eig = jc }).*;
        const lam = res.values[jc];
        try testing.expectApproxEqAbs(lam.re * uu0.re, AT[0][0] * uu0.re + AT[0][1] * uu1.re, 1e-10);
        try testing.expectApproxEqAbs(lam.re * uu1.re, AT[1][0] * uu0.re + AT[1][1] * uu1.re, 1e-10);
    }
    // Right vectors are present too.
    _ = res.right.at(.{ .j = 0, .eig = 0 });
}

test "svdVectorsInplace: thin reconstruction, square row-major" {
    var abuf = [_]f64{ 2, 0, 0, 3 };
    const a = rowMajor2x2(&abuf);
    const A = [_][2]f64{ .{ 2, 0 }, .{ 0, 3 } };
    const res = try svdVectorsInplace(f64, IJ, testing.allocator, a, .thin);
    defer res.deinit(testing.allocator);
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc: f64 = 0;
            var l: usize = 0;
            while (l < 2) : (l += 1)
                acc += res.u.at(.{ .i = i, .sv = l }).* * res.s[l] * res.vt.at(.{ .sv = l, .j = j }).*;
            try testing.expectApproxEqAbs(A[i][j], acc, 1e-10);
        }
    }
}

test "svdVectorsInplace: full mode, square row-major" {
    var abuf = [_]f64{ 2, 0, 0, 3 };
    const a = rowMajor2x2(&abuf);
    const A = [_][2]f64{ .{ 2, 0 }, .{ 0, 3 } };
    const res = try svdVectorsInplace(f64, IJ, testing.allocator, a, .full);
    defer res.deinit(testing.allocator);
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc: f64 = 0;
            var l: usize = 0;
            while (l < 2) : (l += 1)
                acc += res.u.at(.{ .i = i, .sv = l }).* * res.s[l] * res.vt.at(.{ .sv = l, .j = j }).*;
            try testing.expectApproxEqAbs(A[i][j], acc, 1e-10);
        }
    }
}

test "svdVectorsInplace: thin rectangular, column-major (native path)" {
    // A 3x2 column-major: col0=[1,0,1], col1=[0,1,1].
    var abuf = [_]f64{ 1, 0, 1, 0, 1, 1 };
    const a = colMajor(IJ, &abuf, 3, 2);
    const A = [_][2]f64{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } };
    const res = try svdVectorsInplace(f64, IJ, testing.allocator, a, .thin);
    defer res.deinit(testing.allocator);
    // thin shapes: U 3x2, Vt 2x2, s len 2.
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc: f64 = 0;
            var l: usize = 0;
            while (l < 2) : (l += 1)
                acc += res.u.at(.{ .i = i, .sv = l }).* * res.s[l] * res.vt.at(.{ .sv = l, .j = j }).*;
            try testing.expectApproxEqAbs(A[i][j], acc, 1e-10);
        }
    }
}

test "svdVectorsInplace: thin rectangular, row-major (output-swap path)" {
    // A 2x3 row-major -> exercises the transposed relabel with m < n.
    var abuf = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const IJ2 = enum { i, j };
    const a = NamedArray(IJ2, f64).init(NamedIndex(IJ2).initContiguous(.{ .i = 2, .j = 3 }), &abuf);
    const A = [_][3]f64{ .{ 1, 2, 3 }, .{ 4, 5, 6 } };
    const res = try svdVectorsInplace(f64, IJ2, testing.allocator, a, .thin);
    defer res.deinit(testing.allocator);
    // thin shapes: m=2, n=3, k=2 -> U 2x2, Vt 2x3, s len 2.
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 3) : (j += 1) {
            var acc: f64 = 0;
            var l: usize = 0;
            while (l < 2) : (l += 1)
                acc += res.u.at(.{ .i = i, .sv = l }).* * res.s[l] * res.vt.at(.{ .sv = l, .j = j }).*;
            try testing.expectApproxEqAbs(A[i][j], acc, 1e-9);
        }
    }
}

test "lstsqInplace: overdetermined exact fit, row-major A (trans trick)" {
    // A = [[1,0],[0,1],[1,1]] (3x2) row-major; b = [1,2,3] -> x = [1,2].
    const IJ3 = enum { i, j };
    var abuf = [_]f64{ 1, 0, 0, 1, 1, 1 };
    const a = NamedArray(IJ3, f64).init(NamedIndex(IJ3).initContiguous(.{ .i = 3, .j = 2 }), &abuf);
    var bbuf = [_]f64{ 1, 2, 3 };
    const b = colMajor(IK, &bbuf, 3, 1);
    const x = try lstsqInplace(f64, IJ3, IK, testing.allocator, a, b);
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 0, .k = 0 }).*, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 0 }).*, 1e-10);
}

test "lstsqInplace: row-major and column-major A agree" {
    // Overdetermined 3x2 with a non-exact fit; row-major and column-major inputs
    // must produce the same least-squares solution.
    const IJ3 = enum { i, j };
    // A rows: [1,1],[1,2],[1,3]; b = [1,2,2] -> line fit.
    var rbuf = [_]f64{ 1, 1, 1, 2, 1, 3 }; // row-major
    const ar = NamedArray(IJ3, f64).init(NamedIndex(IJ3).initContiguous(.{ .i = 3, .j = 2 }), &rbuf);
    var brbuf = [_]f64{ 1, 2, 2 };
    const br = colMajor(IK, &brbuf, 3, 1);
    const xr = try lstsqInplace(f64, IJ3, IK, testing.allocator, ar, br);

    var cbuf = [_]f64{ 1, 1, 1, 1, 2, 3 }; // same A column-major: col0=1s, col1=[1,2,3]
    const ac = colMajor(IJ3, &cbuf, 3, 2);
    var bcbuf = [_]f64{ 1, 2, 2 };
    const bc = colMajor(IK, &bcbuf, 3, 1);
    const xc = try lstsqInplace(f64, IJ3, IK, testing.allocator, ac, bc);

    try testing.expectApproxEqAbs(xr.at(.{ .j = 0, .k = 0 }).*, xc.at(.{ .j = 0, .k = 0 }).*, 1e-10);
    try testing.expectApproxEqAbs(xr.at(.{ .j = 1, .k = 0 }).*, xc.at(.{ .j = 1, .k = 0 }).*, 1e-10);
}

test "qrInplace: row-major A is packed, reconstructs A, leaves A intact" {
    // A = [[1,2],[3,4]] row-major.
    var abuf = [_]f64{ 1, 2, 3, 4 };
    const a = rowMajor2x2(&abuf);
    const res = try qrInplace(f64, IJ, testing.allocator, a);
    defer res.deinit(testing.allocator);
    // The row-major path copies A, so the caller's buffer is untouched.
    try testing.expectEqual(@as(f64, 1.0), abuf[0]);
    try testing.expectEqual(@as(f64, 4.0), abuf[3]);
    const orig = [_][2]f64{ .{ 1, 2 }, .{ 3, 4 } };
    var ii: usize = 0;
    while (ii < 2) : (ii += 1) {
        var jj: usize = 0;
        while (jj < 2) : (jj += 1) {
            var acc: f64 = 0;
            var kk: usize = 0;
            while (kk < 2) : (kk += 1)
                acc += res.q.at(.{ .i = ii, .qr_rank = kk }).* * res.r.at(.{ .qr_rank = kk, .j = jj }).*;
            try testing.expectApproxEqAbs(orig[ii][jj], acc, 1e-10);
        }
    }
}

test "qrInplace: tall row-major A, reconstruction and orthonormal Q" {
    const IJ3 = enum { i, j };
    // A 3x2 = [[1,1],[1,0],[0,1]] row-major.
    var abuf = [_]f64{ 1, 1, 1, 0, 0, 1 };
    const a = NamedArray(IJ3, f64).init(NamedIndex(IJ3).initContiguous(.{ .i = 3, .j = 2 }), &abuf);
    const orig = [_][2]f64{ .{ 1, 1 }, .{ 1, 0 }, .{ 0, 1 } };
    const res = try qrInplace(f64, IJ3, testing.allocator, a);
    defer res.deinit(testing.allocator);
    // Reconstruct A[i,j] = sum_k Q[i,k] R[k,j] (Q is 3x2, R is 2x2).
    var ii: usize = 0;
    while (ii < 3) : (ii += 1) {
        var jj: usize = 0;
        while (jj < 2) : (jj += 1) {
            var acc: f64 = 0;
            var kk: usize = 0;
            while (kk < 2) : (kk += 1)
                acc += res.q.at(.{ .i = ii, .qr_rank = kk }).* * res.r.at(.{ .qr_rank = kk, .j = jj }).*;
            try testing.expectApproxEqAbs(orig[ii][jj], acc, 1e-10);
        }
    }
    // Q^T Q = I (2x2).
    var cc: usize = 0;
    while (cc < 2) : (cc += 1) {
        var d: usize = 0;
        while (d < 2) : (d += 1) {
            var acc: f64 = 0;
            var ii2: usize = 0;
            while (ii2 < 3) : (ii2 += 1)
                acc += res.q.at(.{ .i = ii2, .qr_rank = cc }).* * res.q.at(.{ .i = ii2, .qr_rank = d }).*;
            const expected: f64 = if (cc == d) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, acc, 1e-10);
        }
    }
}

// ===== Input-preserving *Alloc conveniences ==================================

test "det: preserves A and matches det" {
    var abuf = [_]f64{ 1, 3, 2, 4 }; // column-major [[1,2],[3,4]], det = -2
    const a = colMajor(IJ, &abuf, 2, 2);
    const d = try det(f64, IJ, testing.allocator, a.asConst());
    try testing.expectApproxEqAbs(@as(f64, -2.0), d, 1e-12);
    // A untouched.
    try testing.expectEqualSlices(f64, &.{ 1, 3, 2, 4 }, &abuf);
}

test "eigSym: preserves A, matches eigSym" {
    var abuf = [_]f64{ 2, 1, 1, 2 };
    const a = rowMajor2x2(&abuf);
    const w = try eigSym(f64, IJ, testing.allocator, a.asConst(), .upper);
    defer testing.allocator.free(w);
    try testing.expectApproxEqAbs(@as(f64, 1.0), w[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), w[1], 1e-12);
    try testing.expectEqualSlices(f64, &.{ 2, 1, 1, 2 }, &abuf);
}

test "eig: preserves A" {
    var abuf = [_]f64{ 0, -1, 1, 0 }; // rotation, eigenvalues +-i
    const a = rowMajor2x2(&abuf);
    const vals = try eig(f64, IJ, testing.allocator, a.asConst());
    defer testing.allocator.free(vals);
    try testing.expectApproxEqAbs(@as(f64, 1.0), @abs(vals[0].im), 1e-12);
    try testing.expectEqualSlices(f64, &.{ 0, -1, 1, 0 }, &abuf);
}

test "svd: preserves A, matches svd" {
    var abuf = [_]f64{ 2, 0, 0, 3 };
    const a = rowMajor2x2(&abuf);
    const s = try svd(f64, IJ, testing.allocator, a.asConst());
    defer testing.allocator.free(s);
    try testing.expectApproxEqAbs(@as(f64, 3.0), s[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), s[1], 1e-12);
    try testing.expectEqualSlices(f64, &.{ 2, 0, 0, 3 }, &abuf);
}

test "lstsq: preserves A, solves" {
    // A = [[1,0],[0,1],[1,1]] (3x2) column-major; b = [1,2,3] -> x = [1,2].
    var abuf = [_]f64{ 1, 0, 1, 0, 1, 1 };
    const a = colMajor(IJ, &abuf, 3, 2);
    var bbuf = [_]f64{ 1, 2, 3 };
    const b = colMajor(IK, &bbuf, 3, 1);
    const x = try lstsq(f64, IJ, IK, testing.allocator, a.asConst(), b);
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 0, .k = 0 }).*, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 0 }).*, 1e-10);
    // A untouched (lstsqInplace would overwrite it with QR factors).
    try testing.expectEqualSlices(f64, &.{ 1, 0, 1, 0, 1, 1 }, &abuf);
}

test "qr: preserves a column-major A" {
    // Column-major A: qrInplace would overwrite it in place.
    var abuf = [_]f64{ 1, 3, 2, 4 };
    const a = colMajor(IJ, &abuf, 2, 2);
    const res = try qr(f64, IJ, testing.allocator, a.asConst());
    defer res.deinit(testing.allocator);
    try testing.expectEqualSlices(f64, &.{ 1, 3, 2, 4 }, &abuf);
    const orig = [_][2]f64{ .{ 1, 2 }, .{ 3, 4 } };
    var ii: usize = 0;
    while (ii < 2) : (ii += 1) {
        var jj: usize = 0;
        while (jj < 2) : (jj += 1) {
            var acc: f64 = 0;
            var kk: usize = 0;
            while (kk < 2) : (kk += 1)
                acc += res.q.at(.{ .i = ii, .qr_rank = kk }).* * res.r.at(.{ .qr_rank = kk, .j = jj }).*;
            try testing.expectApproxEqAbs(orig[ii][jj], acc, 1e-10);
        }
    }
}

test "eigVectors: preserves a column-major A" {
    // Column-major A = [[2,0],[1,3]]: eigVectorsInplace would use it as scratch.
    var abuf = [_]f64{ 2, 1, 0, 3 };
    const a = colMajor(IJ, &abuf, 2, 2);
    const res = try eigVectors(f64, IJ, .right, testing.allocator, a.asConst());
    defer res.deinit(testing.allocator);
    try testing.expectEqualSlices(f64, &.{ 2, 1, 0, 3 }, &abuf);
    // Right eigenvectors still satisfy A v = lambda v.
    const A = [_][2]f64{ .{ 2, 0 }, .{ 1, 3 } };
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const v0 = res.right.at(.{ .j = 0, .eig = jc }).*;
        const v1 = res.right.at(.{ .j = 1, .eig = jc }).*;
        const lam = res.values[jc];
        try testing.expectApproxEqAbs(lam.re * v0.re, A[0][0] * v0.re + A[0][1] * v1.re, 1e-10);
        try testing.expectApproxEqAbs(lam.re * v1.re, A[1][0] * v0.re + A[1][1] * v1.re, 1e-10);
    }
}

test "svdVectors: preserves a column-major A, reconstructs" {
    // Column-major A 3x2: svdVectorsInplace would overwrite it.
    var abuf = [_]f64{ 1, 0, 1, 0, 1, 1 };
    const a = colMajor(IJ, &abuf, 3, 2);
    const A = [_][2]f64{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } };
    const res = try svdVectors(f64, IJ, testing.allocator, a.asConst(), .thin);
    defer res.deinit(testing.allocator);
    try testing.expectEqualSlices(f64, &.{ 1, 0, 1, 0, 1, 1 }, &abuf);
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc: f64 = 0;
            var l: usize = 0;
            while (l < 2) : (l += 1)
                acc += res.u.at(.{ .i = i, .sv = l }).* * res.s[l] * res.vt.at(.{ .sv = l, .j = j }).*;
            try testing.expectApproxEqAbs(A[i][j], acc, 1e-10);
        }
    }
}

// ===== Complex decompositions (Q6) ==========================================

const Cf = Complex(f64);

/// Build a column-major complex `NamedArray` view over `buf`.
fn colMajorC(comptime Axis: type, buf: []Cf, rows: usize, cols: usize) NamedArray(Axis, Cf) {
    const names = comptime std.meta.fieldNames(Axis);
    var idx: NamedIndex(Axis) = undefined;
    idx.offset = 0;
    @field(idx.shape, names[0]) = rows;
    @field(idx.shape, names[1]) = cols;
    @field(idx.strides, names[0]) = 1;
    @field(idx.strides, names[1]) = @intCast(rows);
    return NamedArray(Axis, Cf).init(idx, buf);
}

/// Complex multiply-accumulate: acc += x * y.
fn cmadd(acc: *Cf, x: Cf, y: Cf) void {
    acc.re += x.re * y.re - x.im * y.im;
    acc.im += x.re * y.im + x.im * y.re;
}

/// Scale a complex value by a real scalar.
fn cscale(x: Cf, s: f64) Cf {
    return Cf.init(x.re * s, x.im * s);
}

test "eigSymInplace: Hermitian eigenvalues are real and ascending" {
    // A = [[2, i], [-i, 2]] (Hermitian) has eigenvalues 2 ± |i| = 1, 3.
    var abuf = [_]Cf{ Cf.init(2, 0), Cf.init(0, -1), Cf.init(0, 1), Cf.init(2, 0) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const w = try eigSymInplace(Cf, IJ, testing.allocator, a, .upper);
    defer testing.allocator.free(w);
    // `w` is real (RealOf(Cf) == f64), even though A is complex.
    try testing.expect(@TypeOf(w) == []f64);
    try testing.expectApproxEqAbs(@as(f64, 1.0), w[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), w[1], 1e-12);
}

test "eigSymVectors: Hermitian A v = lambda v with real lambda" {
    // A = [[2, i], [-i, 2]], eigenvalues 1 and 3.
    const A = [_][2]Cf{
        .{ Cf.init(2, 0), Cf.init(0, 1) },
        .{ Cf.init(0, -1), Cf.init(2, 0) },
    };
    var abuf = [_]Cf{ Cf.init(2, 0), Cf.init(0, -1), Cf.init(0, 1), Cf.init(2, 0) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const res = try eigSymVectors(Cf, IJ, testing.allocator, a.asConst(), .upper);
    defer res.deinit(testing.allocator);
    try testing.expect(@TypeOf(res.values) == []f64);
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const lam = res.values[jc];
        // Compare A*v against lambda*v (lambda real).
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            var av = Cf.init(0, 0);
            var l: usize = 0;
            while (l < 2) : (l += 1) cmadd(&av, A[i][l], res.vectors.at(.{ .j = l, .eig = jc }).*);
            const v = res.vectors.at(.{ .j = i, .eig = jc }).*;
            try testing.expectApproxEqAbs(lam * v.re, av.re, 1e-10);
            try testing.expectApproxEqAbs(lam * v.im, av.im, 1e-10);
        }
    }
}

test "eigInplace: complex eigenvalues of a complex matrix" {
    // Upper-triangular A = [[1, 5], [0, i]] has eigenvalues 1 and i (the diagonal).
    var abuf = [_]Cf{ Cf.init(1, 0), Cf.init(0, 0), Cf.init(5, 0), Cf.init(0, 1) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const w = try eigInplace(Cf, IJ, testing.allocator, a);
    defer testing.allocator.free(w);
    // Eigenvalues come back in some order; match the set {1, i}.
    const want = [_]Cf{ Cf.init(1, 0), Cf.init(0, 1) };
    for (want) |e| {
        var found = false;
        for (w) |got| {
            if (@abs(got.re - e.re) < 1e-10 and @abs(got.im - e.im) < 1e-10) found = true;
        }
        try testing.expect(found);
    }
}

test "eigVectorsInplace: complex right eigenvectors satisfy A v = lambda v" {
    // A = [[2, 1], [0, i]], eigenvalues 2 and i.
    const A = [_][2]Cf{
        .{ Cf.init(2, 0), Cf.init(1, 0) },
        .{ Cf.init(0, 0), Cf.init(0, 1) },
    };
    var abuf = [_]Cf{ Cf.init(2, 0), Cf.init(0, 0), Cf.init(1, 0), Cf.init(0, 1) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const res = try eigVectorsInplace(Cf, IJ, .right, testing.allocator, a);
    defer res.deinit(testing.allocator);
    var jc: usize = 0;
    while (jc < 2) : (jc += 1) {
        const lam = res.values[jc];
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            var av = Cf.init(0, 0);
            var l: usize = 0;
            while (l < 2) : (l += 1) cmadd(&av, A[i][l], res.right.at(.{ .j = l, .eig = jc }).*);
            const v = res.right.at(.{ .j = i, .eig = jc }).*;
            var lv = Cf.init(0, 0);
            cmadd(&lv, lam, v);
            try testing.expectApproxEqAbs(lv.re, av.re, 1e-10);
            try testing.expectApproxEqAbs(lv.im, av.im, 1e-10);
        }
    }
}

test "svdInplace: singular values of a complex matrix are real, descending" {
    // A = diag(3, 4i): singular values |3|=3, |4i|=4 -> [4, 3].
    var abuf = [_]Cf{ Cf.init(3, 0), Cf.init(0, 0), Cf.init(0, 0), Cf.init(0, 4) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const s = try svdInplace(Cf, IJ, testing.allocator, a);
    defer testing.allocator.free(s);
    try testing.expect(@TypeOf(s) == []f64);
    try testing.expectApproxEqAbs(@as(f64, 4.0), s[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), s[1], 1e-12);
}

test "svdVectorsInplace: complex reconstruction A = U diag(s) V^H (column-major)" {
    // A = [[1+i, 2], [0, 1-i]].
    const A = [_][2]Cf{
        .{ Cf.init(1, 1), Cf.init(2, 0) },
        .{ Cf.init(0, 0), Cf.init(1, -1) },
    };
    var abuf = [_]Cf{ Cf.init(1, 1), Cf.init(0, 0), Cf.init(2, 0), Cf.init(1, -1) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const res = try svdVectorsInplace(Cf, IJ, testing.allocator, a, .thin);
    defer res.deinit(testing.allocator);
    try testing.expect(@TypeOf(res.s) == []f64);
    // vt already holds V^H, so A(i,j) = sum_l U(i,l) * s(l) * Vt(l,j).
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc = Cf.init(0, 0);
            var l: usize = 0;
            while (l < 2) : (l += 1) {
                const us = cscale(res.u.at(.{ .i = i, .sv = l }).*, res.s[l]);
                cmadd(&acc, us, res.vt.at(.{ .sv = l, .j = j }).*);
            }
            try testing.expectApproxEqAbs(A[i][j].re, acc.re, 1e-10);
            try testing.expectApproxEqAbs(A[i][j].im, acc.im, 1e-10);
        }
    }
}

test "svdVectorsInplace: complex row-major A reconstructs (column-major pack path)" {
    // Same A, stored row-major to exercise the complex packing fallback.
    const A = [_][2]Cf{
        .{ Cf.init(1, 1), Cf.init(2, 0) },
        .{ Cf.init(0, 0), Cf.init(1, -1) },
    };
    // Row-major buffer: (0,0),(0,1),(1,0),(1,1).
    var abuf = [_]Cf{ Cf.init(1, 1), Cf.init(2, 0), Cf.init(0, 0), Cf.init(1, -1) };
    const a = NamedArray(IJ, Cf).init(NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 2 }), &abuf);
    const res = try svdVectorsInplace(Cf, IJ, testing.allocator, a, .full);
    defer res.deinit(testing.allocator);
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc = Cf.init(0, 0);
            var l: usize = 0;
            while (l < 2) : (l += 1) {
                const us = cscale(res.u.at(.{ .i = i, .sv = l }).*, res.s[l]);
                cmadd(&acc, us, res.vt.at(.{ .sv = l, .j = j }).*);
            }
            try testing.expectApproxEqAbs(A[i][j].re, acc.re, 1e-10);
            try testing.expectApproxEqAbs(A[i][j].im, acc.im, 1e-10);
        }
    }
}

test "qrInplace: complex A = Q R with unitary Q (Q^H Q = I)" {
    // A = [[1+i, 2], [0, 1-i]] (column-major, factored in place).
    const A = [_][2]Cf{
        .{ Cf.init(1, 1), Cf.init(2, 0) },
        .{ Cf.init(0, 0), Cf.init(1, -1) },
    };
    var abuf = [_]Cf{ Cf.init(1, 1), Cf.init(0, 0), Cf.init(2, 0), Cf.init(1, -1) };
    const a = colMajorC(IJ, &abuf, 2, 2);
    const res = try qrInplace(Cf, IJ, testing.allocator, a);
    defer res.deinit(testing.allocator);
    // Reconstruction Q*R = A.
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc = Cf.init(0, 0);
            var l: usize = 0;
            while (l < 2) : (l += 1)
                cmadd(&acc, res.q.at(.{ .i = i, .qr_rank = l }).*, res.r.at(.{ .qr_rank = l, .j = j }).*);
            try testing.expectApproxEqAbs(A[i][j].re, acc.re, 1e-10);
            try testing.expectApproxEqAbs(A[i][j].im, acc.im, 1e-10);
        }
    }
    // Unitarity: Q^H Q = I (sum_i conj(Q(i,k)) Q(i,l) = delta_kl).
    var kk: usize = 0;
    while (kk < 2) : (kk += 1) {
        var ll: usize = 0;
        while (ll < 2) : (ll += 1) {
            var re: f64 = 0;
            var im: f64 = 0;
            var ii: usize = 0;
            while (ii < 2) : (ii += 1) {
                const qk = res.q.at(.{ .i = ii, .qr_rank = kk }).*;
                const ql = res.q.at(.{ .i = ii, .qr_rank = ll }).*;
                re += qk.re * ql.re + qk.im * ql.im;
                im += qk.re * ql.im - qk.im * ql.re;
            }
            const want: f64 = if (kk == ll) 1 else 0;
            try testing.expectApproxEqAbs(want, re, 1e-10);
            try testing.expectApproxEqAbs(@as(f64, 0), im, 1e-10);
        }
    }
}

test "lstsqInplace: complex overdetermined exact fit, column-major A" {
    // A (3x2) real-valued but complex-typed; exact-fit x = [1, 2+i].
    // b = A x = [1, 2+i, 3+i].
    var abuf = [_]Cf{
        Cf.init(1, 0), Cf.init(0, 0), Cf.init(1, 0), // col 0
        Cf.init(0, 0), Cf.init(1, 0), Cf.init(1, 0), // col 1
    };
    const a = colMajorC(IJ, &abuf, 3, 2);
    var bbuf = [_]Cf{ Cf.init(1, 0), Cf.init(2, 1), Cf.init(3, 1) };
    const b = colMajorC(IK, &bbuf, 3, 1);
    const x = try lstsqInplace(Cf, IJ, IK, testing.allocator, a, b);
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 0, .k = 0 }).re, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), x.at(.{ .j = 0, .k = 0 }).im, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 0 }).re, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 1, .k = 0 }).im, 1e-10);
}

test "lstsqInplace: complex row-major A uses the column-major pack fallback" {
    // Same system, A stored row-major (exercises the complex 'N'-with-pack path).
    // Row-major buffer of [[1,0],[0,1],[1,1]]: rows concatenated.
    var abuf = [_]Cf{
        Cf.init(1, 0), Cf.init(0, 0), // row 0
        Cf.init(0, 0), Cf.init(1, 0), // row 1
        Cf.init(1, 0), Cf.init(1, 0), // row 2
    };
    const a = NamedArray(IJ, Cf).init(NamedIndex(IJ).initContiguous(.{ .i = 3, .j = 2 }), &abuf);
    var bbuf = [_]Cf{ Cf.init(1, 0), Cf.init(2, 1), Cf.init(3, 1) };
    const b = colMajorC(IK, &bbuf, 3, 1);
    const x = try lstsqInplace(Cf, IJ, IK, testing.allocator, a, b);
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 0, .k = 0 }).re, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), x.at(.{ .j = 0, .k = 0 }).im, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), x.at(.{ .j = 1, .k = 0 }).re, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), x.at(.{ .j = 1, .k = 0 }).im, 1e-10);
    // Row-major A was left unmodified (factored from a packed copy).
    try testing.expectApproxEqAbs(@as(f64, 1.0), abuf[0].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), abuf[5].re, 1e-12);
}

test "Q5 sizing helpers: closed-form scratch sizes" {
    try testing.expectEqual(@as(usize, 5), pivotLen(5));
    try testing.expectEqual(@as(usize, 1), pivotLen(0));
    try testing.expectEqual(@as(usize, 24), svdIworkLen(4, 3)); // 8*min(4,3)
    try testing.expectEqual(@as(usize, 7), eigSymRworkLen(3)); // 3n-2
    try testing.expectEqual(@as(usize, 1), eigSymRworkLen(1));
    try testing.expectEqual(@as(usize, 8), eigRworkLen(4)); // 2n
    try testing.expectEqual(@as(usize, 21), svdRworkLen(4, 3, false)); // 7*min
}

// ===== Single-precision + complex-branch coverage (session 10) ===============
// The f64/Complex(f64) suite above never runs the real `s*` dispatch arms nor
// the single-precision complex `zarray_c*` shim forwarders. These tests execute
// each once (the shim's compile-time header check verifies signatures; this
// verifies the runtime interleaved-buffer casts actually work), and cover the
// complex `det`/`inv`/`cholesky` paths the earlier suite skipped — including the
// complex determinant sign×product branch where the `Complex.scale` bug hid.

const Cf32 = Complex(f32);

/// Generic row-major (last axis contiguous) view builder for any scalar `T`.
fn rowMajorG(comptime T: type, comptime Axis: type, buf: []T, rows: usize, cols: usize) NamedArray(Axis, T) {
    const names = comptime std.meta.fieldNames(Axis);
    var idx: NamedIndex(Axis) = undefined;
    idx.offset = 0;
    @field(idx.shape, names[0]) = rows;
    @field(idx.shape, names[1]) = cols;
    @field(idx.strides, names[0]) = @intCast(cols);
    @field(idx.strides, names[1]) = 1;
    return NamedArray(Axis, T).init(idx, buf);
}

/// Generic column-major (first axis contiguous) view builder for any scalar `T`.
fn colMajorG(comptime T: type, comptime Axis: type, buf: []T, rows: usize, cols: usize) NamedArray(Axis, T) {
    const names = comptime std.meta.fieldNames(Axis);
    var idx: NamedIndex(Axis) = undefined;
    idx.offset = 0;
    @field(idx.shape, names[0]) = rows;
    @field(idx.shape, names[1]) = cols;
    @field(idx.strides, names[0]) = 1;
    @field(idx.strides, names[1]) = @intCast(rows);
    return NamedArray(Axis, T).init(idx, buf);
}

/// `acc += x*y` for any `std.math.Complex(T)`.
fn cmaddT(comptime C: type, acc: *C, x: C, y: C) void {
    acc.re += x.re * y.re - x.im * y.im;
    acc.im += x.re * y.im + x.im * y.re;
}

test "f32: LU family routes through s-arms" {
    // solve: [[1,2],[3,4]] x = [5,6] -> [-4, 4.5]
    var sb = [_]f32{ 1, 2, 3, 4 };
    const sa = rowMajorG(f32, IJ, &sb, 2, 2);
    var srhs = [_]f32{ 5, 6 };
    const sbb = colMajorG(f32, IK, &srhs, 2, 1);
    var ipiv: [2]c_int = undefined;
    const sx = try solve(f32, IJ, IK, sa, sbb, &ipiv);
    try testing.expectApproxEqAbs(@as(f32, -4.0), sx.at(.{ .j = 0, .k = 0 }).*, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4.5), sx.at(.{ .j = 1, .k = 0 }).*, 1e-4);

    // det: -2
    var db = [_]f32{ 1, 2, 3, 4 };
    const da = rowMajorG(f32, IJ, &db, 2, 2);
    try testing.expectApproxEqAbs(@as(f32, -2.0), try det(f32, IJ, testing.allocator, da.asConst()), 1e-4);

    // inv: [[1,2],[3,4]] -> [[-2,1],[1.5,-0.5]]
    var ib = [_]f32{ 1, 2, 3, 4 };
    const ia = rowMajorG(f32, IJ, &ib, 2, 2);
    try inv(f32, IJ, testing.allocator, ia);
    try testing.expectApproxEqAbs(@as(f32, -2.0), ia.at(.{ .i = 0, .j = 0 }).*, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), ia.at(.{ .i = 0, .j = 1 }).*, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.5), ia.at(.{ .i = 1, .j = 0 }).*, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -0.5), ia.at(.{ .i = 1, .j = 1 }).*, 1e-4);

    // cholesky + solve: [[4,2],[2,3]] SPD, b=[1,1] -> [0.125, 0.25]
    var cb = [_]f32{ 4, 2, 2, 3 };
    const ca = rowMajorG(f32, IJ, &cb, 2, 2);
    try cholesky(f32, IJ, ca, .lower);
    var crhs = [_]f32{ 1, 1 };
    const cbb = colMajorG(f32, IK, &crhs, 2, 1);
    const cx = try choleskySolve(f32, IJ, IK, ca.asConst(), cbb, .lower);
    try testing.expectApproxEqAbs(@as(f32, 0.125), cx.at(.{ .j = 0, .k = 0 }).*, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.25), cx.at(.{ .j = 1, .k = 0 }).*, 1e-4);
}

test "f32: eigSym/eig/svd route through s-arms" {
    // eigSym [[2,1],[1,2]] -> 1, 3
    var eb = [_]f32{ 2, 1, 1, 2 };
    const ea = rowMajorG(f32, IJ, &eb, 2, 2);
    const w = try eigSym(f32, IJ, testing.allocator, ea.asConst(), .upper);
    defer testing.allocator.free(w);
    try testing.expect(@TypeOf(w) == []f32);
    try testing.expectApproxEqAbs(@as(f32, 1.0), w[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3.0), w[1], 1e-4);

    // svd diag(2,3) -> 3, 2
    var vb = [_]f32{ 2, 0, 0, 3 };
    const va = rowMajorG(f32, IJ, &vb, 2, 2);
    const sv = try svd(f32, IJ, testing.allocator, va.asConst());
    defer testing.allocator.free(sv);
    try testing.expectApproxEqAbs(@as(f32, 3.0), sv[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), sv[1], 1e-4);

    // eig [[0,-1],[1,0]] -> +-i
    var gb = [_]f32{ 0, -1, 1, 0 };
    const ga = rowMajorG(f32, IJ, &gb, 2, 2);
    const ev = try eig(f32, IJ, testing.allocator, ga.asConst());
    defer testing.allocator.free(ev);
    try testing.expect(@TypeOf(ev) == []Complex(f32));
    try testing.expectApproxEqAbs(@as(f32, 0.0), ev[0].re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), @abs(ev[0].im), 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), @abs(ev[1].im), 1e-4);
}

test "f32: qr/lstsq route through s-arms" {
    // qr of [[1,2],[3,4]] column-major -> reconstruct Q*R = A
    var qb = [_]f32{ 1, 3, 2, 4 };
    const qa = colMajorG(f32, IJ, &qb, 2, 2);
    const res = try qr(f32, IJ, testing.allocator, qa.asConst());
    defer res.deinit(testing.allocator);
    const orig = [_][2]f32{ .{ 1, 2 }, .{ 3, 4 } };
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc: f32 = 0;
            var k: usize = 0;
            while (k < 2) : (k += 1)
                acc += res.q.at(.{ .i = i, .qr_rank = k }).* * res.r.at(.{ .qr_rank = k, .j = j }).*;
            try testing.expectApproxEqAbs(orig[i][j], acc, 1e-4);
        }
    }

    // lstsq overdetermined exact fit: A 3x2 col-major, b=[1,2,3] -> x=[1,2]
    var la = [_]f32{ 1, 0, 1, 0, 1, 1 };
    const lam = colMajorG(f32, IJ, &la, 3, 2);
    var lb = [_]f32{ 1, 2, 3 };
    const lbm = colMajorG(f32, IK, &lb, 3, 1);
    const lx = try lstsq(f32, IJ, IK, testing.allocator, lam.asConst(), lbm);
    try testing.expectApproxEqAbs(@as(f32, 1.0), lx.at(.{ .j = 0, .k = 0 }).*, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), lx.at(.{ .j = 1, .k = 0 }).*, 1e-4);
}

test "Complex(f32): LU family routes through c-shim arms" {
    // solve: A = diag(1, 2i), b = [2, 4i] -> x = [2, 2]
    var sb = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(0, 0), Cf32.init(0, 2) };
    const sa = colMajorG(Cf32, IJ, &sb, 2, 2);
    var srhs = [_]Cf32{ Cf32.init(2, 0), Cf32.init(0, 4) };
    const sbb = colMajorG(Cf32, IK, &srhs, 2, 1);
    var ipiv: [2]c_int = undefined;
    const sx = try solve(Cf32, IJ, IK, sa, sbb, &ipiv);
    try testing.expectApproxEqAbs(@as(f32, 2.0), sx.at(.{ .j = 0, .k = 0 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), sx.at(.{ .j = 1, .k = 0 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), sx.at(.{ .j = 1, .k = 0 }).im, 1e-4);

    // det with a row swap -> exercises the complex sign*product branch:
    // det[[0,1],[1,0]] = -1
    var db = [_]Cf32{ Cf32.init(0, 0), Cf32.init(1, 0), Cf32.init(1, 0), Cf32.init(0, 0) };
    const da = colMajorG(Cf32, IJ, &db, 2, 2);
    const d = try det(Cf32, IJ, testing.allocator, da.asConst());
    try testing.expectApproxEqAbs(@as(f32, -1.0), d.re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), d.im, 1e-4);

    // inv: diag(1, 2i) -> diag(1, -0.5i)  (1/(2i) = -i/2)
    var ib = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(0, 0), Cf32.init(0, 2) };
    const ia = colMajorG(Cf32, IJ, &ib, 2, 2);
    try inv(Cf32, IJ, testing.allocator, ia);
    try testing.expectApproxEqAbs(@as(f32, 1.0), ia.at(.{ .i = 0, .j = 0 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), ia.at(.{ .i = 1, .j = 1 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -0.5), ia.at(.{ .i = 1, .j = 1 }).im, 1e-4);

    // cholesky: Hermitian PD [[2,i],[-i,2]], solve b=[1,0] -> (1/3)[2, i]
    var cb = [_]Cf32{ Cf32.init(2, 0), Cf32.init(0, -1), Cf32.init(0, 1), Cf32.init(2, 0) };
    const ca = colMajorG(Cf32, IJ, &cb, 2, 2);
    try cholesky(Cf32, IJ, ca, .upper);
    var crhs = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 0) };
    const cbb = colMajorG(Cf32, IK, &crhs, 2, 1);
    const cx = try choleskySolve(Cf32, IJ, IK, ca.asConst(), cbb, .upper);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), cx.at(.{ .j = 0, .k = 0 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), cx.at(.{ .j = 0, .k = 0 }).im, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), cx.at(.{ .j = 1, .k = 0 }).im, 1e-4);
}

test "Complex(f32): eigSym/eig/svd route through c-shim arms" {
    // eigSym (cheev): Hermitian [[2,i],[-i,2]] -> 1, 3 (real)
    var eb = [_]Cf32{ Cf32.init(2, 0), Cf32.init(0, -1), Cf32.init(0, 1), Cf32.init(2, 0) };
    const ea = colMajorG(Cf32, IJ, &eb, 2, 2);
    const w = try eigSym(Cf32, IJ, testing.allocator, ea.asConst(), .upper);
    defer testing.allocator.free(w);
    try testing.expect(@TypeOf(w) == []f32);
    try testing.expectApproxEqAbs(@as(f32, 1.0), w[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3.0), w[1], 1e-4);

    // eig (cgeev): upper-triangular [[1,5],[0,i]] -> {1, i}
    var gb = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(5, 0), Cf32.init(0, 1) };
    const ga = colMajorG(Cf32, IJ, &gb, 2, 2);
    const ev = try eig(Cf32, IJ, testing.allocator, ga.asConst());
    defer testing.allocator.free(ev);
    var found1 = false;
    var foundi = false;
    for (ev) |z| {
        if (@abs(z.re - 1) < 1e-4 and @abs(z.im) < 1e-4) found1 = true;
        if (@abs(z.re) < 1e-4 and @abs(z.im - 1) < 1e-4) foundi = true;
    }
    try testing.expect(found1 and foundi);

    // svd (cgesdd): diag(3, 4i) -> [4, 3]
    var vb = [_]Cf32{ Cf32.init(3, 0), Cf32.init(0, 0), Cf32.init(0, 0), Cf32.init(0, 4) };
    const va = colMajorG(Cf32, IJ, &vb, 2, 2);
    const sv = try svd(Cf32, IJ, testing.allocator, va.asConst());
    defer testing.allocator.free(sv);
    try testing.expect(@TypeOf(sv) == []f32);
    try testing.expectApproxEqAbs(@as(f32, 4.0), sv[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3.0), sv[1], 1e-4);
}

test "Complex(f32): qr/lstsq route through c-shim arms" {
    // qr (cgeqrf/cungqr): A = [[1+i, 2],[0, 1-i]] column-major, reconstruct Q*R = A
    const A = [_][2]Cf32{
        .{ Cf32.init(1, 1), Cf32.init(2, 0) },
        .{ Cf32.init(0, 0), Cf32.init(1, -1) },
    };
    var qb = [_]Cf32{ Cf32.init(1, 1), Cf32.init(0, 0), Cf32.init(2, 0), Cf32.init(1, -1) };
    const qa = colMajorG(Cf32, IJ, &qb, 2, 2);
    const res = try qr(Cf32, IJ, testing.allocator, qa.asConst());
    defer res.deinit(testing.allocator);
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            var acc = Cf32.init(0, 0);
            var l: usize = 0;
            while (l < 2) : (l += 1)
                cmaddT(Cf32, &acc, res.q.at(.{ .i = i, .qr_rank = l }).*, res.r.at(.{ .qr_rank = l, .j = j }).*);
            try testing.expectApproxEqAbs(A[i][j].re, acc.re, 1e-4);
            try testing.expectApproxEqAbs(A[i][j].im, acc.im, 1e-4);
        }
    }

    // lstsq (cgels): A (3x2) real-valued complex-typed, exact fit x = [1, 2+i]
    var la = [_]Cf32{
        Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(1, 0),
        Cf32.init(0, 0), Cf32.init(1, 0), Cf32.init(1, 0),
    };
    const lam = colMajorG(Cf32, IJ, &la, 3, 2);
    var lb = [_]Cf32{ Cf32.init(1, 0), Cf32.init(2, 1), Cf32.init(3, 1) };
    const lbm = colMajorG(Cf32, IK, &lb, 3, 1);
    const lx = try lstsq(Cf32, IJ, IK, testing.allocator, lam.asConst(), lbm);
    try testing.expectApproxEqAbs(@as(f32, 1.0), lx.at(.{ .j = 0, .k = 0 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), lx.at(.{ .j = 0, .k = 0 }).im, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), lx.at(.{ .j = 1, .k = 0 }).re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), lx.at(.{ .j = 1, .k = 0 }).im, 1e-4);
}

test "Complex(f64): det/inv/cholesky route through z-shim arms" {
    // det with a row swap -> complex sign*product branch: det[[0,1],[1,0]] = -1
    var db = [_]Cf{ Cf.init(0, 0), Cf.init(1, 0), Cf.init(1, 0), Cf.init(0, 0) };
    const da = colMajorC(IJ, &db, 2, 2);
    const d = try det(Cf, IJ, testing.allocator, da.asConst());
    try testing.expectApproxEqAbs(@as(f64, -1.0), d.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), d.im, 1e-12);

    // Non-trivial complex det (upper-tri, no swap): det[[2,1],[0,1-i]] = 2-2i
    var d2 = [_]Cf{ Cf.init(2, 0), Cf.init(0, 0), Cf.init(1, 0), Cf.init(1, -1) };
    const da2 = colMajorC(IJ, &d2, 2, 2);
    const dd = try det(Cf, IJ, testing.allocator, da2.asConst());
    try testing.expectApproxEqAbs(@as(f64, 2.0), dd.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -2.0), dd.im, 1e-12);

    // inv (zgetri): diag(1, 2i) -> diag(1, -0.5i)
    var ib = [_]Cf{ Cf.init(1, 0), Cf.init(0, 0), Cf.init(0, 0), Cf.init(0, 2) };
    const ia = colMajorC(IJ, &ib, 2, 2);
    try inv(Cf, IJ, testing.allocator, ia);
    try testing.expectApproxEqAbs(@as(f64, 1.0), ia.at(.{ .i = 0, .j = 0 }).re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.5), ia.at(.{ .i = 1, .j = 1 }).im, 1e-12);

    // cholesky (zpotrf/zpotrs): Hermitian PD [[2,i],[-i,2]], b=[1,0] -> (1/3)[2, i]
    var cb = [_]Cf{ Cf.init(2, 0), Cf.init(0, -1), Cf.init(0, 1), Cf.init(2, 0) };
    const ca = colMajorC(IJ, &cb, 2, 2);
    try cholesky(Cf, IJ, ca, .upper);
    var crhs = [_]Cf{ Cf.init(1, 0), Cf.init(0, 0) };
    const cbb = colMajorC(IK, &crhs, 2, 1);
    const cx = try choleskySolve(Cf, IJ, IK, ca.asConst(), cbb, .upper);
    try testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), cx.at(.{ .j = 0, .k = 0 }).re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), cx.at(.{ .j = 1, .k = 0 }).im, 1e-12);
}

// ===== Generalized eigenproblem + rank-deficient least squares ===============

test "eigSymGen: generalized eigenvalues A x = lambda B x (real, itype a_bx)" {
    // A = 2I, B = diag(2, 8). A x = lambda B x => 2 = 2*lambda (e1) => lambda=1;
    // 2 = 8*lambda (e2) => lambda = 0.25. Ascending: {0.25, 1}.
    var ab = [_]f64{ 2, 0, 0, 2 };
    var bb = [_]f64{ 2, 0, 0, 8 };
    const a = colMajor(IJ, &ab, 2, 2);
    const b = colMajor(IJ, &bb, 2, 2);
    const w = try eigSymGen(f64, IJ, testing.allocator, a.asConst(), b.asConst(), .a_bx, .upper);
    defer testing.allocator.free(w);
    try testing.expectApproxEqAbs(@as(f64, 0.25), w[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), w[1], 1e-12);
}

test "eigSymGen: preserves A and B" {
    var ab = [_]f64{ 3, 1, 1, 3 };
    var bb = [_]f64{ 2, 0, 0, 1 };
    const a_before = ab;
    const b_before = bb;
    const a = colMajor(IJ, &ab, 2, 2);
    const b = colMajor(IJ, &bb, 2, 2);
    const w = try eigSymGen(f64, IJ, testing.allocator, a.asConst(), b.asConst(), .a_bx, .upper);
    defer testing.allocator.free(w);
    try testing.expectEqualSlices(f64, &a_before, &ab);
    try testing.expectEqualSlices(f64, &b_before, &bb);
    try testing.expect(w[0] < w[1]);
}

test "eigSymGenInplace: matches eigSymGen (real)" {
    var ab = [_]f64{ 3, 1, 1, 3 };
    var bb = [_]f64{ 2, 0, 0, 1 };
    var ab2 = ab;
    var bb2 = bb;
    const w1 = try eigSymGen(f64, IJ, testing.allocator, colMajor(IJ, &ab, 2, 2).asConst(), colMajor(IJ, &bb, 2, 2).asConst(), .a_bx, .upper);
    defer testing.allocator.free(w1);
    const w2 = try eigSymGenInplace(f64, IJ, testing.allocator, colMajor(IJ, &ab2, 2, 2), colMajor(IJ, &bb2, 2, 2), .a_bx, .upper);
    defer testing.allocator.free(w2);
    try testing.expectApproxEqAbs(w1[0], w2[0], 1e-12);
    try testing.expectApproxEqAbs(w1[1], w2[1], 1e-12);
}

test "eigSymGenVectors: A v = lambda B v residual (real)" {
    // A = [[3,1],[1,3]] (upper), B = diag(2,1) PD.
    var ab = [_]f64{ 3, 1, 1, 3 };
    var bb = [_]f64{ 2, 0, 0, 1 };
    const a = colMajor(IJ, &ab, 2, 2);
    const b = colMajor(IJ, &bb, 2, 2);
    const res = try eigSymGenVectors(f64, IJ, testing.allocator, a.asConst(), b.asConst(), .a_bx, .upper);
    defer res.deinit(testing.allocator);
    try testing.expect(res.values[0] < res.values[1]);
    // For each eigenpair j: A v_j - lambda_j B v_j ~ 0.
    var j: usize = 0;
    while (j < 2) : (j += 1) {
        const v0 = res.vectors.at(.{ .j = 0, .eig = j }).*;
        const v1 = res.vectors.at(.{ .j = 1, .eig = j }).*;
        const lam = res.values[j];
        // A v
        const av0 = 3 * v0 + 1 * v1;
        const av1 = 1 * v0 + 3 * v1;
        // B v
        const bv0 = 2 * v0;
        const bv1 = 1 * v1;
        try testing.expectApproxEqAbs(@as(f64, 0), av0 - lam * bv0, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0), av1 - lam * bv1, 1e-12);
    }
}

test "eigSymGenVectors: Hermitian, B = I equals ordinary eig (complex)" {
    // A = [[2, i],[-i, 2]] Hermitian, B = I => ordinary Hermitian eig {1, 3}.
    var ab = [_]Cf{ Cf.init(2, 0), Cf.init(0, -1), Cf.init(0, 1), Cf.init(2, 0) };
    var bb = [_]Cf{ Cf.init(1, 0), Cf.init(0, 0), Cf.init(0, 0), Cf.init(1, 0) };
    const a = colMajorC(IJ, &ab, 2, 2);
    const b = colMajorC(IJ, &bb, 2, 2);
    const res = try eigSymGenVectors(Cf, IJ, testing.allocator, a.asConst(), b.asConst(), .a_bx, .upper);
    defer res.deinit(testing.allocator);
    try testing.expectApproxEqAbs(@as(f64, 1), res.values[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3), res.values[1], 1e-12);
    // Residual A v = lambda v (B = I).
    var j: usize = 0;
    while (j < 2) : (j += 1) {
        const v0 = res.vectors.at(.{ .j = 0, .eig = j }).*;
        const v1 = res.vectors.at(.{ .j = 1, .eig = j }).*;
        const lam = res.values[j];
        // (A v)_0 = 2 v0 + i v1 ; (A v)_1 = -i v0 + 2 v1
        var av0 = Cf.init(0, 0);
        cmadd(&av0, Cf.init(2, 0), v0);
        cmadd(&av0, Cf.init(0, 1), v1);
        var av1 = Cf.init(0, 0);
        cmadd(&av1, Cf.init(0, -1), v0);
        cmadd(&av1, Cf.init(2, 0), v1);
        try testing.expectApproxEqAbs(@as(f64, 0), av0.re - lam * v0.re, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0), av0.im - lam * v0.im, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0), av1.re - lam * v1.re, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0), av1.im - lam * v1.im, 1e-12);
    }
}

test "f32: eigSymGen routes through ssygv arm" {
    var ab = [_]f32{ 2, 0, 0, 2 };
    var bb = [_]f32{ 2, 0, 0, 8 };
    const a = colMajorG(f32, IJ, &ab, 2, 2);
    const b = colMajorG(f32, IJ, &bb, 2, 2);
    const w = try eigSymGen(f32, IJ, testing.allocator, a.asConst(), b.asConst(), .a_bx, .upper);
    defer testing.allocator.free(w);
    try testing.expectApproxEqAbs(@as(f32, 0.25), w[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), w[1], 1e-5);
}

test "Complex(f32): eigSymGenVectors routes through chegv arm" {
    var ab = [_]Cf32{ Cf32.init(2, 0), Cf32.init(0, -1), Cf32.init(0, 1), Cf32.init(2, 0) };
    var bb = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(0, 0), Cf32.init(1, 0) };
    const a = colMajorG(Cf32, IJ, &ab, 2, 2);
    const b = colMajorG(Cf32, IJ, &bb, 2, 2);
    const res = try eigSymGenVectors(Cf32, IJ, testing.allocator, a.asConst(), b.asConst(), .a_bx, .upper);
    defer res.deinit(testing.allocator);
    try testing.expectApproxEqAbs(@as(f32, 1), res.values[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3), res.values[1], 1e-4);
}

test "lstsqSvd: overdetermined full rank (real)" {
    // Fit y = x0 + x1*t through (0,1),(1,2),(2,3): exact x = [1, 1], rank 2.
    var ab = [_]f64{ 1, 1, 1, 0, 1, 2 }; // col-major 3x2: col0=[1,1,1], col1=[0,1,2]
    var bb = [_]f64{ 1, 2, 3 };
    const a = colMajor(IK, &ab, 3, 2);
    const b = colMajor(IJ, &bb, 3, 1);
    const res = try lstsqSvd(f64, IK, IJ, testing.allocator, a.asConst(), b, -1.0);
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), res.rank);
    try testing.expectApproxEqAbs(@as(f64, 1), res.x.at(.{ .k = 0, .j = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), res.x.at(.{ .k = 1, .j = 0 }).*, 1e-12);
    try testing.expect(res.singular_values[0] >= res.singular_values[1]);
    try testing.expect(res.singular_values[1] > 1e-9);
}

test "lstsqSvd: rank-deficient minimum-norm solution (real)" {
    // A = [[1,2],[2,4],[3,6]] (col1 = 2*col0), rank 1. b = col0 = [1,2,3].
    // A x = (x0 + 2 x1) col0; min-norm solution of x0 + 2 x1 = 1 is [0.2, 0.4].
    var ab = [_]f64{ 1, 2, 3, 2, 4, 6 };
    var bb = [_]f64{ 1, 2, 3 };
    const a = colMajor(IK, &ab, 3, 2);
    const b = colMajor(IJ, &bb, 3, 1);
    const res = try lstsqSvd(f64, IK, IJ, testing.allocator, a.asConst(), b, -1.0);
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), res.rank);
    try testing.expectApproxEqAbs(@as(f64, 0.2), res.x.at(.{ .k = 0, .j = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.4), res.x.at(.{ .k = 1, .j = 0 }).*, 1e-12);
    try testing.expect(res.singular_values[1] < 1e-9); // second sv ~ 0
}

test "lstsqSvd: preserves A (row-major)" {
    var ab = [_]f64{ 1, 0, 1, 1, 1, 2 }; // row-major 3x2
    var bb = [_]f64{ 1, 2, 3 };
    const a_before = ab;
    const a = rowMajorG(f64, IK, &ab, 3, 2);
    const b = colMajor(IJ, &bb, 3, 1);
    const res = try lstsqSvd(f64, IK, IJ, testing.allocator, a.asConst(), b, -1.0);
    defer res.deinit(testing.allocator);
    try testing.expectEqualSlices(f64, &a_before, &ab);
    try testing.expectApproxEqAbs(@as(f64, 1), res.x.at(.{ .k = 0, .j = 0 }).*, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), res.x.at(.{ .k = 1, .j = 0 }).*, 1e-12);
}

test "lstsqSvd: complex full rank" {
    // col0=[1,0,1], col1=[0,1,1]; x=[1, i] => b = [1, i, 1+i]. rank 2.
    var ab = [_]Cf{ Cf.init(1, 0), Cf.init(0, 0), Cf.init(1, 0), Cf.init(0, 0), Cf.init(1, 0), Cf.init(1, 0) };
    var bb = [_]Cf{ Cf.init(1, 0), Cf.init(0, 1), Cf.init(1, 1) };
    const a = colMajorC(IK, &ab, 3, 2);
    const b = colMajorC(IJ, &bb, 3, 1);
    const res = try lstsqSvd(Cf, IK, IJ, testing.allocator, a.asConst(), b, -1.0);
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), res.rank);
    const x0 = res.x.at(.{ .k = 0, .j = 0 }).*;
    const x1 = res.x.at(.{ .k = 1, .j = 0 }).*;
    try testing.expectApproxEqAbs(@as(f64, 1), x0.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), x0.im, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), x1.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), x1.im, 1e-12);
}

test "Complex(f32): lstsqSvd routes through cgelsd arm" {
    var ab = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(1, 0), Cf32.init(0, 0), Cf32.init(1, 0), Cf32.init(1, 0) };
    var bb = [_]Cf32{ Cf32.init(1, 0), Cf32.init(0, 1), Cf32.init(1, 1) };
    const a = colMajorG(Cf32, IK, &ab, 3, 2);
    const b = colMajorG(Cf32, IJ, &bb, 3, 1);
    const res = try lstsqSvd(Cf32, IK, IJ, testing.allocator, a.asConst(), b, -1.0);
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), res.rank);
    const x1 = res.x.at(.{ .k = 1, .j = 0 }).*;
    try testing.expectApproxEqAbs(@as(f32, 0), x1.re, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1), x1.im, 1e-4);
}

test "f32: lstsqSvd routes through sgelsd arm" {
    var ab = [_]f32{ 1, 1, 1, 0, 1, 2 };
    var bb = [_]f32{ 1, 2, 3 };
    const a = colMajorG(f32, IK, &ab, 3, 2);
    const b = colMajorG(f32, IJ, &bb, 3, 1);
    const res = try lstsqSvd(f32, IK, IJ, testing.allocator, a.asConst(), b, -1.0);
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), res.rank);
    try testing.expectApproxEqAbs(@as(f32, 1), res.x.at(.{ .k = 0, .j = 0 }).*, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1), res.x.at(.{ .k = 1, .j = 0 }).*, 1e-5);
}
