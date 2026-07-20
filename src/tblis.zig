const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const expect = std.testing.expect;
const Complex = std.math.Complex;

const C = @cImport(@cInclude("tblis_zig.h"));
const TblisScalar = C.tblis_zig_scalar;
const TblisTensor = C.tblis_zig_tensor;
const TblisTypeT = C.zig_type_t;
const TblisReduceT = C.zig_reduce_t;
const c32_tblis = C.scomplex_zig;
const c64_tblis = C.dcomplex_zig;

const arr = @import("named_array.zig");
const idx_ = @import("named_index.zig");
const axis_meta = @import("axis_meta.zig");

pub const Reduce = enum {
    SUM,
    SUM_ABS,
    MAX,
    MAX_ABS,
    MIN,
    MIN_ABS,
    NORM_2,
};

/// Extrema reductions that also expose arg indices.
///
/// For `*_ABS` variants, returned values follow tblis semantics and are
/// magnitudes (non-negative for real scalars).
pub const ArgReduce = enum {
    MAX,
    MAX_ABS,
    MIN,
    MIN_ABS,
};

/// Compile-time guard: `AxisOut` must be a non-empty subset of `AxisIn`.
fn assertAxisSubsetNonEmpty(comptime AxisIn: type, comptime AxisOut: type, comptime caller: []const u8) void {
    const in_names = comptime meta.fieldNames(AxisIn);
    const out_names = comptime meta.fieldNames(AxisOut);

    comptime {
        if (out_names.len == 0)
            @compileError(caller ++ ": AxisOut must be non-empty. Use reduceAll(...) for full reduction.");

        for (out_names) |out_name| {
            var found = false;
            for (in_names) |in_name| {
                if (std.mem.eql(u8, out_name, in_name)) {
                    found = true;
                    break;
                }
            }
            if (!found)
                @compileError(caller ++ ": AxisOut contains axis not present in AxisIn: '" ++ out_name ++ "'");
        }
    }
}

/// Compile-time guard: reductions over `AxisOut` must remove at least one axis.
fn assertHasReducedAxis(comptime AxisIn: type, comptime AxisOut: type, comptime caller: []const u8) void {
    const ReducedAxis = comptime axis_meta.Difference(AxisIn, AxisOut);
    comptime {
        if (meta.fields(ReducedAxis).len == 0)
            @compileError(caller ++ ": no reduced axis. Use reduceAll(...) for full reduction, or pass a strict subset AxisOut.");
    }
}

/// Runtime guard: on kept axes (`AxisOut`), output extents must match input extents.
fn assertOutputShapeMatchesInput(comptime AxisOut: type, in_shape: anytype, out_shape: anytype, comptime caller: []const u8, comptime out_label: []const u8) void {
    inline for (comptime meta.fieldNames(AxisOut)) |name| {
        if (@field(out_shape, name) != @field(in_shape, name))
            @panic(caller ++ ": " ++ out_label ++ " output shape mismatch");
    }
}

/// Build the per-output reduction shape by collapsing kept axes to extent `1`.
fn reducedShapeForOutputAxes(comptime AxisOut: type, in_shape: anytype) @TypeOf(in_shape) {
    var reduced_shape = in_shape;
    inline for (comptime meta.fieldNames(AxisOut)) |name| {
        @field(reduced_shape, name) = 1;
    }
    return reduced_shape;
}

/// Create the non-allocating reduced view used for one `out_key` evaluation.
/// Computes the per-key base offset from kept-axis indices and strides.
fn reducedViewForOutputKey(
    comptime AxisIn: type,
    comptime AxisOut: type,
    comptime Scalar: type,
    a: arr.NamedArrayConst(AxisIn, Scalar),
    reduced_shape: @TypeOf(a.idx.shape),
    out_key: anytype,
) arr.NamedArrayConst(AxisIn, Scalar) {
    var base_offset_signed: isize = @intCast(a.idx.offset);
    inline for (comptime meta.fieldNames(AxisOut)) |name| {
        const idx_val: isize = @intCast(@field(out_key, name));
        const stride_val: isize = @field(a.idx.strides, name);
        base_offset_signed += idx_val * stride_val;
    }
    const base_offset: usize = @intCast(base_offset_signed);

    return .{
        .idx = .{
            .shape = reduced_shape,
            .strides = a.idx.strides,
            .offset = base_offset,
        },
        .buf = a.buf,
    };
}

/// Derive `AxisOut` extents from an input shape by copying only kept-axis sizes.
fn outputShapeFromInput(comptime AxisOut: type, in_shape: anytype) idx_.NamedIndex(AxisOut).Axes {
    const OutIndex = idx_.NamedIndex(AxisOut);
    var out_shape: OutIndex.Axes = undefined;
    inline for (comptime meta.fieldNames(AxisOut)) |name| {
        @field(out_shape, name) = @field(in_shape, name);
    }
    return out_shape;
}

pub fn ReduceWithArgResult(comptime AxisOut: type, comptime Scalar: type, comptime ArgScalar: type) type {
    return struct {
        values: arr.NamedArray(AxisOut, Scalar),
        args: arr.NamedArray(AxisOut, ArgScalar),

        pub fn deinit(self: *const @This(), allocator: std.mem.Allocator) void {
            self.values.deinit(allocator);
            self.args.deinit(allocator);
        }
    };
}

/// Map arg-reduction variants to their value-only reduction counterpart.
fn argReduceToReduce(comptime op: ArgReduce) Reduce {
    return switch (op) {
        .MAX => .MAX,
        .MAX_ABS => .MAX_ABS,
        .MIN => .MIN,
        .MIN_ABS => .MIN_ABS,
    };
}

/// Assign `B <- alpha A + beta B`
/// where `A, B` have the same shape and `alpha, beta` are optional scalars.
/// No implicit broadcasting.
/// Passing different shapes is detectable illegal behavior.
///
/// Unclear behavior as of now:
/// - are A and B allowed to alias?
pub fn add(
    comptime Axis: type,
    comptime Scalar: type,
    a: arr.NamedArrayConst(Axis, Scalar),
    b: arr.NamedArray(Axis, Scalar),
    opt: struct { scale_a: Scalar = one(Scalar), scale_b: Scalar = one(Scalar) },
) void {
    const idx_str = comptime index_strings(&.{Axis})[0];
    const rank = idx_str.len;
    assert(meta.eql(a.idx.shape, b.idx.shape));

    var a_mem: TblisTensorBuf(rank) = undefined;
    var a_tensor = toTblisTensor(Axis, Scalar, a, &a_mem);
    a_tensor.scalar = init_scalar(Scalar, opt.scale_a);

    var b_mem: TblisTensorBuf(rank) = undefined;
    var b_tensor = toTblisTensor(Axis, Scalar, b, &b_mem);
    b_tensor.scalar = init_scalar(Scalar, opt.scale_b);

    C.tblis_zig_tensor_add(null, null, &a_tensor, idx_str.ptr, &b_tensor, idx_str.ptr);
}

test "add strided" {
    const IJ = enum { i, j };
    const T = f32;
    const Arr = arr.NamedArray(IJ, T);
    const ArrConst = arr.NamedArrayConst(IJ, T);

    const a_buf = [_]T{
        1, 2, 3,
        4, 5, 6,
    };
    const a = ArrConst.init(.initContiguous(.{ .i = 2, .j = 3 }), &a_buf);
    var b_buf = [_]T{ 1, 2, 10, 20, 100, 200, 1_000, 2_000, 10_000, 20_000, 100_000, 200_000 };
    const b = Arr.init(.{ .shape = .{ .i = 2, .j = 3 }, .strides = .{ .i = 2, .j = 4 } }, &b_buf);

    const expected = [_]T{ 2, 2, 14, 20, 102, 200, 1_005, 2_000, 10_003, 20_000, 100_006, 200_000 };
    add(IJ, T, a, b, .{});

    try std.testing.expectEqualDeep(&expected, &b_buf);
}

test "add mat + row broadcast" {
    // const J = enum { j };
    const IJ = enum { i, j };
    const T = f32;
    const MatIJ = arr.NamedArray(IJ, T);
    const RowJ = arr.NamedArrayConst(IJ, T);

    const a_buf = [_]T{ 1, 10, 100 };
    var b_buf = [_]T{
        1, 2, 3,
        4, 5, 6,
    };
    var a = RowJ.init(.initContiguous(.{ .i = 1, .j = 3 }), &a_buf);
    const b = MatIJ.init(.initContiguous(.{ .i = 2, .j = 3 }), &b_buf);
    a.idx = a.idx.broadcastAxis(.i, b.idx.shape.i);

    const expected = [_]T{
        2, 12, 103,
        5, 15, 106,
    };
    add(IJ, T, a, b, .{});

    try std.testing.expectEqualDeep(&expected, &b_buf);
}

test "add honors non-zero offset views" {
    const IJ = enum { i, j };
    const T = f32;

    const a_buf = [_]T{
        1, 2, 3,
        4, 5, 6,
    };
    var b_buf = [_]T{
        10, 20, 30,
        40, 50, 60,
    };

    var a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &a_buf);
    var b = arr.NamedArray(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &b_buf);

    a.idx = a.idx.sliceAxis(.j, 1, 3);
    b.idx = b.idx.sliceAxis(.j, 1, 3);

    add(IJ, T, a, b, .{});

    const expected = [_]T{
        10, 22, 33,
        40, 55, 66,
    };
    try std.testing.expectEqualDeep(&expected, &b_buf);
}

/// Return A_i * B_i
/// where `A, B` have the same shape.
/// No implicit broadcasting.
/// Passing different shapes is detectable illegal behavior.
pub fn dot(
    comptime AxisA: type,
    // comptime AxisB: type,
    comptime Scalar: type,
    a: arr.NamedArrayConst(AxisA, Scalar),
    b: arr.NamedArrayConst(AxisA, Scalar),
) Scalar {
    const a_idx = comptime index_strings(&.{AxisA})[0];
    assert(meta.eql(a.idx.shape, b.idx.shape));
    const rank = a_idx.len;

    var a_mem: TblisTensorBuf(rank) = undefined;
    const a_tensor = toTblisTensor(AxisA, Scalar, a, &a_mem);
    var b_mem: TblisTensorBuf(rank) = undefined;
    const b_tensor = toTblisTensor(AxisA, Scalar, b, &b_mem);
    var result = init_scalar(Scalar, undefined);

    C.tblis_zig_tensor_dot(
        null,
        null,
        &a_tensor,
        a_idx.ptr,
        &b_tensor,
        a_idx.ptr,
        &result,
    );
    return get_scalar_val(Scalar, result);
}

test "dot row col" {
    const T = f32;
    const IJ = enum { i, j };
    const m = 1;
    const n = 4;
    const a_data = [m * n]T{ 1, 2, 3, 4 };
    const b_data = [n * m]T{ 1, 10, 100, 1000 };
    const a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = m, .j = n }), &a_data);
    const b = arr.NamedArrayConst(IJ, T).init(.{ .shape = .{ .j = n, .i = m }, .strides = .{ .j = 1, .i = n } }, &b_data);

    const expected: T = 4321;
    const actual = dot(IJ, T, a, b);

    try std.testing.expectEqual(expected, actual);
}

/// Assign `C_k <- C_k + A_i * B_j`
/// where `A, B, C` have compatible shapes.
/// No implicit broadcasting.
/// Passing incompatible shapes is detectable illegal behavior.
pub fn mult(
    comptime AxisA: type,
    comptime AxisB: type,
    comptime AxisC: type,
    comptime Scalar: type,
    a: arr.NamedArrayConst(AxisA, Scalar),
    b: arr.NamedArrayConst(AxisB, Scalar),
    c: arr.NamedArray(AxisC, Scalar),
) void {
    const rank_a = comptime enumLen(AxisA);
    const rank_b = comptime enumLen(AxisB);
    const rank_c = comptime enumLen(AxisC);
    const idx_strs = comptime index_strings(&.{ AxisA, AxisB, AxisC });
    assert(shapesAreConsistent(.{ a.idx.shape, b.idx.shape, c.idx.shape }));

    var a_mem: TblisTensorBuf(rank_a) = undefined;
    var b_mem: TblisTensorBuf(rank_b) = undefined;
    var c_mem: TblisTensorBuf(rank_c) = undefined;
    const a_tensor = toTblisTensor(AxisA, Scalar, a, &a_mem);
    const b_tensor = toTblisTensor(AxisB, Scalar, b, &b_mem);
    var c_tensor = toTblisTensor(AxisC, Scalar, c, &c_mem);

    C.tblis_zig_tensor_mult(
        null,
        null,
        &a_tensor,
        idx_strs[0].ptr,
        &b_tensor,
        idx_strs[1].ptr,
        &c_tensor,
        idx_strs[2].ptr,
    );
}

test "mult ij jk" {
    const IJ = enum { i, j };
    const JK = enum { j, k };
    const IK = enum { i, k };
    const T = f64;

    const m = 2;
    const n = 2;
    const k = 3;

    const a_data = [_]T{ 1, 2, 3, 4 }; // row-major (i,j): strides i=k (n), j=1
    const b_data = [_]T{ 1, 10, 100, 1000, 10_000, 100_000 }; // col-major (j,k): strides j=1, k=n
    var c_data: [m * k]T = undefined;

    const a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = m, .j = n }), &a_data);
    const b = arr.NamedArrayConst(JK, T).init(.{ .shape = .{ .j = n, .k = k }, .strides = .{ .j = 1, .k = n } }, &b_data);
    const c = arr.NamedArray(IK, T).init(.{ .shape = .{ .i = m, .k = k }, .strides = .{ .i = k, .k = 1 } }, &c_data);

    mult(IJ, JK, IK, T, a, b, c);

    const expected = [_]T{ 21, 2100, 210_000, 43, 4300, 430_000 };
    try std.testing.expectEqualDeep(expected, c_data);
}

test "mult i jk -> ijk" {
    const I = enum { i };
    const JK = enum { j, k };
    const IJK = enum { i, j, k };
    const T = f64;

    const m = 2; // i
    const n = 2; // j
    const k = 3; // k

    var a_data = [_]T{ 1.0, 2.0 }; // contiguous 1D over i
    var b_data = [_]T{
        // row-major over (j,k): strides j=k, k=1
        10.0,  20.0,  30.0,
        100.0, 200.0, 300.0,
    };
    var c_data: [m * n * k]T = undefined;

    const a = arr.NamedArrayConst(I, T).init(.{ .shape = .{ .i = m }, .strides = .{ .i = 1 } }, &a_data);
    const b = arr.NamedArrayConst(JK, T).init(.{ .shape = .{ .j = n, .k = k }, .strides = .{ .j = k, .k = 1 } }, &b_data);
    const c = arr.NamedArray(IJK, T).init(.{ .shape = .{ .i = m, .j = n, .k = k }, .strides = .{ .i = n * k, .j = k, .k = 1 } }, &c_data);

    mult(I, JK, IJK, T, a, b, c);

    const expected = [_]T{
        // i = 0 => a[0] = 1
        10.0, 20.0, 30.0, 100.0, 200.0, 300.0,
        // i = 1 => a[1] = 2
        20.0, 40.0, 60.0, 200.0, 400.0, 600.0,
    };
    try std.testing.expectEqualDeep(expected, c_data);
}

// Return a reduction result over `a` using `op`.
// The returned struct contains the reduced scalar value and the index (one per axis)
// at which the extremum occurred for MAX/MIN-type reductions. For SUM/NORM variants,
// the index content is undefined and can be ignored.
pub fn reduceAll(
    comptime AxisA: type,
    comptime Scalar: type,
    op: Reduce,
    a: arr.NamedArrayConst(AxisA, Scalar),
) struct { value: Scalar, index: @TypeOf(a.idx.shape) } {
    const a_idx = comptime index_strings(&.{AxisA})[0];
    const rank = a_idx.len;

    var a_mem: TblisTensorBuf(rank) = undefined;
    var a_tensor = toTblisTensor(AxisA, Scalar, a, &a_mem);

    var out_scalar = init_scalar(Scalar, undefined);
    var out_index: [rank]C.zig_len_type = undefined;

    const c_op: TblisReduceT = switch (op) {
        .SUM => C.ZIG_REDUCE_SUM,
        .SUM_ABS => C.ZIG_REDUCE_SUM_ABS,
        .MAX => C.ZIG_REDUCE_MAX,
        .MAX_ABS => C.ZIG_REDUCE_MAX_ABS,
        .MIN => C.ZIG_REDUCE_MIN,
        .MIN_ABS => C.ZIG_REDUCE_MIN_ABS,
        .NORM_2 => C.ZIG_REDUCE_NORM_2,
    };

    C.tblis_zig_tensor_reduce(
        null,
        null,
        c_op,
        &a_tensor,
        a_idx.ptr,
        &out_scalar,
        &out_index,
    );

    var index: @TypeOf(a.idx.shape) = undefined;
    inline for (comptime meta.fieldNames(AxisA), out_index) |name, val| {
        @field(index, name) = switch (op) {
            .MAX, .MAX_ABS, .MIN, .MIN_ABS => @intCast(val),
            else => 0,
        };
    }

    return .{
        .value = get_scalar_val(Scalar, out_scalar),
        .index = index,
    };
}

/// Reduce `a` over all axes absent from `AxisOut`, writing both extremum values
/// and reduced-axis arg indices in a single pass.
///
/// `out_values` and `out_args` must both have per-axis extents matching `a` on
/// the kept axes (`AxisOut`).
pub fn reduceWithArgInto(
    comptime AxisIn: type,
    comptime AxisOut: type,
    comptime Scalar: type,
    comptime op: ArgReduce,
    a: arr.NamedArrayConst(AxisIn, Scalar),
    out_values: arr.NamedArray(AxisOut, Scalar),
    out_args: arr.NamedArray(AxisOut, axis_meta.DifferenceAxesStruct(AxisIn, AxisOut)),
) void {
    assertAxisSubsetNonEmpty(AxisIn, AxisOut, "reduceWithArgInto");
    assertHasReducedAxis(AxisIn, AxisOut, "reduceWithArgInto");
    const ReducedAxis = comptime axis_meta.Difference(AxisIn, AxisOut);
    const reduced_names = comptime meta.fieldNames(ReducedAxis);
    const ArgScalar = axis_meta.DifferenceAxesStruct(AxisIn, AxisOut);
    const value_op = comptime argReduceToReduce(op);

    assertOutputShapeMatchesInput(AxisOut, a.idx.shape, out_values.idx.shape, "reduceWithArgInto", "values");
    assertOutputShapeMatchesInput(AxisOut, a.idx.shape, out_args.idx.shape, "reduceWithArgInto", "args");

    const reduced_shape = reducedShapeForOutputAxes(AxisOut, a.idx.shape);

    var out_keys = out_values.idx.iterKeys();
    while (out_keys.next()) |out_key| {
        const reduced_view = reducedViewForOutputKey(AxisIn, AxisOut, Scalar, a, reduced_shape, out_key);

        const reduced = reduceAll(AxisIn, Scalar, value_op, reduced_view);
        out_values.at(out_key).* = reduced.value;

        // tblis currently reports a linear offset for extrema location in the
        // first index slot. Recover the full per-axis key by searching the
        // reduced view's key space for the matching linear offset.
        const in_names = comptime meta.fieldNames(AxisIn);
        const raw_linear: usize = @field(reduced.index, in_names[0]);

        var idx_relative = reduced_view.idx;
        idx_relative.offset = 0;

        var found_key: ?@TypeOf(reduced_view.idx).Axes = null;
        var candidate_keys = idx_relative.iterKeys();
        while (candidate_keys.next()) |candidate_key| {
            if (idx_relative.linear(candidate_key) == raw_linear) {
                found_key = candidate_key;
                break;
            }
        }
        const arg_key = found_key orelse @panic("reduceWithArgInto: failed to decode arg index");

        var arg_value: ArgScalar = undefined;
        inline for (reduced_names) |name| {
            @field(arg_value, name) = @field(arg_key, name);
        }
        out_args.at(out_key).* = arg_value;
    }
}

/// Allocate and return both extremum values and reduced-axis arg indices for
/// reducing `a` over all axes absent from `AxisOut`.
pub fn reduceWithArgAlloc(
    comptime AxisIn: type,
    comptime AxisOut: type,
    comptime Scalar: type,
    allocator: std.mem.Allocator,
    comptime op: ArgReduce,
    a: arr.NamedArrayConst(AxisIn, Scalar),
) !ReduceWithArgResult(AxisOut, Scalar, axis_meta.DifferenceAxesStruct(AxisIn, AxisOut)) {
    assertAxisSubsetNonEmpty(AxisIn, AxisOut, "reduceWithArgAlloc");
    assertHasReducedAxis(AxisIn, AxisOut, "reduceWithArgAlloc");

    const ArgScalar = axis_meta.DifferenceAxesStruct(AxisIn, AxisOut);
    const out_shape = outputShapeFromInput(AxisOut, a.idx.shape);

    const values = try arr.NamedArray(AxisOut, Scalar).initAlloc(allocator, out_shape);
    errdefer values.deinit(allocator);

    const args = try arr.NamedArray(AxisOut, ArgScalar).initAlloc(allocator, out_shape);
    errdefer args.deinit(allocator);

    reduceWithArgInto(AxisIn, AxisOut, Scalar, op, a, values, args);

    return .{ .values = values, .args = args };
}

/// Reduce `a` over all axes in `AxisIn` that are absent from `AxisOut`, writing
/// the resulting values into `out`.
///
/// This function does not allocate. `out` must have the same per-axis extents as
/// `a` on the kept axes (`AxisOut`).
pub fn reduceInto(
    comptime AxisIn: type,
    comptime AxisOut: type,
    comptime Scalar: type,
    op: Reduce,
    a: arr.NamedArrayConst(AxisIn, Scalar),
    out: arr.NamedArray(AxisOut, Scalar),
) void {
    assertAxisSubsetNonEmpty(AxisIn, AxisOut, "reduceInto");
    assertOutputShapeMatchesInput(AxisOut, a.idx.shape, out.idx.shape, "reduceInto", "output");

    const reduced_shape = reducedShapeForOutputAxes(AxisOut, a.idx.shape);

    var out_keys = out.idx.iterKeys();
    while (out_keys.next()) |out_key| {
        const reduced_view = reducedViewForOutputKey(AxisIn, AxisOut, Scalar, a, reduced_shape, out_key);
        out.at(out_key).* = reduceAll(AxisIn, Scalar, op, reduced_view).value;
    }
}

/// Allocate and return the values of reducing `a` over all axes absent from
/// `AxisOut` using `op`.
pub fn reduceAlloc(
    comptime AxisIn: type,
    comptime AxisOut: type,
    comptime Scalar: type,
    allocator: std.mem.Allocator,
    op: Reduce,
    a: arr.NamedArrayConst(AxisIn, Scalar),
) !arr.NamedArray(AxisOut, Scalar) {
    assertAxisSubsetNonEmpty(AxisIn, AxisOut, "reduceAlloc");
    const out_shape = outputShapeFromInput(AxisOut, a.idx.shape);

    const out = try arr.NamedArray(AxisOut, Scalar).initAlloc(allocator, out_shape);
    reduceInto(AxisIn, AxisOut, Scalar, op, a, out);
    return out;
}

test "reduceAll i" {
    const T = f64;
    const I = enum { i };
    const data = [_]T{ 1, -2, 3, -4, 5 };
    const a = arr.NamedArrayConst(I, T).init(.initContiguous(.{ .i = data.len }), &data);

    const r_sum = reduceAll(I, T, .SUM, a);
    try std.testing.expectEqual(@as(T, 3.0), r_sum.value);

    const r_sum_abs = reduceAll(I, T, .SUM_ABS, a);
    try std.testing.expectEqual(@as(T, 15.0), r_sum_abs.value);

    const r_max = reduceAll(I, T, .MAX, a);
    try std.testing.expectEqual(@as(T, 5.0), r_max.value);
    try std.testing.expectEqual(@as(C.zig_len_type, 4), r_max.index.i);
}

test "reduceAll honors non-zero offset view" {
    const T = f64;
    const IJ = enum { i, j };

    const data = [_]T{ 1, 2, 3, 4, 5, 6 };
    var a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &data);
    a.idx = a.idx.sliceAxis(.j, 1, 3);

    const r_sum = reduceAll(IJ, T, .SUM, a);
    try std.testing.expectEqual(@as(T, 16), r_sum.value);

    const r_max = reduceAll(IJ, T, .MAX, a);
    try std.testing.expectEqual(@as(T, 6), r_max.value);
}

test "reduceInto supports non-sum op" {
    const T = f64;
    const IJ = enum { i, j };
    const I = enum { i };

    const data = [_]T{ 1, 2, 3, 4, 5, 6 };
    const a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &data);

    var out_buf: [2]T = undefined;
    const out = arr.NamedArray(I, T).init(.initContiguous(.{ .i = 2 }), &out_buf);

    reduceInto(IJ, I, T, .MAX, a, out);
    try std.testing.expectEqual(@as(T, 3), out.scalarAt(.{ .i = 0 }));
    try std.testing.expectEqual(@as(T, 6), out.scalarAt(.{ .i = 1 }));
}

test "reduceInto handles negative reduced stride" {
    const T = f64;
    const IJ = enum { i, j };
    const I = enum { i };

    const data = [_]T{ 1, 2, 3, 4, 5, 6 };
    var a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &data);
    a.idx = a.idx.strideAxis(.j, -1);

    var out_buf: [2]T = undefined;
    const out = arr.NamedArray(I, T).init(.initContiguous(.{ .i = 2 }), &out_buf);

    reduceInto(IJ, I, T, .SUM, a, out);
    try std.testing.expectEqual(@as(T, 6), out.scalarAt(.{ .i = 0 }));
    try std.testing.expectEqual(@as(T, 15), out.scalarAt(.{ .i = 1 }));
}

test "reduceInto sum writes into strided output" {
    const T = f64;
    const IJ = enum { i, j };
    const I = enum { i };

    const data = [_]T{ 1, 2, 3, 4, 5, 6 };
    const a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &data);

    var out_buf = [_]T{ -1, -1, -1, -1 };
    const out = arr.NamedArray(I, T).init(.{ .shape = .{ .i = 2 }, .strides = .{ .i = 2 } }, &out_buf);

    reduceInto(IJ, I, T, .SUM, a, out);

    try std.testing.expectEqual(@as(T, 6), out_buf[0]);
    try std.testing.expectEqual(@as(T, 15), out_buf[2]);
    try std.testing.expectEqual(@as(T, -1), out_buf[1]);
    try std.testing.expectEqual(@as(T, -1), out_buf[3]);
}

test "reduceAlloc sum over selected axes with reordered output axes" {
    const T = f64;
    const IJK = enum { i, j, k };
    const KI = enum { k, i };

    const data = [_]T{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };

    const a = arr.NamedArrayConst(IJK, T).init(.initContiguous(.{ .i = 2, .j = 2, .k = 3 }), &data);

    const out = try reduceAlloc(IJK, KI, T, std.testing.allocator, .SUM, a);
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), out.idx.shape.k);
    try std.testing.expectEqual(@as(usize, 2), out.idx.shape.i);

    try std.testing.expectEqual(@as(T, 5), out.scalarAt(.{ .k = 0, .i = 0 }));
    try std.testing.expectEqual(@as(T, 17), out.scalarAt(.{ .k = 0, .i = 1 }));
    try std.testing.expectEqual(@as(T, 7), out.scalarAt(.{ .k = 1, .i = 0 }));
    try std.testing.expectEqual(@as(T, 19), out.scalarAt(.{ .k = 1, .i = 1 }));
    try std.testing.expectEqual(@as(T, 9), out.scalarAt(.{ .k = 2, .i = 0 }));
    try std.testing.expectEqual(@as(T, 21), out.scalarAt(.{ .k = 2, .i = 1 }));
}

test "reduceAlloc forwards non-sum op" {
    const T = f64;
    const IJK = enum { i, j, k };
    const KI = enum { k, i };

    const data = [_]T{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };

    const a = arr.NamedArrayConst(IJK, T).init(.initContiguous(.{ .i = 2, .j = 2, .k = 3 }), &data);

    const out = try reduceAlloc(IJK, KI, T, std.testing.allocator, .MAX, a);
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(T, 4), out.scalarAt(.{ .k = 0, .i = 0 }));
    try std.testing.expectEqual(@as(T, 10), out.scalarAt(.{ .k = 0, .i = 1 }));
    try std.testing.expectEqual(@as(T, 5), out.scalarAt(.{ .k = 1, .i = 0 }));
    try std.testing.expectEqual(@as(T, 11), out.scalarAt(.{ .k = 1, .i = 1 }));
    try std.testing.expectEqual(@as(T, 6), out.scalarAt(.{ .k = 2, .i = 0 }));
    try std.testing.expectEqual(@as(T, 12), out.scalarAt(.{ .k = 2, .i = 1 }));
}

test "reduceInto sum with all axes kept is identity" {
    const T = f64;
    const IJ = enum { i, j };

    const data = [_]T{ 1, 2, 3, 4, 5, 6 };
    const a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &data);

    var out_buf: [data.len]T = undefined;
    const out = arr.NamedArray(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &out_buf);

    reduceInto(IJ, IJ, T, .SUM, a, out);
    try std.testing.expectEqualDeep(data, out_buf);
}

test "reduceWithArgInto writes values and reduced-axis indices" {
    const T = f64;
    const IJ = enum { i, j };
    const I = enum { i };
    const Arg = axis_meta.DifferenceAxesStruct(IJ, I);

    const data = [_]T{ 1, 2, 3, 4, 5, 6 };
    const a = arr.NamedArrayConst(IJ, T).init(.initContiguous(.{ .i = 2, .j = 3 }), &data);

    var out_val_buf: [2]T = undefined;
    var out_arg_buf: [2]Arg = undefined;

    const out_values = arr.NamedArray(I, T).init(.initContiguous(.{ .i = 2 }), &out_val_buf);
    const out_args = arr.NamedArray(I, Arg).init(.initContiguous(.{ .i = 2 }), &out_arg_buf);

    reduceWithArgInto(IJ, I, T, .MAX, a, out_values, out_args);

    try std.testing.expectEqual(@as(T, 3), out_values.scalarAt(.{ .i = 0 }));
    try std.testing.expectEqual(@as(T, 6), out_values.scalarAt(.{ .i = 1 }));

    try std.testing.expectEqual(@as(usize, 2), out_args.scalarAt(.{ .i = 0 }).j);
    try std.testing.expectEqual(@as(usize, 2), out_args.scalarAt(.{ .i = 1 }).j);
}

test "reduceWithArgAlloc returns values and args in one pass" {
    const T = f64;
    const IJK = enum { i, j, k };
    const KI = enum { k, i };

    const data = [_]T{
        5,  2,
        -1, 4,
        3,  -0.5,
        -7, 9,
        6,  -2,
        1,  -8,
    };

    const a = arr.NamedArrayConst(IJK, T).init(.initContiguous(.{ .i = 2, .j = 3, .k = 2 }), &data);

    var out = try reduceWithArgAlloc(IJK, KI, T, std.testing.allocator, .MIN_ABS, a);
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(T, 1), out.values.scalarAt(.{ .k = 0, .i = 0 }));
    try std.testing.expectEqual(@as(T, 1), out.values.scalarAt(.{ .k = 0, .i = 1 }));
    try std.testing.expectEqual(@as(T, 0.5), out.values.scalarAt(.{ .k = 1, .i = 0 }));
    try std.testing.expectEqual(@as(T, 2), out.values.scalarAt(.{ .k = 1, .i = 1 }));

    try std.testing.expectEqual(@as(usize, 1), out.args.scalarAt(.{ .k = 0, .i = 0 }).j);
    try std.testing.expectEqual(@as(usize, 2), out.args.scalarAt(.{ .k = 0, .i = 1 }).j);
    try std.testing.expectEqual(@as(usize, 2), out.args.scalarAt(.{ .k = 1, .i = 0 }).j);
    try std.testing.expectEqual(@as(usize, 1), out.args.scalarAt(.{ .k = 1, .i = 1 }).j);
}

/// Update `A <- alpha A` where `alpha` is a scalar.
pub fn scale(
    comptime AxisA: type,
    comptime Scalar: type,
    alpha: Scalar,
    a: arr.NamedArray(AxisA, Scalar),
) void {
    const a_idx = comptime index_strings(&.{AxisA})[0];
    const rank = a_idx.len;

    var a_mem: TblisTensorBuf(rank) = undefined;
    var a_tensor = toTblisTensor(AxisA, Scalar, a, &a_mem);
    a_tensor.scalar = init_scalar(Scalar, alpha);

    C.tblis_zig_tensor_scale(
        null,
        null,
        &a_tensor,
        a_idx.ptr,
    );
}

test "scale" {
    const T = Complex(f64);
    const IJKL = enum { i, j, k, l };
    var a_data = [_]T{
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 1.0, .im = -1.0 },
        .{ .re = 0.0, .im = 1.0 },
    };
    const a = arr.NamedArray(IJKL, T).init(.{
        .shape = .{ .i = 1, .j = 1, .k = 3, .l = 1 },
        .strides = .{ .i = 6290, .j = 19348, .k = 1, .l = 6890000 },
    }, &a_data);

    scale(IJKL, T, .init(1.0, 1.0), a);

    const expected = [_]T{
        .init(1.0, 1.0),
        .init(2.0, 0.0),
        .init(-1.0, 1.0),
    };
    try std.testing.expectEqualDeep(expected, a_data);
}

// Update `A <- alpha` where `A` is a tensor and `alpha` is a scalar.
pub fn set(
    comptime AxisA: type,
    comptime Scalar: type,
    alpha: Scalar,
    a: arr.NamedArray(AxisA, Scalar),
) void {
    const a_idx = comptime index_strings(&.{AxisA})[0];
    const rank = a_idx.len;

    var a_mem: TblisTensorBuf(rank) = undefined;
    var a_tensor = toTblisTensor(AxisA, Scalar, a, &a_mem);
    const alpha_scalar = init_scalar(Scalar, alpha);

    C.tblis_zig_tensor_set(
        null,
        null,
        &alpha_scalar,
        &a_tensor,
        a_idx.ptr,
    );
}

test "set f32 contiguous 1d" {
    const T = f32;
    const I = enum { i };
    var data: [5]T = undefined;
    @memset(&data, 0.0);

    const a = arr.NamedArray(I, T).init(.{ .shape = .{ .i = data.len }, .strides = .{ .i = 1 } }, &data);

    set(I, T, 3.5, a);

    const expected = [_]T{3.5} ** 5;
    try std.testing.expectEqualDeep(expected, data);
}

test "set complex strided 3d with gaps" {
    const T = Complex(f64);
    const IJK = enum { i, j, k };
    const m = 2;
    const n = 2;
    const p = 2;

    // Layout: simulate padding by placing planes 8 apart, rows 4 apart, cols 1 apart.
    // Only 8 logical elements, but allocate a larger buffer and verify non-accessed padding unchanged.
    const logical = [_]usize{ 0, 1, 4, 5, 8, 9, 12, 13 };

    var buf: [16]T = undefined;
    @memset(&buf, .init(0, 0));

    const a = arr.NamedArray(IJK, T).init(.{ .shape = .{ .i = m, .j = n, .k = p }, .strides = .{ .i = 8, .j = 4, .k = 1 } }, &buf);

    set(IJK, T, one(T), a);

    // Verify logical positions were modified, and others were not.
    for (0..16) |i| {
        const is_logical = std.mem.containsAtLeastScalar(usize, &logical, 1, i);
        const expected = if (is_logical) one(T) else T.init(0, 0);
        try std.testing.expectEqual(expected, buf[i]);
    }
}

// Update `A <- beta * A + alpha` where `alpha` and `beta` are scalars.
// `beta` defaults to 1.
pub fn shift(
    comptime AxisA: type,
    comptime Scalar: type,
    alpha: Scalar,
    a: arr.NamedArray(AxisA, Scalar),
    opt: struct { scale_a: Scalar = one(Scalar) },
) void {
    const a_idx = comptime index_strings(&.{AxisA})[0];
    const rank = a_idx.len;

    var a_mem: TblisTensorBuf(rank) = undefined;
    var a_tensor = toTblisTensor(AxisA, Scalar, a, &a_mem);
    a_tensor.scalar = init_scalar(Scalar, opt.scale_a);
    const alpha_scalar = init_scalar(Scalar, alpha);

    C.tblis_zig_tensor_shift(
        null,
        null,
        &alpha_scalar,
        &a_tensor,
        a_idx.ptr,
    );
}

test "shift f32 contiguous 1d" {
    const T = f32;
    const I = enum { i };
    var data: [5]T = .{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    const a = arr.NamedArray(I, T).init(.initContiguous(.{ .i = data.len }), &data);

    shift(I, T, 3.5, a, .{ .scale_a = 2.0 });

    const expected = [_]T{ 5.5, 7.5, 9.5, 11.5, 13.5 };
    try std.testing.expectEqualDeep(expected, data);
}

test "shift complex strided 3d with gaps" {
    const T = Complex(f64);
    const IJK = enum { i, j, k };
    const m = 2;
    const n = 2;
    const p = 2;

    // Layout with padding: planes 8 apart, rows 4 apart, cols 1 apart.
    const logical = [_]usize{ 0, 1, 4, 5, 8, 9, 12, 13 };

    var buf: [16]T = .{
        // plane 0 (indices 0..7)
        T.init(0, 0), T.init(1, 0), T.init(0, 0), T.init(0, 0),
        T.init(2, 0), T.init(3, 0), T.init(0, 0), T.init(0, 0),
        // plane 1 (indices 8..15)
        T.init(4, 0), T.init(5, 0), T.init(0, 0), T.init(0, 0),
        T.init(6, 0), T.init(7, 0), T.init(0, 0), T.init(0, 0),
    };

    const a = arr.NamedArray(IJK, T).init(.{ .shape = .{ .i = m, .j = n, .k = p }, .strides = .{ .i = 8, .j = 4, .k = 1 } }, &buf);

    shift(IJK, T, one(T), a, .{});

    // Verify logical positions were incremented by alpha and padding unchanged.
    for (0..16) |i| {
        const is_logical = std.mem.containsAtLeastScalar(usize, &logical, 1, i);
        const before = switch (i) {
            0 => T.init(0, 0),
            1 => T.init(1, 0),
            4 => T.init(2, 0),
            5 => T.init(3, 0),
            8 => T.init(4, 0),
            9 => T.init(5, 0),
            12 => T.init(6, 0),
            13 => T.init(7, 0),
            else => T.init(0, 0),
        };
        const expected = if (is_logical) T.init(before.re + 1.0, before.im) else before;
        try std.testing.expectEqual(expected, buf[i]);
    }
}

fn shapesAreConsistent(shapes: anytype) bool {
    _ = idx_.resolveDimensions(shapes) catch {
        return false;
    };
    return true;
}

fn TblisTensorBuf(comptime rank: usize) type {
    return struct {
        len: [rank]C.zig_len_type,
        stride: [rank]C.zig_stride_type,
    };
}

fn toTblisTensor(comptime AxisA: type, comptime T: type, a: anytype, mem: *TblisTensorBuf(enumLen(AxisA))) TblisTensor {
    const axis_names = comptime meta.fieldNames(AxisA);
    const rank = axis_names.len;

    inline for (axis_names, 0..) |name, i| {
        mem.len[i] = @intCast(@field(a.idx.shape, name));
        mem.stride[i] = @intCast(@field(a.idx.strides, name));
    }

    const base_ptr: [*]T = @constCast(a.buf.ptr) + a.idx.offset;
    const a_tensor = init_tensor(T, rank, &mem.len, &mem.stride, base_ptr);
    return a_tensor;
}

fn enumLen(comptime Axis: type) usize {
    return meta.fields(Axis).len;
}

fn init_type_t(comptime T: type) TblisTypeT {
    return switch (T) {
        f32 => C.ZIG_TYPE_SINGLE,
        f64 => C.ZIG_TYPE_DOUBLE,
        Complex(f32) => C.ZIG_TYPE_SCOMPLEX,
        Complex(f64) => C.ZIG_TYPE_DCOMPLEX,
        else => @compileError("init_type_t: T must be f32, f64 or Complex(...)"),
    };
}

fn init_scalar(comptime T: type, val: T) TblisScalar {
    const type_ = init_type_t(T);
    const data: C.zig_scalar = switch (T) {
        f32 => .{ .s = val },
        f64 => .{ .d = val },
        Complex(f32) => .{ .c = c32_tblis{ .re = val.re, .im = val.im } },
        Complex(f64) => .{ .z = c64_tblis{ .re = val.re, .im = val.im } },
        else => @compileError("init_scalar: T must be f32, f64 or Complex(...)"),
    };
    return .{ .data = data, .type = type_ };
}

fn init_tensor(
    comptime T: type,
    comptime rank: usize,
    shape: *const [rank]C.zig_len_type,
    stride: *const [rank]C.zig_stride_type,
    data: [*]T,
) TblisTensor {
    const type_ = init_type_t(T);
    const scalar = init_scalar(T, one(T));
    return .{
        .type = type_,
        .conj = 0.0,
        .scalar = scalar,
        .data = data,
        .ndim = rank,
        .len = @ptrCast(shape),
        .stride = @ptrCast(stride),
    };
}

test "init_tensor" {
    var shape = [_]C.zig_len_type{ 2, 3, 4 };
    var stride = [_]C.zig_stride_type{ 12, 4, 1 };
    var data: [24]f32 = undefined;
    @memset(&data, 1.0);
    var tf: TblisTensor = undefined;
    const scalar = TblisScalar{
        .data = .{ .s = 1.0 },
        .type = C.ZIG_TYPE_SINGLE,
    };
    tf = .{
        .type = C.ZIG_TYPE_SINGLE,
        .conj = 0,
        .scalar = scalar,
        .data = data[0..].ptr,
        .ndim = 3,
        .len = shape[0..].ptr,
        .stride = stride[0..].ptr,
    };
    const tf_ziginit = init_tensor(f32, 3, &shape, &stride, &data);

    try std.testing.expectEqual(tf.type, tf_ziginit.type);
    try std.testing.expectEqual(tf.conj, tf_ziginit.conj);
    try std.testing.expectEqual(tf.scalar.type, tf_ziginit.scalar.type);
    try std.testing.expectEqual(tf.scalar.data.s, tf_ziginit.scalar.data.s);
    try std.testing.expectEqual(tf.data, tf_ziginit.data);
    try std.testing.expectEqual(tf.ndim, tf_ziginit.ndim);
    try std.testing.expectEqual(tf.len, tf_ziginit.len);
    try std.testing.expectEqual(tf.stride, tf_ziginit.stride);
}

fn get_scalar_val(comptime T: type, scalar: TblisScalar) T {
    const T_type_t = init_type_t(T);
    assert(T_type_t == scalar.type);

    return switch (T) {
        f32 => scalar.data.s,
        f64 => scalar.data.d,
        Complex(f32) => .init(scalar.data.c.re, scalar.data.c.im),
        Complex(f64) => .init(scalar.data.z.re, scalar.data.z.im),
        else => @compileError("get_scalar_val: T must be f32, f64 or Complex(...)"),
    };
}

fn one(comptime T: type) T {
    return switch (T) {
        f32, f64 => 1.0,
        Complex(f32), Complex(f64) => .{ .re = 1.0, .im = 0.0 },
        else => @compileError("one: T must be f32, f64 or Complex(...)"),
    };
}

fn index_strings(comptime Axes: []const type) [Axes.len][]const C.zig_label_type {
    const result: [Axes.len][]const C.zig_label_type = comptime result: {
        const combined_names = axis_meta.unionOfAxisNames(Axes);

        var out: [Axes.len][]const C.zig_label_type = undefined;
        for (Axes, 0..) |AxisT, axis_i| {
            const axis_names = meta.fieldNames(AxisT);
            const rank = axis_names.len;
            assert(rank <= 255 - 'a' + 1);

            var idx_str: [rank]C.zig_label_type = undefined;
            for (axis_names, 0..) |name, i| {
                const combined_i: usize = blk: {
                    for (combined_names, 0..) |uname, j| {
                        if (std.mem.eql(u8, name, uname)) {
                            break :blk j;
                        }
                    }
                    @compileError("Axis name not found in combined names");
                };
                const char_idx: u8 = 'a' + @as(u8, @intCast(combined_i));
                idx_str[i] = char_idx;
            }
            const res = idx_str;
            out[axis_i] = &res;
        }

        break :result out;
    };
    return result;
}

test "index_strings" {
    const IJK = enum { i, j, k };
    const LKMI = enum { l, k, m, i };

    const expected = [_][]const C.zig_label_type{ "abc", "dcea" };
    const actual = index_strings(&.{ IJK, LKMI });

    try std.testing.expectEqualDeep(&expected, &actual);
}

test "C API" {
    _ = struct {
        test "add.h equal shape and stride" {
            const T = Complex(f32);
            const n = 12;
            const rank = 3;
            const a_shape = [rank]C.zig_len_type{ 2, 3, 2 };
            const b_shape = [rank]C.zig_len_type{ 2, 3, 2 };
            const a_stride = [rank]C.zig_stride_type{ 6, 2, 1 };
            const b_stride = [rank]C.zig_stride_type{ 6, 2, 1 };
            const a_idx = "ijk";
            const b_idx = "ijk";
            var a_data: [n]T = undefined;
            @memset(&a_data, .{ .re = 1.0, .im = 2.0 });
            var b_data: [n]T = undefined;
            @memset(&b_data, .{ .re = 2.0, .im = 0.0 });
            const a = init_tensor(T, rank, &a_shape, &a_stride, a_data[0..].ptr);
            var b = init_tensor(T, rank, &b_shape, &b_stride, b_data[0..].ptr);
            C.tblis_zig_tensor_add(
                C.tblis_single,
                null,
                &a,
                a_idx,
                &b,
                b_idx,
            );
            for (b_data) |bi| {
                try std.testing.expectEqualDeep(bi, T{ .re = 3.0, .im = 2.0 });
            }
        }

        test "add.h 2d + 1d" {
            // a = 1d, b = 2d, col major. Adding a to each row of b
            const m = 2;
            const n = 3;

            // b is 2D (m x n), column-major: stride = [1, m]
            const b_shape = [2]C.zig_len_type{ m, n };
            const b_stride = [2]C.zig_stride_type{ 1, m };

            // a is 1D (length n), contiguous
            const a_shape = [1]C.zig_len_type{n};
            const a_stride = [1]C.zig_stride_type{1};

            var b_data: [m * n]f32 = undefined;
            @memset(&b_data, 2.0);

            var a_data: [n]f32 = .{ 1.0, 2.0, 3.0 };

            const a = init_tensor(f32, 1, &a_shape, &a_stride, a_data[0..].ptr);
            var b = init_tensor(f32, 2, &b_shape, &b_stride, b_data[0..].ptr);

            const a_idx = "j";
            const b_idx = "ij";

            C.tblis_zig_tensor_add(
                null,
                null,
                &a,
                a_idx,
                &b,
                b_idx,
            );

            // Verify: b[i, j] = 2.0 + a[j], with column-major indexing: idx = i + j*m
            for (0..n) |j| {
                for (0..m) |i| {
                    const idx = i + j * m;
                    try std.testing.expectEqual(2.0 + a_data[j], b_data[idx]);
                }
            }
        }

        // ij jk seems to work when i = k = 1, but not in the general case.
        test "dot.h row col" {
            const T = f32;
            const m = 1;
            const n = 4;
            const k = 1;
            const a_data = [m * n]T{ 1, 2, 3, 4 };
            const b_data = [n * k]T{ 1, 10, 100, 1000 };
            const a_shape = [2]C.zig_len_type{ m, n };
            const b_shape = [2]C.zig_len_type{ n, k };
            // row-major
            const a_stride = [2]C.zig_stride_type{ n, 1 };
            // col-major
            const b_stride = [2]C.zig_stride_type{ 1, n };

            const expected: T = 4321;

            const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
            const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
            var result_struct = init_scalar(T, undefined);
            C.tblis_zig_tensor_dot(
                null,
                null,
                &a,
                "ij",
                &b,
                "jk",
                &result_struct,
            );
            const actual = result_struct.data.s;

            try std.testing.expectEqual(expected, actual);
        }

        test "dot.h ij ij" {
            const T = f64;
            const m = 2;
            const n = 2;
            const k = 2;
            const a_data = [m * n]T{ 1, 2, 3, 4 };
            const b_data = [n * k]T{ 1, 10, 100, 1000 };
            const a_shape = [2]C.zig_len_type{ m, n };
            const b_shape = [2]C.zig_len_type{ n, k };
            // row-major
            const a_stride = [2]C.zig_stride_type{ n, 1 };
            // row-major
            const b_stride = [2]C.zig_stride_type{ k, 1 };

            const expected: T = 4321;

            const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
            const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
            var result_struct = init_scalar(T, undefined);
            C.tblis_zig_tensor_dot(
                null,
                null,
                &a,
                "ij",
                &b,
                "ij",
                &result_struct,
            );
            const actual = get_scalar_val(T, result_struct);

            try std.testing.expectEqual(expected, actual);
        }

        // I guess dot only works for arrays with equal axes?
        // test "dot.h ij jk" {
        //     const T = f64;
        //     const m = 2;
        //     const n = 2;
        //     const k = 3;
        //     const a_data = [m * n]T{ 1, 2, 3, 4 };
        //     const b_data = [n * k]T{ 1, 10, 100, 1000, 10_000, 100_000 };
        //     const a_shape = [2]C.zig_len_type{ m, n };
        //     const b_shape = [2]C.zig_len_type{ n, k };
        //     // row-major
        //     const a_stride = [2]C.zig_stride_type{ n, 1 };
        //     // col-major
        //     const b_stride = [2]C.zig_stride_type{ 1, n };

        //     // c_11 =      21
        //     // c_12 =   2_100
        //     // c_13 = 210_000
        //     // c_21 =      43
        //     // c_22 =   4_300
        //     // c_23 = 430_000
        //     // sum  = 646_464
        //     const expected: T = 646_464;

        //     const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
        //     const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
        //     var result_struct = init_scalar(T, undefined);
        //     C.tblis_zig_tensor_dot(
        //         null,
        //         null,
        //         &a,
        //         "ij",
        //         &b,
        //         "jk",
        //         &result_struct,
        //     );
        //     const actual = get_scalar_val(T, result_struct);

        //     try std.testing.expectEqual(expected, actual);
        // }

        test "mult.h ij jk" {
            const T = f64;
            const m = 2;
            const n = 2;
            const k = 3;
            const a_data = [m * n]T{ 1, 2, 3, 4 };
            const b_data = [n * k]T{ 1, 10, 100, 1000, 10_000, 100_000 };
            var c_data: [m * k]T = undefined;
            @memset(&c_data, 0.0);
            const a_shape = [2]C.zig_len_type{ m, n };
            const b_shape = [2]C.zig_len_type{ n, k };
            const c_shape = [2]C.zig_len_type{ m, k };
            // row-major
            const a_stride = [2]C.zig_stride_type{ n, 1 };
            // col-major
            const b_stride = [2]C.zig_stride_type{ 1, n };
            // row-major
            const c_stride = [2]C.zig_stride_type{ k, 1 };

            const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
            const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
            var c_ = init_tensor(T, 2, &c_shape, &c_stride, &c_data);

            // c_11 =      21
            // c_12 =   2_100
            // c_13 = 210_000
            // c_21 =      43
            // c_22 =   4_300
            // c_23 = 430_000
            const expected = [m * k]T{ 21, 2100, 210_000, 43, 4300, 430_000 };

            C.tblis_zig_tensor_mult(
                null,
                null,
                &a,
                "ij",
                &b,
                "jk",
                &c_,
                "ik",
            );

            try std.testing.expectEqualSlices(T, &expected, &c_data);
        }

        test "mult.h i jk -> ijk" {
            const T = f64;
            const m = 2; // i-dim
            const n = 2; // j-dim
            const k = 3; // k-dim

            // a: 1D vector over "i"
            const a_shape = [1]C.zig_len_type{m};
            const a_stride = [1]C.zig_stride_type{1};
            var a_data: [m]T = .{ 1.0, 2.0 };

            // b: 2D matrix over "jk" (row-major)
            const b_shape = [2]C.zig_len_type{ n, k };
            const b_stride = [2]C.zig_stride_type{ k, 1 };
            var b_data: [n * k]T = .{
                // j = 0 row
                10.0,  20.0,  30.0,
                // j = 1 row
                100.0, 200.0, 300.0,
            };

            // c: 3D tensor over "ijk" (row-major)
            const c_shape = [3]C.zig_len_type{ m, n, k };
            const c_stride = [3]C.zig_stride_type{ n * k, k, 1 };
            var c_data: [m * n * k]T = undefined;
            @memset(&c_data, 0.0);

            const a = init_tensor(T, 1, &a_shape, &a_stride, a_data[0..].ptr);
            const b = init_tensor(T, 2, &b_shape, &b_stride, b_data[0..].ptr);
            var c_ = init_tensor(T, 3, &c_shape, &c_stride, &c_data);

            // c[i, j, k] = a[i] * b[j, k]
            C.tblis_zig_tensor_mult(
                null,
                null,
                &a,
                "i",
                &b,
                "jk",
                &c_,
                "ijk",
            );

            const expected = [m * n * k]T{
                // i = 0 => a[0] = 1
                10.0, 20.0, 30.0, 100.0, 200.0, 300.0,
                // i = 1 => a[1] = 2
                20.0, 40.0, 60.0, 200.0, 400.0, 600.0,
            };

            try std.testing.expectEqualSlices(T, &expected, &c_data);
        }

        test "reduce.h i" {
            const T = f64;
            const rank = 1;
            const a_shape = [rank]C.zig_len_type{5};
            const a_stride = [rank]C.zig_stride_type{1};
            const a_data = &[_]T{ 1, -2, 3, -4, 5 };
            var a = init_tensor(T, 1, &a_shape, &a_stride, @constCast(a_data[0..].ptr));
            a.scalar = init_scalar(T, 2);

            var a_sum = init_scalar(T, undefined);
            var a_sum_idx: [rank]C.zig_len_type = undefined;
            C.tblis_zig_tensor_reduce(
                null,
                null,
                C.ZIG_REDUCE_SUM,
                &a,
                "i",
                &a_sum,
                &a_sum_idx,
            );
            try std.testing.expectEqual(init_type_t(T), a_sum.type);
            try std.testing.expectEqual(3.0 * 2, a_sum.data.d);

            var a_sum_abs = init_scalar(T, undefined);
            var a_sum_abs_idx: [rank]C.zig_len_type = undefined;
            C.tblis_zig_tensor_reduce(
                null,
                null,
                C.ZIG_REDUCE_SUM_ABS,
                &a,
                "i",
                &a_sum_abs,
                &a_sum_abs_idx,
            );
            try std.testing.expectEqual(init_type_t(T), a_sum_abs.type);
            try std.testing.expectEqual(15.0 * 2, a_sum_abs.data.d);

            var a_max = init_scalar(T, undefined);
            var a_max_idx: [rank]C.zig_len_type = undefined;
            C.tblis_zig_tensor_reduce(
                null,
                null,
                C.ZIG_REDUCE_MAX,
                &a,
                "i",
                &a_max,
                &a_max_idx,
            );
            try std.testing.expectEqual(init_type_t(T), a_max.type);
            try std.testing.expectEqual(5.0 * 2, a_max.data.d);
            try std.testing.expectEqualDeep([_]C.zig_len_type{4}, a_max_idx);
        }

        test "scale.h" {
            const T = Complex(f64);
            const rank = 4;
            const a_shape = [rank]C.zig_len_type{ 1, 1, 3, 1 };
            const a_stride = [rank]C.zig_stride_type{ 6290, 19348, 1, 6890000 };
            var a_data = [_]T{
                .{ .re = 1.0, .im = 0.0 },
                .{ .re = 1.0, .im = -1.0 },
                .{ .re = 0.0, .im = 1.0 },
            };
            var a = init_tensor(T, rank, &a_shape, &a_stride, a_data[0..].ptr);
            a.scalar = init_scalar(T, .init(1.0, 1.0));

            C.tblis_zig_tensor_scale(
                null,
                null,
                &a,
                "ijkl",
            );

            const expected = [_]T{
                .init(1.0, 1.0),
                .init(2.0, 0.0),
                .init(-1.0, 1.0),
            };
            try std.testing.expectEqualDeep(expected, a_data);
            try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
        }

        test "set.h f32 contiguous 1d" {
            const T = f32;
            const rank = 1;
            const len = [rank]C.zig_len_type{5};
            const stride = [rank]C.zig_stride_type{1};
            var data: [5]T = undefined;
            @memset(&data, 0.0);

            var a = init_tensor(T, rank, &len, &stride, data[0..].ptr);
            const alpha = init_scalar(T, 3.5);

            C.tblis_zig_tensor_set(
                null,
                null,
                &alpha,
                &a,
                "i",
            );

            const expected = [_]T{3.5} ** 5;
            try std.testing.expectEqualDeep(expected, data);
            try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
        }

        test "set.h complex strided 3d with gaps" {
            const T = Complex(f64);
            const m = 2;
            const n = 2;
            const p = 2;
            const rank = 3;

            // Layout: simulate padding by placing planes 8 apart, rows 4 apart, cols 1 apart.
            // Only 8 logical elements, but allocate a larger buffer and verify non-accessed padding unchanged.
            const len = [rank]C.zig_len_type{ m, n, p };
            const stride = [rank]C.zig_stride_type{ 8, 4, 1 };
            const logical = [_]usize{ 0, 1, 4, 5, 8, 9, 12, 13 };

            var buf: [16]T = undefined;
            @memset(&buf, .init(0, 0));

            var a = init_tensor(T, rank, &len, &stride, buf[0..].ptr);
            // Check that a.scalar is ignored
            a.scalar = init_scalar(T, .init(2, 0));

            const alpha = init_scalar(T, one(T));
            C.tblis_zig_tensor_set(
                null,
                null,
                &alpha,
                &a,
                "ijk",
            );

            // Verify logical positions were modified, and others were not.
            for (0..16) |i| {
                const is_logical = std.mem.containsAtLeastScalar(usize, &logical, 1, i);
                const expected = if (is_logical) one(T) else T.init(0, 0);
                try std.testing.expectEqual(expected, buf[i]);
            }
            try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
        }

        test "shift.h f32 contiguous 1d" {
            const T = f32;
            const rank = 1;
            const len = [rank]C.zig_len_type{5};
            const stride = [rank]C.zig_stride_type{1};

            var data: [5]T = .{ 1.0, 2.0, 3.0, 4.0, 5.0 };

            var a = init_tensor(T, rank, &len, &stride, data[0..].ptr);
            a.scalar = init_scalar(T, 2.0);

            const alpha = init_scalar(T, 3.5);

            C.tblis_zig_tensor_shift(
                null,
                null,
                &alpha,
                &a,
                "i",
            );

            const expected = [_]T{ 5.5, 7.5, 9.5, 11.5, 13.5 };
            try std.testing.expectEqualDeep(expected, data);
            try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
        }

        test "shift.h complex strided 3d with gaps" {
            const T = Complex(f64);
            const m = 2;
            const n = 2;
            const p = 2;
            const rank = 3;

            // Layout with padding: planes 8 apart, rows 4 apart, cols 1 apart.
            const len = [rank]C.zig_len_type{ m, n, p };
            const stride = [rank]C.zig_stride_type{ 8, 4, 1 };

            // Logical positions (same mapping as in the set test)
            const logical = [_]usize{ 0, 1, 4, 5, 8, 9, 12, 13 };

            var buf: [16]T = .{
                // plane 0 (indices 0..7)
                T.init(0, 0), T.init(1, 0), T.init(0, 0), T.init(0, 0),
                T.init(2, 0), T.init(3, 0), T.init(0, 0), T.init(0, 0),
                // plane 1 (indices 8..15)
                T.init(4, 0), T.init(5, 0), T.init(0, 0), T.init(0, 0),
                T.init(6, 0), T.init(7, 0), T.init(0, 0), T.init(0, 0),
            };

            var a = init_tensor(T, rank, &len, &stride, buf[0..].ptr);

            const alpha = init_scalar(T, one(T));

            C.tblis_zig_tensor_shift(
                null,
                null,
                &alpha,
                &a,
                "ijk",
            );

            // Verify logical positions were incremented by alpha and padding unchanged.
            for (0..16) |i| {
                const is_logical = std.mem.containsAtLeastScalar(usize, &logical, 1, i);
                const before = switch (i) {
                    0 => T.init(0, 0),
                    1 => T.init(1, 0),
                    4 => T.init(2, 0),
                    5 => T.init(3, 0),
                    8 => T.init(4, 0),
                    9 => T.init(5, 0),
                    12 => T.init(6, 0),
                    13 => T.init(7, 0),
                    else => T.init(0, 0),
                };
                const expected = if (is_logical) T.init(before.re + 1.0, before.im) else before;
                try std.testing.expectEqual(expected, buf[i]);
            }

            // Scalar should be reset to 1 after operation
            try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
        }

        test "thread.h" {
            _ = C.tblis_comm;
            _ = C.tblis_single;
            // const TblisComm = c.tblis_comm;
            // comptime for (@typeInfo(TblisComm).@"struct".fields) |field| {
            //     @compileLog(field.name);
            //     @compileLog(@typeInfo(field.type));
            // };
            // std.debug.print("{any}\n", .{c.tblis_single.*});
            // const num_threads_before = c.tblis_get_num_threads();
            // std.debug.print("Default: {} threads\n", .{num_threads_before});
            C.tblis_set_num_threads(2);
            const num_threads_after = C.tblis_get_num_threads();
            try std.testing.expectEqual(2, num_threads_after);
        }
    };
}
