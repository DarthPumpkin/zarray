const std = @import("std");
const mem = std.mem;
const meta = std.meta;

const named_array = @import("named_array.zig");
const named_index = @import("named_index.zig");
const NamedArray = named_array.NamedArray;
const NamedArrayConst = named_array.NamedArrayConst;
const NamedIndex = named_index.NamedIndex;

fn Promote(comptime T: type, comptime U: type) type {
    return @TypeOf(@as(T, 0) + @as(U, 0));
}

fn hasField(names: []const []const u8, name: []const u8) bool {
    for (names) |n| {
        if (mem.eql(u8, n, name)) {
            // @compileLog("Match: " ++ n ++ " " ++ name);
            return true;
        }
    }
    return false;
}

/// If `idx_out` has overlapping linear indices, the output is undefined.
pub fn add(
    comptime Axis: type,
    comptime Scalar: type,
    arr1: NamedArrayConst(Axis, Scalar),
    arr2: NamedArrayConst(Axis, Scalar),
    arr_out: NamedArray(Axis, Scalar),
) void {
    if (arr1.idx.shape != arr2.idx.shape or arr1.idx.shape != arr_out.idx.shape)
        @panic("Incompatible shapes");
    // TODO: Check that arr_out.idx is non-overlapping.
    var keys = arr1.idx.iterKeys();
    while (keys.next()) |key| {
        const l = arr1.scalarAt(key);
        const r = arr2.scalarAt(key);
        arr_out.at(key).* = l + r;
    }
}

pub fn inner(
    comptime Axis: type,
    comptime Scalar1: type,
    comptime Scalar2: type,
    arr1: NamedArrayConst(Axis, Scalar1),
    arr2: NamedArrayConst(Axis, Scalar2),
) Promote(Scalar1, Scalar2) {
    if (arr1.idx.shape != arr2.idx.shape)
        @panic("Incompatible shapes for inner product");

    const ResultType = Promote(Scalar1, Scalar2);
    var sum: ResultType = 0;
    var keys = arr1.idx.iterKeys();
    while (keys.next()) |key| {
        sum += arr1.scalarAt(key) * arr2.scalarAt(key);
    }
    return sum;
}

/// Einstein sum of two arrays.
/// Contracts axes that are present in both arrys, while preserving those that are present in only
/// one array.
/// Two axes are considered equal if their enum fields have the same name.
///
/// For example, given arrays `A` with axes (i, j) and `B` with axes (j, k), the contracted axis is `j`,
/// and the output axes are (i, k). The result is equivalent to matrix multiplication.
///
/// The output axis type must be an enum whose field names are the names of the preserved axes,
/// i.e., symmetric difference (XOR) of the input axis names.
///
/// **Note**: Currently, there must be at least one preserved axis. There is a separate `inner` function
/// for when all axes should be contracted.
///
/// The output array will have shape determined by the preserved axes, and each element will be
/// the sum over all possible values of the contracted axes of the product of the corresponding elements
/// from the input arrays.
/// The contracted axes must have the same size in both arrays.
///
/// Example:
/// ```zig
///     const IJ = enum { i, j };
///     const JK = enum { j, k };
///     const IK = enum { i, k };
///     einsum(IJ, JK, IK, ..., arrA, arrB, allocator)
///```
/// computes matrix multiplication.
/// This function generalizes matrix multiplication, outer product and higher-order tensor products.
///
pub fn einsum(
    comptime AxisA: type,
    comptime AxisB: type,
    comptime AxisOut: type,
    comptime ScalarA: type,
    comptime ScalarB: type,
    arrA: NamedArrayConst(AxisA, ScalarA),
    arrB: NamedArrayConst(AxisB, ScalarB),
    allocator: mem.Allocator,
) !NamedArray(AxisOut, Promote(ScalarA, ScalarB)) {
    const namesA = comptime meta.fieldNames(AxisA);
    const namesB = comptime meta.fieldNames(AxisB);
    const output_names = comptime meta.fieldNames(AxisOut);

    // At least one axis must be preserved
    if (comptime output_names.len == 0) {
        @compileError("einsum with zero output axes (rank-0) is not yet supported. Use inner instead.");
    }

    // Validate axis names
    const xor = named_index.Xor(AxisA, AxisB);
    if (comptime !mem.eql([:0]const u8, output_names, meta.fieldNames(xor))) {
        @compileError("AxisOut is not the XOR of AxisA and AxisB.");
    }

    // Compute contracted axes (intersection)
    const contracted_names = comptime blk: {
        var tmp: [namesA.len]([:0]const u8) = undefined;
        var n = 0;
        for (0..namesA.len) |ai| {
            const nameA = meta.fields(AxisA)[ai].name;
            for (0..namesB.len) |bi| {
                const nameB = meta.fields(AxisB)[bi].name;
                if (mem.eql(u8, nameA, nameB)) {
                    tmp[n] = nameA;
                    n += 1;
                }
            }
        }
        break :blk tmp[0..n];
    };

    // Output scalar type
    const OutputScalar = Promote(ScalarA, ScalarB);
    const OutputIndexType = NamedIndex(AxisOut);

    // Build output shape
    const output_shape: OutputIndexType.Axes = blk: {
        var shape: OutputIndexType.Axes = undefined;
        inline for (output_names) |name| {
            if (comptime hasField(namesA, name)) {
                @field(shape, name) = @field(arrA.idx.shape, name);
            } else if (comptime hasField(namesB, name)) {
                @field(shape, name) = @field(arrB.idx.shape, name);
            } else {
                @compileError(name ++ " " ++ arrA.idx.shape ++ " " ++ arrB.idx.shape);
            }
        }
        break :blk shape;
    };

    // Build output index and buffer
    const output_idx = NamedIndex(AxisOut).initContiguous(output_shape);
    var output_buf = try allocator.alloc(OutputScalar, output_idx.count());

    // For each output key
    var out_keys = output_idx.iterKeys();
    var out_i: usize = 0;
    while (out_keys.next()) |out_key| {
        var sum: OutputScalar = 0;

        if (comptime contracted_names.len == 0) {
            // No contraction: just multiply the corresponding elements
            var keyA: NamedIndex(AxisA).Axes = undefined;
            var keyB: NamedIndex(AxisB).Axes = undefined;
            inline for (namesA) |name| {
                @field(keyA, name) = @field(out_key, name);
            }
            inline for (namesB) |name| {
                @field(keyB, name) = @field(out_key, name);
            }
            sum = arrA.scalarAt(keyA) * arrB.scalarAt(keyB);
        } else {
            // For each contracted key
            const ContractedKey = named_index.AxesStruct(contracted_names);
            var contracted_shape: ContractedKey = undefined;
            inline for (contracted_names) |name| {
                @field(contracted_shape, name) = @field(arrA.idx.shape, name);
            }
            var contracted_idx = NamedIndex(named_index.KeyEnum(contracted_names)).initContiguous(contracted_shape);
            var ckeys = contracted_idx.iterKeys();
            while (ckeys.next()) |ckey| {
                // Build full keys for arrA and arrB
                var keyA: NamedIndex(AxisA).Axes = undefined;
                var keyB: NamedIndex(AxisB).Axes = undefined;
                inline for (namesA) |name| {
                    if (comptime hasField(output_names, name)) {
                        @field(keyA, name) = @field(out_key, name);
                    } else {
                        @field(keyA, name) = @field(ckey, name);
                    }
                }
                inline for (namesB) |name| {
                    if (comptime hasField(output_names, name)) {
                        @field(keyB, name) = @field(out_key, name);
                    } else {
                        @field(keyB, name) = @field(ckey, name);
                    }
                }
                sum += arrA.scalarAt(keyA) * arrB.scalarAt(keyB);
            }
        }
        output_buf[out_i] = sum;
        out_i += 1;
    }

    return NamedArray(AxisOut, OutputScalar){
        .idx = output_idx,
        .buf = output_buf,
    };
}

// test "log" {
//     @compileLog("hi");
// }

test "add inplace" {
    const Axis = enum { i };
    const idx = NamedIndex(Axis).initContiguous(.{ .i = 3 });
    const buf1 = [_]i32{ 1, 2, 3 };
    const arr1 = NamedArrayConst(Axis, i32){
        .idx = idx,
        .buf = &buf1,
    };
    var buf2 = [_]i32{ 2, 2, 2 };
    const arr_out = NamedArray(Axis, i32){
        .idx = idx,
        .buf = &buf2,
    };
    const arr2 = arr_out.asConst();
    add(Axis, i32, arr1, arr2, arr_out);

    const expected = [_]i32{ 3, 4, 5 };
    try std.testing.expectEqualSlices(i32, &expected, &buf2);
}

test "add broadcasted" {
    const I = enum { i };
    const IJ = enum { i, j };
    const idx_broad = NamedIndex(I)
        .initContiguous(.{ .i = 3 })
        .conformAxes(IJ)
        .broadcastAxis(.j, 4);
    const idx_out = NamedIndex(IJ)
        .initContiguous(.{ .i = 3, .j = 4 });
    var buf1 = [_]i32{ 1, 2, 3 };
    var buf2 = [_]i32{ 1, 1, 1 };
    var buf_out: [12]i32 = undefined;
    const arr1 = NamedArrayConst(IJ, i32){
        .idx = idx_broad,
        .buf = &buf1,
    };
    const arr2 = NamedArrayConst(IJ, i32){
        .idx = idx_broad,
        .buf = &buf2,
    };
    const arr_out = NamedArray(IJ, i32){
        .idx = idx_out,
        .buf = &buf_out,
    };
    add(IJ, i32, arr1, arr2, arr_out);

    const expected = [_]i32{
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
    };
    try std.testing.expectEqualSlices(i32, &expected, &buf_out);
}

test "add row-major col-major" {
    const IJ = enum { i, j };
    const idx_row_major = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });
    const idx_col_major = NamedIndex(IJ){
        .shape = .{ .i = 2, .j = 3 },
        .strides = .{ .i = 1, .j = 2 },
    };

    var buf_row_major = [_]i32{
        1, 2, 3,
        4, 5, 6,
    };
    var buf_col_major = [_]i32{
        10, 40,
        20, 50,
        30, 60,
    };
    var buf_out: [6]i32 = undefined;

    const arr_row_major = NamedArrayConst(IJ, i32){ .idx = idx_row_major, .buf = &buf_row_major };
    const arr_col_major = NamedArrayConst(IJ, i32){ .idx = idx_col_major, .buf = &buf_col_major };
    const arr_out = NamedArray(IJ, i32){ .idx = idx_row_major, .buf = &buf_out };

    add(IJ, i32, arr_row_major, arr_col_major, arr_out);

    const expected = [_]i32{ 11, 22, 33, 44, 55, 66 };
    try std.testing.expectEqualSlices(i32, &expected, &buf_out);
}

test "inner 1d mixed types" {
    const Axis = enum { i };
    const idx = NamedIndex(Axis).initContiguous(.{ .i = 3 });
    const arr1 = NamedArrayConst(Axis, f32){ .idx = idx, .buf = &[_]f32{ 1, 2, 3 } };
    const arr2 = NamedArrayConst(Axis, f64){ .idx = idx, .buf = &[_]f64{ 4.0, 5.0, 6.0 } };
    const result = inner(Axis, f32, f64, arr1, arr2);
    try std.testing.expectEqual(result, 32.0); // 1*4.0 + 2*5.0 + 3*6.0 = 32.0
}

test "inner 2d row-major col-major" {
    const IJ = enum { i, j };
    const idx_row_major = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });
    const idx_col_major = NamedIndex(IJ){
        .shape = .{ .i = 2, .j = 3 },
        .strides = .{ .i = 1, .j = 2 },
    };

    const buf_row_major = [_]i32{
        1, 2, 3,
        4, 5, 6,
    };
    const buf_col_major = [_]i32{
        10, 40,
        20, 50,
        30, 60,
    };

    const arr_row_major = NamedArrayConst(IJ, i32){ .idx = idx_row_major, .buf = &buf_row_major };
    const arr_col_major = NamedArrayConst(IJ, i32){ .idx = idx_col_major, .buf = &buf_col_major };

    const result = inner(IJ, i32, i32, arr_row_major, arr_col_major);

    // Calculation:
    // arr_row_major: [ [1,2,3], [4,5,6] ]
    // arr_col_major: [ [10,20,30], [40,50,60] ] (column-major)
    // Inner product: sum over i,j of arr_row_major[i,j] * arr_col_major[i,j]
    // = 1*10 + 2*20 + 3*30 + 4*40 + 5*50 + 6*60
    // = 10 + 40 + 90 + 160 + 250 + 360 = 910

    try std.testing.expectEqual(result, 910);
}

test "einsum matrix multiplication" {
    const IJ = enum { i, j };
    const JK = enum { j, k };
    const IK = enum { i, k };

    const idx_ij = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });
    const idx_jk = NamedIndex(JK).initContiguous(.{ .j = 3, .k = 2 });

    // 2x3 matrix
    const arr_ij = NamedArrayConst(IJ, i32){
        .idx = idx_ij,
        .buf = &[_]i32{ 1, 2, 3, 4, 5, 6 }, // row-major: [ [1,2,3], [4,5,6] ]
    };
    // 3x2 matrix
    const arr_jk = NamedArrayConst(JK, i32){
        .idx = idx_jk,
        .buf = &[_]i32{ 7, 8, 9, 10, 11, 12 }, // row-major: [ [7,8], [9,10], [11,12] ]
    };

    const allocator = std.testing.allocator;
    const arr_ik = try einsum(IJ, JK, IK, i32, i32, arr_ij, arr_jk, allocator);
    defer arr_ik.deinit(allocator);

    // Expected result: [ [58, 64], [139, 154] ]
    const expected = [_]i32{ 58, 64, 139, 154 };
    try std.testing.expectEqualSlices(i32, &expected, arr_ik.buf);
}

test "einsum sum over axis" {
    const IJ = enum { i, j };
    const J = enum { j };
    const I = enum { i };

    const idx_ij = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });

    // 2x3 matrix: [ [1,2,3], [4,5,6] ]
    const arr_ij = NamedArrayConst(IJ, i32){
        .idx = idx_ij,
        .buf = &[_]i32{ 1, 2, 3, 4, 5, 6 },
    };

    // Summing over j: result should be [1+2+3, 4+5+6] = [6, 15]
    const allocator = std.testing.allocator;
    const arr_i = try einsum(IJ, J, I, i32, i32, arr_ij, NamedArrayConst(J, i32){
        .idx = NamedIndex(J).initContiguous(.{ .j = 3 }),
        .buf = &[_]i32{ 1, 1, 1 }, // acts as a "sum" over j
    }, allocator);
    defer arr_i.deinit(allocator);

    const expected = [_]i32{ 6, 15 };
    try std.testing.expectEqualSlices(i32, &expected, arr_i.buf);
}

test "einsum outer product 1d x 1d -> 2d" {
    const I = enum { i };
    const J = enum { j };
    const IJ = enum { i, j };

    const idx_i = NamedIndex(I).initContiguous(.{ .i = 2 });
    const idx_j = NamedIndex(J).initContiguous(.{ .j = 3 });

    const arr_i = NamedArrayConst(I, i32){
        .idx = idx_i,
        .buf = &[_]i32{ 2, 3 },
    };
    const arr_j = NamedArrayConst(J, i32){
        .idx = idx_j,
        .buf = &[_]i32{ 10, 20, 30 },
    };

    const allocator = std.testing.allocator;
    const arr_ij = try einsum(I, J, IJ, i32, i32, arr_i, arr_j, allocator);
    defer arr_ij.deinit(allocator);

    // Expected: [[2*10, 2*20, 2*30], [3*10, 3*20, 3*30]] = [20,40,60, 30,60,90]
    const expected = [_]i32{ 20, 40, 60, 30, 60, 90 };
    try std.testing.expectEqualSlices(i32, &expected, arr_ij.buf);
}
