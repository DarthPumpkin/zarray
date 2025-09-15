const std = @import("std");
const mem = std.mem;
const meta = std.meta;

const named_index = @import("named_index.zig");
const NamedIndex = named_index.NamedIndex;

pub fn NamedArray(comptime Axis: type, comptime Scalar: type) type {
    const Index = NamedIndex(Axis);
    return struct {
        idx: Index,
        buf: []Scalar,

        pub fn initAlloc(allocator: mem.Allocator, shape: Index.Axes) !@This() {
            const idx = Index.initContiguous(shape);
            return .{
                .idx = idx,
                .buf = try allocator.alloc(Scalar, idx.count()),
            };
        }

        pub fn fill(self: *const @This(), val: Scalar) *const @This() {
            var keys = self.idx.iterKeys();
            while (keys.next()) |key| {
                self.buf[self.idx.linear(key)] = val;
            }
            return self;
        }

        pub fn fillArange(self: *const @This()) *const @This() {
            var keys = self.idx.iterKeys();
            var i: Scalar = 0;
            while (keys.next()) |key| {
                self.buf[self.idx.linear(key)] = i;
                i += 1;
            }
            return self;
        }

        pub fn deinit(self: *const @This(), allocator: mem.Allocator) void {
            allocator.free(self.buf);
        }

        pub fn asConst(self: *const @This()) NamedArrayConst(Axis, Scalar) {
            return .{
                .idx = self.idx,
                .buf = self.buf,
            };
        }

        /// If possible, return a 1D slice of the buffer containing the elements of this array.
        /// If the array is non-contiguous, return null.
        /// To get a contiguous copy, see `toContiguous`.
        pub fn flat(self: *const @This()) ?[]Scalar {
            return flatGeneric(self);
        }

        /// Make a contiguous copy of the array.
        /// The new array will have the same shape and default strides.
        /// This allocates `self.idx.count()` scalars.
        pub fn toContiguous(self: *const @This(), allocator: mem.Allocator) !@This() {
            return toContiguousGeneric(Axis, Scalar, self, allocator);
        }

        pub fn getValChecked(self: *const @This(), key: Index.Axes) ?Scalar {
            return getValCheckedGeneric(self, key);
        }

        pub fn getVal(self: *const @This(), key: Index.Axes) Scalar {
            return self.asConst().getVal(key);
        }

        pub fn getPtrChecked(self: *const @This(), key: Index.Axes) ?*Scalar {
            return getPtrCheckedGeneric(self, key);
        }

        pub fn getPtr(self: *const @This(), key: Index.Axes) *Scalar {
            return &self.buf[self.idx.linear(key)];
        }

        pub fn setVal(self: *const @This(), key: Index.Axes, scalar: Scalar) void {
            self.buf[self.idx.linear(key)] = scalar;
        }
    };
}

pub fn NamedArrayConst(comptime Axis: type, comptime Scalar: type) type {
    const Index = NamedIndex(Axis);
    return struct {
        idx: Index,
        buf: []const Scalar,

        /// If possible, return a 1D slice of the buffer containing the elements of this array.
        /// If the array is non-contiguous, return null.
        /// To get a contiguous copy, see `toContiguous`.
        pub fn flat(self: *const @This()) ?[]const Scalar {
            return flatGeneric(self);
        }

        /// Make a contiguous copy of the array.
        /// The new array will have the same shape and default strides.
        /// This allocates `self.idx.count()` scalars.
        pub fn toContiguous(self: *const @This(), allocator: mem.Allocator) !NamedArray(Axis, Scalar) {
            return toContiguousGeneric(Axis, Scalar, self, allocator);
        }

        pub fn getValChecked(self: *const @This(), key: Index.Axes) ?Scalar {
            return getValCheckedGeneric(self, key);
        }

        pub fn getVal(self: *const @This(), key: Index.Axes) Scalar {
            return self.buf[self.idx.linear(key)];
        }

        pub fn getPtrChecked(self: *const @This(), key: Index.Axes) ?*const Scalar {
            return getPtrCheckedGeneric(self, key);
        }

        pub fn getPtr(self: *const @This(), key: Index.Axes) *const Scalar {
            return &self.buf[self.idx.linear(key)];
        }
    };
}

// Works for both NamedArray and NamedArrayConst
fn flatGeneric(self: anytype) ?@TypeOf(self.buf) {
    if (self.idx.isContiguous())
        return self.buf[self.idx.offset..][0..self.idx.count()];
    return null;
}

// Works for both NamedArray and NamedArrayConst
fn toContiguousGeneric(comptime Axis: type, comptime Scalar: type, self: anytype, allocator: mem.Allocator) !NamedArray(Axis, Scalar) {
    const Index = @TypeOf(self.idx);
    var buf = try allocator.alloc(Scalar, self.idx.count());
    errdefer comptime unreachable;
    const new_idx = Index.initContiguous(self.idx.shape);
    {
        var i: usize = 0;
        var keys = new_idx.iterKeys();
        while (keys.next()) |key| {
            buf[i] = self.getVal(key);
            i += 1;
        }
    }
    return .{ .idx = new_idx, .buf = buf };
}

fn getValCheckedGeneric(self: anytype, key: @TypeOf(self.idx).Axes) ?@TypeOf(self.buf[0]) {
    if (self.idx.linearChecked(key)) |key_| {
        return self.buf[key_];
    }
    return null;
}

fn getPtrCheckedGeneric(self: anytype, key: @TypeOf(self.idx).Axes) ?@TypeOf(&self.buf[0]) {
    if (self.idx.linearChecked(key)) |key_| {
        return &self.buf[key_];
    }
    return null;
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
        const l = arr1.getVal(key);
        const r = arr2.getVal(key);
        arr_out.setVal(key, l + r);
    }
}

pub fn dot(
    comptime Axis: type,
    comptime Scalar1: type,
    comptime Scalar2: type,
    arr1: NamedArrayConst(Axis, Scalar1),
    arr2: NamedArrayConst(Axis, Scalar2),
) @TypeOf(arr1.buf[0] * arr2.buf[0]) {
    if (arr1.idx.shape != arr2.idx.shape)
        @panic("Incompatible shapes for dot product");

    const ResultType = @TypeOf(arr1.buf[0] * arr2.buf[0]);
    var sum: ResultType = 0;
    var keys = arr1.idx.iterKeys();
    while (keys.next()) |key| {
        sum += arr1.getVal(key) * arr2.getVal(key);
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
/// **Note**: Currently, there must be at least one preserved axis. There is a separate `dot` function
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
        @compileError("einsum with zero output axes (rank-0) is not yet supported. Use dot instead.");
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
            sum = arrA.getVal(keyA) * arrB.getVal(keyB);
        } else {
            // For each contracted key
            const ContractedKey = named_index.KeyStruct(contracted_names);
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
                sum += arrA.getVal(keyA) * arrB.getVal(keyB);
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

fn hasField(names: []const []const u8, name: []const u8) bool {
    for (names) |n| {
        if (mem.eql(u8, n, name)) {
            // @compileLog("Match: " ++ n ++ " " ++ name);
            return true;
        }
    }
    return false;
}

fn Promote(comptime T: type, comptime U: type) type {
    return @TypeOf(@as(T, 0) + @as(U, 0));
}

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
    const IJ = named_index.KeyEnum(&.{ "i", "j" });
    const idx_broad = NamedIndex(I)
        .initContiguous(.{ .i = 3 })
        .addEmptyAxis("j")
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

test "fill" {
    const Axis = enum { i };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 4 });
    _ = arr.fill(0);
    defer arr.deinit(allocator);

    const expected_zeros = [_]i32{ 0, 0, 0, 0 };
    try std.testing.expectEqualSlices(i32, &expected_zeros, arr.buf);

    _ = arr.fillArange();
    const expected_arange = [_]i32{ 0, 1, 2, 3 };
    try std.testing.expectEqualSlices(i32, &expected_arange, arr.buf);
}

test "flat, toContiguous" {
    const IJ = enum { i, j };

    const al = std.testing.allocator;
    var arr = try NamedArray(IJ, i32).initAlloc(al, .{ .i = 5, .j = 9 });
    defer arr.deinit(al);
    _ = arr.fillArange();

    // Non-contiguous array cannot be flattened
    arr.idx = arr.idx
        .sliceAxis(.i, 0, 4)
        .stride(.{ .j = 3 });
    try std.testing.expectEqual(arr.flat(), null);

    // After making it contiguous, .flat() works.
    const arr_cont = try arr.toContiguous(al);
    defer arr_cont.deinit(al);
    const flat = arr_cont.flat().?;
    const expected = [_]i32{
        0,  3,  6,
        9,  12, 15,
        18, 21, 24,
        27, 30, 33,
    };
    try std.testing.expectEqualSlices(i32, &expected, flat);

    // Test also for Const
    const arr_cont_const = arr_cont.asConst();
    const flat_const = arr_cont_const.flat().?;
    try std.testing.expectEqualSlices(i32, &expected, flat_const);
}

test "get*" {
    // Test all the get* methods, both for NamedArray and NamedArrayConst
    const IJ = enum { i, j };
    const idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });
    var buf = [_]i32{ 10, 11, 12, 13, 14, 15 };
    const arr = NamedArray(IJ, i32){
        .idx = idx,
        .buf = &buf,
    };
    const arr_const = arr.asConst();

    // Test get (in bounds)
    try std.testing.expectEqual(arr.getValChecked(.{ .i = 1, .j = 2 }), 15);
    try std.testing.expectEqual(arr_const.getValChecked(.{ .i = 1, .j = 2 }), 15);

    // Test get (out of bounds)
    try std.testing.expectEqual(arr.getValChecked(.{ .i = 2, .j = 0 }), null);
    try std.testing.expectEqual(arr_const.getValChecked(.{ .i = 2, .j = 0 }), null);

    // Test getUnchecked
    try std.testing.expectEqual(arr.getVal(.{ .i = 0, .j = 1 }), 11);
    try std.testing.expectEqual(arr_const.getVal(.{ .i = 0, .j = 1 }), 11);

    // Test getPtr (in bounds)
    const ptr = arr.getPtrChecked(.{ .i = 1, .j = 0 }).?;
    try std.testing.expectEqual(ptr.*, 13);
    const ptr_const = arr_const.getPtrChecked(.{ .i = 1, .j = 0 }).?;
    try std.testing.expectEqual(ptr_const.*, 13);

    // Test getPtr (out of bounds)
    try std.testing.expectEqual(arr.getPtrChecked(.{ .i = 5, .j = 0 }), null);
    try std.testing.expectEqual(arr_const.getPtrChecked(.{ .i = 5, .j = 0 }), null);

    // Test getPtrUnchecked
    const ptr_unchecked = arr.getPtr(.{ .i = 0, .j = 2 });
    try std.testing.expectEqual(ptr_unchecked.*, 12);
    const ptr_const_unchecked = arr_const.getPtr(.{ .i = 0, .j = 2 });
    try std.testing.expectEqual(ptr_const_unchecked.*, 12);

    // Test setUnchecked
    arr.setVal(.{ .i = 1, .j = 1 }, 99);
    try std.testing.expectEqual(arr.getVal(.{ .i = 1, .j = 1 }), 99);
}

test "dot 1d mixed types" {
    const Axis = enum { i };
    const idx = NamedIndex(Axis).initContiguous(.{ .i = 3 });
    const arr1 = NamedArrayConst(Axis, f32){ .idx = idx, .buf = &[_]f32{ 1, 2, 3 } };
    const arr2 = NamedArrayConst(Axis, f64){ .idx = idx, .buf = &[_]f64{ 4.0, 5.0, 6.0 } };
    const result = dot(Axis, f32, f64, arr1, arr2);
    try std.testing.expectEqual(result, 32.0); // 1*4.0 + 2*5.0 + 3*6.0 = 32.0
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

// test "einstein" {
//     const IJ = enum { i, j };
//     const JK = enum { j, k };

//     const al = std.testing.allocator;

//     const arr_ij = try NamedArray(IJ, f64).initAlloc(al, .{ .i = 4, .j = 3 });
//     defer arr_ij.deinit(al);
//     arr_ij.fillArange();

//     const arr_jk = try NamedArray(JK, f64).initAlloc(al, .{ .j = 3, .k = 2 });
//     defer arr_jk.deinit(al);
//     arr_jk.fill(1);

//     const arr_ik = einstein(al, arr_ij.asConst(), arr_jk.asConst());
//     defer arr_ik.deinit(al);

//     std.testing.expectEqual(arr_ik.shape, .{ .i = 4, .k = 2 });
// }
