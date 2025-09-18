const std = @import("std");
const mem = std.mem;
const meta = std.meta;

const named_index = @import("named_index.zig");
const NamedIndex = named_index.NamedIndex;
const AxisRenamePair = named_index.AxisRenamePair;

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

        pub fn atChecked(self: *const @This(), key: Index.Axes) ?*Scalar {
            return getPtrCheckedGeneric(self, key);
        }

        pub fn at(self: *const @This(), key: Index.Axes) *Scalar {
            return &self.buf[self.idx.linear(key)];
        }
        pub fn scalarAtChecked(self: *const @This(), key: Index.Axes) ?Scalar {
            return getValCheckedGeneric(self, key);
        }

        pub fn scalarAt(self: *const @This(), key: Index.Axes) Scalar {
            return self.asConst().scalarAt(key);
        }

        /// Return a new NamedArray with axes conformed to NewEnum.
        pub fn conformAxes(self: *const @This(), comptime NewEnum: type) NamedArray(NewEnum, Scalar) {
            return .{
                .idx = self.idx.conformAxes(NewEnum),
                .buf = self.buf,
            };
        }

        /// Strictly rename axes according to the provided mapping.
        /// If any axis in NewEnum cannot be mapped, this will fail to compile.
        pub fn renameAxes(self: *const @This(), comptime NewEnum: type, comptime rename_pairs: []const AxisRenamePair) NamedArray(NewEnum, Scalar) {
            return .{
                .idx = self.idx.rename(NewEnum, rename_pairs),
                .buf = self.buf,
            };
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

        pub fn atChecked(self: *const @This(), key: Index.Axes) ?*const Scalar {
            return getPtrCheckedGeneric(self, key);
        }

        pub fn at(self: *const @This(), key: Index.Axes) *const Scalar {
            return &self.buf[self.idx.linear(key)];
        }

        pub fn scalarAtChecked(self: *const @This(), key: Index.Axes) ?Scalar {
            return getValCheckedGeneric(self, key);
        }

        pub fn scalarAt(self: *const @This(), key: Index.Axes) Scalar {
            return self.buf[self.idx.linear(key)];
        }

        /// Return a new NamedArrayConst with axes conformed to NewEnum.
        pub fn conformAxes(self: *const @This(), comptime NewEnum: type) NamedArrayConst(NewEnum, Scalar) {
            return .{
                .idx = self.idx.conformAxes(NewEnum),
                .buf = self.buf,
            };
        }

        /// Strictly rename axes according to the provided mapping.
        /// If any axis in NewEnum cannot be mapped, this will fail to compile.
        pub fn renameAxes(self: *const @This(), comptime NewEnum: type, comptime rename_pairs: []const AxisRenamePair) NamedArrayConst(NewEnum, Scalar) {
            return .{
                .idx = self.idx.rename(NewEnum, rename_pairs),
                .buf = self.buf,
            };
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
            buf[i] = self.scalarAt(key);
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
        const l = arr1.scalarAt(key);
        const r = arr2.scalarAt(key);
        arr_out.at(key).* = l + r;
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
            sum = arrA.scalarAt(keyA) * arrB.scalarAt(keyB);
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
    try std.testing.expectEqual(arr.scalarAtChecked(.{ .i = 1, .j = 2 }).?, 15);
    try std.testing.expectEqual(arr_const.scalarAtChecked(.{ .i = 1, .j = 2 }).?, 15);

    // Test get (out of bounds)
    try std.testing.expectEqual(arr.atChecked(.{ .i = 2, .j = 0 }), null);
    try std.testing.expectEqual(arr_const.atChecked(.{ .i = 2, .j = 0 }), null);

    // Test getUnchecked
    try std.testing.expectEqual(arr.at(.{ .i = 0, .j = 1 }).*, 11);
    try std.testing.expectEqual(arr_const.at(.{ .i = 0, .j = 1 }).*, 11);

    // Test getPtr (in bounds)
    const ptr = arr.atChecked(.{ .i = 1, .j = 0 }).?;
    try std.testing.expectEqual(ptr.*, 13);
    const ptr_const = arr_const.atChecked(.{ .i = 1, .j = 0 }).?;
    try std.testing.expectEqual(ptr_const.*, 13);

    // Test getPtr (out of bounds)
    try std.testing.expectEqual(arr.atChecked(.{ .i = 5, .j = 0 }), null);
    try std.testing.expectEqual(arr_const.atChecked(.{ .i = 5, .j = 0 }), null);

    // Test getPtrUnchecked
    const ptr_unchecked = arr.at(.{ .i = 0, .j = 2 });
    try std.testing.expectEqual(ptr_unchecked.*, 12);
    const ptr_const_unchecked = arr_const.at(.{ .i = 0, .j = 2 });
    try std.testing.expectEqual(ptr_const_unchecked.*, 12);

    // Test setUnchecked
    arr.at(.{ .i = 1, .j = 1 }).* = 99;
    try std.testing.expectEqual(arr.scalarAt(.{ .i = 1, .j = 1 }), 99);
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

test "stride" {
    const IJ = enum { i, j };
    const idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });
    var buf = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = NamedArray(IJ, i32){
        .idx = idx,
        .buf = &buf,
    };
    arr.idx = arr.idx.strideAxis(.j, 2);

    const actual = [_]i32{
        arr.scalarAt(.{ .i = 0, .j = 0 }),
        arr.scalarAt(.{ .i = 0, .j = 1 }),
        arr.scalarAt(.{ .i = 1, .j = 0 }),
        arr.scalarAt(.{ .i = 1, .j = 1 }),
    };
    const expected = [_]i32{ 1, 3, 4, 6 };
    try std.testing.expectEqualSlices(i32, &actual, &expected);

    // Also test that .flat() returns null for non-contiguous
    try std.testing.expectEqual(arr.flat(), null);

    // After making it contiguous, .flat() works.
    const allocator = std.testing.allocator;
    const arr_cont = try arr.toContiguous(allocator);
    defer arr_cont.deinit(allocator);
    const flat = arr_cont.flat().?;
    try std.testing.expectEqualSlices(i32, &actual, flat);
}
test "NamedArray renameAxes strict" {
    const IJ = enum { i, j };
    const XY = enum { x, y };

    var buf = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const arr = NamedArray(IJ, i32){
        .idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }),
        .buf = &buf,
    };
    // Rename i->x, j->y
    const arr_xy = arr.renameAxes(XY, &.{
        .{ .old = "i", .new = "x" },
        .{ .old = "j", .new = "y" },
    });
    try std.testing.expectEqual(arr.buf, arr_xy.buf);
    try std.testing.expectEqual(arr.idx.shape.i, arr_xy.idx.shape.x);
    try std.testing.expectEqual(arr.idx.shape.j, arr_xy.idx.shape.y);
    try std.testing.expectEqual(arr.idx.strides.i, arr_xy.idx.strides.x);
    try std.testing.expectEqual(arr.idx.strides.j, arr_xy.idx.strides.y);

    // Direct match (no rename needed)
    const arr_direct = arr.renameAxes(IJ, &.{});
    try std.testing.expectEqual(arr.idx, arr_direct.idx);

    // Common axes omitted from rename list (only rename one axis)
    const IX = enum { i, x };
    const arr_ix = arr.renameAxes(IX, &.{AxisRenamePair{ .old = "j", .new = "x" }});
    try std.testing.expectEqual(arr.idx.shape.i, arr_ix.idx.shape.i);
    try std.testing.expectEqual(arr.idx.shape.j, arr_ix.idx.shape.x);
    try std.testing.expectEqual(arr.idx.strides.i, arr_ix.idx.strides.i);
    try std.testing.expectEqual(arr.idx.strides.j, arr_ix.idx.strides.x);

    // Should fail to compile if mapping is missing
    if (false) {
        const AB = enum { a, b };
        _ = arr.renameAxes(AB, &.{});
    }

    // Should fail to compile if enums have different numbers of axes
    if (false) {
        const IJK = enum { i, j, k };
        _ = arr.renameAxes(IJK, &.{ AxisRenamePair{ .old = "i", .new = "i" }, AxisRenamePair{ .old = "j", .new = "j" }, AxisRenamePair{ .old = "k", .new = "k" } });
    }
    if (false) {
        const I = enum { i };
        _ = arr.renameAxes(I, &.{AxisRenamePair{ .old = "i", .new = "i" }});
    }
}

test "NamedArrayConst renameAxes strict" {
    const IJ = enum { i, j };
    const XY = enum { x, y };

    const buf = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const arr = NamedArrayConst(IJ, i32){
        .idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }),
        .buf = &buf,
    };
    // Rename i->x, j->y
    const arr_xy = arr.renameAxes(XY, &.{
        .{ .old = "i", .new = "x" },
        .{ .old = "j", .new = "y" },
    });
    try std.testing.expectEqual(arr.buf, arr_xy.buf);
    try std.testing.expectEqual(arr.idx.shape.i, arr_xy.idx.shape.x);
    try std.testing.expectEqual(arr.idx.shape.j, arr_xy.idx.shape.y);
    try std.testing.expectEqual(arr.idx.strides.i, arr_xy.idx.strides.x);
    try std.testing.expectEqual(arr.idx.strides.j, arr_xy.idx.strides.y);

    // Direct match (no rename needed)
    const arr_direct = arr.renameAxes(IJ, &.{});
    try std.testing.expectEqual(arr.idx, arr_direct.idx);

    // Common axes omitted from rename list (only rename one axis)
    const IX = enum { i, x };
    const arr_ix = arr.renameAxes(IX, &.{AxisRenamePair{ .old = "j", .new = "x" }});
    try std.testing.expectEqual(arr.idx.shape.i, arr_ix.idx.shape.i);
    try std.testing.expectEqual(arr.idx.shape.j, arr_ix.idx.shape.x);
    try std.testing.expectEqual(arr.idx.strides.i, arr_ix.idx.strides.i);
    try std.testing.expectEqual(arr.idx.strides.j, arr_ix.idx.strides.x);

    // Should fail to compile if mapping is missing
    if (false) {
        const AB = enum { a, b };
        _ = arr.renameAxes(AB, &.{});
    }

    // Should fail to compile if enums have different numbers of axes
    if (false) {
        const IJK = enum { i, j, k };
        _ = arr.renameAxes(IJK, &.{ AxisRenamePair{ .old = "i", .new = "i" }, AxisRenamePair{ .old = "j", .new = "j" }, AxisRenamePair{ .old = "k", .new = "k" } });
    }
    if (false) {
        const I = enum { i };
        _ = arr.renameAxes(I, &.{AxisRenamePair{ .old = "i", .new = "i" }});
    }
}
test "renameAxes fails if old axis is mapped twice" {
    const IJ = enum { i, j };
    const XY = enum { x, y };
    var buf = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const arr = NamedArray(IJ, i32){
        .idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }),
        .buf = &buf,
    };
    if (false) {
        // "i" is mapped to both "x" and "y"
        _ = arr.renameAxes(XY, &.{
            AxisRenamePair{ .old = "i", .new = "x" },
            AxisRenamePair{ .old = "i", .new = "y" },
        });
    }
}

test "slice" {
    const IJ = enum { i, j };
    var arr = NamedArrayConst(IJ, i32){
        .idx = .initContiguous(.{ .i = 2, .j = 3 }),
        .buf = &[_]i32{ 1, 2, 3, 4, 5, 6 },
    };
    arr.idx = arr.idx.sliceAxis(.j, 1, 2);

    try std.testing.expectEqual(2, arr.scalarAt(.{ .i = 0, .j = 0 }));
    try std.testing.expectEqual(5, arr.scalarAt(.{ .i = 1, .j = 0 }));
    try std.testing.expectEqual(null, arr.scalarAtChecked(.{ .i = 1, .j = 1 }));
}

test "keepOnly" {
    const IJK = enum { i, j, k };
    const IJ = enum { i, j };

    const arr = NamedArrayConst(IJK, i32){
        .idx = .initContiguous(.{ .i = 4, .j = 1, .k = 1 }),
        .buf = &[_]i32{ 1, 2, 3, 4 },
    };

    const squeezed = NamedArrayConst(IJ, i32){
        .idx = arr.idx.keepOnly(IJ),
        .buf = arr.buf,
    };

    try std.testing.expectEqual(1, squeezed.scalarAt(.{ .i = 0, .j = 0 }));
    try std.testing.expectEqual(4, squeezed.scalarAt(.{ .i = 3, .j = 0 }));
    try std.testing.expectEqual(null, squeezed.scalarAtChecked(.{ .i = 0, .j = 1 }));

    //Should panic if any removed axis does not have size 1
    if (false) {
        const idx_bad: NamedIndex(IJK) = .initContiguous(.{ .i = 4, .j = 1, .k = 2 });
        _ = idx_bad.keepOnly(IJ);
    }
}

test "NamedArray conformAxes" {
    const IJK = enum { i, j, k };
    const IKL = enum { i, k, l };

    var buf = [_]i32{ 1, 2, 3, 4 };
    const arr = NamedArray(IJK, i32){
        .idx = NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 1, .k = 1 }),
        .buf = &buf,
    };
    const arr_proj = arr.conformAxes(IKL);
    try std.testing.expectEqual(arr.buf, arr_proj.buf);
    try std.testing.expectEqual(arr.idx.conformAxes(IKL), arr_proj.idx);

    // Should panic if removed axis does not have size 1
    if (false) {
        const arr_bad = NamedArray(IJK, i32){
            .idx = NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 2, .k = 1 }),
            .buf = &buf,
        };
        _ = arr_bad.conformAxes(IKL);
    }
}

test "NamedArrayConst conformAxes" {
    const IJK = enum { i, j, k };
    const IKL = enum { i, k, l };

    const buf = [_]i32{ 1, 2, 3, 4 };
    const arr = NamedArrayConst(IJK, i32){
        .idx = NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 1, .k = 1 }),
        .buf = &buf,
    };
    const arr_proj = arr.conformAxes(IKL);
    try std.testing.expectEqual(arr.buf, arr_proj.buf);
    try std.testing.expectEqual(arr.idx.conformAxes(IKL), arr_proj.idx);

    // Should panic if removed axis does not have size 1
    if (false) {
        const arr_bad = NamedArrayConst(IJK, i32){
            .idx = NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 2, .k = 1 }),
            .buf = &buf,
        };
        _ = arr_bad.conformAxes(IKL);
    }
}

// Rank-0 arrays currently not supported
// test "einsum dot product 2d x 2d -> 0d" {
//     const IJ = enum { i, j };
//     const arr1 = NamedArrayConst(IJ, i32){
//         .idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }),
//         .buf = &[_]i32{ 1, 2, 3, 4, 5, 6 },
//     };
//     const arr2 = NamedArrayConst(IJ, i32){
//         .idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }),
//         .buf = &[_]i32{ 7, 8, 9, 10, 11, 12 },
//     };
//     const allocator = std.testing.allocator;
//     const arr0d = try einsum(IJ, IJ, enum {}, i32, i32, arr1, arr2, allocator);
//     defer arr0d.deinit(allocator);

//     // The einsum should compute sum(arr1[i,j] * arr2[i,j]) over all i,j
//     // (1*7 + 2*8 + 3*9 + 4*10 + 5*11 + 6*12) = 7 + 16 + 27 + 40 + 55 + 72 = 217
//     try std.testing.expectEqual(1, arr0d.buf.len);
//     try std.testing.expectEqual(217, arr0d.buf[0]);
// }
