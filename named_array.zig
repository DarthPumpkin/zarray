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

        /// Merge several axes of this array into a single axis described by `NewEnum`.
        /// This is a zero-copy view transformation; it fails if the axes to be merged
        /// are not laid out contiguously according to stride relationships.
        pub fn mergeAxes(self: *const @This(), comptime NewEnum: type) !NamedArray(NewEnum, Scalar) {
            const new_idx = try self.idx.mergeAxes(NewEnum);
            return .{
                .idx = new_idx,
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

        /// Merge several axes of this const array into a single axis described by `NewEnum`.
        /// This is a zero-copy view transformation; it fails if the axes to be merged
        /// are not laid out contiguously according to stride relationships.
        pub fn mergeAxes(self: *const @This(), comptime NewEnum: type) !NamedArrayConst(NewEnum, Scalar) {
            const new_idx = try self.idx.mergeAxes(NewEnum);
            return .{
                .idx = new_idx,
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
    const arr_ix = arr.renameAxes(IX, &.{.{ .old = "j", .new = "x" }});
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
        _ = arr.renameAxes(IJK, &.{
            .{ .old = "i", .new = "i" },
            .{ .old = "j", .new = "j" },
            .{ .old = "k", .new = "k" },
        });
    }
    if (false) {
        const I = enum { i };
        _ = arr.renameAxes(I, &.{.{ .old = "i", .new = "i" }});
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
    const arr_ix = arr.renameAxes(IX, &.{.{ .old = "j", .new = "x" }});
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
        _ = arr.renameAxes(IJK, &.{
            .{ .old = "i", .new = "i" },
            .{ .old = "j", .new = "j" },
            .{ .old = "k", .new = "k" },
        });
    }
    if (false) {
        const I = enum { i };
        _ = arr.renameAxes(I, &.{.{ .old = "i", .new = "i" }});
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
            .{ .old = "i", .new = "x" },
            .{ .old = "i", .new = "y" },
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

test "mergeAxes" {
    const IJK = enum { i, j, k };
    const IL = enum { i, l };

    const buf_1_through_24 = [_]i32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    };

    const arr_ijk = NamedArrayConst(IJK, i32){
        .idx = .initContiguous(.{
            .i = 2,
            .j = 3,
            .k = 4,
        }),
        .buf = &buf_1_through_24,
    };
    const arr_il = try arr_ijk.mergeAxes(IL);

    try std.testing.expectEqual(NamedIndex(IL).Axes{ .i = 2, .l = 12 }, arr_il.idx.shape);
    try std.testing.expectEqual(2, arr_il.scalarAt(.{ .i = 0, .l = 1 }));
    try std.testing.expectEqual(15, arr_il.scalarAt(.{ .i = 1, .l = 2 }));

    // Failing case: last dim has stride 3, but shape 4 -> cannot merge without copying
    const arr_ijk_strided = NamedArrayConst(IJK, i32){
        .idx = .{
            .strides = .{ .i = 12, .j = 4, .k = 3 },
            .shape = .{ .i = 2, .j = 3, .k = 2 },
        },
        .buf = &buf_1_through_24,
    };

    try std.testing.expectError(named_index.NamedIndexError.StrideMisalignment, arr_ijk_strided.mergeAxes(IL));

    // Failing case: axes not consecutive (j i k -> i l)
    const arr_ijk_noncon = NamedArrayConst(IJK, i32){
        .idx = .{
            .strides = .{ .i = 4, .j = 8, .k = 1 },
            .shape = .{ .i = 2, .j = 3, .k = 4 },
        },
        .buf = &buf_1_through_24,
    };

    try std.testing.expectError(named_index.NamedIndexError.StrideMisalignment, arr_ijk_noncon.mergeAxes(IL));

    // Edge case: shape (1, 1, 1)
    const buf_single = [_]i32{42};
    const arr_ones = NamedArrayConst(IJK, i32){
        .idx = .initContiguous(.{ .i = 1, .j = 1, .k = 1 }),
        .buf = &buf_single,
    };
    const arr_ones_merged = try arr_ones.mergeAxes(IL);
    try std.testing.expectEqual(NamedIndex(IL).Axes{ .i = 1, .l = 1 }, arr_ones_merged.idx.shape);
    try std.testing.expectEqual(42, arr_ones_merged.scalarAt(.{ .i = 0, .l = 0 }));
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
