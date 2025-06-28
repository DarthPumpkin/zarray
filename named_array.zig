const std = @import("std");
const mem = std.mem;

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
                self.buf[self.idx.linearUnchecked(key)] = val;
            }
            return self;
        }

        pub fn fillArange(self: *const @This()) *const @This() {
            var keys = self.idx.iterKeys();
            var i: Scalar = 0;
            while (keys.next()) |key| {
                self.buf[self.idx.linearUnchecked(key)] = i;
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
    };
}

pub fn NamedArrayConst(comptime Axis: type, comptime Scalar: type) type {
    const Index = NamedIndex(Axis);
    return struct {
        idx: Index,
        buf: []const Scalar,
    };
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
    // TODO: Check that arr_out.idx is non-overlapping, instead.
    if (arr1.idx.count() != arr_out.buf.len)
        @panic("Mismatched buffer sizes");
    var keys = arr1.idx.iterKeys();
    while (keys.next()) |key| {
        const l = &arr1.buf[arr1.idx.linearUnchecked(key)];
        const r = &arr2.buf[arr2.idx.linearUnchecked(key)];
        const out = &arr_out.buf[arr_out.idx.linearUnchecked(key)];
        out.* = l.* + r.*;
    }
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
