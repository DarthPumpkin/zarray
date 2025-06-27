const std = @import("std");
const mem = std.mem;

const named_index = @import("named_index.zig");
const NamedIndex = named_index.NamedIndex;

pub fn NamedArray(comptime Axis: type, comptime Scalar: type) type {
    const Index = NamedIndex(Axis);
    return struct {
        idx: Index,
        buf: []Scalar,

        pub fn empty(allocator: mem.Allocator, shape: Index.Axes) !@This() {
            const idx = Index.initContiguous(shape);
            return .{
                .idx = idx,
                .buf = try allocator.alloc(Scalar, idx.count()),
            };
        }

        pub fn zeros(allocator: mem.Allocator, shape: Index.Axes) !@This() {
            var self = try @This().empty(allocator, shape);
            for (0..self.idx.count()) |i| {
                self.buf[i] = 0;
            }
            return self;
        }

        pub fn ones(allocator: mem.Allocator, shape: Index.Axes) !@This() {
            var self = try @This().empty(allocator, shape);
            for (0..self.idx.count()) |i| {
                self.buf[i] = 1;
            }
            return self;
        }

        pub fn arange(allocator: mem.Allocator, shape: Index.Axes) !@This() {
            var self = try @This().empty(allocator, shape);
            for (0..self.idx.count()) |i| {
                self.buf[i] = i;
            }
            return self;
        }

        pub fn deinit(self: *@This(), allocator: mem.Allocator) void {
            allocator.free(self.buf);
        }
    };
}

/// If `idx_out` has overlapping linear indices, the output is undefined.
pub fn add(
    comptime Axis: type,
    comptime Scalar: type,
    idx1: NamedIndex(Axis),
    buf1: []const Scalar,
    idx2: NamedIndex(Axis),
    buf2: []const Scalar,
    idx_out: NamedIndex(Axis),
    buf_out: []Scalar,
) void {
    if (idx1.shape != idx2.shape or idx1.shape != idx_out.shape)
        @panic("Incompatible shapes");
    // TODO: Check that idx_out is non-overlapping, instead.
    if (idx1.count() != buf_out.len)
        @panic("Mismatched buffer sizes");
    var keys = idx1.iterKeys();
    while (keys.next()) |key| {
        const l = &buf1[idx1.linearUnchecked(key)];
        const r = &buf2[idx2.linearUnchecked(key)];
        const out = &buf_out[idx_out.linearUnchecked(key)];
        out.* = l.* + r.*;
    }
}

test "add inplace" {
    const Axis = enum { i };
    const idx = NamedIndex(Axis).initContiguous(.{ .i = 3 });
    const buf1 = [_]i32{ 1, 2, 3 };
    var buf2 = [_]i32{ 2, 2, 2 };
    add(Axis, i32, idx, &buf1, idx, &buf2, idx, &buf2);

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
    const buf1 = [_]i32{ 1, 2, 3 };
    const buf2 = [_]i32{ 1, 1, 1 };
    var buf_out: [12]i32 = undefined;
    add(IJ, i32, idx_broad, &buf1, idx_broad, &buf2, idx_out, &buf_out);

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

    const buf_row_major = [_]i32{
        1, 2, 3,
        4, 5, 6,
    };

    const buf_col_major = [_]i32{
        10, 40,
        20, 50,
        30, 60,
    };

    var buf_out: [6]i32 = undefined;
    // zig fmt: off
    add(IJ, i32,
        idx_row_major, &buf_row_major,
        idx_col_major, &buf_col_major,
        idx_row_major, &buf_out,);
    // zig fmt: on

    const expected = [_]i32{ 11, 22, 33, 44, 55, 66 };
    try std.testing.expectEqualSlices(i32, &expected, &buf_out);
}

test "zeros" {
    const Axis = enum { i };
    const allocator = std.testing.allocator;
    var arr = try NamedArray(Axis, i32).zeros(allocator, .{ .i = 4 });
    defer arr.deinit(allocator);

    const expected = [_]i32{ 0, 0, 0, 0 };
    try std.testing.expectEqualSlices(i32, &expected, arr.buf);
}
