const std = @import("std");
const meta = std.meta;
const Type = std.builtin.Type;

fn NamedIndex(comptime Key: type) type {
    return struct {
        shape: Key,
        strides: Key,
        offset: usize = 0,

        pub const Field = meta.FieldEnum(Key);
        const fields = meta.fields(Field);

        /// Create contiguous index in "row-major" order, where the last field is treated as the
        /// 'row' dimension.
        pub fn initContiguous(shape: Key) @This() {
            const rank = fields.len;
            const fields_rev = comptime rev: {
                var fields_rev_: [rank]Type.EnumField = undefined;
                std.mem.copyForwards(Type.EnumField, &fields_rev_, fields);
                std.mem.reverse(Type.EnumField, &fields_rev_);
                break :rev fields_rev_;
            };

            var strides: Key = undefined;
            var next_stride: usize = 1;
            inline for (fields_rev) |field_info| {
                @field(strides, field_info.name) = next_stride;
                next_stride *= @field(shape, field_info.name);
            }
            return .{
                .shape = shape,
                .strides = strides,
                .offset = 0,
            };
            // Alternative implementation via indexing
            // inline for (0..rank) |fi| {
            //     const field_info = fields[rank - 1 - fi];
            //     @field(strides, field_info.name) = next_stride;
            //     next_stride *= @field(shape, field_info.name);
            // }
        }

        pub fn iterKeys(self: *const @This()) KeyIterator(Key) {
            return KeyIterator(Key).init(self.shape);
        }

        /// Return the offset into the linear buffer for the given structured index.
        /// If the index is out of bounds, return null.
        pub fn linear(self: *const @This(), index: Key) ?usize {
            inline for (fields) |field_info| {
                if (@field(self.shape, field_info.name) <= @field(index, field_info.name))
                    return null;
            }
            return linearUnchecked(self, index);
        }

        pub fn linearUnchecked(self: *const @This(), index: Key) usize {
            var sum = self.offset;
            inline for (fields) |field_info| {
                sum += @field(self.strides, field_info.name) * @field(index, field_info.name);
            }
            return sum;
        }

        /// Stride a given axis by a given step size.
        /// Equivalent to `::step` syntax in python.
        /// Panics if `step` is zero.
        pub fn stride(self: *const @This(), comptime axis: []const u8, step: usize) @This() {
            if (step == 0)
                @panic("step must be positive");
            var new_strides = self.strides;
            var new_shape = self.shape;
            @field(new_strides, axis) *= step;
            var new_size = @field(self.shape, axis) / step;
            if (@field(self.shape, axis) % step > 0) {
                new_size += 1;
            }
            @field(new_shape, axis) = new_size;
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        pub fn strideAll(self: *const @This(), steps: Key) @This() {
            var new_strides = self.strides;
            var new_shape = self.shape;
            inline for (fields) |field| {
                const step = @field(steps, field.name);
                if (step == 0)
                    @panic("step must be positive");
                @field(new_strides, field.name) *= step;
                var new_size = @field(self.shape, field.name) / step;
                if (@field(self.shape, field.name) % step > 0) {
                    new_size += 1;
                }
                @field(new_shape, field.name) = new_size;
            }
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Slice a given axis.
        /// Equivalent to `start:end` syntax in python.
        /// Panics if `start` or `end` are out of bounds.
        pub fn slice(self: *const @This(), comptime axis: []const u8, start: usize, end: usize) @This() {
            const old_size = @field(self.shape, axis);
            if (end > old_size)
                @panic("slice end out of bounds");
            if (start >= end)
                @panic("slice start must be less than end");
            var new_shape = self.shape;
            @field(new_shape, axis) = end - start;
            var offset_lookup: Key = undefined;
            inline for (fields) |field| {
                @field(offset_lookup, field.name) = 0;
            }
            @field(offset_lookup, axis) = start;
            const new_offset = self.linearUnchecked(offset_lookup);
            return .{
                .shape = new_shape,
                .strides = self.strides,
                .offset = new_offset,
            };
        }
    };
}

/// Iterates over all valid indices for a `NamedIndex` with given shape.
/// Iteration order is determined by field order.
/// The last field in the struct varies the quickest.
pub fn KeyIterator(comptime Key: type) type {
    const Field = meta.FieldEnum(Key);
    const fnames = meta.fieldNames(Field);

    const fnames_rev = comptime rev: {
        var fnames_rev_: [fnames.len][]const u8 = undefined;
        @memcpy(&fnames_rev_, fnames);
        std.mem.reverse([]const u8, &fnames_rev_);
        break :rev fnames_rev_;
    };

    return struct {
        next_: Key,
        shape: Key,

        pub fn init(shape: Key) @This() {
            var start: Key = undefined;
            inline for (fnames) |fname| {
                @field(start, fname) = 0;
            }
            return .{ .next_ = start, .shape = shape };
        }

        pub fn next(self: *@This()) ?Key {
            if (@field(self.next_, fnames[0]) >= @field(self.shape, fnames[0]))
                return null;
            const result = self.next_;

            // Update next
            inline for (fnames_rev) |fname| {
                if (@field(self.next_, fname) + 1 < @field(self.shape, fname)) {
                    @field(self.next_, fname) += 1;
                    break;
                } else {
                    // carry over
                    if (!std.mem.eql(u8, fname, fnames[0])) {
                        @field(self.next_, fname) = 0;
                    } else {
                        @field(self.next_, fname) = @field(self.shape, fname);
                    }
                }
            }
            return result;
        }
    };
}

const Index2d = struct { row: usize, col: usize };

test "init strides" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 4, .col = 5 }, .strides = .{ .row = 5, .col = 1 } };
    const idx2 = Structure2d.initContiguous(.{ .row = 4, .col = 5 });
    try std.testing.expectEqual(idx, idx2);
}

test "linear index" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 4, .col = 5 }, .strides = .{ .row = 5, .col = 1 }, .offset = 7 };
    const query: Index2d = .{ .row = 2, .col = 3 };
    const expected = 20;
    try std.testing.expectEqual(expected, idx.linearUnchecked(query));
}

test "linear invalid index" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 4, .col = 5 }, .strides = .{ .row = 5, .col = 1 }, .offset = 7 };
    const query: Index2d = .{ .row = 4, .col = 3 };
    const expected = null;
    try std.testing.expectEqual(expected, idx.linear(query));
}

test "strideAll" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 6, .col = 8 }, .strides = .{ .row = 8, .col = 1 }, .offset = 1 };
    const stepped = idx.strideAll(.{ .row = 2, .col = 1 });
    try std.testing.expectEqual(3, stepped.shape.row);
    try std.testing.expectEqual(8, stepped.shape.col);
    try std.testing.expectEqual(16, stepped.strides.row);
    try std.testing.expectEqual(1, stepped.strides.col);
    try std.testing.expectEqual(1, stepped.offset);
}

test "stride" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 6, .col = 8 }, .strides = .{ .row = 8, .col = 1 }, .offset = 1 };
    const stepped = idx.stride("col", 3);
    try std.testing.expectEqual(6, stepped.shape.row);
    try std.testing.expectEqual(3, stepped.shape.col);
    try std.testing.expectEqual(8, stepped.strides.row);
    try std.testing.expectEqual(3, stepped.strides.col);
    try std.testing.expectEqual(1, stepped.offset);
}

test "slice" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 6, .col = 8 }, .strides = .{ .row = 8, .col = 1 }, .offset = 3 };
    const sliced = idx.slice("row", 2, 5);
    try std.testing.expectEqual(3, sliced.shape.row);
    try std.testing.expectEqual(8, sliced.shape.col);
    try std.testing.expectEqual(8, sliced.strides.row);
    try std.testing.expectEqual(1, sliced.strides.col);
    try std.testing.expectEqual(19, sliced.offset);
}

test "iterKeys" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{ .shape = .{ .row = 2, .col = 3 }, .strides = .{ .row = 3, .col = 1 }, .offset = 0 };
    const expected_indices: [6]Index2d = .{
        .{ .row = 0, .col = 0 },
        .{ .row = 0, .col = 1 },
        .{ .row = 0, .col = 2 },
        .{ .row = 1, .col = 0 },
        .{ .row = 1, .col = 1 },
        .{ .row = 1, .col = 2 },
    };
    var iter = idx.iterKeys();
    var i: usize = 0;
    while (iter.next()) |next| {
        try std.testing.expectEqual(expected_indices[i], next);
        i += 1;
    }
    try std.testing.expectEqual(expected_indices.len, i);
}
