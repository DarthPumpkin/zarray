const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const Type = std.builtin.Type;

fn NamedIndex(comptime Key: type) type {
    const key_info = @typeInfo(Key).@"struct";
    if (key_info.layout != .@"packed")
        @compileError("Key struct must have packed layout.");
    return struct {
        shape: Key,
        strides: Key,
        offset: usize = 0,

        pub const Field = meta.FieldEnum(Key);
        const fields = meta.fields(Field);

        const usize_null: ?usize = null;

        pub const Key_ = Key;
        /// Same fields as `Key`, but they are optional.
        pub const KeyOptional = optional_type: {
            const optional_fields = fields: {
                var optional_fields_: [fields.len]Type.StructField = undefined;
                for (0..fields.len) |fi| {
                    optional_fields_[fi] = .{
                        .name = fields[fi].name,
                        .type = @Type(.{ .optional = .{ .child = usize } }),
                        .default_value_ptr = &usize_null,
                        .is_comptime = false,
                        .alignment = 0,
                    };
                }
                break :fields optional_fields_;
            };
            const type_info: Type = .{ .@"struct" = .{
                .layout = .auto,
                .fields = &optional_fields,
                .decls = &[_]Type.Declaration{},
                .is_tuple = false,
            } };
            const type_ = @Type(type_info);
            break :optional_type type_;
        };

        /// Create contiguous index in "row-major" order, where the last field is treated as the
        /// 'row' dimension.
        pub fn initContiguous(shape: Key) @This() {
            const rank = fields.len;
            const fields_rev = comptime rev: {
                var fields_rev_: [rank]Type.EnumField = undefined;
                mem.copyForwards(Type.EnumField, &fields_rev_, fields);
                mem.reverse(Type.EnumField, &fields_rev_);
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
            return KeyIterator(Key).init(self.shape, self.strides);
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
        pub fn strideAxis(self: *const @This(), comptime axis: []const u8, step: usize) @This() {
            var new_strides = self.strides;
            var new_shape = self.shape;
            const new_dim = &@field(new_shape, axis);
            const new_stride = &@field(new_strides, axis);
            strideInplace(step, new_dim, new_stride);
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Stride multiple axes by given step sizes.
        /// axes whose value is null in `steps` are skipped.
        pub fn stride(self: *const @This(), steps: KeyOptional) @This() {
            var new_strides = self.strides;
            var new_shape = self.shape;
            inline for (fields) |field| {
                if (@field(steps, field.name)) |step| {
                    const new_dim = &@field(new_shape, field.name);
                    const new_stride = &@field(new_strides, field.name);
                    strideInplace(step, new_dim, new_stride);
                }
            }
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        fn strideInplace(step: usize, out_dim: *usize, out_stride: *usize) void {
            if (step == 0)
                @panic("step must be positive");
            out_stride.* *= step;
            const orig_dim = out_dim.*;
            out_dim.* /= step;
            if (orig_dim % step > 0) {
                out_dim.* += 1;
            }
        }

        /// Slice a given axis.
        /// Equivalent to `start:end` syntax in python.
        /// Panics if `start` or `end` are out of bounds.
        // zig fmt: off
        pub fn sliceAxis(self: *const @This(),
            comptime axis: []const u8,
            start: usize,
            end: usize) @This() {
        // zig fmt: on
            const old_size = @field(self.shape, axis);
            if (end > old_size)
                @panic("slice end out of bounds");
            if (start >= end)
                @panic("slice start must be less than end");
            var new_shape = self.shape;
            @field(new_shape, axis) = end - start;
            var offset_lookup = mem.zeroes(Key);
            @field(offset_lookup, axis) = start;
            const new_offset = self.linearUnchecked(offset_lookup);
            return .{
                .shape = new_shape,
                .strides = self.strides,
                .offset = new_offset,
            };
        }

        /// Return the number of elements in this index.
        pub fn count(self: *const @This()) usize {
            var prod: usize = 1;
            inline for (fields) |field| {
                prod *= @field(self.shape, field.name);
            }
            return prod;
        }

        /// Rename an axis.
        // zig fmt: off
        pub fn rename(self: *const @This(),
            comptime old_name: [:0]const u8,
            comptime new_name: [:0]const u8
        ) NamedIndex(RenamedStructField(Key, old_name, new_name)) {
        // zig fmt: on
            const NewKey = RenamedStructField(Key, old_name, new_name);
            const new_shape: NewKey = @bitCast(self.shape);
            const new_strides: NewKey = @bitCast(self.strides);
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }
    };
}

/// Iterates over all valid indices for a `NamedIndex` with given shape.
/// Iteration order is according to stride order.
/// That is, the keys are returned in the order they are encountered in the underlying buffer.
pub fn KeyIterator(comptime Key: type) type {
    const Field = meta.FieldEnum(Key);
    const fnames = meta.fieldNames(Field);

    return struct {
        next_arr: [fnames.len]usize,
        shape_arr: [fnames.len]usize,
        dims_desc: [fnames.len]usize,

        pub fn init(shape: Key, strides: Key) @This() {
            const start = [_]usize{0} ** fnames.len;
            const shape_arr: [fnames.len]usize = @bitCast(shape);
            const strides_arr: [fnames.len]usize = @bitCast(strides);

            const dims_desc = argsort: {
                var argsort: [fnames.len]usize = undefined;
                for (0..fnames.len) |i| {
                    argsort[i] = i;
                }
                const strides_slice: []const usize = strides_arr[0..];
                mem.sort(usize, argsort[0..], strides_slice, fnames_lt);
                break :argsort argsort;
            };
            return .{ .next_arr = start, .shape_arr = shape_arr, .dims_desc = dims_desc };
        }

        pub fn next(self: *@This()) ?Key {
            if (self.next_arr[self.dims_desc[0]] >= self.shape_arr[self.dims_desc[0]])
                return null;
            const result_arr = self.next_arr;

            // Update next
            inline for (0..fnames.len) |di| {
                const dim_idx = fnames.len - 1 - di;
                const dim = self.dims_desc[dim_idx];
                if (self.next_arr[dim] + 1 < self.shape_arr[dim]) {
                    self.next_arr[dim] += 1;
                    break;
                } else {
                    // carry over
                    if (dim_idx != 0) {
                        self.next_arr[dim] = 0;
                    } else {
                        self.next_arr[dim] = self.shape_arr[dim];
                    }
                }
            }

            const result: Key = @bitCast(result_arr);
            return result;
        }

        fn fnames_lt(strides_arr: []const usize, lhs: usize, rhs: usize) bool {
            // descending order
            return strides_arr[lhs] > strides_arr[rhs];
        }
    };
}

fn RenamedStructField(comptime OldKey: type, old_name: [:0]const u8, new_name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"struct";
    const new_fields = comptime fields: {
        var new_fields: [old_struct.fields.len]Type.StructField = undefined;
        var matched = false;
        for (0..old_struct.fields.len) |fi| {
            const old_field = old_struct.fields[fi];
            if (mem.eql(u8, old_field.name, old_name)) {
                const new_field: Type.StructField = .{
                    .alignment = old_field.alignment,
                    .default_value_ptr = old_field.default_value_ptr,
                    .is_comptime = old_field.is_comptime,
                    .type = old_field.type,
                    .name = new_name,
                };
                new_fields[fi] = new_field;
                matched = true;
            } else {
                new_fields[fi] = old_field;
            }
        }
        if (!matched)
            @compileError("rename: field not found in struct: " ++ old_name);
        break :fields new_fields;
    };
    const new_struct: Type.Struct = .{
        .layout = old_struct.layout,
        .backing_integer = old_struct.backing_integer,
        .fields = &new_fields,
        .decls = old_struct.decls,
        .is_tuple = old_struct.is_tuple,
    };
    return @Type(Type{ .@"struct" = new_struct });
}

const Index2d = packed struct { row: usize, col: usize };

test "init strides" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{
        .shape = .{ .row = 4, .col = 5 },
        .strides = .{ .row = 5, .col = 1 },
    };
    const idx2 = Structure2d.initContiguous(.{ .row = 4, .col = 5 });
    try std.testing.expectEqual(idx, idx2);
}

test "linear index" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{
        .shape = .{ .row = 4, .col = 5 },
        .strides = .{ .row = 5, .col = 1 },
        .offset = 7,
    };
    const query: Index2d = .{ .row = 2, .col = 3 };
    const expected = 20;
    try std.testing.expectEqual(expected, idx.linearUnchecked(query));
}

test "linear invalid index" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{
        .shape = .{ .row = 4, .col = 5 },
        .strides = .{ .row = 5, .col = 1 },
        .offset = 7,
    };
    const query: Index2d = .{ .row = 4, .col = 3 };
    const expected = null;
    try std.testing.expectEqual(expected, idx.linear(query));
}

test "strideAxis" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{
        .shape = .{ .row = 6, .col = 8 },
        .strides = .{ .row = 8, .col = 1 },
        .offset = 1,
    };
    const stepped = idx.strideAxis("col", 3);
    try std.testing.expectEqual(6, stepped.shape.row);
    try std.testing.expectEqual(3, stepped.shape.col);
    try std.testing.expectEqual(8, stepped.strides.row);
    try std.testing.expectEqual(3, stepped.strides.col);
    try std.testing.expectEqual(1, stepped.offset);
}

test "stride" {
    const Structure2d = NamedIndex(Index2d);
    const KeyOptional = Structure2d.KeyOptional;

    // Test with both fields set
    const idx: Structure2d = .{
        .shape = .{ .row = 6, .col = 8 },
        .strides = .{ .row = 8, .col = 1 },
        .offset = 1,
    };
    const steps: KeyOptional = .{ .row = 2, .col = 1 };
    const stepped = idx.stride(steps);
    try std.testing.expectEqual(3, stepped.shape.row);
    try std.testing.expectEqual(8, stepped.shape.col);
    try std.testing.expectEqual(16, stepped.strides.row);
    try std.testing.expectEqual(1, stepped.strides.col);
    try std.testing.expectEqual(1, stepped.offset);

    // Test with only one field set
    const stepped_row = idx.stride(.{ .row = 2 });
    try std.testing.expectEqual(3, stepped_row.shape.row);
    try std.testing.expectEqual(8, stepped_row.shape.col);
    try std.testing.expectEqual(16, stepped_row.strides.row);
    try std.testing.expectEqual(1, stepped_row.strides.col);
    try std.testing.expectEqual(1, stepped_row.offset);

    // Test with no fields set (should be identical to original)
    const stepped_none = idx.stride(.{});
    try std.testing.expectEqual(idx, stepped_none);
}

test "sliceAxis" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{
        .shape = .{ .row = 6, .col = 8 },
        .strides = .{ .row = 8, .col = 1 },
        .offset = 3,
    };
    const sliced = idx.sliceAxis("row", 2, 5);
    try std.testing.expectEqual(3, sliced.shape.row);
    try std.testing.expectEqual(8, sliced.shape.col);
    try std.testing.expectEqual(8, sliced.strides.row);
    try std.testing.expectEqual(1, sliced.strides.col);
    try std.testing.expectEqual(19, sliced.offset);
}

test "iterKeys" {
    const Structure2d = NamedIndex(Index2d);
    const idx: Structure2d = .{
        .shape = .{ .row = 2, .col = 3 },
        .strides = .{ .row = 3, .col = 1 },
        .offset = 0,
    };
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

test "iterKeys 3d, major in middle" {
    const Index3d = packed struct { x: usize, y: usize, z: usize };
    const Structure3d = NamedIndex(Index3d);
    // Make y the major dimension (stride 1 for y, then z, then x)
    const idx: Structure3d = .{
        .shape = .{ .x = 2, .y = 3, .z = 4 },
        .strides = .{ .x = 12, .y = 1, .z = 3 },
        .offset = 0,
    };
    // The iteration order should be: y varies fastest, then z, then x
    const expected_indices: [24]Index3d = .{
        .{ .x = 0, .y = 0, .z = 0 },
        .{ .x = 0, .y = 1, .z = 0 },
        .{ .x = 0, .y = 2, .z = 0 },
        .{ .x = 0, .y = 0, .z = 1 },
        .{ .x = 0, .y = 1, .z = 1 },
        .{ .x = 0, .y = 2, .z = 1 },
        .{ .x = 0, .y = 0, .z = 2 },
        .{ .x = 0, .y = 1, .z = 2 },
        .{ .x = 0, .y = 2, .z = 2 },
        .{ .x = 0, .y = 0, .z = 3 },
        .{ .x = 0, .y = 1, .z = 3 },
        .{ .x = 0, .y = 2, .z = 3 },
        .{ .x = 1, .y = 0, .z = 0 },
        .{ .x = 1, .y = 1, .z = 0 },
        .{ .x = 1, .y = 2, .z = 0 },
        .{ .x = 1, .y = 0, .z = 1 },
        .{ .x = 1, .y = 1, .z = 1 },
        .{ .x = 1, .y = 2, .z = 1 },
        .{ .x = 1, .y = 0, .z = 2 },
        .{ .x = 1, .y = 1, .z = 2 },
        .{ .x = 1, .y = 2, .z = 2 },
        .{ .x = 1, .y = 0, .z = 3 },
        .{ .x = 1, .y = 1, .z = 3 },
        .{ .x = 1, .y = 2, .z = 3 },
    };
    var iter = idx.iterKeys();
    var i: usize = 0;
    while (iter.next()) |next| {
        try std.testing.expectEqual(expected_indices[i], next);
        i += 1;
    }
    try std.testing.expectEqual(expected_indices.len, i);
}

test "count" {
    const Structure2d = NamedIndex(Index2d);
    const idx1: Structure2d = .{
        .shape = .{ .row = 5, .col = 2 },
        .strides = .{ .row = 2, .col = 1 },
        .offset = 7,
    };
    try std.testing.expectEqual(@as(usize, 10), idx1.count());

    // Test with a degenerate dimension
    const idx2: Structure2d = .{
        .shape = .{ .row = 0, .col = 4 },
        .strides = .{ .row = 4, .col = 1 },
        .offset = 0,
    };
    try std.testing.expectEqual(@as(usize, 0), idx2.count());
}

test "rename" {
    const IJ = packed struct { i: usize, j: usize };
    const IndexIJ = NamedIndex(IJ);
    const idx: IndexIJ = .initContiguous(.{
        .i = 5,
        .j = 3,
    });
    const idx_from = idx.rename("i", "from");

    // Check that the renamed indices have the expected field names and values
    try std.testing.expectEqual(idx.shape.i, idx_from.shape.from);
    try std.testing.expectEqual(idx.shape.j, idx_from.shape.j);

    // Check that the strides are preserved
    try std.testing.expectEqual(idx.strides.i, idx_from.strides.from);
    try std.testing.expectEqual(idx.strides.j, idx_from.strides.j);

    // Check that the offset is preserved
    try std.testing.expectEqual(idx.offset, idx_from.offset);

    // Toggle this manually to verify that it throws a compileError.
    if (false) {
        _ = idx.rename("nonexisting", "foo");
    }
}
