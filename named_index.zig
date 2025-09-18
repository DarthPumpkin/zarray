const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const assert = std.debug.assert;
const Type = std.builtin.Type;

// Compile with "-framework Accelerate" on macOS
// const acc = @cImport(@cInclude("Accelerate/Accelerate.h"));

// TODO:
// - .init with config struct: validate shapes and strides
// - understand why KeyEnum is incompatible with manually created enums.

// Open questions:
// - Should we support dimensions of size 0?
// - Should we support rank-0 indices? If so, are they empty or scalars?
// - how should we interface with blas?
//   - wrap blas functions, or
//   - export NamedArray to blas arguments
pub fn NamedIndex(comptime AxisEnum: type) type {
    _ = @typeInfo(AxisEnum).@"enum";
    const field_names = meta.fieldNames(AxisEnum);
    return struct {
        shape: Axes,
        strides: Axes,
        offset: usize = 0,

        pub const Axis = AxisEnum;
        pub const Axes = KeyStruct(field_names);

        const usize_null: ?usize = null;
        /// Same fields as `Key`, but they are optional.
        pub const AxesOptional = optional_type: {
            const optional_fields = fields: {
                var optional_fields_: [field_names.len]Type.StructField = undefined;
                const optional_usize = @Type(.{ .optional = .{ .child = usize } });
                for (field_names, 0..) |field_name, fi| {
                    optional_fields_[fi] = .{
                        .name = field_name,
                        .type = optional_usize,
                        .default_value_ptr = &usize_null,
                        .is_comptime = false,
                        .alignment = @alignOf(optional_usize),
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
        pub fn initContiguous(shape: Axes) @This() {
            const rank = field_names.len;
            const field_names_rev = comptime rev: {
                var field_names_rev_: [rank][:0]const u8 = undefined;
                mem.copyForwards([:0]const u8, &field_names_rev_, field_names);
                mem.reverse([:0]const u8, &field_names_rev_);
                break :rev field_names_rev_;
            };

            var strides: Axes = undefined;
            var next_stride: usize = 1;
            inline for (field_names_rev) |field_name| {
                @field(strides, field_name) = next_stride;
                next_stride *= @field(shape, field_name);
            }
            return .{
                .shape = shape,
                .strides = strides,
                .offset = 0,
            };
            // Alternative implementation via indexing
            // inline for (0..rank) |fi| {
            //     const field_name = field_names[rank - 1 - fi];
            //     @field(strides, field_name) = next_stride;
            //     next_stride *= @field(shape, field_name);
            // }
        }

        pub fn iterKeys(self: *const @This()) KeyIterator(Axes) {
            return KeyIterator(Axes).init(self.shape, self.strides);
        }

        /// Return the offset into the linear buffer for the given structured index.
        /// If the index is out of bounds, return null.
        pub fn linearChecked(self: *const @This(), index: Axes) ?usize {
            if (!self.withinBounds(index))
                return null;
            return linear(self, index);
        }

        pub fn linear(self: *const @This(), index: Axes) usize {
            assert(self.withinBounds(index));
            var sum = self.offset;
            inline for (field_names) |field_name| {
                sum += @field(self.strides, field_name) * @field(index, field_name);
            }
            return sum;
        }

        pub fn withinBounds(self: *const @This(), index: Axes) bool {
            inline for (field_names) |field_name| {
                if (@field(self.shape, field_name) <= @field(index, field_name))
                    return false;
            }
            return true;
        }

        /// Stride a single axis by a given step size.
        /// Equivalent to `::step` syntax in python.
        /// Panics if `step` is zero.
        pub fn strideAxis(self: *const @This(), comptime axis: Axis, step: usize) @This() {
            var new_strides = self.strides;
            var new_shape = self.shape;
            const axis_name = field_names[@intFromEnum(axis)];
            const new_dim = &@field(new_shape, axis_name);
            const new_stride = &@field(new_strides, axis_name);

            strideInplace(step, new_dim, new_stride);
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Stride multiple axes by given step sizes.
        /// axes whose value is null in `steps` are skipped.
        pub fn stride(self: *const @This(), steps: AxesOptional) @This() {
            var new_strides = self.strides;
            var new_shape = self.shape;
            inline for (field_names) |field_name| {
                if (@field(steps, field_name)) |step| {
                    const new_dim = &@field(new_shape, field_name);
                    const new_stride = &@field(new_strides, field_name);
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
            comptime axis: Axis,
            start: usize,
            end: usize) @This() {
        // zig fmt: on
            const axis_name = field_names[@intFromEnum(axis)];
            const old_size = @field(self.shape, axis_name);
            if (end > old_size)
                @panic("slice end out of bounds");
            if (start >= end)
                @panic("slice start must be less than end");
            var new_shape = self.shape;
            @field(new_shape, axis_name) = end - start;
            var offset_lookup = mem.zeroes(Axes);
            @field(offset_lookup, axis_name) = start;
            const new_offset = self.linear(offset_lookup);
            return .{
                .shape = new_shape,
                .strides = self.strides,
                .offset = new_offset,
            };
        }

        /// Remove an axis that has size 1.
        /// Panics if size != 1.
        // zig fmt: off
        pub fn squeezeAxis(self: *const @This(),
            comptime axis: Axis
        ) NamedIndex(Removed(Axis, field_names[@intFromEnum(axis)])) {
        // zig fmt: on
            const axis_name = field_names[@intFromEnum(axis)];
            if (@field(self.shape, axis_name) != 1)
                @panic("squeezeAxis: axis size must be 1");
            const NewEnum = Removed(Axis, axis_name);
            const NewKey = KeyStruct(meta.fieldNames(NewEnum));
            const new_shape: NewKey = blk: {
                var tmp: NewKey = undefined;
                inline for (field_names) |field_name| {
                    if (comptime !mem.eql(u8, field_name, axis_name)) {
                        @field(tmp, field_name) = @field(self.shape, field_name);
                    }
                }
                break :blk tmp;
            };
            const new_strides: NewKey = blk: {
                var tmp: NewKey = undefined;
                inline for (field_names) |field_name| {
                    if (comptime !mem.eql(u8, field_name, axis_name)) {
                        @field(tmp, field_name) = @field(self.strides, field_name);
                    }
                }
                break :blk tmp;
            };
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Returns a new NamedIndex with only the axes present in the given subset enum.
        /// All axes not present in the new enum must have size 1, and are squeezed out.
        /// Panics if any removed axis does not have size 1.
        pub fn keepOnly(self: *const @This(), comptime NewEnum: type) NamedIndex(NewEnum) {
            const old_field_names = field_names;
            const new_field_names = comptime meta.fieldNames(NewEnum);

            // Check that NewEnum is a subset of AxisEnum
            inline for (new_field_names) |new_name| {
                var found = false;
                inline for (old_field_names) |old_name| {
                    if (mem.eql(u8, new_name, old_name)) {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    @panic("keepOnly: NewEnum contains axis not present in original enum");
            }

            // For each axis in old_field_names not in new_field_names, check size == 1
            inline for (old_field_names) |old_name| {
                var keep = false;
                inline for (new_field_names) |new_name| {
                    if (mem.eql(u8, old_name, new_name)) {
                        keep = true;
                        break;
                    }
                }
                if (!keep) {
                    if (@field(self.shape, old_name) != 1)
                        @panic("keepOnly: cannot squeeze axis '" ++ old_name ++ "' with size != 1");
                }
            }

            // Build new shape and strides
            const NewKey = KeyStruct(new_field_names);
            var new_shape: NewKey = undefined;
            var new_strides: NewKey = undefined;
            inline for (new_field_names) |new_name| {
                @field(new_shape, new_name) = @field(self.shape, new_name);
                @field(new_strides, new_name) = @field(self.strides, new_name);
            }

            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Returns a new NamedIndex that conforms to a new axes enum by squeezing or unsqueezing
        /// axes.
        /// - Axes present in both: keep shape and stride.
        /// - Axes only in new enum: add with shape 1 and stride 0.
        /// - Axes only in old enum: require shape == 1 and squeeze out.
        /// Panics if any removed axis does not have size 1.
        pub fn conformAxes(self: *const @This(), comptime NewEnum: type) NamedIndex(NewEnum) {
            const old_field_names = field_names;
            const new_field_names = comptime meta.fieldNames(NewEnum);

            // For each axis in old_field_names not in new_field_names, check size == 1
            inline for (old_field_names) |old_name| {
                var keep = false;
                inline for (new_field_names) |new_name| {
                    if (comptime mem.eql(u8, old_name, new_name)) {
                        keep = true;
                        break;
                    }
                }
                if (!keep) {
                    if (@field(self.shape, old_name) != 1)
                        @panic("projectAxes: cannot squeeze axis '" ++ old_name ++ "' with size != 1");
                }
            }

            // Build new shape and strides
            const NewKey = KeyStruct(new_field_names);
            var new_shape: NewKey = undefined;
            var new_strides: NewKey = undefined;
            inline for (new_field_names) |new_name| {
                var found = false;
                inline for (old_field_names) |old_name| {
                    if (comptime mem.eql(u8, new_name, old_name)) {
                        @field(new_shape, new_name) = @field(self.shape, new_name);
                        @field(new_strides, new_name) = @field(self.strides, new_name);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // New axis: add with shape 1 and stride 0
                    @field(new_shape, new_name) = 1;
                    @field(new_strides, new_name) = 0;
                }
            }

            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Return the number of elements in this index.
        pub fn count(self: *const @This()) usize {
            var prod: usize = 1;
            inline for (field_names) |field_name| {
                prod *= @field(self.shape, field_name);
            }
            return prod;
        }

        /// Return the axes ordered descendingly by stride order.
        /// For instance, row-major order returns {.row, .col}.
        pub fn axisOrder(self: *const @This()) [field_names.len]Axis {
            const strides_arr: [field_names.len]usize = @bitCast(self.strides);
            const dims_desc = argsort: {
                var argsort: [field_names.len]usize = undefined;
                for (0..field_names.len) |i| {
                    argsort[i] = i;
                }
                const strides_slice: []const usize = strides_arr[0..];
                mem.sort(usize, argsort[0..], strides_slice, fnames_gt);
                break :argsort argsort;
            };
            var axis_arr: [field_names.len]Axis = undefined;
            inline for (0..field_names.len) |fi| {
                axis_arr[fi] = @enumFromInt(dims_desc[fi]);
            }
            return axis_arr;
        }

        /// A NamedIndex is contiguous if there is a one-to-one mapping between a key in the index and
        /// an element in the underlying buffer.
        pub fn isContiguous(self: *const @This()) bool {
            const axis_order = self.axisOrder();
            const strides_arr: [field_names.len]usize = @bitCast(self.strides);
            const shape_arr: [field_names.len]usize = @bitCast(self.shape);
            var expected_stride: usize = 1;
            inline for (0..field_names.len) |i| {
                const axis = axis_order[field_names.len - 1 - i];
                const stride_mismatch = strides_arr[@intFromEnum(axis)] != expected_stride;
                if (stride_mismatch)
                    return false;
                expected_stride *= shape_arr[@intFromEnum(axis)];
            }
            return true;
        }

        /// Adds an axis of size 1 to the index.
        // zig fmt: off
        pub fn addEmptyAxis(self: *const @This(),
            comptime axis: [:0]const u8
        ) NamedIndex(Added(Axis, axis)) {
        // zig fmt: on
            const NewEnum = Added(Axis, axis);
            const NewKey = KeyStruct(meta.fieldNames(NewEnum));
            const new_shape: NewKey = blk: {
                var tmp: NewKey = undefined;
                inline for (field_names) |field_name| {
                    @field(tmp, field_name) = @field(self.shape, field_name);
                }
                @field(tmp, axis) = 1;
                break :blk tmp;
            };
            const new_strides: NewKey = blk: {
                var tmp: NewKey = undefined;
                inline for (field_names) |field_name| {
                    @field(tmp, field_name) = @field(self.strides, field_name);
                }
                // The stride for the new axis is arbitrary, but 0 is standard for broadcasting semantics.
                @field(tmp, axis) = 0;
                break :blk tmp;
            };
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Helper for strict axis renaming.
        /// If any axis in NewEnum cannot be mapped, this will fail to compile.
        pub fn rename(self: *const @This(), comptime NewEnum: type, comptime rename_pairs: []const AxisRenamePair) NamedIndex(NewEnum) {
            const OldEnum = Axis;
            const old_names = @typeInfo(OldEnum).@"enum".fields;
            const new_names = @typeInfo(NewEnum).@"enum".fields;

            // Compile-time checks
            comptime {
                // number of axes must match
                if (old_names.len != new_names.len) {
                    @compileError("renameAxes: Number of axes in source and target enums must match.");
                }
                // Check for duplicate old axis names in rename_pairs
                for (old_names) |old_field| {
                    var count_: usize = 0;
                    for (rename_pairs) |pair| {
                        if (std.mem.eql(u8, pair.old, old_field.name)) {
                            count_ += 1;
                        }
                    }
                    if (count_ > 1) {
                        @compileError("renameAxes: Old axis '" ++ old_field.name ++ "' is mapped to multiple new axes.");
                    }
                }
            }

            // Build mapping from new_name to old_name
            const map: [new_names.len][:0]const u8 = comptime map: {
                var map: [new_names.len][:0]const u8 = undefined;
                for (new_names, 0..) |new_field, ni| {
                    var found = false;
                    // Check if new_name matches any old_name directly
                    for (old_names) |old_field| {
                        if (std.mem.eql(u8, old_field.name, new_field.name)) {
                            map[ni] = old_field.name;
                            found = true;
                            break;
                        }
                    }
                    // If not, check rename_pairs
                    if (!found) {
                        for (rename_pairs) |pair| {
                            if (std.mem.eql(u8, pair.new, new_field.name)) {
                                // Find old_name in old_names
                                for (old_names) |old_field| {
                                    if (std.mem.eql(u8, old_field.name, pair.old)) {
                                        map[ni] = old_field.name;
                                        found = true;
                                        break;
                                    }
                                }
                                if (found) break;
                            }
                        }
                    }
                    if (!found) {
                        @compileError("renameAxes: Could not map new axis '" ++ new_field.name ++ "' to any old axis.");
                    }
                }
                break :map map;
            };

            // Build new shape and strides
            const NewKey = NamedIndex(NewEnum).Axes;
            var new_shape: NewKey = undefined;
            var new_strides: NewKey = undefined;
            inline for (new_names, 0..) |new_field, ni| {
                const old_name = map[ni];
                @field(new_shape, new_field.name) = @field(self.shape, old_name);
                @field(new_strides, new_field.name) = @field(self.strides, old_name);
            }
            return NamedIndex(NewEnum){
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// Broadcasts an existing axis of size 1 to a new size by setting its stride to 0.
        /// Panics if size is not currently 1.
        ///
        /// To add a new axis instead, see `addEmptyAxis`.
        pub fn broadcastAxis(self: *const @This(), comptime axis: Axis, new_size: usize) @This() {
            const axis_name = field_names[@intFromEnum(axis)];
            if (@field(self.shape, axis_name) != 1)
                @panic("broadcastAxis: axis size must be 1");
            var new_shape = self.shape;
            @field(new_shape, axis_name) = new_size;
            var new_strides = self.strides;
            @field(new_strides, axis_name) = 0;
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        /// DeprecatedRename an axis.
        // zig fmt: off
        pub fn rename_old(self: *const @This(),
            comptime axis: Axis,
            comptime new_name: [:0]const u8
        ) NamedIndex(Renamed(Axis, field_names[@intFromEnum(axis)], new_name)) {
        // zig fmt: on
            const old_name = field_names[@intFromEnum(axis)];
            const NewEnum = Renamed(Axis, old_name, new_name);
            const NewKey = KeyStruct(meta.fieldNames(NewEnum));
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

/// Struct for axis renaming pairs
pub const AxisRenamePair = struct { old: []const u8, new: []const u8 };

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
                mem.sort(usize, argsort[0..], strides_slice, fnames_gt);
                break :argsort argsort;
            };
            return .{ .next_arr = start, .shape_arr = shape_arr, .dims_desc = dims_desc };
        }

        pub fn next(self: *@This()) ?Key {
            if (self.dims_desc.len == 0)
                return null;
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
    };
}

fn fnames_gt(strides_arr: []const usize, lhs: usize, rhs: usize) bool {
    // descending order
    return strides_arr[lhs] > strides_arr[rhs];
}

/// Reify a key struct with given field names.
pub fn KeyStruct(comptime names: []const [:0]const u8) type {
    const rank = names.len;
    const fields = fields: {
        var fields_: [rank]Type.StructField = undefined;
        for (0..rank) |i| {
            fields_[i] = .{
                .name = names[i],
                .type = usize,
                .default_value_ptr = null,
                .is_comptime = false,
                .alignment = 0,
            };
        }
        break :fields fields_;
    };
    const new_struct: Type.Struct = .{
        .layout = .@"packed",
        .backing_integer = null,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    };
    return @Type(Type{ .@"struct" = new_struct });
}

/// Reify an enum with given field names.
pub fn KeyEnum(comptime names: []const [:0]const u8) type {
    // Create an enum type with the given field names.
    const rank = names.len;
    const fields = fields: {
        var fields_: [rank]Type.EnumField = undefined;
        for (0..rank) |i| {
            fields_[i] = .{
                .name = names[i],
                .value = i,
            };
        }
        break :fields fields_;
    };
    const bits = switch (rank) {
        0 => 0,
        else => std.math.log2_int_ceil(usize, rank),
    };
    const TagType = @Type(.{ .int = .{ .bits = bits, .signedness = .unsigned } });
    const enum_type: Type.Enum = .{
        .tag_type = TagType,
        .fields = &fields,
        .decls = &.{},
        .is_exhaustive = true,
    };
    return @Type(Type{ .@"enum" = enum_type });
}

/// Return a copy of a given enum with a given field renamed.
fn Renamed(comptime OldKey: type, old_name: [:0]const u8, new_name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"enum";
    const new_field_names = comptime fields: {
        var new_field_names: [old_struct.fields.len][:0]const u8 = undefined;
        var matched = false;
        for (0..old_struct.fields.len) |fi| {
            const old_field = old_struct.fields[fi];
            if (mem.eql(u8, old_field.name, old_name)) {
                new_field_names[fi] = new_name;
                matched = true;
            } else {
                new_field_names[fi] = old_field.name;
            }
        }
        if (!matched)
            @compileError("Field not found in enum: " ++ old_name);
        break :fields new_field_names;
    };
    return KeyEnum(&new_field_names);
}

/// Return a copy of a given struct with a given field removed
fn Removed(comptime OldKey: type, comptime name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"enum";
    const old_rank = old_struct.fields.len;
    const new_rank = old_rank - 1;
    const new_field_names = comptime fields: {
        var matches: usize = 0;
        // Check if name is actually a field name
        for (old_struct.fields) |field| {
            if (!mem.eql(u8, field.name, name)) {
                matches += 1;
            }
        }
        if (matches != new_rank)
            @compileError("Field not found in enum: " ++ name);
        var new_field_names: [matches][:0]const u8 = undefined;
        var idx: usize = 0;
        for (old_struct.fields) |field| {
            if (!mem.eql(u8, field.name, name)) {
                new_field_names[idx] = field.name;
                idx += 1;
            }
        }
        break :fields new_field_names;
    };
    return KeyEnum(&new_field_names);
}

/// Return a copy of a given enum with a given field added
fn Added(comptime OldKey: type, comptime name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"enum";
    const old_rank = old_struct.fields.len;
    // Check that the field does not already exist
    inline for (old_struct.fields) |field| {
        if (mem.eql(u8, field.name, name)) {
            @compileError("Field already exists in enum: " ++ name);
        }
    }
    const new_rank = old_rank + 1;
    const new_field_names = comptime fields: {
        var new_field_names: [new_rank][:0]const u8 = undefined;
        for (0..old_rank) |i| {
            new_field_names[i] = old_struct.fields[i].name;
        }
        new_field_names[old_rank] = name;
        break :fields new_field_names;
    };
    return KeyEnum(&new_field_names);
}

/// Return a new enum that contains the fields that occur in one of two given enums, but not the
/// other.
///
/// ## Example
/// ```zig
/// AC = Xor(enum {a, b}, enum {b, c});
/// ```
/// `AC` will be equivalent to `enum {a, c}`;
pub fn Xor(comptime Enum1: type, comptime Enum2: type) type {
    const info1 = @typeInfo(Enum1).@"enum";
    const info2 = @typeInfo(Enum2).@"enum";
    var common1 = mem.zeroes([info1.fields.len]bool);
    var common2 = mem.zeroes([info2.fields.len]bool);
    comptime var num_matches: usize = 0;

    inline for (0..info1.fields.len) |fi| {
        inline for (0..info2.fields.len) |fj| fj: {
            const match = mem.eql(u8, info1.fields[fi].name, info2.fields[fj].name);
            if (match) {
                common1[fi] = true;
                common2[fj] = true;
                num_matches += 1;
                break :fj;
            }
        }
    }

    const xor_len = info1.fields.len + info2.fields.len - 2 * num_matches;
    comptime var xor_fnames: [xor_len][:0]const u8 = undefined;
    var i: usize = 0;
    inline for (info1.fields, 0..) |field, fi| {
        if (!common1[fi]) {
            xor_fnames[i] = field.name;
            i += 1;
        }
    }
    inline for (info2.fields, 0..) |field, fj| {
        if (!common2[fj]) {
            xor_fnames[i] = field.name;
            i += 1;
        }
    }

    assert(i == xor_len);

    return KeyEnum(&xor_fnames);
}

// const FieldIntersection = struct {
//     lwr: [][:0]const u8,
//     common: [][:0]const u8,
//     rwl: [][:0]const u8,
// };

const Index2dEnum = enum { row, col };

test "init strides" {
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 4, .col = 5 },
        .strides = .{ .row = 5, .col = 1 },
    };
    const idx2 = Structure2d.initContiguous(.{ .row = 4, .col = 5 });
    try std.testing.expectEqual(idx, idx2);
}

test "linear index" {
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 4, .col = 5 },
        .strides = .{ .row = 5, .col = 1 },
        .offset = 7,
    };
    const query: Structure2d.Axes = .{ .row = 2, .col = 3 };
    const expected = 20;
    try std.testing.expectEqual(expected, idx.linear(query));
}

test "linear invalid index" {
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 4, .col = 5 },
        .strides = .{ .row = 5, .col = 1 },
        .offset = 7,
    };
    const query: Structure2d.Axes = .{ .row = 4, .col = 3 };
    const expected = null;
    try std.testing.expectEqual(expected, idx.linearChecked(query));
}

test "linear size 1" {
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 1, .col = 1 },
        .strides = .{ .row = 18, .col = 2 },
        .offset = 17,
    };
    const query: Structure2d.Axes = .{ .row = 0, .col = 0 };
    const expected = 17;
    try std.testing.expectEqual(expected, idx.linear(query));
    try std.testing.expectEqual(expected, idx.linearChecked(query).?);

    // Out of bounds
    const query_oob: Structure2d.Axes = .{ .row = 1, .col = 0 };
    try std.testing.expectEqual(null, idx.linearChecked(query_oob));
}

test "linear overlapping" {
    // zero strides
    const Structure2d = NamedIndex(Index2dEnum);

    // Overlapping index: both axes have stride 0, shape > 1
    const idx: Structure2d = .{
        .shape = .{ .row = 3, .col = 2 },
        .strides = .{ .row = 0, .col = 0 },
        .offset = 42,
    };

    // All indices should map to the same linear offset
    inline for (0..idx.shape.row) |r| {
        inline for (0..idx.shape.col) |c| {
            const key: Structure2d.Axes = .{ .row = r, .col = c };
            try std.testing.expectEqual(42, idx.linear(key));
            try std.testing.expectEqual(42, idx.linearChecked(key).?);
        }
    }

    // Out of bounds should still return null
    const oob: Structure2d.Axes = .{ .row = 3, .col = 0 };
    try std.testing.expectEqual(null, idx.linearChecked(oob));

    // If only one axis has stride 0, only that axis is broadcasted
    const idx_row_broadcast: Structure2d = .{
        .shape = .{ .row = 3, .col = 4 },
        .strides = .{ .row = 0, .col = 2 },
        .offset = 10,
    };

    const key: Structure2d.Axes = .{ .row = 1, .col = 2 };
    try std.testing.expectEqual(14, idx_row_broadcast.linear(key));
    try std.testing.expectEqual(14, idx_row_broadcast.linearChecked(key).?);
    // Out of bounds for broadcasted axis
    const oob2: Structure2d.Axes = .{ .row = 3, .col = 0 };
    try std.testing.expectEqual(null, idx_row_broadcast.linearChecked(oob2));
}

test "strideAxis" {
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 6, .col = 8 },
        .strides = .{ .row = 8, .col = 1 },
        .offset = 1,
    };
    const stepped = idx.strideAxis(.col, 3);
    try std.testing.expectEqual(6, stepped.shape.row);
    try std.testing.expectEqual(3, stepped.shape.col);
    try std.testing.expectEqual(8, stepped.strides.row);
    try std.testing.expectEqual(3, stepped.strides.col);
    try std.testing.expectEqual(1, stepped.offset);
}

test "stride" {
    const Structure2d = NamedIndex(Index2dEnum);
    const KeyOptional = Structure2d.AxesOptional;

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
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 6, .col = 8 },
        .strides = .{ .row = 8, .col = 1 },
        .offset = 3,
    };
    const sliced = idx.sliceAxis(.row, 2, 5);
    try std.testing.expectEqual(3, sliced.shape.row);
    try std.testing.expectEqual(8, sliced.shape.col);
    try std.testing.expectEqual(8, sliced.strides.row);
    try std.testing.expectEqual(1, sliced.strides.col);
    try std.testing.expectEqual(19, sliced.offset);

    const sliced_again = sliced.sliceAxis(.row, 1, 3);
    try std.testing.expectEqual(2, sliced_again.shape.row);
    try std.testing.expectEqual(8, sliced_again.shape.col);
    try std.testing.expectEqual(8, sliced_again.strides.row);
    try std.testing.expectEqual(1, sliced_again.strides.col);
    try std.testing.expectEqual(27, sliced_again.offset);
}

test "iterKeys" {
    const Structure2d = NamedIndex(Index2dEnum);
    const idx: Structure2d = .{
        .shape = .{ .row = 2, .col = 3 },
        .strides = .{ .row = 3, .col = 1 },
        .offset = 0,
    };
    const expected_indices: [6]Structure2d.Axes = .{
        .{ .row = 0, .col = 0 },
        .{ .row = 0, .col = 1 },
        .{ .row = 0, .col = 2 },
        .{ .row = 1, .col = 0 },
        .{ .row = 1, .col = 1 },
        .{ .row = 1, .col = 2 },
    };
    var i: usize = 0;
    var it = idx.iterKeys();
    while (it.next()) |key| {
        try std.testing.expectEqual(expected_indices[i], key);
        i += 1;
    }
    try std.testing.expectEqual(i, expected_indices.len);
}

test "iterKeys 3d, major in middle" {
    const Index3d = enum { x, y, z };
    const Structure3d = NamedIndex(Index3d);
    // Make y the major dimension (stride 1 for y, then z, then x)
    const idx: Structure3d = .{
        .shape = .{ .x = 2, .y = 3, .z = 4 },
        .strides = .{ .x = 12, .y = 1, .z = 3 },
        .offset = 0,
    };
    // The iteration order should be: y varies fastest, then z, then x
    const expected_indices: [24]Structure3d.Axes = .{
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

test "axisOrder" {
    const Structure2d = NamedIndex(Index2dEnum);

    // Row-major order: row stride > col stride
    const idx_row_major: Structure2d = .{
        .shape = .{ .row = 3, .col = 4 },
        .strides = .{ .row = 4, .col = 1 },
        .offset = 0,
    };
    const order_row_major = idx_row_major.axisOrder();
    try std.testing.expectEqual(.{ .row, .col }, order_row_major);

    // Col-major order: col stride > row stride
    const idx_col_major: Structure2d = .{
        .shape = .{ .row = 3, .col = 4 },
        .strides = .{ .row = 1, .col = 3 },
        .offset = 0,
    };
    const order_col_major = idx_col_major.axisOrder();
    try std.testing.expectEqual(.{ .col, .row }, order_col_major);

    // Strides equal: order should be row, col (original enum order)
    const idx_equal: Structure2d = .{
        .shape = .{ .row = 2, .col = 2 },
        .strides = .{ .row = 5, .col = 5 },
        .offset = 0,
    };
    const order_equal = idx_equal.axisOrder();
    try std.testing.expectEqual(.{ .row, .col }, order_equal);
}

test "isContiguous" {
    const Structure2d = NamedIndex(Index2dEnum);

    // Contiguous row-major
    const idx_row_major: Structure2d = .{
        .shape = .{ .row = 3, .col = 4 },
        .strides = .{ .row = 4, .col = 1 },
        .offset = 0,
    };
    try std.testing.expect(idx_row_major.isContiguous());

    // Contiguous col-major
    const idx_col_major: Structure2d = .{
        .shape = .{ .row = 3, .col = 4 },
        .strides = .{ .row = 1, .col = 3 },
        .offset = 0,
    };
    try std.testing.expect(idx_col_major.isContiguous());

    // Not contiguous: stride mismatch
    const idx_noncontig: Structure2d = .{
        .shape = .{ .row = 3, .col = 4 },
        .strides = .{ .row = 5, .col = 1 },
        .offset = 0,
    };
    try std.testing.expect(!idx_noncontig.isContiguous());

    // Test: isContiguous with 3 dimensions
    const Index3d = enum { x, y, z };
    const Structure3d = NamedIndex(Index3d);

    // Contiguous y-major: y, x, z
    const idx_y_major: Structure3d = .{
        .shape = .{ .x = 2, .y = 3, .z = 4 },
        .strides = .{ .x = 4, .y = 8, .z = 1 },
        .offset = 0,
    };
    try std.testing.expect(idx_y_major.isContiguous());

    // Not contiguous: stride mismatch
    const idx_noncontig_3d: Structure3d = .{
        .shape = .{ .x = 2, .y = 3, .z = 4 },
        .strides = .{ .x = 5, .y = 1, .z = 6 },
        .offset = 0,
    };
    try std.testing.expect(!idx_noncontig_3d.isContiguous());
}

test "count" {
    const Structure2d = NamedIndex(Index2dEnum);
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

// test "cblas" {
//     const order = acc.CBLAS_ORDER;
//     acc.cblas_ddot(__N: c_int, __X: [*c]const f64, __incX: c_int, __Y: [*c]const f64, __incY: c_int)
//     std.debug.print("{any}\n", .{order});
// }

test "rename" {
    const IJEnum = enum { i, j };
    const IndexIJ = NamedIndex(IJEnum);
    const idx: IndexIJ = .initContiguous(.{
        .i = 5,
        .j = 3,
    });
    const idx_from = idx.rename_old(.i, "from");

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
        _ = idx.rename_old("nonexisting", "foo");
    }
}

test "RemovedStructField" {
    const IJKEnum = enum { i, j, k };
    const IK = Removed(IJKEnum, "j");

    // Test field properties
    const ik_info = @typeInfo(IK).@"enum";

    try std.testing.expectEqual(2, ik_info.fields.len);
    try std.testing.expectEqualStrings("i", ik_info.fields[0].name);
    try std.testing.expectEqualStrings("k", ik_info.fields[1].name);

    try std.testing.expectEqual(u1, ik_info.tag_type);
    try std.testing.expectEqualSlices(Type.Declaration, &.{}, ik_info.decls);

    // Toggle to verify that it throws a compileError
    if (false) {
        Removed(IJKEnum, "nonexisting");
    }
}

test "squeezeAxis" {
    const IJKEnum = enum { i, j, k };
    const IndexIJK = NamedIndex(IJKEnum);

    // Squeeze the "j" axis (size 1)
    const idx: IndexIJK = .{
        .shape = .{ .i = 4, .j = 1, .k = 7 },
        .strides = .{ .i = 7, .j = 49, .k = 1 },
        .offset = 0,
    };
    const squeezed = idx.squeezeAxis(.j);

    // The resulting key type should have only "i" and "k"
    const SqueezedKey = Removed(IJKEnum, "j");
    const expected: NamedIndex(SqueezedKey) = .{
        .shape = .{ .i = 4, .k = 7 },
        .strides = .{ .i = 7, .k = 1 },
        .offset = 0,
    };
    try std.testing.expectEqual(expected, squeezed);

    // Toggle to verify that it panics if size is not 1.
    if (false) {
        _ = idx.squeezeAxis(.i);
    }
}

test "keepOnly" {
    const IJKEnum = enum { i, j, k };
    const IJEnum = enum { i, j };
    const IndexIJK = NamedIndex(IJKEnum);

    // keepOnly: keep only "i" and "j" axes
    const idx: IndexIJK = .{
        .shape = .{ .i = 4, .j = 1, .k = 1 },
        .strides = .{ .i = 1, .j = 4, .k = 4 },
        .offset = 0,
    };
    const idx_kept = idx.keepOnly(IJEnum);

    // The resulting key type should have only "i" and "j"
    const expected: NamedIndex(IJEnum) = .{
        .shape = .{ .i = 4, .j = 1 },
        .strides = .{ .i = 1, .j = 4 },
        .offset = 0,
    };
    try std.testing.expectEqual(expected, idx_kept);

    // Should panic if any removed axis does not have size 1
    const idx_bad: IndexIJK = .{
        .shape = .{ .i = 4, .j = 1, .k = 2 },
        .strides = .{ .i = 2, .j = 2, .k = 1 },
        .offset = 0,
    };
    if (false) {
        _ = idx_bad.keepOnly(IJEnum);
    }

    // Should panic if NewEnum contains axis not present in original enum
    const ILEnum = enum { i, l };
    if (false) {
        _ = idx.keepOnly(ILEnum);
    }
}

test "conformAxes" {
    const IJKEnum = enum { i, j, k };
    const IKLEnum = enum { i, k, l };
    const IndexIJK = NamedIndex(IJKEnum);

    // Project to {i, k, l}: keep i and k, add l (size 1, stride 0), squeeze out j (must be size 1)
    const idx: IndexIJK = .{
        .shape = .{ .i = 4, .j = 1, .k = 7 },
        .strides = .{ .i = 7, .j = 28, .k = 1 },
        .offset = 0,
    };
    const idx_proj = idx.conformAxes(IKLEnum);
    const expected: NamedIndex(IKLEnum) = .{
        .shape = .{ .i = 4, .k = 7, .l = 1 },
        .strides = .{ .i = 7, .k = 1, .l = 0 },
        .offset = 0,
    };
    try std.testing.expectEqual(expected, idx_proj);

    // Should panic if removed axis does not have size 1
    const idx_bad: IndexIJK = .{
        .shape = .{ .i = 4, .j = 2, .k = 7 },
        .strides = .{ .i = 7, .j = 28, .k = 1 },
        .offset = 0,
    };
    if (false) {
        _ = idx_bad.conformAxes(IKLEnum);
    }

    // Should allow projecting to a superset (adding multiple axes)
    const IJKLMEnum = enum { i, j, k, l, m };
    const idx_proj2 = idx.conformAxes(IJKLMEnum);
    const expected2: NamedIndex(IJKLMEnum) = .{
        .shape = .{ .i = 4, .j = 1, .k = 7, .l = 1, .m = 1 },
        .strides = .{ .i = 7, .j = 28, .k = 1, .l = 0, .m = 0 },
        .offset = 0,
    };
    try std.testing.expectEqual(expected2, idx_proj2);
}

test "addEmptyAxis" {
    const IJEnum = KeyEnum(&.{ "i", "j" });
    const IndexIJ = NamedIndex(IJEnum);

    // Add an empty axis "k"
    const idx: IndexIJ = .{
        .shape = .{ .i = 4, .j = 7 },
        .strides = .{ .i = 7, .j = 1 },
        .offset = 0,
    };
    const idx_added = idx.addEmptyAxis("k");

    // The resulting key type should have "i", "j", "k"
    // const ExpectedIndex = NamedIndex(IJKEnum);

    // Check that the shape and strides are as expected
    // const IJKEnum = AddedStructField(IJEnum, "k");
    const IJKEnum = KeyEnum(&.{ "i", "j", "k" });
    const expected: NamedIndex(IJKEnum) = .{
        .shape = .{ .i = 4, .j = 7, .k = 1 },
        .strides = .{ .i = 7, .j = 1, .k = 0 },
        .offset = 0,
    };
    try std.testing.expectEqual(expected, idx_added);

    // Toggle to verify that it throws a compileError if axis already exists
    if (false) {
        _ = idx.addEmptyAxis("i");
    }
}

test "broadcastAxis" {
    const IJEnum = enum { i, j };
    const IndexIJ = NamedIndex(IJEnum);

    // Broadcast the "j" axis from size 1 to size 5
    const idx: IndexIJ = .{
        .shape = .{ .i = 4, .j = 1 },
        .strides = .{ .i = 1, .j = 4 },
        .offset = 0,
    };
    const broadcasted = idx.broadcastAxis(.j, 5);

    const expected: IndexIJ = .{
        .shape = .{ .i = 4, .j = 5 },
        .strides = .{ .i = 1, .j = 0 },
        .offset = 0,
    };
    try std.testing.expectEqual(expected, broadcasted);

    // Should panic if axis is not size 1
    if (false) {
        _ = idx.broadcastAxis(.i, 5);
    }
}

test "Xor" {
    // Typical case: some overlapping, some not.
    const ABC = enum { a, b, c };
    const CD = enum { c, d };
    const ABD = enum { a, b, d };
    const Actual = Xor(ABC, CD);

    try std.testing.expectEqual(@typeInfo(ABD), @typeInfo(Actual));

    // Subset case
    const DF = enum { d, f };
    const ABCDF = enum { a, b, c, d, f };

    const info_actual = @typeInfo(Xor(ABC, DF));
    const info_expected = @typeInfo(ABCDF);
    try std.testing.expect(meta.eql(info_expected, info_actual));

    // Disjoint case
    const BC = enum { b, c };
    const A = enum { a };
    try std.testing.expect(meta.eql(@typeInfo(A), @typeInfo(Xor(ABC, BC))));

    try std.testing.expect(meta.eql(@typeInfo(enum {}), @typeInfo(Xor(ABC, ABC))));

    // Empty case
    const EmptyEnum1 = enum {};
    const EmptyEnum2 = enum {};
    try std.testing.expect(meta.eql(@typeInfo(enum {}), @typeInfo(Xor(EmptyEnum1, EmptyEnum2))));
}

test "KeyEnum" {
    const IJ1 = enum { i, j };
    const IJ2 = KeyEnum(&.{ "i", "j" });

    const info1 = @typeInfo(IJ1).@"enum";
    const info2 = @typeInfo(IJ2).@"enum";

    try std.testing.expectEqual(info1, info2);
}
