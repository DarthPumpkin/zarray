const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const assert = std.debug.assert;
const Type = std.builtin.Type;

pub const NamedIndexError = error{
    ShapeMismatch,
    StrideMisalignment,
};

// Generic struct factory for axis collections of arbitrary scalar type.
pub fn AxesStructOf(comptime names: []const [:0]const u8, comptime T: type) type {
    const rank = names.len;
    const fields = comptime blk: {
        var fields_: [rank]Type.StructField = undefined;
        for (0..rank) |i| {
            fields_[i] = .{
                .name = names[i],
                .type = T,
                .default_value_ptr = null,
                .is_comptime = false,
                .alignment = 0,
            };
        }
        break :blk fields_;
    };
    const s: Type.Struct = .{
        .layout = .@"packed",
        .backing_integer = null,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    };
    return @Type(Type{ .@"struct" = s });
}

// Optional variant
pub fn AxesOptionalStructOf(comptime names: []const [:0]const u8, comptime T: type) type {
    const optT = ?T;
    const default_val: optT = null;
    const rank = names.len;
    const fields = comptime blk: {
        var fields_: [rank]Type.StructField = undefined;
        for (0..rank) |i| {
            fields_[i] = .{
                .name = names[i],
                .type = optT,
                .default_value_ptr = &default_val,
                .is_comptime = false,
                .alignment = @alignOf(optT),
            };
        }
        break :blk fields_;
    };
    const s: Type.Struct = .{
        .layout = .auto,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    };
    return @Type(Type{ .@"struct" = s });
}

pub fn AxesStruct(comptime names: []const [:0]const u8) type {
    return AxesStructOf(names, usize);
}
pub fn AxesOptionalStruct(comptime names: []const [:0]const u8) type {
    return AxesOptionalStructOf(names, usize);
}

pub fn NamedIndex(comptime AxisEnum: type) type {
    _ = @typeInfo(AxisEnum).@"enum";
    const field_names = meta.fieldNames(AxisEnum);
    return struct {
        // Shapes remain unsigned
        shape: Axes,
        // Strides now signed to allow negative traversal
        strides: Strides,
        offset: usize = 0,

        pub const Axis = AxisEnum;
        pub const Axes = AxesStruct(field_names);
        pub const Strides = AxesStructOf(field_names, isize);
        pub const AxesOptional = AxesOptionalStruct(field_names); // shape-related optionals (?usize)
        pub const StepsOptional = AxesOptionalStructOf(field_names, isize); // stride step optionals (?isize)

        /// Create contiguous index (row-major: last axis fastest).
        pub fn initContiguous(shape: Axes) @This() {
            const rank = field_names.len;
            const field_names_rev = comptime rev: {
                var tmp: [rank][:0]const u8 = undefined;
                mem.copyForwards([:0]const u8, &tmp, field_names);
                mem.reverse([:0]const u8, &tmp);
                break :rev tmp;
            };

            var strides: Strides = undefined;
            var next_stride: isize = 1;
            inline for (field_names_rev) |fname| {
                @field(strides, fname) = next_stride;
                next_stride *= @intCast(@field(shape, fname));
            }
            return .{ .shape = shape, .strides = strides, .offset = 0 };
        }

        pub fn iterKeys(self: *const @This()) KeyIterator(Axes, Strides) {
            return KeyIterator(Axes, Strides).init(self.shape, self.strides);
        }

        pub fn linearChecked(self: *const @This(), index: Axes) ?usize {
            if (!self.withinBounds(index)) return null;
            return self.linear(index);
        }

        pub fn linear(self: *const @This(), index: Axes) usize {
            assert(self.withinBounds(index));
            var sum: isize = @intCast(self.offset);
            inline for (field_names) |fname| {
                const stride_: isize = @field(self.strides, fname);
                const idx: isize = @intCast(@field(index, fname));
                sum += stride_ * idx;
            }
            if (sum < 0) @panic("linear: negative buffer address (offset/stride mismatch)");
            return @intCast(sum);
        }

        pub fn withinBounds(self: *const @This(), index: Axes) bool {
            inline for (field_names) |fname| {
                if (@field(self.shape, fname) <= @field(index, fname)) return false;
            }
            return true;
        }

        /// Stride a single axis (allow negative step for reversal).
        pub fn strideAxis(self: *const @This(), comptime axis: Axis, step: isize) @This() {
            if (step == 0) @panic("strideAxis: step must be non-zero");
            var new_shape = self.shape;
            var new_strides = self.strides;
            var new_offset: isize = @intCast(self.offset);

            const axis_name = field_names[@intFromEnum(axis)];
            const dim_ptr = &@field(new_shape, axis_name);
            const stride_ptr = &@field(new_strides, axis_name);

            const orig_dim = dim_ptr.*;
            const orig_stride = stride_ptr.*;

            const abs_step: usize = @intCast(@abs(step));
            // Dimension after striding (ceil division)
            dim_ptr.* = if (orig_dim == 0) 0 else (orig_dim + abs_step - 1) / abs_step;

            // New stride
            stride_ptr.* = orig_stride * step;

            // Offset adjustment for negative traversal
            if (step < 0 and orig_dim > 0) {
                const last_index = orig_dim - 1 - ((orig_dim - 1) % abs_step);
                new_offset += @as(isize, @intCast(last_index)) * orig_stride;
            }

            if (new_offset < 0) @panic("strideAxis: negative offset after reversal");

            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = @intCast(new_offset),
            };
        }

        /// Stride multiple axes (steps may be positive or negative).
        pub fn stride(self: *const @This(), steps: StepsOptional) @This() {
            var new_shape = self.shape;
            var new_strides = self.strides;
            var new_offset: isize = @intCast(self.offset);

            inline for (field_names) |fname| {
                if (@field(steps, fname)) |step| {
                    if (step == 0) @panic("stride: step must be non-zero");
                    const dim_ptr = &@field(new_shape, fname);
                    const stride_ptr = &@field(new_strides, fname);

                    const orig_dim = dim_ptr.*;
                    const orig_stride = stride_ptr.*;

                    const abs_step: usize = @intCast(@abs(step));
                    dim_ptr.* = if (orig_dim == 0) 0 else (orig_dim + abs_step - 1) / abs_step;
                    stride_ptr.* = orig_stride * step;

                    if (step < 0 and orig_dim > 0) {
                        const last_index = orig_dim - 1 - ((orig_dim - 1) % abs_step);
                        new_offset += @as(isize, @intCast(last_index)) * orig_stride;
                    }
                }
            }
            if (new_offset < 0) @panic("stride: negative offset after reversal");
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = @intCast(new_offset),
            };
        }

        /// (Internal) Old helper kept for compatibility; now returns offset delta.
        fn strideInplace(step: isize, out_dim: *usize, out_stride: *isize) isize {
            if (step == 0) @panic("step must be non-zero");
            const orig_dim = out_dim.*;
            const orig_stride = out_stride.*;

            const abs_step: usize = @intCast(@abs(step));
            out_dim.* = if (orig_dim == 0) 0 else (orig_dim + abs_step - 1) / abs_step;
            out_stride.* = orig_stride * step;

            if (step < 0 and orig_dim > 0) {
                const last_index = orig_dim - 1 - ((orig_dim - 1) % abs_step);
                return @as(isize, @intCast(last_index)) * orig_stride;
            }
            return 0;
        }

        /// Slice axis (start:end). Does not itself reverse; combine with strideAxis(step = -1).
        pub fn sliceAxis(self: *const @This(), comptime axis: Axis, start: usize, end: usize) @This() {
            const axis_name = field_names[@intFromEnum(axis)];
            const old_size = @field(self.shape, axis_name);
            if (end > old_size) @panic("sliceAxis: end out of bounds");
            if (start >= end) @panic("sliceAxis: start must be < end");

            var new_shape = self.shape;
            @field(new_shape, axis_name) = end - start;

            var offset_lookup = mem.zeroes(Axes);
            @field(offset_lookup, axis_name) = start;
            // linear uses signed strides correctly
            const new_offset = self.linear(offset_lookup);

            return .{
                .shape = new_shape,
                .strides = self.strides,
                .offset = new_offset,
            };
        }

        pub fn squeezeAxis(self: *const @This(), comptime axis: Axis) NamedIndex(Removed(Axis, field_names[@intFromEnum(axis)])) {
            const axis_name = field_names[@intFromEnum(axis)];
            if (@field(self.shape, axis_name) != 1) @panic("squeezeAxis: axis size must be 1");
            const NewEnum = Removed(Axis, axis_name);
            const NewKey = AxesStruct(meta.fieldNames(NewEnum));
            const NewStrides = AxesStructOf(meta.fieldNames(NewEnum), isize);

            const new_shape: NewKey = blk: {
                var tmp: NewKey = undefined;
                inline for (field_names) |fname| if (comptime !mem.eql(u8, fname, axis_name)) {
                    @field(tmp, fname) = @field(self.shape, fname);
                };
                break :blk tmp;
            };
            const new_strides: NewStrides = blk: {
                var tmp: NewStrides = undefined;
                inline for (field_names) |fname| if (comptime !mem.eql(u8, fname, axis_name)) {
                    @field(tmp, fname) = @field(self.strides, fname);
                };
                break :blk tmp;
            };
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        pub fn keepOnly(self: *const @This(), comptime NewEnum: type) NamedIndex(NewEnum) {
            const old_field_names = field_names;
            const new_field_names = comptime meta.fieldNames(NewEnum);

            inline for (new_field_names) |new_name| {
                var found = false;
                inline for (old_field_names) |old_name| {
                    if (mem.eql(u8, new_name, old_name)) {
                        found = true;
                        break;
                    }
                }
                if (!found) @panic("keepOnly: axis not present in original: " ++ new_name);
            }

            inline for (old_field_names) |old_name| {
                var keep = false;
                inline for (new_field_names) |new_name| {
                    if (mem.eql(u8, old_name, new_name)) {
                        keep = true;
                        break;
                    }
                }
                if (!keep and @field(self.shape, old_name) != 1)
                    @panic("keepOnly: cannot squeeze axis '" ++ old_name ++ "' size != 1");
            }

            const NewShape = AxesStruct(new_field_names);
            const NewStrides = AxesStructOf(new_field_names, isize);
            var new_shape: NewShape = undefined;
            var new_strides: NewStrides = undefined;
            inline for (new_field_names) |nm| {
                @field(new_shape, nm) = @field(self.shape, nm);
                @field(new_strides, nm) = @field(self.strides, nm);
            }
            return .{
                .shape = new_shape,
                .strides = new_strides,
                .offset = self.offset,
            };
        }

        pub fn conformAxes(self: *const @This(), comptime NewEnum: type) NamedIndex(NewEnum) {
            const old_field_names = field_names;
            const new_field_names = comptime meta.fieldNames(NewEnum);

            inline for (old_field_names) |old_name| {
                var keep = false;
                inline for (new_field_names) |new_name| {
                    if (comptime mem.eql(u8, old_name, new_name)) {
                        keep = true;
                        break;
                    }
                }
                if (!keep and @field(self.shape, old_name) != 1)
                    @panic("conformAxes: cannot squeeze axis '" ++ old_name ++ "' size != 1");
            }

            const NewShape = AxesStruct(new_field_names);
            const NewStrides = AxesStructOf(new_field_names, isize);
            var new_shape: NewShape = undefined;
            var new_strides: NewStrides = undefined;

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

        pub fn count(self: *const @This()) usize {
            var prod: usize = 1;
            inline for (field_names) |fname| prod *= @field(self.shape, fname);
            return prod;
        }

        pub fn axisOrder(self: *const @This()) [field_names.len]Axis {
            const strides_arr: [field_names.len]isize = @bitCast(self.strides);
            // argsort by descending absolute stride
            var idxs: [field_names.len]usize = undefined;
            inline for (0..field_names.len) |i| idxs[i] = i;
            const strides_slice: []const isize = strides_arr[0..];
            mem.sort(usize, idxs[0..], strides_slice, fnames_gt_signed);
            var axes: [field_names.len]Axis = undefined;
            inline for (0..field_names.len) |i| axes[i] = @enumFromInt(idxs[i]);
            return axes;
        }

        pub fn isContiguous(self: *const @This()) bool {
            const order = self.axisOrder();
            const strides_arr: [field_names.len]isize = @bitCast(self.strides);
            const shape_arr: [field_names.len]usize = @bitCast(self.shape);
            var expected: usize = 1;
            inline for (0..field_names.len) |i| {
                const axis = order[field_names.len - 1 - i];
                const stride_ = strides_arr[@intFromEnum(axis)];
                if (@abs(stride_) != expected) return false;
                expected *= shape_arr[@intFromEnum(axis)];
            }
            return true;
        }

        pub fn addEmptyAxis(self: *const @This(), comptime axis: [:0]const u8) NamedIndex(Added(Axis, axis)) {
            const NewEnum = Added(Axis, axis);
            const NewShape = AxesStruct(meta.fieldNames(NewEnum));
            const NewStrides = AxesStructOf(meta.fieldNames(NewEnum), isize);

            const new_shape: NewShape = blk: {
                var tmp: NewShape = undefined;
                inline for (field_names) |fname| @field(tmp, fname) = @field(self.shape, fname);
                @field(tmp, axis) = 1;
                break :blk tmp;
            };
            const new_strides: NewStrides = blk: {
                var tmp: NewStrides = undefined;
                inline for (field_names) |fname| @field(tmp, fname) = @field(self.strides, fname);
                @field(tmp, axis) = 0;
                break :blk tmp;
            };
            return .{ .shape = new_shape, .strides = new_strides, .offset = self.offset };
        }

        pub fn rename(self: *const @This(), comptime NewEnum: type, comptime rename_pairs: []const AxisRenamePair) NamedIndex(NewEnum) {
            const OldEnum = Axis;
            const old_names = @typeInfo(OldEnum).@"enum".fields;
            const new_names = @typeInfo(NewEnum).@"enum".fields;

            comptime {
                if (old_names.len != new_names.len)
                    @compileError("rename: axis count mismatch");
                // duplicate source axis detection
                for (old_names) |old_field| {
                    var count_: usize = 0;
                    for (rename_pairs) |pair| {
                        if (std.mem.eql(u8, pair.old, old_field.name)) count_ += 1;
                    }
                    if (count_ > 1)
                        @compileError("rename: axis '" ++ old_field.name ++ "' mapped multiple times");
                }
            }

            const map: [new_names.len][:0]const u8 = comptime blk: {
                var m: [new_names.len][:0]const u8 = undefined;
                for (new_names, 0..) |new_field, ni| {
                    var found = false;
                    for (old_names) |old_field| {
                        if (std.mem.eql(u8, old_field.name, new_field.name)) {
                            m[ni] = old_field.name;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        for (rename_pairs) |pair| {
                            if (std.mem.eql(u8, pair.new, new_field.name)) {
                                for (old_names) |old_field| {
                                    if (std.mem.eql(u8, old_field.name, pair.old)) {
                                        m[ni] = old_field.name;
                                        found = true;
                                        break;
                                    }
                                }
                                if (found) break;
                            }
                        }
                    }
                    if (!found)
                        @compileError("rename: cannot map new axis '" ++ new_field.name ++ "'");
                }
                break :blk m;
            };

            const NewShape = AxesStruct(meta.fieldNames(NewEnum));
            const NewStrides = AxesStructOf(meta.fieldNames(NewEnum), isize);

            var new_shape: NewShape = undefined;
            var new_strides: NewStrides = undefined;
            inline for (new_names, 0..) |new_field, ni| {
                const old_name = map[ni];
                @field(new_shape, new_field.name) = @field(self.shape, old_name);
                @field(new_strides, new_field.name) = @field(self.strides, old_name);
            }
            return .{ .shape = new_shape, .strides = new_strides, .offset = self.offset };
        }

        pub fn broadcastAxis(self: *const @This(), comptime axis: Axis, new_size: usize) @This() {
            const axis_name = field_names[@intFromEnum(axis)];
            if (@field(self.shape, axis_name) != 1) @panic("broadcastAxis: axis must have size 1");
            var new_shape = self.shape;
            var new_strides = self.strides;
            @field(new_shape, axis_name) = new_size;
            @field(new_strides, axis_name) = 0;
            return .{ .shape = new_shape, .strides = new_strides, .offset = self.offset };
        }

        pub fn splitAxis(self: *const @This(), comptime TargetEnum: type, splitShapes: AxesOptionalStruct(meta.fieldNames(TargetEnum))) NamedIndex(TargetEnum) {
            const SourceEnum = Axis;
            const source_field_names = comptime meta.fieldNames(SourceEnum);
            const target_field_names = comptime meta.fieldNames(TargetEnum);

            var split_axis_name: ?[]const u8 = null;
            var split_axis_shape: usize = 0;
            var split_axis_stride: isize = 0;
            inline for (source_field_names) |src_name| {
                var found = false;
                inline for (target_field_names) |tgt_name| {
                    if (mem.eql(u8, src_name, tgt_name)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    split_axis_name = src_name;
                    split_axis_shape = @field(self.shape, src_name);
                    split_axis_stride = @field(self.strides, src_name);
                    break;
                }
            }
            if (split_axis_name == null) @panic("splitAxis: no axis to split");

            var new_axes: [target_field_names.len]struct { name: []const u8, shape: usize } = undefined;
            comptime var new_axes_count: usize = 0;
            inline for (target_field_names) |tgt_name| {
                comptime var present = false;
                inline for (source_field_names) |src_name| {
                    if (comptime mem.eql(u8, tgt_name, src_name)) {
                        present = true;
                        break;
                    }
                }
                if (!present) {
                    const shape_opt = @field(splitShapes, tgt_name);
                    if (shape_opt == null)
                        @panic("splitAxis: missing shape for axis '" ++ tgt_name ++ "'");
                    new_axes[new_axes_count] = .{ .name = tgt_name, .shape = shape_opt.? };
                    new_axes_count += 1;
                }
            }

            var prod: usize = 1;
            inline for (0..new_axes_count) |i| prod *= new_axes[i].shape;
            if (prod != split_axis_shape)
                @panic("splitAxis: product of new shapes mismatch");

            const NewShape = AxesStruct(target_field_names);
            const NewStrides = AxesStructOf(target_field_names, isize);
            var new_shape: NewShape = undefined;
            var new_strides: NewStrides = undefined;

            var stride_: isize = split_axis_stride;
            inline for (0..target_field_names.len) |tgt_idx| {
                const tgt_idx_rev = target_field_names.len - 1 - tgt_idx;
                const tgt_name = target_field_names[tgt_idx_rev];
                var found = false;
                inline for (source_field_names) |src_name| {
                    if (mem.eql(u8, tgt_name, src_name)) {
                        @field(new_shape, tgt_name) = @field(self.shape, src_name);
                        @field(new_strides, tgt_name) = @field(self.strides, src_name);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    inline for (0..new_axes_count) |i| {
                        if (mem.eql(u8, tgt_name, new_axes[i].name)) {
                            @field(new_shape, tgt_name) = new_axes[i].shape;
                            @field(new_strides, tgt_name) = stride_;
                            stride_ *= @intCast(new_axes[i].shape);
                        }
                    }
                }
            }
            return .{ .shape = new_shape, .strides = new_strides, .offset = self.offset };
        }

        pub fn mergeAxes(self: *const @This(), comptime TargetEnum: type) error{StrideMisalignment}!NamedIndex(TargetEnum) {
            const source_names = field_names;
            const target_names = comptime meta.fieldNames(TargetEnum);

            const MergeCompile = struct { new_axis_name: [:0]const u8, source_in_target: [source_names.len]bool };
            const compile_phase: MergeCompile = comptime blk: {
                var new_axis_count: usize = 0;
                var new_axis_name: [:0]const u8 = undefined;
                var source_in_target: [source_names.len]bool = .{false} ** source_names.len;
                for (target_names) |tname| {
                    var matched = false;
                    for (source_names, 0..) |sname, si| {
                        if (mem.eql(u8, sname, tname)) {
                            source_in_target[si] = true;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        new_axis_count += 1;
                        new_axis_name = tname;
                    }
                }
                if (new_axis_count != 1)
                    @compileError("mergeAxes: TargetEnum must introduce exactly one new axis.");
                var missing_count: usize = 0;
                for (source_names, 0..) |_, si| {
                    if (!source_in_target[si]) missing_count += 1;
                }
                if (missing_count == 0)
                    @compileError("mergeAxes: must omit at least one source axis.");
                break :blk .{ .new_axis_name = new_axis_name, .source_in_target = source_in_target };
            };
            const new_axis_name = compile_phase.new_axis_name;
            const source_in_target = compile_phase.source_in_target;

            var merged_idxs: [source_names.len]usize = undefined;
            var merged_count: usize = 0;
            inline for (source_in_target, 0..) |keep, si| {
                if (!keep) {
                    merged_idxs[merged_count] = si;
                    merged_count += 1;
                }
            }

            const strides_arr: [source_names.len]isize = @bitCast(self.strides);
            const shapes_arr: [source_names.len]usize = @bitCast(self.shape);

            const order = self.axisOrder();
            var ordered_merge: [source_names.len]usize = undefined;
            var om_count: usize = 0;
            inline for (order) |ax| {
                const si = @intFromEnum(ax);
                var is_merged = false;
                var k: usize = 0;
                while (k < merged_count) : (k += 1)
                    if (merged_idxs[k] == si) {
                        is_merged = true;
                        break;
                    };
                if (is_merged) {
                    ordered_merge[om_count] = si;
                    om_count += 1;
                }
            }

            if (om_count > 1) {
                var mi: usize = 0;
                while (mi + 1 < om_count) : (mi += 1) {
                    const a = ordered_merge[mi];
                    const b = ordered_merge[mi + 1];
                    const expected = @abs(strides_arr[b]) * shapes_arr[b];
                    if (@abs(strides_arr[a]) != expected)
                        return error.StrideMisalignment;
                }
            }

            var merged_shape_product: usize = 1;
            var ci: usize = 0;
            while (ci < om_count) : (ci += 1) merged_shape_product *= shapes_arr[ordered_merge[ci]];
            const merged_stride = strides_arr[ordered_merge[om_count - 1]];

            const NewShape = AxesStruct(target_names);
            const NewStrides = AxesStructOf(target_names, isize);
            var new_shape: NewShape = undefined;
            var new_strides: NewStrides = undefined;
            inline for (target_names) |tname| {
                if (comptime mem.eql(u8, tname, new_axis_name)) {
                    @field(new_shape, tname) = merged_shape_product;
                    @field(new_strides, tname) = merged_stride;
                } else {
                    @field(new_shape, tname) = @field(self.shape, tname);
                    @field(new_strides, tname) = @field(self.strides, tname);
                }
            }
            return .{ .shape = new_shape, .strides = new_strides, .offset = self.offset };
        }

        pub fn rename_old(self: *const @This(), comptime axis: Axis, comptime new_name: [:0]const u8) NamedIndex(Renamed(Axis, field_names[@intFromEnum(axis)], new_name)) {
            const old_name = field_names[@intFromEnum(axis)];
            const NewEnum = Renamed(Axis, old_name, new_name);
            const NewShape = AxesStruct(meta.fieldNames(NewEnum));
            const NewStrides = AxesStructOf(meta.fieldNames(NewEnum), isize);
            const new_shape: NewShape = @bitCast(self.shape);
            const new_strides: NewStrides = @bitCast(self.strides);
            return .{ .shape = new_shape, .strides = new_strides, .offset = self.offset };
        }
    };
}

/// resolveDimensions unchanged (shapes only)
pub fn resolveDimensions(shapes: anytype) NamedIndexError!AxesStruct(unionOfAxisNames(@TypeOf(shapes))) {
    const all_axis_names = comptime unionOfAxisNames(@TypeOf(shapes));
    const ResolvedShape = AxesStruct(all_axis_names);
    const ResolvedShapeOptional = AxesOptionalStruct(all_axis_names);
    var resolved_optional: ResolvedShapeOptional = .{};
    inline for (shapes) |shape| {
        const shape_info = @typeInfo(@TypeOf(shape)).@"struct";
        inline for (shape_info.fields) |field| {
            const current_size = @field(shape, field.name);
            if (@field(resolved_optional, field.name)) |existing_size| {
                if (existing_size != current_size) return NamedIndexError.ShapeMismatch;
            } else {
                @field(resolved_optional, field.name) = current_size;
            }
        }
    }
    var resolved_shape: ResolvedShape = undefined;
    inline for (all_axis_names) |axis_name| {
        if (@field(resolved_optional, axis_name)) |size| {
            @field(resolved_shape, axis_name) = size;
        } else {
            @panic("Missing axis name: " ++ axis_name);
        }
    }
    return resolved_shape;
}

pub const AxisRenamePair = struct { old: []const u8, new: []const u8 };

/// Iterator over keys in buffer order (descending absolute stride).
pub fn KeyIterator(comptime ShapeKey: type, comptime StrideKey: type) type {
    // Ensure field sets match
    comptime {
        const f1 = meta.fieldNames(ShapeKey);
        const f2 = meta.fieldNames(StrideKey);
        if (f1.len != f2.len) @compileError("KeyIterator: shape/stride field mismatch length");
        for (f1, 0..) |n, i| {
            if (!mem.eql(u8, n, f2[i])) @compileError("KeyIterator: field name mismatch");
        }
    }
    const fnames = meta.fieldNames(ShapeKey);
    return struct {
        next_arr: [fnames.len]usize,
        shape_arr: [fnames.len]usize,
        dims_desc: [fnames.len]usize,

        pub fn init(shape: ShapeKey, strides: StrideKey) @This() {
            const start = [_]usize{0} ** fnames.len;
            const shape_arr: [fnames.len]usize = @bitCast(shape);
            const strides_arr: [fnames.len]isize = @bitCast(strides);

            var argsort: [fnames.len]usize = undefined;
            inline for (0..fnames.len) |i| argsort[i] = i;
            const strides_slice: []const isize = strides_arr[0..];
            mem.sort(usize, argsort[0..], strides_slice, fnames_gt_signed);
            return .{ .next_arr = start, .shape_arr = shape_arr, .dims_desc = argsort };
        }

        pub fn next(self: *@This()) ?ShapeKey {
            if (self.dims_desc.len == 0) return null;
            if (self.next_arr[self.dims_desc[0]] >= self.shape_arr[self.dims_desc[0]])
                return null;

            const result_arr = self.next_arr;

            inline for (0..fnames.len) |di| {
                const dim_idx = fnames.len - 1 - di;
                const dim = self.dims_desc[dim_idx];
                if (self.next_arr[dim] + 1 < self.shape_arr[dim]) {
                    self.next_arr[dim] += 1;
                    break;
                } else {
                    if (dim_idx != 0) {
                        self.next_arr[dim] = 0;
                    } else {
                        self.next_arr[dim] = self.shape_arr[dim];
                    }
                }
            }
            return @bitCast(result_arr);
        }
    };
}

// Comparator for axis sorting (descending absolute stride)
fn fnames_gt_signed(strides_arr: []const isize, lhs: usize, rhs: usize) bool {
    const al = @abs(strides_arr[lhs]);
    const ar = @abs(strides_arr[rhs]);
    return al > ar or (al == ar and lhs < rhs); // tie-breaker by enum order
}

// (Old unsigned variant left for legacy code paths, if any)
// fn fnames_gt(_unused: []const usize, _lhs: usize, _rhs: usize) bool {
//     @panic("fnames_gt (unsigned) should not be used with negative strides");
// }

// pub fn AxesOptionalStruct(comptime names: []const [:0]const u8) type {
//     return AxesOptionalStructOf(names, usize);
// }

pub fn KeyEnum(comptime names: []const [:0]const u8) type {
    const rank = names.len;
    const fields = comptime blk: {
        var f: [rank]Type.EnumField = undefined;
        for (0..rank) |i| f[i] = .{ .name = names[i], .value = i };
        break :blk f;
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

fn Renamed(comptime OldKey: type, old_name: [:0]const u8, new_name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"enum";
    const new_field_names = comptime blk: {
        var out: [old_struct.fields.len][:0]const u8 = undefined;
        var matched = false;
        for (0..old_struct.fields.len) |fi| {
            const old_field = old_struct.fields[fi];
            if (mem.eql(u8, old_field.name, old_name)) {
                out[fi] = new_name;
                matched = true;
            } else {
                out[fi] = old_field.name;
            }
        }
        if (!matched) @compileError("Renamed: field not found: " ++ old_name);
        break :blk out;
    };
    return KeyEnum(&new_field_names);
}

fn Removed(comptime OldKey: type, comptime name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"enum";
    const old_rank = old_struct.fields.len;
    const new_rank = old_rank - 1;
    const new_field_names = comptime blk: {
        var matches: usize = 0;
        for (old_struct.fields) |field| {
            if (!mem.eql(u8, field.name, name)) matches += 1;
        }
        if (matches != new_rank) @compileError("Removed: field not found: " ++ name);
        var out: [matches][:0]const u8 = undefined;
        var idx: usize = 0;
        for (old_struct.fields) |field| {
            if (!mem.eql(u8, field.name, name)) {
                out[idx] = field.name;
                idx += 1;
            }
        }
        break :blk out;
    };
    return KeyEnum(&new_field_names);
}

fn Added(comptime OldKey: type, comptime name: [:0]const u8) type {
    const old_struct = @typeInfo(OldKey).@"enum";
    inline for (old_struct.fields) |field| {
        if (mem.eql(u8, field.name, name))
            @compileError("Added: field already exists: " ++ name);
    }
    const new_rank = old_struct.fields.len + 1;
    const new_field_names = comptime blk: {
        var out: [new_rank][:0]const u8 = undefined;
        for (0..old_struct.fields.len) |i| out[i] = old_struct.fields[i].name;
        out[old_struct.fields.len] = name;
        break :blk out;
    };
    return KeyEnum(&new_field_names);
}

pub fn Xor(comptime Enum1: type, comptime Enum2: type) type {
    const info1 = @typeInfo(Enum1).@"enum";
    const info2 = @typeInfo(Enum2).@"enum";
    var common1 = mem.zeroes([info1.fields.len]bool);
    var common2 = mem.zeroes([info2.fields.len]bool);
    comptime var num_matches: usize = 0;

    inline for (0..info1.fields.len) |fi| {
        inline for (0..info2.fields.len) |fj| fj_blk: {
            if (mem.eql(u8, info1.fields[fi].name, info2.fields[fj].name)) {
                common1[fi] = true;
                common2[fj] = true;
                num_matches += 1;
                break :fj_blk;
            }
        }
    }

    const xor_len = info1.fields.len + info2.fields.len - 2 * num_matches;
    comptime var xor_fnames: [xor_len][:0]const u8 = undefined;
    var i: usize = 0;
    inline for (info1.fields, 0..) |field, fi| if (!common1[fi]) {
        xor_fnames[i] = field.name;
        i += 1;
    };
    inline for (info2.fields, 0..) |field, fj| if (!common2[fj]) {
        xor_fnames[i] = field.name;
        i += 1;
    };
    assert(i == xor_len);
    return KeyEnum(&xor_fnames);
}

fn unionOfAxisNames(comptime ShapeTupleType: type) []const [:0]const u8 {
    comptime {
        const tuple_info = @typeInfo(ShapeTupleType).@"struct";
        var sum: usize = 0;
        for (tuple_info.fields) |tuple_field| {
            const info = @typeInfo(tuple_field.type);
            if (info != .@"struct") @compileError("Shape must be a struct");
            for (info.@"struct".fields) |axis_field| {
                if (axis_field.type != usize and axis_field.type != comptime_int)
                    @compileError("Expected axis type usize, found " ++ @typeName(axis_field.type));
            }
            sum += info.@"struct".fields.len;
        }
        var all_names: [sum][:0]const u8 = undefined;
        var count: usize = 0;
        for (tuple_info.fields) |tuple_field| {
            const info = @typeInfo(tuple_field.type);
            for (info.@"struct".fields) |field| {
                var found = false;
                for (all_names[0..count]) |existing|
                    if (mem.eql(u8, existing, field.name)) {
                        found = true;
                        break;
                    };
                if (!found) {
                    all_names[count] = field.name;
                    count += 1;
                }
            }
        }
        return all_names[0..count];
    }
}

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

test "resolveDimensions" {
    const Shape1 = struct { a: usize, b: usize };
    const Shape2 = struct { b: usize, c: usize };
    const Shape3 = struct { d: usize };

    // Consistent sizes
    const s1: Shape1 = .{ .a = 10, .b = 20 };
    var s2: Shape2 = .{ .b = 20, .c = 30 };
    const s3: Shape3 = .{ .d = 40 };

    const resolved = try resolveDimensions(.{ s1, s2, s3 });
    try std.testing.expectEqual(@typeInfo(@TypeOf(resolved)).@"struct".fields.len, 4);
    try std.testing.expectEqual(10, resolved.a);
    try std.testing.expectEqual(20, resolved.b);
    try std.testing.expectEqual(30, resolved.c);
    try std.testing.expectEqual(40, resolved.d);

    // Inconsistent size for 'b'
    s2.b = 99;
    const err = resolveDimensions(.{ s1, s2 });
    try std.testing.expectError(NamedIndexError.ShapeMismatch, err);

    // Test with one shape
    const resolved_one = try resolveDimensions(.{
        .{ .a = 5, .b = 6 },
    });
    try std.testing.expectEqual(5, resolved_one.a);
    try std.testing.expectEqual(6, resolved_one.b);

    // Test with empty tuple
    const resolved_empty = try resolveDimensions(.{});
    try std.testing.expectEqual(@typeInfo(@TypeOf(resolved_empty)).@"struct".fields.len, 0);

    // Shapes with optionals (should fail)
    if (false) {
        const Shape4 = struct { a: usize, c: ?usize };
        _ = try resolveDimensions(.{
            Shape1{ .a = 10, .b = 20 },
            Shape4{ .a = 10, .c = 30 },
        });
    }
}

// test "mergeAxes" {
//     const SourceAxes = enum { i, j, k };
//     const TargetAxes = enum { i, jk };

//     const source_idx: NamedIndex(SourceAxes) = .initContiguous(.{ .i = 2, .j = 3, .k = 5 });
//     const expected_reshaped: NamedIndex(TargetAxes) = .initContiguous(.{ .i = 2, .jk = 15 });

//     const actual_reshaped = try source_idx.mergeAxes(TargetAxes);

//     try std.testing.expectEqual(expected_reshaped, actual_reshaped);

//     // Error if axes cannot be merged due to stride misalignment
//     const error_source: NamedIndex(SourceAxes) = .{
//         .shape = .{ .i = 2, .j = 3, .k = 5 },
//         .strides = .{ .i = 18, .j = 6, .k = 1 },
//     };

//     const error_actual = error_source.mergeAxes(TargetAxes);

//     try std.testing.expectError(.StrideMisalignment, error_actual);
// }

test "splitAxis" {
    const SourceAxes = enum { i, jk };
    const TargetAxes = enum { i, j, k };

    const source_idx: NamedIndex(SourceAxes) = .initContiguous(.{ .i = 2, .jk = 15 });
    const expected_split: NamedIndex(TargetAxes) = .initContiguous(.{ .i = 2, .j = 3, .k = 5 });

    // Cannot error: splitting always possible
    const actual_split = source_idx.splitAxis(TargetAxes, .{ .j = 3, .k = 5 });

    try std.testing.expectEqual(expected_split, actual_split);
}

test "splitAxis non-contiguous" {
    const SourceAxes = enum { i, jk };
    const TargetAxes = enum { i, j, k };

    // Create a non-contiguous source index: jk axis is not contiguous
    const source_idx: NamedIndex(SourceAxes) = .{
        .shape = .{ .i = 2, .jk = 15 },
        .strides = .{ .i = 106, .jk = 7 }, // jk stride is not 1, so not contiguous
        .offset = 5,
    };

    // Split jk into j=3, k=5
    const actual_split = source_idx.splitAxis(TargetAxes, .{ .j = 3, .k = 5 });

    const expected_split: NamedIndex(TargetAxes) = .{
        .shape = .{ .i = 2, .j = 3, .k = 5 },
        .strides = .{ .i = 106, .j = 35, .k = 7 },
        .offset = 5,
    };

    try std.testing.expectEqual(expected_split, actual_split);
}

test "splitAxis into three" {
    const SourceAxes = enum { i, jkl, m };
    const TargetAxes = enum { i, j, k, l, m };

    const source_idx = NamedIndex(SourceAxes).initContiguous(.{ .i = 2, .jkl = 105, .m = 11 });
    const expected_split = NamedIndex(TargetAxes).initContiguous(.{ .i = 2, .j = 3, .k = 5, .l = 7, .m = 11 });
    const actual_split = source_idx.splitAxis(TargetAxes, .{ .j = 3, .k = 5, .l = 7 });

    try std.testing.expectEqual(expected_split, actual_split);
}

// test "reindex: merge axes" {
//     const SourceAxes = enum { i, j, k };
//     const TargetAxes = enum { i, jk };

//     const source_idx: NamedIndex(SourceAxes) = .initContiguous(.{ .i = 2, .j = 3, .k = 5 });
//     const expected_reshaped: NamedIndex(TargetAxes) = .initContiguous(.{ .i = 2, .jk = 15 });

//     // Error if axes cannot be merged due to stride misalignment
//     const actual_reshaped = try source_idx.reindex(TargetAxes, .{
//         // .{ .{ .j, .k }, .jk },
//         "j k -> jk",
//     });

//     try std.testing.assertEqual(expected_reshaped, actual_reshaped);
// }

// test "reindex: split axes" {
//     const SourceAxes = enum { i, jk };
//     const TargetAxes = enum { i, j, k };

//     const source_idx: NamedIndex(SourceAxes) = .initContiguous(.{ .i = 2, .jk = 15 });
//     const expected_split: NamedIndex(TargetAxes) = .initContiguous(.{ .i = 2, .j = 3, .k = 5 });

//     // Error if axes cannot be split due to stride misalignment
//     const actual_split = try source_idx.reindex(TargetAxes, .{
//         "jk -> j k",
//     });

//     try std.testing.assertEqual(expected_split, actual_split);
// }

// test "reindex: add axis" {
//     const SourceAxes = enum { i };
//     const TargetAxes = enum { i, j };

//     const source_idx: NamedIndex(SourceAxes) = .initContiguous(.{ .i = 2 });
//     const expected_added: NamedIndex(TargetAxes) = .initContiguous(.{ .i = 2, .j = 1 });

//     const actual_added = try source_idx.reindex(TargetAxes, .{
//         "-> j",
//     });

//     try std.testing.expectEqual(expected_added, actual_added);
// }
