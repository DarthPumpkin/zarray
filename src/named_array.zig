const std = @import("std");
const mem = std.mem;

const named_index = @import("named_index.zig");
const NamedIndex = named_index.NamedIndex;
const AxisRenamePair = named_index.AxisRenamePair;

const axis_meta = @import("axis_meta.zig");
const DifferenceAxesStruct = axis_meta.DifferenceAxesStruct;

const Writer = std.Io.Writer;

pub fn NamedArray(comptime Axis: type, comptime Scalar: type) type {
    const Index = NamedIndex(Axis);
    return struct {
        idx: Index,
        buf: []Scalar,

        /// Wrap an existing `idx` and `buf` without copying or allocating.
        /// Asserts that every element addressed by `idx` lies within `buf`.
        pub fn init(idx: Index, buf: []Scalar) @This() {
            assertIndexFitsBuffer(idx, buf.len);
            return .{ .idx = idx, .buf = buf };
        }

        pub fn initAlloc(allocator: mem.Allocator, shape: Index.Axes) !@This() {
            const idx = Index.initContiguous(shape);
            const buf = try allocator.alloc(Scalar, idx.count());
            return .init(idx, buf);
        }

        pub fn fill(self: *const @This(), val: Scalar) void {
            var keys = self.idx.iterKeys();
            while (keys.next()) |key| {
                self.buf[self.idx.linear(key)] = val;
            }
        }

        pub fn fillArange(self: *const @This()) void {
            var keys = self.idx.iterKeys();
            var i: Scalar = 0;
            while (keys.next()) |key| {
                self.buf[self.idx.linear(key)] = i;
                i += 1;
            }
        }

        pub fn deinit(self: *const @This(), allocator: mem.Allocator) void {
            allocator.free(self.buf);
        }

        pub fn asConst(self: *const @This()) NamedArrayConst(Axis, Scalar) {
            return .init(self.idx, self.buf);
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
            return .init(self.idx.conformAxes(NewEnum), self.buf);
        }

        /// Fix each axis not present in `NewEnum` to a single position and drop it,
        /// returning a lower-rank view over the same buffer.
        /// `indices` supplies the position for every dropped axis (`Axis \ NewEnum`).
        /// This is `conformAxes` generalized: each dropped axis is first sliced to
        /// its chosen position instead of having to already be size 1.
        pub fn indexAxes(self: *const @This(), comptime NewEnum: type, indices: DifferenceAxesStruct(Axis, NewEnum)) NamedArray(NewEnum, Scalar) {
            return indexAxesGeneric(self, NewEnum, indices);
        }

        /// Like `indexAxes`, but returns null instead of panicking if any dropped
        /// axis index is out of bounds for that axis.
        pub fn indexAxesChecked(self: *const @This(), comptime NewEnum: type, indices: DifferenceAxesStruct(Axis, NewEnum)) ?NamedArray(NewEnum, Scalar) {
            return indexAxesCheckedGeneric(self, NewEnum, indices);
        }

        /// Strictly rename axes according to the provided mapping.
        /// If any axis in NewEnum cannot be mapped, this will fail to compile.
        pub fn renameAxes(self: *const @This(), comptime NewEnum: type, comptime rename_pairs: []const AxisRenamePair) NamedArray(NewEnum, Scalar) {
            return .init(self.idx.rename(NewEnum, rename_pairs), self.buf);
        }

        /// Merge several axes of this array into a single axis described by `NewEnum`.
        /// This is a zero-copy view transformation. It asserts that the merged axes
        /// are laid out contiguously; a non-mergeable layout panics in safe build
        /// modes and is assumed in performance build modes. Use `mergeAxesChecked`
        /// when the layout is not known to be mergeable.
        pub fn mergeAxes(self: *const @This(), comptime NewEnum: type) NamedArray(NewEnum, Scalar) {
            return mergeAxesGeneric(self, NewEnum);
        }

        /// Like `mergeAxes`, but returns null instead of asserting when the axes to
        /// be merged are not laid out contiguously (and so cannot be merged without
        /// copying).
        pub fn mergeAxesChecked(self: *const @This(), comptime NewEnum: type) ?NamedArray(NewEnum, Scalar) {
            return mergeAxesCheckedGeneric(self, NewEnum);
        }

        /// Broadcast `axis` from size 1 to `new_size` by giving it a zero stride,
        /// returning a zero-copy view over the same buffer. Asserts that `axis`
        /// currently has size 1.
        pub fn broadcastAxis(self: *const @This(), comptime axis: Axis, new_size: usize) @This() {
            return .init(self.idx.broadcastAxis(axis, new_size), self.buf);
        }

        /// Broadcast every axis to `target`, returning a zero-copy view over the
        /// same buffer whose shape equals `target`. Each axis must already match
        /// the target size or have size 1 (in which case it repeats via a zero
        /// stride). Returns null if any axis cannot be broadcast.
        pub fn broadcastTo(self: *const @This(), target: Index.Axes) ?@This() {
            const idx = self.idx.broadcastTo(target) orelse return null;
            return .init(idx, self.buf);
        }

        /// Pretty-print the array. Invoked with the `{f}` format specifier.
        pub fn format(self: @This(), w: *Writer) Writer.Error!void {
            return formatArrayGeneric(self, w);
        }

        fn formatDebug(self: @This(), w: *Writer) Writer.Error!void {
            return formatArrayDebugGeneric(self, w);
        }

        /// Returns a `{f}`-printable wrapper that renders shape/strides/offset
        /// diagnostics in addition to the data.
        pub fn fmtDebug(self: @This()) std.fmt.Alt(@This(), formatDebug) {
            return .{ .data = self };
        }
    };
}

pub fn NamedArrayConst(comptime Axis: type, comptime Scalar: type) type {
    const Index = NamedIndex(Axis);
    return struct {
        idx: Index,
        buf: []const Scalar,

        /// Wrap an existing `idx` and `buf` without copying or allocating.
        /// Asserts that every element addressed by `idx` lies within `buf`.
        pub fn init(idx: Index, buf: []const Scalar) @This() {
            assertIndexFitsBuffer(idx, buf.len);
            return .{ .idx = idx, .buf = buf };
        }

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
            return .init(self.idx.conformAxes(NewEnum), self.buf);
        }

        /// Fix each axis not present in `NewEnum` to a single position and drop it,
        /// returning a lower-rank view over the same buffer.
        /// `indices` supplies the position for every dropped axis (`Axis \ NewEnum`).
        /// This is `conformAxes` generalized: each dropped axis is first sliced to
        /// its chosen position instead of having to already be size 1.
        pub fn indexAxes(self: *const @This(), comptime NewEnum: type, indices: DifferenceAxesStruct(Axis, NewEnum)) NamedArrayConst(NewEnum, Scalar) {
            return indexAxesGeneric(self, NewEnum, indices);
        }

        /// Like `indexAxes`, but returns null instead of panicking if any dropped
        /// axis index is out of bounds for that axis.
        pub fn indexAxesChecked(self: *const @This(), comptime NewEnum: type, indices: DifferenceAxesStruct(Axis, NewEnum)) ?NamedArrayConst(NewEnum, Scalar) {
            return indexAxesCheckedGeneric(self, NewEnum, indices);
        }

        /// Strictly rename axes according to the provided mapping.
        /// If any axis in NewEnum cannot be mapped, this will fail to compile.
        pub fn renameAxes(self: *const @This(), comptime NewEnum: type, comptime rename_pairs: []const AxisRenamePair) NamedArrayConst(NewEnum, Scalar) {
            return .init(self.idx.rename(NewEnum, rename_pairs), self.buf);
        }

        /// Merge several axes of this const array into a single axis described by `NewEnum`.
        /// This is a zero-copy view transformation. It asserts that the merged axes
        /// are laid out contiguously; a non-mergeable layout panics in safe build
        /// modes and is assumed in performance build modes. Use `mergeAxesChecked`
        /// when the layout is not known to be mergeable.
        pub fn mergeAxes(self: *const @This(), comptime NewEnum: type) NamedArrayConst(NewEnum, Scalar) {
            return mergeAxesGeneric(self, NewEnum);
        }

        /// Like `mergeAxes`, but returns null instead of asserting when the axes to
        /// be merged are not laid out contiguously (and so cannot be merged without
        /// copying).
        pub fn mergeAxesChecked(self: *const @This(), comptime NewEnum: type) ?NamedArrayConst(NewEnum, Scalar) {
            return mergeAxesCheckedGeneric(self, NewEnum);
        }

        /// Broadcast `axis` from size 1 to `new_size` by giving it a zero stride,
        /// returning a zero-copy view over the same buffer. Asserts that `axis`
        /// currently has size 1.
        pub fn broadcastAxis(self: *const @This(), comptime axis: Axis, new_size: usize) @This() {
            return .init(self.idx.broadcastAxis(axis, new_size), self.buf);
        }

        /// Broadcast every axis to `target`, returning a zero-copy view over the
        /// same buffer whose shape equals `target`. Each axis must already match
        /// the target size or have size 1 (in which case it repeats via a zero
        /// stride). Returns null if any axis cannot be broadcast.
        pub fn broadcastTo(self: *const @This(), target: Index.Axes) ?@This() {
            const idx = self.idx.broadcastTo(target) orelse return null;
            return .init(idx, self.buf);
        }

        /// Pretty-print the array. Invoked with the `{f}` format specifier.
        pub fn format(self: @This(), w: *Writer) Writer.Error!void {
            return formatArrayGeneric(self, w);
        }

        fn formatDebug(self: @This(), w: *Writer) Writer.Error!void {
            return formatArrayDebugGeneric(self, w);
        }

        /// Returns a `{f}`-printable wrapper that renders shape/strides/offset
        /// diagnostics in addition to the data.
        pub fn fmtDebug(self: @This()) std.fmt.Alt(@This(), formatDebug) {
            return .{ .data = self };
        }
    };
}

// Asserts that every element addressed by `idx` lies within a buffer of
// length `buf_len`. Works for both NamedArray and NamedArrayConst indices.
fn assertIndexFitsBuffer(idx: anytype, buf_len: usize) void {
    const fields = @typeInfo(@TypeOf(idx.shape)).@"struct".fields;

    // An empty axis means the array addresses no elements at all, so any
    // buffer (including an empty one) trivially fits.
    inline for (fields) |field| {
        if (@field(idx.shape, field.name) == 0) return;
    }

    // Strides may be negative (reversed/subsampled views), so track the lowest
    // and highest reachable linear addresses separately.
    var min_addr: isize = @intCast(idx.offset);
    var max_addr: isize = @intCast(idx.offset);
    inline for (fields) |field| {
        const last: isize = @intCast(@field(idx.shape, field.name) - 1);
        const span = @as(isize, @field(idx.strides, field.name)) * last;
        if (span < 0) {
            min_addr += span;
        } else {
            max_addr += span;
        }
    }

    std.debug.assert(min_addr >= 0);
    std.debug.assert(max_addr < @as(isize, @intCast(buf_len)));
}

// Works for both NamedArray and NamedArrayConst
fn flatGeneric(self: anytype) ?@TypeOf(self.buf) {
    // Only allow flatten when:
    //  - layout is contiguous (absolute stride chain)
    //  - all strides are non-negative (forward monotonic)
    if (!self.idx.isContiguous())
        return null;

    // Reject negative strides (reverse views); they are logically contiguous
    // but not a single forward slice starting at offset.
    const StridesType = @TypeOf(self.idx.strides);
    const info = @typeInfo(StridesType).@"struct";
    inline for (info.fields) |field| {
        if (@field(self.idx.strides, field.name) < 0)
            return null;
    }

    return self.buf[self.idx.offset..][0..self.idx.count()];
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
    return .init(new_idx, buf);
}

// Selects the mutable or const NamedArray type based on the buffer's constness.
fn ReducedArray(comptime BufType: type, comptime NewAxis: type) type {
    const ptr = @typeInfo(BufType).pointer;
    return if (ptr.is_const)
        NamedArrayConst(NewAxis, ptr.child)
    else
        NamedArray(NewAxis, ptr.child);
}

// Works for both NamedArray and NamedArrayConst
fn indexAxesGeneric(self: anytype, comptime NewEnum: type, indices: anytype) ReducedArray(@TypeOf(self.buf), NewEnum) {
    return .init(self.idx.indexAxes(NewEnum, indices), self.buf);
}

// Works for both NamedArray and NamedArrayConst
fn indexAxesCheckedGeneric(self: anytype, comptime NewEnum: type, indices: anytype) ?ReducedArray(@TypeOf(self.buf), NewEnum) {
    const new_idx = self.idx.indexAxesChecked(NewEnum, indices) orelse return null;
    return .init(new_idx, self.buf);
}

// Works for both NamedArray and NamedArrayConst
fn mergeAxesGeneric(self: anytype, comptime NewEnum: type) ReducedArray(@TypeOf(self.buf), NewEnum) {
    return .init(self.idx.mergeAxes(NewEnum), self.buf);
}

// Works for both NamedArray and NamedArrayConst
fn mergeAxesCheckedGeneric(self: anytype, comptime NewEnum: type) ?ReducedArray(@TypeOf(self.buf), NewEnum) {
    const new_idx = self.idx.mergeAxesChecked(NewEnum) orelse return null;
    return .init(new_idx, self.buf);
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

// ---------------------------------------------------------------------------
// Formatting
//
// Shared by NamedArray and NamedArrayConst. All traversal goes through
// `iterKeys`/`scalarAt`, so reversed, broadcast and otherwise non-contiguous
// views render in correct logical order. No allocation is performed: column
// alignment uses a single upfront pass plus a fixed stack buffer per scalar.
// ---------------------------------------------------------------------------

/// Format specifier used for a single scalar. Numeric scalars use `{d}`;
/// anything else falls back to `{any}`.
fn scalarFmtSpec(comptime Scalar: type) []const u8 {
    return switch (@typeInfo(Scalar)) {
        .int, .comptime_int, .float, .comptime_float => "{d}",
        else => "{any}",
    };
}

/// Rendered width of a scalar, without allocating.
fn scalarWidth(comptime Scalar: type, v: Scalar) usize {
    var buf: [128]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, scalarFmtSpec(Scalar), .{v}) catch return buf.len;
    return s.len;
}

/// Right-align a scalar within `col_width` columns.
fn writePadded(comptime Scalar: type, w: *Writer, v: Scalar, col_width: usize) Writer.Error!void {
    const wdt = scalarWidth(Scalar, v);
    if (col_width > wdt) try w.splatByteAll(' ', col_width - wdt);
    try w.print(scalarFmtSpec(Scalar), .{v});
}

/// Number of head/tail elements shown along a truncated axis. An axis is
/// truncated only when its length exceeds `2 * trunc_edge`, so that the
/// ellipsis always stands in for at least one hidden element.
const trunc_edge: usize = 3;

fn truncated(len: usize) bool {
    return len > 2 * trunc_edge;
}

/// Write the `...` placeholder for a hidden column, right-aligned so that the
/// tail columns stay aligned with the head columns.
fn writeEllipsisCell(w: *Writer, col_width: usize) Writer.Error!void {
    if (col_width > 3) try w.splatByteAll(' ', col_width - 3);
    try w.writeAll("...");
}

/// `NamedArray(axis: size, ...) Scalar`
fn writeHeaderGeneric(self: anytype, w: *Writer) Writer.Error!void {
    const Scalar = @typeInfo(@TypeOf(self.buf)).pointer.child;
    const is_const = @typeInfo(@TypeOf(self.buf)).pointer.is_const;
    const field_names = comptime std.meta.fieldNames(@TypeOf(self.idx).Axis);
    try w.writeAll(if (is_const) "NamedArrayConst(" else "NamedArray(");
    inline for (field_names, 0..) |fname, i| {
        if (i > 0) try w.writeAll(", ");
        try w.print("{s}: {d}", .{ fname, @field(self.idx.shape, fname) });
    }
    try w.print(") {s}", .{@typeName(Scalar)});
}

/// Render the innermost row (the last axis) as `[v0 v1 ...]`, truncating the
/// column axis with an inline ellipsis when it is long. All other axes must
/// already be fixed in `key`.
fn writeRowGeneric(self: anytype, w: *Writer, key: *@TypeOf(self.idx).Axes, col_width: usize) Writer.Error!void {
    const Scalar = @typeInfo(@TypeOf(self.buf)).pointer.child;
    const field_names = comptime std.meta.fieldNames(@TypeOf(self.idx).Axis);
    const rank = field_names.len;
    const col_axis = field_names[rank - 1];
    const cols = @field(self.idx.shape, col_axis);

    try w.writeByte('[');
    if (truncated(cols)) {
        var c: usize = 0;
        while (c < trunc_edge) : (c += 1) {
            if (c > 0) try w.writeByte(' ');
            @field(key.*, col_axis) = c;
            try writePadded(Scalar, w, self.scalarAt(key.*), col_width);
        }
        try w.writeByte(' ');
        try writeEllipsisCell(w, col_width);
        c = cols - trunc_edge;
        while (c < cols) : (c += 1) {
            try w.writeByte(' ');
            @field(key.*, col_axis) = c;
            try writePadded(Scalar, w, self.scalarAt(key.*), col_width);
        }
    } else {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            if (c > 0) try w.writeByte(' ');
            @field(key.*, col_axis) = c;
            try writePadded(Scalar, w, self.scalarAt(key.*), col_width);
        }
    }
    try w.writeByte(']');
}

/// Render the innermost 1D row or 2D grid (the last one or two axes), with
/// `outer` axes already fixed in `key`. Both the row and column axes are
/// truncated with an ellipsis when long.
fn writeBlockGeneric(self: anytype, w: *Writer, key: *@TypeOf(self.idx).Axes, col_width: usize, indent: usize) Writer.Error!void {
    const field_names = comptime std.meta.fieldNames(@TypeOf(self.idx).Axis);
    const rank = field_names.len;
    const inner = @min(rank, 2);
    if (inner == 1) {
        try w.splatByteAll(' ', indent);
        try writeRowGeneric(self, w, key, col_width);
    } else {
        const row_axis = field_names[rank - 2];
        const rows = @field(self.idx.shape, row_axis);
        try w.splatByteAll(' ', indent);
        try w.writeByte('[');
        if (truncated(rows)) {
            var r: usize = 0;
            while (r < trunc_edge) : (r += 1) {
                if (r > 0) {
                    try w.writeByte('\n');
                    try w.splatByteAll(' ', indent + 1);
                }
                @field(key.*, row_axis) = r;
                try writeRowGeneric(self, w, key, col_width);
            }
            try w.writeByte('\n');
            try w.splatByteAll(' ', indent + 1);
            try w.writeAll("...");
            r = rows - trunc_edge;
            while (r < rows) : (r += 1) {
                try w.writeByte('\n');
                try w.splatByteAll(' ', indent + 1);
                @field(key.*, row_axis) = r;
                try writeRowGeneric(self, w, key, col_width);
            }
        } else {
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                if (r > 0) {
                    try w.writeByte('\n');
                    try w.splatByteAll(' ', indent + 1);
                }
                @field(key.*, row_axis) = r;
                try writeRowGeneric(self, w, key, col_width);
            }
        }
        try w.writeByte(']');
    }
}

/// Render the data body: a single bracketed block for rank <= 2, or one
/// labeled block per outer-axis combination for rank >= 3.
fn writeBodyGeneric(self: anytype, w: *Writer) Writer.Error!void {
    const Index = @TypeOf(self.idx);
    const Axes = Index.Axes;
    const Scalar = @typeInfo(@TypeOf(self.buf)).pointer.child;
    const field_names = comptime std.meta.fieldNames(Index.Axis);
    const rank = field_names.len;

    if (rank == 0) {
        const key: Axes = undefined;
        try w.print(scalarFmtSpec(Scalar), .{self.scalarAt(key)});
        return;
    }
    if (self.idx.count() == 0) {
        try w.writeAll("[]");
        return;
    }

    // Single upfront pass to size the value column for aligned output.
    const col_width = blk: {
        var maxw: usize = 0;
        var it = self.idx.iterKeys();
        while (it.next()) |key| {
            const wdt = scalarWidth(Scalar, self.scalarAt(key));
            if (wdt > maxw) maxw = wdt;
        }
        break :blk maxw;
    };

    if (rank <= 2) {
        var key: Axes = undefined;
        try writeBlockGeneric(self, w, &key, col_width, 0);
    } else {
        const num_outer = rank - 2;
        var outer_shape: [num_outer]usize = undefined;
        inline for (0..num_outer) |oi| outer_shape[oi] = @field(self.idx.shape, field_names[oi]);

        var total: usize = 1;
        for (outer_shape) |s| total *= s;

        var key: Axes = undefined;
        const truncate = truncated(total);
        var first = true;
        var i: usize = 0;
        while (i < total) : (i += 1) {
            // Treat the outer-axis combinations as one flat sequence and elide
            // its middle. The per-slice labels keep the shown indices explicit.
            if (truncate and i == trunc_edge) {
                try w.writeAll("\n\n...");
                first = false;
                i = total - trunc_edge;
            }

            // Decompose the flat index into per-outer-axis indices (row-major,
            // last outer axis fastest).
            var outer: [num_outer]usize = undefined;
            var rem = i;
            var d = num_outer;
            while (d > 0) {
                d -= 1;
                outer[d] = rem % outer_shape[d];
                rem /= outer_shape[d];
            }
            inline for (0..num_outer) |oi| @field(key, field_names[oi]) = outer[oi];

            if (!first) try w.writeAll("\n\n");
            first = false;

            // Label the fixed outer axes, e.g. `[batch=0, chan=1]`.
            try w.writeByte('[');
            inline for (0..num_outer) |oi| {
                if (oi > 0) try w.writeAll(", ");
                try w.print("{s}={d}", .{ field_names[oi], outer[oi] });
            }
            try w.writeAll("]\n");

            try writeBlockGeneric(self, w, &key, col_width, 2);
        }
    }
}

fn formatArrayGeneric(self: anytype, w: *Writer) Writer.Error!void {
    try writeHeaderGeneric(self, w);
    try w.writeByte('\n');
    try writeBodyGeneric(self, w);
}

fn formatArrayDebugGeneric(self: anytype, w: *Writer) Writer.Error!void {
    const field_names = comptime std.meta.fieldNames(@TypeOf(self.idx).Axis);
    try writeHeaderGeneric(self, w);
    try w.writeByte('\n');

    try w.writeAll("  shape:      {");
    inline for (field_names, 0..) |fname, i| {
        if (i > 0) try w.writeByte(',');
        try w.print(" {s}: {d}", .{ fname, @field(self.idx.shape, fname) });
    }
    try w.writeAll(" }\n");

    try w.writeAll("  strides:    {");
    inline for (field_names, 0..) |fname, i| {
        if (i > 0) try w.writeByte(',');
        try w.print(" {s}: {d}", .{ fname, @field(self.idx.strides, fname) });
    }
    try w.writeAll(" }\n");

    try w.print("  offset:     {d}\n", .{self.idx.offset});
    try w.print("  contiguous: {}\n", .{self.idx.isContiguous()});

    try writeBodyGeneric(self, w);
}

test "fill" {
    const Axis = enum { i };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 4 });
    arr.fill(0);
    defer arr.deinit(allocator);

    const expected_zeros = [_]i32{ 0, 0, 0, 0 };
    try std.testing.expectEqualSlices(i32, &expected_zeros, arr.buf);

    arr.fillArange();
    const expected_arange = [_]i32{ 0, 1, 2, 3 };
    try std.testing.expectEqualSlices(i32, &expected_arange, arr.buf);
}

test "init wraps existing index and buffer" {
    const IJ = enum { i, j };
    const Index = NamedIndex(IJ);

    var buf = [_]i32{ 0, 1, 2, 3, 4, 5 };
    const idx = Index.initContiguous(.{ .i = 2, .j = 3 });

    const arr = NamedArray(IJ, i32).init(idx, &buf);
    try std.testing.expectEqual(@as(i32, 4), arr.at(.{ .i = 1, .j = 1 }).*);

    // The const variant shares the same validation.
    const carr = NamedArrayConst(IJ, i32).init(idx, &buf);
    try std.testing.expectEqual(@as(i32, 5), carr.at(.{ .i = 1, .j = 2 }).*);

    // A reversed view still fits its buffer (offset points at the last element).
    const rev = idx.strideAxis(.j, -1);
    const rarr = NamedArray(IJ, i32).init(rev, &buf);
    try std.testing.expectEqual(@as(i32, 3), rarr.at(.{ .i = 1, .j = 2 }).*);
}

test "flat, toContiguous" {
    const IJ = enum { i, j };

    const al = std.testing.allocator;
    var arr = try NamedArray(IJ, i32).initAlloc(al, .{ .i = 5, .j = 9 });
    defer arr.deinit(al);
    arr.fillArange();

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

test "NamedArray toContiguous from reversed view retains logical order" {
    const IJ = enum { i, j };
    const allocator = std.testing.allocator;
    var arr = try NamedArray(IJ, i32).initAlloc(allocator, .{ .i = 3, .j = 4 });
    defer arr.deinit(allocator);
    arr.fillArange(); // original layout values: row-major

    // Reverse i
    arr.idx = arr.idx.strideAxis(.i, -1);
    const copy = try arr.toContiguous(allocator);
    defer copy.deinit(allocator);

    // copy is contiguous in logical reversed order; verify first and last rows
    // Logical row 0 of reversed view corresponds to original last row
    var reversed_first_row_ok = true;
    for (0..arr.idx.shape.j) |j| {
        const original_val = (arr.idx.shape.i - 1) * 4 + j;
        const copied_val = copy.scalarAt(.{ .i = 0, .j = j });
        if (original_val != copied_val) reversed_first_row_ok = false;
    }
    try std.testing.expect(reversed_first_row_ok);

    // Logical last row corresponds to original first row
    var reversed_last_row_ok = true;
    for (0..arr.idx.shape.j) |j| {
        const original_val = 0 * 4 + j;
        const copied_val = copy.scalarAt(.{ .i = arr.idx.shape.i - 1, .j = j });
        if (original_val != copied_val) reversed_last_row_ok = false;
    }
    try std.testing.expect(reversed_last_row_ok);
}

test "get*" {
    // Test all the get* methods, both for NamedArray and NamedArrayConst
    const IJ = enum { i, j };
    const idx = NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 });
    var buf = [_]i32{ 10, 11, 12, 13, 14, 15 };
    const arr = NamedArray(IJ, i32).init(idx, &buf);
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
    var arr = NamedArray(IJ, i32).init(idx, &buf);
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

test "NamedArray negative strideAxis reversal contiguous flat" {
    const IJ = enum { i, j };
    const allocator = std.testing.allocator;
    var arr = try NamedArray(IJ, i32).initAlloc(allocator, .{ .i = 3, .j = 4 });
    defer arr.deinit(allocator);
    arr.fillArange();

    // Reverse the major axis (i)
    arr.idx = arr.idx.strideAxis(.i, -1);

    // Contiguous reversed view should still flatten
    const flat_opt = arr.flat();
    try std.testing.expectEqual(null, flat_opt);
}

test "NamedArray negative subsampled stride non-contiguous flat null" {
    const IJ = enum { i, j };
    const allocator = std.testing.allocator;
    var arr = try NamedArray(IJ, i32).initAlloc(allocator, .{ .i = 6, .j = 5 });
    defer arr.deinit(allocator);
    arr.fillArange();

    // Subsample + reverse i axis (step -2)
    arr.idx = arr.idx.strideAxis(.i, -2);
    // Non-contiguous now (abs stride chain breaks), flat should return null
    try std.testing.expectEqual(null, arr.flat());

    // Shape.i = ceil(6/2)=3; check a few mapped elements
    // Logical i mapping: 0->5, 1->3, 2->1
    const vals = [_]i32{
        arr.scalarAt(.{ .i = 0, .j = 0 }),
        arr.scalarAt(.{ .i = 1, .j = 0 }),
        arr.scalarAt(.{ .i = 2, .j = 0 }),
    };
    const expected = [_]i32{
        5 * 5 + 0, // original (5,0)
        3 * 5 + 0, // original (3,0)
        1 * 5 + 0, // original (1,0)
    };
    try std.testing.expectEqualSlices(i32, &expected, &vals);
}

test "NamedArray stride multi-axis negative and positive" {
    const IJ = enum { i, j };
    const allocator = std.testing.allocator;
    var arr = try NamedArray(IJ, i32).initAlloc(allocator, .{ .i = 5, .j = 6 });
    defer arr.deinit(allocator);
    arr.fillArange();

    // Apply multi-axis stride: reverse i (step -1), subsample j by +2
    arr.idx = arr.idx.stride(.{ .i = -1, .j = 2 });

    // Shapes
    try std.testing.expectEqual(5, arr.idx.shape.i);
    try std.testing.expectEqual(3, arr.idx.shape.j);

    // Check mapping for a couple of points:
    // Logical i=0 -> original i=4; logical j=1 -> original j=2
    const v1 = arr.scalarAt(.{ .i = 0, .j = 1 });
    const expected_v1 = 4 * 6 + 2;
    try std.testing.expectEqual(expected_v1, v1);

    // Logical i=4 -> original i=0; logical j=2 -> original j=4
    const v2 = arr.scalarAt(.{ .i = 4, .j = 2 });
    const expected_v2 = 0 * 6 + 4;
    try std.testing.expectEqual(expected_v2, v2);
}

test "NamedArray renameAxes strict" {
    const IJ = enum { i, j };
    const XY = enum { x, y };

    var buf = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const arr = NamedArray(IJ, i32).init(NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }), &buf);
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
    const arr = NamedArrayConst(IJ, i32).init(NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }), &buf);
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
    const arr = NamedArray(IJ, i32).init(NamedIndex(IJ).initContiguous(.{ .i = 2, .j = 3 }), &buf);
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
    var arr = NamedArrayConst(IJ, i32).init(.initContiguous(.{ .i = 2, .j = 3 }), &[_]i32{ 1, 2, 3, 4, 5, 6 });
    arr.idx = arr.idx.sliceAxis(.j, 1, 2);

    try std.testing.expectEqual(2, arr.scalarAt(.{ .i = 0, .j = 0 }));
    try std.testing.expectEqual(5, arr.scalarAt(.{ .i = 1, .j = 0 }));
    try std.testing.expectEqual(null, arr.scalarAtChecked(.{ .i = 1, .j = 1 }));
}

test "broadcastTo" {
    const IJ = enum { i, j };
    // A single row broadcast across the "i" axis.
    const row = NamedArrayConst(IJ, i32).init(.initContiguous(.{ .i = 1, .j = 3 }), &[_]i32{ 1, 2, 3 });
    const broad = row.broadcastTo(.{ .i = 2, .j = 3 }).?;

    try std.testing.expectEqual(@as(i32, 1), broad.scalarAt(.{ .i = 0, .j = 0 }));
    try std.testing.expectEqual(@as(i32, 3), broad.scalarAt(.{ .i = 0, .j = 2 }));
    // Row 1 sees the same underlying data thanks to the zero stride.
    try std.testing.expectEqual(@as(i32, 1), broad.scalarAt(.{ .i = 1, .j = 0 }));
    try std.testing.expectEqual(@as(i32, 3), broad.scalarAt(.{ .i = 1, .j = 2 }));

    // Null when a non-1 axis size mismatches the target.
    try std.testing.expectEqual(null, row.broadcastTo(.{ .i = 2, .j = 4 }));
}

test "keepOnly" {
    const IJK = enum { i, j, k };
    const IJ = enum { i, j };

    const arr = NamedArrayConst(IJK, i32).init(.initContiguous(.{ .i = 4, .j = 1, .k = 1 }), &[_]i32{ 1, 2, 3, 4 });

    const squeezed = NamedArrayConst(IJ, i32).init(arr.idx.keepOnly(IJ), arr.buf);

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
    const arr = NamedArray(IJK, i32).init(NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 1, .k = 1 }), &buf);
    const arr_proj = arr.conformAxes(IKL);
    try std.testing.expectEqual(arr.buf, arr_proj.buf);
    try std.testing.expectEqual(arr.idx.conformAxes(IKL), arr_proj.idx);

    // Should panic if removed axis does not have size 1
    if (false) {
        const arr_bad = NamedArray(IJK, i32).init(NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 2, .k = 1 }), &buf);
        _ = arr_bad.conformAxes(IKL);
    }
}

test "NamedArrayConst conformAxes" {
    const IJK = enum { i, j, k };
    const IKL = enum { i, k, l };

    const buf = [_]i32{ 1, 2, 3, 4 };
    const arr = NamedArrayConst(IJK, i32).init(NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 1, .k = 1 }), &buf);
    const arr_proj = arr.conformAxes(IKL);
    try std.testing.expectEqual(arr.buf, arr_proj.buf);
    try std.testing.expectEqual(arr.idx.conformAxes(IKL), arr_proj.idx);

    // Should panic if removed axis does not have size 1
    if (false) {
        const arr_bad = NamedArrayConst(IJK, i32).init(NamedIndex(IJK).initContiguous(.{ .i = 4, .j = 2, .k = 1 }), &buf);
        _ = arr_bad.conformAxes(IKL);
    }
}

test "NamedArray indexAxes" {
    const IJK = enum { i, j, k };
    const J = enum { j };
    const KI = enum { k, i };

    // Contiguous 2x3x4, buf[n] = n so scalarAt(.{i,j,k}) == i*12 + j*4 + k.
    var buf: [24]i32 = undefined;
    for (&buf, 0..) |*v, n| v.* = @intCast(n);
    const arr = NamedArray(IJK, i32).init(NamedIndex(IJK).initContiguous(.{ .i = 2, .j = 3, .k = 4 }), &buf);

    // Drop i and k, keep j: fix i=1, k=2 -> 1*12 + j*4 + 2.
    const row = arr.indexAxes(J, .{ .i = 1, .k = 2 });
    try std.testing.expectEqual(&buf, row.buf.ptr);
    try std.testing.expectEqual(14, row.scalarAt(.{ .j = 0 }));
    try std.testing.expectEqual(18, row.scalarAt(.{ .j = 1 }));
    try std.testing.expectEqual(22, row.scalarAt(.{ .j = 2 }));

    // Kept axes may be reordered relative to the original: keep {k, i}, drop j=1.
    const plane = arr.indexAxes(KI, .{ .j = 1 });
    try std.testing.expectEqual(4, plane.scalarAt(.{ .i = 0, .k = 0 })); // 0*12 + 4 + 0
    try std.testing.expectEqual(7, plane.scalarAt(.{ .i = 0, .k = 3 })); // 0*12 + 4 + 3
    try std.testing.expectEqual(16, plane.scalarAt(.{ .i = 1, .k = 0 })); // 1*12 + 4 + 0

    // Writes through the reduced view reach the shared buffer.
    row.at(.{ .j = 1 }).* = -1;
    try std.testing.expectEqual(-1, arr.scalarAt(.{ .i = 1, .j = 1, .k = 2 }));

    // Checked variant: in-bounds yields the same view, out-of-bounds yields null.
    const ok = arr.indexAxesChecked(J, .{ .i = 1, .k = 2 });
    try std.testing.expect(ok != null);
    try std.testing.expectEqual(row.idx, ok.?.idx);
    try std.testing.expectEqual(null, arr.indexAxesChecked(J, .{ .i = 2, .k = 2 })); // i == size
    try std.testing.expectEqual(null, arr.indexAxesChecked(J, .{ .i = 0, .k = 4 })); // k == size
}

test "NamedArrayConst indexAxes" {
    const IJK = enum { i, j, k };
    const J = enum { j };

    var buf: [24]i32 = undefined;
    for (&buf, 0..) |*v, n| v.* = @intCast(n);
    const arr = NamedArrayConst(IJK, i32).init(NamedIndex(IJK).initContiguous(.{ .i = 2, .j = 3, .k = 4 }), &buf);

    const row = arr.indexAxes(J, .{ .i = 1, .k = 2 });
    try std.testing.expectEqual(14, row.scalarAt(.{ .j = 0 }));
    try std.testing.expectEqual(18, row.scalarAt(.{ .j = 1 }));
    try std.testing.expectEqual(22, row.scalarAt(.{ .j = 2 }));

    try std.testing.expect(arr.indexAxesChecked(J, .{ .i = 1, .k = 2 }) != null);
    try std.testing.expectEqual(null, arr.indexAxesChecked(J, .{ .i = 0, .k = 4 }));
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

    const arr_ijk = NamedArrayConst(IJK, i32).init(.initContiguous(.{
        .i = 2,
        .j = 3,
        .k = 4,
    }), &buf_1_through_24);
    const arr_il = arr_ijk.mergeAxes(IL);

    try std.testing.expectEqual(NamedIndex(IL).Axes{ .i = 2, .l = 12 }, arr_il.idx.shape);
    try std.testing.expectEqual(2, arr_il.scalarAt(.{ .i = 0, .l = 1 }));
    try std.testing.expectEqual(15, arr_il.scalarAt(.{ .i = 1, .l = 2 }));

    // The checked variant yields the same view when the layout is mergeable.
    const arr_il_checked = arr_ijk.mergeAxesChecked(IL).?;
    try std.testing.expectEqual(NamedIndex(IL).Axes{ .i = 2, .l = 12 }, arr_il_checked.idx.shape);

    // Failing case: last dim has stride 3, but shape 4 -> cannot merge without copying
    const arr_ijk_strided = NamedArrayConst(IJK, i32).init(.{
        .strides = .{ .i = 12, .j = 4, .k = 3 },
        .shape = .{ .i = 2, .j = 3, .k = 2 },
    }, &buf_1_through_24);

    try std.testing.expectEqual(@as(?NamedArrayConst(IL, i32), null), arr_ijk_strided.mergeAxesChecked(IL));

    // Failing case: axes not consecutive (j i k -> i l)
    const arr_ijk_noncon = NamedArrayConst(IJK, i32).init(.{
        .strides = .{ .i = 4, .j = 8, .k = 1 },
        .shape = .{ .i = 2, .j = 3, .k = 4 },
    }, &buf_1_through_24);

    try std.testing.expectEqual(@as(?NamedArrayConst(IL, i32), null), arr_ijk_noncon.mergeAxesChecked(IL));

    // Edge case: shape (1, 1, 1)
    const buf_single = [_]i32{42};
    const arr_ones = NamedArrayConst(IJK, i32).init(.initContiguous(.{ .i = 1, .j = 1, .k = 1 }), &buf_single);
    const arr_ones_merged = arr_ones.mergeAxes(IL);
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

test "format 1d" {
    const Axis = enum { i };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 3 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt("NamedArray(i: 3) i32\n[0 1 2]", "{f}", .{arr});
}

test "format 2d aligned" {
    const Axis = enum { i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 2, .j = 5 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(i: 2, j: 5) i32\n[[0 1 2 3 4]\n [5 6 7 8 9]]",
        "{f}",
        .{arr},
    );
}

test "format 3d labeled slices" {
    const Axis = enum { b, i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .b = 2, .i = 2, .j = 2 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(b: 2, i: 2, j: 2) i32\n" ++
            "[b=0]\n  [[0 1]\n   [2 3]]\n\n" ++
            "[b=1]\n  [[4 5]\n   [6 7]]",
        "{f}",
        .{arr},
    );
}

test "format debug" {
    const Axis = enum { i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 2, .j = 3 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(i: 2, j: 3) i32\n" ++
            "  shape:      { i: 2, j: 3 }\n" ++
            "  strides:    { i: 3, j: 1 }\n" ++
            "  offset:     0\n" ++
            "  contiguous: true\n" ++
            "[[0 1 2]\n [3 4 5]]",
        "{f}",
        .{arr.fmtDebug()},
    );
}

test "format const alignment with wide values" {
    const Axis = enum { i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 2, .j = 2 });
    defer arr.deinit(allocator);
    arr.at(.{ .i = 0, .j = 0 }).* = 1;
    arr.at(.{ .i = 0, .j = 1 }).* = 20;
    arr.at(.{ .i = 1, .j = 0 }).* = 300;
    arr.at(.{ .i = 1, .j = 1 }).* = 4;
    try std.testing.expectFmt(
        "NamedArrayConst(i: 2, j: 2) i32\n[[  1  20]\n [300   4]]",
        "{f}",
        .{arr.asConst()},
    );
}

test "format truncates long 1d column axis" {
    const Axis = enum { i };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 10 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt("NamedArray(i: 10) i32\n[0 1 2 ... 7 8 9]", "{f}", .{arr});
}

test "format truncates long 2d column axis with alignment" {
    const Axis = enum { i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 2, .j = 10 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(i: 2, j: 10) i32\n[[ 0  1  2 ...  7  8  9]\n [10 11 12 ... 17 18 19]]",
        "{f}",
        .{arr},
    );
}

test "format truncates long 2d row axis" {
    const Axis = enum { i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 8, .j = 2 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(i: 8, j: 2) i32\n" ++
            "[[ 0  1]\n [ 2  3]\n [ 4  5]\n ...\n [10 11]\n [12 13]\n [14 15]]",
        "{f}",
        .{arr},
    );
}

test "format truncates long outer slice sequence" {
    const Axis = enum { b, i, j };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .b = 8, .i = 1, .j = 1 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(b: 8, i: 1, j: 1) i32\n" ++
            "[b=0]\n  [[0]]\n\n[b=1]\n  [[1]]\n\n[b=2]\n  [[2]]\n\n" ++
            "...\n\n" ++
            "[b=5]\n  [[5]]\n\n[b=6]\n  [[6]]\n\n[b=7]\n  [[7]]",
        "{f}",
        .{arr},
    );
}

test "format does not truncate at the boundary length" {
    const Axis = enum { i };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .i = 6 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt("NamedArray(i: 6) i32\n[0 1 2 3 4 5]", "{f}", .{arr});
}

test "format 4d with two labeled outer axes" {
    const Axis = enum { a, b, r, c };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .a = 2, .b = 2, .r = 2, .c = 2 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(a: 2, b: 2, r: 2, c: 2) i32\n" ++
            "[a=0, b=0]\n  [[ 0  1]\n   [ 2  3]]\n\n" ++
            "[a=0, b=1]\n  [[ 4  5]\n   [ 6  7]]\n\n" ++
            "[a=1, b=0]\n  [[ 8  9]\n   [10 11]]\n\n" ++
            "[a=1, b=1]\n  [[12 13]\n   [14 15]]",
        "{f}",
        .{arr},
    );
}

test "format 4d truncates flattened outer sequence across axes" {
    const Axis = enum { a, b, r, c };
    const allocator = std.testing.allocator;
    const arr = try NamedArray(Axis, i32).initAlloc(allocator, .{ .a = 2, .b = 8, .r = 1, .c = 1 });
    defer arr.deinit(allocator);
    arr.fillArange();
    try std.testing.expectFmt(
        "NamedArray(a: 2, b: 8, r: 1, c: 1) i32\n" ++
            "[a=0, b=0]\n  [[ 0]]\n\n[a=0, b=1]\n  [[ 1]]\n\n[a=0, b=2]\n  [[ 2]]\n\n" ++
            "...\n\n" ++
            "[a=1, b=5]\n  [[13]]\n\n[a=1, b=6]\n  [[14]]\n\n[a=1, b=7]\n  [[15]]",
        "{f}",
        .{arr},
    );
}
