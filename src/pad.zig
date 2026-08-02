const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const assert = std.debug.assert;

const za = @import("root.zig");
const axis_meta = za.axis_meta;

/// Fill behavior of a padded axis, matching the conventional framework modes
/// (PyTorch `F.pad`, NumPy `np.pad`). `constant` carries the required fill
/// value; the other modes repeat/mirror/wrap the edge values, so a fill value
/// cannot be attached to them (invalid states are unrepresentable).
pub fn PadMode(comptime Scalar: type) type {
    return union(enum(u8)) {
        /// Fill the padded region with a fixed value (0 for zero padding).
        constant: Scalar,
        /// Mirror around the edge without repeating it: `a b c d` padded by 2 on
        /// the left becomes `c b a b c d`. Requires the pad amounts to be smaller
        /// than the axis dimension.
        reflect,
        /// Repeat the edge value: `a b c d` padded by 3 on the left becomes
        /// `a a a a b c d`.
        replicate,
        /// Wrap around: `a b c d` padded by 2 on the left becomes `c d a b c d`.
        circular,
    };
}

/// Padding applied to a single axis. The mode is required: padding behavior is
/// always explicit, never defaulted.
pub fn PadAmount(comptime Scalar: type) type {
    return struct {
        before: usize,
        after: usize,
        mode: PadMode(Scalar),
    };
}

/// Per-axis padding specification for `pad`, keyed by axis name. Axes omitted
/// from the struct literal receive no padding; because the field set is derived
/// from the axis enum, misspelled axis names fail to compile.
///
/// Constant-mode axes with padding must agree on a single fill value; differing
/// fills would make overlap regions ambiguous (asserted in `pad`).
pub fn PaddingSpec(comptime Axis: type, comptime Scalar: type) type {
    return axis_meta.AxesOptionalStructOf(meta.fieldNames(Axis), PadAmount(Scalar));
}

/// Allocating padding of `arr` along any subset of its axes.
///
/// Returns a new contiguous array whose shape is `shape + before + after` per
/// padded axis, with the padded region filled per `PadMode` and the interior
/// holding a copy of `arr`. Works for any rank and any channel structure
/// (ADR-0002: padding is a composable operation external to convolution;
/// same/full/arbitrary amounts are `before`/`after` choices).
///
/// The caller provides the allocator; the result must be deallocated with
/// `deinit`.
pub fn pad(
    comptime Axis: type,
    comptime Scalar: type,
    allocator: mem.Allocator,
    arr: za.NamedArrayConst(Axis, Scalar),
    amounts: PaddingSpec(Axis, Scalar),
) !za.NamedArray(Axis, Scalar) {
    const Index = za.index.NamedIndex(Axis);

    var new_shape = arr.idx.shape;
    inline for (meta.fields(Axis)) |f| {
        if (@field(amounts, f.name)) |a| {
            @field(new_shape, f.name) += a.before + a.after;
        }
    }

    const padded = try za.NamedArray(Axis, Scalar).initAlloc(allocator, new_shape);

    // Validate: padded constant-mode axes must share a single fill value
    // (overlapping padded regions with differing fills would be ambiguous).
    // Zero-padding axes carry no fill and impose no constraint.
    var const_fill: ?Scalar = null;
    var all_constant = true;
    inline for (meta.fields(Axis)) |f| {
        if (@field(amounts, f.name)) |a| {
            if (a.before == 0 and a.after == 0) {
                // Zero-padding axes carry no fill and impose no constraint.
            } else switch (a.mode) {
                .constant => |fill| {
                    if (const_fill) |cf| {
                        assert(cf == fill);
                    } else {
                        const_fill = fill;
                    }
                },
                else => all_constant = false,
            }
        }
    }

    // Fast path: all padded axes are constant mode (single validated fill,
    // covers plain zero padding). Fill the buffer, then copy the interior.
    if (all_constant) {
        padded.fill(const_fill orelse 0);
        var inner = padded.idx;
        inline for (meta.fields(Axis)) |f| {
            if (@field(amounts, f.name)) |a| {
                const orig: usize = @field(arr.idx.shape, f.name);
                inner.sliceAxisInplace(@field(Axis, f.name), a.before, a.before + orig);
            }
        }
        const inner_view = za.NamedArray(Axis, Scalar).init(inner, padded.buf);
        inner_view.fillCopy(arr);
        return padded;
    }

    // Generic path: map every output key to a source key (or a constant fill).
    // `const_fill` is non-null whenever a padded constant axis exists.
    var out_iter = padded.idx.iterKeys();
    while (out_iter.next()) |ok| {
        var src: Index.Axes = undefined;
        var from_constant = false;
        inline for (meta.fields(Axis)) |f| {
            if (@field(amounts, f.name)) |a| {
                const dim: usize = @field(arr.idx.shape, f.name);
                const oi: usize = @field(ok, f.name);
                switch (a.mode) {
                    .constant => {
                        if (oi < a.before or oi >= a.before + dim) {
                            from_constant = true;
                            @field(src, f.name) = 0;
                        } else {
                            @field(src, f.name) = oi - a.before;
                        }
                    },
                    .replicate => {
                        @field(src, f.name) = padIndexReplicate(dim, a.before, oi);
                    },
                    .reflect => {
                        assert(a.before < dim and a.after < dim);
                        @field(src, f.name) = padIndexReflect(dim, a.before, oi);
                    },
                    .circular => {
                        @field(src, f.name) = padIndexCircular(dim, a.before, oi);
                    },
                }
            } else {
                @field(src, f.name) = @field(ok, f.name);
            }
        }
        padded.at(ok).* = if (from_constant) const_fill orelse 0 else arr.scalarAt(src);
    }
    return padded;
}

/// Source index for `replicate`: clamp to the edge.
fn padIndexReplicate(dim: usize, before: usize, idx: usize) usize {
    assert(dim > 0);
    const rel: isize = @as(isize, @intCast(idx)) - @as(isize, @intCast(before));
    const hi: isize = @as(isize, @intCast(dim - 1));
    return @intCast(@min(@max(rel, 0), hi));
}

/// Source index for `reflect`: mirror about the edge without repeating it.
/// Caller must have validated `before`/`after` < `dim`, so a single
/// reflection suffices.
fn padIndexReflect(dim: usize, before: usize, idx: usize) usize {
    assert(dim > 0);
    const rel: isize = @as(isize, @intCast(idx)) - @as(isize, @intCast(before));
    const d: isize = @as(isize, @intCast(dim));
    if (rel < 0) return @intCast(-rel);
    if (rel >= d) return @intCast(2 * (d - 1) - rel);
    return @intCast(rel);
}

/// Source index for `circular`: wrap around the axis.
fn padIndexCircular(dim: usize, before: usize, idx: usize) usize {
    assert(dim > 0);
    const rel: isize = @as(isize, @intCast(idx)) - @as(isize, @intCast(before));
    return @intCast(@mod(rel, @as(isize, @intCast(dim))));
}

fn expectAt(comptime Axis: type, comptime Scalar: type, arr: za.NamedArrayConst(Axis, Scalar), key: za.index.NamedIndex(Axis).Axes, expected: Scalar) !void {
    try std.testing.expectEqual(expected, arr.scalarAt(key));
}

test "pad: 2D asymmetric padding" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 3, .w = 4 });
    defer src.deinit(al);
    src.fillArange();

    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 2, .after = 1, .mode = .{ .constant = 0 } },
        .w = .{ .before = 1, .after = 3, .mode = .{ .constant = 0 } },
    });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 6), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 8), padded.idx.shape.w);

    // Padding region is zero.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 0 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 7 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 5, .w = 0 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 5, .w = 7 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 2 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 4, .w = 5 }, 0);

    // Interior matches the source, shifted by the leading padding.
    var keys = src.idx.iterKeys();
    while (keys.next()) |k| {
        try expectAt(Axes, f, padded.asConst(), .{ .h = k.h + 2, .w = k.w + 1 }, src.scalarAt(k));
    }
}

test "pad: constant mode with non-zero fill" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 2, .w = 2 });
    defer src.deinit(al);
    src.fillArange();

    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .{ .constant = 7 } },
        .w = .{ .before = 1, .after = 1, .mode = .{ .constant = 7 } },
    });
    defer padded.deinit(al);

    // Corners and edges take the shared fill value.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 0 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 1 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 3 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 3, .w = 3 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 0 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 3 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 1 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 2 }, 3);
}

test "pad: 3D (channels) symmetric padding" {
    const Axes = enum { c, h, w };
    const f = f64;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .c = 3, .h = 2, .w = 2 });
    defer src.deinit(al);
    src.fillArange();

    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .{ .constant = 0 } },
        .w = .{ .before = 1, .after = 1, .mode = .{ .constant = 0 } },
    });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 3), padded.idx.shape.c);
    try std.testing.expectEqual(@as(usize, 4), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 4), padded.idx.shape.w);

    var keys = src.idx.iterKeys();
    while (keys.next()) |k| {
        try expectAt(Axes, f, padded.asConst(), .{ .c = k.c, .h = k.h + 1, .w = k.w + 1 }, src.scalarAt(k));
    }
}

test "pad: 1D padding" {
    const Axes = enum { t };
    const i32_ = i32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, i32_).initAlloc(al, .{ .t = 4 });
    defer src.deinit(al);
    src.fillArange();

    const padded = try pad(Axes, i32_, al, src.asConst(), .{ .t = .{ .before = 2, .after = 0, .mode = .{ .constant = 0 } } });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 6), padded.idx.shape.t);
    try expectAt(Axes, i32_, padded.asConst(), .{ .t = 0 }, 0);
    try expectAt(Axes, i32_, padded.asConst(), .{ .t = 1 }, 0);
    try expectAt(Axes, i32_, padded.asConst(), .{ .t = 2 }, 0);
    try expectAt(Axes, i32_, padded.asConst(), .{ .t = 5 }, 3);
}

test "pad: replicate mode repeats edge values" {
    const Axes = enum { t };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .t = 4 });
    defer src.deinit(al);
    src.fillArange();

    // [0 1 2 3] with before=2, after=3 -> [0 0 0 1 2 3 3 3 3]
    const padded = try pad(Axes, f, al, src.asConst(), .{ .t = .{ .before = 2, .after = 3, .mode = .replicate } });
    defer padded.deinit(al);

    const expected = [_]f32{ 0, 0, 0, 1, 2, 3, 3, 3, 3 };
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try expectAt(Axes, f, padded.asConst(), .{ .t = i }, expected[i]);
    }
}

test "pad: reflect mode mirrors without repeating the edge" {
    const Axes = enum { t };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .t = 4 });
    defer src.deinit(al);
    src.fillArange();

    // [0 1 2 3] with before=2, after=3 -> [2 1 0 1 2 3 2 1 0]
    const padded = try pad(Axes, f, al, src.asConst(), .{ .t = .{ .before = 2, .after = 3, .mode = .reflect } });
    defer padded.deinit(al);

    const expected = [_]f32{ 2, 1, 0, 1, 2, 3, 2, 1, 0 };
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try expectAt(Axes, f, padded.asConst(), .{ .t = i }, expected[i]);
    }
}

test "pad: reflect mode in 2D" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 3, .w = 3 });
    defer src.deinit(al);
    src.fillArange();

    // Source rows/cols [1 0 1 2 1]; values per (h,w) below.
    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .reflect },
        .w = .{ .before = 1, .after = 1, .mode = .reflect },
    });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 5), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 5), padded.idx.shape.w);

    // Corner: (0,0) -> src (1,1) = 4.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 0 }, 4);
    // Top row middle: (0,2) -> src (1,1) = 4.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 2 }, 4);
    // (0,1) -> src (1,0) = 3; (1,0) -> src (0,1) = 1.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 1 }, 3);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 0 }, 1);
    // Interior: (1,1) -> (0,0) = 0; (2,2) -> (1,1) = 4.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 1 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 2 }, 4);
    // Bottom row: (4,2) -> src (1,1) = 4; (4,0) -> (1,1) = 4.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 4, .w = 0 }, 4);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 4, .w = 4 }, 4);
    // Right edge middle: (2,4) -> src (1,1) = 4.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 4 }, 4);
}

test "pad: circular mode wraps around" {
    const Axes = enum { t };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .t = 4 });
    defer src.deinit(al);
    src.fillArange();

    // [0 1 2 3] with before=2, after=3 -> [2 3 0 1 2 3 0 1 2]
    const padded = try pad(Axes, f, al, src.asConst(), .{ .t = .{ .before = 2, .after = 3, .mode = .circular } });
    defer padded.deinit(al);

    const expected = [_]f32{ 2, 3, 0, 1, 2, 3, 0, 1, 2 };
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try expectAt(Axes, f, padded.asConst(), .{ .t = i }, expected[i]);
    }
}

test "pad: circular mode wraps multiple times (pad >= dim)" {
    const Axes = enum { t };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .t = 4 });
    defer src.deinit(al);
    src.fillArange();

    // [0 1 2 3] with before=6, after=5: pads larger than the axis wrap via @mod.
    const padded = try pad(Axes, f, al, src.asConst(), .{ .t = .{ .before = 6, .after = 5, .mode = .circular } });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 15), padded.idx.shape.t);
    const expected = [_]f32{ 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0 };
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try expectAt(Axes, f, padded.asConst(), .{ .t = i }, expected[i]);
    }
}

test "pad: replicate and circular on a size-1 axis" {
    const Axes = enum { t };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .t = 1 });
    defer src.deinit(al);
    src.buf[0] = 5;

    // replicate: clamp to the single edge -> constant.
    const rep = try pad(Axes, f, al, src.asConst(), .{ .t = .{ .before = 2, .after = 2, .mode = .replicate } });
    defer rep.deinit(al);
    const rep_expected = [_]f32{ 5, 5, 5, 5, 5 };
    var i: usize = 0;
    while (i < rep_expected.len) : (i += 1) {
        try expectAt(Axes, f, rep.asConst(), .{ .t = i }, rep_expected[i]);
    }

    // circular: @mod wraps to the single element.
    const circ = try pad(Axes, f, al, src.asConst(), .{ .t = .{ .before = 2, .after = 2, .mode = .circular } });
    defer circ.deinit(al);
    const circ_expected = [_]f32{ 5, 5, 5, 5, 5 };
    var j: usize = 0;
    while (j < circ_expected.len) : (j += 1) {
        try expectAt(Axes, f, circ.asConst(), .{ .t = j }, circ_expected[j]);
    }
}

test "pad: zero-padding axis with non-constant mode" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 3, .w = 3 });
    defer src.deinit(al);
    src.fillArange();

    // w is a zero-padding axis but declares `.reflect`: it imposes no fill
    // constraint and must not affect which path is taken.
    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .replicate },
        .w = .{ .before = 0, .after = 0, .mode = .reflect },
    });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 5), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 3), padded.idx.shape.w);

    // h rows repeat [0 0 1 2 2]; w is unchanged.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 0 }, 0);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 2 }, 2);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 1 }, 1);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 0 }, 3);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 4, .w = 2 }, 8);
}

test "pad: mixed modes per axis" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 3, .w = 3 });
    defer src.deinit(al);
    src.fillArange();

    // h: replicate (rows repeat), w: reflect (cols mirror).
    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .replicate },
        .w = .{ .before = 1, .after = 1, .mode = .reflect },
    });
    defer padded.deinit(al);

    // Row source indices: [0 0 1 2 2]; col source indices: [1 0 1 2 1].
    // (0,0) -> src (0,1) = 1
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 0 }, 1);
    // (0,1) -> (0,0) = 0
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 1 }, 0);
    // (2,2) -> (1,1) = 4
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 2 }, 4);
    // (4,4) -> (2,1) = 7
    try expectAt(Axes, f, padded.asConst(), .{ .h = 4, .w = 4 }, 7);
    // (1,0) -> (0,1) = 1
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 0 }, 1);
    // (3,0) -> (2,1) = 7
    try expectAt(Axes, f, padded.asConst(), .{ .h = 3, .w = 0 }, 7);
}

test "pad: strided input view" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const base = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 4, .w = 4 });
    defer base.deinit(al);
    base.fillArange();

    // Subsampled w-axis view: non-contiguous input.
    const strided = za.NamedArray(Axes, f).init(base.idx.strideAxis(.w, 2), base.buf);

    const padded = try pad(Axes, f, al, strided.asConst(), .{ .h = .{ .before = 1, .after = 1, .mode = .{ .constant = 0 } } });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 6), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 2), padded.idx.shape.w);

    var keys = strided.idx.iterKeys();
    while (keys.next()) |k| {
        try expectAt(Axes, f, padded.asConst(), .{ .h = k.h + 1, .w = k.w }, strided.scalarAt(k));
    }
}

test "pad: constant axis mixed with non-constant mode" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 3, .w = 3 });
    defer src.deinit(al);
    src.fillArange();

    // h: constant fill 7, w: replicate. Padded region of h wins over the
    // replicated value in the overlap; interior follows the source.
    const padded = try pad(Axes, f, al, src.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .{ .constant = 7 } },
        .w = .{ .before = 2, .after = 1, .mode = .replicate },
    });
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 5), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 6), padded.idx.shape.w);

    // (0,0): h padded + w replicated -> constant 7 wins.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 0 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 0, .w = 5 }, 7);
    try expectAt(Axes, f, padded.asConst(), .{ .h = 4, .w = 0 }, 7);
    // (1,0): h interior, w replicated -> src (0,0) = 0.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 1, .w = 0 }, 0);
    // (3,5): h interior, w replicated -> src (2,2) = 8.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 3, .w = 5 }, 8);
    // Interior: (2,2) -> src (1,0) = 3.
    try expectAt(Axes, f, padded.asConst(), .{ .h = 2, .w = 2 }, 3);
}

test "pad: zero padding is a copy" {
    const Axes = enum { h, w };
    const f = f32;
    const al = std.testing.allocator;

    const src = try za.NamedArray(Axes, f).initAlloc(al, .{ .h = 2, .w = 3 });
    defer src.deinit(al);
    src.fillArange();

    const padded = try pad(Axes, f, al, src.asConst(), .{});
    defer padded.deinit(al);

    try std.testing.expectEqual(@as(usize, 2), padded.idx.shape.h);
    try std.testing.expectEqual(@as(usize, 3), padded.idx.shape.w);

    var keys = src.idx.iterKeys();
    while (keys.next()) |k| {
        try expectAt(Axes, f, padded.asConst(), k, src.scalarAt(k));
    }
}
