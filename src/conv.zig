const std = @import("std");
const assert = std.debug.assert;
const simd = std.simd;
const testing = std.testing;

const za = @import("root.zig");
const blas = za.bindings.blas;

pub const HW = enum { h, w };
pub const Row = enum { w };

fn swapHWIndex(idx: za.index.NamedIndex(HW)) za.index.NamedIndex(HW) {
    return .{
        .shape = .{ .h = idx.shape.w, .w = idx.shape.h },
        .strides = .{ .h = idx.strides.w, .w = idx.strides.h },
        .offset = idx.offset,
    };
}

fn swapHWConst(comptime Scalar: type, arr: za.NamedArrayConst(HW, Scalar)) za.NamedArrayConst(HW, Scalar) {
    return za.NamedArrayConst(HW, Scalar).init(swapHWIndex(arr.idx), arr.buf);
}

fn swapHW(comptime Scalar: type, arr: za.NamedArray(HW, Scalar)) za.NamedArray(HW, Scalar) {
    return za.NamedArray(HW, Scalar).init(swapHWIndex(arr.idx), arr.buf);
}

/// Single-channel 2D convolution of `im` with `kernel`.
/// No padding, no allocations, no restrictions on layout.
pub fn conv2dSingleChannel(
    comptime Scalar: type,
    im: za.NamedArrayConst(HW, Scalar),
    kernel: za.NamedArrayConst(HW, Scalar),
    // memory: []Scalar,
    out: za.NamedArray(HW, Scalar),
) void {
    assert(im.idx.shape.h == out.idx.shape.h + kernel.idx.shape.h - 1);
    assert(im.idx.shape.w == out.idx.shape.w + kernel.idx.shape.w - 1);

    if (kernel.idx.strides.w == 1 and im.idx.strides.w == 1) {
        if (out.idx.strides.w == 1) {
            convRowContiguous(Scalar, im, kernel, out);
            return;
        }

        // `out` is not w-contiguous: fall back to per-pixel dot products.
        for (0..out.idx.shape.h) |oh| {
            for (0..out.idx.shape.w) |ow| {
                var out_i: Scalar = 0;
                for (0..kernel.idx.shape.h) |kh| {
                    const kernel_row_flat = kernel.indexAxes(Row, .{ .h = kh }).flatUnsafe();
                    var im_row = im.indexAxes(Row, .{ .h = oh + kh });
                    im_row.idx.sliceAxisInplace(.w, ow, ow + kernel.idx.shape.w);
                    const im_row_flat = im_row.flatUnsafe();
                    out_i += za.math.dotFast(Scalar, kernel_row_flat, im_row_flat);
                }
                out.at(.{ .h = oh, .w = ow }).* = out_i;
            }
        }
        return;
    }

    if (kernel.idx.strides.h == 1 and im.idx.strides.h == 1) {
        const kernel_r = swapHWConst(Scalar, kernel);
        const im_r = swapHWConst(Scalar, im);
        const out_r = swapHW(Scalar, out);
        conv2dSingleChannel(Scalar, im_r, kernel_r, out_r);
        return;
    }

    // const cont_axis: HW = if (kernel.idx.strides.h == 1) .h else .w;
    // assert(@field(kernel.idx.strides, @tagName(cont_axis)) == 1);

    // var fba_ = std.heap.FixedBufferAllocator.init(memory);
    var iter_out = out.idx.iterKeys();
    while (iter_out.next()) |idx| {
        var out_i: Scalar = 0;
        // var iter_kern = kernel.idx.iterKeys();
        // while (iter_kern.next()) |ki|
        //     out_i += kernel.scalarAt(ki) * im.scalarAt(.{
        //         .h = idx.h + ki.h,
        //         .w = idx.w + ki.w,
        //     });
        for (0..kernel.idx.shape.h) |i| {
            const kernel_row = kernel.indexAxes(Row, .{ .h = i });
            var im_row = im.indexAxes(Row, .{ .h = idx.h + i });
            im_row.idx = im_row.idx.sliceAxis(.w, idx.w, idx.w + kernel.idx.shape.w);
            out_i += blas.dot(Scalar, Row, im_row, kernel_row, .{});
        }
        out.at(idx).* = out_i;

        // const block_cont = za.NamedArray(HW, Scalar).init(.{
        //     .shape = im_block.idx.shape,
        //     .strides = .{
        //         .h = if (cont_axis == .h) 1 else im_block.idx.shape.w,
        //         .w = if (cont_axis == .w) 1 else im_block.idx.shape.h,
        //     },
        // }, memory);
        // block_cont.fillCopy(im_block);

        // const fba = fba_.allocator();
        // const block_cont = try im_block.toContiguous(fba);
        // assert(im_block.idx.strides.h == kernel.idx.strides.h || )
    }
}

/// Fused multiply-add `a * b + c` for scalars or vectors. Uses a real FMA
/// (single rounding) for float element types and a plain mul+add for integers,
/// where `@mulAdd` is not available.
inline fn fma(comptime V: type, a: V, b: V, c: V) V {
    const Elem = switch (@typeInfo(V)) {
        .vector => |v| v.child,
        else => V,
    };
    return if (@typeInfo(Elem) == .float) @mulAdd(V, a, b, c) else a * b + c;
}

/// Fast path for the fully w-contiguous case (`im`, `kernel` and `out` all have
/// w-stride 1).
///
/// Dispatches common kernel sizes to a specialization with `k_h`/`k_w` known at
/// compile time so the tap loops fully unroll and kernel row offsets fold to
/// constants; all other sizes use a generic (runtime-bounds) instantiation.
fn convRowContiguous(
    comptime Scalar: type,
    im: za.NamedArrayConst(HW, Scalar),
    kernel: za.NamedArrayConst(HW, Scalar),
    out: za.NamedArray(HW, Scalar),
) void {
    const k_h = kernel.idx.shape.h;
    const k_w = kernel.idx.shape.w;
    if (k_h == k_w) {
        switch (k_h) {
            3 => return convRowContiguousImpl(Scalar, 3, 3, im, kernel, out),
            5 => return convRowContiguousImpl(Scalar, 5, 5, im, kernel, out),
            7 => return convRowContiguousImpl(Scalar, 7, 7, im, kernel, out),
            else => {},
        }
    }
    convRowContiguousImpl(Scalar, null, null, im, kernel, out);
}

/// Register-blocked w-contiguous convolution.
///
/// We tile the output row into blocks of SIMD columns and keep the accumulators
/// in registers while looping over every kernel tap, storing each result exactly
/// once. Compared to doing one broadcast multiply-add pass over the whole output
/// row per tap, this removes the `k_h * k_w` read-modify-write passes over the
/// output row (49 of them for a 7x7 kernel) and, by using several independent
/// accumulators, breaks the FP-add latency chain for more instruction-level
/// parallelism.
///
/// `KH`/`KW`, when non-null, pin the kernel height/width at compile time so the
/// tap loops unroll fully. When null the corresponding extent is taken from the
/// runtime shape.
///
/// The per-output-element accumulation order (kh-major, kw-minor) matches the
/// straightforward implementation.
fn convRowContiguousImpl(
    comptime Scalar: type,
    comptime KH: ?usize,
    comptime KW: ?usize,
    im: za.NamedArrayConst(HW, Scalar),
    kernel: za.NamedArrayConst(HW, Scalar),
    out: za.NamedArray(HW, Scalar),
) void {
    const out_h = out.idx.shape.h;
    const out_w = out.idx.shape.w;
    const k_h: usize = KH orelse kernel.idx.shape.h;
    const k_w: usize = KW orelse kernel.idx.shape.w;

    // Vector width, or 1 to signal "no SIMD" for non-vectorizable scalars.
    const lanes: usize = comptime blk: {
        if (simd.suggestVectorLength(Scalar)) |l| {
            if (l >= 2) break :blk l;
        }
        break :blk 1;
    };
    // Number of independent accumulator vectors (unroll factor over columns).
    const unroll = 4;

    for (0..out_h) |oh| {
        const out_row = out.indexAxes(Row, .{ .h = oh }).flatUnsafe();

        var ow: usize = 0;

        if (comptime lanes >= 2) {
            const Vec = @Vector(lanes, Scalar);
            const block = lanes * unroll;

            // Wide blocks: `unroll` accumulator vectors live in registers.
            while (ow + block <= out_w) : (ow += block) {
                var acc = [_]Vec{@splat(0)} ** unroll;
                for (0..k_h) |kh| {
                    const krow = kernel.indexAxes(Row, .{ .h = kh }).flatUnsafe();
                    const irow = im.indexAxes(Row, .{ .h = oh + kh }).flatUnsafe();
                    for (0..k_w) |kw| {
                        const kv: Vec = @splat(krow[kw]);
                        inline for (0..unroll) |u| {
                            const chunk: *const [lanes]Scalar = @ptrCast(irow.ptr + ow + kw + u * lanes);
                            acc[u] = fma(Vec, kv, chunk.*, acc[u]);
                        }
                    }
                }
                inline for (0..unroll) |u| {
                    const dst: *[lanes]Scalar = @ptrCast(out_row.ptr + ow + u * lanes);
                    dst.* = acc[u];
                }
            }

            // Remaining single vectors.
            while (ow + lanes <= out_w) : (ow += lanes) {
                var acc: Vec = @splat(0);
                for (0..k_h) |kh| {
                    const krow = kernel.indexAxes(Row, .{ .h = kh }).flatUnsafe();
                    const irow = im.indexAxes(Row, .{ .h = oh + kh }).flatUnsafe();
                    for (0..k_w) |kw| {
                        const kv: Vec = @splat(krow[kw]);
                        const chunk: *const [lanes]Scalar = @ptrCast(irow.ptr + ow + kw);
                        acc = fma(Vec, kv, chunk.*, acc);
                    }
                }
                const dst: *[lanes]Scalar = @ptrCast(out_row.ptr + ow);
                dst.* = acc;
            }
        }

        // Scalar tail (and the whole row when SIMD is unavailable).
        while (ow < out_w) : (ow += 1) {
            var acc: Scalar = 0;
            for (0..k_h) |kh| {
                const krow = kernel.indexAxes(Row, .{ .h = kh }).flatUnsafe();
                const irow = im.indexAxes(Row, .{ .h = oh + kh }).flatUnsafe();
                for (0..k_w) |kw| {
                    acc = fma(Scalar, krow[kw], irow[ow + kw], acc);
                }
            }
            out_row[ow] = acc;
        }
    }
}

fn conv2dReference(
    comptime Scalar: type,
    im: za.NamedArrayConst(HW, Scalar),
    kernel: za.NamedArrayConst(HW, Scalar),
    out: za.NamedArray(HW, Scalar),
) void {
    assert(im.idx.shape.h == out.idx.shape.h + kernel.idx.shape.h - 1);
    assert(im.idx.shape.w == out.idx.shape.w + kernel.idx.shape.w - 1);

    for (0..out.idx.shape.h) |oh| {
        for (0..out.idx.shape.w) |ow| {
            var acc: Scalar = 0;
            for (0..kernel.idx.shape.h) |kh| {
                for (0..kernel.idx.shape.w) |kw| {
                    acc += kernel.scalarAt(.{ .h = kh, .w = kw }) * im.scalarAt(.{ .h = oh + kh, .w = ow + kw });
                }
            }
            out.at(.{ .h = oh, .w = ow }).* = acc;
        }
    }
}

fn expectSame2d(
    comptime Scalar: type,
    expected: za.NamedArrayConst(HW, Scalar),
    actual: za.NamedArrayConst(HW, Scalar),
) !void {
    try testing.expectEqual(expected.idx.shape.h, actual.idx.shape.h);
    try testing.expectEqual(expected.idx.shape.w, actual.idx.shape.w);

    var keys = expected.idx.iterKeys();
    while (keys.next()) |key| {
        switch (@typeInfo(Scalar)) {
            .float => try testing.expectApproxEqAbs(expected.scalarAt(key), actual.scalarAt(key), @as(Scalar, 1e-4)),
            else => try testing.expectEqual(expected.scalarAt(key), actual.scalarAt(key)),
        }
    }
}

fn runConvAndCheck(
    comptime Scalar: type,
    allocator: std.mem.Allocator,
    im: za.NamedArrayConst(HW, Scalar),
    kernel: za.NamedArrayConst(HW, Scalar),
    out: za.NamedArray(HW, Scalar),
) !void {
    const expected = try za.NamedArray(HW, Scalar).initAlloc(allocator, out.idx.shape);
    defer expected.deinit(allocator);

    conv2dReference(Scalar, im, kernel, expected);
    conv2dSingleChannel(Scalar, im, kernel, out);

    try expectSame2d(Scalar, expected.asConst(), out.asConst());
}

test "conv2dSingleChannel contiguous path matches reference (specialized + generic kernels)" {
    const f = f32;
    const al = testing.allocator;

    const cases = [_]struct {
        kh: usize,
        kw: usize,
        out_h: usize,
        out_w: usize,
    }{
        .{ .kh = 3, .kw = 3, .out_h = 5, .out_w = 19 }, // 3x3 specialization
        .{ .kh = 5, .kw = 5, .out_h = 4, .out_w = 18 }, // 5x5 specialization
        .{ .kh = 7, .kw = 7, .out_h = 3, .out_w = 17 }, // 7x7 specialization
        .{ .kh = 3, .kw = 4, .out_h = 5, .out_w = 21 }, // generic runtime path
    };

    for (cases) |c| {
        const kernel = try za.NamedArray(HW, f).initAlloc(al, .{ .h = c.kh, .w = c.kw });
        defer kernel.deinit(al);
        kernel.fillArange();

        const im = try za.NamedArray(HW, f).initAlloc(al, .{
            .h = c.out_h + c.kh - 1,
            .w = c.out_w + c.kw - 1,
        });
        defer im.deinit(al);
        im.fillArange();

        const out = try za.NamedArray(HW, f).initAlloc(al, .{ .h = c.out_h, .w = c.out_w });
        defer out.deinit(al);

        try runConvAndCheck(f, al, im.asConst(), kernel.asConst(), out);
    }
}

test "conv2dSingleChannel w-contiguous fallback with non-contiguous out" {
    const f = f32;
    const al = testing.allocator;

    const kernel = try za.NamedArray(HW, f).initAlloc(al, .{ .h = 3, .w = 4 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const im = try za.NamedArray(HW, f).initAlloc(al, .{ .h = 7, .w = 12 });
    defer im.deinit(al);
    im.fillArange();

    const out_storage = try za.NamedArray(HW, f).initAlloc(al, .{ .h = 5, .w = 18 });
    defer out_storage.deinit(al);
    out_storage.fill(-777);

    const out = za.NamedArray(HW, f).init(out_storage.idx.strideAxis(.w, 2), out_storage.buf);

    try testing.expectEqual(@as(isize, 1), im.idx.strides.w);
    try testing.expectEqual(@as(isize, 1), kernel.idx.strides.w);
    try testing.expect(out.idx.strides.w != 1);

    try runConvAndCheck(f, al, im.asConst(), kernel.asConst(), out);
}

test "conv2dSingleChannel h-contiguous layout path (axis-rename recursion)" {
    const WH = enum { w, h };

    const f = f32;
    const al = testing.allocator;

    const kernel_wh = try za.NamedArray(WH, f).initAlloc(al, .{ .w = 3, .h = 2 });
    defer kernel_wh.deinit(al);
    kernel_wh.fillArange();
    const kernel = kernel_wh.conformAxes(HW);

    const im_wh = try za.NamedArray(WH, f).initAlloc(al, .{ .w = 11, .h = 8 });
    defer im_wh.deinit(al);
    im_wh.fillArange();
    const im = im_wh.conformAxes(HW);

    const out_h = im.idx.shape.h - kernel.idx.shape.h + 1;
    const out_w = im.idx.shape.w - kernel.idx.shape.w + 1;

    const out_wh = try za.NamedArray(WH, f).initAlloc(al, .{ .w = out_w, .h = out_h });
    defer out_wh.deinit(al);
    const out = out_wh.conformAxes(HW);

    try testing.expect(im.idx.strides.w != 1);
    try testing.expect(kernel.idx.strides.w != 1);
    try testing.expectEqual(@as(isize, 1), im.idx.strides.h);
    try testing.expectEqual(@as(isize, 1), kernel.idx.strides.h);

    try runConvAndCheck(f, al, im.asConst(), kernel.asConst(), out);
}

test "conv2dSingleChannel generic fallback with non-unit strides on im and kernel" {
    const f = f32;
    const al = testing.allocator;

    const kernel_base = try za.NamedArray(HW, f).initAlloc(al, .{ .h = 5, .w = 5 });
    defer kernel_base.deinit(al);
    kernel_base.fillArange();
    const kernel = za.NamedArray(HW, f).init(kernel_base.idx.strideAxis(.h, 2).strideAxis(.w, 2), kernel_base.buf);

    const im_base = try za.NamedArray(HW, f).initAlloc(al, .{ .h = 7, .w = 9 });
    defer im_base.deinit(al);
    im_base.fillArange();
    const im = za.NamedArray(HW, f).init(im_base.idx.strideAxis(.h, 2).strideAxis(.w, 2), im_base.buf);

    const out = try za.NamedArray(HW, f).initAlloc(al, .{
        .h = im.idx.shape.h - kernel.idx.shape.h + 1,
        .w = im.idx.shape.w - kernel.idx.shape.w + 1,
    });
    defer out.deinit(al);

    try testing.expect(im.idx.strides.h != 1 and im.idx.strides.w != 1);
    try testing.expect(kernel.idx.strides.h != 1 and kernel.idx.strides.w != 1);

    try runConvAndCheck(f, al, im.asConst(), kernel.asConst(), out);
}
