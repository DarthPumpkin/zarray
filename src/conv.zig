const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const meta = std.meta;
const simd = std.simd;
const testing = std.testing;

const za = @import("root.zig");
const blas = za.bindings.blas;
const axis_meta = za.axis_meta;
const NamedIndex = za.index.NamedIndex;
const Writer = std.Io.Writer;

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

// ============================================================================
// General convolution (ADR-0002): role inference over named axes.
// ============================================================================
//
// Instead of a fixed positional layout (NCHW, NHWC, ...), each axis's role is
// inferred from which of the three operands (`im`, `kernel`, `out`) share its
// tag:
//
//   | present in…            | role        | shape constraint                          |
//   |------------------------|-------------|-------------------------------------------|
//   | im, kernel, out        | spatial     | im = (out-1)*stride + (kernel-1)*dil + 1  |
//   | im, kernel             | in-channel  | im == kernel                              |
//   | kernel, out            | out-channel | kernel == out                             |
//   | im, out                | batch       | im == out                                 |
//   | one operand only       | —           | @compileError                             |
//
// Channels and arbitrary rank are therefore the same mechanism: every distinct
// tag in `im ∩ kernel` is an in-channel, every tag in `kernel ∩ out` an
// out-channel, and every tag in all three operands a spatial axis.

/// Comptime classification of every axis tag of a convolution.
pub const RoleInfo = struct {
    /// Present in all three operands: sliding-window coupling
    /// `im[s*stride + k*dilation]`.
    spatial: []const [:0]const u8,
    /// Present in im and kernel only: contraction (summed away).
    in_channel: []const [:0]const u8,
    /// Present in kernel and out only: free index sourced from kernel.
    out_channel: []const [:0]const u8,
    /// Present in im and out only: shared free index (pass-through).
    batch: []const [:0]const u8,
};

fn contains(comptime names: []const [:0]const u8, comptime name: [:0]const u8) bool {
    for (names) |n| {
        if (mem.eql(u8, n, name)) return true;
    }
    return false;
}

fn roleName(
    comptime name: [:0]const u8,
    comptime im_names: []const [:0]const u8,
    comptime ker_names: []const [:0]const u8,
    comptime out_names: []const [:0]const u8,
) []const u8 {
    const in_im = contains(im_names, name);
    const in_ker = contains(ker_names, name);
    const in_out = contains(out_names, name);
    if (in_im and in_ker and in_out) return "spatial";
    if (in_im and in_ker) return "in-channel";
    if (in_ker and in_out) return "out-channel";
    if (in_im and in_out) return "batch";
    return "orphan";
}

/// Infer the role of every axis tag from its presence across the three
/// operands (ADR-0002 section 1). Axes present in only one operand are
/// compile errors; the message lists the inferred role of every axis so naming
/// mismatches (e.g. kernel axis `kh` vs image axis `h`) are easy to diagnose.
pub fn inferRoles(comptime ImAxis: type, comptime KerAxis: type, comptime OutAxis: type) RoleInfo {
    const union_names = comptime axis_meta.unionOfAxisNames(&.{ ImAxis, KerAxis, OutAxis });
    const im_names = meta.fieldNames(ImAxis);
    const ker_names = meta.fieldNames(KerAxis);
    const out_names = meta.fieldNames(OutAxis);

    comptime var spatial: [union_names.len][:0]const u8 = undefined;
    comptime var in_channel: [union_names.len][:0]const u8 = undefined;
    comptime var out_channel: [union_names.len][:0]const u8 = undefined;
    comptime var batch: [union_names.len][:0]const u8 = undefined;
    comptime var s_c: usize = 0;
    comptime var i_c: usize = 0;
    comptime var o_c: usize = 0;
    comptime var b_c: usize = 0;
    comptime var orphan: ?[:0]const u8 = null;

    for (union_names) |name| {
        const in_im = contains(im_names, name);
        const in_ker = contains(ker_names, name);
        const in_out = contains(out_names, name);
        if (in_im and in_ker and in_out) {
            spatial[s_c] = name;
            s_c += 1;
        } else if (in_im and in_ker) {
            in_channel[i_c] = name;
            i_c += 1;
        } else if (in_ker and in_out) {
            out_channel[o_c] = name;
            o_c += 1;
        } else if (in_im and in_out) {
            batch[b_c] = name;
            b_c += 1;
        } else {
            if (orphan == null) orphan = name;
        }
    }

    if (comptime orphan) |orphan_name| {
        var buf: [2048]u8 = undefined;
        var w: Writer = .fixed(&buf);
        w.print("conv: axis '{s}' occurs in only one operand and cannot be assigned a role. Inferred roles of every axis:\n", .{orphan_name}) catch unreachable;
        for (union_names) |n| {
            w.print("  {s}: {s}\n", .{ n, roleName(n, im_names, ker_names, out_names) }) catch unreachable;
        }
        w.print("Every axis must be shared by two or three operands. Valid roles: spatial (im+kernel+out), in-channel (im+kernel), out-channel (kernel+out), batch (im+out).", .{}) catch unreachable;
        @compileError(buf[0..w.end]);
    }

    // Copy the comptime-var staging arrays into comptime consts so the
    // returned slices are usable outside comptime-var scope.
    const spatial_arr: [s_c][:0]const u8 = spatial[0..s_c].*;
    const in_channel_arr: [i_c][:0]const u8 = in_channel[0..i_c].*;
    const out_channel_arr: [o_c][:0]const u8 = out_channel[0..o_c].*;
    const batch_arr: [b_c][:0]const u8 = batch[0..b_c].*;

    return .{
        .spatial = &spatial_arr,
        .in_channel = &in_channel_arr,
        .out_channel = &out_channel_arr,
        .batch = &batch_arr,
    };
}

/// Per-spatial-axis stride/dilation specification for `conv`. Keyed by the
/// spatial axis names, so specifying a non-spatial axis fails to compile.
pub fn ConvParams(comptime ImAxis: type, comptime KerAxis: type, comptime OutAxis: type) type {
    const roles = comptime inferRoles(ImAxis, KerAxis, OutAxis);
    const SpatialSpec = axis_meta.AxesOptionalStructOf(roles.spatial, usize);
    return struct {
        /// Output stride per spatial axis; absent axes default to 1.
        stride: SpatialSpec = .{},
        /// Kernel dilation per spatial axis; absent axes default to 1.
        dilation: SpatialSpec = .{},
    };
}

/// Generic convolution of `im` with `kernel` into `out`, valid-mode only.
///
/// Axis roles are inferred at compile time from the operand axis enums (see
/// `inferRoles`), so the operand layouts (NCHW, NHWC, …) need not match any
/// fixed convention — axes communicate by shared tag names. Shape constraints
/// are validated per role (fail fast), and orphan axes are compile errors.
///
/// The engine is layout-agnostic and allocation-free; layout-specific fast
/// paths are added behind dispatch as profiling dictates (ADR-0002 section 5).
pub fn conv(
    comptime Scalar: type,
    comptime ImAxis: type,
    comptime KerAxis: type,
    comptime OutAxis: type,
    im: za.NamedArrayConst(ImAxis, Scalar),
    kernel: za.NamedArrayConst(KerAxis, Scalar),
    out: za.NamedArray(OutAxis, Scalar),
    params: ConvParams(ImAxis, KerAxis, OutAxis),
) void {
    const roles = comptime inferRoles(ImAxis, KerAxis, OutAxis);

    inline for (meta.fields(@TypeOf(params.stride))) |f| {
        const stride: usize = @field(params.stride, f.name) orelse 1;
        const dilation: usize = @field(params.dilation, f.name) orelse 1;
        assert(stride >= 1);
        assert(dilation >= 1);
        const im_n = @field(im.idx.shape, f.name);
        const k_n = @field(kernel.idx.shape, f.name);
        const o_n = @field(out.idx.shape, f.name);
        assert(im_n == (o_n - 1) * stride + (k_n - 1) * dilation + 1);
    }
    inline for (meta.fields(NamedIndex(ImAxis).Axes)) |f| {
        if (comptime contains(roles.in_channel, f.name)) {
            assert(@field(im.idx.shape, f.name) == @field(kernel.idx.shape, f.name));
        }
    }
    inline for (meta.fields(NamedIndex(OutAxis).Axes)) |f| {
        if (comptime contains(roles.out_channel, f.name)) {
            assert(@field(kernel.idx.shape, f.name) == @field(out.idx.shape, f.name));
        }
    }
    inline for (meta.fields(NamedIndex(ImAxis).Axes)) |f| {
        if (comptime contains(roles.batch, f.name)) {
            assert(@field(im.idx.shape, f.name) == @field(out.idx.shape, f.name));
        }
    }

    convGeneric(Scalar, ImAxis, KerAxis, OutAxis, roles, im, kernel, out, params);
}

/// Kernel axes that participate in the reduction: in-channels plus kernel
/// spatial axes (out-channels are free axes fixed per output position).
fn reducedKernelAxis(comptime KerAxis: type, comptime roles: RoleInfo) type {
    const ker_names = meta.fieldNames(KerAxis);
    comptime var red: [ker_names.len][:0]const u8 = undefined;
    comptime var n: usize = 0;
    for (ker_names) |name| {
        if (contains(roles.in_channel, name) or contains(roles.spatial, name)) {
            red[n] = name;
            n += 1;
        }
    }
    return axis_meta.KeyEnum(red[0..n]);
}

/// Position for every kernel axis dropped by the reduction: the out-channel
/// indices, taken from the current output key.
fn kernelIndices(comptime KerAxis: type, comptime RedKer: type, ok: anytype) axis_meta.DifferenceAxesStruct(KerAxis, RedKer) {
    var idx: axis_meta.DifferenceAxesStruct(KerAxis, RedKer) = undefined;
    inline for (meta.fields(axis_meta.DifferenceAxesStruct(KerAxis, RedKer))) |f| {
        @field(idx, f.name) = @field(ok, f.name);
    }
    return idx;
}

/// Rebuild a full kernel key from a reduced kernel key plus the output key
/// (out-channel fields come from `ok`).
fn kernelKey(comptime KerAxis: type, comptime roles: RoleInfo, ok: anytype, kk: anytype) NamedIndex(KerAxis).Axes {
    var key: NamedIndex(KerAxis).Axes = undefined;
    inline for (meta.fields(NamedIndex(KerAxis).Axes)) |f| {
        if (comptime contains(roles.in_channel, f.name) or contains(roles.spatial, f.name)) {
            @field(key, f.name) = @field(kk, f.name);
        } else {
            @field(key, f.name) = @field(ok, f.name);
        }
    }
    return key;
}

/// Image key for one (output position, kernel tap) pair. Batch axes pass the
/// output position through, in-channels come from the tap, and spatial axes
/// couple the two with the sliding-window formula `im = ok*stride + kk*dilation`.
fn imKey(
    comptime ImAxis: type,
    comptime roles: RoleInfo,
    ok: anytype,
    kk: anytype,
    stride_spec: anytype,
    dilation_spec: anytype,
) NamedIndex(ImAxis).Axes {
    var key: NamedIndex(ImAxis).Axes = undefined;
    inline for (meta.fields(NamedIndex(ImAxis).Axes)) |f| {
        if (comptime contains(roles.batch, f.name)) {
            @field(key, f.name) = @field(ok, f.name);
        } else if (comptime contains(roles.in_channel, f.name)) {
            @field(key, f.name) = @field(kk, f.name);
        } else {
            const stride: usize = @field(stride_spec, f.name) orelse 1;
            const dilation: usize = @field(dilation_spec, f.name) orelse 1;
            @field(key, f.name) = @field(ok, f.name) * stride + @field(kk, f.name) * dilation;
        }
    }
    return key;
}

/// Layout-agnostic engine: outer loop over output keys, inner reduction over
/// (in-channel, kernel-spatial) taps. Any stride pattern is handled through
/// the named-index machinery.
fn convGeneric(
    comptime Scalar: type,
    comptime ImAxis: type,
    comptime KerAxis: type,
    comptime OutAxis: type,
    comptime roles: RoleInfo,
    im: za.NamedArrayConst(ImAxis, Scalar),
    kernel: za.NamedArrayConst(KerAxis, Scalar),
    out: za.NamedArray(OutAxis, Scalar),
    params: anytype,
) void {
    const RedKer = comptime reducedKernelAxis(KerAxis, roles);

    var out_iter = out.idx.iterKeys();
    while (out_iter.next()) |ok| {
        var acc: Scalar = 0;
        const k_idx = kernel.idx.indexAxes(RedKer, kernelIndices(KerAxis, RedKer, ok));
        var k_iter = k_idx.iterKeys();
        while (k_iter.next()) |kk| {
            const im_key = imKey(ImAxis, roles, ok, kk, params.stride, params.dilation);
            acc += kernel.scalarAt(kernelKey(KerAxis, roles, ok, kk)) * im.scalarAt(im_key);
        }
        out.at(ok).* = acc;
    }
}

/// Naive reference for tests: same semantics as `conv`, iterating the full
/// kernel without any fast paths.
fn convReference(
    comptime Scalar: type,
    comptime ImAxis: type,
    comptime KerAxis: type,
    comptime OutAxis: type,
    im: za.NamedArrayConst(ImAxis, Scalar),
    kernel: za.NamedArrayConst(KerAxis, Scalar),
    out: za.NamedArray(OutAxis, Scalar),
    params: anytype,
) void {
    const roles = comptime inferRoles(ImAxis, KerAxis, OutAxis);

    var out_iter = out.idx.iterKeys();
    while (out_iter.next()) |ok| {
        var acc: Scalar = 0;
        var k_iter = kernel.idx.iterKeys();
        while (k_iter.next()) |kk| {
            // Skip taps whose out-channel position differs from this output
            // (out-channel axes are free indices sourced from the kernel).
            var matches = true;
            inline for (meta.fields(NamedIndex(KerAxis).Axes)) |f| {
                if (comptime contains(roles.out_channel, f.name)) {
                    if (@field(kk, f.name) != @field(ok, f.name)) matches = false;
                }
            }
            if (!matches) continue;
            const im_key = imKey(ImAxis, roles, ok, kk, params.stride, params.dilation);
            acc += kernel.scalarAt(kk) * im.scalarAt(im_key);
        }
        out.at(ok).* = acc;
    }
}

fn expectSame(
    comptime Scalar: type,
    expected: anytype,
    actual: anytype,
) !void {
    try testing.expectEqual(expected.idx.shape, actual.idx.shape);
    var keys = expected.idx.iterKeys();
    while (keys.next()) |key| {
        switch (@typeInfo(Scalar)) {
            .float => try testing.expectApproxEqAbs(expected.scalarAt(key), actual.scalarAt(key), @as(Scalar, 1e-4)),
            else => try testing.expectEqual(expected.scalarAt(key), actual.scalarAt(key)),
        }
    }
}

fn runConvAndCheckGeneric(
    comptime Scalar: type,
    comptime ImAxis: type,
    comptime KerAxis: type,
    comptime OutAxis: type,
    allocator: std.mem.Allocator,
    im: za.NamedArrayConst(ImAxis, Scalar),
    kernel: za.NamedArrayConst(KerAxis, Scalar),
    out: za.NamedArray(OutAxis, Scalar),
    params: anytype,
) !void {
    const expected = try za.NamedArray(OutAxis, Scalar).initAlloc(allocator, out.idx.shape);
    defer expected.deinit(allocator);

    convReference(Scalar, ImAxis, KerAxis, OutAxis, im, kernel, expected, params);
    conv(Scalar, ImAxis, KerAxis, OutAxis, im, kernel, out, params);

    try expectSame(Scalar, expected.asConst(), out.asConst());
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

test "conv: 2D single channel matches reference" {
    const ImA = enum { h, w };
    const KerA = enum { h, w };
    const OutA = enum { h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .h = 8, .w = 9 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .h = 3, .w = 3 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .h = 6, .w = 7 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
}

test "conv: 2D with in- and out-channels" {
    const ImA = enum { ci, h, w };
    const KerA = enum { ci, co, h, w };
    const OutA = enum { co, h, w };
    const f = f64;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .ci = 4, .h = 7, .w = 6 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 4, .co = 2, .h = 3, .w = 3 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .co = 2, .h = 5, .w = 4 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
}

test "conv: batch axis passes through" {
    const ImA = enum { b, ci, h, w };
    const KerA = enum { ci, co, h, w };
    const OutA = enum { b, co, h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .b = 3, .ci = 2, .h = 5, .w = 5 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 2, .co = 3, .h = 3, .w = 3 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .b = 3, .co = 3, .h = 3, .w = 3 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
}

test "conv: strided output (stride 2 on both spatial axes)" {
    const ImA = enum { ci, h, w };
    const KerA = enum { ci, co, h, w };
    const OutA = enum { co, h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .ci = 2, .h = 11, .w = 9 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 2, .co = 2, .h = 3, .w = 3 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .co = 2, .h = 5, .w = 4 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{ .stride = .{ .h = 2, .w = 2 } });
}

test "conv: dilated kernel (dilation 2)" {
    const ImA = enum { h, w };
    const KerA = enum { h, w };
    const OutA = enum { h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .h = 9, .w = 9 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .h = 3, .w = 3 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .h = 5, .w = 5 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{ .dilation = .{ .h = 2, .w = 2 } });
}

test "conv: 1D (single spatial axis) with channels" {
    const ImA = enum { ci, t };
    const KerA = enum { ci, co, t };
    const OutA = enum { co, t };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .ci = 3, .t = 8 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 3, .co = 2, .t = 4 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .co = 2, .t = 5 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
}

test "conv: 3D spatial (three shared axes)" {
    const ImA = enum { ci, d, h, w };
    const KerA = enum { ci, co, d, h, w };
    const OutA = enum { co, d, h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .ci = 2, .d = 4, .h = 4, .w = 4 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 2, .co = 2, .d = 2, .h = 2, .w = 2 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .co = 2, .d = 3, .h = 3, .w = 3 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
}

test "conv: non-contiguous (column-major) image and kernel" {
    const ImA = enum { h, w };
    const KerA = enum { h, w };
    const OutA = enum { h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im_buf = try al.alloc(f, 8 * 9);
    defer al.free(im_buf);
    const im = za.NamedArray(ImA, f).init(.{
        .shape = .{ .h = 8, .w = 9 },
        .strides = .{ .h = 1, .w = 8 }, // h fastest (column-major)
        .offset = 0,
    }, im_buf);
    var im_i: usize = 0;
    for (im_buf) |*v| {
        v.* = @floatFromInt(im_i);
        im_i += 1;
    }

    const kernel_buf = try al.alloc(f, 3 * 3);
    defer al.free(kernel_buf);
    const kernel = za.NamedArray(KerA, f).init(.{
        .shape = .{ .h = 3, .w = 3 },
        .strides = .{ .h = 1, .w = 3 }, // h fastest
        .offset = 0,
    }, kernel_buf);
    var k_i: usize = 0;
    for (kernel_buf) |*v| {
        v.* = @floatFromInt(k_i);
        k_i += 1;
    }

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .h = 6, .w = 7 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
}

test "conv: pad-then-conv gives same-padding" {
    const ImA = enum { ci, h, w };
    const KerA = enum { ci, co, h, w };
    const OutA = enum { co, h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .ci = 2, .h = 5, .w = 5 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 2, .co = 2, .h = 3, .w = 3 });
    defer kernel.deinit(al);
    kernel.fillArange();

    // Same-padding via the composable pad op: padded im keeps the output the
    // same size as the original input.
    const padded_im = try za.pad.pad(ImA, f, al, im.asConst(), .{
        .h = .{ .before = 1, .after = 1, .mode = .{ .constant = 0 } },
        .w = .{ .before = 1, .after = 1, .mode = .{ .constant = 0 } },
    });
    defer padded_im.deinit(al);

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .co = 2, .h = 5, .w = 5 });
    defer out.deinit(al);

    const expected = try za.NamedArray(OutA, f).initAlloc(al, out.idx.shape);
    defer expected.deinit(al);
    convReference(f, ImA, KerA, OutA, padded_im.asConst(), kernel.asConst(), expected, P{});
    conv(f, ImA, KerA, OutA, padded_im.asConst(), kernel.asConst(), out, P{});

    try expectSame(f, expected.asConst(), out.asConst());
}

test "conv: 1x1 kernel degenerates to channel contraction (gemm-like)" {
    const ImA = enum { ci, h, w };
    const KerA = enum { ci, co, h, w };
    const OutA = enum { co, h, w };
    const f = f32;
    const al = testing.allocator;
    const P = ConvParams(ImA, KerA, OutA);

    const im = try za.NamedArray(ImA, f).initAlloc(al, .{ .ci = 4, .h = 3, .w = 3 });
    defer im.deinit(al);
    im.fillArange();

    const kernel = try za.NamedArray(KerA, f).initAlloc(al, .{ .ci = 4, .co = 3, .h = 1, .w = 1 });
    defer kernel.deinit(al);
    kernel.fillArange();

    const out = try za.NamedArray(OutA, f).initAlloc(al, .{ .co = 3, .h = 3, .w = 3 });
    defer out.deinit(al);

    try runConvAndCheckGeneric(f, ImA, KerA, OutA, al, im.asConst(), kernel.asConst(), out, P{});
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
