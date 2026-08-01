const std = @import("std");
const za = @import("zarray");

const C = @cImport({
    @cInclude("stdio.h");
    @cInclude("time.h");
});

const blas = za.bindings.blas;
const tblis = za.bindings.tblis;
const conv = za.conv;

const T = f32;
const SIMD_LANES: comptime_int = std.simd.suggestVectorLength(T) orelse 8;
const SimdVec = @Vector(SIMD_LANES, T);

const AxisN = enum { n };
const AxisI = enum { i };
const AxisJ = enum { j };
const AxisK = enum { k };
const AxisIJ = enum { i, j };
const AxisJK = enum { j, k };
const AxisIK = enum { i, k };

// Dedicated axis sets for GEMM to avoid overlap with internal temporary IJ
// renaming done by the BLAS wrapper.
const AxisMK = enum { m, k };
const AxisKN = enum { k, n };
const AxisMN = enum { m, n };

const VecN = za.NamedArray(AxisN, T);
const VecI = za.NamedArray(AxisI, T);
const VecJ = za.NamedArray(AxisJ, T);
const MatIJ = za.NamedArray(AxisIJ, T);
const MatJK = za.NamedArray(AxisJK, T);
const MatIK = za.NamedArray(AxisIK, T);
const MatMK = za.NamedArray(AxisMK, T);
const MatKN = za.NamedArray(AxisKN, T);
const MatMN = za.NamedArray(AxisMN, T);
const MatHW = za.NamedArray(conv.HW, T);

const BenchTimer = struct {
    start_ns: u64,

    fn start() !@This() {
        return .{ .start_ns = monotonicNs() };
    }

    fn reset(self: *@This()) void {
        self.start_ns = monotonicNs();
    }

    fn read(self: @This()) u64 {
        return monotonicNs() - self.start_ns;
    }
};

const BenchResult = struct {
    ns_total: u64,
    iters: usize,
    checksum: f64,

    fn nsPerIter(self: @This()) f64 {
        return @as(f64, @floatFromInt(self.ns_total)) / @as(f64, @floatFromInt(self.iters));
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len > 1 and std.mem.eql(u8, args[1], "conv")) {
        try outPrint("conv2dSingleChannel benchmark (f32)\n", .{});
        try outPrint("==================================\n", .{});
        try outPrint("\n", .{});
        try outPrint("Includes mixed-layout cases:\n", .{});
        try outPrint("  - row-major vs column-major combinations\n", .{});
        try outPrint("  - one-dimension-contiguous padded layouts\n", .{});
        try outPrint("\n", .{});
        try outPrint("Run with ReleaseFast for meaningful numbers:\n", .{});
        try outPrint("  zig build -Doptimize=ReleaseFast bench -- conv\n", .{});
        try outPrint("\n", .{});
        try benchConv2dSingleChannel(allocator);
        return;
    }

    try outPrint("TBLIS vs CBLAS vs pure Zig benchmark (f32)\n", .{});
    try outPrint("===========================================\n", .{});
    try outPrint("\n", .{});
    try outPrint("Focus mode: GEMM only\n", .{});
    try outPrint("  cblas_sgemm <-> tblis.mult(A_mk, B_kn -> C_mn), with pure Zig reference\n", .{});
    try outPrint("\n", .{});
    try outPrint("Sizes use a larger geometric step to keep small inputs while reaching bigger ones quickly.\n", .{});

    // Geometric progression (x2): includes small sizes and reaches much larger
    // inputs than before.
    const gemm_sizes = [_]usize{ 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

    try benchGemm(allocator, &gemm_sizes);
    try outPrint("\nTip: run `zig build -Doptimize=ReleaseFast bench -- conv` to benchmark conv2dSingleChannel.\n", .{});
}

fn benchDot(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("dot: x . y");
    for (sizes) |n| {
        const iters = linearIters(n);

        var x = try VecN.initAlloc(allocator, .{ .n = n });
        defer x.deinit(allocator);
        var y = try VecN.initAlloc(allocator, .{ .n = n });
        defer y.deinit(allocator);
        fillDeterministic(x.buf, 0x101);
        fillDeterministic(y.buf, 0x102);

        _ = pureDot(x.buf, y.buf);
        _ = blas.dot(T, AxisN, x.asConst(), y.asConst(), .{});
        _ = tblis.dot(AxisN, T, x.asConst(), y.asConst());

        var timer = try BenchTimer.start();
        var sum_pure: f64 = 0;
        var i: usize = 0;
        while (i < iters) : (i += 1) {
            sum_pure += @as(f64, pureDot(x.buf, y.buf));
        }
        const pure = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_pure };

        timer.reset();
        var sum_cblas: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) {
            sum_cblas += @as(f64, blas.dot(T, AxisN, x.asConst(), y.asConst(), .{}));
        }
        const cblas = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_cblas };

        timer.reset();
        var sum_tblis: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) {
            sum_tblis += @as(f64, tblis.dot(AxisN, T, x.asConst(), y.asConst()));
        }
        const t = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_tblis };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchAxpy(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("axpy: y += alpha * x");
    const alpha: T = 0.5;
    for (sizes) |n| {
        const iters = linearIters(n);

        var x_pure = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_pure.deinit(allocator);
        var y_pure = try VecN.initAlloc(allocator, .{ .n = n });
        defer y_pure.deinit(allocator);
        fillDeterministic(x_pure.buf, 0x201);
        fillDeterministic(y_pure.buf, 0x202);

        pureAxpy(alpha, x_pure.buf, y_pure.buf);

        var timer = try BenchTimer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) pureAxpy(alpha, x_pure.buf, y_pure.buf);
        const pure = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_pure.buf[0]) + @as(f64, y_pure.buf[n - 1]),
        };

        var x_cblas = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_cblas.deinit(allocator);
        var y_cblas = try VecN.initAlloc(allocator, .{ .n = n });
        defer y_cblas.deinit(allocator);
        fillDeterministic(x_cblas.buf, 0x201);
        fillDeterministic(y_cblas.buf, 0x202);

        blas.axpy(T, AxisN, alpha, x_cblas.asConst(), y_cblas);

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) blas.axpy(T, AxisN, alpha, x_cblas.asConst(), y_cblas);
        const cblas = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_cblas.buf[0]) + @as(f64, y_cblas.buf[n - 1]),
        };

        var x_tblis = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_tblis.deinit(allocator);
        var y_tblis = try VecN.initAlloc(allocator, .{ .n = n });
        defer y_tblis.deinit(allocator);
        fillDeterministic(x_tblis.buf, 0x201);
        fillDeterministic(y_tblis.buf, 0x202);

        tblis.add(AxisN, T, x_tblis.asConst(), y_tblis, .{ .scale_a = alpha, .scale_b = 1.0 });

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) {
            tblis.add(AxisN, T, x_tblis.asConst(), y_tblis, .{ .scale_a = alpha, .scale_b = 1.0 });
        }
        const t = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_tblis.buf[0]) + @as(f64, y_tblis.buf[n - 1]),
        };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchCopy(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("copy: y = x");
    for (sizes) |n| {
        const iters = linearIters(n);

        var x_pure = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_pure.deinit(allocator);
        var y_pure = try VecN.initAlloc(allocator, .{ .n = n });
        defer y_pure.deinit(allocator);
        fillDeterministic(x_pure.buf, 0x301);
        fillDeterministic(y_pure.buf, 0x302);

        pureCopy(x_pure.buf, y_pure.buf);

        var timer = try BenchTimer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) pureCopy(x_pure.buf, y_pure.buf);
        const pure = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_pure.buf[0]) + @as(f64, y_pure.buf[n - 1]),
        };

        var x_cblas = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_cblas.deinit(allocator);
        var y_cblas = try VecN.initAlloc(allocator, .{ .n = n });
        defer y_cblas.deinit(allocator);
        fillDeterministic(x_cblas.buf, 0x301);
        fillDeterministic(y_cblas.buf, 0x302);

        blas.copy(T, AxisN, x_cblas.asConst(), y_cblas);

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) blas.copy(T, AxisN, x_cblas.asConst(), y_cblas);
        const cblas = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_cblas.buf[0]) + @as(f64, y_cblas.buf[n - 1]),
        };

        var x_tblis = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_tblis.deinit(allocator);
        var y_tblis = try VecN.initAlloc(allocator, .{ .n = n });
        defer y_tblis.deinit(allocator);
        fillDeterministic(x_tblis.buf, 0x301);
        fillDeterministic(y_tblis.buf, 0x302);

        tblis.add(AxisN, T, x_tblis.asConst(), y_tblis, .{ .scale_a = 1.0, .scale_b = 0.0 });

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) {
            tblis.add(AxisN, T, x_tblis.asConst(), y_tblis, .{ .scale_a = 1.0, .scale_b = 0.0 });
        }
        const t = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_tblis.buf[0]) + @as(f64, y_tblis.buf[n - 1]),
        };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchScal(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("scal: x *= alpha");
    const alpha: T = 0.999;
    for (sizes) |n| {
        const iters = linearIters(n);

        var x_pure = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_pure.deinit(allocator);
        fillDeterministic(x_pure.buf, 0x401);

        pureScal(alpha, x_pure.buf);

        var timer = try BenchTimer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) pureScal(alpha, x_pure.buf);
        const pure = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, x_pure.buf[0]) + @as(f64, x_pure.buf[n - 1]),
        };

        var x_cblas = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_cblas.deinit(allocator);
        fillDeterministic(x_cblas.buf, 0x401);

        blas.scal(T, T, AxisN, alpha, x_cblas);

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) blas.scal(T, T, AxisN, alpha, x_cblas);
        const cblas = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, x_cblas.buf[0]) + @as(f64, x_cblas.buf[n - 1]),
        };

        var x_tblis = try VecN.initAlloc(allocator, .{ .n = n });
        defer x_tblis.deinit(allocator);
        fillDeterministic(x_tblis.buf, 0x401);

        tblis.scale(AxisN, T, alpha, x_tblis);

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) tblis.scale(AxisN, T, alpha, x_tblis);
        const t = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, x_tblis.buf[0]) + @as(f64, x_tblis.buf[n - 1]),
        };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchAsum(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("asum: sum(abs(x))");
    for (sizes) |n| {
        const iters = linearIters(n);

        var x = try VecN.initAlloc(allocator, .{ .n = n });
        defer x.deinit(allocator);
        fillDeterministic(x.buf, 0x501);

        _ = pureAsum(x.buf);
        _ = blas.asum(T, AxisN, x.asConst());
        _ = tblis.reduceAll(AxisN, T, .SUM_ABS, x.asConst());

        var timer = try BenchTimer.start();
        var sum_pure: f64 = 0;
        var i: usize = 0;
        while (i < iters) : (i += 1) sum_pure += @as(f64, pureAsum(x.buf));
        const pure = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_pure };

        timer.reset();
        var sum_cblas: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) sum_cblas += @as(f64, blas.asum(T, AxisN, x.asConst()));
        const cblas = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_cblas };

        timer.reset();
        var sum_tblis: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) sum_tblis += @as(f64, tblis.reduceAll(AxisN, T, .SUM_ABS, x.asConst()));
        const t = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_tblis };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchNrm2(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("nrm2: sqrt(sum(x*x))");
    for (sizes) |n| {
        const iters = linearIters(n);

        var x = try VecN.initAlloc(allocator, .{ .n = n });
        defer x.deinit(allocator);
        fillDeterministic(x.buf, 0x601);

        _ = pureNrm2(x.buf);
        _ = blas.nrm2(T, AxisN, x.asConst());
        _ = tblis.reduceAll(AxisN, T, .NORM_2, x.asConst());

        var timer = try BenchTimer.start();
        var sum_pure: f64 = 0;
        var i: usize = 0;
        while (i < iters) : (i += 1) sum_pure += @as(f64, pureNrm2(x.buf));
        const pure = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_pure };

        timer.reset();
        var sum_cblas: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) sum_cblas += @as(f64, blas.nrm2(T, AxisN, x.asConst()));
        const cblas = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_cblas };

        timer.reset();
        var sum_tblis: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) sum_tblis += @as(f64, tblis.reduceAll(AxisN, T, .NORM_2, x.asConst()));
        const t = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_tblis };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchIamax(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("iamax: argmax(abs(x))");
    for (sizes) |n| {
        const iters = linearIters(n);

        var x = try VecN.initAlloc(allocator, .{ .n = n });
        defer x.deinit(allocator);
        fillDeterministic(x.buf, 0x701);

        _ = pureIamax(x.buf);
        _ = blas.i_amax(T, AxisN, x.asConst());
        _ = tblis.reduceAllWithArg(AxisN, T, .MAX_ABS, x.asConst());

        var timer = try BenchTimer.start();
        var sum_pure: f64 = 0;
        var i: usize = 0;
        while (i < iters) : (i += 1) sum_pure += @as(f64, @floatFromInt(pureIamax(x.buf)));
        const pure = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_pure };

        timer.reset();
        var sum_cblas: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) sum_cblas += @as(f64, @floatFromInt(blas.i_amax(T, AxisN, x.asConst())));
        const cblas = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_cblas };

        timer.reset();
        var sum_tblis: f64 = 0;
        i = 0;
        while (i < iters) : (i += 1) {
            const res = tblis.reduceAllWithArg(AxisN, T, .MAX_ABS, x.asConst());
            sum_tblis += @as(f64, @floatFromInt(res.index.n));
        }
        const t = BenchResult{ .ns_total = timer.read(), .iters = iters, .checksum = sum_tblis };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchGemv(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("gemv accumulation: y += A * x");
    for (sizes) |n| {
        const iters = gemvIters(n, n);

        var a_pure = try MatIJ.initAlloc(allocator, .{ .i = n, .j = n });
        defer a_pure.deinit(allocator);
        var x_pure = try VecJ.initAlloc(allocator, .{ .j = n });
        defer x_pure.deinit(allocator);
        var y_pure = try VecI.initAlloc(allocator, .{ .i = n });
        defer y_pure.deinit(allocator);
        fillDeterministic(a_pure.buf, 0x801);
        fillDeterministic(x_pure.buf, 0x802);
        fillDeterministic(y_pure.buf, 0x803);

        pureGemvAccum(n, n, a_pure.buf, x_pure.buf, y_pure.buf);

        var timer = try BenchTimer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) pureGemvAccum(n, n, a_pure.buf, x_pure.buf, y_pure.buf);
        const pure = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_pure.buf[0]) + @as(f64, y_pure.buf[n - 1]),
        };

        var a_cblas = try MatIJ.initAlloc(allocator, .{ .i = n, .j = n });
        defer a_cblas.deinit(allocator);
        var x_cblas = try VecJ.initAlloc(allocator, .{ .j = n });
        defer x_cblas.deinit(allocator);
        var y_cblas = try VecI.initAlloc(allocator, .{ .i = n });
        defer y_cblas.deinit(allocator);
        fillDeterministic(a_cblas.buf, 0x801);
        fillDeterministic(x_cblas.buf, 0x802);
        fillDeterministic(y_cblas.buf, 0x803);

        blas.gemv(T, AxisIJ, AxisJ, AxisI, a_cblas.asConst(), x_cblas.asConst(), y_cblas, .{ .alpha = 1.0, .beta = 1.0 });

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) {
            blas.gemv(T, AxisIJ, AxisJ, AxisI, a_cblas.asConst(), x_cblas.asConst(), y_cblas, .{ .alpha = 1.0, .beta = 1.0 });
        }
        const cblas = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_cblas.buf[0]) + @as(f64, y_cblas.buf[n - 1]),
        };

        var a_tblis = try MatIJ.initAlloc(allocator, .{ .i = n, .j = n });
        defer a_tblis.deinit(allocator);
        var x_tblis = try VecJ.initAlloc(allocator, .{ .j = n });
        defer x_tblis.deinit(allocator);
        var y_tblis = try VecI.initAlloc(allocator, .{ .i = n });
        defer y_tblis.deinit(allocator);
        fillDeterministic(a_tblis.buf, 0x801);
        fillDeterministic(x_tblis.buf, 0x802);
        fillDeterministic(y_tblis.buf, 0x803);

        tblis.mult(AxisIJ, AxisJ, AxisI, T, a_tblis.asConst(), x_tblis.asConst(), y_tblis);

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) {
            tblis.mult(AxisIJ, AxisJ, AxisI, T, a_tblis.asConst(), x_tblis.asConst(), y_tblis);
        }
        const t = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, y_tblis.buf[0]) + @as(f64, y_tblis.buf[n - 1]),
        };

        try printResultBlock(n, pure, cblas, t);
    }
}

fn benchGemm(allocator: std.mem.Allocator, sizes: []const usize) !void {
    try printSectionHeader("gemm accumulation: C += A * B");
    for (sizes) |n| {
        const iters = gemmIters(n, n, n);

        var a_pure = try MatMK.initAlloc(allocator, .{ .m = n, .k = n });
        defer a_pure.deinit(allocator);
        var b_pure = try MatKN.initAlloc(allocator, .{ .k = n, .n = n });
        defer b_pure.deinit(allocator);
        var c_pure = try MatMN.initAlloc(allocator, .{ .m = n, .n = n });
        defer c_pure.deinit(allocator);
        fillDeterministic(a_pure.buf, 0x901);
        fillDeterministic(b_pure.buf, 0x902);
        fillDeterministic(c_pure.buf, 0x903);

        pureGemmAccum(n, n, n, a_pure.buf, b_pure.buf, c_pure.buf);

        var timer = try BenchTimer.start();
        var i: usize = 0;
        while (i < iters) : (i += 1) pureGemmAccum(n, n, n, a_pure.buf, b_pure.buf, c_pure.buf);
        const pure = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, c_pure.buf[0]) + @as(f64, c_pure.buf[n * n - 1]),
        };

        var a_cblas = try MatMK.initAlloc(allocator, .{ .m = n, .k = n });
        defer a_cblas.deinit(allocator);
        var b_cblas = try MatKN.initAlloc(allocator, .{ .k = n, .n = n });
        defer b_cblas.deinit(allocator);
        var c_cblas = try MatMN.initAlloc(allocator, .{ .m = n, .n = n });
        defer c_cblas.deinit(allocator);
        fillDeterministic(a_cblas.buf, 0x901);
        fillDeterministic(b_cblas.buf, 0x902);
        fillDeterministic(c_cblas.buf, 0x903);

        blas.gemm(T, AxisMK, AxisKN, AxisMN, a_cblas.asConst(), b_cblas.asConst(), c_cblas, .{ .alpha = 1.0, .beta = 1.0 });

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) {
            blas.gemm(T, AxisMK, AxisKN, AxisMN, a_cblas.asConst(), b_cblas.asConst(), c_cblas, .{ .alpha = 1.0, .beta = 1.0 });
        }
        const cblas = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, c_cblas.buf[0]) + @as(f64, c_cblas.buf[n * n - 1]),
        };

        var a_tblis = try MatMK.initAlloc(allocator, .{ .m = n, .k = n });
        defer a_tblis.deinit(allocator);
        var b_tblis = try MatKN.initAlloc(allocator, .{ .k = n, .n = n });
        defer b_tblis.deinit(allocator);
        var c_tblis = try MatMN.initAlloc(allocator, .{ .m = n, .n = n });
        defer c_tblis.deinit(allocator);
        fillDeterministic(a_tblis.buf, 0x901);
        fillDeterministic(b_tblis.buf, 0x902);
        fillDeterministic(c_tblis.buf, 0x903);

        tblis.mult(AxisMK, AxisKN, AxisMN, T, a_tblis.asConst(), b_tblis.asConst(), c_tblis);

        timer.reset();
        i = 0;
        while (i < iters) : (i += 1) {
            tblis.mult(AxisMK, AxisKN, AxisMN, T, a_tblis.asConst(), b_tblis.asConst(), c_tblis);
        }
        const t = BenchResult{
            .ns_total = timer.read(),
            .iters = iters,
            .checksum = @as(f64, c_tblis.buf[0]) + @as(f64, c_tblis.buf[n * n - 1]),
        };

        try printResultBlock(n, pure, cblas, t);
    }
}

const IndexHW = za.index.NamedIndex(conv.HW);

const ConvCase = struct {
    im_h: usize,
    im_w: usize,
    k_h: usize,
    k_w: usize,
};

const ConvLayout = enum {
    row_major,
    col_major,
    row_padded,
    col_padded,
};

const ConvVariant = struct {
    name: []const u8,
    im_layout: ConvLayout,
    kernel_layout: ConvLayout,
    out_layout: ConvLayout,
    im_pad: usize = 0,
    kernel_pad: usize = 0,
    out_pad: usize = 0,
};

fn benchConv2dSingleChannel(allocator: std.mem.Allocator) !void {
    // Smaller than before (for quicker iteration), but with more repetitions per
    // data point for lower timing noise.
    const cases = [_]ConvCase{
        .{ .im_h = 64, .im_w = 64, .k_h = 3, .k_w = 3 },
        .{ .im_h = 128, .im_w = 128, .k_h = 3, .k_w = 3 },
        .{ .im_h = 192, .im_w = 192, .k_h = 3, .k_w = 3 },
        .{ .im_h = 192, .im_w = 192, .k_h = 7, .k_w = 7 },
    };

    const variants = [_]ConvVariant{
        .{
            .name = "all_row_major",
            .im_layout = .row_major,
            .kernel_layout = .row_major,
            .out_layout = .row_major,
        },
        .{
            .name = "im_col__kernel_row__out_row",
            .im_layout = .col_major,
            .kernel_layout = .row_major,
            .out_layout = .row_major,
        },
        .{
            .name = "im_row__kernel_col__out_row",
            .im_layout = .row_major,
            .kernel_layout = .col_major,
            .out_layout = .row_major,
        },
        .{
            .name = "im_row_padded__kernel_col__out_row",
            .im_layout = .row_padded,
            .kernel_layout = .col_major,
            .out_layout = .row_major,
            .im_pad = 7,
        },
        .{
            .name = "im_col_padded__kernel_row_padded__out_col",
            .im_layout = .col_padded,
            .kernel_layout = .row_padded,
            .out_layout = .col_major,
            .im_pad = 5,
            .kernel_pad = 3,
        },
    };

    try outPrint("variant | im(HxW) | k(HxW) | out(HxW) | iters | ns/op | MPix/s | GMAC/s | checksum\n", .{});

    for (variants) |variant| {
        try outPrint("\n[{s}]\n", .{variant.name});

        for (cases) |c| {
            const out_h = c.im_h - c.k_h + 1;
            const out_w = c.im_w - c.k_w + 1;
            const out_elems = out_h * out_w;
            const macs_per_iter = out_elems * c.k_h * c.k_w;
            const iters = conv2dIters(macs_per_iter);

            const im = try allocMatHWWithLayout(allocator, c.im_h, c.im_w, variant.im_layout, variant.im_pad);
            defer im.deinit(allocator);
            const kernel = try allocMatHWWithLayout(allocator, c.k_h, c.k_w, variant.kernel_layout, variant.kernel_pad);
            defer kernel.deinit(allocator);
            const out = try allocMatHWWithLayout(allocator, out_h, out_w, variant.out_layout, variant.out_pad);
            defer out.deinit(allocator);

            fillDeterministic(im.buf, 0xA01 + c.im_h * 17 + c.im_w * 31);
            fillDeterministic(kernel.buf, 0xB01 + c.k_h * 19 + c.k_w * 37);
            fillDeterministic(out.buf, 0xC01 + out_h * 13 + out_w * 29);

            // Warm-up so first timed run isn't dominated by cold-start effects.
            conv.conv2dSingleChannel(T, im.asConst(), kernel.asConst(), out);

            var timer = try BenchTimer.start();
            var i: usize = 0;
            while (i < iters) : (i += 1) {
                conv.conv2dSingleChannel(T, im.asConst(), kernel.asConst(), out);
            }

            const ns_total = timer.read();
            const ns_per_iter = @as(f64, @floatFromInt(ns_total)) / @as(f64, @floatFromInt(iters));
            const sec_per_iter = ns_per_iter / @as(f64, @floatFromInt(std.time.ns_per_s));
            const mpix_per_s = @as(f64, @floatFromInt(out_elems)) / sec_per_iter / 1_000_000.0;
            const gmac_per_s = @as(f64, @floatFromInt(macs_per_iter)) / sec_per_iter / 1_000_000_000.0;
            const checksum = @as(f64, out.scalarAt(.{ .h = 0, .w = 0 })) + @as(f64, out.scalarAt(.{ .h = out_h - 1, .w = out_w - 1 }));

            try outPrint(
                "im={}x{}, k={}x{}, out={}x{}, iters={}, ns/op={d:.2}, MPix/s={d:.2}, GMAC/s={d:.2}, checksum={d:.6}\n",
                .{ c.im_h, c.im_w, c.k_h, c.k_w, out_h, out_w, iters, ns_per_iter, mpix_per_s, gmac_per_s, checksum },
            );
        }
    }
}

fn allocMatHWWithLayout(allocator: std.mem.Allocator, h: usize, w: usize, layout: ConvLayout, pad: usize) !MatHW {
    const idx = makeHWIndex(h, w, layout, pad);
    const len = indexRequiredBufferLen(idx);
    const buf = try allocator.alloc(T, len);
    return MatHW.init(idx, buf);
}

fn makeHWIndex(h: usize, w: usize, layout: ConvLayout, pad: usize) IndexHW {
    return switch (layout) {
        .row_major => .{
            .shape = .{ .h = h, .w = w },
            .strides = .{ .h = @as(isize, @intCast(w)), .w = 1 },
            .offset = 0,
        },
        .col_major => .{
            .shape = .{ .h = h, .w = w },
            .strides = .{ .h = 1, .w = @as(isize, @intCast(h)) },
            .offset = 0,
        },
        .row_padded => .{
            .shape = .{ .h = h, .w = w },
            .strides = .{ .h = @as(isize, @intCast(w + pad)), .w = 1 },
            .offset = 0,
        },
        .col_padded => .{
            .shape = .{ .h = h, .w = w },
            .strides = .{ .h = 1, .w = @as(isize, @intCast(h + pad)) },
            .offset = 0,
        },
    };
}

fn indexRequiredBufferLen(idx: IndexHW) usize {
    if (idx.shape.h == 0 or idx.shape.w == 0) return 0;
    const last = idx.linear(.{ .h = idx.shape.h - 1, .w = idx.shape.w - 1 });
    return last + 1;
}

fn pureDot(x: []const T, y: []const T) T {
    var acc: T = 0;
    for (x, y) |xv, yv| acc += xv * yv;
    return acc;
}

fn pureAxpy(alpha: T, x: []const T, y: []T) void {
    var i: usize = 0;
    while (i < x.len) : (i += 1) {
        y[i] += alpha * x[i];
    }
}

fn pureCopy(x: []const T, y: []T) void {
    @memcpy(y, x);
}

fn pureScal(alpha: T, x: []T) void {
    for (x) |*v| v.* *= alpha;
}

fn pureAsum(x: []const T) T {
    var acc: T = 0;
    for (x) |v| acc += absf(v);
    return acc;
}

fn pureNrm2(x: []const T) T {
    var acc: T = 0;
    for (x) |v| acc += v * v;
    return @sqrt(acc);
}

fn pureIamax(x: []const T) usize {
    var best_idx: usize = 0;
    var best_abs = absf(x[0]);
    var i: usize = 1;
    while (i < x.len) : (i += 1) {
        const cur = absf(x[i]);
        if (cur > best_abs) {
            best_abs = cur;
            best_idx = i;
        }
    }
    return best_idx;
}

fn pureGemvAccum(m: usize, n: usize, a: []const T, x: []const T, y: []T) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var acc: T = 0;
        const row = a[i * n .. (i + 1) * n];
        var j: usize = 0;
        while (j < n) : (j += 1) acc += row[j] * x[j];
        y[i] += acc;
    }
}

fn pureGemmAccum(m: usize, n: usize, k: usize, a: []const T, b: []const T, c: []T) void {
    const lanes: usize = SIMD_LANES;

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var kk: usize = 0;
        while (kk < k) : (kk += 1) {
            const aik = a[i * k + kk];
            const aik_vec: SimdVec = @splat(aik);

            const b_row = b[kk * n .. (kk + 1) * n];
            var c_row = c[i * n .. (i + 1) * n];

            var j: usize = 0;
            while (j + lanes <= n) : (j += lanes) {
                const b_chunk: *const [lanes]T = @ptrCast(b_row.ptr + j);
                const c_chunk: *[lanes]T = @ptrCast(c_row.ptr + j);

                const b_vec: SimdVec = b_chunk.*;
                var c_vec: SimdVec = c_chunk.*;
                c_vec += aik_vec * b_vec;
                c_chunk.* = c_vec;
            }

            while (j < n) : (j += 1) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

fn fillDeterministic(buf: []T, seed: u64) void {
    var state: u64 = (seed | 1);
    for (buf, 0..) |*v, i| {
        state = state *% 6364136223846793005 +% 1442695040888963407 +% @as(u64, i);
        const upper: u32 = @truncate(state >> 32);
        const unit = @as(T, @floatFromInt(upper)) / @as(T, 4294967295.0);
        v.* = unit * 2.0 - 1.0;
    }
}

fn absf(x: T) T {
    return if (x < 0) -x else x;
}

fn linearIters(n: usize) usize {
    // Roughly target a constant total amount of O(n) work per data point.
    return clampUsize(8_000_000 / n, 10, 50_000);
}

fn gemvIters(m: usize, n: usize) usize {
    // Roughly target a constant total amount of O(m*n) work.
    return clampUsize(6_000_000 / (m * n), 2, 2_000);
}

fn gemmIters(m: usize, n: usize, k: usize) usize {
    // Roughly target a constant total amount of O(m*n*k) work.
    return clampUsize(12_000_000 / (m * n * k), 1, 200);
}

fn conv2dIters(macs_per_iter: usize) usize {
    // Target a larger fixed MAC budget to reduce timing noise.
    return clampUsize(600_000_000 / macs_per_iter, 50, 40_000);
}

fn clampUsize(v: usize, lo: usize, hi: usize) usize {
    return @min(@max(v, lo), hi);
}

fn printSectionHeader(title: []const u8) !void {
    try outPrint("\n", .{});
    try outPrint("=== {s} ===\n", .{title});
    try outPrint("size | implementation | ns/op | speedup vs pure | checksum\n", .{});
}

fn printResultBlock(size: usize, pure: BenchResult, cblas: BenchResult, t: BenchResult) !void {
    const pure_ns = pure.nsPerIter();
    const cblas_ns = cblas.nsPerIter();
    const tblis_ns = t.nsPerIter();

    try outPrint("n = {} (iters = {})\n", .{ size, pure.iters });
    try outPrint("  {s:8} | {d:>9.2} | {d:>15.2} | {d:>9.3}\n", .{ "pure", pure_ns, 1.0, pure.checksum });
    try outPrint("  {s:8} | {d:>9.2} | {d:>15.2} | {d:>9.3}\n", .{ "cblas", cblas_ns, pure_ns / cblas_ns, cblas.checksum });
    try outPrint("  {s:8} | {d:>9.2} | {d:>15.2} | {d:>9.3}\n", .{ "tblis", tblis_ns, pure_ns / tblis_ns, t.checksum });
}

fn monotonicNs() u64 {
    var ts: C.struct_timespec = undefined;
    _ = C.clock_gettime(C.CLOCK_MONOTONIC, &ts);
    const sec: u64 = @intCast(ts.tv_sec);
    const nsec: u64 = @intCast(ts.tv_nsec);
    return sec * std.time.ns_per_s + nsec;
}

fn outPrint(comptime fmt: []const u8, args: anytype) !void {
    var buf: [4096]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, fmt, args);
    _ = C.fwrite(msg.ptr, 1, msg.len, @constCast(C.stdout()));
}

test "bench.zig" {
    std.testing.refAllDecls(@This());
}
