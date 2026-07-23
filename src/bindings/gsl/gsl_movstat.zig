//! # Moving-window statistics (`gsl_movstat`)
//!
//! Slide a window of `K` samples over an input signal and emit a same-length
//! output series of a statistic computed within each window (moving mean,
//! median, min/max, robust scale, ...). `f64`-only, so this is a plain
//! namespace rather than `movstat(T)`.
//!
//! Input is a `Strided(f64)` view and output a `StridedMut(f64)` view (fed
//! zero-copy to GSL's `gsl_vector *` API via borrowed views); every routine
//! requires the output length to equal the input length and returns
//! `Error.BadLength` otherwise. The non-aborting error handler is installed on
//! first use so a GSL error surfaces as a Zig `Error` rather than aborting.
//!
//! ## Omissions
//!
//!   - The user-accumulator driver (`gsl_movstat_apply`/`_apply_accum`,
//!     `gsl_movstat_function`, `gsl_movstat_accum`) for custom window functions.
//!   - `gsl_movstat_fill` (the raw window-extraction helper).
//!   - The related digital filters in `gsl_filter.h` (Gaussian, median, RMF,
//!     impulse) — a separate module.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const movstat = @This();
const rstat = gsl.rstat;
const rand = gsl.rand;

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_vector.h");
    @cInclude("gsl/gsl_movstat.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL bindings).
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code.
pub const strerror = gsl.strerror;
/// Shared GSL error set.
pub const Error = gsl.Error;
const check = gsl.check;
/// Strided read-only view. Re-exported from `gsl.zig`.
pub const Strided = gsl.Strided;
/// Strided mutable view. Re-exported from `gsl.zig`.
pub const StridedMut = gsl.StridedMut;

inline fn ensureHandler() void {
    gsl.ensureHandler();
}

inline fn constVectorView(x: Strided(f64)) c.gsl_vector {
    return gsl.constVectorViewOf(c.gsl_vector, x);
}

inline fn mutVectorView(y: StridedMut(f64)) c.gsl_vector {
    return gsl.mutVectorViewOf(c.gsl_vector, y);
}

/// How the window is handled where it overhangs the ends of the signal.
pub const End = enum(c.gsl_movstat_end_t) {
    /// Pad with zeros beyond the boundary.
    pad_zero = c.GSL_MOVSTAT_END_PADZERO,
    /// Pad with the nearest edge value beyond the boundary.
    pad_value = c.GSL_MOVSTAT_END_PADVALUE,
    /// Shrink the window to only the in-range samples at the boundary.
    truncate = c.GSL_MOVSTAT_END_TRUNCATE,
};

/// A moving-window workspace (`gsl_movstat_workspace`) of a fixed width.
/// Owns its GSL allocation; call `deinit` to free it. Reusable across many
/// calls (including different statistics) at the same window size.
///
/// Example:
/// ```
/// var win = try gsl.movstat.Window.init(5);   // symmetric width-5 window
/// defer win.deinit();
/// try win.median(.truncate, gsl.Strided(f64).fromSlice(x),
///                gsl.StridedMut(f64).fromSlice(y));
/// ```
pub const Window = struct {
    ptr: *c.gsl_movstat_workspace,

    /// Allocate a symmetric window of width `k` (`H = J = k/2`). Fails only
    /// if the underlying allocation fails.
    pub fn init(k: usize) error{OutOfMemory}!Window {
        const p = c.gsl_movstat_alloc(k) orelse return error.OutOfMemory;
        return .{ .ptr = p };
    }
    /// Allocate an asymmetric window spanning `back` samples before and
    /// `forward` samples after each point (width `back + forward + 1`).
    pub fn initAsymmetric(back: usize, forward: usize) error{OutOfMemory}!Window {
        const p = c.gsl_movstat_alloc2(back, forward) orelse return error.OutOfMemory;
        return .{ .ptr = p };
    }
    pub fn deinit(self: Window) void {
        c.gsl_movstat_free(self.ptr);
    }

    // Drive a one-input, one-output routine.
    fn run1(
        self: Window,
        comptime f: anytype,
        end: End,
        x: Strided(f64),
        y: StridedMut(f64),
    ) Error!void {
        if (x.len != y.len) return Error.BadLength;
        ensureHandler();
        var xv = constVectorView(x);
        var yv = mutVectorView(y);
        try check(f(@intFromEnum(end), &xv, &yv, self.ptr));
    }

    /// Moving mean.
    pub fn mean(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_mean, end, x, y);
    }
    /// Moving (sample) variance.
    pub fn variance(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_variance, end, x, y);
    }
    /// Moving (sample) standard deviation.
    pub fn sd(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_sd, end, x, y);
    }
    /// Moving median.
    pub fn median(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_median, end, x, y);
    }
    /// Moving minimum.
    pub fn min(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_min, end, x, y);
    }
    /// Moving maximum.
    pub fn max(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_max, end, x, y);
    }
    /// Moving sum.
    pub fn sum(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_sum, end, x, y);
    }

    /// Moving minimum and maximum in one pass (into separate outputs).
    pub fn minMax(
        self: Window,
        end: End,
        x: Strided(f64),
        y_min: StridedMut(f64),
        y_max: StridedMut(f64),
    ) Error!void {
        if (x.len != y_min.len or x.len != y_max.len) return Error.BadLength;
        ensureHandler();
        var xv = constVectorView(x);
        var lo = mutVectorView(y_min);
        var hi = mutVectorView(y_max);
        try check(c.gsl_movstat_minmax(@intFromEnum(end), &xv, &lo, &hi, self.ptr));
    }

    /// Moving median absolute deviation, scaled to estimate the standard
    /// deviation (× 1.4826). Also writes the running median into `xmedian`.
    pub fn mad(
        self: Window,
        end: End,
        x: Strided(f64),
        xmedian: StridedMut(f64),
        y: StridedMut(f64),
    ) Error!void {
        return self.run2(c.gsl_movstat_mad, end, x, xmedian, y);
    }
    /// Moving median absolute deviation, unscaled (raw MAD). Also writes the
    /// running median into `xmedian`.
    pub fn mad0(
        self: Window,
        end: End,
        x: Strided(f64),
        xmedian: StridedMut(f64),
        y: StridedMut(f64),
    ) Error!void {
        return self.run2(c.gsl_movstat_mad0, end, x, xmedian, y);
    }

    // Drive a routine with one input and two outputs (mad/mad0).
    fn run2(
        self: Window,
        comptime f: anytype,
        end: End,
        x: Strided(f64),
        out1: StridedMut(f64),
        out2: StridedMut(f64),
    ) Error!void {
        if (x.len != out1.len or x.len != out2.len) return Error.BadLength;
        ensureHandler();
        var xv = constVectorView(x);
        var o1 = mutVectorView(out1);
        var o2 = mutVectorView(out2);
        try check(f(@intFromEnum(end), &xv, &o1, &o2, self.ptr));
    }

    /// Moving `q`-quantile range: the difference between the `q`- and
    /// `(1-q)`-quantiles within each window (a robust spread measure).
    pub fn qqr(self: Window, end: End, x: Strided(f64), q: f64, y: StridedMut(f64)) Error!void {
        if (x.len != y.len) return Error.BadLength;
        ensureHandler();
        var xv = constVectorView(x);
        var yv = mutVectorView(y);
        try check(c.gsl_movstat_qqr(@intFromEnum(end), &xv, q, &yv, self.ptr));
    }

    /// Moving robust scale estimate S_n (Rousseeuw–Croux).
    pub fn Sn(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_Sn, end, x, y);
    }
    /// Moving robust scale estimate Q_n (Rousseeuw–Croux).
    pub fn Qn(self: Window, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        return self.run1(c.gsl_movstat_Qn, end, x, y);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test "movstat: moving statistics on a known signal (truncate ends)" {
    const x = [_]f64{ 1, 2, 3, 4, 5 };
    var y: [5]f64 = undefined;
    const xv = Strided(f64).fromSlice(&x);
    const yv = StridedMut(f64).fromSlice(&y);

    var win = try movstat.Window.init(3); // symmetric width-3 window
    defer win.deinit();

    // With truncated ends the window shrinks to the in-range samples.
    try win.mean(.truncate, xv, yv);
    try testing.expectEqualSlices(f64, &.{ 1.5, 2, 3, 4, 4.5 }, &y);

    try win.sum(.truncate, xv, yv);
    try testing.expectEqualSlices(f64, &.{ 3, 6, 9, 12, 9 }, &y);

    try win.min(.truncate, xv, yv);
    try testing.expectEqualSlices(f64, &.{ 1, 1, 2, 3, 4 }, &y);

    try win.max(.truncate, xv, yv);
    try testing.expectEqualSlices(f64, &.{ 2, 3, 4, 5, 5 }, &y);

    try win.median(.truncate, xv, yv);
    try testing.expectEqualSlices(f64, &.{ 1.5, 2, 3, 4, 4.5 }, &y);

    // Zero-padded ends pull the boundary averages toward zero.
    try win.mean(.pad_zero, xv, yv);
    try testing.expectApproxEqAbs(@as(f64, 1.0), y[0], 1e-12); // (0+1+2)/3
    try testing.expectApproxEqAbs(@as(f64, 3.0), y[4], 1e-12); // (4+5+0)/3
}

test "movstat: multi-output, robust, and asymmetric routines run" {
    const x = [_]f64{ 1, 2, 100, 4, 5, 6, 7, 8 };
    var y: [8]f64 = undefined;
    var y2: [8]f64 = undefined;
    const xv = Strided(f64).fromSlice(&x);
    const yv = StridedMut(f64).fromSlice(&y);
    const yv2 = StridedMut(f64).fromSlice(&y2);

    var win = try movstat.Window.init(5);
    defer win.deinit();

    // min/max in one pass: min <= max pointwise.
    try win.minMax(.truncate, xv, yv, yv2);
    for (y, y2) |lo, hi| try testing.expect(lo <= hi);

    // mad writes the running median into the first output and the (scaled) MAD
    // into the second; both stay finite and MAD is non-negative.
    try win.mad(.truncate, xv, yv, yv2);
    for (y2) |m| try testing.expect(m >= 0 and std.math.isFinite(m));
    try win.mad0(.truncate, xv, yv, yv2);
    for (y2) |m| try testing.expect(m >= 0 and std.math.isFinite(m));

    // qqr / Sn / Qn are robust spread/scale measures: finite, non-negative.
    try win.qqr(.truncate, xv, 0.25, yv);
    for (y) |m| try testing.expect(m >= 0 and std.math.isFinite(m));
    try win.Sn(.truncate, xv, yv);
    for (y) |m| try testing.expect(m >= 0 and std.math.isFinite(m));
    try win.Qn(.truncate, xv, yv);
    for (y) |m| try testing.expect(m >= 0 and std.math.isFinite(m));

    // variance / sd exist too.
    try win.variance(.truncate, xv, yv);
    for (y) |m| try testing.expect(m >= 0);
    try win.sd(.truncate, xv, yv);
    for (y) |m| try testing.expect(m >= 0);

    // An asymmetric window (2 back, 0 forward = trailing average) also works.
    var trailing = try movstat.Window.initAsymmetric(2, 0);
    defer trailing.deinit();
    try trailing.mean(.truncate, xv, yv);
    try testing.expectApproxEqAbs(x[0], y[0], 1e-12); // only itself in range
}

test "movstat: strided views and length checks" {
    // Interleave the signal into every other slot of a larger buffer to exercise
    // the strided borrowed-view path (stride 2).
    var packed_in = [_]f64{ 1, -1, 2, -1, 3, -1, 4, -1 };
    var packed_out = [_]f64{ 0, 9, 0, 9, 0, 9, 0, 9 };
    const xv = Strided(f64).init(packed_in[0..].ptr, 2, 4);
    const yv = StridedMut(f64).init(packed_out[0..].ptr, 2, 4);

    var win = try movstat.Window.init(3);
    defer win.deinit();
    try win.mean(.truncate, xv, yv);

    // Outputs land only in the strided slots; the interleaved -1/9 are untouched.
    try testing.expectApproxEqAbs(@as(f64, 1.5), packed_out[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), packed_out[2], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), packed_out[4], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.5), packed_out[6], 1e-12);
    try testing.expectEqual(@as(f64, 9), packed_out[1]);

    // A length mismatch between input and output is rejected, not aborted.
    var short: [3]f64 = undefined;
    const full = Strided(f64).fromSlice(&packed_in);
    try testing.expectError(Error.BadLength, win.mean(.truncate, full, StridedMut(f64).fromSlice(&short)));
}
test "rstat/movstat: every wrapped method is invoked (symbol + arity coverage)" {
    // Exhaustively call every binding once with benign inputs, discarding the
    // result and any error. Purpose: force every extern `gsl_rstat_*` /
    // `gsl_movstat_*` symbol to link and its argument order/types to compile.
    // Correctness is checked by the closed-form/oracle tests above.
    const run = struct {
        fn call(eu: anytype) void {
            if (eu) |_| {} else |_| {}
        }
    }.call;

    // --- rstat.Accumulator: mutators + every infallible getter ---
    var acc = try rstat.Accumulator.init();
    defer acc.deinit();
    acc.add(1.0);
    acc.addSlice(&.{ 2.0, 3.0, 4.0 });
    inline for (.{
        rstat.Accumulator.count,    rstat.Accumulator.mean, rstat.Accumulator.variance, rstat.Accumulator.sd,
        rstat.Accumulator.sdMean,   rstat.Accumulator.rms,  rstat.Accumulator.norm,     rstat.Accumulator.skew,
        rstat.Accumulator.kurtosis, rstat.Accumulator.min,  rstat.Accumulator.max,      rstat.Accumulator.median,
    }) |g| _ = g(acc);
    acc.reset();

    // --- rstat.Quantile ---
    var q = try rstat.Quantile.init(0.5);
    defer q.deinit();
    q.add(1.0);
    q.add(2.0);
    _ = q.get();
    q.reset();

    // --- movstat.Window: every routine across every `End` variant ---
    const x = [_]f64{ 5, 1, 4, 1, 5, 9, 2, 6 };
    var y: [8]f64 = undefined;
    var y2: [8]f64 = undefined;
    const xv = Strided(f64).fromSlice(&x);
    const yv = StridedMut(f64).fromSlice(&y);
    const yv2 = StridedMut(f64).fromSlice(&y2);

    var win = try movstat.Window.init(3);
    defer win.deinit();

    inline for (comptime std.enums.values(movstat.End)) |end| {
        inline for (.{
            movstat.Window.mean, movstat.Window.variance, movstat.Window.sd,  movstat.Window.median,
            movstat.Window.min,  movstat.Window.max,      movstat.Window.sum, movstat.Window.Sn,
            movstat.Window.Qn,
        }) |f| run(f(win, end, xv, yv));
        run(win.minMax(end, xv, yv, yv2));
        run(win.mad(end, xv, yv, yv2));
        run(win.mad0(end, xv, yv, yv2));
        run(win.qqr(end, xv, 0.25, yv));
    }

    // The asymmetric allocator is exercised at least once.
    var win2 = try movstat.Window.initAsymmetric(2, 1);
    defer win2.deinit();
    run(win2.mean(.truncate, xv, yv));
}
