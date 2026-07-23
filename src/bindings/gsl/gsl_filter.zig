//! Idiomatic Zig bindings for the GNU Scientific Library's digital-filter
//! module (`gsl_filter`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the digital-filter
//! chapter — a follow-on to the moving-window statistics (`gsl.movstat`), which
//! it shares infrastructure with. It reuses `gsl.zig`'s process-global
//! error-handler switch and its borrowed-`gsl_vector` view helpers
//! (`constVectorViewOf`/`mutVectorViewOf`), but keeps the filter-specific C API
//! behind its own `c`. It is reached as `gsl.filter`.
//!
//! ## Shape of the surface
//!
//! Each filter is a workspace type (`init(K)`/`deinit`) that transforms a
//! `Strided(f64)` input `x` into a same-length `StridedMut(f64)` output `y`.
//! Input and output are fed to GSL's `gsl_vector *` API zero-copy via borrowed
//! views, exactly like `movstat`; every routine requires `y.len == x.len` and
//! returns `Error.BadLength` otherwise.
//!
//!   - `Gaussian` — convolution with a Gaussian kernel (or its derivatives);
//!     `apply` takes the shape parameter `alpha` and derivative `order`, and
//!     the static `kernel` builds the kernel itself.
//!   - `Median` — standard moving median filter (rank filter).
//!   - `RecursiveMedian` — recursive median filter (converges to a root
//!     sequence / local monotonicity).
//!   - `Impulse` — impulse-detection filter: flags and replaces outliers that
//!     deviate from the local median by more than `t` robust scale estimates,
//!     returning the number of outliers found (and, optionally, writing a
//!     per-sample outlier mask).
//!
//! `End` (boundary handling) mirrors `movstat.End`; `Scale` selects the robust
//! scale estimator used by the impulse filter.
//!
//! ## Omissions
//!
//! The full `gsl_filter.h` surface is wrapped, including `Impulse`'s optional
//! integer outlier mask (`ioutlier`, exposed as the `outliers` parameter). The
//! only things not re-exposed here are the internal `movstat`-accumulator
//! plumbing those filters are built on — use `gsl.movstat` (or the raw `c` API)
//! directly for custom window functions.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here (e.g. the
/// integer outlier-index output of the impulse filter).
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_vector.h");
    @cInclude("gsl/gsl_movstat.h");
    @cInclude("gsl/gsl_filter.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;
/// Strided read-only input view. Re-exported from `gsl.zig`.
pub const Strided = gsl.Strided;
/// Strided mutable output view. Re-exported from `gsl.zig`.
pub const StridedMut = gsl.StridedMut;

/// Zig error set for the filter routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_filter_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EDOM` — a value outside the routine's domain.
    Domain,
    /// `GSL_EINVAL` — an invalid argument.
    Invalid,
    /// `GSL_EBADLEN`, or a caller-supplied output view whose length differs
    /// from the input.
    BadLength,
    /// `GSL_ENOMEM` — allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EDOM => Error.Domain,
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADLEN => Error.BadLength,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

inline fn constView(x: Strided(f64)) c.gsl_vector {
    return gsl.constVectorViewOf(c.gsl_vector, x);
}
inline fn mutView(y: StridedMut(f64)) c.gsl_vector {
    return gsl.mutVectorViewOf(c.gsl_vector, y);
}

/// How the filter window is handled where it overhangs the ends of the signal.
/// The underlying values are identical to `movstat.End`.
pub const End = enum(c.gsl_filter_end_t) {
    /// Pad with zeros beyond the boundary.
    pad_zero = c.GSL_FILTER_END_PADZERO,
    /// Pad with the nearest edge value beyond the boundary.
    pad_value = c.GSL_FILTER_END_PADVALUE,
    /// Shrink the window to only the in-range samples at the boundary.
    truncate = c.GSL_FILTER_END_TRUNCATE,
};

/// The robust scale estimator the impulse filter uses to turn the deviation
/// from the local median into a comparable threshold.
pub const Scale = enum(c.gsl_filter_scale_t) {
    /// Median absolute deviation (× 1.4826).
    mad = c.GSL_FILTER_SCALE_MAD,
    /// Interquartile range.
    iqr = c.GSL_FILTER_SCALE_IQR,
    /// Rousseeuw–Croux Sₙ statistic.
    sn = c.GSL_FILTER_SCALE_SN,
    /// Rousseeuw–Croux Qₙ statistic.
    qn = c.GSL_FILTER_SCALE_QN,
};

/// A Gaussian smoothing/derivative filter (`gsl_filter_gaussian_workspace`):
/// convolves the signal with a Gaussian kernel (or a derivative of one). Owns
/// its GSL allocation; call `deinit` to free.
///
/// Example:
/// ```
/// var g = try gsl.filter.Gaussian.init(11);
/// defer g.deinit();
/// try g.apply(.pad_value, 3.0, 0, gsl.Strided(f64).fromSlice(x),
///             gsl.StridedMut(f64).fromSlice(y)); // order 0 = smoothing
/// ```
pub const Gaussian = struct {
    ptr: *c.gsl_filter_gaussian_workspace,

    /// Allocate a Gaussian filter with window size `k` (odd is typical). Fails
    /// only if the underlying allocation fails.
    pub fn init(k: usize) Error!Gaussian {
        gsl.ensureHandler();
        const p = c.gsl_filter_gaussian_alloc(k) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Gaussian) void {
        c.gsl_filter_gaussian_free(self.ptr);
    }

    /// Filter `x` into `y`. `alpha` sets how many standard deviations the
    /// half-window spans (larger = narrower kernel); `order` selects the
    /// derivative of the Gaussian (0 = smoothing, 1 = first derivative, ...).
    /// Requires `y.len == x.len`.
    pub fn apply(
        self: Gaussian,
        end: End,
        alpha: f64,
        order: usize,
        x: Strided(f64),
        y: StridedMut(f64),
    ) Error!void {
        if (x.len != y.len) return Error.BadLength;
        gsl.ensureHandler();
        var xv = constView(x);
        var yv = mutView(y);
        try check(c.gsl_filter_gaussian(@intFromEnum(end), alpha, order, &xv, &yv, self.ptr));
    }

    /// Compute the Gaussian kernel itself into `out` (of length equal to the
    /// desired kernel size). `normalize` scales a smoothing kernel to sum to 1.
    /// Independent of any workspace.
    pub fn kernel(alpha: f64, order: usize, normalize: bool, out: StridedMut(f64)) Error!void {
        gsl.ensureHandler();
        var kv = mutView(out);
        try check(c.gsl_filter_gaussian_kernel(alpha, order, @intFromBool(normalize), &kv));
    }
};

/// A standard moving median filter (`gsl_filter_median_workspace`): replaces
/// each sample with the median of its window. Owns its GSL allocation; call
/// `deinit` to free.
pub const Median = struct {
    ptr: *c.gsl_filter_median_workspace,

    /// Allocate a median filter with window size `k`.
    pub fn init(k: usize) Error!Median {
        gsl.ensureHandler();
        const p = c.gsl_filter_median_alloc(k) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Median) void {
        c.gsl_filter_median_free(self.ptr);
    }

    /// Filter `x` into `y`. Requires `y.len == x.len`.
    pub fn apply(self: Median, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        if (x.len != y.len) return Error.BadLength;
        gsl.ensureHandler();
        var xv = constView(x);
        var yv = mutView(y);
        try check(c.gsl_filter_median(@intFromEnum(end), &xv, &yv, self.ptr));
    }
};

/// A recursive median filter (`gsl_filter_rmedian_workspace`): like the median
/// filter but feeds already-filtered outputs back into each window, converging
/// to a locally monotone ("root") sequence in a single pass. Owns its GSL
/// allocation; call `deinit` to free.
pub const RecursiveMedian = struct {
    ptr: *c.gsl_filter_rmedian_workspace,

    /// Allocate a recursive median filter with window size `k`.
    pub fn init(k: usize) Error!RecursiveMedian {
        gsl.ensureHandler();
        const p = c.gsl_filter_rmedian_alloc(k) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: RecursiveMedian) void {
        c.gsl_filter_rmedian_free(self.ptr);
    }

    /// Filter `x` into `y`. Requires `y.len == x.len`.
    pub fn apply(self: RecursiveMedian, end: End, x: Strided(f64), y: StridedMut(f64)) Error!void {
        if (x.len != y.len) return Error.BadLength;
        gsl.ensureHandler();
        var xv = constView(x);
        var yv = mutView(y);
        try check(c.gsl_filter_rmedian(@intFromEnum(end), &xv, &yv, self.ptr));
    }
};

/// An impulse-detection filter (`gsl_filter_impulse_workspace`): flags samples
/// that deviate from their local median by more than `t` robust scale estimates
/// and replaces those with the median, leaving the rest untouched. Owns its GSL
/// allocation; call `deinit` to free.
pub const Impulse = struct {
    ptr: *c.gsl_filter_impulse_workspace,

    /// Allocate an impulse filter with window size `k`.
    pub fn init(k: usize) Error!Impulse {
        gsl.ensureHandler();
        const p = c.gsl_filter_impulse_alloc(k) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Impulse) void {
        c.gsl_filter_impulse_free(self.ptr);
    }

    /// Filter `x` into `y`, replacing outliers with the local median. A sample
    /// is an outlier when `|x[i] - median| > t * scale`, where `scale` is the
    /// robust window scale chosen by `scale_type`. Also writes the running
    /// median into `xmedian` and the running scale into `xsigma`, and — if
    /// `outliers` is non-null — a per-sample mask into it (`1` where the sample
    /// was flagged as an outlier, `0` otherwise). Returns the number of outliers
    /// detected. All supplied views must have length `x.len`.
    pub fn apply(
        self: Impulse,
        end: End,
        scale_type: Scale,
        t: f64,
        x: Strided(f64),
        xmedian: StridedMut(f64),
        xsigma: StridedMut(f64),
        y: StridedMut(f64),
        outliers: ?StridedMut(i32),
    ) Error!usize {
        if (x.len != y.len or x.len != xmedian.len or x.len != xsigma.len) {
            return Error.BadLength;
        }
        gsl.ensureHandler();
        var xv = constView(x);
        var yv = mutView(y);
        var med = mutView(xmedian);
        var sig = mutView(xsigma);
        var iov: c.gsl_vector_int = undefined;
        var iov_ptr: [*c]c.gsl_vector_int = null;
        if (outliers) |o| {
            if (o.len != x.len) return Error.BadLength;
            iov = .{ .size = o.len, .stride = o.stride, .data = o.ptr, .block = null, .owner = 0 };
            iov_ptr = &iov;
        }
        var noutlier: usize = 0;
        try check(c.gsl_filter_impulse(
            @intFromEnum(end),
            @intFromEnum(scale_type),
            t,
            &xv,
            &yv,
            &med,
            &sig,
            &noutlier,
            iov_ptr,
            self.ptr,
        ));
        return noutlier;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "filter: median filter removes an isolated spike" {
    const x = [_]f64{ 1, 1, 1, 5, 1, 1, 1 };
    var y = [_]f64{0} ** x.len;
    var m = try Median.init(3);
    defer m.deinit();
    try m.apply(.truncate, Strided(f64).fromSlice(&x), StridedMut(f64).fromSlice(&y));
    // The width-3 median annihilates the lone spike; everything is 1.
    for (y) |v| try testing.expectApproxEqAbs(@as(f64, 1.0), v, 1e-12);
}

test "filter: gaussian smoothing preserves a constant signal" {
    const x = [_]f64{7} ** 16;
    var y = [_]f64{0} ** x.len;
    var g = try Gaussian.init(7);
    defer g.deinit();
    // order 0 with a normalized kernel is a weighted average of equal values.
    try g.apply(.pad_value, 3.0, 0, Strided(f64).fromSlice(&x), StridedMut(f64).fromSlice(&y));
    for (y) |v| try testing.expectApproxEqAbs(@as(f64, 7.0), v, 1e-9);
}

test "filter: gaussian kernel is normalized and symmetric" {
    var k = [_]f64{0} ** 9;
    try Gaussian.kernel(3.0, 0, true, StridedMut(f64).fromSlice(&k));
    var sum: f64 = 0;
    for (k) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
    // Symmetric about the centre.
    for (0..k.len / 2) |i| {
        try testing.expectApproxEqAbs(k[i], k[k.len - 1 - i], 1e-12);
    }
    // The centre tap is the largest.
    for (k) |v| try testing.expect(v <= k[k.len / 2] + 1e-15);
}

test "filter: gaussian first-derivative filter estimates a constant slope on a ramp" {
    // A first-derivative Gaussian filter (order 1) of a linear ramp y = m·x + b
    // should report an (approximately) constant slope in the interior, and
    // vanish on a flat signal. The exact scaling of GSL's derivative kernel is
    // convention-dependent, so we assert constancy/sign rather than a value.
    var ramp: [24]f64 = undefined;
    for (&ramp, 0..) |*v, i| v.* = 2.0 * @as(f64, @floatFromInt(i)) + 5.0; // slope +2
    var dy = [_]f64{0} ** ramp.len;

    var g = try Gaussian.init(9);
    defer g.deinit();
    try g.apply(.pad_value, 3.0, 1, Strided(f64).fromSlice(&ramp), StridedMut(f64).fromSlice(&dy));

    // Interior samples (clear of the padded edges) are positive and mutually
    // consistent: the derivative of a straight line is constant.
    const ref = dy[ramp.len / 2];
    try testing.expect(ref > 0.0);
    for (dy[6 .. ramp.len - 6]) |v| {
        try testing.expect(v > 0.0);
        try testing.expectApproxEqRel(ref, v, 1e-6);
    }

    // A flat signal has zero first derivative everywhere.
    const flat = [_]f64{4.0} ** 20;
    var dflat = [_]f64{0} ** flat.len;
    try g.apply(.pad_value, 3.0, 1, Strided(f64).fromSlice(&flat), StridedMut(f64).fromSlice(&dflat));
    for (dflat) |v| try testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-9);
}

test "filter: gaussian derivative kernel is antisymmetric and sums to zero" {
    // The order-1 (first-derivative) Gaussian kernel integrates to zero and is
    // antisymmetric about its centre, with a zero centre tap.
    var k = [_]f64{0} ** 11;
    try Gaussian.kernel(3.0, 1, false, StridedMut(f64).fromSlice(&k));
    var sum: f64 = 0;
    for (k) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f64, 0.0), sum, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), k[k.len / 2], 1e-12);
    for (0..k.len / 2) |i| {
        try testing.expectApproxEqAbs(k[i], -k[k.len - 1 - i], 1e-12);
    }
}

test "filter: recursive median runs and stays within the data range" {
    const x = [_]f64{ 2, 1, 8, 3, 2, 9, 1, 2, 3, 2 };
    var y = [_]f64{0} ** x.len;
    var r = try RecursiveMedian.init(5);
    defer r.deinit();
    try r.apply(.pad_value, Strided(f64).fromSlice(&x), StridedMut(f64).fromSlice(&y));
    // A median-type filter never introduces values outside the input range.
    for (y) |v| try testing.expect(v >= 1.0 and v <= 9.0);
}

test "filter: pad_zero boundary handling leaves the interior of a median filter intact" {
    // Exercise the End.pad_zero path through an apply (the other tests use
    // truncate/pad_value). With zero-padding at the edges, a width-3 median of a
    // constant signal is unchanged in the interior; only the very ends can dip
    // toward the zero pad.
    const x = [_]f64{5} ** 9;
    var y = [_]f64{0} ** x.len;
    var m = try Median.init(3);
    defer m.deinit();
    try m.apply(.pad_zero, Strided(f64).fromSlice(&x), StridedMut(f64).fromSlice(&y));
    for (y[1 .. y.len - 1]) |v| try testing.expectApproxEqAbs(@as(f64, 5.0), v, 1e-12);
    for (y) |v| try testing.expect(std.math.isFinite(v));
}

test "filter: impulse detection flags and replaces an outlier" {
    // A mildly varying baseline with one gross outlier at index 4.
    const x = [_]f64{ 2, 3, 2, 3, 40, 3, 2, 3, 2, 3, 2 };
    var y = [_]f64{0} ** x.len;
    var xmedian = [_]f64{0} ** x.len;
    var xsigma = [_]f64{0} ** x.len;
    var imp = try Impulse.init(5);
    defer imp.deinit();

    const n = try imp.apply(
        .truncate,
        .mad,
        3.0,
        Strided(f64).fromSlice(&x),
        StridedMut(f64).fromSlice(&xmedian),
        StridedMut(f64).fromSlice(&xsigma),
        StridedMut(f64).fromSlice(&y),
        null,
    );

    try testing.expect(n >= 1); // at least the gross outlier is caught
    // The outlier was replaced by something near the local median (~2-3).
    try testing.expect(y[4] < 10.0);
    // A clearly-inlier sample passes through untouched.
    try testing.expectApproxEqAbs(x[0], y[0], 1e-12);
}

test "filter: impulse outlier mask agrees with the count across every scale estimator" {
    // A flat-ish baseline with a single gross spike at index 5.
    const x = [_]f64{ 10, 11, 10, 11, 10, 100, 10, 11, 10, 11, 10, 11, 10 };
    inline for (.{ Scale.mad, Scale.iqr, Scale.sn, Scale.qn }) |sc| {
        var y = [_]f64{0} ** x.len;
        var xmedian = [_]f64{0} ** x.len;
        var xsigma = [_]f64{0} ** x.len;
        var mask = [_]i32{0} ** x.len;
        var imp = try Impulse.init(7);
        defer imp.deinit();

        const n = try imp.apply(
            .truncate,
            sc,
            5.0,
            Strided(f64).fromSlice(&x),
            StridedMut(f64).fromSlice(&xmedian),
            StridedMut(f64).fromSlice(&xsigma),
            StridedMut(f64).fromSlice(&y),
            StridedMut(i32).fromSlice(&mask),
        );

        // The spike is flagged, and the returned count equals the set flags.
        try testing.expectEqual(@as(i32, 1), mask[5]);
        var flags: usize = 0;
        for (mask) |m| {
            if (m != 0) flags += 1;
        }
        try testing.expectEqual(n, flags);
        // The spike is replaced by the local median; everything stays finite.
        try testing.expect(y[5] < 50.0);
        for (y) |v| try testing.expect(std.math.isFinite(v));
    }
}

test "filter: mismatched output length is rejected" {
    const x = [_]f64{ 1, 2, 3, 4 };
    var y_short = [_]f64{0} ** 3;
    var m = try Median.init(3);
    defer m.deinit();
    try testing.expectError(
        Error.BadLength,
        m.apply(.truncate, Strided(f64).fromSlice(&x), StridedMut(f64).fromSlice(&y_short)),
    );
}

test "filter: End values match movstat.End" {
    // The two boundary-handling enums are documented to share underlying values.
    try testing.expectEqual(
        @intFromEnum(gsl.movstat.End.pad_zero),
        @as(c.gsl_movstat_end_t, @intCast(@intFromEnum(End.pad_zero))),
    );
    try testing.expectEqual(
        @intFromEnum(gsl.movstat.End.pad_value),
        @as(c.gsl_movstat_end_t, @intCast(@intFromEnum(End.pad_value))),
    );
    try testing.expectEqual(
        @intFromEnum(gsl.movstat.End.truncate),
        @as(c.gsl_movstat_end_t, @intCast(@intFromEnum(End.truncate))),
    );
}
