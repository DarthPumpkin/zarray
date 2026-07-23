//! # Running (streaming) statistics (`gsl_rstat`)
//!
//! Accumulate statistics over a stream of `f64` values in O(1) memory, without
//! storing the data. Feed values one at a time (or in bulk with `addSlice`) and
//! query moments/extremes at any point. `f64`-only (GSL provides no other
//! element type here), so this is a plain namespace rather than `rstat(T)`.
//!
//! This wraps all of `gsl_rstat.h` — nothing is omitted. Note that the moments
//! (mean/variance/sd/skew/kurtosis) and extremes are exact, but the median and
//! `Quantile` are estimated with the streaming P² algorithm (approximate, and
//! order-sensitive); use `stats(f64).median`/`quantileFromSorted` on buffered
//! data if you need an exact quantile.
//!
//! ## Omissions
//!
//! None — this wraps all of `gsl_rstat.h`.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");
const rstat = @This();
const rand = gsl.rand;
const stats = gsl.stats;
const Strided = gsl.Strided;

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_rstat.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL bindings).
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code.
pub const strerror = gsl.strerror;

/// A streaming accumulator (`gsl_rstat_workspace`). Owns its GSL allocation;
/// call `deinit` to free it.
///
/// Example:
/// ```
/// var acc = try gsl.rstat.Accumulator.init();
/// defer acc.deinit();
/// acc.addSlice(&.{ 1.0, 2.0, 3.0, 4.0 });
/// const m = acc.mean();   // 2.5
/// const s = acc.sd();     // sample sd
/// ```
pub const Accumulator = struct {
    ptr: *c.gsl_rstat_workspace,

    /// Allocate an empty accumulator. Fails only if the underlying
    /// allocation fails.
    pub fn init() error{OutOfMemory}!Accumulator {
        const p = c.gsl_rstat_alloc() orelse return error.OutOfMemory;
        return .{ .ptr = p };
    }
    pub fn deinit(self: Accumulator) void {
        c.gsl_rstat_free(self.ptr);
    }
    /// Discard all accumulated data, returning to the empty state.
    pub fn reset(self: Accumulator) void {
        _ = c.gsl_rstat_reset(self.ptr);
    }

    /// Add a single sample.
    pub fn add(self: Accumulator, x: f64) void {
        _ = c.gsl_rstat_add(x, self.ptr);
    }
    /// Add every sample in `xs`.
    pub fn addSlice(self: Accumulator, xs: []const f64) void {
        for (xs) |x| _ = c.gsl_rstat_add(x, self.ptr);
    }
    /// Number of samples added so far.
    pub fn count(self: Accumulator) usize {
        return c.gsl_rstat_n(self.ptr);
    }

    pub fn mean(self: Accumulator) f64 {
        return c.gsl_rstat_mean(self.ptr);
    }
    /// Sample variance (divides by n-1).
    pub fn variance(self: Accumulator) f64 {
        return c.gsl_rstat_variance(self.ptr);
    }
    /// Sample standard deviation (divides by n-1).
    pub fn sd(self: Accumulator) f64 {
        return c.gsl_rstat_sd(self.ptr);
    }
    /// Standard deviation of the mean, `sd / sqrt(n)`.
    pub fn sdMean(self: Accumulator) f64 {
        return c.gsl_rstat_sd_mean(self.ptr);
    }
    /// Root mean square of the samples.
    pub fn rms(self: Accumulator) f64 {
        return c.gsl_rstat_rms(self.ptr);
    }
    /// Euclidean norm (`sqrt(sum of squares)`) of the samples.
    pub fn norm(self: Accumulator) f64 {
        return c.gsl_rstat_norm(self.ptr);
    }
    pub fn skew(self: Accumulator) f64 {
        return c.gsl_rstat_skew(self.ptr);
    }
    pub fn kurtosis(self: Accumulator) f64 {
        return c.gsl_rstat_kurtosis(self.ptr);
    }
    pub fn min(self: Accumulator) f64 {
        return c.gsl_rstat_min(self.ptr);
    }
    pub fn max(self: Accumulator) f64 {
        return c.gsl_rstat_max(self.ptr);
    }
    /// Running median, estimated with the P² algorithm (approximate for a
    /// stream; exact medians require the full data).
    pub fn median(self: Accumulator) f64 {
        return c.gsl_rstat_median(self.ptr);
    }
};

/// A streaming estimator for a single `p`-quantile via the P² algorithm
/// (`gsl_rstat_quantile_workspace`): dynamic quantile tracking in O(1)
/// memory. Owns its GSL allocation; call `deinit` to free it.
///
/// Example:
/// ```
/// var q = try gsl.rstat.Quantile.init(0.5);  // running median
/// defer q.deinit();
/// for (data) |x| q.add(x);
/// const med = q.get();
/// ```
pub const Quantile = struct {
    ptr: *c.gsl_rstat_quantile_workspace,

    /// Allocate an estimator for the `p`-quantile (`0 < p < 1`). Fails only
    /// if the underlying allocation fails.
    pub fn init(p: f64) error{OutOfMemory}!Quantile {
        const ws = c.gsl_rstat_quantile_alloc(p) orelse return error.OutOfMemory;
        return .{ .ptr = ws };
    }
    pub fn deinit(self: Quantile) void {
        c.gsl_rstat_quantile_free(self.ptr);
    }
    pub fn reset(self: Quantile) void {
        _ = c.gsl_rstat_quantile_reset(self.ptr);
    }
    /// Add a single sample.
    pub fn add(self: Quantile, x: f64) void {
        _ = c.gsl_rstat_quantile_add(x, self.ptr);
    }
    /// Current estimate of the `p`-quantile.
    pub fn get(self: Quantile) f64 {
        return c.gsl_rstat_quantile_get(self.ptr);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test "rstat: streaming accumulator matches whole-array stats" {
    const data = [_]f64{ 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };

    var acc = try rstat.Accumulator.init();
    defer acc.deinit();
    acc.addSlice(&data);

    // Cross-check the streaming moments against the batch `stats(f64)` module,
    // which is the oracle here (same library, same estimator definitions).
    const S = stats(f64);
    const v = Strided(f64).fromSlice(&data);

    try testing.expectEqual(@as(usize, data.len), acc.count());
    try testing.expectApproxEqAbs(S.mean(v), acc.mean(), 1e-12);
    try testing.expectApproxEqAbs(S.variance(v), acc.variance(), 1e-12);
    try testing.expectApproxEqAbs(S.sd(v), acc.sd(), 1e-12);
    // rstat's sdMean is the standard error of the mean, sd / sqrt(n).
    try testing.expectApproxEqAbs(S.sd(v) / @sqrt(@as(f64, data.len)), acc.sdMean(), 1e-12);
    try testing.expectApproxEqAbs(S.skew(v), acc.skew(), 1e-9);
    try testing.expectApproxEqAbs(S.kurtosis(v), acc.kurtosis(), 1e-9);
    try testing.expectEqual(@as(f64, 2.0), acc.min());
    try testing.expectEqual(@as(f64, 9.0), acc.max());

    // rms = sqrt(mean of squares); norm = sqrt(sum of squares).
    var sumsq: f64 = 0;
    for (data) |x| sumsq += x * x;
    try testing.expectApproxEqAbs(@sqrt(sumsq / data.len), acc.rms(), 1e-12);
    try testing.expectApproxEqAbs(@sqrt(sumsq), acc.norm(), 1e-12);

    // The P²-estimated median lands near the exact median (5.0 -> midpoint of
    // the two central 5s... here the exact median is 4.5).
    try testing.expectApproxEqAbs(@as(f64, 4.5), acc.median(), 1.0);

    // reset returns to the empty state; a fresh stream recomputes from scratch.
    acc.reset();
    try testing.expectEqual(@as(usize, 0), acc.count());
    acc.add(1.0);
    acc.add(3.0);
    try testing.expectEqual(@as(usize, 2), acc.count());
    try testing.expectApproxEqAbs(@as(f64, 2.0), acc.mean(), 1e-12);
}

test "rstat: P² quantile estimator approaches the true quantile" {
    // Feed a proper uniform(0,1) i.i.d. sample (what P² is designed for); the
    // 0.5- and 0.9-quantile estimates should approach 0.5 and 0.9.
    var median_est = try rstat.Quantile.init(0.5);
    defer median_est.deinit();
    var p90 = try rstat.Quantile.init(0.9);
    defer p90.deinit();

    var rng = try rand.Rng.init(.mt19937);
    defer rng.deinit();
    rng.seed(20240521);
    for (0..5000) |_| {
        const x = rng.uniform();
        median_est.add(x);
        p90.add(x);
    }
    try testing.expectApproxEqAbs(@as(f64, 0.5), median_est.get(), 0.03);
    try testing.expectApproxEqAbs(@as(f64, 0.9), p90.get(), 0.03);

    median_est.reset();
    median_est.add(0.25);
    try testing.expectApproxEqAbs(@as(f64, 0.25), median_est.get(), 1e-12);
}
