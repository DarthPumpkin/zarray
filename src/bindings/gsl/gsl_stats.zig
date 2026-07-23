//! Idiomatic Zig bindings for the GNU Scientific Library's statistics module (`gsl_statistics`).
//!
//! This file extends `gsl.zig` with descriptive, robust, and weighted statistics
//! over strided views of supported numeric element types, reached as `gsl.stats(T)`.
//!
//! ## Omissions
//!
//!   - The `long double` module (`gsl_stats_long_double_*`) is intentionally
//!     not wrapped (ABI portability).
//!   - Signed 8-bit support depends on target `c_char` signedness, matching the
//!     underlying GSL module availability.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_statistics.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL bindings).
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code.
pub const strerror = gsl.strerror;
/// Strided read-only view. Re-exported from `gsl.zig`.
pub const Strided = gsl.Strided;
/// Strided mutable view. Re-exported from `gsl.zig`.
pub const StridedMut = gsl.StridedMut;

fn statsInfix(comptime T: type) []const u8 {
    return gsl.numericModuleInfix("gsl.stats", T);
}

/// Descriptive statistics over `Strided(T)` views, wrapping GSL's per-element-
/// type `gsl_stats_*` modules. `stats(T)` selects the module for `T` (e.g.
/// `stats(f64)` for `f64` data, `stats(i32)` for `i32` data).
///
/// Supported `T`: `f32`, `f64`, and any Zig-native integer whose size and
/// signedness match one of GSL's C modules on the target platform (`i8`/`u8`
/// through `i64`/`u64`, subject to platform C-type sizes). Instantiating with
/// an unsupported type is a compile error.
///
/// Return types follow GSL exactly. Moment, dispersion, correlation, and
/// quantile estimators — along with the *scaled* robust estimators (`mad`,
/// `snFromSorted`, `qnFromSorted`) — always return `f64`. Value-selecting
/// routines (`max`, `min`, `select`, `sn0FromSorted`, `qn0FromSorted`, and the
/// `min`/`max` fields of `minMax`) return the element type `T`.
///
/// Everything here is allocation-free; the robust estimators that need scratch
/// space take an explicit caller-provided `work` buffer whose required length
/// is given by the matching `*WorkLen` helper. `spearman`, `mad`, and `mad0`
/// scratch is `[]f64`; `Sn`/`Qn` scratch is `[]T` (with an extra `[]c_int` for
/// `Qn`).
///
/// The `*Mean`/`*MeanSd` variants correspond to GSL's `_m` forms (pass a
/// precomputed mean/sd to skip recomputing it); `*WithFixedMean` corresponds to
/// the `_with_fixed_mean` forms (known population mean, divides by `n`).
///
/// Weighted statistics exist only for floating-point element types in GSL, so
/// `stats(T).weighted` is a compile error for integer `T`.
///
/// ## Omissions
///
///   - The `long double` module (`gsl_stats_long_double_*`) is not wrapped:
///     Zig has no portable fixed-width type matching C `long double`'s ABI.
///     Use `f64`, or the raw `c.gsl_stats_long_double_*` symbols, if you need
///     it.
///   - Signed 8-bit data is unavailable on the rare targets where C `char` is
///     unsigned (e.g. some AArch64 platforms): `stats(i8)` is a compile error
///     there rather than risk misreading values. `u8` is always available
///     through the `uchar` module.
pub fn stats(comptime T: type) type {
    return struct {
        const F = Strided(T);
        const FMut = StridedMut(T);
        const p = "gsl_stats_" ++ statsInfix(T);
        const is_float = switch (@typeInfo(T)) {
            .float => true,
            else => false,
        };

        pub const MinMax = struct { min: T, max: T };
        pub const MinMaxIndex = struct { min: usize, max: usize };

        // --- Moments and dispersion (single sample) ---------------------------

        pub fn mean(x: F) f64 {
            return @field(c, p ++ "mean")(@ptrCast(x.ptr), x.stride, x.len);
        }

        /// Sample variance (divides by n-1).
        pub fn variance(x: F) f64 {
            return @field(c, p ++ "variance")(@ptrCast(x.ptr), x.stride, x.len);
        }
        /// Sample variance about a precomputed mean (divides by n-1).
        pub fn varianceMean(x: F, m: f64) f64 {
            return @field(c, p ++ "variance_m")(@ptrCast(x.ptr), x.stride, x.len, m);
        }
        /// Variance about a known population mean (divides by n).
        pub fn varianceWithFixedMean(x: F, m: f64) f64 {
            return @field(c, p ++ "variance_with_fixed_mean")(@ptrCast(x.ptr), x.stride, x.len, m);
        }

        /// Sample standard deviation (divides by n-1).
        pub fn sd(x: F) f64 {
            return @field(c, p ++ "sd")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn sdMean(x: F, m: f64) f64 {
            return @field(c, p ++ "sd_m")(@ptrCast(x.ptr), x.stride, x.len, m);
        }
        pub fn sdWithFixedMean(x: F, m: f64) f64 {
            return @field(c, p ++ "sd_with_fixed_mean")(@ptrCast(x.ptr), x.stride, x.len, m);
        }

        /// Total sum of squares about the mean.
        pub fn tss(x: F) f64 {
            return @field(c, p ++ "tss")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn tssMean(x: F, m: f64) f64 {
            return @field(c, p ++ "tss_m")(@ptrCast(x.ptr), x.stride, x.len, m);
        }

        /// Mean absolute deviation about the mean.
        pub fn absdev(x: F) f64 {
            return @field(c, p ++ "absdev")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn absdevMean(x: F, m: f64) f64 {
            return @field(c, p ++ "absdev_m")(@ptrCast(x.ptr), x.stride, x.len, m);
        }

        /// Skewness.
        pub fn skew(x: F) f64 {
            return @field(c, p ++ "skew")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn skewMeanSd(x: F, m: f64, sd_: f64) f64 {
            return @field(c, p ++ "skew_m_sd")(@ptrCast(x.ptr), x.stride, x.len, m, sd_);
        }

        /// Excess kurtosis (0 for a normal distribution).
        pub fn kurtosis(x: F) f64 {
            return @field(c, p ++ "kurtosis")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn kurtosisMeanSd(x: F, m: f64, sd_: f64) f64 {
            return @field(c, p ++ "kurtosis_m_sd")(@ptrCast(x.ptr), x.stride, x.len, m, sd_);
        }

        /// Lag-1 autocorrelation.
        pub fn lag1Autocorrelation(x: F) f64 {
            return @field(c, p ++ "lag1_autocorrelation")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn lag1AutocorrelationMean(x: F, m: f64) f64 {
            return @field(c, p ++ "lag1_autocorrelation_m")(@ptrCast(x.ptr), x.stride, x.len, m);
        }

        // --- Extrema ----------------------------------------------------------

        pub fn max(x: F) T {
            return @bitCast(@field(c, p ++ "max")(@ptrCast(x.ptr), x.stride, x.len));
        }
        pub fn min(x: F) T {
            return @bitCast(@field(c, p ++ "min")(@ptrCast(x.ptr), x.stride, x.len));
        }
        pub fn minMax(x: F) MinMax {
            var lo: T = undefined;
            var hi: T = undefined;
            @field(c, p ++ "minmax")(@ptrCast(&lo), @ptrCast(&hi), @ptrCast(x.ptr), x.stride, x.len);
            return .{ .min = lo, .max = hi };
        }
        pub fn maxIndex(x: F) usize {
            return @field(c, p ++ "max_index")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn minIndex(x: F) usize {
            return @field(c, p ++ "min_index")(@ptrCast(x.ptr), x.stride, x.len);
        }
        pub fn minMaxIndex(x: F) MinMaxIndex {
            var lo: usize = undefined;
            var hi: usize = undefined;
            @field(c, p ++ "minmax_index")(&lo, &hi, @ptrCast(x.ptr), x.stride, x.len);
            return .{ .min = lo, .max = hi };
        }

        // --- Two-sample -------------------------------------------------------

        /// Covariance of two equal-length samples.
        pub fn covariance(a: F, b: F) f64 {
            std.debug.assert(a.len == b.len);
            return @field(c, p ++ "covariance")(@ptrCast(a.ptr), a.stride, @ptrCast(b.ptr), b.stride, a.len);
        }
        pub fn covarianceMean(a: F, b: F, mean_a: f64, mean_b: f64) f64 {
            std.debug.assert(a.len == b.len);
            return @field(c, p ++ "covariance_m")(@ptrCast(a.ptr), a.stride, @ptrCast(b.ptr), b.stride, a.len, mean_a, mean_b);
        }
        /// Pearson correlation of two equal-length samples.
        pub fn correlation(a: F, b: F) f64 {
            std.debug.assert(a.len == b.len);
            return @field(c, p ++ "correlation")(@ptrCast(a.ptr), a.stride, @ptrCast(b.ptr), b.stride, a.len);
        }
        /// Required `work` length for `spearman` over `n` elements.
        pub fn spearmanWorkLen(n: usize) usize {
            return 2 * n;
        }
        /// Spearman rank correlation. `work` must be at least `spearmanWorkLen(n)`.
        pub fn spearman(a: F, b: F, work: []f64) f64 {
            std.debug.assert(a.len == b.len);
            std.debug.assert(work.len >= spearmanWorkLen(a.len));
            return @field(c, p ++ "spearman")(@ptrCast(a.ptr), a.stride, @ptrCast(b.ptr), b.stride, a.len, work.ptr);
        }
        /// Pooled variance of two samples (lengths may differ).
        pub fn pvariance(a: F, b: F) f64 {
            return @field(c, p ++ "pvariance")(@ptrCast(a.ptr), a.stride, a.len, @ptrCast(b.ptr), b.stride, b.len);
        }
        /// t-statistic for the difference of two sample means.
        pub fn ttest(a: F, b: F) f64 {
            return @field(c, p ++ "ttest")(@ptrCast(a.ptr), a.stride, a.len, @ptrCast(b.ptr), b.stride, b.len);
        }

        // --- Order statistics and quantiles -----------------------------------

        /// The `k`-th smallest element (0-based). Rearranges the input in place.
        pub fn select(x: FMut, k: usize) T {
            return @bitCast(@field(c, p ++ "select")(@ptrCast(x.ptr), x.stride, x.len, k));
        }
        /// Median. Does not require sorted input, but rearranges it in place.
        pub fn median(x: FMut) f64 {
            return @field(c, p ++ "median")(@ptrCast(x.ptr), x.stride, x.len);
        }
        /// Median of an already-ascending-sorted view (input untouched).
        pub fn medianFromSorted(sorted: F) f64 {
            return @field(c, p ++ "median_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len);
        }
        /// The `f`-quantile (0 <= f <= 1) of an already-ascending-sorted view.
        pub fn quantileFromSorted(sorted: F, f: f64) f64 {
            std.debug.assert(f >= 0.0 and f <= 1.0);
            return @field(c, p ++ "quantile_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len, f);
        }
        /// Trimmed mean of an already-ascending-sorted view; `trim` in [0, 0.5).
        pub fn trmeanFromSorted(trim: f64, sorted: F) f64 {
            return @field(c, p ++ "trmean_from_sorted_data")(trim, @ptrCast(sorted.ptr), sorted.stride, sorted.len);
        }
        /// Gastwirth robust location estimate of an already-sorted view.
        pub fn gastwirthFromSorted(sorted: F) f64 {
            return @field(c, p ++ "gastwirth_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len);
        }

        // --- Robust scale estimators (explicit work buffers) ------------------

        /// Required `work` length for `mad`/`mad0` over `n` elements.
        pub fn madWorkLen(n: usize) usize {
            return n;
        }
        /// Median absolute deviation (scaled for consistency with the sd of a
        /// normal distribution). `work` must be at least `madWorkLen(n)`.
        pub fn mad(x: F, work: []f64) f64 {
            std.debug.assert(work.len >= madWorkLen(x.len));
            return @field(c, p ++ "mad")(@ptrCast(x.ptr), x.stride, x.len, work.ptr);
        }
        /// Unscaled median absolute deviation. `work` must be `>= madWorkLen(n)`.
        pub fn mad0(x: F, work: []f64) f64 {
            std.debug.assert(work.len >= madWorkLen(x.len));
            return @field(c, p ++ "mad0")(@ptrCast(x.ptr), x.stride, x.len, work.ptr);
        }

        /// Required `work` length for `snFromSorted`/`sn0FromSorted`.
        pub fn snWorkLen(n: usize) usize {
            return n;
        }
        /// Rousseeuw-Croux Sn scale estimator over an already-sorted view.
        pub fn snFromSorted(sorted: F, work: []T) f64 {
            std.debug.assert(work.len >= snWorkLen(sorted.len));
            return @field(c, p ++ "Sn_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len, @ptrCast(work.ptr));
        }
        /// Unscaled Sn over an already-sorted view.
        pub fn sn0FromSorted(sorted: F, work: []T) T {
            std.debug.assert(work.len >= snWorkLen(sorted.len));
            return @bitCast(@field(c, p ++ "Sn0_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len, @ptrCast(work.ptr)));
        }

        /// Required `work` length (in `T`) for `qnFromSorted`/`qn0FromSorted`.
        pub fn qnWorkLen(n: usize) usize {
            return 3 * n;
        }
        /// Required `work_int` (c_int) length for `qnFromSorted`/`qn0FromSorted`.
        pub fn qnWorkIntLen(n: usize) usize {
            return 5 * n;
        }
        /// Rousseeuw-Croux Qn scale estimator over an already-sorted view. `work`
        /// must be `>= qnWorkLen(n)` and `work_int` `>= qnWorkIntLen(n)`.
        pub fn qnFromSorted(sorted: F, work: []T, work_int: []c_int) f64 {
            std.debug.assert(work.len >= qnWorkLen(sorted.len));
            std.debug.assert(work_int.len >= qnWorkIntLen(sorted.len));
            return @field(c, p ++ "Qn_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len, @ptrCast(work.ptr), work_int.ptr);
        }
        /// Unscaled Qn over an already-sorted view.
        pub fn qn0FromSorted(sorted: F, work: []T, work_int: []c_int) T {
            std.debug.assert(work.len >= qnWorkLen(sorted.len));
            std.debug.assert(work_int.len >= qnWorkIntLen(sorted.len));
            return @bitCast(@field(c, p ++ "Qn0_from_sorted_data")(@ptrCast(sorted.ptr), sorted.stride, sorted.len, @ptrCast(work.ptr), work_int.ptr));
        }

        /// Weighted statistics. Each takes a weights view `w` and a data view `x`
        /// of equal length; wraps GSL's `gsl_stats_w*` family. GSL only provides
        /// these for floating-point element types, so this is a compile error
        /// for integer `T`.
        pub const weighted = if (is_float) struct {
            pub fn mean(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wmean")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn variance(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wvariance")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn varianceMean(w: F, x: F, wmean: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wvariance_m")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, wmean);
            }
            pub fn varianceWithFixedMean(w: F, x: F, mean_: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wvariance_with_fixed_mean")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, mean_);
            }
            pub fn sd(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wsd")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn sdMean(w: F, x: F, wmean: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wsd_m")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, wmean);
            }
            pub fn sdWithFixedMean(w: F, x: F, mean_: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wsd_with_fixed_mean")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, mean_);
            }
            pub fn tss(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wtss")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn tssMean(w: F, x: F, wmean: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wtss_m")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, wmean);
            }
            pub fn absdev(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wabsdev")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn absdevMean(w: F, x: F, wmean: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wabsdev_m")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, wmean);
            }
            pub fn skew(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wskew")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn skewMeanSd(w: F, x: F, wmean: f64, wsd: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wskew_m_sd")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, wmean, wsd);
            }
            pub fn kurtosis(w: F, x: F) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wkurtosis")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len);
            }
            pub fn kurtosisMeanSd(w: F, x: F, wmean: f64, wsd: f64) f64 {
                std.debug.assert(w.len == x.len);
                return @field(c, p ++ "wkurtosis_m_sd")(@ptrCast(w.ptr), w.stride, @ptrCast(x.ptr), x.stride, x.len, wmean, wsd);
            }
        } else @compileError("gsl.stats: weighted statistics are only available for floating-point element types (f32, f64); GSL provides no weighted routines for integer types");
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test "stats: basic descriptive statistics" {
    const data = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const v = Strided(f64).fromSlice(&data);
    try testing.expectApproxEqAbs(@as(f64, 5.0), stats(f64).mean(v), 1e-12);
    // Sample variance (n-1) of this classic dataset is 32/7.
    try testing.expectApproxEqAbs(@as(f64, 32.0 / 7.0), stats(f64).variance(v), 1e-12);
    try testing.expectEqual(@as(f64, 9.0), stats(f64).max(v));
    try testing.expectEqual(@as(f64, 2.0), stats(f64).min(v));
    const mm = stats(f64).minMax(v);
    try testing.expectEqual(@as(f64, 2.0), mm.min);
    try testing.expectEqual(@as(f64, 9.0), mm.max);
    // Data is already sorted, so we can take the median directly.
    try testing.expectApproxEqAbs(@as(f64, 4.5), stats(f64).medianFromSorted(v), 1e-12);
}

test "stats: strided view selects every other element" {
    // Even indices hold the dataset; odd indices are noise to be skipped.
    const interleaved = [_]f64{ 2, -9, 4, -9, 4, -9, 4, -9, 5, -9, 5, -9, 7, -9, 9, -9 };
    const v = Strided(f64).init(&interleaved, 2, 8);
    try testing.expectApproxEqAbs(@as(f64, 5.0), stats(f64).mean(v), 1e-12);
    try testing.expectEqual(@as(f64, 9.0), stats(f64).max(v));
}

test "stats: select and median rearrange in place" {
    var data = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    // 0-based 3rd smallest is 3.
    try testing.expectEqual(@as(f64, 3.0), stats(f64).select(.fromSlice(&data), 2));

    var data2 = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    try testing.expectEqual(@as(f64, 3.0), stats(f64).median(.fromSlice(&data2)));
}

test "stats: weighted mean matches unweighted for equal weights" {
    const x = [_]f64{ 1, 2, 3, 4 };
    const w = [_]f64{ 1, 1, 1, 1 };
    try testing.expectApproxEqAbs(
        stats(f64).mean(.fromSlice(&x)),
        stats(f64).weighted.mean(.fromSlice(&w), .fromSlice(&x)),
        1e-12,
    );
    // All weight on the last point pulls the mean to it.
    const w2 = [_]f64{ 0, 0, 0, 1 };
    try testing.expectApproxEqAbs(@as(f64, 4.0), stats(f64).weighted.mean(.fromSlice(&w2), .fromSlice(&x)), 1e-12);
}

test "stats: robust scale estimators with explicit work buffers" {
    // Already-sorted data with a single gross outlier.
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 };
    const v = Strided(f64).fromSlice(&data);
    const n = v.len;

    var work: [40]f64 = undefined; // >= max(madWorkLen, snWorkLen, qnWorkLen) = 3n
    var work_int: [50]c_int = undefined; // >= qnWorkIntLen = 5n

    const m = stats(f64).mad(v, work[0..stats(f64).madWorkLen(n)]);
    const s = stats(f64).snFromSorted(v, work[0..stats(f64).snWorkLen(n)]);
    const q = stats(f64).qnFromSorted(v, work[0..stats(f64).qnWorkLen(n)], work_int[0..stats(f64).qnWorkIntLen(n)]);

    // Robust scales ignore the outlier, staying far below the (inflated) sd.
    try testing.expect(m > 0 and s > 0 and q > 0);
    try testing.expect(m < stats(f64).sd(v));
    try testing.expect(s < stats(f64).sd(v));
    try testing.expect(q < stats(f64).sd(v));
}

test "stats: correlation of a perfectly linear relationship is 1" {
    const x = [_]f64{ 1, 2, 3, 4, 5 };
    const y = [_]f64{ 3, 5, 7, 9, 11 }; // y = 2x + 1
    try testing.expectApproxEqAbs(@as(f64, 1.0), stats(f64).correlation(.fromSlice(&x), .fromSlice(&y)), 1e-12);
}

test "stats: integer element type mirrors the f64 module's results" {
    const S = stats(i32);
    const idata = [_]i32{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const iv = Strided(i32).fromSlice(&idata);

    // Moments are accumulated in double precision, so they match the f64 module.
    const fdata = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const fv = Strided(f64).fromSlice(&fdata);
    try testing.expectApproxEqAbs(stats(f64).mean(fv), S.mean(iv), 1e-12);
    try testing.expectApproxEqAbs(stats(f64).variance(fv), S.variance(iv), 1e-12);

    // Value-selecting routines return the element type itself.
    const hi: i32 = S.max(iv);
    const lo: i32 = S.min(iv);
    try testing.expectEqual(@as(i32, 9), hi);
    try testing.expectEqual(@as(i32, 2), lo);
    const mm = S.minMax(iv);
    try testing.expect(@TypeOf(mm.min) == i32);
    try testing.expectEqual(@as(i32, 2), mm.min);
    try testing.expectEqual(@as(i32, 9), mm.max);
}

test "stats: select over an unsigned integer view returns the element type" {
    const S = stats(u16);
    var data = [_]u16{ 5, 3, 1, 4, 2 };
    const third: u16 = S.select(StridedMut(u16).fromSlice(&data), 2);
    try testing.expectEqual(@as(u16, 3), third);
}

test "stats: f32 element type, including weighted statistics" {
    const S = stats(f32);
    const x = [_]f32{ 1, 2, 3, 4 };
    const w = [_]f32{ 1, 1, 1, 1 };
    try testing.expectApproxEqAbs(@as(f64, 2.5), S.mean(Strided(f32).fromSlice(&x)), 1e-6);
    // Weighted mean with equal weights collapses to the plain mean.
    try testing.expectApproxEqAbs(
        S.mean(.fromSlice(&x)),
        S.weighted.mean(.fromSlice(&w), .fromSlice(&x)),
        1e-6,
    );
}

test "stats: integer robust scale estimators use T-typed work buffers" {
    const S = stats(i32);
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 };
    const v = Strided(i32).fromSlice(&data);
    const n = v.len;

    var work: [30]i32 = undefined; // >= qnWorkLen = 3n
    var work_int: [50]c_int = undefined; // >= qnWorkIntLen = 5n

    // Unscaled estimators return the element type; the scaled Sn returns f64.
    const s0: i32 = S.sn0FromSorted(v, work[0..S.snWorkLen(n)]);
    const q0: i32 = S.qn0FromSorted(v, work[0..S.qnWorkLen(n)], work_int[0..S.qnWorkIntLen(n)]);
    const s: f64 = S.snFromSorted(v, work[0..S.snWorkLen(n)]);

    try testing.expect(s0 > 0 and q0 > 0);
    try testing.expect(std.math.isFinite(s) and s > 0);
}
test "stats: every descriptive routine is invoked and cross-checked" {
    const S = stats(f64);
    const data = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const v = Strided(f64).fromSlice(&data);
    const n = data.len;
    const m = S.mean(v);
    const s = S.sd(v);

    // The `_m` variants must agree with the plain ones when given the true mean/sd.
    try testing.expectApproxEqAbs(S.variance(v), S.varianceMean(v, m), 1e-10);
    try testing.expectApproxEqAbs(S.sd(v), S.sdMean(v, m), 1e-10);
    try testing.expectApproxEqAbs(S.tss(v), S.tssMean(v, m), 1e-10);
    try testing.expectApproxEqAbs(S.absdev(v), S.absdevMean(v, m), 1e-10);
    try testing.expectApproxEqAbs(S.skew(v), S.skewMeanSd(v, m, s), 1e-10);
    try testing.expectApproxEqAbs(S.kurtosis(v), S.kurtosisMeanSd(v, m, s), 1e-10);
    try testing.expectApproxEqAbs(S.lag1Autocorrelation(v), S.lag1AutocorrelationMean(v, m), 1e-10);

    // The fixed-mean forms divide by n instead of n-1.
    try testing.expectApproxEqAbs(
        S.variance(v) * @as(f64, @floatFromInt(n - 1)) / @as(f64, @floatFromInt(n)),
        S.varianceWithFixedMean(v, m),
        1e-10,
    );
    try testing.expectApproxEqAbs(
        S.sdWithFixedMean(v, m) * S.sdWithFixedMean(v, m),
        S.varianceWithFixedMean(v, m),
        1e-10,
    );

    // Index extrema locate the known min (2 @ idx 0) and max (9 @ idx 7).
    try testing.expectEqual(@as(usize, 0), S.minIndex(v));
    try testing.expectEqual(@as(usize, 7), S.maxIndex(v));
    const mmi = S.minMaxIndex(v);
    try testing.expectEqual(@as(usize, 0), mmi.min);
    try testing.expectEqual(@as(usize, 7), mmi.max);

    // Quantile at 0.5 equals the median; a 0-trim mean equals the plain mean.
    try testing.expectApproxEqAbs(S.medianFromSorted(v), S.quantileFromSorted(v, 0.5), 1e-12);
    try testing.expectApproxEqAbs(m, S.trmeanFromSorted(0.0, v), 1e-10);
    try testing.expect(std.math.isFinite(S.gastwirthFromSorted(v)));

    // Scaled vs. unscaled robust scale estimators (also invokes every work-buffer path).
    var work: [24]f64 = undefined; // >= qnWorkLen(8) = 24
    var work_int: [40]c_int = undefined; // >= qnWorkIntLen(8) = 40
    const mad_scaled = S.mad(v, work[0..S.madWorkLen(n)]);
    const mad_raw = S.mad0(v, work[0..S.madWorkLen(n)]);
    const sn = S.snFromSorted(v, work[0..S.snWorkLen(n)]);
    const sn0 = S.sn0FromSorted(v, work[0..S.snWorkLen(n)]);
    const qn = S.qnFromSorted(v, work[0..S.qnWorkLen(n)], work_int[0..S.qnWorkIntLen(n)]);
    const qn0 = S.qn0FromSorted(v, work[0..S.qnWorkLen(n)], work_int[0..S.qnWorkIntLen(n)]);
    try testing.expect(mad_scaled > mad_raw and mad_raw > 0);
    try testing.expect(sn > 0 and sn0 > 0 and qn > 0 and qn0 > 0);

    // Two-sample routines.
    const data2 = [_]f64{ 1, 3, 3, 5, 4, 6, 8, 10 };
    const w = Strided(f64).fromSlice(&data2);
    try testing.expectApproxEqAbs(
        S.covariance(v, w),
        S.covarianceMean(v, w, S.mean(v), S.mean(w)),
        1e-10,
    );
    try testing.expect(std.math.isFinite(S.correlation(v, w)));
    try testing.expect(std.math.isFinite(S.pvariance(v, w)));
    try testing.expect(std.math.isFinite(S.ttest(v, w)));
    var swork: [16]f64 = undefined; // >= spearmanWorkLen(8) = 16
    try testing.expect(std.math.isFinite(S.spearman(v, w, swork[0..S.spearmanWorkLen(n)])));
}

test "stats: weighted routines match unweighted under equal weights" {
    const S = stats(f64);
    const x = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const wt = [_]f64{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const xv = Strided(f64).fromSlice(&x);
    const wv = Strided(f64).fromSlice(&wt);
    const m = S.mean(xv);

    // With unit weights, GSL's weighted estimators reduce to the unweighted ones.
    const wm = S.weighted.mean(wv, xv);
    const wv_var = S.weighted.variance(wv, xv);
    const wsd = S.weighted.sd(wv, xv);
    try testing.expectApproxEqAbs(m, wm, 1e-9);
    try testing.expectApproxEqAbs(S.variance(xv), wv_var, 1e-9);
    try testing.expectApproxEqAbs(S.sd(xv), wsd, 1e-9);
    try testing.expectApproxEqAbs(S.tss(xv), S.weighted.tss(wv, xv), 1e-9);
    try testing.expectApproxEqAbs(S.absdev(xv), S.weighted.absdev(wv, xv), 1e-9);
    try testing.expectApproxEqAbs(S.skew(xv), S.weighted.skew(wv, xv), 1e-9);
    try testing.expectApproxEqAbs(S.kurtosis(xv), S.weighted.kurtosis(wv, xv), 1e-9);

    // The `_m` weighted forms agree with their auto-mean counterparts.
    try testing.expectApproxEqAbs(wv_var, S.weighted.varianceMean(wv, xv, wm), 1e-9);
    try testing.expectApproxEqAbs(wsd, S.weighted.sdMean(wv, xv, wm), 1e-9);
    try testing.expectApproxEqAbs(S.weighted.tss(wv, xv), S.weighted.tssMean(wv, xv, wm), 1e-9);
    try testing.expectApproxEqAbs(S.weighted.absdev(wv, xv), S.weighted.absdevMean(wv, xv, wm), 1e-9);
    try testing.expectApproxEqAbs(S.weighted.skew(wv, xv), S.weighted.skewMeanSd(wv, xv, wm, wsd), 1e-9);
    try testing.expectApproxEqAbs(S.weighted.kurtosis(wv, xv), S.weighted.kurtosisMeanSd(wv, xv, wm, wsd), 1e-9);

    // The fixed-mean weighted forms divide by the summed weight (= n here).
    try testing.expectApproxEqAbs(
        S.varianceWithFixedMean(xv, m),
        S.weighted.varianceWithFixedMean(wv, xv, m),
        1e-9,
    );
    try testing.expectApproxEqAbs(
        S.sdWithFixedMean(xv, m),
        S.weighted.sdWithFixedMean(wv, xv, m),
        1e-9,
    );
}

test "stats: every supported integer module instantiates and computes" {
    // One representative dataset whose mean is 3 and whose extrema are 1 and 5,
    // instantiated across every integer element type GSL provides a module for.
    inline for (.{ u8, i16, u16, i32, u32, i64, u64, c_short, c_ushort, c_int, c_uint, c_long, c_ulong, c_char }) |T| {
        const S = stats(T);
        const data = [_]T{ 1, 2, 3, 4, 5 };
        const v = Strided(T).fromSlice(&data);
        try testing.expectApproxEqAbs(@as(f64, 3.0), S.mean(v), 1e-12);
        try testing.expectEqual(@as(T, 5), S.max(v));
        try testing.expectEqual(@as(T, 1), S.min(v));
    }

    // Signed 8-bit maps onto GSL's `char` module only where C `char` is signed.
    if (@typeInfo(c_char).int.signedness == .signed) {
        const S = stats(i8);
        const data = [_]i8{ 1, 2, 3, 4, 5 };
        const v = Strided(i8).fromSlice(&data);
        try testing.expectApproxEqAbs(@as(f64, 3.0), S.mean(v), 1e-12);
        try testing.expectEqual(@as(i8, 5), S.max(v));
    }
}
test "stats: unsupported instantiations are rejected at comptime" {
    // Toggle any block to `if (true)` to manually verify it throws a
    // compileError. These guards live in `statsInfix`/`Stats` and can't be
    // exercised by a normal (passing) test, so they're checked by hand.

    // No GSL statistics module matches a 128-bit integer element type.
    if (false) {
        _ = statsInfix(i128);
    }

    // f16 has no corresponding GSL floating-point module (only f32/f64 exist).
    if (false) {
        _ = statsInfix(f16);
    }

    // Weighted statistics exist only for floating-point element types, so
    // reaching `weighted` on an integer specialization is a compile error.
    if (false) {
        _ = stats(i32).weighted;
    }
}
