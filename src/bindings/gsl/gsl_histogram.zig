//! Idiomatic Zig bindings for the GNU Scientific Library's histogram modules
//! (`gsl_histogram`, `gsl_histogram2d`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the histogram
//! chapter. It reuses that module's process-global error-handler switch (so a
//! GSL domain error surfaces as a Zig `Error` instead of aborting), but keeps
//! the histogram-specific C API behind its own `c` (the `gsl_histogram*`
//! headers are not pulled in by `gsl.zig`). It is reached as `gsl.histogram`.
//!
//! ## Shape of the surface
//!
//! A histogram is a set of `n` contiguous bins covering `[range[0], range[n]]`,
//! each holding a (weighted) count. GSL separates *allocation* from *setting
//! the bin ranges*; these bindings fuse the two into the `init*` constructors
//! (per the house style) and expose `setRanges`/`setRangesUniform`/`reset` for
//! the reuse path.
//!
//!   - `Histogram` owns a `gsl_histogram`. Build it with `init` (uniform
//!     integer ranges `0..n`), `initUniform` (`n` equal-width bins over
//!     `[min, max)`), or `initWithRanges` (explicit, strictly-increasing bin
//!     edges). Fill it with `increment`/`accumulate`, query bins and summary
//!     statistics, and combine two same-binning histograms with the in-place
//!     arithmetic (`add`/`sub`/`mul`/`div`/`scale`/`shift`).
//!   - `Pdf` turns a filled `Histogram` into an empirical distribution you can
//!     sample from with a uniform deviate (e.g. `rng.uniform()`).
//!   - `Histogram2d`/`Pdf2d` mirror the above for `(x, y)` pairs.
//!
//! ## Omissions
//!
//!   - The `FILE*` serialization forms (`gsl_histogram_fread`/`fwrite`/
//!     `fprintf`/`fscanf`, and the 2-D equivalents) are intentionally omitted as
//!     non-idiomatic; reach them through the raw `c.gsl_histogram_*` symbols.
//!   - Copy-into-an-existing-instance (`gsl_histogram_memcpy`) is not exposed;
//!     use `clone`, which allocates a fresh independent copy.
//!   - `init` uses GSL's `calloc` form (bins zeroed, default `0..n` integer
//!     ranges); the bare `gsl_histogram_alloc` (uninitialized ranges) is not
//!     surfaced — set the ranges you want with `setRanges`/`setRangesUniform`,
//!     or build with `initUniform`/`initWithRanges`.
//!   - `get` (and the 2-D `get`) return 0 for an out-of-range index rather than
//!     erroring, mirroring GSL; the installed no-op handler keeps that from
//!     aborting.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for the `FILE*` serialization forms and any
/// symbol not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_histogram.h");
    @cInclude("gsl/gsl_histogram2d.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the histogram routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_histogram_*` symbol if you need the
/// exact code.
pub const Error = error{
    /// `GSL_EDOM` — a value lies outside the histogram's range, or an index is
    /// out of bounds.
    Domain,
    /// `GSL_EINVAL` — invalid input, e.g. combining two histograms whose bins
    /// do not match, or ranges that are not strictly increasing.
    Invalid,
    /// `GSL_EBADLEN`, or a caller-supplied ranges/size that is inconsistent
    /// with the histogram (`ranges.len != bins + 1`), or `bins == 0`.
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

/// The lower/upper edges of a single bin.
pub const Range = struct { lower: f64, upper: f64 };
/// A pair of bin indices, e.g. the location of the extremal bin of a 2-D
/// histogram.
pub const BinIndex = struct { i: usize, j: usize };

// ===========================================================================
// 1-D histogram
// ===========================================================================

/// A 1-D histogram (`gsl_histogram`): `n` bins over `[range[0], range[n]]`, each
/// holding a (weighted) count. Owns its GSL allocation; call `deinit` to free.
///
/// Example:
/// ```
/// var h = try gsl.histogram.Histogram.initUniform(10, 0.0, 1.0);
/// defer h.deinit();
/// try h.increment(0.3);
/// try h.accumulate(0.7, 2.5);
/// const total = h.sum();
/// ```
pub const Histogram = struct {
    ptr: *c.gsl_histogram,

    /// Allocate a histogram of `bins` bins with default uniform integer ranges
    /// `[0, 1, ..., bins]` and all bins zeroed. Set the ranges afterwards with
    /// `setRanges`/`setRangesUniform` if you need something else.
    pub fn init(n_bins: usize) Error!Histogram {
        if (n_bins == 0) return Error.BadLength;
        gsl.ensureHandler();
        const p = c.gsl_histogram_calloc(n_bins) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate a histogram of `bins` equal-width bins spanning `[min, max)`,
    /// all bins zeroed. Requires `min < max`.
    pub fn initUniform(n_bins: usize, lo: f64, hi: f64) Error!Histogram {
        if (n_bins == 0) return Error.BadLength;
        if (!(lo < hi)) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_histogram_calloc_uniform(n_bins, lo, hi) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate a histogram whose bin edges are given explicitly by `ranges`
    /// (which must be strictly increasing). The histogram has `ranges.len - 1`
    /// bins, all zeroed; requires `ranges.len >= 2`.
    pub fn initWithRanges(ranges: []const f64) Error!Histogram {
        if (ranges.len < 2) return Error.BadLength;
        gsl.ensureHandler();
        const p = c.gsl_histogram_calloc(ranges.len - 1) orelse return Error.OutOfMemory;
        errdefer c.gsl_histogram_free(p);
        try check(c.gsl_histogram_set_ranges(p, ranges.ptr, ranges.len));
        return .{ .ptr = p };
    }

    pub fn deinit(self: Histogram) void {
        c.gsl_histogram_free(self.ptr);
    }

    /// Allocate an independent copy of this histogram (ranges and bin contents).
    pub fn clone(self: Histogram) Error!Histogram {
        const p = c.gsl_histogram_clone(self.ptr) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Zero all bin counts, leaving the ranges intact.
    pub fn reset(self: Histogram) void {
        c.gsl_histogram_reset(self.ptr);
    }

    /// Replace the bin edges with `ranges` (strictly increasing); requires
    /// `ranges.len == bins() + 1`. Also zeroes the bins.
    pub fn setRanges(self: Histogram, ranges: []const f64) Error!void {
        try check(c.gsl_histogram_set_ranges(self.ptr, ranges.ptr, ranges.len));
    }

    /// Replace the bin edges with `bins()` equal-width bins over `[min, max)`.
    /// Also zeroes the bins.
    pub fn setRangesUniform(self: Histogram, lo: f64, hi: f64) Error!void {
        try check(c.gsl_histogram_set_ranges_uniform(self.ptr, lo, hi));
    }

    // --- fill ---

    /// Add 1 to the bin containing `x`. Returns `Error.Domain` if `x` lies
    /// outside `[min(), max())`.
    pub fn increment(self: Histogram, x: f64) Error!void {
        try check(c.gsl_histogram_increment(self.ptr, x));
    }

    /// Add `weight` to the bin containing `x`. Returns `Error.Domain` if `x`
    /// lies outside `[min(), max())`.
    pub fn accumulate(self: Histogram, x: f64, weight: f64) Error!void {
        try check(c.gsl_histogram_accumulate(self.ptr, x, weight));
    }

    // --- query ---

    /// The number of bins.
    pub fn bins(self: Histogram) usize {
        return c.gsl_histogram_bins(self.ptr);
    }

    /// The (weighted) count in bin `i`. Returns 0 if `i >= bins()`.
    pub fn get(self: Histogram, i: usize) f64 {
        return c.gsl_histogram_get(self.ptr, i);
    }

    /// The lower/upper edges of bin `i`.
    pub fn getRange(self: Histogram, i: usize) Error!Range {
        var r: Range = undefined;
        try check(c.gsl_histogram_get_range(self.ptr, i, &r.lower, &r.upper));
        return r;
    }

    /// The index of the bin containing `x`, or `null` if `x` is out of range.
    pub fn find(self: Histogram, x: f64) ?usize {
        var i: usize = undefined;
        return switch (c.gsl_histogram_find(self.ptr, x, &i)) {
            c.GSL_SUCCESS => i,
            else => null,
        };
    }

    /// The lower bound of the histogram's range (`range[0]`).
    pub fn min(self: Histogram) f64 {
        return c.gsl_histogram_min(self.ptr);
    }
    /// The upper bound of the histogram's range (`range[n]`).
    pub fn max(self: Histogram) f64 {
        return c.gsl_histogram_max(self.ptr);
    }

    /// The maximum bin count.
    pub fn maxVal(self: Histogram) f64 {
        return c.gsl_histogram_max_val(self.ptr);
    }
    /// The index of the bin with the maximum count.
    pub fn maxBin(self: Histogram) usize {
        return c.gsl_histogram_max_bin(self.ptr);
    }
    /// The minimum bin count.
    pub fn minVal(self: Histogram) f64 {
        return c.gsl_histogram_min_val(self.ptr);
    }
    /// The index of the bin with the minimum count.
    pub fn minBin(self: Histogram) usize {
        return c.gsl_histogram_min_bin(self.ptr);
    }

    /// The mean of the histogram, treating bin midpoints as values weighted by
    /// their counts.
    pub fn mean(self: Histogram) f64 {
        return c.gsl_histogram_mean(self.ptr);
    }
    /// The standard deviation of the histogram (see `mean`).
    pub fn sigma(self: Histogram) f64 {
        return c.gsl_histogram_sigma(self.ptr);
    }
    /// The sum of all bin counts.
    pub fn sum(self: Histogram) f64 {
        return c.gsl_histogram_sum(self.ptr);
    }

    // --- whole-histogram arithmetic (in place on self) ---

    /// `self[i] += other[i]` for every bin. Requires identical binning.
    pub fn add(self: Histogram, other: Histogram) Error!void {
        try check(c.gsl_histogram_add(self.ptr, other.ptr));
    }
    /// `self[i] -= other[i]` for every bin. Requires identical binning.
    pub fn sub(self: Histogram, other: Histogram) Error!void {
        try check(c.gsl_histogram_sub(self.ptr, other.ptr));
    }
    /// `self[i] *= other[i]` for every bin. Requires identical binning.
    pub fn mul(self: Histogram, other: Histogram) Error!void {
        try check(c.gsl_histogram_mul(self.ptr, other.ptr));
    }
    /// `self[i] /= other[i]` for every bin. Requires identical binning.
    pub fn div(self: Histogram, other: Histogram) Error!void {
        try check(c.gsl_histogram_div(self.ptr, other.ptr));
    }
    /// Multiply every bin count by `s`.
    pub fn scale(self: Histogram, s: f64) void {
        _ = c.gsl_histogram_scale(self.ptr, s);
    }
    /// Add `s` to every bin count.
    pub fn shift(self: Histogram, s: f64) void {
        _ = c.gsl_histogram_shift(self.ptr, s);
    }

    /// Whether `self` and `other` have the same number of bins and identical
    /// bin edges.
    pub fn equalBins(self: Histogram, other: Histogram) bool {
        return c.gsl_histogram_equal_bins_p(self.ptr, other.ptr) != 0;
    }
};

/// An empirical probability distribution derived from a filled `Histogram`
/// (`gsl_histogram_pdf`). Sample it with a uniform deviate in `[0, 1)`. Owns its
/// GSL allocation; call `deinit` to free.
pub const Pdf = struct {
    ptr: *c.gsl_histogram_pdf,

    /// Build a sampling distribution from `h`. Requires all bin counts to be
    /// non-negative (otherwise `Error.Domain`).
    pub fn init(h: Histogram) Error!Pdf {
        gsl.ensureHandler();
        const p = c.gsl_histogram_pdf_alloc(c.gsl_histogram_bins(h.ptr)) orelse
            return Error.OutOfMemory;
        errdefer c.gsl_histogram_pdf_free(p);
        try check(c.gsl_histogram_pdf_init(p, h.ptr));
        return .{ .ptr = p };
    }

    pub fn deinit(self: Pdf) void {
        c.gsl_histogram_pdf_free(self.ptr);
    }

    /// Map a uniform deviate `r` in `[0, 1)` to a sample from the distribution.
    pub fn sample(self: Pdf, r: f64) f64 {
        return c.gsl_histogram_pdf_sample(self.ptr, r);
    }
};

// ===========================================================================
// 2-D histogram
// ===========================================================================

/// A 2-D histogram (`gsl_histogram2d`): an `nx * ny` grid of bins over
/// `[xrange[0], xrange[nx]] x [yrange[0], yrange[ny]]`, each holding a
/// (weighted) count. Owns its GSL allocation; call `deinit` to free.
pub const Histogram2d = struct {
    ptr: *c.gsl_histogram2d,

    /// Allocate an `nx * ny` histogram with default uniform integer ranges and
    /// all bins zeroed.
    pub fn init(nx_bins: usize, ny_bins: usize) Error!Histogram2d {
        if (nx_bins == 0 or ny_bins == 0) return Error.BadLength;
        gsl.ensureHandler();
        const p = c.gsl_histogram2d_calloc(nx_bins, ny_bins) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate an `nx * ny` histogram of equal-width bins over
    /// `[xmin, xmax) x [ymin, ymax)`, all bins zeroed.
    pub fn initUniform(
        nx_bins: usize,
        ny_bins: usize,
        x_lo: f64,
        x_hi: f64,
        y_lo: f64,
        y_hi: f64,
    ) Error!Histogram2d {
        if (nx_bins == 0 or ny_bins == 0) return Error.BadLength;
        if (!(x_lo < x_hi) or !(y_lo < y_hi)) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_histogram2d_calloc_uniform(nx_bins, ny_bins, x_lo, x_hi, y_lo, y_hi) orelse
            return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate a histogram with explicit, strictly-increasing bin edges. The
    /// histogram has `xranges.len - 1` by `yranges.len - 1` bins, all zeroed;
    /// requires each of `xranges`/`yranges` to have length >= 2.
    pub fn initWithRanges(xranges: []const f64, yranges: []const f64) Error!Histogram2d {
        if (xranges.len < 2 or yranges.len < 2) return Error.BadLength;
        gsl.ensureHandler();
        const p = c.gsl_histogram2d_calloc(xranges.len - 1, yranges.len - 1) orelse
            return Error.OutOfMemory;
        errdefer c.gsl_histogram2d_free(p);
        try check(c.gsl_histogram2d_set_ranges(p, xranges.ptr, xranges.len, yranges.ptr, yranges.len));
        return .{ .ptr = p };
    }

    pub fn deinit(self: Histogram2d) void {
        c.gsl_histogram2d_free(self.ptr);
    }

    /// Allocate an independent copy of this histogram.
    pub fn clone(self: Histogram2d) Error!Histogram2d {
        const p = c.gsl_histogram2d_clone(self.ptr) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Zero all bin counts, leaving the ranges intact.
    pub fn reset(self: Histogram2d) void {
        c.gsl_histogram2d_reset(self.ptr);
    }

    /// Replace both axes' bin edges (each strictly increasing); requires
    /// `xranges.len == nx() + 1` and `yranges.len == ny() + 1`. Zeroes the bins.
    pub fn setRanges(self: Histogram2d, xranges: []const f64, yranges: []const f64) Error!void {
        try check(c.gsl_histogram2d_set_ranges(
            self.ptr,
            xranges.ptr,
            xranges.len,
            yranges.ptr,
            yranges.len,
        ));
    }

    /// Replace both axes with equal-width bins over `[xmin, xmax) x [ymin, ymax)`.
    /// Zeroes the bins.
    pub fn setRangesUniform(self: Histogram2d, x_lo: f64, x_hi: f64, y_lo: f64, y_hi: f64) Error!void {
        try check(c.gsl_histogram2d_set_ranges_uniform(self.ptr, x_lo, x_hi, y_lo, y_hi));
    }

    // --- fill ---

    /// Add 1 to the bin containing `(x, y)`. Returns `Error.Domain` if the point
    /// is outside the histogram's ranges.
    pub fn increment(self: Histogram2d, x: f64, y: f64) Error!void {
        try check(c.gsl_histogram2d_increment(self.ptr, x, y));
    }

    /// Add `weight` to the bin containing `(x, y)`. Returns `Error.Domain` if the
    /// point is outside the histogram's ranges.
    pub fn accumulate(self: Histogram2d, x: f64, y: f64, weight: f64) Error!void {
        try check(c.gsl_histogram2d_accumulate(self.ptr, x, y, weight));
    }

    // --- query ---

    /// The number of bins along the x axis.
    pub fn nx(self: Histogram2d) usize {
        return c.gsl_histogram2d_nx(self.ptr);
    }
    /// The number of bins along the y axis.
    pub fn ny(self: Histogram2d) usize {
        return c.gsl_histogram2d_ny(self.ptr);
    }

    /// The (weighted) count in bin `(i, j)`. Returns 0 if the indices are out of
    /// range.
    pub fn get(self: Histogram2d, i: usize, j: usize) f64 {
        return c.gsl_histogram2d_get(self.ptr, i, j);
    }

    /// The lower/upper x edges of column `i`.
    pub fn getXRange(self: Histogram2d, i: usize) Error!Range {
        var r: Range = undefined;
        try check(c.gsl_histogram2d_get_xrange(self.ptr, i, &r.lower, &r.upper));
        return r;
    }
    /// The lower/upper y edges of row `j`.
    pub fn getYRange(self: Histogram2d, j: usize) Error!Range {
        var r: Range = undefined;
        try check(c.gsl_histogram2d_get_yrange(self.ptr, j, &r.lower, &r.upper));
        return r;
    }

    /// The `(i, j)` bin containing `(x, y)`, or `null` if the point is out of
    /// range.
    pub fn find(self: Histogram2d, x: f64, y: f64) ?BinIndex {
        var idx: BinIndex = undefined;
        return switch (c.gsl_histogram2d_find(self.ptr, x, y, &idx.i, &idx.j)) {
            c.GSL_SUCCESS => idx,
            else => null,
        };
    }

    pub fn xmin(self: Histogram2d) f64 {
        return c.gsl_histogram2d_xmin(self.ptr);
    }
    pub fn xmax(self: Histogram2d) f64 {
        return c.gsl_histogram2d_xmax(self.ptr);
    }
    pub fn ymin(self: Histogram2d) f64 {
        return c.gsl_histogram2d_ymin(self.ptr);
    }
    pub fn ymax(self: Histogram2d) f64 {
        return c.gsl_histogram2d_ymax(self.ptr);
    }

    /// The maximum bin count.
    pub fn maxVal(self: Histogram2d) f64 {
        return c.gsl_histogram2d_max_val(self.ptr);
    }
    /// The `(i, j)` index of the bin with the maximum count.
    pub fn maxBin(self: Histogram2d) BinIndex {
        var idx: BinIndex = undefined;
        c.gsl_histogram2d_max_bin(self.ptr, &idx.i, &idx.j);
        return idx;
    }
    /// The minimum bin count.
    pub fn minVal(self: Histogram2d) f64 {
        return c.gsl_histogram2d_min_val(self.ptr);
    }
    /// The `(i, j)` index of the bin with the minimum count.
    pub fn minBin(self: Histogram2d) BinIndex {
        var idx: BinIndex = undefined;
        c.gsl_histogram2d_min_bin(self.ptr, &idx.i, &idx.j);
        return idx;
    }

    /// The mean of the x values (bin midpoints weighted by count).
    pub fn xmean(self: Histogram2d) f64 {
        return c.gsl_histogram2d_xmean(self.ptr);
    }
    /// The mean of the y values (bin midpoints weighted by count).
    pub fn ymean(self: Histogram2d) f64 {
        return c.gsl_histogram2d_ymean(self.ptr);
    }
    /// The standard deviation of the x values.
    pub fn xsigma(self: Histogram2d) f64 {
        return c.gsl_histogram2d_xsigma(self.ptr);
    }
    /// The standard deviation of the y values.
    pub fn ysigma(self: Histogram2d) f64 {
        return c.gsl_histogram2d_ysigma(self.ptr);
    }
    /// The covariance of the x and y values.
    pub fn cov(self: Histogram2d) f64 {
        return c.gsl_histogram2d_cov(self.ptr);
    }
    /// The sum of all bin counts.
    pub fn sum(self: Histogram2d) f64 {
        return c.gsl_histogram2d_sum(self.ptr);
    }

    // --- whole-histogram arithmetic (in place on self) ---

    pub fn add(self: Histogram2d, other: Histogram2d) Error!void {
        try check(c.gsl_histogram2d_add(self.ptr, other.ptr));
    }
    pub fn sub(self: Histogram2d, other: Histogram2d) Error!void {
        try check(c.gsl_histogram2d_sub(self.ptr, other.ptr));
    }
    pub fn mul(self: Histogram2d, other: Histogram2d) Error!void {
        try check(c.gsl_histogram2d_mul(self.ptr, other.ptr));
    }
    pub fn div(self: Histogram2d, other: Histogram2d) Error!void {
        try check(c.gsl_histogram2d_div(self.ptr, other.ptr));
    }
    pub fn scale(self: Histogram2d, s: f64) void {
        _ = c.gsl_histogram2d_scale(self.ptr, s);
    }
    pub fn shift(self: Histogram2d, s: f64) void {
        _ = c.gsl_histogram2d_shift(self.ptr, s);
    }

    /// Whether `self` and `other` have identical binning on both axes.
    pub fn equalBins(self: Histogram2d, other: Histogram2d) bool {
        return c.gsl_histogram2d_equal_bins_p(self.ptr, other.ptr) != 0;
    }
};

/// A sampled `(x, y)` point from a 2-D empirical distribution.
pub const Point = struct { x: f64, y: f64 };

/// An empirical 2-D distribution derived from a filled `Histogram2d`
/// (`gsl_histogram2d_pdf`). Sample it with a pair of uniform deviates. Owns its
/// GSL allocation; call `deinit` to free.
pub const Pdf2d = struct {
    ptr: *c.gsl_histogram2d_pdf,

    /// Build a sampling distribution from `h`. Requires all bin counts to be
    /// non-negative (otherwise `Error.Domain`).
    pub fn init(h: Histogram2d) Error!Pdf2d {
        gsl.ensureHandler();
        const p = c.gsl_histogram2d_pdf_alloc(
            c.gsl_histogram2d_nx(h.ptr),
            c.gsl_histogram2d_ny(h.ptr),
        ) orelse return Error.OutOfMemory;
        errdefer c.gsl_histogram2d_pdf_free(p);
        try check(c.gsl_histogram2d_pdf_init(p, h.ptr));
        return .{ .ptr = p };
    }

    pub fn deinit(self: Pdf2d) void {
        c.gsl_histogram2d_pdf_free(self.ptr);
    }

    /// Map a pair of uniform deviates `(r1, r2)` in `[0, 1)` to a sampled
    /// `(x, y)` point.
    pub fn sample(self: Pdf2d, r1: f64, r2: f64) Error!Point {
        var out: Point = undefined;
        try check(c.gsl_histogram2d_pdf_sample(self.ptr, r1, r2, &out.x, &out.y));
        return out;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "histogram: uniform binning counts and locates values" {
    var h = try Histogram.initUniform(10, 0.0, 10.0);
    defer h.deinit();

    try testing.expectEqual(@as(usize, 10), h.bins());
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.min(), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10.0), h.max(), 1e-12);

    // Three values in bin 2 ([2,3)), one in bin 7.
    try h.increment(2.1);
    try h.increment(2.5);
    try h.accumulate(2.9, 3.0);
    try h.increment(7.4);

    try testing.expectEqual(@as(usize, 2), h.find(2.5).?);
    try testing.expectApproxEqAbs(@as(f64, 5.0), h.get(2), 1e-12); // 1 + 1 + 3
    try testing.expectApproxEqAbs(@as(f64, 1.0), h.get(7), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6.0), h.sum(), 1e-12);

    const r = try h.getRange(2);
    try testing.expectApproxEqAbs(@as(f64, 2.0), r.lower, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), r.upper, 1e-12);

    // Bin 2 dominates.
    try testing.expectEqual(@as(usize, 2), h.maxBin());
    try testing.expectApproxEqAbs(@as(f64, 5.0), h.maxVal(), 1e-12);
}

test "histogram: out-of-range fills are a domain error, and find returns null" {
    var h = try Histogram.initUniform(4, 0.0, 1.0);
    defer h.deinit();
    try testing.expectError(Error.Domain, h.increment(1.5));
    try testing.expectError(Error.Domain, h.accumulate(-0.1, 2.0));
    try testing.expect(h.find(2.0) == null);
}

test "histogram: explicit ranges, reset, and setRanges reuse" {
    const edges = [_]f64{ 0.0, 1.0, 4.0, 9.0 }; // 3 unequal bins
    var h = try Histogram.initWithRanges(&edges);
    defer h.deinit();
    try testing.expectEqual(@as(usize, 3), h.bins());

    try h.increment(2.0); // bin 1 ([1,4))
    try testing.expectApproxEqAbs(@as(f64, 1.0), h.get(1), 1e-12);
    h.reset();
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.sum(), 1e-12);

    // Reuse the allocation with a fresh uniform binning.
    try h.setRangesUniform(0.0, 3.0);
    try h.increment(2.5); // bin 2
    try testing.expectEqual(@as(usize, 2), h.find(2.5).?);
}

test "histogram: whole-histogram arithmetic requires matching bins" {
    var a = try Histogram.initUniform(5, 0.0, 5.0);
    defer a.deinit();
    var b = try a.clone();
    defer b.deinit();

    try a.increment(0.5);
    try b.increment(0.5);
    try testing.expect(a.equalBins(b));

    try a.add(b);
    try testing.expectApproxEqAbs(@as(f64, 2.0), a.get(0), 1e-12);

    a.scale(2.0);
    try testing.expectApproxEqAbs(@as(f64, 4.0), a.get(0), 1e-12);
    a.shift(1.0);
    try testing.expectApproxEqAbs(@as(f64, 5.0), a.get(0), 1e-12);

    // Different binning is rejected.
    var mismatched = try Histogram.initUniform(6, 0.0, 5.0);
    defer mismatched.deinit();
    try testing.expect(!a.equalBins(mismatched));
    try testing.expectError(Error.Invalid, a.add(mismatched));
}

test "histogram: pdf samples land in populated bins" {
    var h = try Histogram.initUniform(4, 0.0, 4.0);
    defer h.deinit();
    // Only bin 1 ([1,2)) and bin 3 ([3,4)) get any weight.
    try h.accumulate(1.5, 3.0);
    try h.accumulate(3.5, 1.0);

    var pdf = try Pdf.init(h);
    defer pdf.deinit();

    // Every sample must fall in one of the two populated bins.
    var r: f64 = 0.0;
    while (r < 1.0) : (r += 0.05) {
        const x = pdf.sample(r);
        const in_bin1 = (x >= 1.0 and x <= 2.0);
        const in_bin3 = (x >= 3.0 and x <= 4.0);
        try testing.expect(in_bin1 or in_bin3);
    }
}

test "histogram: too-few bins or ranges are rejected" {
    try testing.expectError(Error.BadLength, Histogram.init(0));
    try testing.expectError(Error.Invalid, Histogram.initUniform(4, 1.0, 1.0));
    const one_edge = [_]f64{0.0};
    try testing.expectError(Error.BadLength, Histogram.initWithRanges(&one_edge));
}

test "histogram: remaining query methods and sub/mul/div arithmetic" {
    var a = try Histogram.initUniform(4, 0.0, 4.0);
    defer a.deinit();
    try a.accumulate(0.5, 4.0); // bin 0
    try a.accumulate(1.5, 2.0); // bin 1
    try a.accumulate(2.5, 6.0); // bin 2
    try a.accumulate(3.5, 1.0); // bin 3

    // Least-populated bin is bin 3 (count 1); mean/sigma are finite.
    try testing.expectApproxEqAbs(@as(f64, 1.0), a.minVal(), 1e-12);
    try testing.expectEqual(@as(usize, 3), a.minBin());
    try testing.expect(std.math.isFinite(a.mean()));
    try testing.expect(std.math.isFinite(a.sigma()));

    var b = try a.clone();
    defer b.deinit();
    try a.mul(b); // square every bin
    try testing.expectApproxEqAbs(@as(f64, 16.0), a.get(0), 1e-9); // 4^2
    try a.div(b); // divide back (all b bins non-zero)
    try testing.expectApproxEqAbs(@as(f64, 4.0), a.get(0), 1e-9);
    try a.sub(b); // now all zero
    try testing.expectApproxEqAbs(@as(f64, 0.0), a.sum(), 1e-12);
}

test "histogram2d: remaining accessors and arithmetic" {
    var h = try Histogram2d.initUniform(2, 3, 0.0, 2.0, 0.0, 3.0);
    defer h.deinit();
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.xmin(), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), h.xmax(), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.ymin(), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), h.ymax(), 1e-12);

    const yr = try h.getYRange(2); // row 2 spans [2, 3)
    try testing.expectApproxEqAbs(@as(f64, 2.0), yr.lower, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), yr.upper, 1e-12);

    try h.increment(0.5, 0.5); // populate bin (0,0)
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.minVal(), 1e-12); // an empty bin
    _ = h.minBin(); // links and returns a valid (i, j)

    var g = try h.clone();
    defer g.deinit();
    try h.sub(g); // back to all-zero
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.sum(), 1e-12);
    h.shift(1.0); // every bin += 1
    try testing.expectApproxEqAbs(@as(f64, 1.0), h.get(0, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6.0), h.sum(), 1e-12); // 2*3 bins
}

test "histogram2d: counts, ranges, and summary statistics" {
    var h = try Histogram2d.initUniform(4, 4, 0.0, 4.0, 0.0, 4.0);
    defer h.deinit();
    try testing.expectEqual(@as(usize, 4), h.nx());
    try testing.expectEqual(@as(usize, 4), h.ny());

    try h.increment(0.5, 0.5); // bin (0,0)
    try h.accumulate(2.5, 1.5, 4.0); // bin (2,1)
    try testing.expectApproxEqAbs(@as(f64, 1.0), h.get(0, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.0), h.get(2, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0), h.sum(), 1e-12);

    const found = h.find(2.5, 1.5).?;
    try testing.expectEqual(@as(usize, 2), found.i);
    try testing.expectEqual(@as(usize, 1), found.j);

    const mx = h.maxBin();
    try testing.expectEqual(@as(usize, 2), mx.i);
    try testing.expectEqual(@as(usize, 1), mx.j);
    try testing.expectApproxEqAbs(@as(f64, 4.0), h.maxVal(), 1e-12);

    const xr = try h.getXRange(2);
    try testing.expectApproxEqAbs(@as(f64, 2.0), xr.lower, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), xr.upper, 1e-12);

    // Both summary means and covariance link and are finite.
    try testing.expect(std.math.isFinite(h.xmean()));
    try testing.expect(std.math.isFinite(h.ymean()));
    try testing.expect(std.math.isFinite(h.xsigma()));
    try testing.expect(std.math.isFinite(h.ysigma()));
    try testing.expect(std.math.isFinite(h.cov()));
}

test "histogram2d: out-of-range fill errors, clone, arithmetic, and pdf" {
    var a = try Histogram2d.initUniform(3, 3, 0.0, 3.0, 0.0, 3.0);
    defer a.deinit();
    try testing.expectError(Error.Domain, a.increment(5.0, 0.5));

    try a.accumulate(1.5, 1.5, 2.0);
    var b = try a.clone();
    defer b.deinit();
    try testing.expect(a.equalBins(b));
    try a.add(b);
    try testing.expectApproxEqAbs(@as(f64, 4.0), a.get(1, 1), 1e-12);
    a.scale(0.5);
    try testing.expectApproxEqAbs(@as(f64, 2.0), a.get(1, 1), 1e-12);

    var pdf = try Pdf2d.init(a);
    defer pdf.deinit();
    // The only populated bin is (1,1) = [1,2) x [1,2).
    const s = try pdf.sample(0.5, 0.5);
    try testing.expect(s.x >= 1.0 and s.x <= 2.0);
    try testing.expect(s.y >= 1.0 and s.y <= 2.0);
}

test "histogram2d: explicit ranges and setRanges reuse" {
    const xr = [_]f64{ 0.0, 1.0, 3.0 };
    const yr = [_]f64{ 0.0, 2.0, 5.0 };
    var h = try Histogram2d.initWithRanges(&xr, &yr);
    defer h.deinit();
    try testing.expectEqual(@as(usize, 2), h.nx());
    try testing.expectEqual(@as(usize, 2), h.ny());

    try h.increment(2.0, 3.0); // bin (1,1)
    try testing.expectApproxEqAbs(@as(f64, 1.0), h.get(1, 1), 1e-12);

    try h.setRangesUniform(0.0, 2.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), h.sum(), 1e-12); // reset by setRanges
}
