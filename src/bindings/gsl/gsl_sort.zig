//! Idiomatic Zig bindings for the GNU Scientific Library's sorting chapter
//! (`gsl_sort`).
//!
//! This file extends `gsl.zig` with element-type-specialized sorting over
//! strided views, reached as `gsl.sort.sort(T)`.
//!
//! ## Shape of the surface
//!
//! The chapter is comptime-specialized by element type, mirroring `gsl.stats`:
//! `sort(T)` picks the matching `gsl_sort_*` module for `T` and exposes
//! in-place sort (`sort`), co-sort (`sort2`), indirect sort (`sortIndex`), and
//! top-k extraction (`smallest`/`largest` and index variants).
//!
//! ## Omissions
//!
//!   - Project-wide, all `FILE*` forms are intentionally not bound. (`gsl_sort`
//!     has no `FILE*` API.)
//!   - Copy-into-existing (`_memcpy`) forms are intentionally omitted
//!     project-wide; use chapter-specific `clone` APIs where applicable.
//!   - Vector convenience variants (`gsl_sort_vector*`) are omitted; pass
//!     `Strided(T)` / `StridedMut(T)` over caller-owned storage instead.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_sort.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;
/// Strided read-only view. Re-exported from `gsl.zig`.
pub const Strided = gsl.Strided;
/// Strided mutable view. Re-exported from `gsl.zig`.
pub const StridedMut = gsl.StridedMut;

/// Zig error set for the status-returning top-k sorting helpers.
pub const Error = error{
    /// `GSL_EINVAL` — invalid argument (e.g. `k > n`).
    Invalid,
    /// Caller-supplied output view length mismatch.
    BadLength,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADLEN => Error.BadLength,
        else => Error.Unspecified,
    };
}

/// Element-type-specialized wrappers for `gsl_sort_*`.
///
/// Supported `T`: `f32`, `f64`, and integer types whose size/signedness match
/// one of GSL's `char`/`short`/`int`/`long` modules on the target platform.
pub fn sort(comptime T: type) type {
    return struct {
        const F = Strided(T);
        const FMut = StridedMut(T);

        const stem = gsl.numericModuleStem("gsl.sort", T);
        const infix = gsl.numericModuleInfix("gsl.sort", T);

        const fn_sort = if (stem.len == 0) "gsl_sort" else "gsl_sort_" ++ stem;
        const fn_sort2 = if (stem.len == 0) "gsl_sort2" else "gsl_sort2_" ++ stem;
        const fn_sort_index = "gsl_sort_" ++ infix ++ "index";
        const fn_smallest = "gsl_sort_" ++ infix ++ "smallest";
        const fn_smallest_index = "gsl_sort_" ++ infix ++ "smallest_index";
        const fn_largest = "gsl_sort_" ++ infix ++ "largest";
        const fn_largest_index = "gsl_sort_" ++ infix ++ "largest_index";

        /// In-place ascending sort.
        pub fn sort(data: FMut) void {
            @field(c, fn_sort)(@ptrCast(data.ptr), data.stride, data.len);
        }

        /// Co-sort two arrays by `data1` (applying the same permutation to
        /// `data2`). Both views must have the same length.
        pub fn sort2(data1: FMut, data2: FMut) void {
            std.debug.assert(data1.len == data2.len);
            @field(c, fn_sort2)(@ptrCast(data1.ptr), data1.stride, @ptrCast(data2.ptr), data2.stride, data1.len);
        }

        /// Indirect ascending sort: write the sorting permutation into `dest`.
        /// `dest.len` must equal `src.len`.
        pub fn sortIndex(dest: []usize, src: F) void {
            std.debug.assert(dest.len == src.len);
            @field(c, fn_sort_index)(dest.ptr, @ptrCast(src.ptr), src.stride, src.len);
        }

        /// Write the `k` smallest elements of `src` (ascending) into `dest`.
        /// Requires `dest.len == k` and `k <= src.len`.
        pub fn smallest(dest: []T, k: usize, src: F) Error!void {
            if (dest.len != k) return Error.BadLength;
            if (k > src.len) return Error.Invalid;
            gsl.ensureHandler();
            try check(@field(c, fn_smallest)(@ptrCast(dest.ptr), k, @ptrCast(src.ptr), src.stride, src.len));
        }

        /// Write indices of the `k` smallest elements into `dest`.
        /// Requires `dest.len == k` and `k <= src.len`.
        pub fn smallestIndex(dest: []usize, k: usize, src: F) Error!void {
            if (dest.len != k) return Error.BadLength;
            if (k > src.len) return Error.Invalid;
            gsl.ensureHandler();
            try check(@field(c, fn_smallest_index)(dest.ptr, k, @ptrCast(src.ptr), src.stride, src.len));
        }

        /// Write the `k` largest elements of `src` into `dest`.
        /// Requires `dest.len == k` and `k <= src.len`.
        pub fn largest(dest: []T, k: usize, src: F) Error!void {
            if (dest.len != k) return Error.BadLength;
            if (k > src.len) return Error.Invalid;
            gsl.ensureHandler();
            try check(@field(c, fn_largest)(@ptrCast(dest.ptr), k, @ptrCast(src.ptr), src.stride, src.len));
        }

        /// Write indices of the `k` largest elements into `dest`.
        /// Requires `dest.len == k` and `k <= src.len`.
        pub fn largestIndex(dest: []usize, k: usize, src: F) Error!void {
            if (dest.len != k) return Error.BadLength;
            if (k > src.len) return Error.Invalid;
            gsl.ensureHandler();
            try check(@field(c, fn_largest_index)(dest.ptr, k, @ptrCast(src.ptr), src.stride, src.len));
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "sort: in-place sort, co-sort, and indirect permutation" {
    const S = sort(f64);

    var xs = [_]f64{ 3, 1, 4, 1, 5, 2 };
    S.sort(StridedMut(f64).fromSlice(&xs));
    try testing.expectEqualSlices(f64, &.{ 1, 1, 2, 3, 4, 5 }, &xs);

    var keys = [_]f64{ 3, 1, 2, 1 };
    var payload = [_]f64{ 30, 10, 20, 11 };
    S.sort2(StridedMut(f64).fromSlice(&keys), StridedMut(f64).fromSlice(&payload));
    try testing.expectEqualSlices(f64, &.{ 1, 1, 2, 3 }, &keys);
    // Equal keys do not guarantee stable tie ordering; only the grouped mapping
    // is constrained.
    try testing.expect(payload[2] == 20 and payload[3] == 30);
    try testing.expect((payload[0] == 10 and payload[1] == 11) or (payload[0] == 11 and payload[1] == 10));

    const raw = [_]f64{ 3, 1, 4, 1, 5, 2 };
    var p: [raw.len]usize = undefined;
    S.sortIndex(&p, Strided(f64).fromSlice(&raw));

    var indirect: [raw.len]f64 = undefined;
    for (p, 0..) |j, i| indirect[i] = raw[j];
    try testing.expectEqualSlices(f64, &.{ 1, 1, 2, 3, 4, 5 }, &indirect);
}

test "sort: top-k values and indices" {
    const S = sort(i32);
    const src = [_]i32{ 9, 4, 1, 7, 3, 8 };

    var small: [3]i32 = undefined;
    try S.smallest(&small, small.len, Strided(i32).fromSlice(&src));
    try testing.expectEqualSlices(i32, &.{ 1, 3, 4 }, &small);

    var small_idx: [3]usize = undefined;
    try S.smallestIndex(&small_idx, small_idx.len, Strided(i32).fromSlice(&src));
    var by_idx: [3]i32 = undefined;
    for (small_idx, 0..) |j, i| by_idx[i] = src[j];
    try testing.expectEqualSlices(i32, &small, &by_idx);

    var large: [2]i32 = undefined;
    try S.largest(&large, large.len, Strided(i32).fromSlice(&src));
    // GSL returns largest values in descending order.
    try testing.expectEqualSlices(i32, &.{ 9, 8 }, &large);
}

test "sort: top-k boundary and length errors" {
    const S = sort(f64);
    const src = [_]f64{ 1, 2, 3, 4 };

    var bad_len: [2]f64 = undefined;
    try testing.expectError(Error.BadLength, S.smallest(&bad_len, 3, Strided(f64).fromSlice(&src)));

    var too_many: [5]f64 = undefined;
    try testing.expectError(Error.Invalid, S.smallest(&too_many, too_many.len, Strided(f64).fromSlice(&src)));

    var idx_bad_len: [1]usize = undefined;
    try testing.expectError(Error.BadLength, S.largestIndex(&idx_bad_len, 2, Strided(f64).fromSlice(&src)));
}
