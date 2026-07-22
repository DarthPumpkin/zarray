//! Idiomatic Zig bindings for the GNU Scientific Library multiset chapter
//! (`gsl_multiset`).
//!
//! This file extends `gsl.zig` with owning multiset handles, reached as
//! `gsl.multiset`.
//!
//! ## Shape of the surface
//!
//! A `Multiset` represents choose-`k`-from-`n` **with replacement** and sorted
//! order (elements are non-decreasing). It supports constructors (`init`,
//! `initFirst`), in-place resets (`resetFirst`, `initLast`), cloning,
//! validation, indexed access, and lexicographic `next`/`prev` iteration.
//!
//! `next`/`prev` use `bool` control flow (`true` advanced, `false` exhausted)
//! even though GSL reports exhaustion as `GSL_FAILURE`.
//!
//! ## Omissions
//!
//!   - Project-wide, all `FILE*` forms are intentionally not bound.
//!   - Copy-into-existing (`gsl_multiset_memcpy`) is intentionally omitted;
//!     use `clone`, which allocates a fresh independent copy.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_multiset.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for multiset routines.
pub const Error = error{
    /// `GSL_EINVAL` — invalid argument.
    Invalid,
    /// Size mismatch between inputs/outputs.
    BadLength,
    /// `GSL_ENOMEM` — allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADLEN => Error.BadLength,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// Owning wrapper for `gsl_multiset`.
pub const Multiset = struct {
    ptr: *c.gsl_multiset,

    /// Allocate `(n, k)` multiset storage (contents unspecified until reset).
    pub fn init(n_items: usize, k_items: usize) Error!Multiset {
        gsl.ensureHandler();
        const p = c.gsl_multiset_alloc(n_items, k_items) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate and initialize to the first multiset (`{0,0,...,0}`).
    pub fn initFirst(n_items: usize, k_items: usize) Error!Multiset {
        gsl.ensureHandler();
        const p = c.gsl_multiset_calloc(n_items, k_items) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Multiset) void {
        c.gsl_multiset_free(self.ptr);
    }

    /// Reset in place to the first multiset (`{0,0,...,0}`).
    pub fn resetFirst(self: Multiset) void {
        c.gsl_multiset_init_first(self.ptr);
    }

    /// Reset in place to the last multiset (`{n-1,...,n-1}`).
    pub fn initLast(self: Multiset) void {
        c.gsl_multiset_init_last(self.ptr);
    }

    /// Allocate an independent copy.
    pub fn clone(self: Multiset) Error!Multiset {
        var out = try init(self.n(), self.k());
        errdefer out.deinit();
        gsl.ensureHandler();
        try check(c.gsl_multiset_memcpy(out.ptr, self.ptr));
        return out;
    }

    pub fn n(self: Multiset) usize {
        return c.gsl_multiset_n(self.ptr);
    }

    pub fn k(self: Multiset) usize {
        return c.gsl_multiset_k(self.ptr);
    }

    /// Borrow the underlying multiset data (`k()` elements). Lifetime is tied to
    /// `self`.
    pub fn slice(self: Multiset) []const usize {
        const kk = self.k();
        if (kk == 0) return &[_]usize{};
        return @as([*]const usize, @ptrCast(c.gsl_multiset_data(self.ptr)))[0..kk];
    }

    pub fn get(self: Multiset, i: usize) usize {
        std.debug.assert(i < self.k());
        return c.gsl_multiset_get(self.ptr, i);
    }

    /// Whether this container currently holds a valid multiset.
    pub fn isValid(self: Multiset) bool {
        return c.gsl_multiset_valid(self.ptr) == c.GSL_SUCCESS;
    }

    /// Advance to the next multiset in lexicographic order.
    /// Returns `false` when already at the last one.
    pub fn next(self: Multiset) Error!bool {
        gsl.ensureHandler();
        const status = c.gsl_multiset_next(self.ptr);
        if (status == c.GSL_SUCCESS) return true;
        if (status == c.GSL_FAILURE) return false;
        try check(status);
        unreachable;
    }

    /// Move to the previous multiset in lexicographic order.
    /// Returns `false` when already at the first one.
    pub fn prev(self: Multiset) Error!bool {
        gsl.ensureHandler();
        const status = c.gsl_multiset_prev(self.ptr);
        if (status == c.GSL_SUCCESS) return true;
        if (status == c.GSL_FAILURE) return false;
        try check(status);
        unreachable;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "multiset: initFirst and full lexicographic enumeration" {
    var m = try Multiset.initFirst(3, 2);
    defer m.deinit();

    try testing.expect(m.isValid());
    try testing.expectEqualSlices(usize, &.{ 0, 0 }, m.slice());

    var seen: [6][2]usize = undefined;
    var count: usize = 0;
    while (true) {
        const s = m.slice();
        // Invariants: non-decreasing and in range [0, n).
        try testing.expect(s[0] <= s[1]);
        try testing.expect(s[0] < m.n() and s[1] < m.n());

        seen[count] = .{ s[0], s[1] };
        count += 1;
        if (!(try m.next())) break;
    }

    try testing.expectEqual(@as(usize, 6), count);
    const expected = [_][2]usize{
        .{ 0, 0 },
        .{ 0, 1 },
        .{ 0, 2 },
        .{ 1, 1 },
        .{ 1, 2 },
        .{ 2, 2 },
    };
    for (expected, 0..) |e, i| try testing.expectEqualSlices(usize, &e, &seen[i]);

    try testing.expect(!(try m.next()));
}

test "multiset: clone and prev()" {
    var m = try Multiset.initFirst(4, 3);
    defer m.deinit();
    try testing.expect(try m.next());
    try testing.expect(try m.next());

    var copied = try m.clone();
    defer copied.deinit();
    try testing.expectEqualSlices(usize, m.slice(), copied.slice());

    try testing.expect(try copied.prev());
    try testing.expect(copied.isValid());
}
