//! Idiomatic Zig bindings for the GNU Scientific Library combination chapter
//! (`gsl_combination`).
//!
//! This file extends `gsl.zig` with owning combination handles, reached as
//! `gsl.combination`.
//!
//! ## Shape of the surface
//!
//! A `Combination` represents choose-`k`-from-`n` with strictly increasing
//! elements. It supports constructors (`init`, `initFirst`), in-place resets
//! (`resetFirst`, `initLast`), cloning, validation, indexed access, and
//! lexicographic `next`/`prev` iteration.
//!
//! `next`/`prev` use `bool` control flow (`true` advanced, `false` exhausted)
//! even though GSL reports exhaustion as `GSL_FAILURE`.
//!
//! ## Omissions
//!
//!   - Project-wide, all `FILE*` forms are intentionally not bound.
//!   - Copy-into-existing (`gsl_combination_memcpy`) is intentionally omitted;
//!     use `clone`, which allocates a fresh independent copy.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_combination.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for combination routines.
pub const Error = error{
    /// `GSL_EINVAL` — invalid argument (e.g. `k > n`).
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

/// Owning wrapper for `gsl_combination`.
pub const Combination = struct {
    ptr: *c.gsl_combination,

    /// Allocate `(n, k)` combination storage (contents unspecified until reset).
    pub fn init(n_items: usize, k_items: usize) Error!Combination {
        if (k_items > n_items) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_combination_alloc(n_items, k_items) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate and initialize to the first combination (`{0,1,...,k-1}`).
    pub fn initFirst(n_items: usize, k_items: usize) Error!Combination {
        if (k_items > n_items) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_combination_calloc(n_items, k_items) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Combination) void {
        c.gsl_combination_free(self.ptr);
    }

    /// Reset in place to the first combination (`{0,1,...,k-1}`).
    pub fn resetFirst(self: Combination) void {
        c.gsl_combination_init_first(self.ptr);
    }

    /// Reset in place to the last combination (`{n-k,...,n-1}`).
    pub fn initLast(self: Combination) void {
        c.gsl_combination_init_last(self.ptr);
    }

    /// Allocate an independent copy.
    pub fn clone(self: Combination) Error!Combination {
        var out = try init(self.n(), self.k());
        errdefer out.deinit();
        gsl.ensureHandler();
        try check(c.gsl_combination_memcpy(out.ptr, self.ptr));
        return out;
    }

    pub fn n(self: Combination) usize {
        return c.gsl_combination_n(self.ptr);
    }

    pub fn k(self: Combination) usize {
        return c.gsl_combination_k(self.ptr);
    }

    /// Borrow the underlying combination data (`k()` elements). Lifetime is tied
    /// to `self`.
    pub fn slice(self: Combination) []const usize {
        const kk = self.k();
        if (kk == 0) return &[_]usize{};
        return @as([*]const usize, @ptrCast(c.gsl_combination_data(self.ptr)))[0..kk];
    }

    pub fn get(self: Combination, i: usize) usize {
        std.debug.assert(i < self.k());
        return c.gsl_combination_get(self.ptr, i);
    }

    /// Whether this container currently holds a valid combination.
    pub fn isValid(self: Combination) bool {
        return c.gsl_combination_valid(self.ptr) == c.GSL_SUCCESS;
    }

    /// Advance to the next combination in lexicographic order.
    /// Returns `false` when already at the last one.
    pub fn next(self: Combination) Error!bool {
        gsl.ensureHandler();
        const status = c.gsl_combination_next(self.ptr);
        if (status == c.GSL_SUCCESS) return true;
        if (status == c.GSL_FAILURE) return false;
        try check(status);
        unreachable;
    }

    /// Move to the previous combination in lexicographic order.
    /// Returns `false` when already at the first one.
    pub fn prev(self: Combination) Error!bool {
        gsl.ensureHandler();
        const status = c.gsl_combination_prev(self.ptr);
        if (status == c.GSL_SUCCESS) return true;
        if (status == c.GSL_FAILURE) return false;
        try check(status);
        unreachable;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "combination: initFirst and full lexicographic enumeration" {
    var c2 = try Combination.initFirst(4, 2);
    defer c2.deinit();

    try testing.expect(c2.isValid());
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, c2.slice());

    var seen: [6][2]usize = undefined;
    var count: usize = 0;
    while (true) {
        const s = c2.slice();
        // Invariants: strictly increasing and in range [0, n).
        try testing.expect(s[0] < s[1]);
        try testing.expect(s[0] < c2.n() and s[1] < c2.n());

        seen[count] = .{ s[0], s[1] };
        count += 1;
        if (!(try c2.next())) break;
    }

    try testing.expectEqual(@as(usize, 6), count);
    const expected = [_][2]usize{
        .{ 0, 1 },
        .{ 0, 2 },
        .{ 0, 3 },
        .{ 1, 2 },
        .{ 1, 3 },
        .{ 2, 3 },
    };
    for (expected, 0..) |e, i| try testing.expectEqualSlices(usize, &e, &seen[i]);

    try testing.expect(!(try c2.next()));
}

test "combination: clone and prev()" {
    var c2 = try Combination.initFirst(5, 3);
    defer c2.deinit();
    try testing.expect(try c2.next());
    try testing.expect(try c2.next());

    var copied = try c2.clone();
    defer copied.deinit();
    try testing.expectEqualSlices(usize, c2.slice(), copied.slice());

    try testing.expect(try copied.prev());
    try testing.expect(copied.isValid());
}

test "combination: invalid parameters are rejected" {
    try testing.expectError(Error.Invalid, Combination.init(3, 4));
    try testing.expectError(Error.Invalid, Combination.initFirst(1, 2));
}
