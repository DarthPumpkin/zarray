//! Idiomatic Zig bindings for GNU Scientific Library permutation chapters
//! (`gsl_permutation`, `gsl_permute`).
//!
//! This file extends `gsl.zig` with owning permutation handles and
//! element-type-specialized permutation application to strided arrays, reached
//! as `gsl.permutation`.
//!
//! ## Shape of the surface
//!
//!   - `Permutation` owns a `gsl_permutation` and exposes identity/init,
//!     `swap`/`reverse`, validity checks, inverse/composition, cycle metrics,
//!     and lexicographic `next`/`prev` iteration.
//!   - `permute(T)` exposes `apply`/`applyInverse` for `gsl_permute_*` over
//!     `StridedMut(T)`.
//!
//! `next`/`prev` use `bool` control flow (`true` advanced, `false` exhausted)
//! even though GSL reports exhaustion as `GSL_FAILURE`.
//!
//! ## Omissions
//!
//!   - Project-wide, all `FILE*` forms are intentionally not bound.
//!   - Copy-into-existing (`gsl_permutation_memcpy`) is intentionally omitted;
//!     use `clone`, which allocates a fresh independent copy.
//!   - Vector/matrix permutation convenience variants
//!     (`gsl_permute_vector*`/`gsl_permute_matrix*`) are omitted; pass strided
//!     views over caller-owned storage instead.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_permutation.h");
    @cInclude("gsl/gsl_permute.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;
/// Strided mutable view. Re-exported from `gsl.zig`.
pub const StridedMut = gsl.StridedMut;

/// Zig error set for permutation routines.
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

/// Owning wrapper for `gsl_permutation`.
pub const Permutation = struct {
    ptr: *c.gsl_permutation,

    /// Allocate a permutation of size `n` (contents unspecified until set).
    pub fn init(n: usize) Error!Permutation {
        gsl.ensureHandler();
        const p = c.gsl_permutation_alloc(n) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Allocate an identity permutation of size `n`.
    pub fn initIdentity(n: usize) Error!Permutation {
        gsl.ensureHandler();
        const p = c.gsl_permutation_calloc(n) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Permutation) void {
        c.gsl_permutation_free(self.ptr);
    }

    /// Reset in place to identity.
    pub fn reset(self: Permutation) void {
        c.gsl_permutation_init(self.ptr);
    }

    /// Allocate an independent copy of this permutation.
    pub fn clone(self: Permutation) Error!Permutation {
        var out = try init(self.size());
        errdefer out.deinit();
        gsl.ensureHandler();
        try check(c.gsl_permutation_memcpy(out.ptr, self.ptr));
        return out;
    }

    /// Number of elements.
    pub fn size(self: Permutation) usize {
        return c.gsl_permutation_size(self.ptr);
    }

    /// Borrow the underlying permutation data (`size()` elements). Lifetime is
    /// tied to `self`.
    pub fn slice(self: Permutation) []const usize {
        const n = self.size();
        if (n == 0) return &[_]usize{};
        return @as([*]const usize, @ptrCast(c.gsl_permutation_data(self.ptr)))[0..n];
    }

    /// `self[i]`.
    pub fn get(self: Permutation, i: usize) usize {
        std.debug.assert(i < self.size());
        return c.gsl_permutation_get(self.ptr, i);
    }

    /// Swap two entries.
    pub fn swap(self: Permutation, i: usize, j: usize) Error!void {
        gsl.ensureHandler();
        try check(c.gsl_permutation_swap(self.ptr, i, j));
    }

    /// Whether this is a valid permutation.
    pub fn isValid(self: Permutation) bool {
        return c.gsl_permutation_valid(self.ptr) == c.GSL_SUCCESS;
    }

    /// Reverse the permutation in place.
    pub fn reverse(self: Permutation) void {
        c.gsl_permutation_reverse(self.ptr);
    }

    /// Allocate the inverse permutation.
    pub fn inverse(self: Permutation) Error!Permutation {
        var inv = try init(self.size());
        errdefer inv.deinit();
        gsl.ensureHandler();
        try check(c.gsl_permutation_inverse(inv.ptr, self.ptr));
        return inv;
    }

    /// Advance to the next lexicographic permutation.
    /// Returns `false` when already at the last one.
    pub fn next(self: Permutation) Error!bool {
        gsl.ensureHandler();
        const status = c.gsl_permutation_next(self.ptr);
        if (status == c.GSL_SUCCESS) return true;
        if (status == c.GSL_FAILURE) return false;
        try check(status);
        unreachable;
    }

    /// Move to the previous lexicographic permutation.
    /// Returns `false` when already at the first one.
    pub fn prev(self: Permutation) Error!bool {
        gsl.ensureHandler();
        const status = c.gsl_permutation_prev(self.ptr);
        if (status == c.GSL_SUCCESS) return true;
        if (status == c.GSL_FAILURE) return false;
        try check(status);
        unreachable;
    }

    /// Compose two permutations and return the result.
    pub fn mul(pa: Permutation, pb: Permutation) Error!Permutation {
        if (pa.size() != pb.size()) return Error.BadLength;
        var out = try init(pa.size());
        errdefer out.deinit();
        gsl.ensureHandler();
        try check(c.gsl_permutation_mul(out.ptr, pa.ptr, pb.ptr));
        return out;
    }

    /// Convert linear-cycle notation to canonical-cycle notation.
    pub fn linearToCanonical(self: Permutation) Error!Permutation {
        var out = try init(self.size());
        errdefer out.deinit();
        gsl.ensureHandler();
        try check(c.gsl_permutation_linear_to_canonical(out.ptr, self.ptr));
        return out;
    }

    /// Convert canonical-cycle notation to linear-cycle notation.
    pub fn canonicalToLinear(self: Permutation) Error!Permutation {
        var out = try init(self.size());
        errdefer out.deinit();
        gsl.ensureHandler();
        try check(c.gsl_permutation_canonical_to_linear(out.ptr, self.ptr));
        return out;
    }

    pub fn inversions(self: Permutation) usize {
        return c.gsl_permutation_inversions(self.ptr);
    }

    pub fn linearCycles(self: Permutation) usize {
        return c.gsl_permutation_linear_cycles(self.ptr);
    }

    pub fn canonicalCycles(self: Permutation) usize {
        return c.gsl_permutation_canonical_cycles(self.ptr);
    }
};

/// Element-type-specialized wrappers for `gsl_permute_*`.
pub fn permute(comptime T: type) type {
    return struct {
        const FMut = StridedMut(T);
        const stem = gsl.numericModuleStem("gsl.permute", T);

        const fn_apply = if (stem.len == 0) "gsl_permute" else "gsl_permute_" ++ stem;
        const fn_apply_inverse = if (stem.len == 0) "gsl_permute_inverse" else "gsl_permute_" ++ stem ++ "_inverse";

        /// Apply permutation `p` to `data` in place.
        pub fn apply(p: Permutation, data: FMut) Error!void {
            if (data.len != p.size()) return Error.BadLength;
            gsl.ensureHandler();
            try check(@field(c, fn_apply)(p.slice().ptr, @ptrCast(data.ptr), data.stride, data.len));
        }

        /// Apply the inverse of permutation `p` to `data` in place.
        pub fn applyInverse(p: Permutation, data: FMut) Error!void {
            if (data.len != p.size()) return Error.BadLength;
            gsl.ensureHandler();
            try check(@field(c, fn_apply_inverse)(p.slice().ptr, @ptrCast(data.ptr), data.stride, data.len));
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "permutation: identity, lexicographic next(), and terminal false" {
    var p = try Permutation.initIdentity(3);
    defer p.deinit();

    try testing.expect(p.isValid());
    for (0..p.size()) |i| try testing.expectEqual(i, p.get(i));

    var seen: [6][3]usize = undefined;
    var count: usize = 0;
    while (true) {
        const s = p.slice();
        seen[count] = .{ s[0], s[1], s[2] };
        count += 1;
        if (!(try p.next())) break;
    }
    try testing.expectEqual(@as(usize, 6), count);

    const expected = [_][3]usize{
        .{ 0, 1, 2 },
        .{ 0, 2, 1 },
        .{ 1, 0, 2 },
        .{ 1, 2, 0 },
        .{ 2, 0, 1 },
        .{ 2, 1, 0 },
    };
    for (expected, 0..) |e, i| try testing.expectEqualSlices(usize, &e, &seen[i]);

    try testing.expect(!(try p.next()));
}

test "permutation: inverse and composition produce identity" {
    var p = try Permutation.initIdentity(5);
    defer p.deinit();

    try p.swap(0, 4);
    try p.swap(1, 3);
    try testing.expect(p.isValid());

    var inv = try p.inverse();
    defer inv.deinit();

    var composed = try Permutation.mul(p, inv);
    defer composed.deinit();

    for (0..composed.size()) |i| {
        try testing.expectEqual(i, composed.get(i));
    }
}

test "permutation: reverse, inversions, and cycle notation roundtrip" {
    var p = try Permutation.initIdentity(4);
    defer p.deinit();

    p.reverse();
    try testing.expectEqualSlices(usize, &.{ 3, 2, 1, 0 }, p.slice());
    try testing.expectEqual(@as(usize, 6), p.inversions());

    var canonical = try p.linearToCanonical();
    defer canonical.deinit();
    var linear = try canonical.canonicalToLinear();
    defer linear.deinit();

    try testing.expectEqualSlices(usize, p.slice(), linear.slice());
}

test "permute(T): apply and applyInverse are inverses" {
    const P = permute(i32);

    var p = try Permutation.initIdentity(5);
    defer p.deinit();
    try testing.expect(try p.next());
    try testing.expect(try p.next());

    var x = [_]i32{ 10, 20, 30, 40, 50 };
    const original = x;

    try P.apply(p, StridedMut(i32).fromSlice(&x));
    try P.applyInverse(p, StridedMut(i32).fromSlice(&x));
    try testing.expectEqualSlices(i32, &original, &x);

    var short = [_]i32{ 1, 2, 3 };
    try testing.expectError(Error.BadLength, P.apply(p, StridedMut(i32).fromSlice(&short)));
}
