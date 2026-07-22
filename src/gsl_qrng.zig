//! Idiomatic Zig bindings for the GNU Scientific Library's quasi-random
//! sequence generators (`gsl_qrng`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the quasi-random
//! (low-discrepancy) sequence chapter. It reuses that module's process-global
//! error-handler switch but keeps the `gsl_qrng` C API behind its own `c`. It
//! is reached as `gsl.qrng`, and replaces the reserved placeholder that used to
//! live in `gsl.zig`.
//!
//! ## Shape of the surface
//!
//! Quasi-random generators produce deterministic, space-filling point sets for
//! quasi-Monte Carlo integration — they are *not* pseudo-random streams, so
//! they have no seed and live outside `gsl.rand`.
//!
//!   - `Type` selects the algorithm (Sobol, Halton, reverse-Halton,
//!     Niederreiter). Each has a maximum dimension (`Type.maxDimension`).
//!   - `Sequence` owns a `gsl_qrng` of a fixed dimension. `get` fills the next
//!     `dimension`-length point (each component in `[0, 1)`); `reset` restarts
//!     the sequence; `clone` duplicates the current position; and
//!     `saveState`/`loadState` checkpoint/restore it across a process boundary
//!     (same-binary blob, mirroring `rand.Rng`).
//!
//! ## Omissions
//!
//!   - The `FILE*` serialization forms are intentionally omitted; the
//!     byte-buffer `saveState`/`loadState` cover state persistence.
//!   - Copy-into-an-existing-instance (`gsl_qrng_memcpy`) is not exposed; use
//!     `clone`, which allocates a fresh independent copy.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for anything not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_qrng.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the quasi-random routines. The raw `c_int` status is
/// always available from the underlying `c.gsl_qrng_*` symbol if you need the
/// exact code.
pub const Error = error{
    /// A dimension of 0, or one exceeding the generator type's
    /// `maxDimension()`.
    Invalid,
    /// A caller-supplied output slice whose length differs from the sequence's
    /// dimension.
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

/// The quasi-random algorithm, mirroring GSL's `gsl_qrng_*` types.
pub const Type = enum {
    /// Sobol sequence (max dimension 40).
    sobol,
    /// Halton sequence (max dimension 1229).
    halton,
    /// Reverse-Halton sequence (max dimension 1229).
    reverse_halton,
    /// Niederreiter base-2 sequence (max dimension 12).
    niederreiter2,

    fn typePtr(self: Type) *const c.gsl_qrng_type {
        return switch (self) {
            .sobol => c.gsl_qrng_sobol,
            .halton => c.gsl_qrng_halton,
            .reverse_halton => c.gsl_qrng_reversehalton,
            .niederreiter2 => c.gsl_qrng_niederreiter_2,
        };
    }

    /// The largest space dimension this generator type supports.
    pub fn maxDimension(self: Type) u32 {
        return @intCast(self.typePtr().*.max_dimension);
    }
};

/// A quasi-random sequence generator (`gsl_qrng`) of a fixed dimension. Owns its
/// GSL allocation; call `deinit` to free.
///
/// Example:
/// ```
/// var q = try gsl.qrng.Sequence.init(.sobol, 2);
/// defer q.deinit();
/// var point: [2]f64 = undefined;
/// try q.get(&point); // next low-discrepancy point in [0,1)^2
/// ```
pub const Sequence = struct {
    ptr: *c.gsl_qrng,

    /// Allocate a generator of type `t` over `dim` dimensions. `dim` must be in
    /// `1..=t.maxDimension()`.
    pub fn init(t: Type, dim: u32) Error!Sequence {
        if (dim == 0 or dim > t.maxDimension()) return Error.Invalid;
        gsl.ensureHandler();
        const p = c.gsl_qrng_alloc(t.typePtr(), dim) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: Sequence) void {
        c.gsl_qrng_free(self.ptr);
    }

    /// Duplicate this generator, including its current position in the sequence.
    pub fn clone(self: Sequence) Error!Sequence {
        const p = c.gsl_qrng_clone(self.ptr) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    /// Restart the sequence from its beginning.
    pub fn reset(self: Sequence) void {
        c.gsl_qrng_init(self.ptr);
    }

    /// The standardized name of the generator (e.g. "sobol").
    pub fn name(self: Sequence) [:0]const u8 {
        return std.mem.span(c.gsl_qrng_name(self.ptr));
    }

    /// The space dimension this generator was allocated for.
    pub fn dimension(self: Sequence) u32 {
        return @intCast(self.ptr.*.dimension);
    }

    /// Write the next point of the sequence into `out`. `out.len` must equal
    /// `dimension()`; each component is in `[0, 1)`.
    pub fn get(self: Sequence, out: []f64) Error!void {
        if (out.len != self.dimension()) return Error.BadLength;
        try check(c.gsl_qrng_get(self.ptr, out.ptr));
    }

    // --- State serialization (mirrors rand.Rng) ---------------------------
    //
    // Checkpoint/restore of a sequence position across a process boundary. The
    // byte format is *not portable* (algorithm/platform specific); treat a
    // snapshot as an opaque, same-binary blob, not an archival format.

    /// Number of bytes `saveState` writes for this generator (runtime-known).
    pub fn stateSize(self: Sequence) usize {
        return c.gsl_qrng_size(self.ptr);
    }

    /// Snapshot the generator's internal state into `buf`, returning the written
    /// sub-slice (`buf[0..stateSize()]`). `buf.len` must be at least
    /// `stateSize()`. Restore later with `loadState`.
    pub fn saveState(self: Sequence, buf: []u8) []u8 {
        const n = self.stateSize();
        std.debug.assert(buf.len >= n);
        const src: [*]const u8 = @ptrCast(c.gsl_qrng_state(self.ptr).?);
        @memcpy(buf[0..n], src[0..n]);
        return buf[0..n];
    }

    /// Restore internal state previously captured by `saveState`. `bytes.len`
    /// must equal this generator's `stateSize()`, and the bytes must have come
    /// from the same `Type`/dimension on a compatible build.
    pub fn loadState(self: Sequence, bytes: []const u8) void {
        const n = self.stateSize();
        std.debug.assert(bytes.len == n);
        const dst: [*]u8 = @ptrCast(c.gsl_qrng_state(self.ptr).?);
        @memcpy(dst[0..n], bytes[0..n]);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "qrng: sobol points are in the unit square and deterministic" {
    var q = try Sequence.init(.sobol, 2);
    defer q.deinit();
    try testing.expectEqualStrings("sobol", q.name());
    try testing.expectEqual(@as(u32, 2), q.dimension());

    var first: [10][2]f64 = undefined;
    for (&first) |*p| {
        try q.get(p);
        try testing.expect(p[0] >= 0.0 and p[0] < 1.0);
        try testing.expect(p[1] >= 0.0 and p[1] < 1.0);
    }

    // Restarting the sequence reproduces the exact same points.
    q.reset();
    for (first) |expected| {
        var p: [2]f64 = undefined;
        try q.get(&p);
        try testing.expectEqual(expected[0], p[0]);
        try testing.expectEqual(expected[1], p[1]);
    }
}

test "qrng: a fresh generator of the same type matches point-for-point" {
    var a = try Sequence.init(.halton, 3);
    defer a.deinit();
    var b = try Sequence.init(.halton, 3);
    defer b.deinit();

    for (0..8) |_| {
        var pa: [3]f64 = undefined;
        var pb: [3]f64 = undefined;
        try a.get(&pa);
        try b.get(&pb);
        try testing.expectEqualSlices(f64, &pa, &pb);
    }
}

test "qrng: clone continues from the current position" {
    var q = try Sequence.init(.sobol, 2);
    defer q.deinit();
    // Advance a few points.
    var scratch: [2]f64 = undefined;
    for (0..5) |_| try q.get(&scratch);

    var cloned = try q.clone();
    defer cloned.deinit();

    for (0..5) |_| {
        var p: [2]f64 = undefined;
        var pc: [2]f64 = undefined;
        try q.get(&p);
        try cloned.get(&pc);
        try testing.expectEqualSlices(f64, &p, &pc);
    }
}

test "qrng: state save/load round-trips a sequence across a checkpoint" {
    var q = try Sequence.init(.niederreiter2, 4);
    defer q.deinit();
    var scratch: [4]f64 = undefined;
    for (0..3) |_| try q.get(&scratch);

    const buf = try testing.allocator.alloc(u8, q.stateSize());
    defer testing.allocator.free(buf);
    const snapshot = q.saveState(buf);

    // Draw a reference continuation.
    var expected: [6][4]f64 = undefined;
    for (&expected) |*p| try q.get(p);

    // Rewind to the checkpoint and confirm the continuation is identical.
    q.loadState(snapshot);
    for (expected) |exp| {
        var p: [4]f64 = undefined;
        try q.get(&p);
        try testing.expectEqualSlices(f64, &exp, &p);
    }
}

test "qrng: invalid dimensions and output lengths are rejected" {
    try testing.expectError(Error.Invalid, Sequence.init(.sobol, 0));
    // Sobol tops out at 40 dimensions.
    try testing.expectError(Error.Invalid, Sequence.init(.sobol, 41));

    var q = try Sequence.init(.sobol, 2);
    defer q.deinit();
    var wrong: [3]f64 = undefined;
    try testing.expectError(Error.BadLength, q.get(&wrong));
}

test "qrng: every type instantiates, names itself, and advances" {
    inline for (comptime std.enums.values(Type)) |t| {
        try testing.expect(t.maxDimension() >= 2);
        var q = try Sequence.init(t, 2);
        defer q.deinit();
        try testing.expect(q.name().len > 0);
        var p: [2]f64 = undefined;
        try q.get(&p);
        try testing.expect(p[0] >= 0.0 and p[0] < 1.0);
    }
}

test "qrng: maxDimension reflects each algorithm's limit and bounds init" {
    // Sobol tops out at 40 dimensions; the Halton family reaches into the
    // thousands; Niederreiter-2 is comparatively small but supports >= 2.
    try testing.expectEqual(@as(u32, 40), Type.sobol.maxDimension());
    try testing.expect(Type.halton.maxDimension() >= 1000);
    try testing.expect(Type.reverse_halton.maxDimension() >= 1000);
    try testing.expect(Type.niederreiter2.maxDimension() >= 2);

    // Allocating at exactly the maximum works; one past it is rejected.
    var q = try Sequence.init(.sobol, Type.sobol.maxDimension());
    defer q.deinit();
    try testing.expectEqual(@as(u32, 40), q.dimension());
    try testing.expectError(Error.Invalid, Sequence.init(.sobol, Type.sobol.maxDimension() + 1));
}
