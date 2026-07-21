//! Idiomatic Zig bindings for the GNU Scientific Library's Fast Fourier
//! Transform modules (`gsl_fft_complex`, `gsl_fft_real`, and
//! `gsl_fft_halfcomplex`, in both `double`/`f64` and `float`/`f32` precisions).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the FFT chapter.
//! It reuses that module's strided-view helpers (`Strided`/`StridedMut`) and
//! error-handler switch so that the whole GSL surface stays consistent; the
//! FFT-specific C API lives behind this file's own `c` (the `gsl_fft_*`
//! headers are not pulled in by `gsl.zig`).
//!
//! ## Layout of the public surface
//!
//! Three comptime factories select the element precision `T` (`f32` or `f64`),
//! each mirroring one GSL sub-module:
//!
//!   - `complex(T)` — `gsl_fft_complex[_float]`: in-place complex→complex FFTs
//!     over packed complex data (`std.math.Complex(T)`).
//!   - `real(T)` — `gsl_fft_real[_float]`: the forward transform of real data
//!     (real→half-complex), as `real(T).radix2Forward` or a mixed-radix
//!     `real(T).Plan`.
//!   - `halfcomplex(T)` — `gsl_fft_halfcomplex[_float]`: the inverse/backward
//!     transform (half-complex→real), plus `unpack` helpers that expand a
//!     half-complex sequence into ordinary complex numbers.
//!
//! Each mixed-radix `Plan` owns its wavetable and, via `init(n)`, its own
//! scratch `Workspace` too — so the common path is one self-contained object
//! with workspace-free calls (`plan.forward(data)`). To avoid a redundant
//! workspace when a `real(T).Plan` (forward) and a `halfcomplex(T).Plan`
//! (inverse) transform the same length, build them with
//! `initWithWorkspace(n, ws)` to share a single explicitly-owned `Workspace`
//! (`real(T).Workspace` and `halfcomplex(T).Workspace` are the same type).
//!
//! GSL keeps real forward and half-complex inverse in two separate modules
//! because a real signal's spectrum is conjugate-symmetric ("half-complex") and
//! is stored compactly in `n` reals rather than `n` complex numbers. These
//! bindings preserve that split. Use `halfcomplex(T).unpack` /
//! `real(T).unpack` to expand a half-complex/real buffer into an ordinary
//! `[]Complex(T)` for inspection.
//!
//! ## Packed complex data
//!
//! GSL's complex routines operate on *packed* arrays: interleaved
//! `re, im, re, im, ...` floats. `std.math.Complex(T)` is an `extern struct
//! { re: T, im: T }` with exactly that layout, so a `[]Complex(T)` is passed
//! straight through with no copy or repacking.
//!
//! ## Strided views
//!
//! Every transform takes a `StridedMut(...)` view (re-exported from `gsl.zig`),
//! so a single row/column/axis of a larger array can be transformed in place
//! without copying. `stride`/`len` are in *element* units — complex numbers for
//! `complex(T)`, reals for `real(T)`/`halfcomplex(T)`. Use `.fromSlice(buf)`
//! for the common contiguous (stride-1) case:
//!
//! ```
//! var buf: [8]fft.Complex(f64) = ...;         // 8 complex samples
//! try fft.complex(f64).radix2Forward(.fromSlice(&buf));
//! try fft.complex(f64).radix2Inverse(.fromSlice(&buf)); // back to the input
//! ```
//!
//! ## Radix-2 vs. mixed-radix
//!
//!   - Radix-2 free functions (`radix2*`) transform in place with no scratch
//!     space, but require the length to be a power of two.
//!   - Mixed-radix works for *any* length but needs a trigonometric wavetable
//!     (per length *and* transform kind) plus a scratch workspace (per length).
//!     A `Plan` owns its wavetable and, by default (`init`), its own workspace,
//!     so the common case is a single self-contained object with workspace-free
//!     calls (`plan.forward(data)`). When several same-length plans would each
//!     allocate a redundant workspace (e.g. a real forward plan and its
//!     half-complex inverse plan), build them with `initWithWorkspace(n, ws)`
//!     to share one explicitly-owned `Workspace`. Build a plan once and reuse it
//!     for many transforms of that length. `Plan.factors()` exposes GSL's chosen
//!     factorization (large prime factors fall back to a slow O(n^2) module).
//!
//! ## Error convention
//!
//! Checkable contract violations are rejected *before* calling GSL, so they are
//! reported as Zig errors regardless of which GSL error handler is installed:
//! `error.NotPowerOfTwo` (radix-2 length), `error.LengthMismatch` (data length
//! vs. a `Plan`'s length), and `error.ZeroLength`. Any other nonzero GSL return
//! code surfaces as `error.TransformFailed`.
//!
//! Note that GSL's *default* error handler calls `abort()`, so for a genuine
//! internal GSL error you would never observe the return code. Install the
//! non-aborting handler once at startup with `disableDefaultErrorHandler()`
//! (re-exported from `gsl.zig`) if you want fallible behavior there too. With
//! the pre-validation above, the common mistakes never reach that path.
//!
//! ## Omitted from GSL
//!
//! These parts of GSL's FFT API are intentionally *not* wrapped (the raw
//! symbols remain reachable through `c`):
//!
//!   - The decimation-in-frequency radix-2 variants
//!     (`gsl_fft_complex_radix2_dif_*`) produce the same results as the bound
//!     decimation-in-time routines; only the default algorithm is surfaced.
//!   - `gsl_fft_complex_memcpy` (duplicating a wavetable) — allocate a fresh
//!     `Plan` instead.
//!   - GSL ships no `long double` FFT module, so only `f32`/`f64` exist.

const std = @import("std");
const testing = std.testing;
const gsl = @import("gsl.zig");

pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_complex.h");
    @cInclude("gsl/gsl_fft_complex.h");
    @cInclude("gsl/gsl_fft_complex_float.h");
    @cInclude("gsl/gsl_fft_real.h");
    @cInclude("gsl/gsl_fft_real_float.h");
    @cInclude("gsl/gsl_fft_halfcomplex.h");
    @cInclude("gsl/gsl_fft_halfcomplex_float.h");
});

// Re-exported from `gsl.zig` so FFT users can toggle the error handler and turn
// codes into messages without importing two modules. GSL's error handler is
// process-global, so flipping it here affects every GSL call.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
pub const strerror = gsl.strerror;

/// Read-only strided view (`ptr`/`stride`/`len` in element units), re-exported
/// from `gsl.zig`. Used for the read-only source of the `unpack` helpers.
pub const Strided = gsl.Strided;
/// Mutable strided view, re-exported from `gsl.zig`. In-place transforms take
/// this; `.fromSlice(buf)` covers the contiguous case.
pub const StridedMut = gsl.StridedMut;

/// `std.math.Complex`, re-exported for convenience. `complex(T).Value` is
/// `Complex(T)`, an `extern struct { re: T, im: T }` matching GSL's packed
/// layout.
pub const Complex = std.math.Complex;

/// Sign of the exponential in the transform, matching `gsl_fft_direction`
/// (GSL/FFTPACK convention: the forward transform uses the *negative*
/// exponent). Only the `transform` entry points take a direction; the
/// `forward`/`backward`/`inverse` shorthands fix it.
pub const Direction = enum(c_int) {
    /// e^{-2 pi i j k / n}; the plain (unscaled) analysis transform.
    forward = -1,
    /// e^{+2 pi i j k / n}; the unscaled synthesis transform. Equivalent to the
    /// inverse without the `1/n` normalization.
    backward = 1,
};

/// Errors returned by the FFT routines.
pub const Error = error{
    /// The transform length is zero (`GSL_EDOM`).
    ZeroLength,
    /// A radix-2 routine was given a length that is not a power of two.
    NotPowerOfTwo,
    /// A `Plan` was applied to data whose length differs from the length the
    /// plan was built for.
    LengthMismatch,
    /// A workspace passed to `Plan.initWithWorkspace` was built for a different
    /// length than the plan.
    WorkspaceLengthMismatch,
    /// A wavetable/workspace allocation failed.
    OutOfMemory,
    /// GSL returned an unexpected nonzero status code.
    TransformFailed,
};

fn requirePow2(n: usize) Error!void {
    if (n == 0) return error.ZeroLength;
    if (!std.math.isPowerOfTwo(n)) return error.NotPowerOfTwo;
}

fn requireLen(actual: usize, expected: usize) Error!void {
    if (actual == 0) return error.ZeroLength;
    if (actual != expected) return error.LengthMismatch;
}

fn check(code: c_int) Error!void {
    if (code != 0) return error.TransformFailed;
}

fn requirePrecision(comptime T: type) void {
    if (T != f32 and T != f64)
        @compileError("gsl.fft: element type must be f32 or f64, got '" ++ @typeName(T) ++ "' (GSL has no long-double FFT module)");
}

/// A GSL FFT scratch workspace of a fixed length, wrapping one of the
/// `gsl_fft_*_workspace` types. `WorkspaceC` is the C struct (which exposes a
/// `size_t n` used to validate length matches), and `alloc_name`/`free_name`
/// are the corresponding GSL symbols.
fn WorkspaceHandle(comptime WorkspaceC: type, comptime alloc_name: [:0]const u8, comptime free_name: [:0]const u8) type {
    return struct {
        ptr: *WorkspaceC,
        const Self = @This();

        /// Allocate scratch space for a length-`n` transform (`n >= 1`).
        pub fn init(n: usize) error{ OutOfMemory, ZeroLength }!Self {
            if (n == 0) return error.ZeroLength;
            const p = @field(c, alloc_name)(n) orelse return error.OutOfMemory;
            return .{ .ptr = p };
        }
        /// Free the scratch space.
        pub fn deinit(self: Self) void {
            @field(c, free_name)(self.ptr);
        }
        /// The length this workspace was built for.
        pub fn len(self: Self) usize {
            return self.ptr.n;
        }
    };
}

/// Scratch workspace for mixed-radix *complex* transforms of a fixed length
/// (`complex(T).Workspace`).
fn ComplexWorkspace(comptime T: type) type {
    requirePrecision(T);
    const is_f64 = T == f64;
    return WorkspaceHandle(
        if (is_f64) c.gsl_fft_complex_workspace else c.gsl_fft_complex_workspace_float,
        if (is_f64) "gsl_fft_complex_workspace_alloc" else "gsl_fft_complex_workspace_float_alloc",
        if (is_f64) "gsl_fft_complex_workspace_free" else "gsl_fft_complex_workspace_float_free",
    );
}

/// Scratch workspace for mixed-radix real *and* half-complex transforms of a
/// fixed length. GSL's half-complex inverse reuses the real workspace type, so
/// a single instance drives both directions: `real(T).Workspace` and
/// `halfcomplex(T).Workspace` are the very same type (allocate once with
/// `init(n)`, then share it via `Plan.initWithWorkspace` across a `real(T).Plan`
/// and a `halfcomplex(T).Plan` of the same length).
fn RealWorkspace(comptime T: type) type {
    requirePrecision(T);
    const is_f64 = T == f64;
    return WorkspaceHandle(
        if (is_f64) c.gsl_fft_real_workspace else c.gsl_fft_real_workspace_float,
        if (is_f64) "gsl_fft_real_workspace_alloc" else "gsl_fft_real_workspace_float_alloc",
        if (is_f64) "gsl_fft_real_workspace_free" else "gsl_fft_real_workspace_float_free",
    );
}

/// # Complex FFTs (`gsl_fft_complex[_float]`)
///
/// In-place complex→complex transforms over packed complex data
/// (`Value == Complex(T)`). `complex(f64)` selects the `double` module,
/// `complex(f32)` the `float` module.
///
/// Radix-2 helpers (`radix2Forward`/`radix2Backward`/`radix2Inverse`/
/// `radix2Transform`) run in place with no scratch space but require a
/// power-of-two length. For arbitrary lengths, build a `Plan` (mixed radix).
///
/// A forward transform followed by an inverse reproduces the input (the inverse
/// applies the `1/n` normalization); `backward` is the unscaled inverse.
pub fn complex(comptime T: type) type {
    requirePrecision(T);
    return struct {
        /// The packed complex element type handed to GSL.
        pub const Value = Complex(T);
        /// The in-place strided view every transform accepts.
        pub const View = StridedMut(Value);
        /// Scratch space for the mixed-radix `Plan`.
        pub const Workspace = ComplexWorkspace(T);

        const is_f64 = T == f64;
        const inf = if (is_f64) "" else "_float";
        const fp = "gsl_fft_complex" ++ inf ++ "_";
        const Wavetable = if (is_f64) c.gsl_fft_complex_wavetable else c.gsl_fft_complex_wavetable_float;
        const wt_alloc = "gsl_fft_complex_wavetable" ++ inf ++ "_alloc";
        const wt_free = "gsl_fft_complex_wavetable" ++ inf ++ "_free";

        // --- Radix-2 (power-of-two length, in place, no workspace) -------------

        /// Forward radix-2 FFT in place. `data.len` must be a power of two.
        pub fn radix2Forward(data: View) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_forward")(@ptrCast(data.ptr), data.stride, data.len));
        }
        /// Backward (unscaled inverse) radix-2 FFT in place.
        pub fn radix2Backward(data: View) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_backward")(@ptrCast(data.ptr), data.stride, data.len));
        }
        /// Inverse radix-2 FFT in place (normalized by `1/n`).
        pub fn radix2Inverse(data: View) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_inverse")(@ptrCast(data.ptr), data.stride, data.len));
        }
        /// Radix-2 FFT in place with an explicit `direction` (unscaled).
        pub fn radix2Transform(data: View, direction: Direction) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_transform")(@ptrCast(data.ptr), data.stride, data.len, @intFromEnum(direction)));
        }

        // --- Mixed radix (any length; owns a wavetable, owns/borrows a workspace)

        /// A reusable mixed-radix plan for complex transforms of a fixed length.
        /// Owns GSL's trigonometric wavetable, and by default (`init`) its own
        /// scratch workspace too; build once and reuse across many transforms of
        /// length `n`. Use `initWithWorkspace` to borrow a shared `Workspace`.
        pub const Plan = struct {
            n: usize,
            wavetable: *Wavetable,
            workspace: Workspace,
            owns_workspace: bool,

            fn create(n: usize, workspace: Workspace, owns: bool) error{OutOfMemory}!Plan {
                const wt = @field(c, wt_alloc)(n) orelse return error.OutOfMemory;
                return .{ .n = n, .wavetable = wt, .workspace = workspace, .owns_workspace = owns };
            }
            /// Allocate the wavetable and a private workspace for length `n`
            /// (`n >= 1`). The plan owns both; `deinit` frees both.
            pub fn init(n: usize) error{ OutOfMemory, ZeroLength }!Plan {
                if (n == 0) return error.ZeroLength;
                const ws = try Workspace.init(n);
                errdefer ws.deinit();
                return create(n, ws, true);
            }
            /// Allocate the wavetable but borrow `workspace` (which must have
            /// been built for the same length). The plan does *not* free the
            /// workspace on `deinit`, so one workspace can be shared across
            /// several same-length plans.
            pub fn initWithWorkspace(n: usize, workspace: Workspace) error{ OutOfMemory, ZeroLength, WorkspaceLengthMismatch }!Plan {
                if (n == 0) return error.ZeroLength;
                if (workspace.len() != n) return error.WorkspaceLengthMismatch;
                return create(n, workspace, false);
            }
            /// Free the wavetable, and the workspace too if this plan owns it
            /// (i.e. it was built with `init`, not `initWithWorkspace`).
            pub fn deinit(self: Plan) void {
                if (self.owns_workspace) self.workspace.deinit();
                @field(c, wt_free)(self.wavetable);
            }

            /// Forward FFT in place. `data.len` must equal the plan's length.
            pub fn forward(self: Plan, data: View) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "forward")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr));
            }
            /// Backward (unscaled inverse) FFT in place.
            pub fn backward(self: Plan, data: View) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "backward")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr));
            }
            /// Inverse FFT in place (normalized by `1/n`).
            pub fn inverse(self: Plan, data: View) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "inverse")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr));
            }
            /// FFT in place with an explicit `direction` (unscaled).
            pub fn transform(self: Plan, data: View, direction: Direction) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "transform")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr, @intFromEnum(direction)));
            }

            /// GSL's chosen factorization of `n` (only sub-transforms of size
            /// 2, 3, 4, 5, 6, 7 are optimized; other factors use a slow
            /// general module). Handy for estimating run-time.
            pub fn factors(self: Plan) []const usize {
                return self.wavetable.factor[0..self.wavetable.nf];
            }
        };
    };
}

/// # Real-data forward FFTs (`gsl_fft_real[_float]`)
///
/// The forward transform of a real sequence (real→half-complex). Because the
/// spectrum of a real signal is conjugate-symmetric, GSL stores it as a compact
/// *half-complex* sequence of `n` reals rather than `n` complex numbers (invert
/// it with `halfcomplex(T)`, or expand it with `halfcomplex(T).unpack`).
///
/// `Value == T` (real samples). `radix2Forward` needs a power-of-two length; the
/// mixed-radix `Plan` works for any length and transforms against a separately
/// owned `Workspace` (see the module docs for why the workspace is external).
pub fn real(comptime T: type) type {
    requirePrecision(T);
    return struct {
        /// Real sample type.
        pub const Value = T;
        /// Packed complex type produced by `unpack`.
        pub const ComplexValue = Complex(T);
        /// In-place strided view over real samples.
        pub const View = StridedMut(T);
        /// Scratch space for the mixed-radix `Plan`. Identical to
        /// `halfcomplex(T).Workspace`, so one instance can be shared across a
        /// forward+inverse round trip.
        pub const Workspace = RealWorkspace(T);

        const is_f64 = T == f64;
        const inf = if (is_f64) "" else "_float";
        const fp = "gsl_fft_real" ++ inf ++ "_";
        const Wavetable = if (is_f64) c.gsl_fft_real_wavetable else c.gsl_fft_real_wavetable_float;
        const wt_alloc = "gsl_fft_real_wavetable" ++ inf ++ "_alloc";
        const wt_free = "gsl_fft_real_wavetable" ++ inf ++ "_free";

        /// Forward radix-2 real FFT in place. `data.len` must be a power of two.
        /// The output is a half-complex sequence in the radix-2 storage scheme
        /// (expand it with `halfcomplex(T).radix2Unpack`, invert it with
        /// `halfcomplex(T).radix2Inverse`).
        pub fn radix2Forward(data: View) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_transform")(@ptrCast(data.ptr), data.stride, data.len));
        }

        /// Expand a real array into a packed complex array (imaginary parts set
        /// to zero), ready for the `complex(T)` routines. `src` and `dst` must
        /// share the same length and stride (GSL uses a single stride for both).
        pub fn unpack(src: Strided(T), dst: StridedMut(ComplexValue)) Error!void {
            std.debug.assert(src.len == dst.len and src.stride == dst.stride);
            try check(@field(c, fp ++ "unpack")(@ptrCast(src.ptr), @ptrCast(dst.ptr), src.stride, src.len));
        }

        /// A reusable mixed-radix plan for the *forward* real transform of a
        /// fixed length. Owns the real wavetable, and by default (`init`) its own
        /// scratch workspace; a forward-only user allocates just this one
        /// wavetable and one workspace. Use `initWithWorkspace` to share a
        /// workspace with a `halfcomplex(T).Plan` across a round trip. See
        /// `complex(T).Plan` for the ownership model.
        pub const Plan = struct {
            n: usize,
            wavetable: *Wavetable,
            workspace: Workspace,
            owns_workspace: bool,

            fn create(n: usize, workspace: Workspace, owns: bool) error{OutOfMemory}!Plan {
                const wt = @field(c, wt_alloc)(n) orelse return error.OutOfMemory;
                return .{ .n = n, .wavetable = wt, .workspace = workspace, .owns_workspace = owns };
            }
            /// Allocate the wavetable and a private workspace for length `n`
            /// (`n >= 1`). The plan owns both; `deinit` frees both.
            pub fn init(n: usize) error{ OutOfMemory, ZeroLength }!Plan {
                if (n == 0) return error.ZeroLength;
                const ws = try Workspace.init(n);
                errdefer ws.deinit();
                return create(n, ws, true);
            }
            /// Allocate the wavetable but borrow `workspace` (built for the same
            /// length). The plan does *not* free the workspace on `deinit`; e.g.
            /// share one `real(T).Workspace` with a `halfcomplex(T).Plan`.
            pub fn initWithWorkspace(n: usize, workspace: Workspace) error{ OutOfMemory, ZeroLength, WorkspaceLengthMismatch }!Plan {
                if (n == 0) return error.ZeroLength;
                if (workspace.len() != n) return error.WorkspaceLengthMismatch;
                return create(n, workspace, false);
            }
            /// Free the wavetable, and the workspace too if this plan owns it.
            pub fn deinit(self: Plan) void {
                if (self.owns_workspace) self.workspace.deinit();
                @field(c, wt_free)(self.wavetable);
            }

            /// Forward real FFT in place (real→half-complex, FFTPACK storage).
            /// `data.len` must equal the plan's length. Invert with a
            /// `halfcomplex(T).Plan`, or expand with `halfcomplex(T).unpack`.
            pub fn forward(self: Plan, data: View) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "transform")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr));
            }

            /// GSL's chosen factorization of `n`.
            pub fn factors(self: Plan) []const usize {
                return self.wavetable.factor[0..self.wavetable.nf];
            }
        };
    };
}

/// # Half-complex inverse FFTs (`gsl_fft_halfcomplex[_float]`)
///
/// The inverse/backward transform of a half-complex sequence produced by
/// `real(T)`, reconstructing the original real signal. `Value == T` (the data
/// is stored as `n` reals in FFTPACK half-complex order for the mixed-radix
/// `Plan`, or the radix-2 order for the `radix2*` routines). `unpack` /
/// `radix2Unpack` expand a half-complex sequence into an ordinary packed complex
/// array.
///
/// The mixed-radix `Plan` owns only the half-complex wavetable and takes a
/// `Workspace` per call; that `Workspace` is the same type as `real(T)`'s, so a
/// forward plan and an inverse plan of the same length share one workspace.
pub fn halfcomplex(comptime T: type) type {
    requirePrecision(T);
    return struct {
        /// Real sample type (half-complex data is stored as reals).
        pub const Value = T;
        /// Packed complex type produced by `unpack`/`radix2Unpack`.
        pub const ComplexValue = Complex(T);
        /// In-place strided view over the half-complex reals.
        pub const View = StridedMut(T);
        /// Scratch space for the mixed-radix `Plan`. Identical to
        /// `real(T).Workspace`.
        pub const Workspace = RealWorkspace(T);

        const is_f64 = T == f64;
        const inf = if (is_f64) "" else "_float";
        const fp = "gsl_fft_halfcomplex" ++ inf ++ "_";
        const Wavetable = if (is_f64) c.gsl_fft_halfcomplex_wavetable else c.gsl_fft_halfcomplex_wavetable_float;
        const wt_alloc = "gsl_fft_halfcomplex_wavetable" ++ inf ++ "_alloc";
        const wt_free = "gsl_fft_halfcomplex_wavetable" ++ inf ++ "_free";

        // --- Radix-2 (power-of-two length, in place, no workspace) -------------

        /// Inverse radix-2 half-complex→real FFT in place (normalized by `1/n`).
        /// Expects data in the radix-2 half-complex storage scheme (the output
        /// of `real(T).radix2Forward`).
        pub fn radix2Inverse(data: View) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_inverse")(@ptrCast(data.ptr), data.stride, data.len));
        }
        /// Backward (unscaled inverse) radix-2 half-complex→real FFT in place.
        pub fn radix2Backward(data: View) Error!void {
            try requirePow2(data.len);
            try check(@field(c, fp ++ "radix2_backward")(@ptrCast(data.ptr), data.stride, data.len));
        }

        /// Expand a radix-2 half-complex array into a full packed complex array
        /// using conjugate symmetry. `src` and `dst` must share length/stride.
        pub fn radix2Unpack(src: Strided(T), dst: StridedMut(ComplexValue)) Error!void {
            std.debug.assert(src.len == dst.len and src.stride == dst.stride);
            try check(@field(c, fp ++ "radix2_unpack")(@ptrCast(src.ptr), @ptrCast(dst.ptr), src.stride, src.len));
        }

        /// Expand a mixed-radix (FFTPACK) half-complex array into a full packed
        /// complex array using conjugate symmetry. `src` and `dst` must share
        /// length/stride.
        pub fn unpack(src: Strided(T), dst: StridedMut(ComplexValue)) Error!void {
            std.debug.assert(src.len == dst.len and src.stride == dst.stride);
            try check(@field(c, fp ++ "unpack")(@ptrCast(src.ptr), @ptrCast(dst.ptr), src.stride, src.len));
        }

        // --- Mixed radix (any length; owns a wavetable, borrows a workspace) ---

        /// A reusable mixed-radix plan for the *inverse* half-complex→real
        /// transform of a fixed length. Owns the half-complex wavetable, and by
        /// default (`init`) its own scratch workspace. Use `initWithWorkspace` to
        /// borrow a workspace shared with a `real(T).Plan` (its `Workspace` is
        /// the same type). See `complex(T).Plan` for the ownership model.
        pub const Plan = struct {
            n: usize,
            wavetable: *Wavetable,
            workspace: Workspace,
            owns_workspace: bool,

            fn create(n: usize, workspace: Workspace, owns: bool) error{OutOfMemory}!Plan {
                const wt = @field(c, wt_alloc)(n) orelse return error.OutOfMemory;
                return .{ .n = n, .wavetable = wt, .workspace = workspace, .owns_workspace = owns };
            }
            /// Allocate the wavetable and a private workspace for length `n`
            /// (`n >= 1`). The plan owns both; `deinit` frees both.
            pub fn init(n: usize) error{ OutOfMemory, ZeroLength }!Plan {
                if (n == 0) return error.ZeroLength;
                const ws = try Workspace.init(n);
                errdefer ws.deinit();
                return create(n, ws, true);
            }
            /// Allocate the wavetable but borrow `workspace` (built for the same
            /// length). The plan does *not* free the workspace on `deinit`; e.g.
            /// share one `real(T).Workspace` with a `real(T).Plan`.
            pub fn initWithWorkspace(n: usize, workspace: Workspace) error{ OutOfMemory, ZeroLength, WorkspaceLengthMismatch }!Plan {
                if (n == 0) return error.ZeroLength;
                if (workspace.len() != n) return error.WorkspaceLengthMismatch;
                return create(n, workspace, false);
            }
            /// Free the wavetable, and the workspace too if this plan owns it.
            pub fn deinit(self: Plan) void {
                if (self.owns_workspace) self.workspace.deinit();
                @field(c, wt_free)(self.wavetable);
            }

            /// Inverse half-complex→real FFT in place (normalized by `1/n`).
            /// Expects FFTPACK half-complex storage (the output of
            /// `real(T).Plan.forward`). `data.len` must equal the plan's length.
            pub fn inverse(self: Plan, data: View) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "inverse")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr));
            }
            /// Backward (unscaled inverse) half-complex→real FFT in place.
            pub fn backward(self: Plan, data: View) Error!void {
                try requireLen(data.len, self.n);
                try check(@field(c, fp ++ "backward")(@ptrCast(data.ptr), data.stride, data.len, self.wavetable, self.workspace.ptr));
            }

            /// GSL's chosen factorization of `n`.
            pub fn factors(self: Plan) []const usize {
                return self.wavetable.factor[0..self.wavetable.nf];
            }
        };
    };
}

// ===== Tests =================================================================

const cf64 = Complex(f64);
const cf32 = Complex(f32);

fn expectComplexClose(expected: cf64, actual: cf64, tol: f64) !void {
    try testing.expectApproxEqAbs(expected.re, actual.re, tol);
    try testing.expectApproxEqAbs(expected.im, actual.im, tol);
}

test "complex: radix-2 forward of an impulse is a flat spectrum" {
    // x[0] = 1, rest 0  ->  X[k] = 1 for all k.
    var data = [_]cf64{ cf64.init(1, 0), cf64.init(0, 0), cf64.init(0, 0), cf64.init(0, 0) };
    try complex(f64).radix2Forward(.fromSlice(&data));
    for (data) |x| try expectComplexClose(cf64.init(1, 0), x, 1e-12);
}

test "complex: radix-2 forward of a constant is an impulse at DC" {
    // x[k] = 1  ->  X[0] = n, rest 0.
    var data = [_]cf64{ cf64.init(1, 0), cf64.init(1, 0), cf64.init(1, 0), cf64.init(1, 0) };
    try complex(f64).radix2Forward(.fromSlice(&data));
    try expectComplexClose(cf64.init(4, 0), data[0], 1e-12);
    try expectComplexClose(cf64.init(0, 0), data[1], 1e-12);
    try expectComplexClose(cf64.init(0, 0), data[2], 1e-12);
    try expectComplexClose(cf64.init(0, 0), data[3], 1e-12);
}

test "complex: radix-2 forward then inverse round-trips" {
    const orig = [_]cf64{
        cf64.init(1, -1), cf64.init(2, 0.5), cf64.init(-3, 2), cf64.init(0.25, -0.75),
        cf64.init(4, 4),  cf64.init(-1, 0),  cf64.init(0, 3),  cf64.init(2.5, -2.5),
    };
    var data = orig;
    try complex(f64).radix2Forward(.fromSlice(&data));
    try complex(f64).radix2Inverse(.fromSlice(&data));
    for (orig, data) |e, a| try expectComplexClose(e, a, 1e-12);
}

test "complex: backward is the unscaled inverse (scales by n)" {
    const orig = [_]cf64{ cf64.init(1, 0), cf64.init(2, -1), cf64.init(-1, 3), cf64.init(0.5, 0.5) };
    var data = orig;
    try complex(f64).radix2Forward(.fromSlice(&data));
    try complex(f64).radix2Backward(.fromSlice(&data));
    for (orig, data) |e, a| try expectComplexClose(cf64.init(e.re * 4, e.im * 4), a, 1e-12);
}

test "complex: radix-2 transform with explicit direction matches forward" {
    const orig = [_]cf64{ cf64.init(1, 0), cf64.init(0, 2), cf64.init(3, -1), cf64.init(-2, 0.5) };
    var a = orig;
    var b = orig;
    try complex(f64).radix2Forward(.fromSlice(&a));
    try complex(f64).radix2Transform(.fromSlice(&b), .forward);
    for (a, b) |x, y| try expectComplexClose(x, y, 1e-12);
}

test "complex: radix-2 rejects non-power-of-two length before calling GSL" {
    var data = [_]cf64{ cf64.init(1, 0), cf64.init(2, 0), cf64.init(3, 0) };
    try testing.expectError(error.NotPowerOfTwo, complex(f64).radix2Forward(.fromSlice(&data)));
    var empty = [_]cf64{};
    try testing.expectError(error.ZeroLength, complex(f64).radix2Forward(.fromSlice(&empty)));
}

test "complex: mixed-radix plan round-trips a non-power-of-two length" {
    const n = 6;
    const orig = [_]cf64{
        cf64.init(1, 0),   cf64.init(2, -1), cf64.init(-3, 2),
        cf64.init(0.5, 1), cf64.init(4, 0),  cf64.init(-1, -2),
    };
    var data = orig;

    var plan = try complex(f64).Plan.init(n);
    defer plan.deinit();

    // 6 = 3 * 2 (GSL orders factors starting from the smallest optimized ones).
    var product: usize = 1;
    for (plan.factors()) |f| product *= f;
    try testing.expectEqual(@as(usize, n), product);

    try plan.forward(.fromSlice(&data));
    try plan.inverse(.fromSlice(&data));
    for (orig, data) |e, a| try expectComplexClose(e, a, 1e-12);
}

test "complex: mixed-radix plan round-trips a large prime length (slow general module)" {
    // A prime length has no small factors, so GSL falls back to its O(n^2)
    // general module. `factors()` should report the single prime factor.
    const n = 233;
    var orig: [n]cf64 = undefined;
    var prng = std.Random.DefaultPrng.init(0xFEEDBEEF);
    const rand = prng.random();
    for (&orig) |*x| x.* = cf64.init(rand.floatNorm(f64), rand.floatNorm(f64));
    var data = orig;

    var plan = try complex(f64).Plan.init(n);
    defer plan.deinit();

    try testing.expectEqualSlices(usize, &.{n}, plan.factors());

    try plan.forward(.fromSlice(&data));
    try plan.inverse(.fromSlice(&data));
    for (orig, data) |e, a| try expectComplexClose(e, a, 1e-10);
}

test "complex: plan rejects a length mismatch" {
    var plan = try complex(f64).Plan.init(4);
    defer plan.deinit();
    var data = [_]cf64{ cf64.init(1, 0), cf64.init(2, 0), cf64.init(3, 0) };
    try testing.expectError(error.LengthMismatch, plan.forward(.fromSlice(&data)));
}

test "complex: f32 precision round-trips" {
    const orig = [_]cf32{ cf32.init(1, -1), cf32.init(2, 0.5), cf32.init(-3, 2), cf32.init(0.25, -0.75) };
    var data = orig;
    try complex(f32).radix2Forward(.fromSlice(&data));
    try complex(f32).radix2Inverse(.fromSlice(&data));
    for (orig, data) |e, a| {
        try testing.expectApproxEqAbs(e.re, a.re, 1e-5);
        try testing.expectApproxEqAbs(e.im, a.im, 1e-5);
    }
}

test "complex: strided view transforms every other element in place" {
    // Interleave the length-4 signal with sentinel values and transform the
    // even-indexed sub-array via stride 2, leaving the odd slots untouched.
    var buf = [_]cf64{
        cf64.init(1, 0), cf64.init(99, 99), cf64.init(1, 0), cf64.init(99, 99),
        cf64.init(1, 0), cf64.init(99, 99), cf64.init(1, 0), cf64.init(99, 99),
    };
    const view = StridedMut(cf64).init(&buf, 2, 4);
    try complex(f64).radix2Forward(view);
    // Constant signal -> impulse at DC on the strided elements.
    try expectComplexClose(cf64.init(4, 0), buf[0], 1e-12);
    try expectComplexClose(cf64.init(0, 0), buf[2], 1e-12);
    // Untouched sentinels.
    try expectComplexClose(cf64.init(99, 99), buf[1], 0);
    try expectComplexClose(cf64.init(99, 99), buf[3], 0);
}

test "real: radix-2 forward then half-complex inverse round-trips" {
    const orig = [_]f64{ 1, 2, 3, 4, 3, 2, 1, 0 };
    var data = orig;
    try real(f64).radix2Forward(.fromSlice(&data));
    try halfcomplex(f64).radix2Inverse(.fromSlice(&data));
    for (orig, data) |e, a| try testing.expectApproxEqAbs(e, a, 1e-12);
}

test "real: mixed-radix forward then inverse round-trips with a shared workspace" {
    const n = 9;
    const orig = [_]f64{ 1, 2, 3, 4, 5, 4, 3, 2, 1 };
    var data = orig;

    // One workspace, shared by the forward and inverse plans via
    // initWithWorkspace; neither plan allocates a second workspace.
    var ws = try real(f64).Workspace.init(n);
    defer ws.deinit();
    var fwd = try real(f64).Plan.initWithWorkspace(n, ws);
    defer fwd.deinit();
    var inv = try halfcomplex(f64).Plan.initWithWorkspace(n, ws);
    defer inv.deinit();

    try fwd.forward(.fromSlice(&data));
    try inv.inverse(.fromSlice(&data));
    for (orig, data) |e, a| try testing.expectApproxEqAbs(e, a, 1e-12);
}

test "plan: initWithWorkspace rejects a workspace built for a different length" {
    var ws = try complex(f64).Workspace.init(8);
    defer ws.deinit();
    try testing.expectError(error.WorkspaceLengthMismatch, complex(f64).Plan.initWithWorkspace(4, ws));
}

test "half-complex unpack matches a full complex FFT of the same signal" {
    const n = 6;
    const signal = [_]f64{ 1, 3, -2, 5, 0, 4 };

    // Path A: real forward transform (bundled workspace), then unpack.
    var hc = signal;
    var fwd = try real(f64).Plan.init(n);
    defer fwd.deinit();
    try fwd.forward(.fromSlice(&hc));

    var unpacked: [n]cf64 = undefined;
    try halfcomplex(f64).unpack(.fromSlice(&hc), .fromSlice(&unpacked));

    // Path B: embed the real signal as complex and run a full complex FFT.
    var cdata: [n]cf64 = undefined;
    for (signal, 0..) |s, i| cdata[i] = cf64.init(s, 0);
    var cplan = try complex(f64).Plan.init(n);
    defer cplan.deinit();
    try cplan.forward(.fromSlice(&cdata));

    for (unpacked, cdata) |u, e| try expectComplexClose(e, u, 1e-11);
}

test "real: unpack embeds a real array as complex with zero imaginary parts" {
    const src = [_]f64{ 1, -2, 3, -4 };
    var dst: [4]cf64 = undefined;
    try real(f64).unpack(.fromSlice(&src), .fromSlice(&dst));
    for (src, dst) |s, d| {
        try testing.expectEqual(s, d.re);
        try testing.expectEqual(@as(f64, 0), d.im);
    }
}

test "fft: real and half-complex share a single workspace type" {
    // The same workspace instance can drive both directions of a round trip.
    try testing.expect(real(f64).Workspace == halfcomplex(f64).Workspace);
    try testing.expect(real(f32).Workspace == halfcomplex(f32).Workspace);
}

test "fft: precision guard rejects unsupported element types at comptime" {
    // These would each be a compile error; kept as documentation of the guard:
    //   _ = complex(f16);
    //   _ = real(i32);
    try testing.expect(@hasDecl(complex(f64), "Plan"));
    try testing.expect(@hasDecl(real(f32), "Plan"));
    try testing.expect(@hasDecl(halfcomplex(f64), "Plan"));
}
