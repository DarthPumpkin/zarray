//! Idiomatic Zig bindings for the GNU Scientific Library.
//!
//! This file is the **hub**: it re-exports one namespace per GSL chapter (each
//! implemented in its own `gsl_*.zig` file) and holds the small pieces of
//! infrastructure they all share. Reach any chapter as `gsl.<name>`.
//!
//! ## Chapters
//!
//!   - `rand`     — `gsl_rng`/`gsl_randist`/`gsl_cdf`: generator algorithms and
//!                  the stateful `Rng` handle, random distributions (with the
//!                  generic shuffle/choose/sample helpers on `Rng`), and
//!                  scipy-style `cdf`/`sf`/`ppf`/`isf` methods per distribution.
//!   - `qrng`     — `gsl_qrng`: quasi-random (low-discrepancy) sequences.
//!   - `stats`    — `gsl_statistics`: descriptive, robust, and weighted
//!                  statistics over strided views. `stats(T)` selects the module
//!                  for element type `T` (e.g. `stats(f64)`, `stats(i32)`).
//!   - `rstat`    — `gsl_rstat`: running/streaming statistics (O(1)-memory
//!                  `Accumulator` + a P²-algorithm single-`Quantile`).
//!   - `movstat`  — `gsl_movstat`: moving-window statistics.
//!   - `filter`   — `gsl_filter`: digital filters (Gaussian, median, recursive
//!                  median, impulse).
//!   - `interp`   — `gsl_interp`/`gsl_spline`: 1-D interpolation and splines.
//!   - `histogram`— `gsl_histogram`/`gsl_histogram2d`: 1-D/2-D histograms + PDFs.
//!   - `poly`     — `gsl_poly`: polynomial evaluation and root finding.
//!   - `deriv`    — `gsl_deriv`: numerical differentiation (finite differences).
//!   - `integration` — `gsl_integration`: adaptive/non-adaptive Gauss-Kronrod
//!                  quadrature over finite and infinite intervals.
//!   - `roots`    — `gsl_roots`: one-dimensional root finding (bracketing and
//!                  derivative-based solvers).
//!   - `min`      — `gsl_min`: one-dimensional function minimization.
//!   - `cheb`     — `gsl_chebyshev`: Chebyshev-series function approximation.
//!   - `sort`     — `gsl_sort`: typed sorting / k-smallest / index sorts.
//!   - `permutation` — `gsl_permutation` (+ `gsl_permute`): permutations.
//!   - `combination` — `gsl_combination`: combinations.
//!   - `multiset` — `gsl_multiset`: multisets.
//!   - `sf`       — `gsl_sf_*`: special functions.
//!   - `fft`      — `gsl_fft_*`: fast Fourier transforms.
//!
//! ## Shared infrastructure (defined here, used by the chapters)
//!
//!   - `Error`/`check` — the common Zig error set and status-code translator.
//!   - `ensureHandler`/`disableDefaultErrorHandler`/`strerror` — GSL's
//!     process-global (non-aborting) error handler and message lookup.
//!   - `Strided`/`StridedMut` — borrowed strided views over caller memory.
//!   - `constVectorViewOf`/`mutVectorViewOf` — generic borrowed-`gsl_vector`
//!     view builders (each chapter's `@cImport` yields a distinct `gsl_vector`
//!     type, so these are generic over it).
//!   - `numericModuleStem`/`numericModuleInfix` — map an element type to GSL's
//!     per-type symbol suffix (shared by `stats` and `sort`).
//!   - `callback` — the bridge that turns a Zig callable into GSL's
//!     `gsl_function`-style callback structs. Chapters expose a `Callback` value
//!     type built with `.initFn(f)` (a plain function) or `.initCtx(&ctx)` (a
//!     `*struct` with an `eval` method); used by the callback chapters
//!     (currently `deriv`, `integration`, `roots`, `min`, and `cheb`).
//!
//! ## Conventions
//!
//! These bindings are not exhaustive: some GSL symbols are intentionally left
//! unwrapped. See each chapter file's `## Omissions` section for the specifics.
//! Every chapter keeps its own raw C API behind its own `c` (this file's `c`
//! only pulls in `gsl_errno.h` for the shared error codes).
//!
//! Error convention: by default GSL's error handler calls `abort()` on error.
//! Many routines never report errors, but the fallible bindings install a
//! non-aborting handler lazily (see `ensureHandler`) so a GSL error surfaces as
//! a Zig `Error`; call `disableDefaultErrorHandler()` yourself to install it
//! earlier or for the fallible parts of a chapter's raw `c` API.

const std = @import("std");
const testing = std.testing;

pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
});

/// Replace GSL's default (aborting) error handler with the no-op handler, so
/// that fallible GSL routines return error codes instead of terminating the
/// process. Returns nothing; call once during program setup if desired.
///
/// The `movstat` bindings install this automatically on first use (see
/// `ensureHandler`); call it yourself only if you want it installed earlier or
/// for the fallible parts of the raw `c` API.
pub fn disableDefaultErrorHandler() void {
    _ = c.gsl_set_error_handler_off();
    handler_installed.store(true, .monotonic);
}

/// Human-readable message for a GSL error code (the `int` many GSL functions
/// return, where `0` == success).
pub fn strerror(gsl_errno: c_int) [:0]const u8 {
    return std.mem.span(c.gsl_strerror(gsl_errno));
}

// One-time, lazy install of the non-aborting error handler, shared by every
// fallible binding here. Installing the no-op handler is idempotent (it just
// stores a handler pointer), so the guard races benignly: at worst a few
// threads each install it before the flag flips. Mirrors `gsl_sf.zig`.
var handler_installed = std.atomic.Value(bool).init(false);

/// Idempotently install GSL's non-aborting error handler (once, lazily). Shared
/// by the fallible bindings across the GSL sub-modules so a GSL error surfaces
/// as a Zig `Error` instead of aborting the process; safe to call on every
/// fallible entry point (it is a single relaxed atomic load in the common case).
pub inline fn ensureHandler() void {
    if (!handler_installed.load(.monotonic)) {
        _ = c.gsl_set_error_handler_off();
        handler_installed.store(true, .monotonic);
    }
}

/// Zig error set covering the status codes GSL's fallible routines raise. Codes
/// without a dedicated variant map to `Unspecified`; the raw `c_int` is always
/// available from the underlying `c.gsl_*` symbol if you need the exact value.
pub const Error = error{
    /// `GSL_EDOM` — input outside the routine's domain.
    Domain,
    /// `GSL_ERANGE` — output outside the representable range.
    Range,
    /// `GSL_EINVAL` — an invalid argument.
    Invalid,
    /// `GSL_EBADLEN`, or a caller-supplied output view of the wrong length.
    BadLength,
    /// `GSL_ENOMEM` — allocation failed.
    OutOfMemory,
    /// Any other nonzero GSL status code.
    Unspecified,
};

/// Translate a GSL status code into `Error!void` (`GSL_SUCCESS` -> `{}`).
pub fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EDOM => Error.Domain,
        c.GSL_ERANGE => Error.Range,
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADLEN => Error.BadLength,
        c.GSL_ENOMEM => Error.OutOfMemory,
        else => Error.Unspecified,
    };
}

/// # Mathematical constants (`gsl_math.h` `M_*`) — use `std.math` instead
///
/// GSL's `gsl_math.h` defines 17 `M_*` mathematical constants (`M_PI`, `M_E`,
/// `M_EULER`, ...). We deliberately do *not* re-bind them: they are plain
/// `#define`d `double` literals, not linkable symbols, and Zig's standard
/// library already provides them (or a trivial expression for them):
///
///   - Direct `std.math` equivalents: `M_E`→`e`, `M_LOG2E`→`log2e`,
///     `M_LOG10E`→`log10e`, `M_LN2`→`ln2`, `M_LN10`→`ln10`, `M_PI`→`pi`,
///     `M_SQRT2`→`sqrt2`, `M_SQRT1_2`→`sqrt1_2`, `M_2_SQRTPI`→`two_sqrtpi`.
///   - Trivial from `std.math.pi`: `M_PI_2` (`pi / 2.0`), `M_PI_4`
///     (`pi / 4.0`), `M_1_PI` (`1.0 / pi`), `M_2_PI` (`2.0 / pi`).
///   - The few remaining literals (`M_SQRT3`, `M_SQRTPI`, `M_LNPI`, `M_EULER`)
///     are one-line `comptime_float`s if you need them.
///
/// Binding GSL's macros would just duplicate hand-copied literals with no C
/// symbol behind them, so this namespace is intentionally a compile error.
pub const constants = @compileError(
    "GSL's M_* mathematical constants are not bound; use std.math instead " ++
        "(e.g. std.math.pi, std.math.e, std.math.sqrt2). See the `constants` docs.",
);

/// # Fast Fourier Transforms (`gsl_fft_*`)
///
/// Complex, real, and half-complex FFTs (radix-2 and mixed-radix, `f32`/`f64`).
/// Kept in its own file (`gsl_fft.zig`) because it needs the `gsl_fft_*` C
/// headers that this module does not include; see that file's docs for the
/// `complex(T)` / `real(T)` / `halfcomplex(T)` surface.
pub const fft = @import("gsl_fft.zig");

/// # Special functions (`gsl_sf_*`)
///
/// Gamma/beta, Bessel, error/exponential/elliptic integrals, zeta, orthogonal
/// polynomials, and the rest of GSL's special-function chapter. Kept in its own
/// file (`gsl_sf.zig`) because it needs the `gsl_sf_*` C headers that this
/// module does not include; every wrapper returns `Error!f64` and the surface
/// is grouped into per-header namespaces (`sf.gamma`, `sf.bessel`, ...).
pub const sf = @import("gsl_sf.zig");

/// # Interpolation & splines (`gsl_interp`/`gsl_spline`)
///
/// 1-D interpolation (linear, polynomial, cubic/Akima/Steffen splines) over
/// caller-owned `x`/`y` data, evaluating the value, first/second derivative, or
/// definite integral. Kept in its own file (`gsl_interp.zig`); reached as
/// `gsl.interp` (`interp.Spline`, `interp.Accel`, `interp.Type`).
pub const interp = @import("gsl_interp.zig");

/// # Histograms (`gsl_histogram`/`gsl_histogram2d`)
///
/// 1-D and 2-D histograms with uniform or explicit binning, (weighted) fills,
/// summary statistics, whole-histogram arithmetic, and empirical-distribution
/// sampling. Kept in its own file (`gsl_histogram.zig`); reached as
/// `gsl.histogram` (`histogram.Histogram`, `histogram.Pdf`,
/// `histogram.Histogram2d`, `histogram.Pdf2d`).
pub const histogram = @import("gsl_histogram.zig");

/// # Polynomials (`gsl_poly`)
///
/// Horner evaluation (real/complex), successive derivatives, closed-form real
/// and complex roots of quadratics and cubics, and general-degree complex root
/// finding via the companion matrix. Kept in its own file (`gsl_poly.zig`);
/// reached as `gsl.poly` (`poly.eval`, `poly.solveQuadratic`,
/// `poly.ComplexSolver`, ...). Complex values are `std.math.Complex(f64)`.
pub const poly = @import("gsl_poly.zig");

/// # Digital filters (`gsl_filter`)
///
/// Gaussian (smoothing/derivative), median, recursive-median, and
/// impulse-detection filters over `Strided`/`StridedMut` signals — a follow-on
/// to `gsl.movstat` that shares its borrowed-`gsl_vector` view helpers. Kept in
/// its own file (`gsl_filter.zig`); reached as `gsl.filter` (`filter.Gaussian`,
/// `filter.Median`, `filter.RecursiveMedian`, `filter.Impulse`).
pub const filter = @import("gsl_filter.zig");

/// # Sorting (`gsl_sort`)
///
/// Element-type-specialized in-place sorting, indirect sorting (index
/// permutations), and top-k extraction over strided views. Kept in its own file
/// (`gsl_sort.zig`); reached as `gsl.sort.sort(T)`.
pub const sort = @import("gsl_sort.zig");

/// # Permutations (`gsl_permutation` + `gsl_permute`)
///
/// Owning permutation handles (`Permutation`) plus element-type-specialized
/// permutation application to strided arrays (`permute(T)`). Kept in its own
/// file (`gsl_permutation.zig`); reached as `gsl.permutation`.
pub const permutation = @import("gsl_permutation.zig");

/// # Combinations (`gsl_combination`)
///
/// Owning combination handles with lexicographic `next`/`prev` enumeration.
/// Kept in its own file (`gsl_combination.zig`); reached as `gsl.combination`.
pub const combination = @import("gsl_combination.zig");

/// # Multisets (`gsl_multiset`)
///
/// Owning multiset handles (combinations with replacement) with
/// lexicographic `next`/`prev` enumeration. Kept in its own file
/// (`gsl_multiset.zig`); reached as `gsl.multiset`.
pub const multiset = @import("gsl_multiset.zig");

/// # Random numbers (`gsl_rng` + `gsl_randist` + `gsl_cdf`)
///
/// Pseudo-random generators and random distributions. Kept in its own file
/// (`gsl_rand.zig`); reached as `gsl.rand` (`rand.Rng`, `rand.Gaussian`, ...).
pub const rand = @import("gsl_rand.zig");

const stats_ref = @import("gsl_stats.zig");
/// # Statistics (`gsl_statistics`)
///
/// Descriptive, robust, and weighted statistics over strided views. Kept in
/// its own file (`gsl_stats.zig`); reached as `gsl.stats(T)`.
pub const stats = stats_ref.stats;

/// # Running (streaming) statistics (`gsl_rstat`)
///
/// O(1)-memory streaming statistics over `f64` samples. Kept in its own file
/// (`gsl_rstat.zig`); reached as `gsl.rstat`.
pub const rstat = @import("gsl_rstat.zig");

/// # Moving-window statistics (`gsl_movstat`)
///
/// Sliding-window statistics over `Strided(f64)` inputs. Kept in its own file
/// (`gsl_movstat.zig`); reached as `gsl.movstat`.
pub const movstat = @import("gsl_movstat.zig");

/// # Quasi-random sequences (`gsl_qrng`)
///
/// GSL's quasi-random generators (`gsl_qrng`) produce low-discrepancy
/// deterministic sequences for quasi-Monte Carlo integration. Kept in its own
/// file (`gsl_qrng.zig`); reached as `gsl.qrng` (`qrng.Sequence`, `qrng.Type`).
pub const qrng = @import("gsl_qrng.zig");

/// # Callback bridge (`gsl.callback`)
///
/// Shared glue that turns an idiomatic Zig callable (a bare `fn(f64) f64`, a
/// `*const fn(f64) f64`, or a `*context` with an `eval` method) into the C
/// function-pointer structs GSL's callback chapters expect. Used by `deriv`
/// (and, as they land, the other callback chapters). Kept in its own file
/// (`gsl_callback.zig`); reached as `gsl.callback`.
pub const callback = @import("gsl_callback.zig");

/// # Numerical differentiation (`gsl_deriv`)
///
/// Central/forward/backward finite-difference estimates of `f'(x)` (with an
/// error bound) for any callable the `callback` bridge accepts. Kept in its own
/// file (`gsl_deriv.zig`); reached as `gsl.deriv` (`deriv.central`, ...).
pub const deriv = @import("gsl_deriv.zig");

/// # Numerical integration (`gsl_integration`)
///
/// Adaptive and non-adaptive Gauss-Kronrod quadrature (with an error bound) for
/// any integrand the `callback` bridge accepts: `qng` (no workspace), and the
/// adaptive `qag`/`qags`/`qagi`/`qagiu`/`qagil` family over finite and infinite
/// intervals. Kept in its own file (`gsl_integration.zig`); reached as
/// `gsl.integration` (`integration.qng`, `integration.Workspace`, ...).
pub const integration = @import("gsl_integration.zig");

/// # One-dimensional root finding (`gsl_roots`)
///
/// Iterative solvers for `f(x) = 0`: bracketing (`bisection`/`brent`/
/// `falsepos`) via `Solver`, and derivative-based (`newton`/`secant`/
/// `steffenson`) via `PolishSolver`, with `testInterval`/`testDelta`/
/// `testResidual` convergence helpers. Kept in its own file (`gsl_roots.zig`);
/// reached as `gsl.roots` (`roots.Solver`, `roots.PolishSolver`, ...).
pub const roots = @import("gsl_roots.zig");

/// # One-dimensional minimization (`gsl_min`)
///
/// Iterative bracketing minimizers for a scalar function
/// (`goldensection`/`brent`/`quad_golden`) via `Minimizer`, with a
/// `testInterval` convergence helper. A close twin of `roots`. Kept in its own
/// file (`gsl_min.zig`); reached as `gsl.min` (`min.Minimizer`, ...).
pub const min = @import("gsl_min.zig");

/// # Chebyshev approximation (`gsl_chebyshev`)
///
/// Fit a smooth function over `[a, b]` to a truncated Chebyshev series
/// (`Chebyshev.fit`), then evaluate it cheaply (`eval`/`evalN`/`evalErr`) and
/// derive `deriv`/`integ` series. Kept in its own file (`gsl_chebyshev.zig`);
/// reached as `gsl.cheb` (`cheb.Chebyshev`).
pub const cheb = @import("gsl_chebyshev.zig");

test {
    // Pull re-exported sub-module files into test discovery. `zig build test`
    // only collects tests from files that are analyzed, so reference each
    // chapter file explicitly.
    _ = fft;
    _ = sf;
    _ = interp;
    _ = histogram;
    _ = poly;
    _ = filter;
    _ = qrng;
    _ = sort;
    _ = permutation;
    _ = combination;
    _ = multiset;
    _ = rand;
    _ = stats_ref;
    _ = rstat;
    _ = movstat;
    _ = callback;
    _ = deriv;
    _ = integration;
    _ = roots;
    _ = min;
    _ = cheb;
}

/// A strided, read-only view over `T`: `len` elements spaced `stride` apart
/// starting at `ptr`. GSL routines that take strided input (currently the
/// statistics module) operate on exactly this shape, so a column/row/axis of a
/// larger array can be passed without copying. Use `fromSlice` for the common
/// contiguous (stride-1) case.
pub fn Strided(comptime T: type) type {
    return struct {
        ptr: [*]const T,
        stride: usize,
        len: usize,

        const Self = @This();

        pub fn init(ptr: [*]const T, stride: usize, len: usize) Self {
            return .{ .ptr = ptr, .stride = stride, .len = len };
        }
        pub fn fromSlice(s: []const T) Self {
            return .{ .ptr = s.ptr, .stride = 1, .len = s.len };
        }
    };
}

/// Mutable counterpart of `Strided`, required by the few statistics routines
/// that rearrange their input in place (`select`, `median`).
pub fn StridedMut(comptime T: type) type {
    return struct {
        ptr: [*]T,
        stride: usize,
        len: usize,

        const Self = @This();

        pub fn init(ptr: [*]T, stride: usize, len: usize) Self {
            return .{ .ptr = ptr, .stride = stride, .len = len };
        }
        pub fn fromSlice(s: []T) Self {
            return .{ .ptr = s.ptr, .stride = 1, .len = s.len };
        }
        pub fn asConst(self: Self) Strided(T) {
            return .{ .ptr = self.ptr, .stride = self.stride, .len = self.len };
        }
    };
}

/// Maps a Zig numeric element type to the GSL chapter stem used in symbol
/// names (`""` for `f64`, `"float"` for `f32`, integer stems like `"int"`,
/// `"ulong"`, ...).
///
/// The chosen stem is the one whose C element type has identical size and
/// signedness on the target platform, so pointers cast across `@cImport`
/// boundaries are representation-safe.
pub fn numericModuleStem(comptime chapter: []const u8, comptime T: type) []const u8 {
    switch (@typeInfo(T)) {
        .float => switch (T) {
            f32 => return "float",
            f64 => return "",
            else => @compileError(chapter ++ ": unsupported float element type '" ++ @typeName(T) ++ "'; only f32 and f64 are supported (GSL's long double module is intentionally omitted)"),
        },
        .int => |info| {
            // (stem, C element type) candidates. Unsigned char first so `u8`
            // deterministically resolves to `uchar` on unsigned-char targets.
            const candidates = .{
                .{ "uchar", u8 },        .{ "char", c_char },
                .{ "ushort", c_ushort }, .{ "short", c_short },
                .{ "uint", c_uint },     .{ "int", c_int },
                .{ "ulong", c_ulong },   .{ "long", c_long },
            };
            inline for (candidates) |cand| {
                const CT = cand[1];
                if (@sizeOf(T) == @sizeOf(CT) and info.signedness == @typeInfo(CT).int.signedness)
                    return cand[0];
            }
            @compileError(std.fmt.comptimePrint(
                "{s}: no GSL numeric module matches element type '{s}' ({d}-bit {s}); GSL provides only char/short/int/long-sized modules",
                .{ chapter, @typeName(T), @bitSizeOf(T), @tagName(info.signedness) },
            ));
        },
        else => @compileError(chapter ++ ": unsupported element type '" ++ @typeName(T) ++ "'; expected a Zig integer or f32/f64"),
    }
}

/// `numericModuleStem` with GSL's trailing-underscore infix convention used by
/// many chapter symbols (`""` for `f64`, otherwise e.g. `"int_"`).
pub fn numericModuleInfix(comptime chapter: []const u8, comptime T: type) []const u8 {
    const stem = numericModuleStem(chapter, T);
    return if (stem.len == 0) "" else stem ++ "_";
}

// ---------------------------------------------------------------------------
// gsl_vector views (borrowed, zero-copy)
// ---------------------------------------------------------------------------
//
// GSL's moving-window routines take `gsl_vector *`, but a `gsl_vector` is just
// our `Strided(f64)` triple (`data`/`stride`/`size`) plus ownership bookkeeping
// (`block`/`owner`). A borrowed view sets `block = null, owner = 0` and owns
// nothing — exactly what `gsl_vector_view_array_with_stride` produces — so we
// can stack-construct one over caller memory with no allocation and no copy.
//
// Each GSL sub-module has its own `@cImport`, hence its own *distinct*
// `c.gsl_vector` type, so the shared constructors are generic over the target
// vector type `Vec`: the struct literal coerces to whichever `gsl_vector` the
// caller passes. `gsl_filter.zig` reuses these (see D12).

/// Stack-construct a borrowed (non-owning) read-only `gsl_vector` of the
/// caller's type `Vec` over the strided data `s`. Internal shared helper for the
/// `gsl_vector`-based bindings (movstat, filter).
pub fn constVectorViewOf(comptime Vec: type, s: Strided(f64)) Vec {
    return .{
        .size = s.len,
        .stride = s.stride,
        // Safe: GSL only reads through the `const gsl_vector *` input parameters.
        .data = @constCast(s.ptr),
        .block = null,
        .owner = 0,
    };
}

/// Stack-construct a borrowed (non-owning) mutable `gsl_vector` of the caller's
/// type `Vec` over the strided data `s`. Internal shared helper (see
/// `constVectorViewOf`).
pub fn mutVectorViewOf(comptime Vec: type, s: StridedMut(f64)) Vec {
    return .{
        .size = s.len,
        .stride = s.stride,
        .data = s.ptr,
        .block = null,
        .owner = 0,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "gsl: error helpers report success and install the non-aborting handler" {
    // Code 0 is GSL_SUCCESS.
    try testing.expectEqualStrings("success", strerror(0));
    // A nonzero code yields a non-empty diagnostic string.
    try testing.expect(strerror(4).len > 0);
    // Smoke-test the setup call (it only swaps GSL's global error handler).
    disableDefaultErrorHandler();
}
