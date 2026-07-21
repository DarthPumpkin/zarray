//! Idiomatic Zig bindings for the GNU Scientific Library's random number
//! generation (`gsl_rng`), random distributions (`gsl_randist`), cumulative
//! distribution functions (`gsl_cdf`), and statistics (`gsl_statistics`)
//! modules.
//!
//! The public surface is organized into two namespaces, each mapping onto
//! specific GSL modules:
//!   - `rand`  — `gsl_rng` (generator algorithms + the stateful `Rng` handle),
//!               `gsl_randist` (random distributions, plus the generic
//!               shuffling/sampling helpers on `Rng`), and `gsl_cdf`
//!               (cumulative distribution functions, surfaced as scipy-style
//!               `cdf`/`sf`/`ppf`/`isf` methods on the distribution structs).
//!   - `stats` — `gsl_statistics` (descriptive, robust, and weighted statistics
//!               over strided views of any element type GSL supports).
//!               `stats(T)` selects the module for element type `T` (e.g.
//!               `stats(f64)`, `stats(i32)`).
//!
//! These bindings are not exhaustive: some GSL symbols are intentionally left
//! unwrapped. See each namespace's "Omitted from GSL" documentation for the
//! specifics, and reach for the raw C API below when you need something else.
//!
//! The raw C API is available via `c` if you need something not wrapped here.
//!
//! Error convention: by default GSL's error handler calls `abort()` on error.
//! For most RNG/stats routines this never triggers (they don't report errors),
//! but if you call into fallible parts of GSL you can install the non-aborting
//! handler once at startup with `disableDefaultErrorHandler()`.

const std = @import("std");
const testing = std.testing;

pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_rng.h");
    @cInclude("gsl/gsl_qrng.h");
    @cInclude("gsl/gsl_randist.h");
    @cInclude("gsl/gsl_cdf.h");
    // Pulls in every element-type statistics module (double, float, int, uint,
    // long, ulong, short, ushort, char, uchar, and long double).
    @cInclude("gsl/gsl_statistics.h");
});

/// Replace GSL's default (aborting) error handler with the no-op handler, so
/// that fallible GSL routines return error codes instead of terminating the
/// process. Returns nothing; call once during program setup if desired.
pub fn disableDefaultErrorHandler() void {
    _ = c.gsl_set_error_handler_off();
}

/// Human-readable message for a GSL error code (the `int` many GSL functions
/// return, where `0` == success).
pub fn strerror(gsl_errno: c_int) [:0]const u8 {
    return std.mem.span(c.gsl_strerror(gsl_errno));
}

/// # Fast Fourier Transforms (`gsl_fft_*`)
///
/// Complex, real, and half-complex FFTs (radix-2 and mixed-radix, `f32`/`f64`).
/// Kept in its own file (`gsl_fft.zig`) because it needs the `gsl_fft_*` C
/// headers that this module does not include; see that file's docs for the
/// `complex(T)` / `real(T)` / `halfcomplex(T)` surface.
pub const fft = @import("gsl_fft.zig");

/// # Random numbers (`gsl_rng` + `gsl_randist` + `gsl_cdf`)
///
/// A single namespace for everything stochastic: the generator handle and every
/// distribution family.
///
/// ## Generators
///
/// `Generator` selects an algorithm and `Rng` is the stateful handle. `Rng`
/// also carries GSL's generic shuffling and sampling helpers
/// (`shuffle`/`choose`/`sample`), which operate on caller-owned data rather
/// than on any particular distribution.
///
/// ## Distributions
///
/// Each distribution is its own struct that follows a small comptime "static
/// interface" (in the spirit of `std.io` readers/writers), rather than being a
/// variant of one big union. This keeps every family's types honest:
///
///   - `pub const Sample = T` — the type a draw produces
///     (`f64`, `u32`, `bool`, `[]f64`, `[]u32`, `[2]f64`, ...).
///   - Scalar families: `sample(r) Sample` and `pdf(x: Sample) f64`.
///   - Vector families: `sample(r, out: []Elem) void` (fills caller storage,
///     no hidden allocation) and `pdf(x: []const Elem) f64`.
///   - CDF-bearing families additionally expose scipy-style `cdf`/`sf`/`ppf`/
///     `isf` (lower tail / upper tail / inverse-cdf / inverse-sf), and only
///     those families GSL actually supports them for. Generic code can detect
///     any capability at comptime with `@hasDecl`.
///
/// GSL provides no moment functions, so distributions intentionally expose no
/// `mean`/`variance` — those would be hand-derived math, not a binding. If you
/// need to hold heterogeneous distributions at runtime, wrap the ones you care
/// about in your own tagged union grouped by `Sample` type.
///
/// ```
/// var gen = try rand.Rng.init(.mt19937);
/// defer gen.deinit();
/// gen.seed(12345);
///
/// const g = rand.Gaussian{ .sigma = 2.0 };
/// const x: f64 = g.sample(gen);
/// const p = g.pdf(x);
///
/// const d = rand.Dirichlet{ .alpha = &.{ 1.0, 2.0, 3.0 } };
/// var theta: [3]f64 = undefined;
/// d.sample(gen, &theta);   // writes a point on the 2-simplex
/// ```
///
/// ## Omitted from GSL
///
/// These parts of GSL's RNG/distribution API are intentionally *not* wrapped
/// (the raw symbols remain reachable through `c`):
///
///   - Generator zoo: `Generator` surfaces GSL's recommended algorithms; the
///     ~60 total include many legacy/compatibility generators. Reach any of them
///     by name with `rand.Rng.initByName`, or use a raw `c.gsl_rng_*` type
///     pointer with `c.gsl_rng_alloc`.
///   - Quasi-random generators (`gsl_qrng_*`: Sobol, Halton, ...) are a separate
///     GSL module with a different shape; see the reserved `qrng` namespace.
///   - Multivariate families: the multivariate Gaussian
///     (`gsl_ran_multivariate_gaussian*`) and Wishart (`gsl_ran_wishart*`) are
///     deferred until this project grows matrix/vector (linear-algebra)
///     bindings for their arguments.
///   - Redundant sampling algorithms: where GSL offers several samplers for one
///     distribution we bind only the default and skip the alternates — e.g.
///     `gsl_ran_gaussian_ziggurat`/`_ratio_method`,
///     `gsl_ran_ugaussian_ratio_method`, `gsl_ran_gamma_knuth`/`_mt`/`_int`,
///     `gsl_ran_binomial_knuth`/`_tpe`, and `gsl_ran_dir_2d_trig_method`.
///   - Bulk helpers: array-filling variants such as `gsl_ran_poisson_array`.
pub const rand = struct {
    // ===== Generators (gsl_rng) ===============================================

    /// A curated selection of GSL's recommended generator algorithms — the
    /// modern, well-tested families the GSL manual endorses. GSL registers ~60
    /// generators in total, but most of the rest exist only to reproduce
    /// bit-exact output from legacy software (old Unix `random()`, historical
    /// Fortran RNGs, deliberately-flawed generators like `randu`). Reach any of
    /// those by name with `Rng.initByName`, or use a raw `c.gsl_rng_*` type
    /// pointer with `c.gsl_rng_alloc`.
    pub const Generator = enum {
        /// Mersenne Twister. GSL's default; a good general-purpose choice.
        mt19937,
        /// Tausworthe generator (maximally equidistributed, very fast).
        taus2,
        /// Improved Tausworthe (L'Ecuyer 1999), period ~2^113.
        taus113,
        /// Combined multiple recursive generator (long period).
        cmrg,
        /// Multiple recursive generator (L'Ecuyer et al.), period ~10^46.
        mrg,
        /// Lagged-Fibonacci, four-tap; very fast.
        gfsr4,
        /// Original RANLUX (single precision, 24-bit), default luxury level.
        ranlux,
        /// Original RANLUX at the highest luxury level (best decorrelation,
        /// slowest).
        ranlux389,
        /// RANLUX, second-generation single-precision, luxury level 0.
        ranlxs0,
        /// RANLUX, second-generation single-precision, luxury level 1.
        ranlxs1,
        /// RANLUX, second-generation single-precision, luxury level 2.
        ranlxs2,
        /// RANLUX, second-generation double-precision, luxury level 1.
        ranlxd1,
        /// RANLUX, second-generation double-precision, luxury level 2
        /// (highest quality, slowest).
        ranlxd2,

        // Invariant: each tag's name is identical to GSL's own algorithm name
        // (the string `Rng.name` reports). Keep new variants aligned with their
        // `gsl_rng_*` name so a single `@tagName` loop can verify the mapping.
        fn typePtr(self: Generator) [*c]const c.gsl_rng_type {
            return switch (self) {
                .mt19937 => c.gsl_rng_mt19937,
                .taus2 => c.gsl_rng_taus2,
                .taus113 => c.gsl_rng_taus113,
                .cmrg => c.gsl_rng_cmrg,
                .mrg => c.gsl_rng_mrg,
                .gfsr4 => c.gsl_rng_gfsr4,
                .ranlux => c.gsl_rng_ranlux,
                .ranlux389 => c.gsl_rng_ranlux389,
                .ranlxs0 => c.gsl_rng_ranlxs0,
                .ranlxs1 => c.gsl_rng_ranlxs1,
                .ranlxs2 => c.gsl_rng_ranlxs2,
                .ranlxd1 => c.gsl_rng_ranlxd1,
                .ranlxd2 => c.gsl_rng_ranlxd2,
            };
        }
    };

    /// A GSL random number generator. Owns the underlying `gsl_rng` allocation;
    /// call `deinit` to free it.
    ///
    /// Example:
    /// ```
    /// var gen = try rand.Rng.init(.mt19937);
    /// defer gen.deinit();
    /// gen.seed(12345);
    /// const u = gen.uniform();                        // f64 in [0, 1)
    /// const x = rand.Gaussian{ .sigma = 2.0 }.sample(gen); // N(0, sigma=2)
    /// ```
    pub const Rng = struct {
        ptr: *c.gsl_rng,

        /// Allocate a generator of the given algorithm. Fails only if the
        /// underlying allocation fails.
        pub fn init(gen: Generator) error{OutOfMemory}!Rng {
            const p = c.gsl_rng_alloc(gen.typePtr()) orelse return error.OutOfMemory;
            return .{ .ptr = p };
        }

        /// Allocate a generator by its GSL algorithm name (e.g. "mt19937",
        /// "ranlux389", "randu"). This is the escape hatch to the ~60 generators
        /// GSL registers, including the legacy/compatibility ones not surfaced by
        /// `Generator`. Returns `error.UnknownGenerator` if no algorithm matches
        /// the given name, or `error.OutOfMemory` if allocation fails.
        pub fn initByName(gen_name: [:0]const u8) error{ UnknownGenerator, OutOfMemory }!Rng {
            var it: [*c]const [*c]const c.gsl_rng_type = c.gsl_rng_types_setup();
            while (it.* != null) : (it += 1) {
                const t = it.*;
                if (std.mem.orderZ(u8, t.*.name, gen_name) == .eq) {
                    const p = c.gsl_rng_alloc(t) orelse return error.OutOfMemory;
                    return .{ .ptr = p };
                }
            }
            return error.UnknownGenerator;
        }

        /// Allocate GSL's library default generator (`gsl_rng_default`), which is
        /// `mt19937` unless a prior `initFromEnv` overrode the global. The result
        /// is unseeded; call `seed` before use for reproducibility.
        pub fn initDefault() error{OutOfMemory}!Rng {
            const p = c.gsl_rng_alloc(c.gsl_rng_default) orelse return error.OutOfMemory;
            return .{ .ptr = p };
        }

        /// Honor the `GSL_RNG_TYPE` and `GSL_RNG_SEED` environment variables
        /// (falling back to `mt19937` / seed 0), allocate that generator, and
        /// pre-seed it with `gsl_rng_default_seed`. This mirrors GSL's
        /// env-configurable default so the generator and seed can be chosen at
        /// runtime without recompiling.
        ///
        /// Note: `gsl_rng_env_setup` mutates process-global GSL state, so this also
        /// changes what `initDefault` returns afterwards.
        pub fn initFromEnv() error{OutOfMemory}!Rng {
            const t = c.gsl_rng_env_setup();
            const p = c.gsl_rng_alloc(t) orelse return error.OutOfMemory;
            c.gsl_rng_set(p, c.gsl_rng_default_seed);
            return .{ .ptr = p };
        }

        /// Free the underlying generator.
        pub fn deinit(self: Rng) void {
            c.gsl_rng_free(self.ptr);
        }

        /// Seed the generator. The same seed reproduces the same sequence.
        pub fn seed(self: Rng, s: u64) void {
            c.gsl_rng_set(self.ptr, @intCast(s));
        }

        /// Duplicate this generator, including its current internal state.
        pub fn clone(self: Rng) error{OutOfMemory}!Rng {
            const p = c.gsl_rng_clone(self.ptr) orelse return error.OutOfMemory;
            return .{ .ptr = p };
        }

        // --- State serialization ----------------------------------------------
        //
        // For checkpoint/restore of a long stream: snapshot the generator's full
        // internal state to bytes and later resume exactly where it left off.
        // `seed`/`clone` cover in-process reproducibility; these cover crossing a
        // process boundary (e.g. writing a checkpoint to disk).
        //
        // Caveat: the byte format is *not portable*. It depends on the algorithm
        // and the platform (word size, endianness, struct layout), so a snapshot
        // only restores into a generator of the same `Generator` on a compatible
        // build. Treat it as an opaque, same-binary blob — not an archival format.

        /// Number of bytes `saveState` writes for this generator. Runtime-known
        /// (it is `sizeof` an algorithm-specific internal struct), so query it to
        /// size a buffer: `const buf = try alloc(rng.stateSize())`.
        pub fn stateSize(self: Rng) usize {
            return c.gsl_rng_size(self.ptr);
        }

        /// Snapshot the generator's internal state into `buf` and return the
        /// written sub-slice (`buf[0..stateSize()]`). `buf.len` must be at least
        /// `stateSize()`. Restore later with `loadState`. The same buffer can be
        /// reused across repeated checkpoints — no allocation happens here.
        pub fn saveState(self: Rng, buf: []u8) []u8 {
            const n = self.stateSize();
            std.debug.assert(buf.len >= n);
            const src: [*]const u8 = @ptrCast(c.gsl_rng_state(self.ptr).?);
            @memcpy(buf[0..n], src[0..n]);
            return buf[0..n];
        }

        /// Restore internal state previously captured by `saveState`. `bytes.len`
        /// must equal this generator's `stateSize()`, and the bytes must have come
        /// from the same `Generator` on a compatible build (see caveat above).
        pub fn loadState(self: Rng, bytes: []const u8) void {
            const n = self.stateSize();
            std.debug.assert(bytes.len == n);
            const dst: [*]u8 = @ptrCast(c.gsl_rng_state(self.ptr).?);
            @memcpy(dst[0..n], bytes[0..n]);
        }

        /// Name of the underlying algorithm (e.g. "mt19937").
        pub fn name(self: Rng) [:0]const u8 {
            return std.mem.span(c.gsl_rng_name(self.ptr));
        }

        /// Smallest value `next()` can return.
        pub fn minValue(self: Rng) u64 {
            return @intCast(c.gsl_rng_min(self.ptr));
        }

        /// Largest value `next()` can return.
        pub fn maxValue(self: Rng) u64 {
            return @intCast(c.gsl_rng_max(self.ptr));
        }

        // --- Raw / uniform sampling -------------------------------------------

        /// Next raw integer in `[minValue(), maxValue()]`.
        pub fn next(self: Rng) u64 {
            return @intCast(c.gsl_rng_get(self.ptr));
        }

        /// Uniform `f64` in `[0, 1)`.
        pub fn uniform(self: Rng) f64 {
            return c.gsl_rng_uniform(self.ptr);
        }

        /// Uniform `f64` in `(0, 1)` (excludes zero).
        pub fn uniformPos(self: Rng) f64 {
            return c.gsl_rng_uniform_pos(self.ptr);
        }

        /// Uniform integer in `[0, n)`. `n` must be > 0 and <= the generator's
        /// range; otherwise GSL reports an error via its error handler.
        pub fn uniformInt(self: Rng, n: u64) u64 {
            std.debug.assert(n > 0);
            return @intCast(c.gsl_rng_uniform_int(self.ptr, @intCast(n)));
        }

        // --- Shuffling and sampling -------------------------------------------
        //
        // GSL files these under `gsl_randist`, but they are generic RNG operations
        // over caller data (no distribution parameters), so they live on `Rng`.

        /// Randomly permute `slice` in place (a uniform random permutation).
        pub fn shuffle(self: Rng, comptime T: type, slice: []T) void {
            if (slice.len < 2) return;
            c.gsl_ran_shuffle(self.ptr, @ptrCast(slice.ptr), slice.len, @sizeOf(T));
        }

        /// Choose `dest.len` elements from `src` at random *without* replacement,
        /// preserving their relative order, and write them into `dest`. Requires
        /// `dest.len <= src.len`.
        pub fn choose(self: Rng, comptime T: type, dest: []T, src: []const T) void {
            std.debug.assert(dest.len <= src.len);
            _ = c.gsl_ran_choose(
                self.ptr,
                @ptrCast(dest.ptr),
                dest.len,
                @ptrCast(@constCast(src.ptr)),
                src.len,
                @sizeOf(T),
            );
        }

        /// Sample `dest.len` elements from `src` at random *with* replacement,
        /// writing them into `dest`.
        pub fn sample(self: Rng, comptime T: type, dest: []T, src: []const T) void {
            c.gsl_ran_sample(
                self.ptr,
                @ptrCast(dest.ptr),
                dest.len,
                @ptrCast(@constCast(src.ptr)),
                src.len,
                @sizeOf(T),
            );
        }
    };

    // ===== Distributions (gsl_randist + gsl_cdf) ==============================

    // --- Continuous scalar families -------------------------------------------

    /// Normal with mean 0 and standard deviation `sigma`.
    pub const Gaussian = struct {
        sigma: f64,
        pub const Sample = f64;
        pub fn sample(self: Gaussian, r: Rng) f64 {
            return c.gsl_ran_gaussian(r.ptr, self.sigma);
        }
        pub fn pdf(self: Gaussian, x: f64) f64 {
            return c.gsl_ran_gaussian_pdf(x, self.sigma);
        }
        pub fn cdf(self: Gaussian, x: f64) f64 {
            return c.gsl_cdf_gaussian_P(x, self.sigma);
        }
        pub fn sf(self: Gaussian, x: f64) f64 {
            return c.gsl_cdf_gaussian_Q(x, self.sigma);
        }
        pub fn ppf(self: Gaussian, p: f64) f64 {
            return c.gsl_cdf_gaussian_Pinv(p, self.sigma);
        }
        pub fn isf(self: Gaussian, q: f64) f64 {
            return c.gsl_cdf_gaussian_Qinv(q, self.sigma);
        }
    };

    /// Exponential with mean `mu`.
    pub const Exponential = struct {
        mu: f64,
        pub const Sample = f64;
        pub fn sample(self: Exponential, r: Rng) f64 {
            return c.gsl_ran_exponential(r.ptr, self.mu);
        }
        pub fn pdf(self: Exponential, x: f64) f64 {
            return c.gsl_ran_exponential_pdf(x, self.mu);
        }
        pub fn cdf(self: Exponential, x: f64) f64 {
            return c.gsl_cdf_exponential_P(x, self.mu);
        }
        pub fn sf(self: Exponential, x: f64) f64 {
            return c.gsl_cdf_exponential_Q(x, self.mu);
        }
        pub fn ppf(self: Exponential, p: f64) f64 {
            return c.gsl_cdf_exponential_Pinv(p, self.mu);
        }
        pub fn isf(self: Exponential, q: f64) f64 {
            return c.gsl_cdf_exponential_Qinv(q, self.mu);
        }
    };

    /// Uniform on `[a, b)`.
    pub const Flat = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Flat, r: Rng) f64 {
            return c.gsl_ran_flat(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Flat, x: f64) f64 {
            return c.gsl_ran_flat_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Flat, x: f64) f64 {
            return c.gsl_cdf_flat_P(x, self.a, self.b);
        }
        pub fn sf(self: Flat, x: f64) f64 {
            return c.gsl_cdf_flat_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Flat, p: f64) f64 {
            return c.gsl_cdf_flat_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Flat, q: f64) f64 {
            return c.gsl_cdf_flat_Qinv(q, self.a, self.b);
        }
    };

    /// Gamma with shape `a` and scale `b`.
    pub const Gamma = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Gamma, r: Rng) f64 {
            return c.gsl_ran_gamma(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Gamma, x: f64) f64 {
            return c.gsl_ran_gamma_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Gamma, x: f64) f64 {
            return c.gsl_cdf_gamma_P(x, self.a, self.b);
        }
        pub fn sf(self: Gamma, x: f64) f64 {
            return c.gsl_cdf_gamma_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Gamma, p: f64) f64 {
            return c.gsl_cdf_gamma_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Gamma, q: f64) f64 {
            return c.gsl_cdf_gamma_Qinv(q, self.a, self.b);
        }
    };

    /// Beta with shape parameters `a` and `b`.
    pub const Beta = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Beta, r: Rng) f64 {
            return c.gsl_ran_beta(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Beta, x: f64) f64 {
            return c.gsl_ran_beta_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Beta, x: f64) f64 {
            return c.gsl_cdf_beta_P(x, self.a, self.b);
        }
        pub fn sf(self: Beta, x: f64) f64 {
            return c.gsl_cdf_beta_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Beta, p: f64) f64 {
            return c.gsl_cdf_beta_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Beta, q: f64) f64 {
            return c.gsl_cdf_beta_Qinv(q, self.a, self.b);
        }
    };

    /// Chi-squared with `nu` degrees of freedom.
    pub const ChiSquared = struct {
        nu: f64,
        pub const Sample = f64;
        pub fn sample(self: ChiSquared, r: Rng) f64 {
            return c.gsl_ran_chisq(r.ptr, self.nu);
        }
        pub fn pdf(self: ChiSquared, x: f64) f64 {
            return c.gsl_ran_chisq_pdf(x, self.nu);
        }
        pub fn cdf(self: ChiSquared, x: f64) f64 {
            return c.gsl_cdf_chisq_P(x, self.nu);
        }
        pub fn sf(self: ChiSquared, x: f64) f64 {
            return c.gsl_cdf_chisq_Q(x, self.nu);
        }
        pub fn ppf(self: ChiSquared, p: f64) f64 {
            return c.gsl_cdf_chisq_Pinv(p, self.nu);
        }
        pub fn isf(self: ChiSquared, q: f64) f64 {
            return c.gsl_cdf_chisq_Qinv(q, self.nu);
        }
    };

    // --- Discrete scalar families ---------------------------------------------

    /// Poisson with mean `mu`.
    pub const Poisson = struct {
        mu: f64,
        pub const Sample = u32;
        pub fn sample(self: Poisson, r: Rng) u32 {
            return @intCast(c.gsl_ran_poisson(r.ptr, self.mu));
        }
        pub fn pdf(self: Poisson, k: u32) f64 {
            return c.gsl_ran_poisson_pdf(k, self.mu);
        }
        pub fn cdf(self: Poisson, k: u32) f64 {
            return c.gsl_cdf_poisson_P(k, self.mu);
        }
        pub fn sf(self: Poisson, k: u32) f64 {
            return c.gsl_cdf_poisson_Q(k, self.mu);
        }
    };

    /// Bernoulli with success probability `p`.
    pub const Bernoulli = struct {
        p: f64,
        pub const Sample = bool;
        pub fn sample(self: Bernoulli, r: Rng) bool {
            return c.gsl_ran_bernoulli(r.ptr, self.p) != 0;
        }
        pub fn pdf(self: Bernoulli, x: bool) f64 {
            return c.gsl_ran_bernoulli_pdf(@intFromBool(x), self.p);
        }
    };

    /// Binomial: `n` trials each with success probability `p`.
    pub const Binomial = struct {
        p: f64,
        n: u32,
        pub const Sample = u32;
        pub fn sample(self: Binomial, r: Rng) u32 {
            return @intCast(c.gsl_ran_binomial(r.ptr, self.p, @intCast(self.n)));
        }
        pub fn pdf(self: Binomial, k: u32) f64 {
            return c.gsl_ran_binomial_pdf(k, self.p, @intCast(self.n));
        }
        pub fn cdf(self: Binomial, k: u32) f64 {
            return c.gsl_cdf_binomial_P(k, self.p, self.n);
        }
        pub fn sf(self: Binomial, k: u32) f64 {
            return c.gsl_cdf_binomial_Q(k, self.p, self.n);
        }
    };

    // --- Vector-valued families -----------------------------------------------

    /// Bivariate Gaussian with per-axis standard deviations and correlation `rho`.
    /// A fixed-size vector family, so its sample is returned by value as `[2]f64`.
    pub const BivariateGaussian = struct {
        sigma_x: f64,
        sigma_y: f64,
        rho: f64,
        pub const Sample = [2]f64;
        pub fn sample(self: BivariateGaussian, r: Rng) [2]f64 {
            var out: [2]f64 = undefined;
            c.gsl_ran_bivariate_gaussian(r.ptr, self.sigma_x, self.sigma_y, self.rho, &out[0], &out[1]);
            return out;
        }
        pub fn pdf(self: BivariateGaussian, x: [2]f64) f64 {
            return c.gsl_ran_bivariate_gaussian_pdf(x[0], x[1], self.sigma_x, self.sigma_y, self.rho);
        }
    };

    /// Dirichlet distribution over the `alpha.len - 1` simplex. Sampling writes a
    /// probability vector (summing to 1) into caller-provided storage.
    pub const Dirichlet = struct {
        alpha: []const f64,
        pub const Elem = f64;
        pub const Sample = []f64;

        pub fn dim(self: Dirichlet) usize {
            return self.alpha.len;
        }
        pub fn sample(self: Dirichlet, r: Rng, out: []f64) void {
            std.debug.assert(out.len == self.alpha.len);
            c.gsl_ran_dirichlet(r.ptr, self.alpha.len, self.alpha.ptr, out.ptr);
        }
        pub fn pdf(self: Dirichlet, theta: []const f64) f64 {
            std.debug.assert(theta.len == self.alpha.len);
            return c.gsl_ran_dirichlet_pdf(self.alpha.len, self.alpha.ptr, theta.ptr);
        }
        pub fn logpdf(self: Dirichlet, theta: []const f64) f64 {
            std.debug.assert(theta.len == self.alpha.len);
            return c.gsl_ran_dirichlet_lnpdf(self.alpha.len, self.alpha.ptr, theta.ptr);
        }
    };

    /// Multinomial: `n` independent trials over `p.len` categories with (possibly
    /// unnormalized) weights `p`. Sampling writes per-category counts into `out`.
    pub const Multinomial = struct {
        n: u32,
        p: []const f64,
        pub const Elem = u32;
        pub const Sample = []u32;

        pub fn categories(self: Multinomial) usize {
            return self.p.len;
        }
        pub fn sample(self: Multinomial, r: Rng, out: []u32) void {
            std.debug.assert(out.len == self.p.len);
            c.gsl_ran_multinomial(r.ptr, self.p.len, @intCast(self.n), self.p.ptr, @ptrCast(out.ptr));
        }
        pub fn pdf(self: Multinomial, x: []const u32) f64 {
            std.debug.assert(x.len == self.p.len);
            return c.gsl_ran_multinomial_pdf(self.p.len, self.p.ptr, @ptrCast(x.ptr));
        }
        pub fn logpdf(self: Multinomial, x: []const u32) f64 {
            std.debug.assert(x.len == self.p.len);
            return c.gsl_ran_multinomial_lnpdf(self.p.len, self.p.ptr, @ptrCast(x.ptr));
        }
    };

    // --- More continuous scalar families --------------------------------------

    /// Standard normal (mean 0, variance 1). Carries the full CDF family.
    pub const UnitGaussian = struct {
        pub const Sample = f64;
        pub fn sample(self: UnitGaussian, r: Rng) f64 {
            _ = self;
            return c.gsl_ran_ugaussian(r.ptr);
        }
        pub fn pdf(self: UnitGaussian, x: f64) f64 {
            _ = self;
            return c.gsl_ran_ugaussian_pdf(x);
        }
        pub fn cdf(self: UnitGaussian, x: f64) f64 {
            _ = self;
            return c.gsl_cdf_ugaussian_P(x);
        }
        pub fn sf(self: UnitGaussian, x: f64) f64 {
            _ = self;
            return c.gsl_cdf_ugaussian_Q(x);
        }
        pub fn ppf(self: UnitGaussian, p: f64) f64 {
            _ = self;
            return c.gsl_cdf_ugaussian_Pinv(p);
        }
        pub fn isf(self: UnitGaussian, q: f64) f64 {
            _ = self;
            return c.gsl_cdf_ugaussian_Qinv(q);
        }
    };

    /// Upper tail of `Gaussian(0, sigma)` conditioned on `x > a` (with `a > 0`).
    pub const GaussianTail = struct {
        a: f64,
        sigma: f64,
        pub const Sample = f64;
        pub fn sample(self: GaussianTail, r: Rng) f64 {
            return c.gsl_ran_gaussian_tail(r.ptr, self.a, self.sigma);
        }
        pub fn pdf(self: GaussianTail, x: f64) f64 {
            return c.gsl_ran_gaussian_tail_pdf(x, self.a, self.sigma);
        }
    };

    /// Upper tail of the standard normal conditioned on `x > a` (with `a > 0`).
    pub const UnitGaussianTail = struct {
        a: f64,
        pub const Sample = f64;
        pub fn sample(self: UnitGaussianTail, r: Rng) f64 {
            return c.gsl_ran_ugaussian_tail(r.ptr, self.a);
        }
        pub fn pdf(self: UnitGaussianTail, x: f64) f64 {
            return c.gsl_ran_ugaussian_tail_pdf(x, self.a);
        }
    };

    /// Laplace (double exponential) with width `a`.
    pub const Laplace = struct {
        a: f64,
        pub const Sample = f64;
        pub fn sample(self: Laplace, r: Rng) f64 {
            return c.gsl_ran_laplace(r.ptr, self.a);
        }
        pub fn pdf(self: Laplace, x: f64) f64 {
            return c.gsl_ran_laplace_pdf(x, self.a);
        }
        pub fn cdf(self: Laplace, x: f64) f64 {
            return c.gsl_cdf_laplace_P(x, self.a);
        }
        pub fn sf(self: Laplace, x: f64) f64 {
            return c.gsl_cdf_laplace_Q(x, self.a);
        }
        pub fn ppf(self: Laplace, p: f64) f64 {
            return c.gsl_cdf_laplace_Pinv(p, self.a);
        }
        pub fn isf(self: Laplace, q: f64) f64 {
            return c.gsl_cdf_laplace_Qinv(q, self.a);
        }
    };

    /// Exponential power family with scale `a` and exponent `b` (`b == 2` is
    /// Gaussian, `b == 1` is Laplace). GSL provides only the lower/upper CDF.
    pub const ExpPower = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: ExpPower, r: Rng) f64 {
            return c.gsl_ran_exppow(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: ExpPower, x: f64) f64 {
            return c.gsl_ran_exppow_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: ExpPower, x: f64) f64 {
            return c.gsl_cdf_exppow_P(x, self.a, self.b);
        }
        pub fn sf(self: ExpPower, x: f64) f64 {
            return c.gsl_cdf_exppow_Q(x, self.a, self.b);
        }
    };

    /// Cauchy (Lorentz) with scale `a`.
    pub const Cauchy = struct {
        a: f64,
        pub const Sample = f64;
        pub fn sample(self: Cauchy, r: Rng) f64 {
            return c.gsl_ran_cauchy(r.ptr, self.a);
        }
        pub fn pdf(self: Cauchy, x: f64) f64 {
            return c.gsl_ran_cauchy_pdf(x, self.a);
        }
        pub fn cdf(self: Cauchy, x: f64) f64 {
            return c.gsl_cdf_cauchy_P(x, self.a);
        }
        pub fn sf(self: Cauchy, x: f64) f64 {
            return c.gsl_cdf_cauchy_Q(x, self.a);
        }
        pub fn ppf(self: Cauchy, p: f64) f64 {
            return c.gsl_cdf_cauchy_Pinv(p, self.a);
        }
        pub fn isf(self: Cauchy, q: f64) f64 {
            return c.gsl_cdf_cauchy_Qinv(q, self.a);
        }
    };

    /// Rayleigh with scale `sigma`.
    pub const Rayleigh = struct {
        sigma: f64,
        pub const Sample = f64;
        pub fn sample(self: Rayleigh, r: Rng) f64 {
            return c.gsl_ran_rayleigh(r.ptr, self.sigma);
        }
        pub fn pdf(self: Rayleigh, x: f64) f64 {
            return c.gsl_ran_rayleigh_pdf(x, self.sigma);
        }
        pub fn cdf(self: Rayleigh, x: f64) f64 {
            return c.gsl_cdf_rayleigh_P(x, self.sigma);
        }
        pub fn sf(self: Rayleigh, x: f64) f64 {
            return c.gsl_cdf_rayleigh_Q(x, self.sigma);
        }
        pub fn ppf(self: Rayleigh, p: f64) f64 {
            return c.gsl_cdf_rayleigh_Pinv(p, self.sigma);
        }
        pub fn isf(self: Rayleigh, q: f64) f64 {
            return c.gsl_cdf_rayleigh_Qinv(q, self.sigma);
        }
    };

    /// Upper tail of `Rayleigh(sigma)` conditioned on `x > a`.
    pub const RayleighTail = struct {
        a: f64,
        sigma: f64,
        pub const Sample = f64;
        pub fn sample(self: RayleighTail, r: Rng) f64 {
            return c.gsl_ran_rayleigh_tail(r.ptr, self.a, self.sigma);
        }
        pub fn pdf(self: RayleighTail, x: f64) f64 {
            return c.gsl_ran_rayleigh_tail_pdf(x, self.a, self.sigma);
        }
    };

    /// Landau distribution (no parameters). Sampling and density only.
    pub const Landau = struct {
        pub const Sample = f64;
        pub fn sample(self: Landau, r: Rng) f64 {
            _ = self;
            return c.gsl_ran_landau(r.ptr);
        }
        pub fn pdf(self: Landau, x: f64) f64 {
            _ = self;
            return c.gsl_ran_landau_pdf(x);
        }
    };

    /// Symmetric Levy alpha-stable with scale `c` and exponent `alpha` in `(0, 2]`
    /// (`alpha == 2` is Gaussian, `alpha == 1` is Cauchy). GSL exposes sampling
    /// only — there is no closed-form density.
    pub const Levy = struct {
        c: f64,
        alpha: f64,
        pub const Sample = f64;
        pub fn sample(self: Levy, r: Rng) f64 {
            return c.gsl_ran_levy(r.ptr, self.c, self.alpha);
        }
    };

    /// Skew Levy alpha-stable with scale `c`, exponent `alpha`, and skewness
    /// `beta` in `[-1, 1]`. Sampling only.
    pub const LevySkew = struct {
        c: f64,
        alpha: f64,
        beta: f64,
        pub const Sample = f64;
        pub fn sample(self: LevySkew, r: Rng) f64 {
            return c.gsl_ran_levy_skew(r.ptr, self.c, self.alpha, self.beta);
        }
    };

    /// Log-normal: `log(x)` is `Gaussian(zeta, sigma)`.
    pub const Lognormal = struct {
        zeta: f64,
        sigma: f64,
        pub const Sample = f64;
        pub fn sample(self: Lognormal, r: Rng) f64 {
            return c.gsl_ran_lognormal(r.ptr, self.zeta, self.sigma);
        }
        pub fn pdf(self: Lognormal, x: f64) f64 {
            return c.gsl_ran_lognormal_pdf(x, self.zeta, self.sigma);
        }
        pub fn cdf(self: Lognormal, x: f64) f64 {
            return c.gsl_cdf_lognormal_P(x, self.zeta, self.sigma);
        }
        pub fn sf(self: Lognormal, x: f64) f64 {
            return c.gsl_cdf_lognormal_Q(x, self.zeta, self.sigma);
        }
        pub fn ppf(self: Lognormal, p: f64) f64 {
            return c.gsl_cdf_lognormal_Pinv(p, self.zeta, self.sigma);
        }
        pub fn isf(self: Lognormal, q: f64) f64 {
            return c.gsl_cdf_lognormal_Qinv(q, self.zeta, self.sigma);
        }
    };

    /// F-distribution with `nu1` and `nu2` degrees of freedom.
    pub const FDist = struct {
        nu1: f64,
        nu2: f64,
        pub const Sample = f64;
        pub fn sample(self: FDist, r: Rng) f64 {
            return c.gsl_ran_fdist(r.ptr, self.nu1, self.nu2);
        }
        pub fn pdf(self: FDist, x: f64) f64 {
            return c.gsl_ran_fdist_pdf(x, self.nu1, self.nu2);
        }
        pub fn cdf(self: FDist, x: f64) f64 {
            return c.gsl_cdf_fdist_P(x, self.nu1, self.nu2);
        }
        pub fn sf(self: FDist, x: f64) f64 {
            return c.gsl_cdf_fdist_Q(x, self.nu1, self.nu2);
        }
        pub fn ppf(self: FDist, p: f64) f64 {
            return c.gsl_cdf_fdist_Pinv(p, self.nu1, self.nu2);
        }
        pub fn isf(self: FDist, q: f64) f64 {
            return c.gsl_cdf_fdist_Qinv(q, self.nu1, self.nu2);
        }
    };

    /// Student's t-distribution with `nu` degrees of freedom.
    pub const TDist = struct {
        nu: f64,
        pub const Sample = f64;
        pub fn sample(self: TDist, r: Rng) f64 {
            return c.gsl_ran_tdist(r.ptr, self.nu);
        }
        pub fn pdf(self: TDist, x: f64) f64 {
            return c.gsl_ran_tdist_pdf(x, self.nu);
        }
        pub fn cdf(self: TDist, x: f64) f64 {
            return c.gsl_cdf_tdist_P(x, self.nu);
        }
        pub fn sf(self: TDist, x: f64) f64 {
            return c.gsl_cdf_tdist_Q(x, self.nu);
        }
        pub fn ppf(self: TDist, p: f64) f64 {
            return c.gsl_cdf_tdist_Pinv(p, self.nu);
        }
        pub fn isf(self: TDist, q: f64) f64 {
            return c.gsl_cdf_tdist_Qinv(q, self.nu);
        }
    };

    /// Logistic with scale `a`.
    pub const Logistic = struct {
        a: f64,
        pub const Sample = f64;
        pub fn sample(self: Logistic, r: Rng) f64 {
            return c.gsl_ran_logistic(r.ptr, self.a);
        }
        pub fn pdf(self: Logistic, x: f64) f64 {
            return c.gsl_ran_logistic_pdf(x, self.a);
        }
        pub fn cdf(self: Logistic, x: f64) f64 {
            return c.gsl_cdf_logistic_P(x, self.a);
        }
        pub fn sf(self: Logistic, x: f64) f64 {
            return c.gsl_cdf_logistic_Q(x, self.a);
        }
        pub fn ppf(self: Logistic, p: f64) f64 {
            return c.gsl_cdf_logistic_Pinv(p, self.a);
        }
        pub fn isf(self: Logistic, q: f64) f64 {
            return c.gsl_cdf_logistic_Qinv(q, self.a);
        }
    };

    /// Pareto with exponent `a` and scale `b` (support `x >= b`).
    pub const Pareto = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Pareto, r: Rng) f64 {
            return c.gsl_ran_pareto(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Pareto, x: f64) f64 {
            return c.gsl_ran_pareto_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Pareto, x: f64) f64 {
            return c.gsl_cdf_pareto_P(x, self.a, self.b);
        }
        pub fn sf(self: Pareto, x: f64) f64 {
            return c.gsl_cdf_pareto_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Pareto, p: f64) f64 {
            return c.gsl_cdf_pareto_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Pareto, q: f64) f64 {
            return c.gsl_cdf_pareto_Qinv(q, self.a, self.b);
        }
    };

    /// Weibull with scale `a` and shape `b`.
    pub const Weibull = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Weibull, r: Rng) f64 {
            return c.gsl_ran_weibull(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Weibull, x: f64) f64 {
            return c.gsl_ran_weibull_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Weibull, x: f64) f64 {
            return c.gsl_cdf_weibull_P(x, self.a, self.b);
        }
        pub fn sf(self: Weibull, x: f64) f64 {
            return c.gsl_cdf_weibull_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Weibull, p: f64) f64 {
            return c.gsl_cdf_weibull_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Weibull, q: f64) f64 {
            return c.gsl_cdf_weibull_Qinv(q, self.a, self.b);
        }
    };

    /// Type-1 Gumbel (extreme value) with parameters `a` and `b`.
    pub const Gumbel1 = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Gumbel1, r: Rng) f64 {
            return c.gsl_ran_gumbel1(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Gumbel1, x: f64) f64 {
            return c.gsl_ran_gumbel1_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Gumbel1, x: f64) f64 {
            return c.gsl_cdf_gumbel1_P(x, self.a, self.b);
        }
        pub fn sf(self: Gumbel1, x: f64) f64 {
            return c.gsl_cdf_gumbel1_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Gumbel1, p: f64) f64 {
            return c.gsl_cdf_gumbel1_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Gumbel1, q: f64) f64 {
            return c.gsl_cdf_gumbel1_Qinv(q, self.a, self.b);
        }
    };

    /// Type-2 Gumbel (extreme value) with parameters `a` and `b`.
    pub const Gumbel2 = struct {
        a: f64,
        b: f64,
        pub const Sample = f64;
        pub fn sample(self: Gumbel2, r: Rng) f64 {
            return c.gsl_ran_gumbel2(r.ptr, self.a, self.b);
        }
        pub fn pdf(self: Gumbel2, x: f64) f64 {
            return c.gsl_ran_gumbel2_pdf(x, self.a, self.b);
        }
        pub fn cdf(self: Gumbel2, x: f64) f64 {
            return c.gsl_cdf_gumbel2_P(x, self.a, self.b);
        }
        pub fn sf(self: Gumbel2, x: f64) f64 {
            return c.gsl_cdf_gumbel2_Q(x, self.a, self.b);
        }
        pub fn ppf(self: Gumbel2, p: f64) f64 {
            return c.gsl_cdf_gumbel2_Pinv(p, self.a, self.b);
        }
        pub fn isf(self: Gumbel2, q: f64) f64 {
            return c.gsl_cdf_gumbel2_Qinv(q, self.a, self.b);
        }
    };

    /// Erlang with scale `a` and (real-valued) shape `n`; the Gamma distribution
    /// specialized to integer shape. GSL provides no CDF for this family.
    pub const Erlang = struct {
        a: f64,
        n: f64,
        pub const Sample = f64;
        pub fn sample(self: Erlang, r: Rng) f64 {
            return c.gsl_ran_erlang(r.ptr, self.a, self.n);
        }
        pub fn pdf(self: Erlang, x: f64) f64 {
            return c.gsl_ran_erlang_pdf(x, self.a, self.n);
        }
    };

    // --- More discrete scalar families ----------------------------------------

    /// Geometric: number of trials up to and including the first success, each
    /// trial succeeding with probability `p` (support `k >= 1`).
    pub const Geometric = struct {
        p: f64,
        pub const Sample = u32;
        pub fn sample(self: Geometric, r: Rng) u32 {
            return @intCast(c.gsl_ran_geometric(r.ptr, self.p));
        }
        pub fn pdf(self: Geometric, k: u32) f64 {
            return c.gsl_ran_geometric_pdf(k, self.p);
        }
        pub fn cdf(self: Geometric, k: u32) f64 {
            return c.gsl_cdf_geometric_P(k, self.p);
        }
        pub fn sf(self: Geometric, k: u32) f64 {
            return c.gsl_cdf_geometric_Q(k, self.p);
        }
    };

    /// Negative binomial: number of failures before the `n`-th success, where `n`
    /// may be real-valued. Success probability `p`.
    pub const NegativeBinomial = struct {
        p: f64,
        n: f64,
        pub const Sample = u32;
        pub fn sample(self: NegativeBinomial, r: Rng) u32 {
            return @intCast(c.gsl_ran_negative_binomial(r.ptr, self.p, self.n));
        }
        pub fn pdf(self: NegativeBinomial, k: u32) f64 {
            return c.gsl_ran_negative_binomial_pdf(k, self.p, self.n);
        }
        pub fn cdf(self: NegativeBinomial, k: u32) f64 {
            return c.gsl_cdf_negative_binomial_P(k, self.p, self.n);
        }
        pub fn sf(self: NegativeBinomial, k: u32) f64 {
            return c.gsl_cdf_negative_binomial_Q(k, self.p, self.n);
        }
    };

    /// Pascal: the negative binomial restricted to an integer number of successes
    /// `n`. Success probability `p`.
    pub const Pascal = struct {
        p: f64,
        n: u32,
        pub const Sample = u32;
        pub fn sample(self: Pascal, r: Rng) u32 {
            return @intCast(c.gsl_ran_pascal(r.ptr, self.p, self.n));
        }
        pub fn pdf(self: Pascal, k: u32) f64 {
            return c.gsl_ran_pascal_pdf(k, self.p, self.n);
        }
        pub fn cdf(self: Pascal, k: u32) f64 {
            return c.gsl_cdf_pascal_P(k, self.p, self.n);
        }
        pub fn sf(self: Pascal, k: u32) f64 {
            return c.gsl_cdf_pascal_Q(k, self.p, self.n);
        }
    };

    /// Hypergeometric: draws of one type in a sample of size `t` taken without
    /// replacement from a population of `n1` of that type and `n2` of the other.
    pub const Hypergeometric = struct {
        n1: u32,
        n2: u32,
        t: u32,
        pub const Sample = u32;
        pub fn sample(self: Hypergeometric, r: Rng) u32 {
            return @intCast(c.gsl_ran_hypergeometric(r.ptr, self.n1, self.n2, self.t));
        }
        pub fn pdf(self: Hypergeometric, k: u32) f64 {
            return c.gsl_ran_hypergeometric_pdf(k, self.n1, self.n2, self.t);
        }
        pub fn cdf(self: Hypergeometric, k: u32) f64 {
            return c.gsl_cdf_hypergeometric_P(k, self.n1, self.n2, self.t);
        }
        pub fn sf(self: Hypergeometric, k: u32) f64 {
            return c.gsl_cdf_hypergeometric_Q(k, self.n1, self.n2, self.t);
        }
    };

    /// Logarithmic distribution with parameter `p` in `(0, 1)` (support `k >= 1`).
    /// GSL provides no CDF for this family.
    pub const Logarithmic = struct {
        p: f64,
        pub const Sample = u32;
        pub fn sample(self: Logarithmic, r: Rng) u32 {
            return @intCast(c.gsl_ran_logarithmic(r.ptr, self.p));
        }
        pub fn pdf(self: Logarithmic, k: u32) f64 {
            return c.gsl_ran_logarithmic_pdf(k, self.p);
        }
    };

    // --- Stateful general discrete --------------------------------------------

    /// An arbitrary discrete distribution over `0..K` given (possibly unnormalized)
    /// nonnegative weights. Unlike the other families this one owns a precomputed
    /// lookup table (Walker's alias method), so it must be freed with `deinit`.
    ///
    /// ```
    /// var g = try rand.General.init(&.{ 0.1, 0.3, 0.6 });
    /// defer g.deinit();
    /// const k = g.sample(gen);   // 0, 1, or 2
    /// ```
    pub const General = struct {
        table: *c.gsl_ran_discrete_t,
        pub const Sample = usize;

        /// Precompute the lookup table for the given weights. Fails only if the
        /// underlying allocation fails.
        pub fn init(weights: []const f64) error{OutOfMemory}!General {
            const t = c.gsl_ran_discrete_preproc(weights.len, weights.ptr) orelse
                return error.OutOfMemory;
            return .{ .table = t };
        }
        /// Free the precomputed lookup table.
        pub fn deinit(self: General) void {
            c.gsl_ran_discrete_free(self.table);
        }
        pub fn sample(self: General, r: Rng) usize {
            return c.gsl_ran_discrete(r.ptr, self.table);
        }
        pub fn pdf(self: General, k: usize) f64 {
            return c.gsl_ran_discrete_pdf(k, self.table);
        }
    };

    // --- Random directions ----------------------------------------------------
    //
    // Uniformly distributed unit vectors. These are genuine random variates
    // (unlike the generic shuffling helpers on `Rng`), so they sit among the
    // distribution families rather than on the generator.

    /// A uniform random unit vector in the plane, returned as `.{ x, y }`.
    pub fn dir2d(r: Rng) [2]f64 {
        var out: [2]f64 = undefined;
        c.gsl_ran_dir_2d(r.ptr, &out[0], &out[1]);
        return out;
    }

    /// A uniform random unit vector on the sphere, returned as `.{ x, y, z }`.
    pub fn dir3d(r: Rng) [3]f64 {
        var out: [3]f64 = undefined;
        c.gsl_ran_dir_3d(r.ptr, &out[0], &out[1], &out[2]);
        return out;
    }

    /// A uniform random unit vector in `out.len` dimensions, written into `out`.
    pub fn dirNd(r: Rng, out: []f64) void {
        c.gsl_ran_dir_nd(r.ptr, out.len, out.ptr);
    }
};

/// # Quasi-random sequences (`gsl_qrng`) — reserved, not yet implemented
///
/// GSL's quasi-random generators (`gsl_qrng`) produce low-discrepancy sequences
/// (Sobol, Halton, reverse-Halton, Niederreiter) for quasi-Monte Carlo
/// integration. They deliberately live *outside* `rand`: they are a separate GSL
/// module (`gsl_qrng.h`) with a different shape — deterministic, dimension-
/// parameterized space-filling sequences with no seeding and no distributions
/// attached, rather than pseudo-random streams. Folding them into `rand` would
/// blur that boundary, so when they are wrapped they will get their own
/// namespace here.
///
/// Until then this namespace is intentionally a compile error; drop down to the
/// raw `c.gsl_qrng_*` API (e.g. `c.gsl_qrng_alloc(c.gsl_qrng_sobol, dim)`) if you
/// need quasi-random sequences today.
pub const qrng = @compileError(
    "rand-adjacent quasi-random generators (gsl_qrng) are not yet wrapped; " ++
        "use the raw c.gsl_qrng_* API for now (see the `qrng` docs).",
);

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

/// Maps a Zig element type to the GSL statistics module infix used in the C
/// symbol name `gsl_stats_<infix>...`. The chosen module is the one whose C
/// element type has an identical size and signedness on the target platform,
/// so the raw pointer handed to GSL is always reinterpreted correctly. Types
/// GSL has no matching module for are rejected at comptime.
fn statsInfix(comptime T: type) [:0]const u8 {
    switch (@typeInfo(T)) {
        .float => switch (T) {
            f32 => return "float_",
            f64 => return "",
            else => @compileError("gsl.stats: unsupported float element type '" ++ @typeName(T) ++ "'; only f32 and f64 are supported (GSL's long double module is intentionally omitted)"),
        },
        .int => |info| {
            // (infix, C element type) candidates. The unsigned variants come
            // first so that on the rare unsigned-`char` platform `u8`
            // deterministically resolves to `uchar_` rather than `char_`.
            const candidates = .{
                .{ "uchar_", u8 },        .{ "char_", c_char },
                .{ "ushort_", c_ushort }, .{ "short_", c_short },
                .{ "uint_", c_uint },     .{ "int_", c_int },
                .{ "ulong_", c_ulong },   .{ "long_", c_long },
            };
            inline for (candidates) |cand| {
                const CT = cand[1];
                if (@sizeOf(T) == @sizeOf(CT) and info.signedness == @typeInfo(CT).int.signedness)
                    return cand[0];
            }
            @compileError(std.fmt.comptimePrint("gsl.stats: no GSL statistics module matches element type '{s}' ({d}-bit {s}); GSL provides only char/short/int/long-sized modules", .{ @typeName(T), @bitSizeOf(T), @tagName(info.signedness) }));
        },
        else => @compileError("gsl.stats: unsupported element type '" ++ @typeName(T) ++ "'; expected a Zig integer or f32/f64"),
    }
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
/// ## Omitted from GSL
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
//
// Testing conventions for these bindings
// --------------------------------------
// These tests exercise the *binding layer*, not GSL's math. GSL is already a
// mature, heavily tested library, so re-deriving its numerics here would add
// little and be brittle. Instead the suite is built to catch the bugs a binding
// actually introduces: a mistyped C symbol name, swapped or wrong-typed
// arguments, a bad `Sample`/return type, or API-shape drift. Concretely:
//
//   - Invoke every wrapped symbol at least once. A wrapper that no test calls is
//     effectively unverified; new wrappers should come with a test that reaches
//     them (the `inline for` sweeps over distribution families, `Generator`
//     variants, and `stats(T)` element types exist to make this cheap).
//   - Prefer oracle checks over hand-computed expectations. The strongest test
//     seeds two identical generators and asserts a wrapper produces the
//     bit-identical result of the raw `c.gsl_*` call (see the "equals the
//     underlying GSL call" tests). A handful of known closed-form anchors
//     (e.g. the standard-normal peak) guard against a wrapper that calls the
//     *wrong* GSL routine but still returns plausible numbers.
//   - Statistical assertions stay loose. Where a test samples and checks a mean
//     or spread, use wide tolerances (it verifies "this is wired up", not the
//     quality of GSL's RNG).
//
// Compile-time contract checks use `if (false)` toggle blocks
// -----------------------------------------------------------
// Some guarantees are compile errors by design (e.g. `stats(i128)`,
// `stats(i32).weighted`, referencing the reserved `qrng` namespace). A passing
// test can't contain code that fails to compile, so those cases live in
// `if (false) { ... }` blocks (see "unsupported instantiations are rejected at
// comptime"). To verify one, flip its block to `if (true)`, confirm you get the
// intended `@compileError`, then revert to `if (false)`.

test "rand: default constructors" {
    // The library default is mt19937 (unless env overrode the global).
    var d = try rand.Rng.initDefault();
    defer d.deinit();
    try testing.expectEqualStrings("mt19937", d.name());

    // initFromEnv honors GSL_RNG_TYPE/SEED; with none set it also yields
    // mt19937, and it is pre-seeded so it produces values without an explicit
    // seed call.
    var e = try rand.Rng.initFromEnv();
    defer e.deinit();
    try testing.expectEqualStrings("mt19937", e.name());
    _ = e.next();
}

test "rand: deterministic sequence for a fixed seed" {
    var a = try rand.Rng.init(.mt19937);
    defer a.deinit();
    var b = try rand.Rng.init(.mt19937);
    defer b.deinit();

    a.seed(42);
    b.seed(42);
    try testing.expectEqualStrings("mt19937", a.name());
    for (0..16) |_| {
        try testing.expectEqual(a.next(), b.next());
    }
}

test "rand: uniform stays in range and uniformInt respects bound" {
    var r = try rand.Rng.init(.taus2);
    defer r.deinit();
    r.seed(1);

    for (0..1000) |_| {
        const u = r.uniform();
        try testing.expect(u >= 0.0 and u < 1.0);
        try testing.expect(r.uniformInt(6) < 6);
        try testing.expect(r.uniformPos() > 0.0);
    }
}

test "rand: clone reproduces subsequent draws" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(7);
    _ = r.next();

    var copy = try r.clone();
    defer copy.deinit();
    for (0..8) |_| {
        try testing.expectEqual(r.next(), copy.next());
    }
}

test "randist: sample means converge to distribution means" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(2024);

    const n = 200_000;
    var g_sum: f64 = 0;
    var e_sum: f64 = 0;
    var heads: u64 = 0;
    const gaussian = rand.Gaussian{ .sigma = 2.0 };
    const expo = rand.Exponential{ .mu = 3.0 };
    const bern = rand.Bernoulli{ .p = 0.25 };
    for (0..n) |_| {
        g_sum += gaussian.sample(r);
        e_sum += expo.sample(r);
        if (bern.sample(r)) heads += 1;
    }
    try testing.expectApproxEqAbs(@as(f64, 0.0), g_sum / n, 0.05); // mean 0
    try testing.expectApproxEqAbs(@as(f64, 3.0), e_sum / n, 0.05); // mean mu
    const frac: f64 = @as(f64, @floatFromInt(heads)) / n;
    try testing.expectApproxEqAbs(@as(f64, 0.25), frac, 0.01);
}

test "pdf: standard normal peak and symmetry" {
    const g = rand.Gaussian{ .sigma = 1.0 };
    try testing.expectApproxEqAbs(@as(f64, 0.3989422804014327), g.pdf(0.0), 1e-12);
    try testing.expectApproxEqAbs(g.pdf(-1.3), g.pdf(1.3), 1e-15);
}

test "distribution: scalar sample equals the underlying GSL call under the same seed" {
    var a = try rand.Rng.init(.mt19937);
    defer a.deinit();
    var b = try rand.Rng.init(.mt19937);
    defer b.deinit();
    a.seed(99);
    b.seed(99);

    const g = rand.Gaussian{ .sigma = 2.0 };
    // Same generator state + same algorithm => bit-identical draw as the raw C call.
    try testing.expectEqual(c.gsl_ran_gaussian(a.ptr, 2.0), g.sample(b));
}

test "distribution: typed pdf for discrete families" {
    const p = rand.Poisson{ .mu = 4.0 };
    // Sample and pdf speak u32, not f64.
    const k: u32 = 4;
    try testing.expect(p.pdf(k) > 0.0);

    const bern = rand.Bernoulli{ .p = 0.3 };
    // Bernoulli's Sample type is bool.
    comptime std.debug.assert(rand.Bernoulli.Sample == bool);
    try testing.expectApproxEqAbs(@as(f64, 1.0), bern.pdf(true) + bern.pdf(false), 1e-12);

    const bino = rand.Binomial{ .p = 0.5, .n = 10 };
    try testing.expect(bino.pdf(5) > 0.0);
}

test "distribution: sampled Gaussian has the expected spread" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(2024);

    var buf: [50_000]f64 = undefined;
    const g = rand.Gaussian{ .sigma = 3.0 };
    for (&buf) |*x| x.* = g.sample(r);
    try testing.expectApproxEqAbs(@as(f64, 0.0), stats(f64).mean(.fromSlice(&buf)), 0.05);
    try testing.expectApproxEqAbs(@as(f64, 3.0), stats(f64).sd(.fromSlice(&buf)), 0.05);
}

test "distribution: Dirichlet sample lands on the simplex" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(1);

    const d = rand.Dirichlet{ .alpha = &.{ 1.0, 2.0, 3.0 } };
    var theta: [3]f64 = undefined;
    d.sample(r, &theta);

    var sum: f64 = 0;
    for (theta) |t| {
        try testing.expect(t >= 0.0 and t <= 1.0);
        sum += t;
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
    try testing.expect(d.pdf(&theta) > 0.0);
}

test "distribution: Multinomial counts sum to n" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(7);

    const m = rand.Multinomial{ .n = 100, .p = &.{ 0.2, 0.3, 0.5 } };
    var counts: [3]u32 = undefined;
    m.sample(r, &counts);

    var total: u32 = 0;
    for (counts) |c_| total += c_;
    try testing.expectEqual(@as(u32, 100), total);
    try testing.expect(m.pdf(&counts) > 0.0);
}

test "distribution: BivariateGaussian returns a fixed-size vector" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(3);

    const bv = rand.BivariateGaussian{ .sigma_x = 1.0, .sigma_y = 2.0, .rho = 0.5 };
    const v: [2]f64 = bv.sample(r);
    try testing.expect(std.math.isFinite(v[0]) and std.math.isFinite(v[1]));
    try testing.expect(bv.pdf(v) > 0.0);
}

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

test "rand: scipy-style cdf/sf are complementary and ppf inverts cdf" {
    const d = rand.Gaussian{ .sigma = 2.0 };
    inline for (.{ -3.1, -0.4, 0.0, 1.7, 5.0 }) |x| {
        // cdf + sf == 1 for a continuous family.
        try testing.expectApproxEqAbs(@as(f64, 1.0), d.cdf(x) + d.sf(x), 1e-12);
        // ppf is the inverse of cdf.
        try testing.expectApproxEqAbs(@as(f64, x), d.ppf(d.cdf(x)), 1e-9);
        // isf is the inverse of sf.
        try testing.expectApproxEqAbs(@as(f64, x), d.isf(d.sf(x)), 1e-9);
    }
    // The median of a zero-mean Gaussian is 0.
    try testing.expectApproxEqAbs(@as(f64, 0.0), d.ppf(0.5), 1e-12);
}

test "rand: unit gaussian cdf matches known quantiles" {
    const z = rand.UnitGaussian{};
    try testing.expectApproxEqAbs(@as(f64, 0.5), z.cdf(0.0), 1e-12);
    // ~95% of mass lies below 1.6448536269514722.
    try testing.expectApproxEqAbs(@as(f64, 1.6448536269514722), z.ppf(0.95), 1e-9);
}

test "rand: new continuous families sample finitely and integrate via cdf" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(11);

    // A representative spread of the newly added continuous families.
    inline for (.{
        rand.Laplace{ .a = 1.5 },
        rand.Cauchy{ .a = 2.0 },
        rand.Rayleigh{ .sigma = 1.0 },
        rand.Lognormal{ .zeta = 0.0, .sigma = 1.0 },
        rand.TDist{ .nu = 5.0 },
        rand.FDist{ .nu1 = 3.0, .nu2 = 8.0 },
        rand.Logistic{ .a = 1.0 },
        rand.Pareto{ .a = 3.0, .b = 1.0 },
        rand.Weibull{ .a = 1.0, .b = 1.5 },
        rand.Gumbel1{ .a = 1.0, .b = 1.0 },
    }) |d| {
        const x = d.sample(r);
        try testing.expect(std.math.isFinite(x));
        try testing.expect(d.pdf(x) >= 0.0);
        // cdf/sf partition the probability mass.
        try testing.expectApproxEqAbs(@as(f64, 1.0), d.cdf(x) + d.sf(x), 1e-9);
    }
}

test "rand: sampling-only families (levy, landau) stay finite" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(5);

    const l = rand.Levy{ .c = 1.0, .alpha = 1.5 };
    const ls = rand.LevySkew{ .c = 1.0, .alpha = 1.5, .beta = 0.5 };
    const lan = rand.Landau{};
    try testing.expect(std.math.isFinite(l.sample(r)));
    try testing.expect(std.math.isFinite(ls.sample(r)));
    try testing.expect(std.math.isFinite(lan.sample(r)));
    // Levy families are intentionally sampling-only.
    comptime std.debug.assert(!@hasDecl(rand.Levy, "pdf"));
}

test "rand: new discrete families sample in support with valid pmf" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(21);

    const geo = rand.Geometric{ .p = 0.25 };
    const k = geo.sample(r);
    try testing.expect(k >= 1); // support starts at 1
    try testing.expect(geo.pdf(k) > 0.0);
    // cdf includes k, so it is at least the pmf at k; both tails are valid
    // probabilities and the cdf saturates to 1 far into the tail.
    try testing.expect(geo.cdf(k) >= geo.pdf(k) - 1e-12);
    try testing.expect(geo.sf(k) >= 0.0 and geo.sf(k) <= 1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), geo.cdf(1000), 1e-9);

    const nb = rand.NegativeBinomial{ .p = 0.4, .n = 3.5 };
    try testing.expect(nb.pdf(nb.sample(r)) > 0.0);

    const hg = rand.Hypergeometric{ .n1 = 5, .n2 = 7, .t = 4 };
    const h = hg.sample(r);
    try testing.expect(h <= 4 and hg.pdf(h) > 0.0);

    const logd = rand.Logarithmic{ .p = 0.6 };
    try testing.expect(logd.sample(r) >= 1);
}

test "rand: general discrete honors its weights" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(2024);

    // Category 2 carries 60% of the mass.
    var g = try rand.General.init(&.{ 0.1, 0.3, 0.6 });
    defer g.deinit();

    try testing.expectApproxEqAbs(@as(f64, 0.6), g.pdf(2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.1), g.pdf(0), 1e-12);

    var counts = [_]usize{ 0, 0, 0 };
    const n = 100_000;
    for (0..n) |_| counts[g.sample(r)] += 1;
    const frac2 = @as(f64, @floatFromInt(counts[2])) / n;
    try testing.expectApproxEqAbs(@as(f64, 0.6), frac2, 0.01);
}

test "rand: random directions are unit vectors" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(7);

    const v2 = rand.dir2d(r);
    try testing.expectApproxEqAbs(@as(f64, 1.0), v2[0] * v2[0] + v2[1] * v2[1], 1e-12);

    const v3 = rand.dir3d(r);
    try testing.expectApproxEqAbs(@as(f64, 1.0), v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2], 1e-12);

    var v5: [5]f64 = undefined;
    rand.dirNd(r, &v5);
    var norm2: f64 = 0;
    for (v5) |c_| norm2 += c_ * c_;
    try testing.expectApproxEqAbs(@as(f64, 1.0), norm2, 1e-12);
}

test "rand: shuffle permutes, choose/sample draw subsets" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(3);

    // Shuffle preserves the multiset of elements.
    var deck = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    r.shuffle(u32, &deck);
    var sum: u32 = 0;
    for (deck) |x| sum += x;
    try testing.expectEqual(@as(u32, 45), sum);

    // choose: k distinct, order-preserving elements taken from src.
    const src = [_]u32{ 10, 11, 12, 13, 14, 15 };
    var picked: [3]u32 = undefined;
    r.choose(u32, &picked, &src);
    for (0..picked.len) |i| {
        try testing.expect(picked[i] >= 10 and picked[i] <= 15);
        if (i > 0) try testing.expect(picked[i - 1] < picked[i]); // order preserved
    }

    // sample: with replacement, so values just need to come from src.
    var withrep: [8]u32 = undefined;
    r.sample(u32, &withrep, &src);
    for (withrep) |x| try testing.expect(x >= 10 and x <= 15);
}

// ---------------------------------------------------------------------------
// Exhaustive binding-surface coverage
//
// These tests exist to catch binding-layer mistakes (a wrong C symbol name or a
// swapped argument), not to re-verify GSL's numerics. The guiding rule is that
// every wrapped symbol should be *invoked* at least once, with a handful of
// stronger oracle/known-value checks where a plausible-looking wrong answer
// could otherwise slip through.
// ---------------------------------------------------------------------------

/// Exercise whatever slice of the scalar-distribution interface `d` implements,
/// detected at comptime via `Sample`/`@hasDecl`. This lets one loop cover every
/// scalar family regardless of which of sample/pdf/cdf/ppf it carries.
fn exerciseScalarDist(r: rand.Rng, d: anytype) !void {
    const D = @TypeOf(d);
    const S = D.Sample;
    const x = d.sample(r);
    comptime std.debug.assert(@TypeOf(x) == S);

    switch (@typeInfo(S)) {
        .float => try testing.expect(std.math.isFinite(x)),
        else => {},
    }

    if (@hasDecl(D, "pdf")) {
        try testing.expect(d.pdf(x) >= 0.0);
        if (S == bool) {
            // A Bernoulli-style density must sum to 1 over its two outcomes.
            try testing.expectApproxEqAbs(@as(f64, 1.0), d.pdf(true) + d.pdf(false), 1e-12);
        }
    }

    if (@hasDecl(D, "cdf")) {
        const lo = d.cdf(x);
        const hi = d.sf(x);
        try testing.expect(lo >= -1e-9 and lo <= 1.0 + 1e-9);
        try testing.expect(hi >= -1e-9 and hi <= 1.0 + 1e-9);
        // Lower and upper tails partition the probability mass.
        try testing.expectApproxEqAbs(@as(f64, 1.0), lo + hi, 1e-6);
    }

    if (@hasDecl(D, "ppf")) {
        try testing.expect(std.math.isFinite(d.ppf(0.5)));
        try testing.expect(std.math.isFinite(d.isf(0.5)));
        // ppf inverts cdf at an interior probability (catches a Pinv/Qinv swap).
        const pr: f64 = 0.37;
        try testing.expectApproxEqAbs(pr, d.cdf(d.ppf(pr)), 1e-6);
    }
}

test "distribution: every scalar family exercises its full method set" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(20240607);

    // Continuous families carrying the full CDF set.
    try exerciseScalarDist(r, rand.Gaussian{ .sigma = 1.5 });
    try exerciseScalarDist(r, rand.Exponential{ .mu = 2.0 });
    try exerciseScalarDist(r, rand.Flat{ .a = -1.0, .b = 2.0 });
    try exerciseScalarDist(r, rand.Gamma{ .a = 2.0, .b = 1.5 });
    try exerciseScalarDist(r, rand.Beta{ .a = 2.0, .b = 3.0 });
    try exerciseScalarDist(r, rand.ChiSquared{ .nu = 4.0 });
    try exerciseScalarDist(r, rand.UnitGaussian{});
    try exerciseScalarDist(r, rand.Laplace{ .a = 1.0 });
    try exerciseScalarDist(r, rand.Cauchy{ .a = 1.0 });
    try exerciseScalarDist(r, rand.Rayleigh{ .sigma = 1.0 });
    try exerciseScalarDist(r, rand.Lognormal{ .zeta = 0.0, .sigma = 1.0 });
    try exerciseScalarDist(r, rand.FDist{ .nu1 = 3.0, .nu2 = 8.0 });
    try exerciseScalarDist(r, rand.TDist{ .nu = 5.0 });
    try exerciseScalarDist(r, rand.Logistic{ .a = 1.0 });
    try exerciseScalarDist(r, rand.Pareto{ .a = 3.0, .b = 1.0 });
    try exerciseScalarDist(r, rand.Weibull{ .a = 1.0, .b = 1.5 });
    try exerciseScalarDist(r, rand.Gumbel1{ .a = 1.0, .b = 1.0 });
    try exerciseScalarDist(r, rand.Gumbel2{ .a = 1.0, .b = 1.0 });

    // Continuous families with cdf/sf but no inverse (ExpPower).
    try exerciseScalarDist(r, rand.ExpPower{ .a = 1.0, .b = 2.5 });

    // Continuous families with sampling + density only.
    try exerciseScalarDist(r, rand.GaussianTail{ .a = 1.0, .sigma = 1.0 });
    try exerciseScalarDist(r, rand.UnitGaussianTail{ .a = 1.0 });
    try exerciseScalarDist(r, rand.RayleighTail{ .a = 0.5, .sigma = 1.0 });
    try exerciseScalarDist(r, rand.Landau{});
    try exerciseScalarDist(r, rand.Erlang{ .a = 1.0, .n = 3.0 });

    // Sampling-only families (no density).
    try exerciseScalarDist(r, rand.Levy{ .c = 1.0, .alpha = 1.5 });
    try exerciseScalarDist(r, rand.LevySkew{ .c = 1.0, .alpha = 1.5, .beta = 0.5 });

    // Discrete families.
    try exerciseScalarDist(r, rand.Poisson{ .mu = 3.0 });
    try exerciseScalarDist(r, rand.Bernoulli{ .p = 0.3 });
    try exerciseScalarDist(r, rand.Binomial{ .p = 0.5, .n = 10 });
    try exerciseScalarDist(r, rand.Geometric{ .p = 0.25 });
    try exerciseScalarDist(r, rand.NegativeBinomial{ .p = 0.4, .n = 3.5 });
    try exerciseScalarDist(r, rand.Pascal{ .p = 0.4, .n = 3 });
    try exerciseScalarDist(r, rand.Hypergeometric{ .n1 = 5, .n2 = 7, .t = 4 });
    try exerciseScalarDist(r, rand.Logarithmic{ .p = 0.6 });
}

test "distribution: representative samplers equal the raw GSL call under the same seed" {
    const seed = 0xC0FFEE;

    // A one-parameter continuous sampler.
    {
        var a = try rand.Rng.init(.mt19937);
        defer a.deinit();
        var b = try rand.Rng.init(.mt19937);
        defer b.deinit();
        a.seed(seed);
        b.seed(seed);
        const d = rand.Exponential{ .mu = 2.5 };
        try testing.expectEqual(c.gsl_ran_exponential(a.ptr, 2.5), d.sample(b));
    }
    // A two-parameter continuous sampler.
    {
        var a = try rand.Rng.init(.mt19937);
        defer a.deinit();
        var b = try rand.Rng.init(.mt19937);
        defer b.deinit();
        a.seed(seed);
        b.seed(seed);
        const d = rand.Gamma{ .a = 2.0, .b = 1.5 };
        try testing.expectEqual(c.gsl_ran_gamma(a.ptr, 2.0, 1.5), d.sample(b));
    }
    // A discrete sampler (also checks the u32 narrowing).
    {
        var a = try rand.Rng.init(.mt19937);
        defer a.deinit();
        var b = try rand.Rng.init(.mt19937);
        defer b.deinit();
        a.seed(seed);
        b.seed(seed);
        const d = rand.Poisson{ .mu = 4.0 };
        try testing.expectEqual(@as(u32, @intCast(c.gsl_ran_poisson(a.ptr, 4.0))), d.sample(b));
    }
}

test "distribution: cdf/pdf/ppf hit known closed-form values" {
    // Exponential(mu=2): cdf(x) = 1 - e^{-x/mu}, median = mu*ln 2, pdf = (1/mu)e^{-x/mu}.
    {
        const d = rand.Exponential{ .mu = 2.0 };
        try testing.expectApproxEqAbs(@as(f64, 1.0 - @exp(-1.0)), d.cdf(2.0), 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 2.0 * @log(2.0)), d.ppf(0.5), 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0.5 * @exp(-0.5)), d.pdf(1.0), 1e-12);
    }
    // Flat(0,4) is uniform: cdf(1)=1/4, pdf=1/4, median=2.
    {
        const d = rand.Flat{ .a = 0.0, .b = 4.0 };
        try testing.expectApproxEqAbs(@as(f64, 0.25), d.cdf(1.0), 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0.25), d.pdf(1.0), 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 2.0), d.ppf(0.5), 1e-12);
    }
    // Beta(1,1) is uniform on [0,1].
    {
        const d = rand.Beta{ .a = 1.0, .b = 1.0 };
        try testing.expectApproxEqAbs(@as(f64, 0.3), d.cdf(0.3), 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 1.0), d.pdf(0.5), 1e-12);
    }
    // Poisson(mu=3): pmf(0) = e^{-3}, and cdf(0) = pmf(0).
    {
        const d = rand.Poisson{ .mu = 3.0 };
        try testing.expectApproxEqAbs(@as(f64, @exp(-3.0)), d.pdf(0), 1e-12);
        try testing.expectApproxEqAbs(@as(f64, @exp(-3.0)), d.cdf(0), 1e-12);
    }
    // Binomial(n=10, p=0.5): pmf(5) = C(10,5)/2^10 = 252/1024.
    {
        const d = rand.Binomial{ .p = 0.5, .n = 10 };
        try testing.expectApproxEqAbs(@as(f64, 252.0 / 1024.0), d.pdf(5), 1e-9);
    }
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

test "rand: every generator algorithm allocates, names itself, and advances" {
    // The enum tag names are chosen to match GSL's own algorithm names, so a
    // single loop over every variant doubles as a check that `typePtr` maps each
    // one to the intended `gsl_rng_*` type.
    inline for (comptime std.enums.values(rand.Generator)) |gen| {
        var g = try rand.Rng.init(gen);
        defer g.deinit();
        try testing.expectEqualStrings(@tagName(gen), g.name());
        try testing.expect(g.maxValue() > g.minValue());
        _ = g.next();
    }
}

test "rand: initByName reaches curated and legacy generators, rejects unknown names" {
    // A curated algorithm also reachable via the enum.
    var m = try rand.Rng.initByName("mt19937");
    defer m.deinit();
    try testing.expectEqualStrings("mt19937", m.name());

    // A legacy/compatibility generator deliberately *not* in `Generator`
    // (the infamously flawed IBM RANDU) is still reachable by name.
    var legacy = try rand.Rng.initByName("randu");
    defer legacy.deinit();
    try testing.expectEqualStrings("randu", legacy.name());
    legacy.seed(1);
    _ = legacy.next();

    // An unrecognized name is reported, not silently defaulted or aborted.
    try testing.expectError(error.UnknownGenerator, rand.Rng.initByName("definitely_not_a_generator"));
}

test "rand: state save/load round-trips a stream across a checkpoint" {
    var r = try rand.Rng.init(.mt19937);
    defer r.deinit();
    r.seed(20240521);

    // Advance past the initial state so the snapshot captures mid-stream state.
    for (0..37) |_| _ = r.next();

    // Snapshot into a caller buffer; the used slice is exactly `stateSize()`.
    var buf: [8192]u8 = undefined;
    try testing.expect(r.stateSize() <= buf.len);
    const snapshot = r.saveState(&buf);
    try testing.expectEqual(r.stateSize(), snapshot.len);

    // Record the next draws, then rewind by loading the snapshot back.
    var expected: [16]u64 = undefined;
    for (&expected) |*e| e.* = r.next();

    r.loadState(snapshot);
    for (expected) |e| try testing.expectEqual(e, r.next());

    // A snapshot also restores into a *different* handle of the same algorithm,
    // which is the cross-process checkpoint/restore use case.
    var fresh = try rand.Rng.init(.mt19937);
    defer fresh.deinit();
    fresh.seed(1); // deliberately different starting point
    fresh.loadState(snapshot);
    for (expected) |e| try testing.expectEqual(e, fresh.next());
}

test "gsl: error helpers report success and install the non-aborting handler" {
    // Code 0 is GSL_SUCCESS.
    try testing.expectEqualStrings("success", strerror(0));
    // A nonzero code yields a non-empty diagnostic string.
    try testing.expect(strerror(4).len > 0);
    // Smoke-test the setup call (it only swaps GSL's global error handler).
    disableDefaultErrorHandler();
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

    // The quasi-random namespace is a reserved placeholder: referencing it at
    // all triggers its `@compileError`, directing callers to the raw C API.
    if (false) {
        _ = qrng;
    }
}
