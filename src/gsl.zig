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
//!               `stats` is the `f64` specialization; use `Stats(T)` for other
//!               element types.
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

pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_rng.h");
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
///   - Generator zoo: only the five `Generator` algorithms below are exposed,
///     out of the ~60 GSL ships. Use a raw `c.gsl_rng_*` type pointer with
///     `c.gsl_rng_alloc` if you need one that isn't listed.
///   - Generator plumbing: RNG state serialization (`gsl_rng_fwrite`/`fread`),
///     raw state access (`gsl_rng_state`/`gsl_rng_size`), and the whole
///     quasi-random generator family (`gsl_qrng_*`: Sobol, Halton, ...).
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

    /// A selection of GSL's built-in generator algorithms. GSL ships many more
    /// (see `gsl_rng.h`); these are the commonly used, well-tested ones. Reach
    /// for the raw `c.gsl_rng_*` type pointers if you need one not listed.
    pub const Generator = enum {
        /// Mersenne Twister. GSL's default; a good general-purpose choice.
        mt19937,
        /// Tausworthe generator (maximally equidistributed, very fast).
        taus2,
        /// RANLUX, second-generation double-precision, highest quality/slowest.
        ranlxd2,
        /// Combined multiple recursive generator (long period).
        cmrg,
        /// Lagged-Fibonacci, four-tap; very fast.
        gfsr4,

        fn typePtr(self: Generator) [*c]const c.gsl_rng_type {
            return switch (self) {
                .mt19937 => c.gsl_rng_mt19937,
                .taus2 => c.gsl_rng_taus2,
                .ranlxd2 => c.gsl_rng_ranlxd2,
                .cmrg => c.gsl_rng_cmrg,
                .gfsr4 => c.gsl_rng_gfsr4,
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

/// A strided, read-only view over `T`: `len` elements spaced `stride` apart
/// starting at `ptr`. GSL's statistics routines operate on exactly this shape,
/// so a column/row/axis of a larger array can be passed without copying. Use
/// `fromSlice` for the common contiguous (stride-1) case.
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
            else => @compileError("gsl.Stats: unsupported float element type '" ++ @typeName(T) ++ "'; only f32 and f64 are supported (GSL's long double module is intentionally omitted)"),
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
            @compileError(std.fmt.comptimePrint("gsl.Stats: no GSL statistics module matches element type '{s}' ({d}-bit {s}); GSL provides only char/short/int/long-sized modules", .{ @typeName(T), @bitSizeOf(T), @tagName(info.signedness) }));
        },
        else => @compileError("gsl.Stats: unsupported element type '" ++ @typeName(T) ++ "'; expected a Zig integer or f32/f64"),
    }
}

/// Descriptive statistics over `Strided(T)` views, wrapping GSL's per-element-
/// type `gsl_stats_*` modules. `Stats(T)` selects the module for `T`; the
/// ready-made `stats` namespace below is the `f64` specialization.
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
/// `Stats(T).weighted` is a compile error for integer `T`.
///
/// ## Omitted from GSL
///
///   - The `long double` module (`gsl_stats_long_double_*`) is not wrapped:
///     Zig has no portable fixed-width type matching C `long double`'s ABI.
///     Use `f64`, or the raw `c.gsl_stats_long_double_*` symbols, if you need
///     it.
///   - Signed 8-bit data is unavailable on the rare targets where C `char` is
///     unsigned (e.g. some AArch64 platforms): `Stats(i8)` is a compile error
///     there rather than risk misreading values. `u8` is always available
///     through the `uchar` module.
pub fn Stats(comptime T: type) type {
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
        } else @compileError("gsl.Stats: weighted statistics are only available for floating-point element types (f32, f64); GSL provides no weighted routines for integer types");
    };
}

/// Descriptive statistics over `Strided(f64)` views: the ready-made `f64`
/// specialization of `Stats`. See `Stats` for the full contract and for how to
/// obtain the same API over other element types.
pub const stats = Stats(f64);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

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
    try testing.expectApproxEqAbs(@as(f64, 0.0), stats.mean(.fromSlice(&buf)), 0.05);
    try testing.expectApproxEqAbs(@as(f64, 3.0), stats.sd(.fromSlice(&buf)), 0.05);
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
    try testing.expectApproxEqAbs(@as(f64, 5.0), stats.mean(v), 1e-12);
    // Sample variance (n-1) of this classic dataset is 32/7.
    try testing.expectApproxEqAbs(@as(f64, 32.0 / 7.0), stats.variance(v), 1e-12);
    try testing.expectEqual(@as(f64, 9.0), stats.max(v));
    try testing.expectEqual(@as(f64, 2.0), stats.min(v));
    const mm = stats.minMax(v);
    try testing.expectEqual(@as(f64, 2.0), mm.min);
    try testing.expectEqual(@as(f64, 9.0), mm.max);
    // Data is already sorted, so we can take the median directly.
    try testing.expectApproxEqAbs(@as(f64, 4.5), stats.medianFromSorted(v), 1e-12);
}

test "stats: strided view selects every other element" {
    // Even indices hold the dataset; odd indices are noise to be skipped.
    const interleaved = [_]f64{ 2, -9, 4, -9, 4, -9, 4, -9, 5, -9, 5, -9, 7, -9, 9, -9 };
    const v = Strided(f64).init(&interleaved, 2, 8);
    try testing.expectApproxEqAbs(@as(f64, 5.0), stats.mean(v), 1e-12);
    try testing.expectEqual(@as(f64, 9.0), stats.max(v));
}

test "stats: select and median rearrange in place" {
    var data = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    // 0-based 3rd smallest is 3.
    try testing.expectEqual(@as(f64, 3.0), stats.select(.fromSlice(&data), 2));

    var data2 = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    try testing.expectEqual(@as(f64, 3.0), stats.median(.fromSlice(&data2)));
}

test "stats: weighted mean matches unweighted for equal weights" {
    const x = [_]f64{ 1, 2, 3, 4 };
    const w = [_]f64{ 1, 1, 1, 1 };
    try testing.expectApproxEqAbs(
        stats.mean(.fromSlice(&x)),
        stats.weighted.mean(.fromSlice(&w), .fromSlice(&x)),
        1e-12,
    );
    // All weight on the last point pulls the mean to it.
    const w2 = [_]f64{ 0, 0, 0, 1 };
    try testing.expectApproxEqAbs(@as(f64, 4.0), stats.weighted.mean(.fromSlice(&w2), .fromSlice(&x)), 1e-12);
}

test "stats: robust scale estimators with explicit work buffers" {
    // Already-sorted data with a single gross outlier.
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 };
    const v = Strided(f64).fromSlice(&data);
    const n = v.len;

    var work: [40]f64 = undefined; // >= max(madWorkLen, snWorkLen, qnWorkLen) = 3n
    var work_int: [50]c_int = undefined; // >= qnWorkIntLen = 5n

    const m = stats.mad(v, work[0..stats.madWorkLen(n)]);
    const s = stats.snFromSorted(v, work[0..stats.snWorkLen(n)]);
    const q = stats.qnFromSorted(v, work[0..stats.qnWorkLen(n)], work_int[0..stats.qnWorkIntLen(n)]);

    // Robust scales ignore the outlier, staying far below the (inflated) sd.
    try testing.expect(m > 0 and s > 0 and q > 0);
    try testing.expect(m < stats.sd(v));
    try testing.expect(s < stats.sd(v));
    try testing.expect(q < stats.sd(v));
}

test "stats: correlation of a perfectly linear relationship is 1" {
    const x = [_]f64{ 1, 2, 3, 4, 5 };
    const y = [_]f64{ 3, 5, 7, 9, 11 }; // y = 2x + 1
    try testing.expectApproxEqAbs(@as(f64, 1.0), stats.correlation(.fromSlice(&x), .fromSlice(&y)), 1e-12);
}

test "stats: integer element type mirrors the f64 module's results" {
    const S = Stats(i32);
    const idata = [_]i32{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const iv = Strided(i32).fromSlice(&idata);

    // Moments are accumulated in double precision, so they match the f64 module.
    const fdata = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const fv = Strided(f64).fromSlice(&fdata);
    try testing.expectApproxEqAbs(stats.mean(fv), S.mean(iv), 1e-12);
    try testing.expectApproxEqAbs(stats.variance(fv), S.variance(iv), 1e-12);

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
    const S = Stats(u16);
    var data = [_]u16{ 5, 3, 1, 4, 2 };
    const third: u16 = S.select(StridedMut(u16).fromSlice(&data), 2);
    try testing.expectEqual(@as(u16, 3), third);
}

test "stats: f32 element type, including weighted statistics" {
    const S = Stats(f32);
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
    const S = Stats(i32);
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
