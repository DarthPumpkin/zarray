//! Idiomatic Zig bindings for the GNU Scientific Library's polynomial module
//! (`gsl_poly`).
//!
//! This file *extends* the GSL bindings in `gsl.zig` with the polynomial
//! chapter. It reuses that module's process-global error-handler switch, but
//! keeps the polynomial-specific C API behind its own `c` (the `gsl_poly`
//! header is not pulled in by `gsl.zig`). It is reached as `gsl.poly`.
//!
//! ## Shape of the surface
//!
//! A polynomial is passed as a slice of coefficients in *ascending* order:
//! `coeffs[0] + coeffs[1]·x + coeffs[2]·x² + ...`.
//!
//!   - `eval`/`evalComplex`/`evalComplexPoly` — Horner evaluation of a real
//!     polynomial at a real or complex point, or a complex polynomial at a
//!     complex point.
//!   - `evalDerivs` — the value and successive derivatives at a point.
//!   - `solveQuadratic`/`solveCubic` — the **real** roots of a quadratic or
//!     (monic) cubic, returned as `{ n, roots }` (only `roots[0..n]` are valid).
//!   - `solveQuadraticComplex`/`solveCubicComplex` — all roots as
//!     `std.math.Complex(f64)`.
//!   - `ComplexSolver` — all complex roots of a general real polynomial of any
//!     degree (companion-matrix method), owning a reusable workspace.
//!   - `ddInit`/`ddEval`/`ddTaylor`/`ddHermiteInit` — Newton divided-difference
//!     interpolation through a set of points (and the Hermite variant that
//!     matches derivatives too). The caller supplies the coefficient/scratch
//!     buffers, keeping this module allocation-free.
//!
//! Complex values use `std.math.Complex(f64)`, whose in-memory layout matches
//! GSL's `gsl_complex`, so slices cast straight through with no copy.
//!
//! ## Preconditions & omissions
//!
//!   - `eval`, `evalComplex`, `evalComplexPoly`, and `ddEval` require a
//!     **non-empty** coefficient slice: GSL's Horner recurrence reads
//!     `c[len-1]`, so an empty slice is a programmer error (asserted in safe
//!     builds), not a runtime condition.
//!   - The full `gsl_poly.h` surface is otherwise wrapped. Only the plain
//!     (non-`_e`, non-workspace) helpers are reached through the raw `c` API —
//!     e.g. `c.gsl_poly_eval` if you want the bare `f64` form without the
//!     non-empty assertion.
//!   - `evalComplex`/`evalComplexPoly` are native Zig Horner loops rather than
//!     calls to `gsl_poly_complex_eval`/`gsl_complex_poly_complex_eval`, which
//!     pass and return `gsl_complex` *by value* — an ABI that Zig's `@cImport`
//!     does not reliably reproduce. The results are identical; the raw C symbols
//!     remain reachable via `c` if you accept that caveat.

const std = @import("std");
const testing = std.testing;
const Complex = std.math.Complex;
const gsl = @import("gsl.zig");

/// The raw C API. Use it directly for the divided-difference forms and any
/// symbol not wrapped here.
pub const c = @cImport({
    @cInclude("gsl/gsl_errno.h");
    @cInclude("gsl/gsl_complex.h");
    @cInclude("gsl/gsl_poly.h");
});

/// Toggle GSL's process-global error handler (shared with the rest of the GSL
/// bindings). Re-exported from `gsl.zig`; installed automatically on first use.
pub const disableDefaultErrorHandler = gsl.disableDefaultErrorHandler;
/// Human-readable message for a GSL status code. Re-exported from `gsl.zig`.
pub const strerror = gsl.strerror;

/// Zig error set for the polynomial routines. The raw `c_int` status is always
/// available from the underlying `c.gsl_poly_*` symbol if you need the exact
/// code.
pub const Error = error{
    /// `GSL_EINVAL` — an invalid argument, e.g. a general polynomial whose
    /// leading coefficient is zero.
    Invalid,
    /// A caller-supplied slice of the wrong length (`roots.len + 1 != coeffs.len`)
    /// or too few coefficients to solve.
    BadLength,
    /// `GSL_ENOMEM` — allocation failed.
    OutOfMemory,
    /// `GSL_EFAILED` — the QR iteration for the general root finder did not
    /// converge.
    Failure,
    /// Any other nonzero GSL status code.
    Unspecified,
};

fn check(status: c_int) Error!void {
    return switch (status) {
        c.GSL_SUCCESS => {},
        c.GSL_EINVAL => Error.Invalid,
        c.GSL_EBADLEN => Error.BadLength,
        c.GSL_ENOMEM => Error.OutOfMemory,
        c.GSL_EFAILED => Error.Failure,
        else => Error.Unspecified,
    };
}

inline fn toComplex(z: c.gsl_complex) Complex(f64) {
    return .{ .re = z.dat[0], .im = z.dat[1] };
}

/// The real roots of a low-degree polynomial: only `roots[0..n]` are
/// meaningful, and they are returned in ascending order.
pub const RealRoots2 = struct { n: usize, roots: [2]f64 };
/// The real roots of a cubic: only `roots[0..n]` are meaningful (ascending).
pub const RealRoots3 = struct { n: usize, roots: [3]f64 };

// --- evaluation ---

/// Horner evaluation of the real polynomial `coeffs[0] + coeffs[1]·x + ...` at
/// the real point `x`. `coeffs` must be non-empty.
pub fn eval(coeffs: []const f64, x: f64) f64 {
    std.debug.assert(coeffs.len > 0);
    return c.gsl_poly_eval(coeffs.ptr, @intCast(coeffs.len), x);
}

/// Evaluate the real polynomial `coeffs` at the complex point `z`. Implemented
/// as a native Horner recurrence (equivalent to `gsl_poly_complex_eval`, but
/// without relying on GSL's by-value `gsl_complex` ABI).
pub fn evalComplex(coeffs: []const f64, z: Complex(f64)) Complex(f64) {
    std.debug.assert(coeffs.len > 0);
    var ans = Complex(f64){ .re = coeffs[coeffs.len - 1], .im = 0 };
    var i = coeffs.len - 1;
    while (i > 0) : (i -= 1) {
        ans = ans.mul(z).add(.{ .re = coeffs[i - 1], .im = 0 });
    }
    return ans;
}

/// Evaluate the complex polynomial `coeffs` at the complex point `z`. Native
/// Horner recurrence (equivalent to `gsl_complex_poly_complex_eval`).
pub fn evalComplexPoly(coeffs: []const Complex(f64), z: Complex(f64)) Complex(f64) {
    std.debug.assert(coeffs.len > 0);
    var ans = coeffs[coeffs.len - 1];
    var i = coeffs.len - 1;
    while (i > 0) : (i -= 1) {
        ans = ans.mul(z).add(coeffs[i - 1]);
    }
    return ans;
}

/// Evaluate the polynomial `coeffs` and its successive derivatives at `x`,
/// writing `out.len` results: `out[0]` is the value, `out[1]` the first
/// derivative, and so on (derivatives past the polynomial's degree are 0).
/// `coeffs` must be non-empty.
pub fn evalDerivs(coeffs: []const f64, x: f64, out: []f64) Error!void {
    if (coeffs.len == 0) return Error.BadLength;
    gsl.ensureHandler();
    try check(c.gsl_poly_eval_derivs(coeffs.ptr, coeffs.len, x, out.ptr, out.len));
}

// --- Newton divided differences ---
//
// Newton's divided-difference form of the polynomial interpolating the points
// `(xs[i], ys[i])`. Unlike the rest of this module these routines need scratch
// storage; to keep the module allocation-free (no injected allocator, matching
// its pure-function style) the caller supplies the output/work buffers.

/// Compute the divided-difference coefficients of the polynomial interpolating
/// `(xs[i], ys[i])` into `dd`. All three slices must share the same length `n`
/// (`n >= 1`). Evaluate the result with `ddEval` using the same `xs`.
pub fn ddInit(dd: []f64, xs: []const f64, ys: []const f64) Error!void {
    if (dd.len == 0 or dd.len != xs.len or xs.len != ys.len) return Error.BadLength;
    gsl.ensureHandler();
    try check(c.gsl_poly_dd_init(dd.ptr, xs.ptr, ys.ptr, dd.len));
}

/// Evaluate a divided-difference polynomial (from `ddInit`) at `x`. `dd` and
/// `xs` must be the same length passed to `ddInit` (and non-empty).
pub fn ddEval(dd: []const f64, xs: []const f64, x: f64) f64 {
    std.debug.assert(dd.len == xs.len and dd.len > 0);
    return c.gsl_poly_dd_eval(dd.ptr, xs.ptr, dd.len, x);
}

/// Convert a divided-difference polynomial (from `ddInit`) into ordinary Taylor
/// coefficients about the point `xp`, written to `out` in ascending powers of
/// `(x - xp)`. `out`, `dd`, and `xs` share length `n`; `work` is scratch of
/// length at least `n`.
pub fn ddTaylor(out: []f64, xp: f64, dd: []const f64, xs: []const f64, work: []f64) Error!void {
    const n = dd.len;
    if (n == 0 or out.len != n or xs.len != n or work.len < n) return Error.BadLength;
    gsl.ensureHandler();
    try check(c.gsl_poly_dd_taylor(out.ptr, xp, dd.ptr, xs.ptr, n, work.ptr));
}

/// Compute the divided-difference coefficients of the Hermite interpolant that
/// matches both the values `ys` and the derivatives `dys` at each `xs[i]`.
/// `xs`/`ys`/`dys` share length `n` (`n >= 1`); the outputs `dd` and `za` must
/// each have length `2*n`. Evaluate the result with `ddEval` using `za` (not
/// `xs`) as the abscissae.
pub fn ddHermiteInit(
    dd: []f64,
    za: []f64,
    xs: []const f64,
    ys: []const f64,
    dys: []const f64,
) Error!void {
    const n = xs.len;
    if (n == 0 or ys.len != n or dys.len != n) return Error.BadLength;
    if (dd.len != 2 * n or za.len != 2 * n) return Error.BadLength;
    gsl.ensureHandler();
    try check(c.gsl_poly_dd_hermite_init(dd.ptr, za.ptr, xs.ptr, ys.ptr, dys.ptr, n));
}

// --- low-degree closed-form roots ---
//
// Coefficients are named by the power they multiply: `c2·x² + c1·x + c0` for the
// quadratic, and the *monic* cubic `x³ + c2·x² + c1·x + c0` (GSL's cubic solver
// assumes a leading coefficient of 1).

/// Real roots of the quadratic `c2·x² + c1·x + c0 = 0`. `n` is the number of
/// distinct real roots (0, 1, or 2), returned ascending in `roots[0..n]`.
pub fn solveQuadratic(c2: f64, c1: f64, c0: f64) RealRoots2 {
    var r: [2]f64 = undefined;
    const n = c.gsl_poly_solve_quadratic(c2, c1, c0, &r[0], &r[1]);
    return .{ .n = @intCast(n), .roots = r };
}

/// Both roots of the quadratic `c2·x² + c1·x + c0 = 0` as complex numbers
/// (always well-defined, including the complex-conjugate case).
pub fn solveQuadraticComplex(c2: f64, c1: f64, c0: f64) [2]Complex(f64) {
    var z0: c.gsl_complex = undefined;
    var z1: c.gsl_complex = undefined;
    _ = c.gsl_poly_complex_solve_quadratic(c2, c1, c0, &z0, &z1);
    return .{ toComplex(z0), toComplex(z1) };
}

/// Real roots of the monic cubic `x³ + c2·x² + c1·x + c0 = 0`. `n` is the
/// number of distinct real roots (1 or 3), returned ascending in `roots[0..n]`.
pub fn solveCubic(c2: f64, c1: f64, c0: f64) RealRoots3 {
    var r: [3]f64 = undefined;
    const n = c.gsl_poly_solve_cubic(c2, c1, c0, &r[0], &r[1], &r[2]);
    return .{ .n = @intCast(n), .roots = r };
}

/// All three roots of the monic cubic `x³ + c2·x² + c1·x + c0 = 0` as complex
/// numbers.
pub fn solveCubicComplex(c2: f64, c1: f64, c0: f64) [3]Complex(f64) {
    var z0: c.gsl_complex = undefined;
    var z1: c.gsl_complex = undefined;
    var z2: c.gsl_complex = undefined;
    _ = c.gsl_poly_complex_solve_cubic(c2, c1, c0, &z0, &z1, &z2);
    return .{ toComplex(z0), toComplex(z1), toComplex(z2) };
}

// --- general-degree root finding ---

/// A reusable workspace for finding all complex roots of a general real
/// polynomial via its companion matrix (`gsl_poly_complex_workspace`). Owns its
/// GSL allocation; call `deinit` to free.
///
/// Example:
/// ```
/// // x^3 - 1 = 0  ->  coeffs (ascending) { -1, 0, 0, 1 }
/// const coeffs = [_]f64{ -1, 0, 0, 1 };
/// var solver = try gsl.poly.ComplexSolver.init(coeffs.len);
/// defer solver.deinit();
/// var roots: [3]std.math.Complex(f64) = undefined;
/// try solver.solve(&coeffs, &roots);
/// ```
pub const ComplexSolver = struct {
    ptr: *c.gsl_poly_complex_workspace,

    /// Allocate a workspace sized for a polynomial with `num_coeffs`
    /// coefficients (degree `num_coeffs - 1`). Requires `num_coeffs >= 2`.
    pub fn init(num_coeffs: usize) Error!ComplexSolver {
        if (num_coeffs < 2) return Error.BadLength;
        gsl.ensureHandler();
        const p = c.gsl_poly_complex_workspace_alloc(num_coeffs) orelse return Error.OutOfMemory;
        return .{ .ptr = p };
    }

    pub fn deinit(self: ComplexSolver) void {
        c.gsl_poly_complex_workspace_free(self.ptr);
    }

    /// Find all `coeffs.len - 1` complex roots of the polynomial whose
    /// ascending coefficients are `coeffs`. `roots.len` must equal
    /// `coeffs.len - 1`, and the leading coefficient `coeffs[len-1]` must be
    /// non-zero (else `Error.Invalid`). The workspace must have been sized for
    /// `coeffs.len` (see `init`).
    pub fn solve(self: ComplexSolver, coeffs: []const f64, roots: []Complex(f64)) Error!void {
        if (coeffs.len < 2 or roots.len + 1 != coeffs.len) return Error.BadLength;
        const zp: [*c]f64 = @ptrCast(roots.ptr);
        try check(c.gsl_poly_complex_solve(coeffs.ptr, coeffs.len, self.ptr, zp));
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "poly: Horner evaluation and derivatives" {
    // p(x) = 1 + 2x + 3x^2 ; p'(x) = 2 + 6x ; p''(x) = 6.
    const coeffs = [_]f64{ 1, 2, 3 };
    try testing.expectApproxEqAbs(@as(f64, 17.0), eval(&coeffs, 2.0), 1e-12); // 1+4+12

    var d: [4]f64 = undefined;
    try evalDerivs(&coeffs, 2.0, &d);
    try testing.expectApproxEqAbs(@as(f64, 17.0), d[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 14.0), d[1], 1e-12); // 2 + 6*2
    try testing.expectApproxEqAbs(@as(f64, 6.0), d[2], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), d[3], 1e-12); // beyond degree
}

test "poly: real quadratic roots, both real and none" {
    // x^2 - 3x + 2 = (x-1)(x-2)
    const q = solveQuadratic(1.0, -3.0, 2.0);
    try testing.expectEqual(@as(usize, 2), q.n);
    try testing.expectApproxEqAbs(@as(f64, 1.0), q.roots[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), q.roots[1], 1e-12);

    // x^2 + 1 has no real roots.
    const none = solveQuadratic(1.0, 0.0, 1.0);
    try testing.expectEqual(@as(usize, 0), none.n);
}

test "poly: complex quadratic roots are the conjugate pair ±i" {
    const z = solveQuadraticComplex(1.0, 0.0, 1.0); // x^2 + 1
    // Roots are +i and -i in some order.
    try testing.expectApproxEqAbs(@as(f64, 0.0), z[0].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), z[1].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), @abs(z[0].im), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), @abs(z[1].im), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), z[0].im + z[1].im, 1e-12); // conjugates
}

test "poly: monic cubic real roots" {
    // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    const cub = solveCubic(-6.0, 11.0, -6.0);
    try testing.expectEqual(@as(usize, 3), cub.n);
    try testing.expectApproxEqAbs(@as(f64, 1.0), cub.roots[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), cub.roots[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), cub.roots[2], 1e-10);
}

test "poly: complex cubic returns three roots" {
    // x^3 - 1 = 0 (monic: c2=0, c1=0, c0=-1). Roots are the cube roots of unity.
    const z = solveCubicComplex(0.0, 0.0, -1.0);
    for (z) |root| {
        // Every cube root of unity has modulus 1.
        try testing.expectApproxEqAbs(@as(f64, 1.0), root.magnitude(), 1e-10);
    }
    // Exactly one root is real (≈ 1).
    var real_count: usize = 0;
    for (z) |root| {
        if (@abs(root.im) < 1e-10) real_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), real_count);
}

test "poly: general complex solver finds all roots of x^3 - 1" {
    const coeffs = [_]f64{ -1, 0, 0, 1 }; // -1 + x^3, ascending
    var solver = try ComplexSolver.init(coeffs.len);
    defer solver.deinit();

    var roots: [3]Complex(f64) = undefined;
    try solver.solve(&coeffs, &roots);

    for (roots) |root| {
        try testing.expectApproxEqAbs(@as(f64, 1.0), root.magnitude(), 1e-10);
        // Each root satisfies z^3 = 1, i.e. the polynomial evaluates to ~0.
        const v = evalComplex(&coeffs, root);
        try testing.expectApproxEqAbs(@as(f64, 0.0), v.magnitude(), 1e-9);
    }
}

test "poly: general solver rejects mismatched output length" {
    const coeffs = [_]f64{ -1, 0, 1 }; // degree 2 -> 2 roots
    var solver = try ComplexSolver.init(coeffs.len);
    defer solver.deinit();
    var too_few: [1]Complex(f64) = undefined;
    try testing.expectError(Error.BadLength, solver.solve(&coeffs, &too_few));
    try testing.expectError(Error.BadLength, ComplexSolver.init(1));
}

test "poly: complex-coefficient evaluation at a complex point" {
    // p(z) = (1 + 0i) + (0 + 1i) z ; at z = i, p = 1 + i*i = 1 - 1 = 0.
    const coeffs = [_]Complex(f64){ .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 1 } };
    const v = evalComplexPoly(&coeffs, .{ .re = 0, .im = 1 });
    try testing.expectApproxEqAbs(@as(f64, 0.0), v.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), v.im, 1e-12);
}

test "poly: real polynomial evaluated at a genuinely complex point" {
    // p(x) = 1 + 2x + 3x^2 at z = 1 + i:
    //   (1+i)^2 = 2i, so p = 1 + 2(1+i) + 3(2i) = 3 + 8i.
    const coeffs = [_]f64{ 1, 2, 3 };
    const v = evalComplex(&coeffs, .{ .re = 1, .im = 1 });
    try testing.expectApproxEqAbs(@as(f64, 3.0), v.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), v.im, 1e-12);
}

test "poly: evalDerivs rejects an empty coefficient slice" {
    var out: [3]f64 = undefined;
    try testing.expectError(Error.BadLength, evalDerivs(&[_]f64{}, 1.0, &out));
}

test "poly: a repeated-root quadratic reports a double real root" {
    // (x - 3)^2 = x^2 - 6x + 9 has a single value with multiplicity two; GSL's
    // solver returns two (equal) real roots for the zero-discriminant case.
    const r = solveQuadratic(1.0, -6.0, 9.0);
    try testing.expectEqual(@as(usize, 2), r.n);
    try testing.expectApproxEqAbs(@as(f64, 3.0), r.roots[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0), r.roots[1], 1e-12);
}

test "poly: a single-coefficient (constant) polynomial evaluates everywhere" {
    // The len==1 boundary: a constant polynomial, exercised on every eval form.
    const cst = [_]f64{3.5};
    try testing.expectApproxEqAbs(@as(f64, 3.5), eval(&cst, 100.0), 1e-12);
    const z = evalComplex(&cst, .{ .re = 2, .im = 5 });
    try testing.expectApproxEqAbs(@as(f64, 3.5), z.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), z.im, 1e-12);
    const cz = [_]Complex(f64){.{ .re = 3.5, .im = 0 }};
    const zz = evalComplexPoly(&cz, .{ .re = -1, .im = 4 });
    try testing.expectApproxEqAbs(@as(f64, 3.5), zz.re, 1e-12);
}

test "poly: divided-difference interpolation reproduces a cubic and its Taylor form" {
    // Sample p(x) = 1 - 2x + 0.5x^2 + 3x^3 at 4 points; dd interpolates it exactly.
    const coeffs = [_]f64{ 1, -2, 0.5, 3 };
    const xs = [_]f64{ -1, 0, 1, 2 };
    var ys: [4]f64 = undefined;
    for (xs, &ys) |xv, *yv| yv.* = eval(&coeffs, xv);

    var dd: [4]f64 = undefined;
    try ddInit(&dd, &xs, &ys);
    // Matches the source polynomial at off-node points.
    try testing.expectApproxEqAbs(eval(&coeffs, 0.5), ddEval(&dd, &xs, 0.5), 1e-12);
    try testing.expectApproxEqAbs(eval(&coeffs, -0.3), ddEval(&dd, &xs, -0.3), 1e-12);

    // Taylor expansion about xp = 0 recovers the original ascending coefficients.
    var taylor: [4]f64 = undefined;
    var work: [4]f64 = undefined;
    try ddTaylor(&taylor, 0.0, &dd, &xs, &work);
    for (coeffs, taylor) |cf, tf| try testing.expectApproxEqAbs(cf, tf, 1e-12);

    // Length checks.
    var too_small: [2]f64 = undefined;
    try testing.expectError(Error.BadLength, ddInit(&too_small, &xs, &ys));
}

test "poly: hermite divided differences match values and derivatives" {
    // p(x) = 2 + x - x^2 ; p'(x) = 1 - 2x.
    const coeffs = [_]f64{ 2, 1, -1 };
    const dcoeffs = [_]f64{ 1, -2 };
    const xs = [_]f64{ 0, 1, 2 };
    var ys: [3]f64 = undefined;
    var dys: [3]f64 = undefined;
    for (xs, &ys, &dys) |xv, *yv, *dv| {
        yv.* = eval(&coeffs, xv);
        dv.* = eval(&dcoeffs, xv);
    }
    var dd: [6]f64 = undefined;
    var za: [6]f64 = undefined;
    try ddHermiteInit(&dd, &za, &xs, &ys, &dys);
    // The Hermite interpolant (evaluated with `za`) equals p off-node.
    try testing.expectApproxEqAbs(eval(&coeffs, 1.5), ddEval(&dd, &za, 1.5), 1e-12);
    try testing.expectApproxEqAbs(eval(&coeffs, 0.25), ddEval(&dd, &za, 0.25), 1e-12);

    // Output-length checks (dd/za must be 2*n).
    var short_dd: [3]f64 = undefined;
    var short_za: [3]f64 = undefined;
    try testing.expectError(Error.BadLength, ddHermiteInit(&short_dd, &short_za, &xs, &ys, &dys));
}
