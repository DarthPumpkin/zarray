//! Shared geometry kernel for viewing a 2-axis `NamedArray` as a strided matrix.
//!
//! This is the *mechanism* layer of a deliberate two-layer split. It answers one
//! policy-free question about an index's `(shape, strides)`:
//!
//!   > Can these two axes be seen as a matrix with a single leading dimension
//!   > (one axis contiguous), and if so — is it row- or column-major, what are
//!   > its dimensions, and what is its major-axis stride (lda)?
//!
//! It knows nothing about any particular library's requirements, ABI enums,
//! integer widths, error-vs-panic policy, ownership bookkeeping, or even the
//! element pointer. Those all belong to the *policy* layer: each backend
//! (LAPACK's `describe`, BLAS's `Blas2d`, a future GSL `gsl_matrix` view) wraps
//! `analyze2d` in its own descriptor type that enforces its own rules —
//!
//!   - **LAPACK** (column-major native): accepts either layout, mapping
//!     row-major to a `trans` flag; propagates `NotContiguous` as a recoverable
//!     error.
//!   - **BLAS/CBLAS** (both layouts native): maps `layout` to `CBLAS_ORDER`;
//!     treats non-contiguity as a programmer error (`@panic`).
//!   - **GSL** (row-major only): would require `layout == .row_major` (rejecting
//!     or transpose-copying column-major) and attach `block`/`owner`.
//!
//! Deliberately takes only the *index* (not the array), so it stays
//! type-agnostic (no `Scalar`/const generics) and unit-testable with a bare
//! index. Each policy layer derives its own offset-aware base pointer via
//! `arr.at(zeroes)` — the one type-dependent line, and where const-ness is
//! decided (`[*]T` vs `[*]const T`).

const std = @import("std");

/// Which axis of the matrix is contiguous in memory (has stride 1).
/// `col_major`: the row axis is contiguous (Fortran order).
/// `row_major`: the column axis is contiguous (C order).
pub const Layout = enum { col_major, row_major };

/// The purely dimensional facts about a 2-axis view seen as a matrix. No
/// pointer, no element type, no ABI-specific integer width — just geometry.
pub const Geometry2d = struct {
    layout: Layout,
    rows: usize,
    cols: usize,
    /// Major-axis (outer) stride. The minor axis is contiguous (stride 1).
    /// For `col_major` this is the column stride; for `row_major`, the row
    /// stride. Clamped up to `max(1, minor_extent)` for degenerate single
    /// row/column views, so it always satisfies the `lda >= max(1, m)`
    /// requirement shared by LAPACK and BLAS.
    lda: usize,
};

pub const Error = error{
    /// Neither axis is unit-stride (a doubly-strided view) or a stride is
    /// non-positive (zero from a broadcast, or negative from a reversed view).
    /// A matrix needs a single leading dimension over positive strides; copy to
    /// a contiguous buffer and retry.
    NotContiguous,
};

/// Analyze two axes of an index as a single-`lda` strided matrix. `rows_name`
/// is the logical first (row) axis; `cols_name` the second (column) axis.
/// `idx` is any `NamedIndex` (taken as `anytype` to avoid a generic dependency).
///
/// A row or column *vector* (one extent equal to 1) is layout-ambiguous — the
/// same memory is validly both row- and column-major. `prefer` picks the label
/// in that tie only; it never affects a genuine matrix (both extents > 1), where
/// at most one layout is contiguous. The two backends deliberately differ here:
/// LAPACK (column-major native) prefers `.col_major`; BLAS prefers `.row_major`
/// so a lone column/row keeps the same label as the row-major operands it sits
/// beside.
pub fn analyze2d(
    idx: anytype,
    comptime rows_name: [:0]const u8,
    comptime cols_name: [:0]const u8,
    comptime prefer: Layout,
) Error!Geometry2d {
    const nr = @field(idx.shape, rows_name);
    const nc = @field(idx.shape, cols_name);
    if (nr == 0 or nc == 0) return error.NotContiguous;
    const sr = @field(idx.strides, rows_name);
    const sc = @field(idx.strides, cols_name);
    const nr_i: isize = @intCast(nr);
    const nc_i: isize = @intCast(nc);

    // Column-major: rows contiguous. A single column never steps by lda, so
    // clamp lda up to the row count (a lone column can have any/undefined
    // column stride).
    const col_ok = sr == 1 and (nc == 1 or sc >= nr_i);
    // Row-major: cols contiguous. A library reading this column-major sees Aᵀ.
    const row_ok = sc == 1 and (nr == 1 or sr >= nc_i);

    const col_result: Geometry2d = .{
        .layout = .col_major,
        .rows = nr,
        .cols = nc,
        .lda = if (nc == 1) @max(nr, 1) else @intCast(sc),
    };
    const row_result: Geometry2d = .{
        .layout = .row_major,
        .rows = nr,
        .cols = nc,
        .lda = if (nr == 1) @max(nc, 1) else @intCast(sr),
    };

    // Only ambiguous when an extent is 1 (a row/column vector); honor `prefer`.
    if (col_ok and row_ok) return switch (prefer) {
        .col_major => col_result,
        .row_major => row_result,
    };
    if (col_ok) return col_result;
    if (row_ok) return row_result;
    return error.NotContiguous;
}

// ===== Tests =================================================================

const NamedIndex = @import("named_index.zig").NamedIndex;

test "analyze2d: dense column-major" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 3, .j = 4 },
        .strides = .{ .i = 1, .j = 3 }, // columns contiguous
    };
    const g = try analyze2d(idx, "i", "j", .col_major);
    try std.testing.expectEqual(Layout.col_major, g.layout);
    try std.testing.expectEqual(@as(usize, 3), g.rows);
    try std.testing.expectEqual(@as(usize, 4), g.cols);
    try std.testing.expectEqual(@as(usize, 3), g.lda);
}

test "analyze2d: dense row-major" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 3, .j = 4 },
        .strides = .{ .i = 4, .j = 1 }, // rows contiguous
    };
    const g = try analyze2d(idx, "i", "j", .col_major);
    try std.testing.expectEqual(Layout.row_major, g.layout);
    try std.testing.expectEqual(@as(usize, 4), g.lda);
}

test "analyze2d: padded column-major submatrix (lda > rows)" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 3, .j = 2 },
        .strides = .{ .i = 1, .j = 5 }, // column stride 5 > 3 rows (padding)
        .offset = 7,
    };
    const g = try analyze2d(idx, "i", "j", .col_major);
    try std.testing.expectEqual(Layout.col_major, g.layout);
    try std.testing.expectEqual(@as(usize, 5), g.lda);
}

test "analyze2d: single column clamps lda to row count" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 4, .j = 1 },
        .strides = .{ .i = 1, .j = 999 }, // lone column: column stride irrelevant
    };
    const g = try analyze2d(idx, "i", "j", .col_major);
    try std.testing.expectEqual(Layout.col_major, g.layout);
    try std.testing.expectEqual(@as(usize, 4), g.lda);
}

test "analyze2d: vector tie-break honors prefer" {
    const IJ = enum { i, j };
    // A contiguous 4x1 column vector: both layouts describe the same memory.
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 4, .j = 1 },
        .strides = .{ .i = 1, .j = 1 },
    };
    const gc = try analyze2d(idx, "i", "j", .col_major);
    try std.testing.expectEqual(Layout.col_major, gc.layout);
    try std.testing.expectEqual(@as(usize, 4), gc.lda); // clamped to rows
    const gr = try analyze2d(idx, "i", "j", .row_major);
    try std.testing.expectEqual(Layout.row_major, gr.layout);
    try std.testing.expectEqual(@as(usize, 1), gr.lda); // clamped to cols
}

test "analyze2d: doubly strided is NotContiguous" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 3, .j = 4 },
        .strides = .{ .i = 2, .j = 8 }, // neither axis unit-stride
    };
    try std.testing.expectError(error.NotContiguous, analyze2d(idx, "i", "j", .col_major));
}

test "analyze2d: zero-size is NotContiguous" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 0, .j = 4 },
        .strides = .{ .i = 1, .j = 1 },
    };
    try std.testing.expectError(error.NotContiguous, analyze2d(idx, "i", "j", .col_major));
}

test "analyze2d: broadcast (stride 0) is NotContiguous" {
    const IJ = enum { i, j };
    const idx: NamedIndex(IJ) = .{
        .shape = .{ .i = 3, .j = 4 },
        .strides = .{ .i = 0, .j = 1 }, // broadcast row
    };
    // Row axis broadcast: cols contiguous but row stride 0 < cols ⇒ rejected.
    try std.testing.expectError(error.NotContiguous, analyze2d(idx, "i", "j", .col_major));
}
