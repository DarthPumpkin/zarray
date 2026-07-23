//! Non-portable BLAS extension namespaces.
//!
//! `bindings.blas` is the portability-oriented core. Backend-specific or
//! non-mandated CBLAS entry points live here.
//!
//! Current backends:
//! - `accelerate` — Apple Accelerate CBLAS extension symbols.

const std = @import("std");

pub const accelerate = @import("blas_ext/accelerate.zig");

/// Alias to the extension set of the currently selected BLAS backend.
/// Today the project targets Accelerate only.
pub const active = accelerate;

test "blas_ext" {
    std.testing.refAllDecls(@This());
}
