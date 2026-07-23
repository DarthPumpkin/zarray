//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const arr = @import("named_array.zig");

pub const NamedArray = arr.NamedArray;
pub const NamedArrayConst = arr.NamedArrayConst;

pub const index = @import("named_index.zig");
pub const axis_meta = @import("axis_meta.zig");
pub const bindings = struct {
    pub const blas = @import("bindings/blas.zig").blas;
    pub const lapack = @import("bindings/lapack/lapack.zig");
    pub const tblis = @import("bindings/tblis/tblis.zig");
    pub const gsl = @import("bindings/gsl/gsl.zig");

    test "bindings" {
        std.testing.refAllDecls(@This());
    }
};

pub const libs = bindings;

// temporary for testing purposes
pub const math = @import("math.zig");
pub const mlp_example = @import("mlp_example.zig");

test "root.zig" {
    std.testing.refAllDecls(@This());
}
