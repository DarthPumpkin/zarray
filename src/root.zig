//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const arr = @import("named_array.zig");

pub const NamedArray = arr.NamedArray;
pub const NamedArrayConst = arr.NamedArrayConst;

pub const index = @import("named_index.zig");
pub const math = @import("math.zig");
pub const libs = struct {
    pub const blas = @import("accelerate.zig").blas;
    // pub const lapack = @import("accelerate.zig").lapack;
    pub const tblis = @import("tblis.zig");

    test "libs" {
        std.testing.refAllDecls(@This());
    }
};

test "root.zig" {
    std.testing.refAllDecls(@This());
}
