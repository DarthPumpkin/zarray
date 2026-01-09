//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const arr = @import("named_array.zig");

pub const NamedArray = arr.NamedArray;
pub const NamedArrayConst = arr.NamedArrayConst;

pub const index = @import("named_index.zig");
pub const math = @import("math.zig");
pub const libs = struct {
    pub const blas = @import("accelerate.zig").blas;
    pub const tblis = @import("tblis.zig");

    test "libs" {
        std.testing.refAllDecls(@This());
    }
};

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "root.zig" {
    std.testing.refAllDecls(@This());
    // const I = enum { i };
    // const Arr = NamedArray(I, f32);

    // const al = std.testing.allocator;
    // const a = try Arr.initAlloc(al, .{ .i = 2 });
    // defer a.deinit(al);
    // const b = try Arr.initAlloc(al, .{ .i = 2 });
    // defer b.deinit(al);
    // const c = try Arr.initAlloc(al, .{ .i = 2 });
    // defer c.deinit(al);

    // @memset(a.buf, 1.0);
    // @memset(b.buf, 2.0);
    // math.add(I, f32, a.asConst(), b.asConst(), c);
    // try std.testing.expectEqualDeep(&[_]f32{ 3.0, 3.0 }, c.buf);

    // const sum = libs.blas.asum(I, f32, c.asConst());
    // try std.testing.expectEqual(6.0, sum);
}
