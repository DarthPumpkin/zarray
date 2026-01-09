const std = @import("std");
const za = @import("zarray");

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    std.debug.print("To do: run some benchmarks here.\n", .{});
}

fn bench_1() !void {
    // TO DO
}

test "bench.zig" {
    std.testing.refAllDecls(@This());
}
