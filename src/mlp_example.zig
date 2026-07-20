const std = @import("std");
const mem = std.mem;

const named_array = @import("named_array.zig");
const NamedArray = named_array.NamedArray;
const NamedArrayConst = named_array.NamedArrayConst;
const tblis = @import("tblis.zig");

const InputAxis = enum { batch, in };
const OutputAxis = enum { batch, out };
const WeightsAxis = enum { in, out };
const BiasAxis = enum { out };

test "main" {
    try main();
}

// pub fn main(init: std.process.Init) !void {
pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer std.debug.assert(gpa.deinit() == .ok);

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const al = arena.allocator();

    const mlp_buffer = try MLP(f32).alloc(al, &[_]usize{ 3, 3, 2 });
    const mlp = MLP(f32).initZeros(mlp_buffer);
    @memset(mlp_buffer.weights_flat, 1.0);

    const batch = try NamedArray(InputAxis, f32).initAlloc(al, .{
        .batch = 2,
        .in = 3,
    });
    batch.fillArange();
    const output = try mlp.forward(al, batch.asConst());
    std.log.debug("{f}\n", .{output});
}

fn MLP(comptime Scalar_: type) type {
    const MlpInput = NamedArray(InputAxis, Scalar_);
    const MlpInputConst = NamedArrayConst(InputAxis, Scalar_);
    const MlpOutput = NamedArray(OutputAxis, Scalar_);
    // const MlpOutputConst = NamedArrayConst(OutputAxis, Scalar);

    const Buffer_ = struct {
        const Scalar = Scalar_;

        weights_flat: []Scalar,
        biases_flat: []Scalar,
        layer_sizes: []const usize,

        pub fn initAlloc(al: mem.Allocator, layer_sizes: []const usize) !@This() {
            const self_layer_sizes: []usize = try al.alloc(usize, layer_sizes.len);
            @memcpy(self_layer_sizes, layer_sizes);

            const n_layers = layer_sizes.len - 1;
            var n_biases: usize = 0;
            var n_weights: usize = 0;
            for (layer_sizes[0..n_layers], layer_sizes[1..]) |lin, lout| {
                n_biases += lout;
                n_weights += lin * lout;
            }
            const biases_flat = try al.alloc(Scalar_, n_biases);
            const weights_flat = try al.alloc(Scalar_, n_weights);

            return .{
                .weights_flat = weights_flat,
                .biases_flat = biases_flat,
                .layer_sizes = self_layer_sizes,
            };
        }

        pub fn deinit(self: @This(), al: mem.Allocator) void {
            al.free(self.weights_flat);
            al.free(self.biases_flat);
            al.free(self.layer_sizes);
        }
    };

    return struct {
        const Scalar: type = Scalar_;
        const Buffer: type = Buffer_;

        buffer: Buffer,

        pub fn alloc(al: mem.Allocator, layer_sizes: []const usize) !Buffer {
            return try Buffer.initAlloc(al, layer_sizes);
        }

        pub fn initZeros(buffer: Buffer) @This() {
            var self: @This() = .{ .buffer = buffer };
            var layers = self.iterLayers();
            while (layers.next()) |layer| {
                fillZeros(Scalar, layer);
            }
            return self;
        }

        pub fn iterLayers(self: @This()) LayerIterator(Scalar) {
            return .{ .buffer = self.buffer };
        }

        pub fn forward(self: @This(), al: mem.Allocator, batch: MlpInputConst) !MlpOutput {
            var input: MlpInput = try batch.toContiguous(al);
            var layers = self.iterLayers();
            while (layers.next()) |layer| {
                const batch_size = input.idx.shape.batch;
                const biases_2d: MlpOutput = layer.biases_1d.conformAxes(OutputAxis).broadcastAxis(.batch, batch_size);
                const output = try biases_2d.toContiguous(al);
                tblis.mult(InputAxis, WeightsAxis, OutputAxis, Scalar, input.asConst(), layer.weights_2d.asConst(), output);

                al.free(input.buf);
                input = output.renameAxes(InputAxis, &.{.{ .old = "out", .new = "in" }});
            }
            return input.renameAxes(OutputAxis, &.{.{ .old = "in", .new = "out" }});
        }
    };
}

fn Layer(comptime Scalar: type) type {
    return struct {
        weights_2d: NamedArray(WeightsAxis, Scalar),
        biases_1d: NamedArray(BiasAxis, Scalar),
    };
}

fn LayerIterator(comptime Scalar: type) type {
    return struct {
        buffer: MLP(Scalar).Buffer,
        layer_offset: usize = 0,
        weights_offset: usize = 0,
        biases_offset: usize = 0,

        pub fn next(self: *@This()) ?Layer(Scalar) {
            const layer_sizes = self.buffer.layer_sizes;
            if (self.layer_offset + 1 >= layer_sizes.len) {
                return null;
            }
            const lin = layer_sizes[self.layer_offset];
            const lout = layer_sizes[self.layer_offset + 1];
            const layer: Layer(Scalar) = .{
                .weights_2d = NamedArray(WeightsAxis, Scalar).init(.{
                    .shape = .{ .in = @intCast(lin), .out = lout },
                    .strides = .{ .in = @intCast(lout), .out = 1 },
                }, self.buffer.weights_flat[self.weights_offset..][0 .. lin * lout]),
                .biases_1d = NamedArray(BiasAxis, Scalar).init(.initContiguous(.{ .out = lout }), self.buffer.biases_flat[self.biases_offset..][0..lout]),
            };

            self.layer_offset += 1;
            self.weights_offset += lin * lout;
            self.biases_offset += lout;
            return layer;
        }
    };
}

fn fillZeros(comptime Scalar: type, layer: Layer(Scalar)) void {
    layer.biases_1d.fill(0);
    layer.weights_2d.fill(0);
}
