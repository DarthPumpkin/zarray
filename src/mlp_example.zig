// Sources for weights and data
// Weights: https://huggingface.co/dacorvo/mnist-mlp/resolve/main/model.safetensors
// Test images: https://github.com/aimacode/aima-data/raw/f6cbea61ad0c21c6b7be826d17af5a8d3a7c2c86/MNIST/Digits/t10k-images-idx3-ubyte
// Test labels: https://github.com/aimacode/aima-data/raw/f6cbea61ad0c21c6b7be826d17af5a8d3a7c2c86/MNIST/Digits/t10k-labels-idx1-ubyte
const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const mem = std.mem;
const json = std.json;

const root = @import("zarray");
const NamedArray = root.NamedArray;
const NamedArrayConst = root.NamedArrayConst;
const blas = root.bindings.blas;

const InputAxis = enum { batch, in };
const OutputAxis = enum { batch, out };
const WeightsAxis = enum { in, out };
const BiasAxis = enum { out };

const mean = 0.1307;
const stddev = 0.3081;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const batch_size = 500;

    // Open files
    var args = init.minimal.args.iterate();
    _ = args.next();
    const data_path = args.next() orelse "data";
    const cwd = std.Io.Dir.cwd();
    const datadir = cwd.openDir(io, data_path, .{}) catch |err| {
        log.err("Could not open data directory '{s}': {s}", .{ data_path, @errorName(err) });
        return err;
    };
    const checkpoint_path = "model.safetensors";
    const images_path = "t10k-images-idx3-ubyte";
    const labels_path = "t10k-labels-idx1-ubyte";
    const checkpoint_file = try datadir.openFile(io, checkpoint_path, .{ .mode = .read_write });
    defer checkpoint_file.close(io);
    const images_file = try datadir.openFile(io, images_path, .{ .mode = .read_only });
    defer images_file.close(io);
    const labels_file = try datadir.openFile(io, labels_path, .{ .mode = .read_only });
    defer labels_file.close(io);

    // Memory-map the tensors
    var mmap = try checkpoint_file.createMemoryMap(io, .{ .len = try checkpoint_file.length(io) });
    defer mmap.destroy(io);
    try mmap.read(io);
    const weights_header_size_buffer: *const [8]u8 = mmap.memory[0..8];
    const weights_header_size = std.mem.readInt(u64, weights_header_size_buffer, .little);
    log.debug("Weights header size: {d}", .{weights_header_size});
    const weights_header_buffer = mmap.memory[8..][0..weights_header_size];
    const mlp_header = try json.parseFromSlice(DacorvoMlpHeader, gpa, weights_header_buffer, .{});
    defer mlp_header.deinit();
    log.debug("Parsed Safetensors header:\n{any}", .{mlp_header.value});
    const tensor_data: []const u8 = mmap.memory[8 + weights_header_size ..];
    const mlp = try mlp_header.value.readMlpBuffer(f32, gpa, tensor_data);
    defer mlp.deinit(gpa);

    // Load data headers
    var images_header_buffer: [4096]u8 = undefined;
    var images_reader = images_file.reader(io, &images_header_buffer);
    const images_shape = try readIdxHeader(&images_reader.interface, 3, 2051);
    log.info("Images shape: {any}", .{images_shape});
    var labels_header_buffer: [4096]u8 = undefined;
    var labels_reader = labels_file.reader(io, &labels_header_buffer);
    const labels_shape = try readIdxHeader(&labels_reader.interface, 1, 2049);
    log.info("Labels shape: {any}", .{labels_shape});

    if (images_shape[0] != labels_shape[0]) {
        log.err("Image count ({d}) does not match label count ({d})", .{ images_shape[0], labels_shape[0] });
        return error.ImageLabelCountMismatch;
    }
    const total_n = images_shape[0];
    const feature_dim = images_shape[1] * images_shape[2];

    // Pre-allocate batch memory and re-use across iterations
    const activations_buffer = try mlp.allocActivations(gpa, batch_size);
    defer gpa.free(activations_buffer);
    const labels_buffer = try gpa.alloc(u8, batch_size);
    defer gpa.free(labels_buffer);
    const images_buffer = try gpa.alloc(u8, batch_size * feature_dim);
    defer gpa.free(images_buffer);
    const batch_buffer = try gpa.alloc(f32, batch_size * feature_dim);
    defer gpa.free(batch_buffer);

    var n_correct: usize = 0;
    var total_prob: f64 = 0;
    var n_processed: usize = 0;

    while (n_processed < total_n) : (n_processed += @min(batch_size, total_n - n_processed)) {
        const actual_batch_size = @min(batch_size, total_n - n_processed);

        // Prepare network inputs
        const images_bytes = images_buffer[0 .. actual_batch_size * images_shape[1] * images_shape[2]];
        try images_reader.interface.readSliceAll(images_bytes);
        const images_proper = batch_buffer[0 .. actual_batch_size * images_shape[1] * images_shape[2]];
        const inv_255: f32 = 1.0 / 255.0;
        const inv_stddev: f32 = 1.0 / stddev;
        for (images_bytes, images_proper) |byt, *pro| {
            const x: f32 = @floatFromInt(byt);
            pro.* = (x * inv_255 - mean) * inv_stddev;
        }
        const batch = NamedArrayConst(InputAxis, f32).init(
            .initContiguous(.{ .batch = actual_batch_size, .in = feature_dim }),
            images_proper,
        );
        var batch_sample = batch;
        batch_sample.idx = batch_sample.idx.sliceAxis(.batch, 0, 2).sliceAxis(.in, 28 * 14, 29 * 14);
        const labels = labels_buffer[0..actual_batch_size];
        try labels_reader.interface.readSliceAll(labels);

        // Run through the network
        const output = try mlp.forward(batch, activations_buffer);
        const out_size = output.idx.shape.out;

        if (n_processed == 0) {
            log.debug("Batch sample (two images, center crop):\n{f}", .{batch_sample.fmtScalars("d:.4")});
            log.debug("Logits:\n{f}", .{output.fmtScalars("d:.4")});
        }

        softmaxInplace(f32, output);
        if (n_processed == 0) log.debug("Probs:\n{f}", .{output.fmtScalars("d:.4")});

        // Track metrics
        for (0..actual_batch_size) |b| {
            const row = output.indexAxes(enum { out }, .{ .batch = b }).flat().?;
            var best_class: usize = 0;
            var best_value = row[0];
            for (1..out_size) |j| {
                const value = row[j];
                if (value > best_value) {
                    best_value = value;
                    best_class = j;
                }
            }
            const label: usize = labels[b];
            if (best_class == label) {
                n_correct += 1;
            }
            total_prob += row[label];
        }
    }

    const accuracy: f64 = @as(f64, @floatFromInt(n_correct)) / total_n;
    const avg_prob = total_prob / total_n;
    log.info("Accuracy: {d}", .{accuracy});
    log.info("Average probability assigned to the true label: {d}", .{avg_prob});
}

// Borrows tensor storage. Owns only the layer metadata slice.
fn MLP(comptime Scalar_: type) type {
    const MlpInputConst = NamedArrayConst(InputAxis, Scalar_);
    const MlpOutput = NamedArray(OutputAxis, Scalar_);

    return struct {
        pub const Scalar: type = Scalar_;

        layers: []const Layer(Scalar),

        /// Forward pass.
        /// Allocate `memory` with `allocActivations`.
        /// Returned array is a view into `memory`.
        pub fn forward(self: @This(), batch: MlpInputConst, memory: []Scalar) !MlpOutput {
            const batch_size = batch.idx.shape.batch;
            const n_relu_layers = self.layers.len - 1; // no activation in final layer

            var input = batch;
            var output: MlpOutput = undefined;
            for (self.layers, 0..) |layer, li| {
                const out_size = layer.biases_1d.idx.shape.out;
                const out_scalars = batch_size * out_size;
                const output_buf = if (li % 2 == 0) memory[0..out_scalars] else memory[memory.len - out_scalars ..];
                output = .{ .buf = output_buf, .idx = .initContiguous(.{
                    .batch = batch_size,
                    .out = out_size,
                }) };
                for (0..batch_size) |bi| {
                    const row = output.indexAxes(enum { out }, .{ .batch = bi });
                    @memcpy(row.flat().?, layer.biases_1d.buf);
                }
                matmul(input, layer.weights_2d, output);

                if (li < n_relu_layers) {
                    for (output.flat().?) |*x| x.* = relu(x.*);
                    input = output.renameAxes(InputAxis, &.{.{ .old = "out", .new = "in" }}).asConst();
                }
            }
            return output;
        }

        pub fn allocActivations(self: @This(), al: mem.Allocator, batch_size: usize) ![]Scalar {
            // Pre-allocate a single buffer to hold two subsequent layers of activations.
            // We alternate between using the head and the tail as storage for the next layer.
            var memsize: usize = 0;
            for (self.layers[1..]) |layer| { // the first layer's inputs don't need allocation (input arg)
                const lsize = layer.weights_2d.idx.shape.in + layer.weights_2d.idx.shape.out;
                memsize = @max(memsize, lsize);
            }
            memsize *= batch_size;
            const memory = try al.alloc(Scalar, memsize);
            return memory;
        }

        pub fn deinit(self: @This(), al: mem.Allocator) void {
            al.free(self.layers);
        }

        fn matmul(
            x: NamedArrayConst(InputAxis, Scalar),
            w: NamedArrayConst(WeightsAxis, Scalar),
            out: NamedArray(OutputAxis, Scalar),
        ) void {
            blas.gemm(Scalar, InputAxis, WeightsAxis, OutputAxis, x, w, out, .{});
        }
    };
}

fn Layer(comptime Scalar: type) type {
    return struct {
        weights_2d: NamedArrayConst(WeightsAxis, Scalar),
        biases_1d: NamedArrayConst(BiasAxis, Scalar),
    };
}

/// Read the header of an IDX file (4-byte big-endian magic number followed by
/// `n_dims` big-endian dimension sizes), returning the shape. Leaves the reader
/// positioned at the start of the tensor data.
fn readIdxHeader(reader: anytype, comptime n_dims: usize, expected_magic: u32) ![n_dims]u32 {
    var header: [4 + n_dims * 4]u8 = undefined;
    try reader.readSliceAll(&header);
    const magic = std.mem.readInt(u32, header[0..4], .big);
    if (magic != expected_magic) {
        log.err("IDX file: expected magic number {d}, got {d}", .{ expected_magic, magic });
        return error.MagicNumberMismatch;
    }
    var shape: [n_dims]u32 = undefined;
    for (0..n_dims) |i| {
        shape[i] = std.mem.readInt(u32, header[(i + 1) * 4 ..][0..4], .big);
    }
    return shape;
}

fn relu(x: anytype) @TypeOf(x) {
    return @max(0, x);
}

fn softmaxInplace(comptime Scalar: type, x: NamedArray(OutputAxis, Scalar)) void {
    for (0..x.idx.shape.batch) |b| {
        const row = x.indexAxes(enum { out }, .{ .batch = b });
        std.debug.assert(row.idx.isContiguous());
        const row_buf = row.flat().?;

        var max: Scalar = -std.math.inf(Scalar);
        for (row_buf) |v| max = @max(max, v);

        var sum: Scalar = 0;
        for (row_buf) |*v| {
            v.* = @exp(v.* - max);
            sum += v.*;
        }

        for (row_buf) |*v| v.* /= sum;
    }
}

const DacorvoMlpHeader = struct {
    __metadata__: json.Value,
    @"input_layer.bias": SafetensorsInfo,
    @"input_layer.weight": SafetensorsInfo,
    @"mid_layer.bias": SafetensorsInfo,
    @"mid_layer.weight": SafetensorsInfo,
    @"output_layer.bias": SafetensorsInfo,
    @"output_layer.weight": SafetensorsInfo,

    pub fn maxTensorDataEnd(self: @This()) usize {
        var max_end: usize = 0;
        inline for (std.meta.fields(@This())) |field| {
            if (field.type == SafetensorsInfo) {
                const info: SafetensorsInfo = @field(self, field.name);
                max_end = @max(max_end, info.data_offsets[1]);
            }
        }
        return max_end;
    }

    pub fn readMlpBuffer(self: @This(), comptime Scalar: type, al: mem.Allocator, tensor_data: []const u8) !MLP(Scalar) {
        const layer_names = [_][]const u8{ "input_layer", "mid_layer", "output_layer" };
        const layers = try al.alloc(Layer(Scalar), layer_names.len);
        errdefer al.free(layers);

        inline for (layer_names, 0..) |lname, li| {
            const bias_name = lname ++ ".bias";
            const bias_info: SafetensorsInfo = @field(self, bias_name);
            if (bias_info.shape.len != 1) return error.InvalidTensorShape;
            const bias_slice = try viewSafetensorsBytesAsScalars(Scalar, bias_info, tensor_data);
            const bias_na = NamedArrayConst(BiasAxis, Scalar).init(
                .initContiguous(.{ .out = bias_info.shape[0] }),
                bias_slice,
            );

            const weight_name = lname ++ ".weight";
            const weight_info: SafetensorsInfo = @field(self, weight_name);
            if (weight_info.shape.len != 2) return error.InvalidTensorShape;
            const weight_slice = try viewSafetensorsBytesAsScalars(Scalar, weight_info, tensor_data);
            const weight_na = NamedArrayConst(WeightsAxis, Scalar).init(.{
                .shape = .{
                    .in = weight_info.shape[1],
                    .out = weight_info.shape[0],
                },
                .strides = .{
                    .in = 1,
                    .out = @intCast(weight_info.shape[1]),
                },
            }, weight_slice);

            layers[li] = Layer(Scalar){
                .biases_1d = bias_na,
                .weights_2d = weight_na,
            };
        }
        return .{ .layers = layers };
    }
};

fn viewSafetensorsBytesAsScalars(comptime Scalar: type, info: SafetensorsInfo, tensor_data: []const u8) ![]const Scalar {
    // Safetensors always stores tensor data little-endian. We reinterpret the
    // raw bytes as host-native scalars below, so this only works on
    // little-endian hosts. A big-endian host would need to byte-swap.
    comptime std.debug.assert(builtin.cpu.arch.endian() == .little);

    const expected_dtype = comptime scalarSafetensorDtype(Scalar);
    if (!mem.eql(u8, info.dtype, expected_dtype)) return error.UnexpectedTensorDType;

    const begin = info.data_offsets[0];
    const end = info.data_offsets[1];
    if (end < begin) return error.InvalidTensorOffsets;
    if (end > tensor_data.len) return error.TensorOffsetOutOfBounds;

    const expected_n_scalars = tensorElementCount(info.shape);
    const expected_n_bytes = expected_n_scalars * @sizeOf(Scalar);
    if (end - begin != expected_n_bytes) return error.UnexpectedTensorSize;

    if (begin % @alignOf(Scalar) != 0) return error.UnalignedTensorData;

    const src = tensor_data[begin..end];
    const aligned_src: []align(@alignOf(Scalar)) const u8 = @alignCast(src);
    return std.mem.bytesAsSlice(Scalar, aligned_src);
}

fn tensorElementCount(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

fn scalarSafetensorDtype(comptime Scalar: type) []const u8 {
    return switch (Scalar) {
        f16 => "F16",
        f32 => "F32",
        f64 => "F64",
        else => @compileError("Unsupported Scalar type for safetensors"),
    };
}

const SafetensorsInfo = struct { dtype: []u8, shape: []usize, data_offsets: [2]usize };
