// x implement ReLU in MLP
// x implement softmax
// x implement parse safetensors header
// x implement reading data (idx3-ubyte, id1-ubyte)
// Improvements:
// - implement ReLU with iter slices
// - implement ReLU with simd
// x test TBLIS vs CBLAS vs naïve
// Weights: https://huggingface.co/dacorvo/mnist-mlp/resolve/main/model.safetensors
// Test images: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
// Test labels: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
const std = @import("std");
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

    // Open files
    var args = init.minimal.args.iterate();
    _ = args.next();
    const data_path = args.next() orelse "data";
    const cwd = std.Io.Dir.cwd();
    const datadir = cwd.openDir(io, data_path, .{}) catch {
        std.debug.panic("{s} does not exist", .{data_path});
    };
    const checkpoint_path = "model.safetensors";
    const images_path = "t10k-images-idx3-ubyte";
    const labels_path = "t10k-labels-idx1-ubyte";
    const checkpoint_file = try datadir.openFile(io, checkpoint_path, .{ .mode = .read_only });
    defer checkpoint_file.close(io);
    const images_file = try datadir.openFile(io, images_path, .{ .mode = .read_only });
    defer images_file.close(io);
    const labels_file = try datadir.openFile(io, labels_path, .{ .mode = .read_only });
    defer labels_file.close(io);

    // Load images
    const magic_number_images = 2051;
    var images_buffer: [4096]u8 = undefined;
    var images_reader = images_file.reader(io, &images_buffer);
    var images_header: [16]u8 = undefined;
    try images_reader.interface.readSliceAll(&images_header);
    const actual_magic_number_images = std.mem.readInt(u32, images_header[0..4], .big);
    if (actual_magic_number_images != magic_number_images) {
        std.debug.panic("Images file: Expected magic number {d}, got {d}", .{ magic_number_images, actual_magic_number_images });
    }
    var images_shape: [3]u32 = undefined;
    for (0..3) |i| {
        images_shape[i] = std.mem.readInt(u32, images_header[(i + 1) * 4 ..][0..4], .big);
    }
    log.info("Images shape: {any}", .{images_shape});
    const images_bytes = try images_reader.interface.readAlloc(
        gpa,
        images_shape[0] * images_shape[1] * images_shape[2],
    );
    defer gpa.free(images_bytes);

    // Load labels
    const magic_number_labels = 2049;
    var labels_buffer: [4096]u8 = undefined;
    var labels_reader = labels_file.reader(io, &labels_buffer);
    var labels_header: [8]u8 = undefined;
    try labels_reader.interface.readSliceAll(&labels_header);
    const actual_magic_number_labels = std.mem.readInt(u32, labels_header[0..4], .big);
    if (actual_magic_number_labels != magic_number_labels) {
        std.debug.panic("Labels file: Expected magic number {d}, got {d}", .{ magic_number_labels, actual_magic_number_labels });
    }
    var labels_shape: [1]u32 = undefined;
    for (0..1) |i| {
        labels_shape[i] = std.mem.readInt(u32, labels_header[(i + 1) * 4 ..][0..4], .big);
    }
    log.info("Labels shape: {any}", .{labels_shape});
    const labels = try labels_reader.interface.readAlloc(gpa, labels_shape[0]);
    defer gpa.free(labels);

    // Load weights
    var weights_buffer: [4096]u8 = undefined;
    var weights_reader = checkpoint_file.reader(io, &weights_buffer);
    var weights_header_size_buffer: [8]u8 = undefined;
    try weights_reader.interface.readSliceAll(&weights_header_size_buffer);
    const weights_header_size = std.mem.readInt(u64, &weights_header_size_buffer, .little);
    log.debug("Weights header size: {d}", .{weights_header_size});
    const weights_header_buffer = try gpa.alloc(u8, weights_header_size);
    defer gpa.free(weights_header_buffer);
    try weights_reader.interface.readSliceAll(weights_header_buffer);
    const json_is_valid = try json.validate(gpa, weights_header_buffer);
    if (!json_is_valid) {
        std.debug.panic("JSON is invalid:\n{s}", .{weights_header_buffer});
    }
    const mlp_header = try json.parseFromSlice(DacorvoMlpHeader, gpa, weights_header_buffer, .{});
    defer mlp_header.deinit();
    log.debug("Parsed Safetensors header:\n{any}", .{mlp_header.value});
    const tensor_data_len = mlp_header.value.maxTensorDataEnd();
    const tensor_data = try gpa.alloc(u8, tensor_data_len);
    defer gpa.free(tensor_data);
    try weights_reader.interface.readSliceAll(tensor_data);
    const mlp_buffer = try mlp_header.value.readMlpBuffer(f32, gpa, tensor_data);
    defer mlp_buffer.deinit(gpa);

    // Prepare network inputs
    const images_proper = try gpa.alloc(f32, images_shape[0] * images_shape[1] * images_shape[2]);
    defer gpa.free(images_proper);
    for (images_bytes, images_proper) |byt, *pro| {
        pro.* = @floatFromInt(byt);
    }
    for (images_proper) |*pro| {
        pro.* /= 255.0;
        pro.* -= mean;
        pro.* /= stddev;
    }
    const batch = NamedArrayConst(InputAxis, f32).init(
        .initContiguous(.{
            .batch = labels_shape[0],
            .in = images_shape[1] * images_shape[2],
        }),
        images_proper,
    );
    var batch_sample = batch;
    batch_sample.idx = batch_sample.idx.sliceAxis(.batch, 0, 2).sliceAxis(.in, 28 * 14 + 7, 28 * 14 + 21);
    log.debug("Batch sample (two images):\n{f}", .{batch_sample});

    // Run through the network
    const mlp = MLP(f32){ .buffer = mlp_buffer };
    log.debug("Final layer:\n{f}", .{mlp.buffer.layers[2].biases_1d});
    const output = try mlp.forward(gpa, batch);
    defer output.deinit(gpa);
    const output_sample = output.indexAxesChecked(enum { out }, .{ .batch = 0 }).?;
    log.info("Logits:\n{any}", .{output_sample.buf[0..10]});
    softmaxInplace(f32, output);
    log.info("Probs:\n{any}", .{output_sample.buf[0..10]});

    // Calculate accuracy
    var n_correct: usize = 0;
    var total_prob: f64 = 0;
    if (output.idx.strides.out != 1) {
        @panic("Expected output.out to be unit stride");
    }
    const out_size = output.idx.shape.out;
    for (0..labels_shape[0]) |b| {
        const row_start = output.idx.linear(.{ .batch = b, .out = 0 });
        const row = output.buf[row_start..];
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
    const accuracy: f64 = @as(f64, @floatFromInt(n_correct)) / labels_shape[0];
    const avg_prob = total_prob / labels_shape[0];
    log.info("Accuracy: {d}", .{accuracy});
    log.info("Average confidence in correct label: {d}", .{avg_prob});
}

fn MLP(comptime Scalar_: type) type {
    const MlpInput = NamedArray(InputAxis, Scalar_);
    const MlpInputConst = NamedArrayConst(InputAxis, Scalar_);
    const MlpOutput = NamedArray(OutputAxis, Scalar_);

    // Owns its data. Must call `deinit()`.
    const Buffer_ = struct {
        const Scalar = Scalar_;

        layers: []const Layer(Scalar),

        pub fn initAlloc(al: mem.Allocator, layer_sizes: []const usize) !@This() {
            const n_layers = layer_sizes.len - 1;
            const layers = try al.alloc(Layer(Scalar), n_layers);
            for (layer_sizes[0..n_layers], layer_sizes[1..], layers) |lin, lout, *layer| {
                const weights = try NamedArray(WeightsAxis, Scalar).initAlloc(al, .{
                    .in = lin,
                    .out = lout,
                });
                const biases = try NamedArray(BiasAxis, Scalar).initAlloc(al, .{
                    .out = lout,
                });
                layer.* = .{
                    .weights_2d = weights,
                    .biases_1d = biases,
                };
            }
            return .{ .layers = layers };
        }

        pub fn deinit(self: @This(), al: mem.Allocator) void {
            for (self.layers) |layer| {
                layer.biases_1d.deinit(al);
                layer.weights_2d.deinit(al);
            }
            al.free(self.layers);
        }
    };

    return struct {
        pub const Scalar: type = Scalar_;
        pub const Buffer: type = Buffer_;

        buffer: Buffer,

        pub fn initZeros(buffer: Buffer) @This() {
            for (buffer.layers) |layer| {
                fillZeros(Scalar, layer);
            }
            return .{ .buffer = buffer };
        }

        pub fn iterLayers(self: @This()) LayerIterator(Scalar) {
            return .{ .buffer = self.buffer };
        }

        pub fn forward(self: @This(), al: mem.Allocator, batch: MlpInputConst) !MlpOutput {
            var input: MlpInput = try batch.toContiguous(al);
            const n_relu_layers = self.buffer.layers.len - 1; // no activation in final layer
            for (self.buffer.layers, 0..) |layer, li| {
                const batch_size = input.idx.shape.batch;
                const biases_2d: MlpOutput = layer.biases_1d.conformAxes(OutputAxis).broadcastAxis(.batch, batch_size);
                const output = try biases_2d.toContiguous(al);
                blas.gemm(
                    Scalar,
                    InputAxis,
                    WeightsAxis,
                    OutputAxis,
                    input.asConst(),
                    layer.weights_2d.asConst(),
                    output,
                    .{},
                );
                if (li < n_relu_layers) {
                    std.debug.assert(output.idx.isContiguous());
                    for (output.buf) |*x| {
                        x.* = relu(x.*);
                    }
                }

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

fn relu(x: anytype) @TypeOf(x) {
    return @max(0, x);
}

fn softmaxInplace(comptime Scalar: type, x: NamedArray(OutputAxis, Scalar)) void {
    for (0..x.idx.shape.batch) |b| {
        var max: Scalar = -std.math.inf(Scalar);
        for (0..x.idx.shape.out) |j| {
            max = @max(max, x.at(.{ .batch = b, .out = j }).*);
        }
        var sum: Scalar = 0;
        for (0..x.idx.shape.out) |j| {
            const x_j = x.at(.{ .batch = b, .out = j }).*;
            const term = @exp(x_j - max);
            sum += term;
        }
        for (0..x.idx.shape.out) |j| {
            const ptr = x.at(.{ .batch = b, .out = j });
            ptr.* = @exp(ptr.* - max) / sum;
        }
    }
}

fn fillZeros(comptime Scalar: type, layer: Layer(Scalar)) void {
    layer.biases_1d.fill(0);
    layer.weights_2d.fill(0);
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
        const tensor_names = [_][]const u8{
            "input_layer.bias",
            "input_layer.weight",
            "mid_layer.bias",
            "mid_layer.weight",
            "output_layer.bias",
            "output_layer.weight",
        };

        var max_end: usize = 0;
        inline for (tensor_names) |tname| {
            const info: SafetensorsInfo = @field(self, tname);
            max_end = @max(max_end, info.data_offsets[1]);
        }
        return max_end;
    }

    pub fn readMlpBuffer(self: @This(), comptime Scalar: type, al: mem.Allocator, tensor_data: []const u8) !MLP(Scalar).Buffer {
        const layer_names = [_][]const u8{ "input_layer", "mid_layer", "output_layer" };
        const layers = try al.alloc(Layer(Scalar), layer_names.len);
        var initialized_layers: usize = 0;
        errdefer {
            for (layers[0..initialized_layers]) |layer| {
                layer.biases_1d.deinit(al);
                layer.weights_2d.deinit(al);
            }
            al.free(layers);
        }

        inline for (layer_names, 0..) |lname, li| {
            const bias_name = lname ++ ".bias";
            const bias_info: SafetensorsInfo = @field(self, bias_name);
            if (bias_info.shape.len != 1) return error.InvalidTensorShape;
            const bias_na = try NamedArray(BiasAxis, Scalar).initAlloc(al, .{ .out = bias_info.shape[0] });
            errdefer bias_na.deinit(al);
            try @This().copyTensorIntoScalars(Scalar, bias_na.buf, bias_info, tensor_data);

            const weight_name = lname ++ ".weight";
            const weight_info: SafetensorsInfo = @field(self, weight_name);
            if (weight_info.shape.len != 2) return error.InvalidTensorShape;
            const weight_buf = try al.alloc(Scalar, weight_info.shape[0] * weight_info.shape[1]);
            errdefer al.free(weight_buf);
            const weight_na = NamedArray(WeightsAxis, Scalar).init(.{
                .shape = .{
                    .in = weight_info.shape[1],
                    .out = weight_info.shape[0],
                },
                .strides = .{
                    .in = 1,
                    .out = @intCast(weight_info.shape[1]),
                },
            }, weight_buf);
            try @This().copyTensorIntoScalars(Scalar, weight_na.buf, weight_info, tensor_data);

            layers[li] = Layer(Scalar){
                .biases_1d = bias_na,
                .weights_2d = weight_na,
            };
            initialized_layers += 1;
        }
        return .{ .layers = layers };
    }

    fn copyTensorIntoScalars(comptime Scalar: type, out: []Scalar, info: SafetensorsInfo, tensor_data: []const u8) !void {
        const expected_dtype = comptime scalarSafetensorDtype(Scalar);
        if (!mem.eql(u8, info.dtype, expected_dtype)) return error.UnexpectedTensorDType;

        const begin = info.data_offsets[0];
        const end = info.data_offsets[1];
        if (end < begin) return error.InvalidTensorOffsets;
        if (end > tensor_data.len) return error.TensorOffsetOutOfBounds;

        const expected_n_bytes = out.len * @sizeOf(Scalar);
        if (end - begin != expected_n_bytes) return error.UnexpectedTensorSize;

        const src = tensor_data[begin..end];
        const dst = std.mem.sliceAsBytes(out);
        std.mem.copyForwards(u8, dst, src);
    }

    fn scalarSafetensorDtype(comptime Scalar: type) []const u8 {
        return switch (Scalar) {
            f16 => "F16",
            f32 => "F32",
            f64 => "F64",
            else => @compileError("Unsupported Scalar type for safetensors"),
        };
    }
};

const SafetensorsInfo = struct { dtype: []u8, shape: []usize, data_offsets: [2]usize };
