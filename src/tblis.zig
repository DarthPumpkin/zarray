const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const Complex = std.math.Complex;

const c = @cImport(@cInclude("tblis_zig.h"));
// const c = @cImport(@cInclude("tblis.h"));

const TblisScalar = c.tblis_zig_scalar;
const TblisTensor = c.tblis_zig_tensor;
const TblisTypeT = c.zig_type_t;
const TblisReduceT = c.reduce_t;
const c32_tblis = c.scomplex_zig;
const c64_tblis = c.dcomplex_zig;

fn init_type_t(comptime T: type) TblisTypeT {
    return switch (T) {
        f32 => c.ZIG_TYPE_SINGLE,
        f64 => c.ZIG_TYPE_DOUBLE,
        Complex(f32) => c.ZIG_TYPE_SCOMPLEX,
        Complex(f64) => c.ZIG_TYPE_DCOMPLEX,
        else => @compileError("init_type_t: T must be f32, f64 or Complex(...)"),
    };
}

fn init_scalar(comptime T: type, val: T) TblisScalar {
    const type_ = init_type_t(T);
    const data: c.zig_scalar = switch (T) {
        f32 => .{ .s = val },
        f64 => .{ .d = val },
        Complex(f32) => .{ .c = c32_tblis{ .re = val.re, .im = val.im } },
        Complex(f64) => .{ .z = c64_tblis{ .re = val.re, .im = val.im } },
        else => @compileError("init_scalar: T must be f32, f64 or Complex(...)"),
    };
    return .{ .data = data, .type = type_ };
}

fn init_tensor(
    comptime T: type,
    comptime rank: usize,
    shape: *const [rank]c.zig_len_type,
    stride: *const [rank]c.zig_stride_type,
    data: [*]T,
) TblisTensor {
    const type_ = init_type_t(T);
    const scalar = init_scalar(T, one(T));
    return .{
        .type = type_,
        .conj = 0.0,
        .scalar = scalar,
        .data = data,
        .ndim = rank,
        .len = @ptrCast(shape),
        .stride = @ptrCast(stride),
    };
}

fn get_scalar_val(comptime T: type, scalar: TblisScalar) T {
    const T_type_t = init_type_t(T);
    assert(T_type_t == scalar.type);

    return switch (T) {
        f32 => scalar.data.s,
        f64 => scalar.data.d,
        Complex(f32) => .init(scalar.data.c.re, scalar.data.c.im),
        Complex(f64) => .init(scalar.data.z.re, scalar.data.z.im),
        else => @compileError("get_scalar_val: T must be f32, f64 or Complex(...)"),
    };
}

fn one(comptime T: type) T {
    return switch (T) {
        f32, f64 => 1.0,
        Complex(f32), Complex(f64) => .{ .re = 1.0, .im = 0.0 },
        else => @compileError("one: T must be f32, f64 or Complex(...)"),
    };
}

test "tensor init" {
    var shape = [_]c.zig_len_type{ 2, 3, 4 };
    var stride = [_]c.zig_stride_type{ 12, 4, 1 };
    var data: [24]f32 = undefined;
    @memset(&data, 1.0);
    var tf: TblisTensor = undefined;
    const scalar = TblisScalar{
        .data = .{ .s = 1.0 },
        .type = c.ZIG_TYPE_SINGLE,
    };
    tf = .{
        .type = c.ZIG_TYPE_SINGLE,
        .conj = 0,
        .scalar = scalar,
        .data = data[0..].ptr,
        .ndim = 3,
        .len = shape[0..].ptr,
        .stride = stride[0..].ptr,
    };
    const tf_ziginit = init_tensor(f32, 3, &shape, &stride, &data);

    try std.testing.expectEqual(tf.type, tf_ziginit.type);
    try std.testing.expectEqual(tf.conj, tf_ziginit.conj);
    try std.testing.expectEqual(tf.scalar.type, tf_ziginit.scalar.type);
    try std.testing.expectEqual(tf.scalar.data.s, tf_ziginit.scalar.data.s);
    try std.testing.expectEqual(tf.data, tf_ziginit.data);
    try std.testing.expectEqual(tf.ndim, tf_ziginit.ndim);
    try std.testing.expectEqual(tf.len, tf_ziginit.len);
    try std.testing.expectEqual(tf.stride, tf_ziginit.stride);
}

test "add.h equal shape and stride" {
    const T = Complex(f32);
    const n = 12;
    const rank = 3;
    const a_shape = [rank]c.zig_len_type{ 2, 3, 2 };
    const b_shape = [rank]c.zig_len_type{ 2, 3, 2 };
    const a_stride = [rank]c.zig_stride_type{ 6, 2, 1 };
    const b_stride = [rank]c.zig_stride_type{ 6, 2, 1 };
    const a_idx = "ijk";
    const b_idx = "ijk";
    var a_data: [n]T = undefined;
    @memset(&a_data, .{ .re = 1.0, .im = 2.0 });
    var b_data: [n]T = undefined;
    @memset(&b_data, .{ .re = 2.0, .im = 0.0 });
    const a = init_tensor(T, rank, &a_shape, &a_stride, a_data[0..].ptr);
    var b = init_tensor(T, rank, &b_shape, &b_stride, b_data[0..].ptr);
    c.tblis_zig_tensor_add(
        c.tblis_single,
        null,
        &a,
        a_idx,
        &b,
        b_idx,
    );
    for (b_data) |bi| {
        try std.testing.expectEqualDeep(bi, T{ .re = 3.0, .im = 2.0 });
    }
}

test "add.h 2d + 1d" {
    // a = 1d, b = 2d, col major. Adding a to each row of b
    const m = 2;
    const n = 3;

    // b is 2D (m x n), column-major: stride = [1, m]
    const b_shape = [2]c.zig_len_type{ m, n };
    const b_stride = [2]c.zig_stride_type{ 1, m };

    // a is 1D (length n), contiguous
    const a_shape = [1]c.zig_len_type{n};
    const a_stride = [1]c.zig_stride_type{1};

    var b_data: [m * n]f32 = undefined;
    @memset(&b_data, 2.0);

    var a_data: [n]f32 = .{ 1.0, 2.0, 3.0 };

    const a = init_tensor(f32, 1, &a_shape, &a_stride, a_data[0..].ptr);
    var b = init_tensor(f32, 2, &b_shape, &b_stride, b_data[0..].ptr);

    const a_idx = "j";
    const b_idx = "ij";

    c.tblis_zig_tensor_add(
        null,
        null,
        &a,
        a_idx,
        &b,
        b_idx,
    );

    // Verify: b[i, j] = 2.0 + a[j], with column-major indexing: idx = i + j*m
    for (0..n) |j| {
        for (0..m) |i| {
            const idx = i + j * m;
            try std.testing.expectEqual(2.0 + a_data[j], b_data[idx]);
        }
    }
}

test "dot.h row col" {
    const T = f32;
    const m = 1;
    const n = 4;
    const k = 1;
    const a_data = [m * n]T{ 1, 2, 3, 4 };
    const b_data = [n * k]T{ 1, 10, 100, 1000 };
    const a_shape = [2]c.zig_len_type{ m, n };
    const b_shape = [2]c.zig_len_type{ n, k };
    // row-major
    const a_stride = [2]c.zig_stride_type{ n, 1 };
    // col-major
    const b_stride = [2]c.zig_stride_type{ 1, n };

    const expected: T = 4321;

    const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
    const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
    var result_struct = init_scalar(T, undefined);
    c.tblis_zig_tensor_dot(
        null,
        null,
        &a,
        "ij",
        &b,
        "jk",
        &result_struct,
    );
    const actual = result_struct.data.s;

    try std.testing.expectEqual(expected, actual);
}

test "dot.h ij ij" {
    const T = f64;
    const m = 2;
    const n = 2;
    const k = 2;
    const a_data = [m * n]T{ 1, 2, 3, 4 };
    const b_data = [n * k]T{ 1, 10, 100, 1000 };
    const a_shape = [2]c.zig_len_type{ m, n };
    const b_shape = [2]c.zig_len_type{ n, k };
    // row-major
    const a_stride = [2]c.zig_stride_type{ n, 1 };
    // row-major
    const b_stride = [2]c.zig_stride_type{ k, 1 };

    const expected: T = 4321;

    const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
    const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
    var result_struct = init_scalar(T, undefined);
    c.tblis_zig_tensor_dot(
        null,
        null,
        &a,
        "ij",
        &b,
        "ij",
        &result_struct,
    );
    const actual = get_scalar_val(T, result_struct);

    try std.testing.expectEqual(expected, actual);
}

// I guess dot only works for arrays with equal axes?
// test "dot.h ij jk" {
//     const T = f64;
//     const m = 2;
//     const n = 2;
//     const k = 3;
//     const a_data = [m * n]T{ 1, 2, 3, 4 };
//     const b_data = [n * k]T{ 1, 10, 100, 1000, 10_000, 100_000 };
//     const a_shape = [2]c.zig_len_type{ m, n };
//     const b_shape = [2]c.zig_len_type{ n, k };
//     // row-major
//     const a_stride = [2]c.zig_stride_type{ n, 1 };
//     // col-major
//     const b_stride = [2]c.zig_stride_type{ 1, n };

//     // c_11 =      21
//     // c_12 =   2_100
//     // c_13 = 210_000
//     // c_21 =      43
//     // c_22 =   4_300
//     // c_23 = 430_000
//     // sum  = 646_464
//     const expected: T = 646_464;

//     const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
//     const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
//     var result_struct = init_scalar(T, undefined);
//     c.tblis_zig_tensor_dot(
//         null,
//         null,
//         &a,
//         "ij",
//         &b,
//         "jk",
//         &result_struct,
//     );
//     const actual = get_scalar_val(T, result_struct);

//     try std.testing.expectEqual(expected, actual);
// }

test "mult.h ij jk" {
    const T = f64;
    const m = 2;
    const n = 2;
    const k = 3;
    const a_data = [m * n]T{ 1, 2, 3, 4 };
    const b_data = [n * k]T{ 1, 10, 100, 1000, 10_000, 100_000 };
    var c_data: [m * k]T = undefined;
    @memset(&c_data, 0.0);
    const a_shape = [2]c.zig_len_type{ m, n };
    const b_shape = [2]c.zig_len_type{ n, k };
    const c_shape = [2]c.zig_len_type{ m, k };
    // row-major
    const a_stride = [2]c.zig_stride_type{ n, 1 };
    // col-major
    const b_stride = [2]c.zig_stride_type{ 1, n };
    // row-major
    const c_stride = [2]c.zig_stride_type{ k, 1 };

    const a = init_tensor(T, 2, &a_shape, &a_stride, @constCast(&a_data));
    const b = init_tensor(T, 2, &b_shape, &b_stride, @constCast(&b_data));
    var c_ = init_tensor(T, 2, &c_shape, &c_stride, &c_data);

    // c_11 =      21
    // c_12 =   2_100
    // c_13 = 210_000
    // c_21 =      43
    // c_22 =   4_300
    // c_23 = 430_000
    const expected = [m * k]T{ 21, 2100, 210_000, 43, 4300, 430_000 };

    c.tblis_zig_tensor_mult(
        null,
        null,
        &a,
        "ij",
        &b,
        "jk",
        &c_,
        "ik",
    );

    try std.testing.expectEqualSlices(T, &expected, &c_data);
}

test "mult.h i jk -> ijk" {
    const T = f64;
    const m = 2; // i-dim
    const n = 2; // j-dim
    const k = 3; // k-dim

    // a: 1D vector over "i"
    const a_shape = [1]c.zig_len_type{m};
    const a_stride = [1]c.zig_stride_type{1};
    var a_data: [m]T = .{ 1.0, 2.0 };

    // b: 2D matrix over "jk" (row-major)
    const b_shape = [2]c.zig_len_type{ n, k };
    const b_stride = [2]c.zig_stride_type{ k, 1 };
    var b_data: [n * k]T = .{
        // j = 0 row
        10.0,  20.0,  30.0,
        // j = 1 row
        100.0, 200.0, 300.0,
    };

    // c: 3D tensor over "ijk" (row-major)
    const c_shape = [3]c.zig_len_type{ m, n, k };
    const c_stride = [3]c.zig_stride_type{ n * k, k, 1 };
    var c_data: [m * n * k]T = undefined;
    @memset(&c_data, 0.0);

    const a = init_tensor(T, 1, &a_shape, &a_stride, a_data[0..].ptr);
    const b = init_tensor(T, 2, &b_shape, &b_stride, b_data[0..].ptr);
    var c_ = init_tensor(T, 3, &c_shape, &c_stride, &c_data);

    // c[i, j, k] = a[i] * b[j, k]
    c.tblis_zig_tensor_mult(
        null,
        null,
        &a,
        "i",
        &b,
        "jk",
        &c_,
        "ijk",
    );

    const expected = [m * n * k]T{
        // i = 0 => a[0] = 1
        10.0, 20.0, 30.0, 100.0, 200.0, 300.0,
        // i = 1 => a[1] = 2
        20.0, 40.0, 60.0, 200.0, 400.0, 600.0,
    };

    try std.testing.expectEqualSlices(T, &expected, &c_data);
}

test "reduce.h i" {
    const T = f64;
    const rank = 1;
    const a_shape = [rank]c.zig_len_type{5};
    const a_stride = [rank]c.zig_stride_type{1};
    const a_data = &[_]T{ 1, -2, 3, -4, 5 };
    var a = init_tensor(T, 1, &a_shape, &a_stride, @constCast(a_data[0..].ptr));
    a.scalar = init_scalar(T, 2);

    var a_sum = init_scalar(T, undefined);
    var a_sum_idx: [rank]c.zig_len_type = undefined;
    c.tblis_zig_tensor_reduce(
        null,
        null,
        c.ZIG_REDUCE_SUM,
        &a,
        "i",
        &a_sum,
        &a_sum_idx,
    );
    try std.testing.expectEqual(init_type_t(T), a_sum.type);
    try std.testing.expectEqual(3.0 * 2, a_sum.data.d);

    var a_sum_abs = init_scalar(T, undefined);
    var a_sum_abs_idx: [rank]c.zig_len_type = undefined;
    c.tblis_zig_tensor_reduce(
        null,
        null,
        c.ZIG_REDUCE_SUM_ABS,
        &a,
        "i",
        &a_sum_abs,
        &a_sum_abs_idx,
    );
    try std.testing.expectEqual(init_type_t(T), a_sum_abs.type);
    try std.testing.expectEqual(15.0 * 2, a_sum_abs.data.d);

    var a_max = init_scalar(T, undefined);
    var a_max_idx: [rank]c.zig_len_type = undefined;
    c.tblis_zig_tensor_reduce(
        null,
        null,
        c.ZIG_REDUCE_MAX,
        &a,
        "i",
        &a_max,
        &a_max_idx,
    );
    try std.testing.expectEqual(init_type_t(T), a_max.type);
    try std.testing.expectEqual(5.0 * 2, a_max.data.d);
    try std.testing.expectEqualDeep([_]c.zig_len_type{4}, a_max_idx);
}

test "scale.h" {
    const T = Complex(f64);
    const rank = 4;
    const a_shape = [rank]c.zig_len_type{ 1, 1, 3, 1 };
    const a_stride = [rank]c.zig_stride_type{ 6290, 19348, 1, 6890000 };
    var a_data = [_]T{
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 1.0, .im = -1.0 },
        .{ .re = 0.0, .im = 1.0 },
    };
    var a = init_tensor(T, rank, &a_shape, &a_stride, a_data[0..].ptr);
    a.scalar = init_scalar(T, .init(1.0, 1.0));

    c.tblis_zig_tensor_scale(
        null,
        null,
        &a,
        "ijkl",
    );

    const expected = [_]T{
        .init(1.0, 1.0),
        .init(2.0, 0.0),
        .init(-1.0, 1.0),
    };
    try std.testing.expectEqualDeep(expected, a_data);
    try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
}

test "set.h f32 contiguous 1d" {
    const T = f32;
    const rank = 1;
    const len = [rank]c.zig_len_type{5};
    const stride = [rank]c.zig_stride_type{1};
    var data: [5]T = undefined;
    @memset(&data, 0.0);

    var a = init_tensor(T, rank, &len, &stride, data[0..].ptr);
    const alpha = init_scalar(T, 3.5);

    c.tblis_zig_tensor_set(
        null,
        null,
        &alpha,
        &a,
        "i",
    );

    const expected = [_]T{3.5} ** 5;
    try std.testing.expectEqualDeep(expected, data);
    try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
}

test "set.h complex strided 3d with gaps" {
    const T = Complex(f64);
    const m = 2;
    const n = 2;
    const p = 2;
    const rank = 3;

    // Layout: simulate padding by placing planes 8 apart, rows 4 apart, cols 1 apart.
    // Only 8 logical elements, but allocate a larger buffer and verify non-accessed padding unchanged.
    const len = [rank]c.zig_len_type{ m, n, p };
    const stride = [rank]c.zig_stride_type{ 8, 4, 1 };
    const logical = [_]usize{ 0, 1, 4, 5, 8, 9, 12, 13 };

    var buf: [16]T = undefined;
    @memset(&buf, .init(0, 0));

    var a = init_tensor(T, rank, &len, &stride, buf[0..].ptr);
    // Check that a.scalar is ignored
    a.scalar = init_scalar(T, .init(2, 0));

    const alpha = init_scalar(T, one(T));
    c.tblis_zig_tensor_set(
        null,
        null,
        &alpha,
        &a,
        "ijk",
    );

    // Verify logical positions were modified, and others were not.
    for (0..16) |i| {
        const is_logical = std.mem.containsAtLeastScalar(usize, &logical, 1, i);
        const expected = if (is_logical) one(T) else T.init(0, 0);
        try std.testing.expectEqual(expected, buf[i]);
    }
    try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
}

test "shift.h f32 contiguous 1d" {
    const T = f32;
    const rank = 1;
    const len = [rank]c.zig_len_type{5};
    const stride = [rank]c.zig_stride_type{1};

    var data: [5]T = .{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var a = init_tensor(T, rank, &len, &stride, data[0..].ptr);
    a.scalar = init_scalar(T, 2.0);

    const alpha = init_scalar(T, 3.5);

    c.tblis_zig_tensor_shift(
        null,
        null,
        &alpha,
        &a,
        "i",
    );

    const expected = [_]T{ 5.5, 7.5, 9.5, 11.5, 13.5 };
    try std.testing.expectEqualDeep(expected, data);
    try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
}

test "shift.h complex strided 3d with gaps" {
    const T = Complex(f64);
    const m = 2;
    const n = 2;
    const p = 2;
    const rank = 3;

    // Layout with padding: planes 8 apart, rows 4 apart, cols 1 apart.
    const len = [rank]c.zig_len_type{ m, n, p };
    const stride = [rank]c.zig_stride_type{ 8, 4, 1 };

    // Logical positions (same mapping as in the set test)
    const logical = [_]usize{ 0, 1, 4, 5, 8, 9, 12, 13 };

    var buf: [16]T = .{
        // plane 0 (indices 0..7)
        T.init(0, 0), T.init(1, 0), T.init(0, 0), T.init(0, 0),
        T.init(2, 0), T.init(3, 0), T.init(0, 0), T.init(0, 0),
        // plane 1 (indices 8..15)
        T.init(4, 0), T.init(5, 0), T.init(0, 0), T.init(0, 0),
        T.init(6, 0), T.init(7, 0), T.init(0, 0), T.init(0, 0),
    };

    var a = init_tensor(T, rank, &len, &stride, buf[0..].ptr);

    const alpha = init_scalar(T, one(T));

    c.tblis_zig_tensor_shift(
        null,
        null,
        &alpha,
        &a,
        "ijk",
    );

    // Verify logical positions were incremented by alpha and padding unchanged.
    for (0..16) |i| {
        const is_logical = std.mem.containsAtLeastScalar(usize, &logical, 1, i);
        const before = switch (i) {
            0 => T.init(0, 0),
            1 => T.init(1, 0),
            4 => T.init(2, 0),
            5 => T.init(3, 0),
            8 => T.init(4, 0),
            9 => T.init(5, 0),
            12 => T.init(6, 0),
            13 => T.init(7, 0),
            else => T.init(0, 0),
        };
        const expected = if (is_logical) T.init(before.re + 1.0, before.im) else before;
        try std.testing.expectEqual(expected, buf[i]);
    }

    // Scalar should be reset to 1 after operation
    try std.testing.expectEqualDeep(one(T), get_scalar_val(T, a.scalar));
}

test "thread.h" {
    _ = c.tblis_comm;
    _ = c.tblis_single;
    // const TblisComm = c.tblis_comm;
    // comptime for (@typeInfo(TblisComm).@"struct".fields) |field| {
    //     @compileLog(field.name);
    //     @compileLog(@typeInfo(field.type));
    // };
    // std.debug.print("{any}\n", .{c.tblis_single.*});
    // const num_threads_before = c.tblis_get_num_threads();
    // std.debug.print("Default: {} threads\n", .{num_threads_before});
    c.tblis_set_num_threads(2);
    const num_threads_after = c.tblis_get_num_threads();
    try std.testing.expectEqual(2, num_threads_after);
}
