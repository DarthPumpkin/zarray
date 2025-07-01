# zarray

*🚧 This repository is under construction. 🚧*

A safe and convenient n-dimensional array (a.k.a. tensor) library for Zig.

## Features
### Named dimensions
Array dimensions are named at compile time so you don't have to remember what the fourth axis of your five-dimensional array represents.
- Axis blunders are caught at compile time
- Less cognitive overhead
- More explicit code
```zig
const  IJ = enum { i, j };
const IJK = enum { i, j, k };

// Create 3x4x5 array
const a = za.arange(IJK, i32, allocator, .{
    .i = 3,
    .j = 4,
    .k = 5,
});
defer a.deinit(allocator);

// sums over all axes except { i, j }
// equivalent to np.sum(a, axis=2)
const b = a.sumAxes(IJ, allocator);
defer b.deinit(allocator);

// equivalent to a[:, :, 1] in numpy
const c = a.subArray(IJ, .{ .k = 1 });
```
<!-- Now, `b` is a 2x3 array with entries `.{ 1,  5,  9, 13, 17, 21 }`. -->

### Clear separation between view and copy semantics
A common pitfall with numpy-style fancy indexing is that it obscures which operations view the underlying array and which make a copy.
In `zarray`, all allocating functions take an allocator parameter.
```zig
// select every other element along j, skip first element along k.
// No allocation = no copy!
const c = a
    .step(.j, 2)
    .slice(.k, 1, 5);

assert(a.get({ .j = 2, .k = 1 }) == 11);

c.set({ .j = 1, .k = 0 }, -1);

assert(a.get({ .j = 2, .k = 1 }) == -1);
```

### Convenient `einsum`
[Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) is a general way to multiply and sum over axes by labeling them.
`zarray` makes it trivial to `einsum`, because axes are labeled from the get-go.
```zig
const IJ = enum { i, j };
const JK = enum { j, k };
const IK = enum { i, k };

const a = za.arange(JK, f64, allocator, .{
    .j = 3, .k = 4
});
defer a.deinit(allocator);

const b = za.ones(IJ, f64, allocator, .{
    .i = 2, .j = 3
});
defer b.deinit(allocator);

// implements matrix multiplication
// result has shape { .i = 2, .k = 4 };
const ab = za.einstein(IK, allocator, a, b);
defer ab.deinit(allocator);
```

### Wrap arbitrary slices
`zarray` arrays are just multi-dimensional views of regular Zig slices.
You can view any slice through a named array.

**Example:** View newline-delimited strings as 2D arrays.
```zig
const TextAxis = enum { line, col };

const buf = "abc\ndef\n";

// a 2-dimensional read-only view of buf that skips '\n'
const arr = NamedArrayConst(TextAxis, u8) {
    .idx = .{
        .shape = { .line = 2, .col = 3 },
        .strides = { .line = 4, .col = 1 },
    },
    .buf = &buf,
};

const arr11 = arr.get({ .line = 1, .col = 1 });
assert(arr11 == 'e');
```
