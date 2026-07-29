# Formatting `NamedArray`: user guide & cookbook

This guide shows how to control `NamedArray` printing in day-to-day use.

Works the same for:

- `NamedArray(...)`
- `NamedArrayConst(...)`

---

## Quick start

Print with defaults:

```zig
try std.debug.print("{f}\n", .{arr});
```

Default behavior is tuned for readability:

- summarize only when the array is large (`threshold = 1000`)
- show head/tail slices (`edgeitems = 4`)
- wrap long rows around ~100 chars (`linewidth = 100`)
- align columns (`align_columns = true`)

---

## Most useful methods

- `arr.fmtWith(opts)`
  - change layout/wrapping/summarization options.
- `arr.fmtFull()`
  - disable summarization and print all values.
- `arr.fmtScalars(comptime scalar_fmt)`
  - keep default layout, but change scalar formatting.
- `arr.fmtWithScalars(opts, comptime scalar_fmt)`
  - customize both layout and scalar formatting.
- `arr.fmtFullScalars(comptime scalar_fmt)`
  - full print + custom scalar format.
- `arr.fmtCallback(comptime scalar_write)`
  - use a user-supplied scalar writer callback with default layout options.
- `arr.fmtWithCallback(opts: CallbackFormatOptions, comptime scalar_write)`
  - callback + custom layout options.
- `arr.fmtFullCallback(comptime scalar_write)`
  - callback + full output (no summarization).
- `arr.fmtDebug()`
  - includes shape/strides/offset/contiguous diagnostics.

---

## Layout options (`ArrayFormatOptions`)

```zig
pub const ArrayFormatOptions = struct {
    edgeitems: usize = 4,
    threshold: usize = 1000,
    linewidth: usize = 100,
    align_columns: bool = true,
};
```

What each does:

- `threshold`: summarize only if `count > threshold`
- `edgeitems`: number of values shown at each edge when summarized
- `linewidth`: preferred max row width before wrapping
- `align_columns`: right-align values in columns (`false` = compact)

Callback mode uses a separate options struct:

```zig
pub const CallbackFormatOptions = struct {
    edgeitems: usize = 4,
    threshold: usize = 1000,
    linewidth: usize = 100,
};
```

---

## Cookbook

### 1) Force summarization (even for small arrays)

```zig
try std.debug.print("{f}\n", .{
    arr.fmtWith(.{ .threshold = 0, .edgeitems = 2 }),
});
```

Output style:

```text
[0 1 ... 8 9]
```

### 2) Print everything (no `...`)

```zig
try std.debug.print("{f}\n", .{arr.fmtFull()});
```

### 3) Wrap earlier for narrow terminals

```zig
try std.debug.print("{f}\n", .{
    arr.fmtWith(.{ .linewidth = 12 }),
});
```

Output style:

```text
[ 0  1  2  3
  4  5  6  7
  8  9 10 11]
```

### 4) Compact output (no alignment padding)

```zig
try std.debug.print("{f}\n", .{
    arr.fmtWith(.{ .align_columns = false }),
});
```

### 5) Hex formatting for integer scalars

```zig
try std.debug.print("{f}\n", .{
    arr.fmtWithScalars(.{ .align_columns = false }, "x"),
});
```

Output style:

```text
[a b ff 1000]
```

### 6) Scientific notation for floats

```zig
try std.debug.print("{f}\n", .{arr.fmtScalars("e")});
```

### 7) Custom scalar types

For custom struct scalar types, use scalar format `"f"` and implement `format`:

```zig
const Writer = std.Io.Writer;

const Money = struct {
    cents: i64,

    pub fn format(self: @This(), w: *Writer) Writer.Error!void {
        try w.print("USD{d}", .{self.cents});
    }
};

try std.debug.print("{f}\n", .{
    prices.fmtWithScalars(.{ .align_columns = false }, "f"),
});
```

### 8) Supply a scalar writer callback

Sometimes you want complete control over scalar text (prefixes, units, custom escaping, etc.).

```zig
const Writer = std.Io.Writer;

const cb = struct {
    fn write(w: *Writer, v: i32) Writer.Error!void {
        try w.print("<{d}>", .{v});
    }
}.write;

try std.debug.print("{f}\n", .{
    arr.fmtWithCallback(.{ .linewidth = 14 }, cb),
});
```

Notes:

- Callback options intentionally have no `align_columns` field.
- `threshold`, `edgeitems`, and `linewidth` still apply.

---

## Scalar format string rules

`scalar_fmt` is the inside of `{...}`.

Examples:

- `"d"` decimal
- `"x"` hex
- `"e"` scientific
- `"f"` custom struct formatting hook
- `""` default `{}`

`scalar_fmt` is a **comptime** string (Zig-style), while layout options are runtime values.

For callback mode (`fmtWithCallback` / `fmtCallback`), scalar formatting is provided by your callback instead of `scalar_fmt`, and options come from `CallbackFormatOptions`.

---

## High-rank arrays: what to expect

For rank > 2, current summarization truncates the **flattened outer slice sequence** (row-major over outer axes), then prints labeled blocks.

That means you may see first/last outer blocks rather than per-axis “corner” blocks.

Example style:

```text
[a=0, b=0]
  [[ 0]]

...

[a=3, b=3]
  [[15]]
```

---

## NumPy users: quick mapping

Current defaults are intentionally NumPy-like in spirit:

- `threshold = 1000`
- `edgeitems = 4` (note: NumPy default is 3)
- `linewidth = 100` (NumPy default is 75)

If you want a tighter NumPy look, reduce `edgeitems`/`linewidth` in `fmtWith(...)`.
