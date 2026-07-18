const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const assert = std.debug.assert;

// Axis-keyed struct factories.
//
// These are the name-list generalization of `std.enums.EnumFieldStruct`: given a
// list of axis names they build a struct with one `T`-typed field per axis, in
// declaration order. In fact `AxesStructOf(std.meta.fieldNames(E), T)` is
// structurally identical to `std.enums.EnumFieldStruct(E, T, null)` (both lower to
// the same `@Struct(.auto, ...)` call); the equivalence is pinned by the
// "factories match std.enums.EnumFieldStruct" test below.
//
// We key on names rather than an enum, and keep a single factory as the sole
// constructor, for two reasons:
//   1. Several axis transforms build structs for *derived* name lists (unions,
//      splits, renames) that don't correspond to a single input enum, so an
//      enum-only helper like `EnumFieldStruct` can't express them. Where names do
//      come from an enum they're obtained via `std.meta.fieldNames`.
//   2. `@Struct` does not deduplicate: two separate calls with identical
//      arguments produce *distinct* types. Type identity across the codebase
//      therefore relies on comptime memoization of these factory functions, so
//      every axis-keyed struct must be minted here rather than by calling
//      `EnumFieldStruct` (or `@Struct`) directly at each site.
//
// Layout is the default `auto`, exactly like `EnumFieldStruct`.
pub fn AxesStructOf(comptime names: []const [:0]const u8, comptime T: type) type {
    return @Struct(.auto, null, names, &@splat(T), &@splat(.{}));
}

// Optional variant, used for partial specifications (e.g. per-axis steps), where
// unspecified axes default to null. Matches `EnumFieldStruct(E, ?T, <null default>)`.
pub fn AxesOptionalStructOf(comptime names: []const [:0]const u8, comptime T: type) type {
    const optT = ?T;
    const default_val: optT = null;
    return @Struct(
        .auto,
        null,
        names,
        &@splat(optT),
        &@splat(.{ .default_value_ptr = &default_val }),
    );
}

pub fn AxesStruct(comptime names: []const [:0]const u8) type {
    return AxesStructOf(names, usize);
}

pub fn AxesOptionalStruct(comptime names: []const [:0]const u8) type {
    return AxesOptionalStructOf(names, usize);
}

pub fn KeyEnum(comptime names: []const [:0]const u8) type {
    const rank = names.len;
    const bits = switch (rank) {
        0 => 0,
        else => std.math.log2_int_ceil(usize, rank),
    };
    const TagType = @Int(.unsigned, bits);
    const field_values = comptime blk: {
        var fv_: [rank]TagType = undefined;
        for (0..rank) |i| fv_[i] = i;
        break :blk fv_;
    };
    return @Enum(TagType, .exhaustive, names, &field_values);
}

pub fn Xor(comptime Enum1: type, comptime Enum2: type) type {
    const info1 = @typeInfo(Enum1).@"enum";
    const info2 = @typeInfo(Enum2).@"enum";
    var common1 = mem.zeroes([info1.fields.len]bool);
    var common2 = mem.zeroes([info2.fields.len]bool);
    comptime var num_matches: usize = 0;

    inline for (0..info1.fields.len) |fi| {
        inline for (0..info2.fields.len) |fj| fj_blk: {
            if (mem.eql(u8, info1.fields[fi].name, info2.fields[fj].name)) {
                common1[fi] = true;
                common2[fj] = true;
                num_matches += 1;
                break :fj_blk;
            }
        }
    }

    const xor_len = info1.fields.len + info2.fields.len - 2 * num_matches;
    comptime var xor_fnames: [xor_len][:0]const u8 = undefined;
    var i: usize = 0;
    inline for (info1.fields, 0..) |field, fi| if (!common1[fi]) {
        xor_fnames[i] = field.name;
        i += 1;
    };
    inline for (info2.fields, 0..) |field, fj| if (!common2[fj]) {
        xor_fnames[i] = field.name;
        i += 1;
    };
    assert(i == xor_len);
    return KeyEnum(&xor_fnames);
}

/// Difference of axis names `EnumAll \ EnumSubset` as an enum.
///
/// Requires that every field in `EnumSubset` appears in `EnumAll`.
pub fn Difference(comptime EnumAll: type, comptime EnumSubset: type) type {
    const all_info = @typeInfo(EnumAll).@"enum";
    const subset_info = @typeInfo(EnumSubset).@"enum";

    comptime {
        for (subset_info.fields) |subset_field| {
            var found = false;
            for (all_info.fields) |all_field| {
                if (mem.eql(u8, subset_field.name, all_field.name)) {
                    found = true;
                    break;
                }
            }
            if (!found)
                @compileError("Difference: subset axis not present in superset: " ++ subset_field.name);
        }
    }

    // Under the subset precondition, symmetric difference equals set difference.
    return Xor(EnumAll, EnumSubset);
}

/// Axis-keyed `usize` struct for `EnumAll \ EnumSubset`.
pub fn DifferenceAxesStruct(comptime EnumAll: type, comptime EnumSubset: type) type {
    return AxesStruct(meta.fieldNames(Difference(EnumAll, EnumSubset)));
}

/// Deduplicated union of the field names across the given axis types, in
/// first-seen order. `types` may be enums or shape structs (or a mix), since
/// `std.meta.fieldNames` reports field names for both.
pub fn unionOfAxisNames(comptime types: []const type) []const [:0]const u8 {
    comptime {
        var sum: usize = 0;
        for (types) |T| sum += meta.fieldNames(T).len;
        var all_names: [sum][:0]const u8 = undefined;
        var count: usize = 0;
        for (types) |T| {
            for (meta.fieldNames(T)) |name| {
                var found = false;
                for (all_names[0..count]) |existing|
                    if (mem.eql(u8, existing, name)) {
                        found = true;
                        break;
                    };
                if (!found) {
                    all_names[count] = name;
                    count += 1;
                }
            }
        }
        return all_names[0..count];
    }
}

// Pins the documented equivalence between our name-list factories and
// `std.enums.EnumFieldStruct`. The types are deliberately *not* compared with
// `==` (separate `@Struct` calls never dedupe, see the `AxesStructOf` header);
// we assert structural equivalence field-by-field plus matching defaults, which
// is what the codebase actually relies on.
test "factories match std.enums.EnumFieldStruct for enum-backed names" {
    const E = enum { i, j, k };
    const names = comptime meta.fieldNames(E);

    // Plain variant: AxesStructOf(fieldNames(E), T) ~ EnumFieldStruct(E, T, null).
    const Ours = AxesStructOf(names, u32);
    const Std = std.enums.EnumFieldStruct(E, u32, null);
    const ours_fields = @typeInfo(Ours).@"struct".fields;
    const std_fields = @typeInfo(Std).@"struct".fields;
    try std.testing.expectEqual(std_fields.len, ours_fields.len);
    inline for (ours_fields, std_fields) |of, sf| {
        try std.testing.expect(mem.eql(u8, of.name, sf.name));
        try std.testing.expectEqual(sf.type, of.type);
        try std.testing.expectEqual(sf.alignment, of.alignment);
    }
    try std.testing.expectEqual(@sizeOf(Std), @sizeOf(Ours));

    // Optional variant: AxesOptionalStructOf(fieldNames(E), T)
    // ~ EnumFieldStruct(E, ?T, null), including the per-field null default.
    const OursOpt = AxesOptionalStructOf(names, u32);
    const StdOpt = std.enums.EnumFieldStruct(E, ?u32, @as(?u32, null));
    const ours_opt = OursOpt{};
    const std_opt = StdOpt{};
    inline for (@typeInfo(OursOpt).@"struct".fields, @typeInfo(StdOpt).@"struct".fields) |of, sf| {
        try std.testing.expect(mem.eql(u8, of.name, sf.name));
        try std.testing.expectEqual(sf.type, of.type);
        try std.testing.expectEqual(@field(std_opt, sf.name), @field(ours_opt, of.name));
    }
}

test "unionOfAxisNames" {
    comptime {
        const IJK = enum { i, j, k };
        const JLI = enum { j, l, i };

        // First-seen order: i, j, k (from IJK), then l (the only new name in JLI).
        const expected = [_][:0]const u8{ "i", "j", "k", "l" };
        const actual = unionOfAxisNames(&.{ IJK, JLI });

        try std.testing.expectEqualDeep(&expected, actual);
    }
}

test "Difference" {
    const IJK = enum { i, j, k };
    const IJ = enum { i, j };
    const K = enum { k };

    const D = Difference(IJK, IJ);
    try std.testing.expectEqualDeep(meta.fieldNames(K), meta.fieldNames(D));

    if (false) {
        const KL = enum { k, l };
        _ = Difference(IJK, KL);
    }
}

test "DifferenceAxesStruct" {
    const IJK = enum { i, j, k };
    const IK = enum { i, k };

    const D = DifferenceAxesStruct(IJK, IK);
    const d: D = .{ .j = 3 };
    try std.testing.expectEqual(@as(usize, 3), d.j);
}
