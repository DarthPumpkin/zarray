# LAPACK — Session 14 (const-correctness of the input-preserving surface)

**Status:** **Complete.** Makes every LAPACK routine that does not write to a
matrix argument take `NamedArrayConst` instead of `NamedArray`, so read-only
views and `[]const` buffers can be passed and the type system enforces the
"input-preserving" contract the docstrings already promised.

**Tests (latest run):**
- `zig test src/lapack.zig src/lapack_shim.c -framework Accelerate -lc` →
  **All 147 tests pass** (unchanged count; call sites migrated).
- `zig build test` → **passes** (whole project).
- `diagnostics` on `src/lapack.zig` → clean.

---

## 1. Motivation

The maintainer noticed the input-preserving routines took `NamedArray`, not
`NamedArrayConst`, even though they never write to the matrix. That was a
file-wide house convention (uniform `NamedArray` avoids `.asConst()` at call
sites) but it (a) understated the const-correctness the docstrings promised and
(b) prevented callers from passing genuinely read-only views (`NamedArrayConst`
or `[]const T`-backed arrays). This pass fixes it.

## 2. What changed (signatures)

Matrix params the routine only *reads* are now `NamedArrayConst`:

| Routine | Now-const param(s) | Mechanism |
|---|---|---|
| `det`, `qr`, `eigSym`, `eig`, `svd`, `eigVectors`, `svdVectors` | `a` | copy-first (`toContiguous`) — no body change |
| `lstsq`, `lstsqSvd` | `a` (`b` stays mutable — it receives X) | copy-first |
| `eigSymGen` | `a`, `b` | copy-first |
| `eigSymVectors`, `eigSymGenVectors` | `a` (and `b`) | `describeConst` (read-only) |
| `luSolve` | `a_lu` (reads LU factors) | `describeConst` → `@constCast` at FFI call |
| `choleskySolve` | `a_chol` (reads Cholesky factor) | `describeConst` → `@constCast` at FFI call |

Unchanged (they legitimately write their matrix, so stay `NamedArray`):
`solve`, `lu`, `inv`, `cholesky`, `detInplace`, `lstsqInplace`, `lstsqSvdInplace`,
`qrInplace`, `eigSymInplace`, `eigInplace`, `svdInplace`, `eigVectorsInplace`,
`svdVectorsInplace`, `eigSymGenInplace`. (`b`/`ipiv` outputs stay mutable too.)

## 3. Handing const inputs to `describe` (the one non-trivial bit)

The copy-first routines needed no body change (`NamedArrayConst.toContiguous`
returns a mutable `NamedArray`, so the `*Inplace` core still gets a writable
copy). But four routines call the private `describe()` on the *original* input,
and the original `describe` returned a mutable `[*]T` (`@ptrCast` of `arr.at(...)`,
which is `*Scalar`) — and `@ptrCast` cannot drop `const`.

The fix is to let `const` ride the descriptor all the way to the FFI boundary,
where LAPACK's non-`const` C ABI forces the `@constCast` anyway (its Fortran
prototypes type even read-only matrices as plain non-`const` pointers, e.g.
`getrs`'s factored `A` is `__CLPK_real *`). Concretely:

- `Descriptor(T)` is now `DescriptorOf([*]T)`; a sibling `DescriptorConst(T)` =
  `DescriptorOf([*]const T)` carries a read-only pointer.
- The layout/dimension math is factored into `describeGeom` (reads only the
  *index*, no buffer), shared by both variants — no body duplication.
- `describe` = `describeGeom` + mutable base pointer; new `describeConst` =
  `describeGeom` + `[*]const T` base pointer, taking a `NamedArrayConst`.
- `readElem` takes `am: anytype` (it only reads), so it accepts either
  descriptor. `toColMajorSquare` stays `Descriptor(T)` — it transposes in place
  and is only ever called on `*Inplace` (mutable) inputs.

The four routines then split cleanly:
- `eigSymVectors`/`eigSymGenVectors` — `describeConst` and copy via `readElem`;
  **no `@constCast` at all**.
- `luSolve`/`choleskySolve` — `describeConst`, then `@constCast(am.ptr)` at the
  single `getrs`/`potrs` call (next to the pre-existing `ipiv` cast), where the
  C signature demands a mutable pointer for an argument it only reads.

This keeps the const-drop co-located with the ABI mismatch that necessitates it,
rather than fabricating a mutable `NamedArray` over a `const` buffer up front.
(Aside — GSL, also bound in this project, *is* const-correct at the API level
(`const gsl_vector *`), yet `constVectorViewOf` still `@constCast`s the borrowed
view's `.data` at the boundary. Same principle: cast at the edge, not early.)

## 4. Call sites

`NamedArray` and `NamedArrayConst` are distinct generic structs with no implicit
coercion, so ~40 test call sites gained `.asConst()` on the now-const argument(s).
No production callers exist outside `src/lapack.zig` (grep confirmed), so only the
in-file tests changed. Test count is unchanged (147); this is a signature/const
migration, not new behavior.

## 5. Files changed
- **Modified:** `src/lapack.zig` — import `NamedArrayConst`; parameterize the
  descriptor (`DescriptorOf`/`Descriptor`/`DescriptorConst`); extract
  `describeGeom`; add `describeConst`; `readElem` now `anytype`; 14 signature
  changes; `describeConst` wiring in the 4 describe-the-original routines
  (with `@constCast` at the `getrs`/`potrs` calls); ~40 test call sites gained
  `.asConst()`.
- No shim or `build.zig` changes.

## 6. Notes / follow-ups
- Docstrings' `Input: … left unmodified` lines are now also enforced by the type
  system; left as-is (still accurate).
- This does not touch `NamedArray`/`NamedArrayConst` themselves (no implicit
  coercion added) — that's a `named_array.zig` design question out of scope here.
- Resolved in Session 15: the `view.zig` adapter extraction (the shared
  `analyze2d` geometry kernel now backs both `describe` and `Blas2d`).
