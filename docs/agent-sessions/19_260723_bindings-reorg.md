# Bindings Reorganization — Session 19 (completed)

## Goal
Reorganize `src` bindings so that:
1. GSL/TBLIS/BLAS/LAPACK live under `src/bindings`.
2. Bindings are reachable via a `bindings` namespace (e.g. `bindings.lapack`).
3. `src/accelerate.zig` becomes `blas.zig` (BLAS-only).

## Progress log
- ✅ Audited current `src` layout and identified binding files.
- ✅ Audited current references/imports (`root.zig`, `mlp_example.zig`, `build.zig`, GSL internal module imports).
- ✅ Moved bindings into `src/bindings` (`blas`, `lapack`, `tblis`, `gsl/*`).
- ✅ Updated root namespace: new `bindings.*` exports in `src/root.zig`; `libs` now aliases `bindings`.
- ✅ Updated path-sensitive imports in moved files (`blas`, `lapack`, `tblis`) and updated `build.zig` C source paths.
- ✅ Validation run completed: `zig build test && echo PASS` (`PASS` observed).
- ✅ Diagnostics refreshed: no project errors/warnings.
- ✅ Follow-up: flattened `src/bindings/blas.zig` so BLAS declarations are module-level (removed outer `blas` struct), and switched `src/root.zig` to `pub const blas = @import("bindings/blas.zig");`.
- ✅ Kept BLAS tests green after flattening (adjusted internal axis helper name and test-local `IJ` names to avoid top-level shadowing under Zig 0.16).
- ✅ Follow-up complete: removed `const blas = @This()` and in-file `blas.` references from `src/bindings/blas.zig`.
- ✅ Validation rerun after alias removal: `zig build test && echo PASS` (`PASS` observed).
- ✅ Added a file-level `//!` module docstring to `src/bindings/blas.zig`, analogous to other binding modules.
- ✅ Validation rerun after BLAS doc update: `zig build test && echo PASS` (`PASS` observed).
- ✅ Exhaustive BLAS symbol audit vs Accelerate `cblas.h` (CLT/Xcode SDKs): initially 144/148 (`cblas_csrot`, `cblas_zdrot`, `cblas_xerbla`, `cblas_errprn` omitted).
- ✅ Implemented the BLAS omissions in `src/bindings/blas.zig`:
  - `rot_complex` now uses `cblas_csrot` / `cblas_zdrot` semantics.
  - Added thin wrappers for `cblas_errprn` and `cblas_xerbla`.
  - Added/updated `rot_complex` tests and removed the stale "currently broken" TODO block.
- ✅ Tightened `rot_complex` API to strict BLAS semantics (`rot: GivensRotationReal(T)` only); removed temporary compatibility acceptance of `GivensRotationComplex` and updated tests/docs accordingly.
- ✅ Follow-up cleanup: removed `rotg_complex` and `GivensRotationComplex` entirely (no apply-path in this module by design).
- ✅ Updated BLAS file-level docstring to document current omissions precisely (`cblas_crotg`, `cblas_zrotg`).
- ✅ Re-audited symbol coverage after removal: `blas.zig` now references 146/148 `cblas_*` symbols from Accelerate `cblas.h` (missing exactly `cblas_crotg`, `cblas_zrotg`).
- ✅ Validation rerun after BLAS API cleanup: `zig build test && echo PASS` (`PASS` observed).
- ✅ Portability-oriented tightening: removed `cblas_errprn`/`cblas_xerbla` wrappers and removed `rot_complex` from core `bindings.blas` to keep core strictly mandated-BLAS/CBLAS only.
- ✅ Updated BLAS omissions docs accordingly; re-audit now reports core `blas.zig` uses 142/148 `cblas_*` (missing exactly `cblas_{crotg,zrotg,csrot,zdrot,errprn,xerbla}`).
- ✅ Added backend-extensions namespace: `bindings.blas_ext` with `bindings.blas_ext.accelerate` backend module.
- ✅ Moved non-portable Accelerate/ATLAS BLAS entry points into `src/bindings/blas_ext/accelerate.zig`; retained extension wrappers for `cblas_{csrot,zdrot,errprn,xerbla}` and intentionally left `cblas_{crotg,zrotg}` unwrapped (no corresponding complex-`s` apply API exposed).
- ✅ Consistency cleanup in `blas_ext.accelerate`: no intermediate symbol-by-symbol helpers; extension API is exposed via typed dispatchers (`rot_complex`, plus `errprn`/`xerbla`).
- ✅ Validation rerun after strict-core split: `zig build test && echo PASS` (`PASS` observed); diagnostics clean.

## Planned mapping
- `src/accelerate.zig` -> `src/bindings/blas.zig`
- `src/lapack.zig` -> `src/bindings/lapack/lapack.zig`
- `src/lapack_shim.c` -> `src/bindings/lapack/lapack_shim.c`
- `src/lapack_shim.h` -> `src/bindings/lapack/lapack_shim.h`
- `src/tblis.zig` -> `src/bindings/tblis/tblis.zig`
- `src/tblis_zig.c` -> `src/bindings/tblis/tblis_zig.c`
- `src/gsl*.zig` -> `src/bindings/gsl/`

## Notes
- GSL now lives as a grouped subdirectory under `src/bindings/gsl/`.
- `libs` is kept as a compatibility alias to `bindings` (`pub const libs = bindings;`).
