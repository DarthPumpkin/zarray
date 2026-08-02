# conv2d hot-path optimization + general convolution ADR — Session 22

## Goal
Speed up the w-contiguous hot path of `conv2dSingleChannel` in `src/conv.zig`
(both `w` strides == 1), and plan how to extend convolution to multiple
channels, padding, and arbitrary rank.

## Progress log
- ✅ Established baseline with `zig build -Doptimize=ReleaseFast bench -- conv`.
  Hot path (`[all_row_major]`) was ~1.75 GMAC/s for 3x3, ~4.04 for 7x7. Root
  cause: `dotFast` reduced over the tiny kernel width (3/7), degenerating to
  scalar code.
- ✅ Optimization 1 — vectorize over output width (AXPY): for each kernel tap,
  broadcast-multiply-accumulate the shifted image row into the output row.
  Added `math.axpy`. Result ~7–9 GMAC/s. Checksums bit-identical.
- ✅ Compared `math.axpy` vs BLAS `cblas_saxpy` empirically (temporary swap).
  Hand-written inline version wins on short rows (call overhead + no inlining
  for the external symbol); BLAS ties/edges ahead only on the longest rows.
  Kept `math.axpy`.
- ✅ Optimization 2 — register/accumulator blocking: tile each output row into
  `lanes*4` columns, keep 4 SIMD accumulators in registers, run all taps, store
  once. Eliminates `k_h*k_w` read-modify-write passes over the output row and
  breaks the FP-add latency chain. Introduced `convRowContiguous`. Result
  ~10–17 GMAC/s. Per-element accumulation order preserved → checksums unchanged.
- ✅ Optimization 3 — FMA via `@mulAdd` (with an integer fallback helper `fma`).
  ~+15–20% on 3x3 (compute-bound); neutral on 7x7 (load-bound). Checksums stable
  to 6 digits.
- ✅ Optimization 4 — comptime kernel specialization: dispatcher sends square
  3x3/5x5/7x7 to `convRowContiguousImpl(Scalar, KH, KW, ...)` with kernel extents
  known at compile time; LLVM fully unrolls the tap loops. Big win for 3x3
  (→ ~20–23.5 GMAC/s); 7x7 unchanged (bound by L1 load throughput).
- ✅ Removed now-unused `math.axpy` after the register-blocked kernel superseded
  it (bench still uses BLAS `axpy`, untouched).
- ✅ Discussed general-convolution design. Concluded axis roles can be inferred
  from operand set membership across `{im, kernel, out}` (einsum-style):
  all-three = spatial, im+kernel = in-channel, kernel+out = out-channel,
  im+out = batch; single-operand axes are compile errors. This unifies channels
  and arbitrary rank into one engine. Padding kept as a separate composable op.
  Identified the grouped/depthwise limitation (an all-three axis that is diagonal,
  not windowed) needing an explicit annotation.
- ✅ Authored ADR-0002 (convolution via role inference over named axes) at
  `docs/adr/ADR-0002-convolution-role-inference.md`.

## Performance summary (`[all_row_major]`, GMAC/s)
| case | baseline | AXPY | reg-block | +FMA | +comptime sizes |
|---|---|---|---|---|---|
| 64x64, 3x3 | 1.74 | 7.23 | 10.05 | 11.68 | 20.34 |
| 128x128, 3x3 | 1.75 | 8.24 | 13.17 | 15.78 | 22.81 |
| 192x192, 3x3 | 1.76 | 8.44 | 14.41 | 17.37 | 23.55 |
| 192x192, 7x7 | 4.04 | 8.94 | 16.68 | 16.07 | 16.26 |

~13x for 3x3, ~4x for 7x7, allocation-free, checksums preserved. Also beats BLAS
`saxpy` on this workload.

## Files changed
- `src/conv.zig` (w-contiguous fast path rewritten: `convRowContiguous` dispatcher
  + `convRowContiguousImpl` register-blocked kernel with FMA and comptime kernel
  specialization; `fma` helper).
- `src/math.zig` (added then removed `axpy` — net no change).
- `docs/adr/ADR-0002-convolution-role-inference.md` (new).

## Validation run in this session
- ✅ `zig build test` (after each optimization)
- ✅ `zig build -Doptimize=ReleaseFast bench -- conv`
- ✅ `diagnostics` on `src/conv.zig` (clean)

## Likely next steps
1. Standalone composable `pad` op (spatial padding, any rank/channels).
2. Generic role-inference conv engine + comptime role/shape validation with rich
   `@compileError` messages.
3. Fast-path dispatch: reuse the spatial-contiguous kernel (NCHW-like) and add a
   channel-contiguous BLAS `gemm` path (NHWC-like).
4. Optional perf: fused padding, output-height/channel blocking to attack the 7x7
   load bottleneck.
5. Optional: grouped/depthwise via the reserved group-axis annotation.
