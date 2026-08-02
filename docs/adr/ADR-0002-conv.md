# ADR-0002: Convolution via role inference over named axes

- **Status:** Proposed
- **Date:** 2026-08-02

## Context

The current `conv2dSingleChannel` (`src/conv.zig`) is a fixed 2D, single-channel,
valid-mode (no padding) convolution over the `HW` axis enum, with an optimized
register-blocked hot path for the fully w-contiguous layout.

We want to grow this into a general convolution that supports:

1. **Multiple channels** (in-channels reduced away, out-channels produced),
2. **Padding** (same/full/arbitrary), and
3. **Arbitrary rank** (1D/2D/3D/ND spatial).

The core design question is how axes map to convolution roles. Rather than a
fixed positional layout (NCHW, NHWC, …), we exploit the library's compile-time
named-axis model (ADR-0001): an axis's role can be **inferred from which of the
three operands (`im`, `kernel`, `out`) it appears in**, matched by shared enum
tag. This is the einsum/tensor-contraction formulation of convolution, with the
twist that the axis shared by all three operands carries a shifted (windowed)
coupling rather than a shared index.

The comptime set-algebra needed already exists (`Difference`, `conformAxes`,
`keepOnly`, `mergeAxes`, `splitAxis`, tag-name matching over `fieldNames`).

## Decision

### 1) Axis roles are inferred from operand set membership

For each distinct axis tag, its presence across `{im, kernel, out}` determines
its role. There are seven non-empty subsets; four are valid roles and three are
compile errors:

| present in… | role | semantics | shape constraint |
|---|---|---|---|
| im, kernel, out | **spatial** | sliding-window coupling `im[s*stride + k*dilation]` | `im = (out - 1)*stride + (kernel - 1)*dilation + 1` |
| im, kernel | **in-channel** | contraction (summed away) | `im == kernel` |
| kernel, out | **out-channel** | free index sourced from kernel | `kernel == out` |
| im, out | **batch** | shared free index (pass-through) | `im == out` |
| im only | — | source axis with no sink | `@compileError` |
| kernel only | — | weight axis with no meaning | `@compileError` |
| out only | — | output axis with no source | `@compileError` |

For valid (unpadded, unit-stride, undilated) convolution the spatial constraint
reduces to `im = out + kernel - 1`.

Because axes are shared enum tags, operand-to-operand correspondence (which `h`
in the kernel maps to which `h` in the image) is automatic; there is no
positional guessing.

### 2) Channels and arbitrary rank are the same mechanism

Under role inference, "multiple channels" and "arbitrary rank" are not separate
features — both are simply additional tags in the operand sets:

- N spatial axes → N tags present in all three operands.
- Channels → tags in the im+kernel (in-channel) and kernel+out (out-channel) sets.

A single generic engine therefore delivers channels and arbitrary rank together.
Degenerate cases fall out for free: a spatial extent of 1 (1x1 conv) becomes a
pure channel contraction (gemm); zero spatial axes become a plain tensordot.

### 3) Padding is a separate, composable operation

Convolution stays **valid-mode only**. Same/full/arbitrary padding is provided by
a standalone `pad` operation (allocating; takes an allocator per ADR-0001's
view-vs-alloc rule) that produces a padded array which is then fed to `conv`.

- Keeps the hot kernel free of boundary logic.
- Avoids interior/border domain-splitting, whose region count grows as `2^N`
  with spatial rank.
- Fused zero-padding inside a fast kernel is deferred to a later performance
  pass, once the kernel's final shape is settled.

### 4) Grouped/depthwise convolution requires an explicit annotation

Set membership alone cannot express grouped/depthwise convolution: there, a
channel axis is present in all three operands but behaves as a diagonal/batched
index, not a sliding window. The default for an all-three axis is **spatial**.
Grouped/depthwise support is provided by an explicit, comptime list of axis tags
to treat as diagonal ("group") axes. The signature reserves room for this even
if grouped conv is implemented later.

### 5) Generic engine first, then profile-driven specialization

- Implement a correct generic engine over any layout using the existing
  iterators: outer loops over free axes (batch, out-channel, output-spatial),
  inner reduction over (in-channel, kernel-spatial).
- Add fast paths by detecting favorable stride patterns and dispatching:
  - **spatial-contiguous** (channels-first / NCHW-like): reuse the existing
    register-blocked width kernel, with in-channels as an outer accumulation.
  - **channel-contiguous** (channels-last / NHWC-like): reduction over in-channel
    is contiguous; map onto BLAS `gemm` (1x1 or im2col) and vectorize over
    out-channel.
- Do not bless a single layout up front. Let profiling on real shapes decide
  which specializations earn their complexity.

### 6) Strict, informative compile-time validation

All role inference and shape checks happen at comptime where representable
(ADR-0001). Because spatial correspondence depends on shared tags, a naming
mismatch (e.g. kernel axis `kh` vs image axis `h`) silently reclassifies axes
(orphan error, or `h` becoming batch). Mitigate with a rich `@compileError` that
lists the inferred role of every axis and flags orphan axes.

## Non-goals (this milestone)

1. Padding fused into the conv kernel (composable `pad` only for now).
2. Automatic layout selection/transposition (caller chooses layout).
3. Grouped/depthwise implementation (annotation reserved, not required).
4. GPU kernels.

## Consequences

### Positive
- One generic engine covers channels and arbitrary rank.
- Layout-agnostic API; correctness independent of NCHW/NHWC/etc.
- Roles and shapes validated at compile time.
- Composable padding keeps the hot kernel simple.
- Reuses existing named-axis comptime set-algebra.

### Tradeoffs
- Naming discipline is load-bearing (spatial axes must share tags); mitigated by
  strong compile errors.
- Grouped/depthwise needs an escape hatch beyond pure inference.
- Composable padding materializes a padded copy until fused padding lands.
- Peak performance requires layout-specific specializations, added incrementally.

## Enforceable invariants (design guardrails)

1. **Role determinism:** an axis's role is a pure function of its `{im, kernel,
   out}` membership (plus the explicit group-axis annotation).
2. **Orphan rejection:** axes present in only one operand are compile errors.
3. **Shape checking:** per-role shape constraints are checked at compile time
   when representable, else fail fast at runtime.
4. **Valid-mode kernel:** the conv kernel never performs boundary handling;
   padding is external.
5. **No hidden allocation:** the conv engine allocates only when given an
   allocator (e.g. an im2col fast path); `pad` is the explicit allocating step.

## Roadmap

1. Standalone `pad` operation (spatial-axis padding, any rank/channels).
2. Generic role-inference conv engine + comptime validation and error messages.
3. Fast-path dispatch: spatial-contiguous kernel (reuse current) and
   channel-contiguous gemm path.
4. Optional perf pass: fused padding, additional blocking (e.g. over output
   height/channels).
5. Optional: grouped/depthwise via the reserved group-axis annotation.

---

If a future milestone needs padding fused into the kernel or automatic layout
selection, revisit this ADR intentionally rather than drifting defaults.
