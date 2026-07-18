# ADR-0001: Named-axis core with strict semantics and explicit memory behavior

- **Status:** Accepted (for current milestone)
- **Date:** 2026-07-18

## Context

`zarray` aims to become foundational infrastructure for scientific computing in Zig, analogous to NumPy's ecosystem role, but with different priorities:

1. **Correctness first** (prevent API-induced bugs)
2. **Efficiency second** (memory + compute)
3. Convenience/ergonomics later

The current core already centers on:
- compile-time axis names (enum-based)
- index model: `shape + strides + offset`
- zero-copy view transforms where possible
- explicit allocator use for allocating operations

## Decision

### 1) Axis model and typing
- Keep axis identity as **compile-time enum labels**.
- Keep core array/index types free of runtime axis metadata.
- If metadata is needed later, add it in a separate wrapper layer (not in base ND array types).

### 2) Strictness policy
- Prefer **compile-time errors whenever possible**.
- If a violation cannot be checked at compile time, fail fast at runtime (panic or explicit error union according to API class).
- Default behavior for core transformations is strict/fail-fast, not convenience-oriented best effort.

### 3) View vs allocation semantics
- Preserve the explicit split between non-allocating view operations and allocating operations.
- Allocator parameter is the primary signal that an operation allocates.
- When both allocating and non-allocating variants exist with similar names, use a naming disambiguator (e.g. `*Alloc` suffix) to avoid ambiguity.

### 4) Mutability/read-only model
- Keep support for both mutable and read-only backing buffers (`[]T` and `[]const T`).
- Continue with distinct array types/factories for now (`NamedArray` and `NamedArrayConst`) because the mutability distinction is explicit and maps cleanly to Zig semantics.
- Revisit unification only if it improves clarity without weakening compile-time guarantees.

### 5) Broadcasting and reshape
- Broadcasting policy is **explicit only** (no implicit broadcasting).
- `reshape` should only succeed as a view when layout constraints permit it; otherwise require an explicit copy path (e.g. `toContiguous` then reshape).

### 6) Numerical backends
- No backend abstraction layer at this stage.
- Keep external numerical libraries as explicit bindings with explicit scalar support constraints.

## Non-goals (current milestone)

The following are explicitly out of scope for now:

1. Implicit broadcasting
2. Runtime axis metadata in core array/index types
3. Automatic backend dispatch/abstraction across BLAS/TBLIS/etc.
4. Sparse array infrastructure
5. Autodiff framework integration

> Note: GPU kernels are not a current milestone goal, but may be explored later.

## Consequences

### Positive
- Strong prevention of axis and layout errors
- Predictable memory behavior
- Easier reasoning about aliasing/view semantics
- Clear extension boundary between core arrays and optional advanced layers

### Tradeoffs
- Some operations are more verbose than convenience-first libraries
- API surface may include paired methods (view vs alloc)
- Users may need explicit conversion steps (`toContiguous`) in more situations

## Enforceable invariants (design guardrails)

1. **No hidden allocation:** core methods without allocator arguments must not allocate.
2. **Axis safety:** cross-axis misuse should fail at compile time when representable in types.
3. **View purity:** view transforms must preserve backing buffer identity.
4. **Contiguous copy explicitness:** any operation requiring relocation must require explicit allocation path.
5. **Broadcast explicitness:** no implicit expansion in arithmetic/reduction APIs.
6. **Backend explicitness:** library binding modules must declare supported scalar types and fail clearly on unsupported ones.

## Near-term roadmap priorities

1. Docs/examples/tests hardening (make contracts explicit and verifiable)
2. Reduction ergonomics over selected subsets of axes (convenience wrappers that preserve strict semantics)

---

If future milestones prioritize ergonomics, revisit this ADR intentionally rather than drifting defaults.