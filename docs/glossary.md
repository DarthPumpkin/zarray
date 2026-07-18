# zarray Glossary (v1)

## Core modeling terms

- **Axis**
  A compile-time enum tag identifying one logical dimension (e.g. `i`, `j`, `k`).

- **Rank**
  Number of axes in an array.

- **Shape**
  Per-axis extents (sizes), keyed by axis name.

- **Stride**
  Per-axis linear step in the underlying 1D buffer for incrementing an index along that axis.

- **Offset**
  Base linear index into the backing buffer for a view.

- **Index model**
  The tuple `shape + strides + offset` that maps axis-keyed indices to buffer positions.

- **Linearization**
  Conversion of an axis-keyed index into a single buffer index using strides and offset.

## Memory and semantics terms

- **View (zero-copy)**
  A transformed array/index that references the same backing buffer.

- **Allocating operation**
  Any operation that creates new storage; in `zarray` this is signaled by requiring an allocator.

- **In-place operation**
  Mutation of existing storage without allocation.

- **Contiguous layout**
  Layout where elements are arranged in default forward order for the shape (compatible with a flat forward slice).

- **Non-contiguous view**
  A view whose strides/offset do not permit exposure as a single forward 1D slice.

## Axis-transform terms

- **Conform axes**
  Re-express an array in a target axis set, allowing insertion of size-1 axes and removal (squeeze) of size-1 axes under strict rules.

- **Squeeze axis**
  Remove an axis of extent 1.

- **Rename axes**
  Rebind axis identities via an explicit mapping.

- **Merge axes**
  Combine multiple axes into one when stride/layout compatibility allows a zero-copy view.

- **Kept axes**
  Axes that remain in the output of a subset-axis operation (e.g. partial reduction).

- **Reduced axes**
  Axes eliminated by a reduction operation (summed/maxed/etc.).

- **Reshape (view-safe)**
  Change shape without moving data only when layout constraints allow; otherwise requires explicit copy path.

## Policy terms

- **Strict/fail-fast**
  Prefer compile-time rejection where possible; otherwise fail immediately at runtime for contract violations.

- **Explicit broadcasting**
  Broadcasting must be requested via explicit API; never inferred implicitly.

- **Backend binding**
  Module-level wrapper over a specific numeric library (e.g. TBLIS), with explicit supported scalar types.

- **No hidden allocation**
  Contract that core APIs without allocator arguments must not allocate.
