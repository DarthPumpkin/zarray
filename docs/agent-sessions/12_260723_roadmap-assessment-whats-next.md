# Roadmap Assessment — Session 12 (what's next after LAPACK?)

**Status:** **Advisory / no code changes.** This session is a strategy pass: the
maintainer asked, after the session-10 `*Inplace` rename, whether the LAPACK
bindings are in a good place, whether to bind more of LAPACK, or whether to
switch focus — measured against the original session-1 brainstorming.

**Method:** worked backward through the LAPACK chain (1 → 2 → 4 → 6 → 9 → 10) and
cross-read the parallel GSL track (3, 5, 7, 8) plus `src/root.zig` and `Readme.md`.

---

## 1. Where the LAPACK track actually stands

Every open question from the session-1 roadmap is now closed:

| Q | Topic | Status |
|---|---|---|
| Q1 | Eigen/singular **vectors** | ✅ session 2 |
| Q2 | Row-major `qr`/`lstsq` | ✅ session 2 |
| Q3 | Result-axis consistency (reuse input labels + shared inner name) | ✅ session 4 |
| Q4 | Input-preserving conveniences + ownership docs | ✅ session 4 |
| Q5 | Scratch-sizing helpers (Option A) | ✅ session 9, trimmed in 10 |
| Q6 | Complex decompositions via C shim | ✅ session 9 |
| — | `*Inplace` rename (mutation-based naming) | ✅ session 10 |

The dense **workhorse surface is complete**: LU / Cholesky / `lstsq` / `qr` /
`eig`/`eigSym`/`svd` (values **and** vectors), real **and** complex,
layout-transparent, input-preserving by default (`*Inplace` opt-out),
signature-checked complex via `lapack_shim.c`, **134 tests** green, clean
diagnostics.

**Verdict: the LAPACK bindings are in a very good place.** Session 10 was the last
*ergonomics* pass on a feature-complete surface; there is no obvious "next Q" left
in the original queue.

---

## 2. Should we bind more LAPACK?

Mostly **no** — consistent with the maintainer's bottom-up philosophy ("draft the
workhorse surface per-library, abstract/expand later"). What remains unbound is
the specialization tail, whose value drops off fast:

- **Genuinely high-value, if any:** the **generalized eigenproblem**
  (`sygv`/`hegv`, `ggev` — solving `A x = λ B x`; ubiquitous in PCA / GLS /
  vibration modes / quantum). This is the one gap a scientific user is realistically
  likely to hit that the current surface **cannot express at all**.
- **Second tier:** **rank-deficient least squares** (`gelsd`/`gelsy`). Today's
  `lstsq` uses `gels`, which assumes full rank; session 10 itself noted the
  rank-deficient `error.Singular` path is untested — because the routine can't
  really handle it. A `gelsd`-backed variant would be a real *capability*, not a
  convenience.
- **Skip unless demanded:** banded/packed/tridiagonal storage solvers
  (`gbsv`/`gtsv`/`pbsv`), Schur (`gees`/`gges`), condition estimation (`gecon`),
  iterative refinement (`gerfs`), pivoted/rank-revealing QR (`geqp3`), Sylvester
  (`trsyl`). These need new storage adapters or serve narrow audiences — classic
  diminishing returns.

**Recommendation:** declare the dense workhorse surface **done and freeze it**;
treat generalized-eigenproblem + rank-deficient-`lstsq` as an *optional,
demand-driven* follow-up rather than default next work.

---

## 3. Parallel GSL track — state (for context)

The GSL track (a different agent lineage) has advanced far in parallel:

- ✅ `gsl_sf` (special functions, session 5), `gsl_fft` (session 3),
  `rstat`/`movstat` (running & moving-window stats, session 7).
- ✅ Session 8 non-callback batch: `interp`, `histogram`, `poly`, `filter`,
  `qrng` — all done. Whole project ~**474 tests** green.
- **Remaining GSL frontier:** the **callback-driven families** (integration,
  roots, minimization, ODE, multifit, monte, deriv, Chebyshev) behind a shared
  `gsl_function` bridge; plus `sort`/`permutation`/`combination`/`multiset`; plus
  a planned extraction of `rand`/`stats`/`rstat`/`movstat` out of `gsl.zig` into
  their own files.

---

## 4. What to do next (recommendation: switch focus)

Comparing against the **original session-1 brainstorming**, two roadmap items are
now ripe and higher-leverage than more LAPACK:

### 4.1 The deferred `view.zig` extraction — its trigger condition is now met
Session 1 said to extract the shared "NamedArray → column-major matrix" adapter
*"once a third consumer makes the right abstraction clear."* That third consumer
now exists. The pattern is duplicated across:
- `Blas2d`/`Blas2dMut` in `src/accelerate.zig`,
- `describe`/`wrapMat` in `src/lapack.zig`,
- the generic `constVectorViewOf`/`mutVectorViewOf` GSL vector-view helpers
  (session 8, D12).

Every LAPACK session deliberately kept `describe` and `Blas2d` mechanically
similar *"so a future extraction is a delete-duplication job, not a
reconcile-conventions job."* This is the intended payoff moment. Most
self-contained option; clear scope.

### 4.2 The GSL callback-family design pass — the big untapped frontier
Session 1 flagged integration/roots/minimization/ODE as needing *"a
Zig-idiomatic callback convention designed once."* Session 8 confirms this is the
main remaining GSL boundary (everything non-callback is done). This is where the
most **new capability per unit effort** now lives — but it needs a design decision
first (the `gsl_function` bridge: how to thread a Zig closure + error through a C
callback), like the design-only sessions 6/8. Highest strategic value.

### 4.3 Cheap LAPACK cleanup (optional, ~1 hour)
The four low-priority test gaps session 10 deferred, if you want to fully close
out LAPACK before freezing:
- `lstsq` rank-deficient path (`error.Singular`),
- the layout-error returns (`error.NotColumnMajor` / `error.RhsNotColumnMajor`),
- an `inv` round-trip (`A·A⁻¹ ≈ I`),
- full-mode **rectangular** SVD.

---

## 5. Bottom line

- **LAPACK bindings:** complete and healthy — **freeze the dense workhorse
  surface.**
- **More LAPACK:** only the **generalized eigenproblem** (`sygv`/`ggev`) and
  **rank-deficient `lstsq`** (`gelsd`) are worth it, and only on demand.
- **Switch focus.** The roadmap's own trigger conditions now point elsewhere:
  (a) do the `view.zig` adapter extraction now that a third consumer exists,
  and/or (b) open the GSL callback-convention design pass — the last major unbound
  frontier.

**Suggested pick:** the GSL callback design pass unblocks the most future
capability, but starts as a design/brainstorm (à la sessions 6/8). If a concrete
consolidation win is preferred first, the `view.zig` extraction is the safer,
more mechanical choice.

---

## 6. Coordination notes
- No source files changed this session. Only this report was added.
- The `view.zig` extraction would touch `src/accelerate.zig`, `src/lapack.zig`,
  and `src/gsl.zig` (shared with other agents) — coordinate before starting.
