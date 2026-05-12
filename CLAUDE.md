# keyten — project orientation for contributors and agents

This file is loaded automatically by Claude Code (and similar agents) when
working in this repo. Read top-to-bottom; the **performance mandate** at the
top is non-negotiable and shapes every decision below it.

---

## Performance mandate (non-negotiable)

Performance is mandatory in keyten, not optional. Every design decision and
code change must be evaluated against **five concerns simultaneously**, not
picked off one at a time:

1. **Cache layout** — 16-byte alignment for SIMD; working sets sized for L1/L2/L3
   (the `kernels::*_CHUNK` constants exist for this reason); no false sharing
   across cores; `CachePadded` where shared mutable state crosses thread
   boundaries.
2. **Chunks** — every kernel is a `ChunkStep` state machine with bounded
   per-step memory and a per-chunk cancellation/progress observation point.
   Never "all-at-once" allocation or processing.
3. **Parallelizable** — every primitive must have a `parallel_for_each_mut` or
   `parallel_reduce` path, gated on `Runtime.parallel_enabled()` and
   `PARALLEL_THRESHOLD`. Sequential-only is technical debt that compounds
   upward — once a primitive isn't parallel, nothing above it can be either.
4. **Streaming** — don't materialise intermediate buffers when avoidable.
   Per-chunk `madvise_dontneed_slice` bounds RSS on long folds. Consider
   fusion when an obvious closed form exists (e.g. `(sum (til N))` →
   `N*(N-1)/2`).
5. **Vectorization** — SIMD via the `simd` module (scalar loops the
   autovectorizer turns into `Simd<T, N>` reductions). Hand-written
   non-SIMD-friendly scalar loops over hot data are a regression.

### Why this is mandatory

keyten is the substrate for a vector-database engine. The basic blocks
established now compound for years. v2 Stage 2 revealed an **8× perf gap**
hiding behind two trivial omissions: `til` wasn't parallelised, and the REPL
never enabled parallel by default. Both individually invisible; both fixed
in one commit (`2a5c80a`). Multiply that across N kernels and you have a
codebase that's permanently 5× slower than it should be, fixable only by a
top-to-bottom refactor.

Bench-driven development is the discipline that prevents this. See
`benches/bench_parallel.rs` and the median numbers committed alongside
material changes.

### How this applies in practice

- **New kernel?** Build the `ChunkStep` first, wire the parallel branch via
  `parallel_for_each_mut` or `parallel_reduce` in the **same PR**.
  Sequential-only is "incomplete," not "land it later."
- **New value type?** Ask: 16-byte aligned? splittable into per-worker
  sub-slices? streaming-consumable? autovectorisable on a hot loop?
- **Reviewing a PR?** "Doing it the obvious way" carries a perf cost that
  must be explicit and justified. Silent regressions accumulate.
- **Commit messages** for material perf changes should include median ns/elem
  and worker-count sweep numbers vs the prior baseline.
- **When in doubt**, `cargo bench -p keyten --bench bench_parallel` and
  compare against the recent baselines (`2a5c80a` for rayforce parity,
  `c879fd0` for parallel_reduce).

---

## Project layout

Cargo workspace with two crates:

```
keyten/        runtime library  — object layout, kernels, parser, eval, parallel
keyten-cli/    binary `keyten`  — reedline REPL, formatter, banner, sys-cmds
```

Both inherit `[workspace.package]` (edition 2021, Rust 1.80+, dual MIT/Apache).

### Key modules (`keyten/src/`)

- **`obj.rs`** — `Obj` header (8 bytes: meta, attr, kind, _resv, rc as
  `AtomicU32`), `RefObj` (transparent `NonNull<Obj>`, `Send` but not `Sync`).
- **`alloc.rs`** — atom freelist (per-thread TLS), vector heap-or-mmap path
  (`mmap_threshold` ~1 MiB), atomic-rc clone/drop.
- **`chunk.rs`** — `ChunkStep` trait, `drive_sync` (loop + cancel observation)
  and `drive_async` (same + per-chunk yield).
- **`parallel/`** — `parallel_for_each_mut`, `parallel_reduce`,
  `partition::balanced`. `std::thread::scope` per kernel for now;
  persistent-pool upgrade is a known Stage 2.1 item.
- **`kernels/`** — `plus`, `minus`, `times`, `div`, `til`. Each kernel is
  a `ChunkStep` impl plus async dispatch entries (`_async`).
- **`adverb.rs`** — `+/` (over) for I64 and F64 via `parallel_reduce`.
- **`ctx.rs`** — `Ctx<'r>`: runtime ref, cancellation flag, progress counter,
  optional render sink, chunk-size override. All fields are `Sync` refs.
- **`runtime.rs`** — process-wide `Runtime` (atomic mmap threshold, global
  cancel, parallel flag, worker count). Per-thread atom-cell freelist via
  `thread_local!`.
- **`eval/tree.rs`** — tree-walking evaluator, both sync and async surfaces.
- **`parse/`** — lexer + grammar.
- **`ast.rs`** — `Expr` enum, `Span`, `AtomLit`, `AdvId`.

### Architecture documents

Design decisions live in `docs/plans/`. Reading order for a new contributor:

1. `2026-05-12-keyten-cli-architecture.md` — v1's design, with three
   carry-forward properties (cancellation-within-one-chunk,
   kernel-driven UI wakes, `Ctx` shape). Some sections are now historical
   (`RefObj: !Send + !Sync` was relaxed in v2 stage 0); the architectural
   properties still hold.
2. `2026-05-12-lambda-v1-1.md` — Lambda v1.1 design (first-class functions,
   K-style `{x+y}` syntax with auto-detected implicit `x`/`y`/`z`, bracket
   apply + juxtaposition, scope-chain env, no closures until v1.2).
   **Prerequisite for v3 reactor.**

### Test layout

- `keyten/src/*` — unit tests inline under `#[cfg(test)] mod tests`.
- `keyten/tests/` — integration tests by topic: `layout.rs`,
  `parse_eval.rs`, `progress.rs`, `refcount.rs`, `parallel.rs`.
- `keyten/benches/` — `bench_rc.rs` (refcount overhead, the Stage 0
  baseline), `bench_parallel.rs` (vec+vec and +/ across worker counts
  and input sizes).

---

## Conventions

### Commits

- One logical change per commit; bench numbers in the body for perf changes.
- Subject line in imperative mood, present tense, ≤72 chars.
- Body explains *why*, not what (the diff explains what).
- For v2-stage work, prefix with `v2 stage N:` so the history is navigable.

### Code style

- Default to no comments. Add one only when *why* is non-obvious (hidden
  constraint, subtle invariant, workaround for a specific bug). Never
  explain *what* the code does — names and structure should make it
  obvious.
- Cancellation, progress, and the optional render sink are observed at
  *every* chunk boundary, never less often. The cost is ~3 ns and the
  property is "ctrl-C within one chunk," which is a hard guarantee.
- `unsafe` is used at three levels: object layout reads/writes, allocator
  routing, and SIMD intrinsics. Document the safety contract above each
  `unsafe fn`.

### Branch and PR flow

- master is the default branch (renamed from main implicitly).
- License is dual MIT / Apache-2.0 (LICENSE-MIT and LICENSE-APACHE at root).
- Author: Anton Kundenko (`anton.kundenko@gmail.com`).

---

## Quick start

```bash
# REPL (parallel on by default; `\p 0` to disable)
cargo run -p keyten-cli --release

# Library tests
cargo test --workspace

# Benches (release; `--no-run` to just check it compiles)
cargo bench -p keyten --bench bench_parallel

# TSan run on the parallel paths (needs nightly + rust-src)
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test \
    --target x86_64-unknown-linux-gnu -Zbuild-std
```

In the REPL:

```
𝕜 +/!100000000           / sum of 0..1e8, parallel by default
4999999950000000   (37 ms)

𝕜 \p 0                   / disable parallel for A/B comparison
parallel: off

𝕜 \t +/!100000000
4999999950000000   (243 ms)

𝕜 \p 1                   / re-enable
parallel: on   workers: 28
```

---

## Reference projects

Sibling K-family vector engines on the same machine, useful for performance
comparisons:

- `../rayforce/rayforce` — production K-style vector engine. Run with
  `-t 1 -c 0` for per-stage timing. Maintains parity ±5% on `+/!1e8` after
  v2 commit `2a5c80a`.
- `../theplatform/kernel/o/` — earlier-generation vector engine with a
  reactor crate (`reactor/`) implementing join-calculus reactions. **Use as
  conceptual reference for the v3 reactor design, not implementation
  source — we target tokio + tokio_stream for the plumbing instead of
  hand-rolling.**
