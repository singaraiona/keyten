# keyten

> a streaming array language

A K-inspired array language with a runtime written in Rust. Evaluation is
**streaming and chunked** — long computations advance in small fixed-size
chunks, so `Ctrl-C` reaches the running kernel within one chunk-time
(roughly 6–60 µs for arithmetic kernels) and the REPL stays responsive even
mid-fold. The runtime is single-threaded on a `tokio` current-thread
executor; the REPL is built on [`reedline`](https://github.com/nushell/reedline).

## Why keyten?

<!-- TODO: your voice — 3–6 lines.
     What you're exploring, who it's for, what it isn't.
     Examples of framing to consider:
       - research-flavored ("an experiment in streaming K-style evaluation
         with cooperative cancellation as a first-class property")
       - practical ("a small K dialect with a nice REPL, useful for X")
     Delete this comment once filled in. -->

## Status

Early. The current vertical slice covers parser, tree-walking eval, atom and
typed-vector pretty-printing, the primitives `+ − × ÷`, the `+/` (sum) adverb,
the monadic `!` (til / iota), `Ctrl-C` cancellation, history persistence, and
K-style system commands. Not yet: dictionaries, tables, IO, user-defined
verbs, multi-threaded execution.

## Crate layout

```
keyten/        runtime library  — object layout, kernels, parser, eval
keyten-cli/    binary `keyten`  — reedline REPL, formatter, banner, sys-cmds
```

The library has no `tokio` runtime dependency by default; the CLI owns the
runtime. Sync entry points exist alongside the `*_async` ones for embedders
that don't want `tokio`.

## Build & run

Rust **1.80+** is required (edition 2021).

```bash
# launch the REPL
cargo run -p keyten-cli --release

# library tests + CLI tests
cargo test --workspace

# minimal embedding example
cargo run -p keyten --example calc
```

## A short REPL session

```
𝕜 1 + 2
3

𝕜 +/!1000000
499999500000

𝕜 \t +/!10000000
4999999950000  (12.4 ms)

𝕜 x: !5
0 1 2 3 4

𝕜 \v
x

𝕜 \h
  \t expr   time evaluation of expr
  \v        list bound variables
  \h \?     help
  \\        quit

𝕜 \\
```

`Ctrl-C` during a long fold cancels the kernel without exiting the REPL.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work, as defined in the Apache-2.0 license,
shall be dual-licensed as above, without any additional terms or conditions.
