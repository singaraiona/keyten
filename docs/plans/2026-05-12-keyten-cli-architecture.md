# Keyten CLI — Stage 1: Tokio-driven TUI REPL

## Context

Build a tokio-driven terminal application on top of the keyten runtime: a rich
REPL with multi-line editing, K pretty-printed output, live progress display
during long evaluations, and Ctrl-C cancellation that interrupts the running
computation without exiting the program.

Three properties carry over from the engine and **must not regress**:

1. **Cancellation observed within one chunk** — Ctrl-C must reach the running
   kernel via `ctx.cancelled` and surface in the UI within one chunk-time
   (~6-60 µs for arithmetic kernels) plus one redraw.
2. **Progress visible during streaming ops** — the running kernel itself
   wakes the UI at chunk boundaries when progress has advanced enough to
   be worth painting. No wall-clock timer in the event loop.
3. **`RefObj` never crosses threads** — the runtime is single-threaded
   (`tokio::runtime::Builder::new_current_thread()`); eval is `spawn_local`'d
   on the same thread as the UI loop. The design carries over to v2's
   multi-threaded executor pool without architectural change.

## Decisions confirmed

| Axis | Choice |
|---|---|
| Workspace shape | Cargo workspace with `keyten/` (library) + `keyten-cli/` (binary) |
| TUI editor | `tui-textarea` widget rendered inside the ratatui frame |
| Parser placement | In the `keyten` library: `keyten/src/parse/`, `keyten/src/ast.rs`, `keyten/src/eval/` |
| First ship scope | MVP: parser + tree-walking eval + 3-pane TUI + progress + Ctrl-C cancel + history persistence + K pretty-print for atoms and typed vectors |
| Tokio runtime | `new_current_thread()` — single-threaded; eval lives on the UI thread, yields cooperatively |
| Async kernel surface | Library grows `*_async` variants of every dispatch entry that internally use `drive_async`; sync variants stay for tests and library users that don't want tokio |

## Workspace restructure

```
keyten/                                # workspace root (renamed from current crate)
├── Cargo.toml                         # [workspace]
├── keyten/                            # the runtime library (current contents move here)
│   ├── Cargo.toml
│   └── src/
│       └── …                          # existing modules
└── keyten-cli/                        # new binary crate
    ├── Cargo.toml
    └── src/
        ├── main.rs
        └── …
```

Mechanical migration:
- Create `keyten-workspace/Cargo.toml` with `[workspace] members = ["keyten", "keyten-cli"]`.
- Move existing `Cargo.toml` + `src/` + `tests/` + `examples/` + `Cargo.lock` into `keyten/`.
- Re-run `cargo test` from the workspace root to confirm nothing breaks.
- The git history is preserved via `git mv`.

## Library additions

### New modules in `keyten/`

```
keyten/src/
├── ast.rs                              # Expr, Stmt, Verb, Adverb, Atom enums
├── parse/
│   ├── mod.rs                          # pub use parse::parse
│   ├── lex.rs                          # tokeniser
│   └── grammar.rs                      # recursive-descent parser
├── eval/
│   ├── mod.rs                          # pub use eval::{eval, eval_async, Env, EvalErr}
│   ├── env.rs                          # Env: HashMap<Sym, RefObj> scoped variable map
│   └── tree.rs                         # tree-walking evaluator (sync + async)
└── …                                   # existing modules unchanged
```

### `ast::Expr` shape

```rust
// keyten/src/ast.rs
pub enum Expr {
    /// A bare literal that fits in one heap atom.
    AtomLit(AtomLit),
    /// A typed vector literal (sequence of homogeneous atoms).
    VecLit { kind: Kind, items: Vec<AtomLit> },
    /// Generic list `(a; b; c)` of arbitrary sub-expressions.
    ListLit(Vec<Expr>),
    /// Variable reference.
    Name(Sym),
    /// `name: expr`
    Assign(Sym, Box<Expr>),
    /// Dyadic verb application: `x V y`.
    Dyad { verb: VerbId, lhs: Box<Expr>, rhs: Box<Expr> },
    /// Monadic verb application: `V x`.
    Monad { verb: VerbId, arg: Box<Expr> },
    /// Adverb composition: `V/x` (over), `V\x` (scan).
    Adverb { adv: AdvId, verb: VerbId, arg: Box<Expr> },
    /// Multiple top-level statements separated by `;` — returns the last.
    Seq(Vec<Expr>),
}

pub enum AtomLit {
    Bool(bool),
    I64(i64),
    F64(f64),
    Char(u8),
    Sym(Sym),
    NullI64,
    NullF64,
    InfI64,
    InfF64,
}

pub enum AdvId { Over, Scan }    // Scan reserved for v1.1
```

### Parser scope for v1

The parser accepts exactly what v1 keyten can evaluate. Out of scope until
v1.1: dict / table literals, conditionals (`$[…]`), user functions, monadic
adverbs beyond `/`, named verbs.

| Construct | Example |
|---|---|
| Int / float / null literals | `42`, `3.14`, `0N`, `0n`, `0W`, `0w` |
| Char and string | `"a"`, `"hello"` |
| Symbol | `` `x `` |
| Typed vector | `1 2 3`, `1.5 2.5 3.5`, `` `a`b `c `` |
| Generic list | `(1; 2.0; "ab")` |
| Variable assignment | `x: 1 2 3` |
| Name reference | `x` |
| Dyadic verbs | `x + y`, `x - y`, `x * y`, `x % y` (right-associative per K convention) |
| Monadic prefix | `-x`, `+x` |
| `+/` adverb | `+/x`, `+/1 2 3` |
| Parenthesised grouping | `(1 + 2) * 3` |
| Multiple statements | `x: 1 2 3; +/x` |

Error reporting: every `Expr` carries a `Span { start, end }` so eval errors
can point back to the source. Parse errors are formatted with a caret line
beneath the offending span.

### Evaluator

```rust
// keyten/src/eval/env.rs
pub struct Env {
    bindings: HashMap<Sym, RefObj>,
}

impl Env {
    pub fn new() -> Self;
    pub fn lookup(&self, name: Sym) -> Option<RefObj>;       // returns a clone (rc bump)
    pub fn bind(&mut self, name: Sym, v: RefObj);
    pub fn names(&self) -> impl Iterator<Item = (Sym, &RefObj)>;
}

// keyten/src/eval/tree.rs
pub enum EvalErr {
    UndefinedName(Sym, Span),
    KernelErr(KernelErr, Span),
    TypeMismatch { expected: Kind, got: Kind, span: Span },
}

pub async fn eval_async(
    expr: &Expr,
    env: &mut Env,
    ctx: &Ctx<'_>,
) -> Result<RefObj, EvalErr>;

pub fn eval(expr: &Expr, env: &mut Env, ctx: &Ctx) -> Result<RefObj, EvalErr>;
```

`eval_async` recurses through `Box::pin(eval_async(...)).await` for sub-
expressions (the `async fn` recursion idiom) and reaches the kernel layer via
the new `dispatch_*_async` entries. Each kernel uses `drive_async` internally
so each chunk boundary is a yield point — the UI loop runs concurrently.

### Async kernel surface (library addition)

Every existing `dispatch_*` and `*_vec_vec` / `*_scalar_vec` / `*_vec_scalar`
entry gets an `_async` sibling that swaps `drive_sync` for `drive_async`.
Kernel state machines (`AddI64VecVec`, etc.) are unchanged — they're already
the right shape. Mechanical change, ~150 LOC.

```rust
pub async fn dispatch_plus_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> { … }
pub async fn dispatch_minus_async(...) -> Result<RefObj, KernelErr> { … }
// etc., one per op
pub async fn over_async(op: OpId, x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> { … }
```

Sync entries (`dispatch_plus` etc.) remain as-is. Tests against them stay; new
tests cover the async variants.

## CLI app structure

```
keyten-cli/src/
├── main.rs                 # tokio rt init, terminal init, top-level select! loop
├── app.rs                  # AppState; on_term / on_tick / on_eval_done / on_interrupt
├── format.rs               # RefObj → display String (K idiomatic)
├── history.rs              # persisted ring buffer at $XDG_STATE_HOME/keyten/history
├── sig.rs                  # SIGINT debounce, terminal restore on panic
└── tui/
    ├── mod.rs              # pub re-exports
    ├── layout.rs           # ratatui frame composition (output / input / status)
    ├── output.rs           # OutputBuffer: scrollback widget, line wrapping
    ├── input.rs            # tui-textarea wrapper with K-aware key handling
    ├── status.rs           # status / progress widget
    └── theme.rs            # colour palette (prompt, error, ok, dim, accent)
```

### `AppState` (single source of truth)

```rust
pub struct AppState {
    // Persistent environment.
    pub env: Env,

    // Output history.
    pub output: OutputBuffer,        // entries (prompt, formatted result, optional error)

    // Active evaluation, if any.
    pub eval: Option<ActiveEval>,

    // Editor.
    pub input: TextArea<'static>,
    pub history: History,            // up/down arrow ring

    // Last-tick instrumentation for the status bar.
    pub last_progress: u64,
    pub last_progress_at: Instant,
    pub throughput_eps: f64,         // smoothed elements/sec

    // Ctrl-C debounce — second within 500 ms exits the app.
    pub last_interrupt_at: Option<Instant>,

    // Render-dirty bit.
    pub dirty: bool,

    pub should_quit: bool,
}

pub struct ActiveEval {
    pub cancelled: Arc<AtomicBool>,
    pub progress:  Arc<AtomicU64>,
    pub started_at: Instant,
    pub source: String,              // original input, for the prompt line
    pub join: tokio::task::JoinHandle<EvalOutcome>,
}

pub enum EvalOutcome {
    Ok(RefObj),
    Err(EvalErr),
    Cancelled,
}
```

`cancelled` and `progress` are `Arc<Atomic*>` so the eval task and the UI loop
share them. (The `Ctx` constructor on the keyten side already accepts
`&AtomicBool` / `&AtomicU64`; the CLI uses `Arc` and dereferences when calling.)

## Event-loop topology — fully edge-triggered, no clock

The substrate is mio's readiness model (epoll / kqueue / IOCP under the OS hood),
exposed via tokio's runtime (which adds the executor, the timer wheel, and the
`AsyncRead`/`AsyncWrite` plumbing for `tokio::net`). Every wakeup is an OS-level
readiness event — terminal stdin, an `eventfd` backing `Notify`, a timerfd for
`tokio::time`, a socket fd for `tokio::net`. There is no internal poll loop and
no wall-clock timer the runtime keeps for its own sake.

```rust
// keyten-cli/src/main.rs
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut terminal = init_terminal()?;
    let mut app = AppState::new(load_history()?);
    let mut events = EventStream::new();
    let r = run(&mut app, &mut terminal, &mut events).await;
    restore_terminal(&mut terminal)?;
    save_history(&app.history)?;
    r
}

async fn run(app, term, events) -> Result<()> {
    loop {
        tokio::select! {
            // 1) terminal input (mio: stdin fd readiness)
            Some(Ok(ev)) = events.next()                       => app.on_term(ev),

            // 2) running kernel says "progress crossed stride; please redraw"
            //    (Notify is an eventfd under the hood)
            _ = app.render_sink.notify.notified()              => { /* nothing — fall through to draw */ }

            // 3) eval task completed
            Some(out) = app.poll_eval()                        => app.on_eval_done(out),

            // 4) SIGINT (signalfd / IOCP signal port)
            _ = tokio::signal::ctrl_c()                        => app.on_interrupt(),
        }
        terminal.draw(|f| app.render(f))?;
        if app.should_quit { break; }
    }
    Ok(())
}
```

`poll_eval` is a helper that `await`s the active `JoinHandle` only when there is
one (otherwise it returns a `Pending` future and that arm of the select! is
inert). Cancellation is implemented entirely through
`app.eval.cancelled.store(true)` — the eval task's `drive_async` notices at its
next chunk boundary and returns `Err(KernelErr::Cancelled)`.

### Kernel-driven UI wakes via `RenderSink`

The chunk loop is itself the heartbeat. At each chunk boundary the kernel
decides whether enough work has happened to merit a redraw, and if so, wakes
the UI task. No fixed-interval timer participates.

```rust
// keyten/src/render.rs (NEW, in the library so kernels can reach it)
pub struct RenderSink {
    pub notify: Arc<Notify>,
    pub last_notified_progress: AtomicU64,
    /// Notify when progress has advanced by at least this many elements.
    /// 0 disables intermediate notifications (only final notify fires).
    pub stride: AtomicU64,
}

// keyten/src/ctx.rs — `Ctx` gains one optional field
pub struct Ctx<'r> {
    pub runtime:     &'r Runtime,
    pub cancelled:   &'r AtomicBool,
    pub progress:    &'r AtomicU64,
    pub render:      Option<&'r RenderSink>,
    pub chunk_elems: usize,
}

// keyten/src/chunk.rs — `drive_async` becomes notification-aware
pub async fn drive_async<K: ChunkStep>(k: &mut K, ctx: &Ctx<'_>) -> Result<(), KernelErr> {
    while let Some(n) = k.step() {
        let p = ctx.progress.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
        if ctx.cancelled.load(Ordering::Relaxed)
            || ctx.runtime.global_cancel.load(Ordering::Relaxed)
        {
            return Err(KernelErr::Cancelled);
        }
        if let Some(sink) = ctx.render {
            let stride = sink.stride.load(Ordering::Relaxed);
            if stride > 0 {
                let last = sink.last_notified_progress.load(Ordering::Relaxed);
                if p.wrapping_sub(last) >= stride {
                    sink.last_notified_progress.store(p, Ordering::Relaxed);
                    sink.notify.notify_one();
                }
            }
        }
        YieldNow::new().await;
    }
    if let Some(sink) = ctx.render {
        sink.notify.notify_one();   // final notify on completion
    }
    Ok(())
}
```

Per-chunk overhead: two relaxed atomic loads + an integer compare. ~3 ns —
below the noise floor of even the smallest chunk.

### Stride is a property of the work, not of the clock

When the CLI submits an eval, it sets `stride` based on input size:

```rust
fn pick_stride(input_elements: u64) -> u64 {
    // Aim for ~20 visible updates over the whole run.
    // < ~1M elements: zero — single notify at completion, no spinner needed.
    // Otherwise: n / 20, clamped to [256K, 64M].
    if input_elements < 1_000_000 { 0 }
    else { (input_elements / 20).clamp(256 * 1024, 64 * 1024 * 1024) }
}
```

For ops with unknown total (e.g. `seq` of statements), the CLI starts at 1M
and doubles `stride` after each notify until the observed redraw rate falls
under ~20 Hz. ~5 lines of adaptive logic in `app.on_render_signal`.

This makes the UI's visible update rate **a property of the work**, not of a
wall clock. A 1 ms op produces zero intermediate notifications, completes in
one redraw. A 10 s op produces ~20 redraws over its lifetime, evenly spread
across the actual work. No "fast op produces a flicker storm, slow op feels
frozen" failure modes.

### How this generalises to timers, sockets, feeds (v1.x / v2)

Every event source we'll add — local timers, IPC connections, market-data
feeds — becomes one more `select!` arm. None of them poll. Each is an OS
readiness wake (timerfd, socket fd, eventfd) routed through mio and presented
as a future.

```rust
loop {
    tokio::select! {
        // v1 today
        Some(Ok(ev)) = events.next()                  => app.on_term(ev),
        Some(out)    = app.poll_eval()                => app.on_eval_done(out),
        _ = app.render_sink.notify.notified()         => { /* will redraw */ }
        _ = tokio::signal::ctrl_c()                   => app.on_interrupt(),

        // v1.x — scheduled tasks / timeouts
        _ = app.timers.next_fire()                    => app.on_timer_fire(),

        // v2 — IPC server (TcpListener accept loop)
        Some(conn) = app.ipc.next_conn()              => app.on_ipc_open(conn),
        Some(msg)  = app.ipc.next_message()           => app.on_ipc_msg(msg),

        // v2 — live data feed
        Some(tick) = app.market_feed.next_tick()      => app.on_market_tick(tick),
    }
    terminal.draw(|f| app.render(f))?;
}
```

When the executor flips to multi-threaded in v2, none of this changes
shape: `Notify`, mio `Waker`, atomic counters, channels — all are `Send +
Sync`. The kernel runs across worker threads, calls `notify_one()` at its
chunk boundaries, the UI task wakes on whichever thread it's currently
scheduled on, and the redraw happens. The `current_thread` runtime in v1 is a
configuration choice, not an architectural one.

### Why tokio, not mio directly

Tokio is mio plus the executor plus the timer wheel plus
`tokio::net`/`tokio::signal`/`tokio::sync`. Going to mio directly would mean
re-implementing those for the same wakeup discipline. For a runtime that will
eventually expose IPC and consume live feeds, that's a lot of well-trodden
ground we'd be re-walking. The architectural commitments (no polling, every
wake is OS-driven, every event source is a `select!` arm) are honoured either
way; tokio just hands us the upper layers pre-wired.

### Submitting an expression

```rust
fn spawn_eval(&mut self, source: String) {
    let cancelled = Arc::new(AtomicBool::new(false));
    let progress  = Arc::new(AtomicU64::new(0));
    let env = self.env.clone();      // shallow: clones refcounted RefObjs
    let c = cancelled.clone();
    let p = progress.clone();
    let join = tokio::task::spawn_local(async move {
        let parsed = parse(&source)?;
        let ctx = Ctx::new(&RUNTIME, &c, &p);
        let mut env = env;
        let result = eval_async(&parsed, &mut env, &ctx).await;
        // env may have new bindings; we return it for merging on the UI thread
        EvalOutcome::from(result, env)
    });
    self.eval = Some(ActiveEval { cancelled, progress, started_at: Instant::now(), source, join });
    self.dirty = true;
}
```

### Ctrl-C handling (debounced)

```rust
fn on_interrupt(&mut self) {
    let now = Instant::now();
    if let Some(prev) = self.last_interrupt_at {
        if now.duration_since(prev) < Duration::from_millis(500) && self.eval.is_none() {
            self.should_quit = true;
            return;
        }
    }
    self.last_interrupt_at = Some(now);
    if let Some(active) = &self.eval {
        active.cancelled.store(true, Ordering::Relaxed);
    }
}
```

First Ctrl-C: cancels the active eval (or no-op if idle).
Second Ctrl-C within 500 ms while idle: quits.

## TUI layout

```
┌─ keyten ──────────────────────────────────────────────────────┐
│ k) x: !1e6                                                    │  output area
│ 0 1 2 3 4 5 6 7 8 …                                           │  (scrollable;
│                                                               │   PgUp/PgDn)
│ k) +/x                                                        │
│ 499999500000                                                  │
│                                                               │
│ k) +/!1e10                                                    │
│ ⠋ 47% (4.7B / 10B)  2.1s  1.84G elems/s    [^C] cancel        │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│ k) _                                                          │  input area
│                                                               │  (tui-textarea;
│                                                               │   multi-line)
├───────────────────────────────────────────────────────────────┤
│ ready  •  rss 24 MiB  •  vars 1  •  history 142  •  ? help    │  status line
└───────────────────────────────────────────────────────────────┘
```

Vertical layout: `Constraint::Min(0)` for output (takes remaining), `Length(3)`
for input (grows to 5 on overflow with a scrollbar), `Length(1)` for status.

### Output rendering

`OutputBuffer` is a `VecDeque<OutEntry>` capped at, say, 1000 entries
(configurable). Each entry:

```rust
pub struct OutEntry {
    pub prompt: String,       // "k) " by convention
    pub source: String,       // user input
    pub body: Body,
}
pub enum Body {
    Ok { value_display: Vec<Line<'static>>, took: Duration },
    Err { message: String, source_span: Option<Span> },
    Active,                   // entry pinned for the running eval; shows live progress
}
```

Lines are pre-rendered (with ratatui `Line` + `Span` to carry colour) and
re-wrapped only on terminal resize.

### Input rendering

`tui-textarea` already provides:
- Multi-line editing with Up/Down/Left/Right.
- Backspace, Delete, paste (bracketed if available).
- Cursor positioning, selection, copy/cut, undo.

We add on top:
- **K syntax tinting** (post-MVP): a token-aware highlighter that colours
  verbs/adverbs distinctly. Off by default in v1; flag-enabled.
- **History recall**: Up at empty buffer pulls previous entry; Down moves
  toward newer entries; original buffer restored when Down past newest.
- **Enter behaviour**: submits if the bracket stack is balanced; otherwise
  inserts a newline. Shift-Enter always inserts a newline. Alt-Enter always
  submits.
- **Tab**: inserts two spaces (no completion in MVP).

### Status / progress widget

Two states:

- **Idle:** `ready  •  rss 24 MiB  •  vars 1  •  history 142  •  ? help`.
- **Active:** `<spinner> NN% (done / total)  Ts elapsed  XG elems/s    [^C] cancel`.
  - `done` comes from `app.eval.progress.load(Relaxed)`.
  - `total` is the input length: we get it from the expression's outermost
    operand size before submitting (cheap — read `len()` from the input
    RefObj). When the total is unknown (e.g. a `seq` of statements), we show
    only elapsed and throughput.
  - Throughput = exponential moving average over the last ~1 s of progress
    deltas.

RSS is read via `procfs`-equivalent (`/proc/self/statm` on Linux, `task_info`
on macOS). One platform-specific helper, gated on cfg.

## K pretty-printing

```rust
// keyten-cli/src/format.rs
pub struct PrintOpts {
    pub max_width: u16,         // terminal columns
    pub max_lines: u16,         // vertical budget for one value
    pub max_elems_per_line: usize,
}

pub fn format(r: &RefObj, opts: &PrintOpts) -> Vec<Line<'static>>;
```

- **Atom:** bare value. `42`, `3.14`, `` `abc ``. Nulls render as `0N` / `0n` /
  `` ` `` (empty sym). Inf as `0W` / `0w`.
- **Typed vector:** space-separated values; switch to multi-line if it overflows
  width × max_lines. Print first/last N elements with `…` ellipsis beyond.
- **Generic list:** `(a; b; c)` with recursive formatting of children.
- **Composite:** out-of-scope for v1 (no semantics yet); placeholder
  `<dict ...>` / `<table ...>`.
- **HAS_NULLS attr** affects rendering: when set, nulls show as `0N` rather
  than the raw sentinel.

For `+/!1e10`-style atoms, the result is a single i64 — trivial.

## History persistence

```
$XDG_STATE_HOME/keyten/history       (default: ~/.local/state/keyten/history)
```

One line per entry, newline-separated, UTF-8. Capped at 10 000 lines (rolling).
Loaded on startup, saved on shutdown. Up/Down arrows traverse it from newest
to oldest. Survives crashes via "save every N submissions" (N=10 by default).

## Stage-1 CLI deliverables

- [ ] Workspace `Cargo.toml` with `keyten`, `keyten-cli` members.
- [ ] Move existing keyten code into `keyten/`.
- [ ] `keyten/src/ast.rs` (Expr, Stmt, …, Span).
- [ ] `keyten/src/parse/` (lex + grammar) covering the listed v1 syntax.
- [ ] `keyten/src/eval/` (Env, eval, eval_async).
- [ ] Async kernel variants for `+ − × ÷` and `+/`.
- [ ] `keyten-cli/Cargo.toml` (tokio current_thread, ratatui, crossterm, tui-textarea, dirs).
- [ ] Terminal init/restore with panic hook.
- [ ] `AppState` + main `select!` loop.
- [ ] Input editor (tui-textarea) with history recall and Enter heuristics.
- [ ] Output buffer with K pretty-printer.
- [ ] Status / progress widget (idle + active states, throughput EMA).
- [ ] Ctrl-C debounce: cancel active eval, second press while idle quits.
- [ ] History persistence (XDG state path, rolling cap).
- [ ] Parse-error display with caret pointer.
- [ ] Eval-error display with span highlight in the source line.

## Verification

- `cargo test -p keyten`: all existing tests pass after move; parser and eval
  unit tests added.
- `cargo test -p keyten`: async kernel variants produce identical results to
  sync variants on randomised inputs.
- `cargo test -p keyten-cli`: pretty-printer round-trips a representative set
  of atoms / vectors / nulls; history file is loaded and saved correctly.
- Smoke test (manual or scripted): `x: !1e8 ; +/x` prints the correct sum,
  status bar shows progress, throughput ≥ 1 G elems/s in release.
- Smoke test (manual): start `+/!1e10`, press Ctrl-C within the first second,
  observe cancellation within one chunk and the prompt returning to ready.
- `cargo run -p keyten-cli` boots into the TUI, accepts input, displays output.

## Out of scope for CLI v1 (deferred to v1.1+)

- Syntax highlighting in the input editor (a flag-gated extension).
- Completion (variables and verbs).
- Side panels (variable inspector, in-app help).
- Conditionals `$[…]` and user-defined functions in the parser.
- Dict and Table literals and pretty-printing.
- Composite verbs in the eval surface.
- A `:script` / `:run` mode for non-interactive file execution.
- Multiple windows or splits inside the TUI.
- Mouse interaction (selection, click-to-position).

## Out of scope (still): runtime concerns deferred to keyten v2

These are inherited from the engine's plan and remain deferred:
- Parallel kernel execution (driver-level swap).
- File-backed mmap loader (the `IS_EXTERNAL` bit is wired; no loader yet).
- Out-of-core algorithms (chunk source interface ready).
- Parse-time monomorphisation for scalar-heavy paths.
- Threading (`RefObj: !Send + !Sync`; CLI lives within this constraint).
