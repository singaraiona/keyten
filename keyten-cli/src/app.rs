//! Main application loop. One `select!`, four event sources, no clock.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind};
use futures::StreamExt;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use tokio::task::{JoinHandle, LocalSet};

use keyten::{Ctx, Env, EvalErr, KernelErr, RefObj, RenderSink, RUNTIME};

use crate::format::{format, PrintOpts};
use crate::history::History;
use crate::tui::input::{InputArea, InputOutcome};
use crate::tui::output::{Body, OutEntry, OutputBuffer};
use crate::tui::status::{Progress, StatusBar, Throughput, SPINNER};
use crate::tui::{layout, theme, Term};

pub async fn run(term: &mut Term) -> Result<()> {
    let local = LocalSet::new();
    local.run_until(async {
        let mut app = App::new();
        let res = app.event_loop(term).await;
        app.history.save().ok();
        res
    }).await
}

struct App {
    env: Env,
    output: OutputBuffer,
    input: InputArea,
    history: History,

    eval: Option<ActiveEval>,
    throughput: Throughput,
    spinner_tick: usize,

    last_interrupt_at: Option<Instant>,
    should_quit: bool,
}

struct ActiveEval {
    cancelled: Arc<AtomicBool>,
    progress: Arc<AtomicU64>,
    sink: Arc<RenderSink>,
    started_at: Instant,
    source: String,
    join: JoinHandle<EvalOutcome>,
}

enum EvalOutcome {
    Ok(RefObj, Env),
    Err(EvalErr, Env),
    Kernel(KernelErr, Env),
}

impl App {
    fn new() -> Self {
        Self {
            env: Env::new(),
            output: OutputBuffer::default(),
            input: InputArea::default(),
            history: History::load(),
            eval: None,
            throughput: Throughput::new(),
            spinner_tick: 0,
            last_interrupt_at: None,
            should_quit: false,
        }
    }

    async fn event_loop(&mut self, term: &mut Term) -> Result<()> {
        let mut events = EventStream::new();
        // Initial draw.
        self.draw(term)?;

        loop {
            // Snapshot the sink (cheap Arc clone) so the notify-await arm can
            // borrow it without touching self.eval (which the join-await arm
            // borrows mutably).
            let sink = self.eval.as_ref().map(|a| a.sink.clone());

            tokio::select! {
                Some(Ok(ev)) = events.next() => {
                    self.on_terminal(ev);
                }
                outcome = wait_join(&mut self.eval) => {
                    self.on_eval_done(outcome).await;
                }
                _ = wait_notify(sink.as_ref()) => {
                    self.on_render_notify();
                }
                _ = tokio::signal::ctrl_c() => {
                    self.on_interrupt();
                }
            }
            self.draw(term)?;
            if self.should_quit {
                break;
            }
        }
        Ok(())
    }

    fn draw(&mut self, term: &mut Term) -> Result<()> {
        term.draw(|f| {
            let area = f.area();
            let (output_area, input_area, status_area) = layout::split(area);
            self.render_output(output_area, f.buffer_mut());
            self.input.render(input_area, f.buffer_mut());
            self.render_status(status_area, f.buffer_mut());
        })?;
        Ok(())
    }

    fn render_output(&self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let busy = self.busy_status_line();
        self.output.render(area, buf, busy);
    }

    fn busy_status_line(&self) -> Option<Line<'static>> {
        let active = self.eval.as_ref()?;
        let elapsed = active.started_at.elapsed();
        let processed = active.progress.load(Ordering::Relaxed);
        let frame = SPINNER[self.spinner_tick % SPINNER.len()];
        let s = format!(
            "  {frame} {} elems   {:.1}s   {} elems/s    [^C] cancel",
            fmt_count(processed),
            elapsed.as_secs_f64(),
            fmt_count(self.throughput.rate_eps as u64),
        );
        Some(Line::from(Span::styled(s, theme::STATUS_BUSY)))
    }

    fn render_status(&self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        use ratatui::widgets::Widget;
        let bar = if let Some(active) = &self.eval {
            let p = active.progress.load(Ordering::Relaxed);
            StatusBar {
                idle_text: "",
                progress: Some(Progress {
                    elapsed: active.started_at.elapsed(),
                    processed: p,
                    total: None,
                    throughput_eps: self.throughput.rate_eps,
                    spinner_idx: self.spinner_tick,
                }),
            }
        } else {
            let idle = format!(
                "ready  \u{2022}  vars {}  \u{2022}  history {}  \u{2022}  Ctrl-C \u{00D7}2 to quit",
                self.env.len(),
                self.history.len(),
            );
            // Store in `self` would need mutability; render with a static string built here.
            let leaked: &'static str = Box::leak(idle.into_boxed_str());
            StatusBar {
                idle_text: leaked,
                progress: None,
            }
        };
        bar.render(area, buf);
    }

    fn on_terminal(&mut self, ev: Event) {
        if let Event::Key(key) = ev {
            // Ignore key-release events on Windows-style backends.
            if key.kind != KeyEventKind::Press && key.kind != KeyEventKind::Repeat {
                return;
            }
            // Ctrl-C is delivered separately by tokio::signal::ctrl_c, but most
            // terminals also emit it as a Ctrl-C key event. Treat both equivalently.
            if matches!(key.code, KeyCode::Char('c'))
                && key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL)
            {
                self.on_interrupt();
                return;
            }
            // Eval running? Only Ctrl-C and scroll are meaningful.
            if self.eval.is_some() {
                if matches!(key.code, KeyCode::PageUp) {
                    self.output.scroll_up(4);
                } else if matches!(key.code, KeyCode::PageDown) {
                    self.output.scroll_down(4);
                }
                return;
            }
            match self.input.handle_key(key, &self.history) {
                InputOutcome::Submit(text) => {
                    self.history.push(text.clone());
                    self.spawn_eval(text);
                }
                InputOutcome::ScrollUp => self.output.scroll_up(4),
                InputOutcome::ScrollDown => self.output.scroll_down(4),
                _ => {}
            }
        }
    }

    fn spawn_eval(&mut self, source: String) {
        // Push the input as an "Active" entry so the user sees it immediately.
        self.output.push(OutEntry {
            prompt: "k) ".into(),
            source: source.clone(),
            body: Body::Active,
        });

        let cancelled = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(AtomicU64::new(0));
        let sink = Arc::new(RenderSink::with_stride(pick_stride_default()));

        // Move ownership of the env into the task; on completion the task
        // returns it back so we can re-install with new bindings.
        let env = std::mem::take(&mut self.env);
        let cancelled_clone = cancelled.clone();
        let progress_clone = progress.clone();
        let sink_clone = sink.clone();
        let src_for_task = source.clone();
        let join = tokio::task::spawn_local(async move {
            let mut env = env;
            let mut ctx = Ctx::new(&RUNTIME, &cancelled_clone, &progress_clone);
            ctx.render = Some(&sink_clone);
            let parsed = match keyten::parse(&src_for_task) {
                Ok(e) => e,
                Err(e) => {
                    return EvalOutcome::Err(
                        EvalErr::Type {
                            msg: format!("parse error: {}", e.msg),
                            span: e.span,
                        },
                        env,
                    );
                }
            };
            match keyten::eval_async(&parsed, &mut env, &ctx).await {
                Ok(v) => EvalOutcome::Ok(v, env),
                Err(EvalErr::Kernel { err, .. }) => EvalOutcome::Kernel(err, env),
                Err(e) => EvalOutcome::Err(e, env),
            }
        });

        self.throughput.reset();
        self.eval = Some(ActiveEval {
            cancelled,
            progress,
            sink,
            started_at: Instant::now(),
            source,
            join,
        });
    }

    fn on_render_notify(&mut self) {
        if let Some(active) = self.eval.as_ref() {
            let p = active.progress.load(Ordering::Relaxed);
            self.throughput.observe(p);
            self.spinner_tick = self.spinner_tick.wrapping_add(1);
        }
    }

    async fn on_eval_done(&mut self, outcome: Option<EvalOutcome>) {
        let Some(active) = self.eval.take() else {
            return;
        };
        let outcome = match outcome {
            Some(o) => o,
            None => {
                if let Some(entry) = self.output.last_mut() {
                    entry.body = Body::Err {
                        message: "eval task crashed".into(),
                        caret: None,
                    };
                }
                return;
            }
        };
        let final_progress = active.progress.load(Ordering::Relaxed);
        self.throughput.observe(final_progress);
        let started_at = active.started_at;
        let source = active.source.clone();
        drop(active); // release the JoinHandle etc.
        let _ = started_at;
        let _ = source;

        match outcome {
            EvalOutcome::Ok(value, env) => {
                self.env = env;
                let lines = format(&value, &PrintOpts::default());
                let took = started_at.elapsed();
                if let Some(entry) = self.output.last_mut() {
                    entry.body = Body::Ok { lines, took };
                }
            }
            EvalOutcome::Err(err, env) => {
                self.env = env;
                let (message, caret) = format_eval_err(err, &source);
                if let Some(entry) = self.output.last_mut() {
                    entry.body = Body::Err { message, caret };
                }
            }
            EvalOutcome::Kernel(err, env) => {
                self.env = env;
                let msg = match err {
                    KernelErr::Cancelled => "cancelled".into(),
                    KernelErr::Oom => "out of memory".into(),
                    KernelErr::Type => "type error".into(),
                    KernelErr::Shape => "shape error".into(),
                };
                if let Some(entry) = self.output.last_mut() {
                    entry.body = Body::Err {
                        message: msg,
                        caret: None,
                    };
                }
            }
        }
    }

    fn on_interrupt(&mut self) {
        let now = Instant::now();
        if let Some(active) = &self.eval {
            active.cancelled.store(true, Ordering::Relaxed);
            self.last_interrupt_at = Some(now);
            return;
        }
        // Idle: two presses within 500 ms quit.
        if let Some(prev) = self.last_interrupt_at {
            if now.duration_since(prev) < Duration::from_millis(500) {
                self.should_quit = true;
                return;
            }
        }
        self.last_interrupt_at = Some(now);
    }
}

/// Wait for the active eval's JoinHandle to resolve, without consuming it
/// from the Option. Returns `Some(outcome)` on success, `None` if the task
/// panicked, and never resolves when there is no active eval.
async fn wait_join(eval: &mut Option<ActiveEval>) -> Option<EvalOutcome> {
    match eval.as_mut() {
        Some(active) => {
            let res = std::future::poll_fn(|cx| Pin::new(&mut active.join).poll(cx)).await;
            match res {
                Ok(o) => Some(o),
                Err(_) => None,
            }
        }
        None => {
            std::future::pending::<()>().await;
            unreachable!()
        }
    }
}

/// Wait on the active eval's RenderSink notification. Resolves never when
/// there is no active eval.
async fn wait_notify(sink: Option<&Arc<RenderSink>>) {
    match sink {
        Some(s) => s.notify.notified().await,
        None => std::future::pending().await,
    }
}

fn pick_stride_default() -> u64 {
    // Default: signal every ~1M elements processed. Adaptive tuning lands
    // when we observe how fast the UI is actually being woken.
    1 << 20
}

fn fmt_count(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else if n < 1_000_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    }
}

fn format_eval_err(err: EvalErr, _source: &str) -> (String, Option<String>) {
    match err {
        EvalErr::UndefinedName { name, .. } => {
            let bytes = name.0.to_le_bytes();
            let s: String = bytes.iter().copied().take_while(|b| *b != 0).map(|b| b as char).collect();
            (format!("undefined name `{s}`"), None)
        }
        EvalErr::Kernel { err, .. } => (format!("kernel: {err:?}"), None),
        EvalErr::Type { msg, .. } => (msg, None),
        EvalErr::Empty => ("empty expression".into(), None),
    }
}
