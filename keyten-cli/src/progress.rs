//! Progress line printer.
//!
//! Renders a single line of `\r…\x1b[K` (overprint + clear-to-end) on stderr
//! whenever the kernel's `RenderSink::notify` fires. Tracks an exponential
//! moving average of elements/sec.

use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use keyten::RenderSink;
use nu_ansi_term::{Color, Style};

fn stderr_is_tty() -> bool {
    static IS_TTY: OnceLock<bool> = OnceLock::new();
    *IS_TTY.get_or_init(|| {
        #[cfg(unix)]
        unsafe {
            extern "C" {
                fn isatty(fd: i32) -> i32;
            }
            isatty(2) != 0
        }
        #[cfg(not(unix))]
        {
            true
        }
    })
}

pub const SPINNER: &[&str] = &[
    "\u{2807}", "\u{2811}", "\u{2819}", "\u{2839}", "\u{2879}", "\u{28F8}", "\u{28D8}", "\u{28D4}",
    "\u{28C6}", "\u{2847}",
];

pub struct ProgressPrinter {
    started_at: Instant,
    last_progress: u64,
    last_instant: Instant,
    rate_eps: f64,
    frame: usize,
}

impl ProgressPrinter {
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            last_progress: 0,
            last_instant: Instant::now(),
            rate_eps: 0.0,
            frame: 0,
        }
    }

    /// Print one progress line based on the latest `progress` counter value.
    pub fn tick(&mut self, processed: u64) {
        if !stderr_is_tty() {
            return;
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_instant).as_secs_f64();
        if dt > 0.0 {
            let delta = processed.saturating_sub(self.last_progress) as f64;
            let inst = delta / dt;
            self.rate_eps = if self.rate_eps == 0.0 {
                inst
            } else {
                0.3 * inst + 0.7 * self.rate_eps
            };
            self.last_progress = processed;
            self.last_instant = now;
        }
        self.frame = self.frame.wrapping_add(1);
        let spinner = SPINNER[self.frame % SPINNER.len()];
        let elapsed = now.duration_since(self.started_at).as_secs_f64();
        let line = format!(
            "  {sp} {n} elems  {e:.1}s  {r}/s   {hint}",
            sp = Style::new().fg(Color::Yellow).bold().paint(spinner),
            n = fmt_count(processed),
            e = elapsed,
            r = fmt_count(self.rate_eps as u64),
            hint = Style::new().fg(Color::Cyan).paint("[^C] cancel"),
        );
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\r\x1b[K{line}");
        let _ = stderr.flush();
    }

    pub fn clear() {
        if !stderr_is_tty() {
            return;
        }
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\r\x1b[K");
        let _ = stderr.flush();
    }
}

/// Async loop: await on `sink.notify`, read `progress`, render once per wake.
pub async fn run_progress_loop(sink: Arc<RenderSink>, progress: Arc<AtomicU64>) {
    let mut printer = ProgressPrinter::new();
    loop {
        sink.notify.notified().await;
        let p = progress.load(Ordering::Relaxed);
        printer.tick(p);
    }
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
