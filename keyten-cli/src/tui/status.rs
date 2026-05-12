//! Status / progress line at the bottom of the frame.

use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::tui::theme;

/// Braille spinner frames. Ten frames; one rotation per ten redraws.
pub const SPINNER: &[&str] = &[
    "\u{2807}", "\u{2811}", "\u{2819}", "\u{2839}", "\u{2879}", "\u{28F8}", "\u{28D8}", "\u{28D4}",
    "\u{28C6}", "\u{2847}",
];

pub struct StatusBar<'a> {
    pub idle_text: &'a str,
    pub progress: Option<Progress>,
}

pub struct Progress {
    pub elapsed: Duration,
    pub processed: u64,
    pub total: Option<u64>,
    pub throughput_eps: f64,
    pub spinner_idx: usize,
}

impl<'a> Widget for StatusBar<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let line = match self.progress {
            None => Line::from(Span::styled(self.idle_text.to_string(), theme::STATUS_IDLE)),
            Some(p) => {
                let frame = SPINNER[p.spinner_idx % SPINNER.len()];
                let mut spans = vec![Span::styled(format!("{frame} "), theme::STATUS_BUSY)];
                if let Some(total) = p.total {
                    let pct = if total > 0 {
                        (100.0 * (p.processed as f64) / (total as f64)).min(100.0)
                    } else {
                        100.0
                    };
                    spans.push(Span::styled(
                        format!("{pct:.0}% ({} / {})", fmt_count(p.processed), fmt_count(total)),
                        theme::STATUS_BUSY,
                    ));
                } else {
                    spans.push(Span::styled(
                        format!("{} elems", fmt_count(p.processed)),
                        theme::STATUS_BUSY,
                    ));
                }
                spans.push(Span::raw("  "));
                spans.push(Span::styled(
                    format!("{:.1}s", p.elapsed.as_secs_f64()),
                    theme::DIM,
                ));
                spans.push(Span::raw("  "));
                spans.push(Span::styled(
                    format!("{} elems/s", fmt_rate(p.throughput_eps)),
                    theme::DIM,
                ));
                spans.push(Span::raw("   "));
                spans.push(Span::styled("[\u{005E}C] cancel", theme::ACCENT));
                Line::from(spans)
            }
        };
        Paragraph::new(line).render(area, buf);
    }
}

/// Exponential moving average of throughput.
pub struct Throughput {
    last_progress: u64,
    last_instant: Instant,
    pub rate_eps: f64,
}

impl Throughput {
    pub fn new() -> Self {
        Self {
            last_progress: 0,
            last_instant: Instant::now(),
            rate_eps: 0.0,
        }
    }

    pub fn observe(&mut self, progress: u64) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_instant).as_secs_f64();
        if dt <= 0.0 {
            return;
        }
        let delta = progress.saturating_sub(self.last_progress) as f64;
        let instant_rate = delta / dt;
        let alpha = 0.3;
        if self.rate_eps == 0.0 {
            self.rate_eps = instant_rate;
        } else {
            self.rate_eps = alpha * instant_rate + (1.0 - alpha) * self.rate_eps;
        }
        self.last_progress = progress;
        self.last_instant = now;
    }

    pub fn reset(&mut self) {
        self.last_progress = 0;
        self.last_instant = Instant::now();
        self.rate_eps = 0.0;
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

fn fmt_rate(eps: f64) -> String {
    fmt_count(eps as u64)
}
