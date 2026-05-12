//! Output scrollback: a ring buffer of (prompt, source, body) entries with a
//! ratatui widget that wraps and renders the visible window.

use std::collections::VecDeque;
use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme;

const MAX_ENTRIES: usize = 1000;

pub enum Body {
    /// A completed successful evaluation.
    Ok {
        lines: Vec<Line<'static>>,
        took: Duration,
    },
    /// A failed evaluation: error message + optional caret line.
    Err {
        message: String,
        caret: Option<String>,
    },
    /// Evaluation in flight; we render a spinner+progress until it completes.
    Active,
}

pub struct OutEntry {
    pub prompt: String,
    pub source: String,
    pub body: Body,
}

pub struct OutputBuffer {
    entries: VecDeque<OutEntry>,
    /// Scroll offset from the bottom in "logical lines". 0 = pinned to newest.
    pub scroll: u16,
}

impl Default for OutputBuffer {
    fn default() -> Self {
        Self {
            entries: VecDeque::with_capacity(64),
            scroll: 0,
        }
    }
}

impl OutputBuffer {
    pub fn push(&mut self, entry: OutEntry) {
        while self.entries.len() >= MAX_ENTRIES {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        // Re-pin to newest on each new entry.
        self.scroll = 0;
    }

    pub fn last_mut(&mut self) -> Option<&mut OutEntry> {
        self.entries.back_mut()
    }

    pub fn scroll_up(&mut self, n: u16) {
        self.scroll = self.scroll.saturating_add(n);
    }
    pub fn scroll_down(&mut self, n: u16) {
        self.scroll = self.scroll.saturating_sub(n);
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer, busy_status: Option<Line<'static>>) {
        // Flatten entries into a Vec<Line>, oldest first.
        let mut lines: Vec<Line<'static>> = Vec::with_capacity(self.entries.len() * 2 + 1);
        for entry in &self.entries {
            // Source line: prompt + source text.
            lines.push(Line::from(vec![
                Span::styled(entry.prompt.clone(), theme::PROMPT),
                Span::raw(entry.source.clone()),
            ]));
            match &entry.body {
                Body::Ok { lines: result, took } => {
                    for l in result {
                        lines.push(l.clone());
                    }
                    if took.as_millis() > 100 {
                        lines.push(Line::from(Span::styled(
                            format!("  ({:.2}s)", took.as_secs_f64()),
                            theme::DIM,
                        )));
                    }
                }
                Body::Err { message, caret } => {
                    lines.push(Line::from(Span::styled(
                        format!("  error: {message}"),
                        theme::ERROR,
                    )));
                    if let Some(c) = caret {
                        lines.push(Line::from(Span::styled(c.clone(), theme::DIM)));
                    }
                }
                Body::Active => {
                    if let Some(line) = busy_status.as_ref() {
                        lines.push(line.clone());
                    } else {
                        lines.push(Line::from(Span::styled(
                            "  working\u{2026}",
                            theme::STATUS_BUSY,
                        )));
                    }
                }
            }
            lines.push(Line::raw(""));
        }

        let block = Block::default().borders(Borders::NONE);
        let inner = block.inner(area);
        block.render(area, buf);

        let total = lines.len() as u16;
        // Wrap into the inner area and render with scroll.
        let visible_rows = inner.height;
        let max_scroll = total.saturating_sub(visible_rows);
        let scroll = self.scroll.min(max_scroll);
        // We want newest at the bottom; ratatui's `scroll` is "skip first N lines".
        let skip = max_scroll.saturating_sub(scroll);
        let para = Paragraph::new(lines)
            .style(Style::default())
            .wrap(Wrap { trim: false })
            .scroll((skip, 0));
        para.render(inner, buf);
    }
}
