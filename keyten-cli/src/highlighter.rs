//! reedline `Highlighter` — colours verbs, atoms, names, strings.
//!
//! Operates on a `SharedNames` snapshot rather than the live `Env` so it
//! satisfies reedline's `Send` bound. Names are refreshed by the REPL loop
//! after each successful eval.

use nu_ansi_term::{Color, Style};
use reedline::{Highlighter, StyledText};

use crate::names::SharedNames;

pub struct KHighlighter {
    names: SharedNames,
}

impl KHighlighter {
    pub fn new(names: SharedNames) -> Self {
        Self { names }
    }
}

impl Highlighter for KHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut out = StyledText::new();
        let mut i = 0;
        let b = line.as_bytes();
        while i < b.len() {
            let c = b[i];
            if c == b' ' || c == b'\t' {
                out.push((Style::new(), (c as char).to_string()));
                i += 1;
                continue;
            }
            if c == b'"' {
                let start = i;
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if i < b.len() {
                    i += 1;
                }
                out.push((Style::new().fg(Color::Green), line[start..i].to_string()));
                continue;
            }
            if c == b'`' {
                let start = i;
                i += 1;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push((Style::new().fg(Color::Cyan), line[start..i].to_string()));
                continue;
            }
            if c.is_ascii_digit() {
                let start = i;
                if c == b'0' && i + 1 < b.len() && matches!(b[i + 1], b'N' | b'n' | b'W' | b'w') {
                    i += 2;
                    out.push((
                        Style::new().fg(Color::DarkGray).italic(),
                        line[start..i].to_string(),
                    ));
                    continue;
                }
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && b[i] == b'.' {
                    i += 1;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    i += 1;
                    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                        i += 1;
                    }
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                out.push((Style::new().fg(Color::White), line[start..i].to_string()));
                continue;
            }
            if c.is_ascii_alphabetic() || c == b'_' {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let s = &line[start..i];
                let bound = self
                    .names
                    .lock()
                    .map(|n| n.contains(s))
                    .unwrap_or(false);
                let style = if bound {
                    Style::new().fg(Color::LightYellow).bold()
                } else {
                    Style::new().fg(Color::LightGray)
                };
                out.push((style, s.to_string()));
                continue;
            }
            let style = match c {
                b'+' | b'-' | b'*' | b'%' => Style::new().fg(Color::Cyan).bold(),
                b'/' | b'\\' => Style::new().fg(Color::Magenta).bold(),
                b':' | b';' => Style::new().fg(Color::DarkGray),
                b'(' | b')' => Style::new().fg(Color::LightGray),
                _ => Style::new(),
            };
            out.push((style, (c as char).to_string()));
            i += 1;
        }
        out
    }
}
