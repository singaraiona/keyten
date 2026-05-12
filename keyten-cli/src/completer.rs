//! Tab-completion against the shared `Names` registry.

use reedline::{Completer, Span as ReedSpan, Suggestion};

use crate::names::SharedNames;

pub struct KCompleter {
    names: SharedNames,
}

impl KCompleter {
    pub fn new(names: SharedNames) -> Self {
        Self { names }
    }
}

impl Completer for KCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let bytes = line.as_bytes();
        let end = pos.min(bytes.len());
        let mut start = end;
        while start > 0 {
            let c = bytes[start - 1];
            if c.is_ascii_alphanumeric() || c == b'_' {
                start -= 1;
            } else {
                break;
            }
        }
        let prefix = &line[start..end];
        if prefix.is_empty() {
            return Vec::new();
        }
        let guard = match self.names.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let mut sugs = Vec::new();
        for name in guard.prefix_matches(prefix) {
            sugs.push(Suggestion {
                value: name.to_string(),
                description: None,
                style: None,
                extra: None,
                span: ReedSpan::new(start, end),
                append_whitespace: false,
                display_override: None,
                match_indices: None,
            });
        }
        sugs
    }
}
