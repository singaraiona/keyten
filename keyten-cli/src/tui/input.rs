//! Input editor (tui-textarea) with K-aware key handling: history recall,
//! balanced-bracket Enter, two-space tab.

use crossterm::event::{KeyEvent, KeyModifiers};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders, Widget};
use tui_textarea::{Input, Key, TextArea};

use crate::history::History;

pub struct InputArea {
    pub textarea: TextArea<'static>,
    history_idx: Option<usize>,
    pending_buffer: Option<Vec<String>>,
}

impl Default for InputArea {
    fn default() -> Self {
        let mut ta = TextArea::default();
        ta.set_cursor_line_style(Style::default());
        ta.set_block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
                .title("k)"),
        );
        Self {
            textarea: ta,
            history_idx: None,
            pending_buffer: None,
        }
    }
}

pub enum InputOutcome {
    /// User submitted a complete expression — pass `text` to the evaluator.
    Submit(String),
    /// Move output scrollback up/down (PageUp / PageDown).
    ScrollUp,
    ScrollDown,
    /// Nothing user-visible changed for the app loop to react to; just redraw.
    None,
    /// Editor consumed the event but it has no app-level meaning.
    Edited,
}

impl InputArea {
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        self.textarea.render(area, buf);
    }

    /// Handle one terminal key event.
    pub fn handle_key(&mut self, key: KeyEvent, history: &History) -> InputOutcome {
        // Translate KeyEvent -> tui-textarea Input.
        let input: Input = key.into();

        match input.key {
            Key::Enter => {
                let shift = input.shift;
                let alt = input.alt;
                // Shift-Enter or Alt-Enter: force newline.
                if shift || alt {
                    self.textarea.insert_newline();
                    return InputOutcome::Edited;
                }
                // Else: submit iff brackets are balanced.
                let text = self.textarea.lines().join("\n");
                if brackets_balanced(&text) {
                    if !text.trim().is_empty() {
                        // Reset editor.
                        self.textarea = TextArea::default();
                        self.textarea.set_cursor_line_style(Style::default());
                        self.textarea.set_block(
                            Block::default()
                                .borders(Borders::TOP | Borders::BOTTOM)
                                .title("k)"),
                        );
                        self.history_idx = None;
                        self.pending_buffer = None;
                        return InputOutcome::Submit(text);
                    }
                    return InputOutcome::None;
                }
                self.textarea.insert_newline();
                InputOutcome::Edited
            }
            Key::Up if self.is_empty_or_at_top() => {
                self.recall_prev(history);
                InputOutcome::Edited
            }
            Key::Down if self.history_idx.is_some() => {
                self.recall_next(history);
                InputOutcome::Edited
            }
            Key::PageUp => InputOutcome::ScrollUp,
            Key::PageDown => InputOutcome::ScrollDown,
            Key::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Move to start-of-line — tui-textarea's standard. Let it through.
                self.textarea.input(input);
                InputOutcome::Edited
            }
            Key::Tab => {
                self.textarea.insert_str("  ");
                InputOutcome::Edited
            }
            _ => {
                self.textarea.input(input);
                InputOutcome::Edited
            }
        }
    }

    fn is_empty_or_at_top(&self) -> bool {
        let (row, _col) = self.textarea.cursor();
        row == 0
    }

    fn snapshot(&self) -> Vec<String> {
        self.textarea.lines().to_vec()
    }

    fn restore(&mut self, lines: Vec<String>) {
        self.textarea = TextArea::new(lines);
        self.textarea.set_cursor_line_style(Style::default());
        self.textarea.set_block(
            Block::default()
                .borders(Borders::TOP | Borders::BOTTOM)
                .title("k)"),
        );
    }

    fn recall_prev(&mut self, history: &History) {
        let entries = history.entries();
        if entries.is_empty() {
            return;
        }
        if self.history_idx.is_none() {
            self.pending_buffer = Some(self.snapshot());
            self.history_idx = Some(entries.len());
        }
        let idx = self.history_idx.unwrap();
        if idx == 0 {
            return; // already at oldest
        }
        let new_idx = idx - 1;
        self.history_idx = Some(new_idx);
        self.restore(vec![entries[new_idx].clone()]);
    }

    fn recall_next(&mut self, history: &History) {
        let entries = history.entries();
        let idx = match self.history_idx {
            Some(i) => i,
            None => return,
        };
        let new_idx = idx + 1;
        if new_idx >= entries.len() {
            // Past the newest: restore the pending buffer.
            self.history_idx = None;
            if let Some(buf) = self.pending_buffer.take() {
                self.restore(buf);
            } else {
                self.restore(vec![String::new()]);
            }
        } else {
            self.history_idx = Some(new_idx);
            self.restore(vec![entries[new_idx].clone()]);
        }
    }

}

fn brackets_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev = '\0';
    for c in s.chars() {
        if in_str {
            if c == '"' && prev != '\\' {
                in_str = false;
            }
        } else {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                '"' => in_str = true,
                _ => {}
            }
        }
        prev = c;
        if depth < 0 {
            return false;
        }
    }
    depth == 0 && !in_str
}
