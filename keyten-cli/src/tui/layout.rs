//! Frame composition: output area + input area + status line.

use ratatui::layout::{Constraint, Direction, Layout, Rect};

pub fn split(area: Rect) -> (Rect, Rect, Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),    // output
            Constraint::Length(5), // input editor (multi-line capable)
            Constraint::Length(1), // status
        ])
        .split(area);
    (chunks[0], chunks[1], chunks[2])
}
