//! Colour palette and named styles.

use ratatui::style::{Color, Modifier, Style};

pub const PROMPT: Style = Style::new()
    .fg(Color::Cyan)
    .add_modifier(Modifier::BOLD);

pub const RESULT_NUMERIC: Style = Style::new().fg(Color::White);

pub const RESULT_STRING: Style = Style::new().fg(Color::Green);

pub const RESULT_NULL: Style = Style::new()
    .fg(Color::DarkGray)
    .add_modifier(Modifier::ITALIC);

pub const ERROR: Style = Style::new()
    .fg(Color::Red)
    .add_modifier(Modifier::BOLD);

pub const DIM: Style = Style::new().fg(Color::DarkGray);

pub const STATUS_IDLE: Style = Style::new().fg(Color::DarkGray);

pub const STATUS_BUSY: Style = Style::new()
    .fg(Color::Yellow)
    .add_modifier(Modifier::BOLD);

pub const ACCENT: Style = Style::new().fg(Color::Cyan);
