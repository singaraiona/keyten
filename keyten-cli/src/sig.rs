//! Signal / panic terminal-state recovery.
//!
//! On panic we need to leave raw mode and the alternate screen, otherwise the
//! terminal stays in a broken state after the process dies.

use std::io;
use std::panic;

use crossterm::{
    execute,
    terminal::{disable_raw_mode, LeaveAlternateScreen},
};

pub fn install_panic_hook() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        prev(info);
    }));
}
