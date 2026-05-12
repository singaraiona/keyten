//! Keyten REPL — a tokio-driven terminal application atop the keyten runtime.

use anyhow::Result;
use keyten_cli::{app, sig, tui};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    sig::install_panic_hook();
    let mut terminal = tui::init_terminal()?;
    let res = app::run(&mut terminal).await;
    tui::restore_terminal(&mut terminal)?;
    res
}
