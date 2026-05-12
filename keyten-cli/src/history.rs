//! Persistent history at `$XDG_STATE_HOME/keyten/history`.

use std::path::PathBuf;

use reedline::FileBackedHistory;

const CAP: usize = 10_000;

pub fn open() -> Option<FileBackedHistory> {
    let path = history_path()?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    FileBackedHistory::with_file(CAP, path).ok()
}

fn history_path() -> Option<PathBuf> {
    let base = dirs::state_dir().or_else(dirs::data_local_dir)?;
    Some(base.join("keyten").join("history"))
}
