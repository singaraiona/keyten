//! Persisted command history.
//!
//! Storage path: `$XDG_STATE_HOME/keyten/history`, defaulting to
//! `~/.local/state/keyten/history` on Unix. One entry per line, UTF-8. Rolling
//! cap of 10 000 lines.

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const CAP: usize = 10_000;

pub struct History {
    entries: VecDeque<String>,
    dirty: bool,
}

impl History {
    pub fn load() -> Self {
        let path = file_path();
        let entries = path
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok())
            .map(|s| {
                s.lines()
                    .map(|l| l.trim_end().to_string())
                    .filter(|l| !l.is_empty())
                    .collect::<VecDeque<_>>()
            })
            .unwrap_or_default();
        Self {
            entries,
            dirty: false,
        }
    }

    pub fn push(&mut self, line: String) {
        if line.trim().is_empty() {
            return;
        }
        if self.entries.back().map(|s| s.as_str()) == Some(line.as_str()) {
            return; // skip exact dups of the most recent entry
        }
        self.entries.push_back(line);
        while self.entries.len() > CAP {
            self.entries.pop_front();
        }
        self.dirty = true;
    }

    pub fn entries(&self) -> &VecDeque<String> {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn save(&self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let path = match file_path() {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut tmp = path.clone();
        tmp.set_extension("tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            for line in &self.entries {
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }
}

fn file_path() -> Option<PathBuf> {
    let base = dirs::state_dir().or_else(dirs::data_local_dir)?;
    Some(base.join("keyten").join("history"))
}
