//! A `Send + Sync` registry of bound names, decoupled from `Env`.
//!
//! reedline's `Highlighter` / `Completer` traits require `Send + Sync`. As of
//! v2 stage 0 `RefObj` is `Send` (atomic refcount), but it is **not** `Sync`
//! — concurrent in-place mutation is gated by `is_unique()` which assumes a
//! single writer at a time. That makes `Env` (a `HashMap<Sym, RefObj>`)
//! `Send` but not `Sync`, so it can't be shared via `Arc<Mutex<…>>` to the
//! reedline traits cleanly.
//!
//! The highlighter/completer don't need RefObjs anyway — they only need to
//! know which identifiers are currently bound. We maintain this small
//! registry alongside `Env` and refresh it after each successful eval.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use keyten::{Env, Sym};

#[derive(Default)]
pub struct Names {
    names: BTreeSet<String>,
}

impl Names {
    pub fn refresh_from(&mut self, env: &Env) {
        self.names.clear();
        for (sym, _) in env.iter() {
            if let Some(name) = decode(*sym) {
                self.names.insert(name);
            }
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn prefix_matches<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.names
            .iter()
            .filter(move |n| n.starts_with(prefix))
            .map(|s| s.as_str())
    }
}

pub type SharedNames = Arc<Mutex<Names>>;

fn decode(s: Sym) -> Option<String> {
    let bytes = s.0.to_le_bytes();
    let v: Vec<u8> = bytes.iter().copied().take_while(|b| *b != 0).collect();
    if v.is_empty() {
        return None;
    }
    String::from_utf8(v).ok()
}
