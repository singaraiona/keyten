//! Variable environment for the evaluator.
//!
//! A simple flat namespace `Sym → RefObj`. `Sym` is a packed-i64 hash key by
//! construction (see `keyten::sym`), so lookup is a single `HashMap` probe on
//! an `i64` key — cheap.

use std::collections::HashMap;

use crate::obj::RefObj;
use crate::sym::Sym;

pub struct Env {
    bindings: HashMap<Sym, RefObj>,
}

impl Default for Env {
    fn default() -> Self {
        Env::new()
    }
}

impl Env {
    pub fn new() -> Self {
        Env { bindings: HashMap::new() }
    }

    /// Fetch a name. Returns a refcount-bumped clone (the caller owns one ref).
    pub fn lookup(&self, name: Sym) -> Option<RefObj> {
        self.bindings.get(&name).cloned()
    }

    /// Bind a name, replacing any previous value.
    pub fn bind(&mut self, name: Sym, value: RefObj) {
        self.bindings.insert(name, value);
    }

    /// Remove a binding. Used by lambda apply to restore the env after
    /// a param was previously unbound.
    pub fn unbind(&mut self, name: Sym) {
        self.bindings.remove(&name);
    }

    /// Iterate over current bindings (for the variable inspector).
    pub fn iter(&self) -> impl Iterator<Item = (&Sym, &RefObj)> {
        self.bindings.iter()
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}
