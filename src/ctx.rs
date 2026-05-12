//! Per-call execution context.
//!
//! `Ctx` is threaded through every kernel call. It carries:
//! - a reference to the process `Runtime`,
//! - a task-local cancellation flag and progress counter,
//! - an optional chunk-size override.
//!
//! `Ctx::quiet()` returns a zero-cost default backed by static atomics.

use core::sync::atomic::{AtomicBool, AtomicU64};

use crate::runtime::{Runtime, RUNTIME};

pub struct Ctx<'r> {
    pub runtime: &'r Runtime,
    pub cancelled: &'r AtomicBool,
    pub progress: &'r AtomicU64,
    /// Per-call chunk-size override; `0` means "use kernel default".
    pub chunk_elems: usize,
}

pub static QUIET_CANCEL: AtomicBool = AtomicBool::new(false);
pub static QUIET_PROGRESS: AtomicU64 = AtomicU64::new(0);

impl Ctx<'static> {
    /// Default context: shared static counters, kernel-default chunk size.
    /// Suitable for synchronous one-off calls where the caller does not care
    /// about cancellation or progress.
    pub fn quiet() -> Ctx<'static> {
        Ctx {
            runtime: &RUNTIME,
            cancelled: &QUIET_CANCEL,
            progress: &QUIET_PROGRESS,
            chunk_elems: 0,
        }
    }
}

impl<'r> Ctx<'r> {
    pub fn new(
        runtime: &'r Runtime,
        cancelled: &'r AtomicBool,
        progress: &'r AtomicU64,
    ) -> Self {
        Ctx {
            runtime,
            cancelled,
            progress,
            chunk_elems: 0,
        }
    }

    #[inline]
    pub fn with_chunk(mut self, chunk_elems: usize) -> Self {
        self.chunk_elems = chunk_elems;
        self
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum KernelErr {
    Cancelled,
    Oom,
    Type,
    Shape,
}
