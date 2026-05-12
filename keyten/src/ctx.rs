//! Per-call execution context.
//!
//! `Ctx` is threaded through every kernel call. It carries:
//! - a reference to the process `Runtime`,
//! - a task-local cancellation flag and progress counter,
//! - an optional `RenderSink` whose `Notify` is signalled at chunk boundaries
//!   when the kernel has advanced enough to merit a UI redraw,
//! - an optional chunk-size override.
//!
//! `Ctx::quiet()` returns a zero-cost default backed by static atomics.

use core::sync::atomic::{AtomicBool, AtomicU64};

use crate::render::RenderSink;
use crate::runtime::{Runtime, RUNTIME};

pub struct Ctx<'r> {
    pub runtime: &'r Runtime,
    pub cancelled: &'r AtomicBool,
    pub progress: &'r AtomicU64,
    pub render: Option<&'r RenderSink>,
    /// Per-call chunk-size override; `0` means "use kernel default".
    pub chunk_elems: usize,
}

pub static QUIET_CANCEL: AtomicBool = AtomicBool::new(false);
pub static QUIET_PROGRESS: AtomicU64 = AtomicU64::new(0);

impl Ctx<'static> {
    /// Default context: shared static counters, no render sink, kernel-default
    /// chunk size. Suitable for one-off synchronous calls where the caller does
    /// not care about cancellation, progress, or UI updates.
    pub fn quiet() -> Ctx<'static> {
        Ctx {
            runtime: &RUNTIME,
            cancelled: &QUIET_CANCEL,
            progress: &QUIET_PROGRESS,
            render: None,
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
            render: None,
            chunk_elems: 0,
        }
    }

    #[inline]
    pub fn with_chunk(mut self, chunk_elems: usize) -> Self {
        self.chunk_elems = chunk_elems;
        self
    }

    #[inline]
    pub fn with_render(mut self, render: &'r RenderSink) -> Self {
        self.render = Some(render);
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
