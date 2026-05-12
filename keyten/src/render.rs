//! Wake source for an interactive UI watching a running kernel.
//!
//! Kernels write `progress` and `cancelled` flags unconditionally; the UI's
//! `notify` is only signalled when progress has advanced by at least `stride`
//! elements. The chunk loop pays one relaxed atomic load and one compare per
//! chunk to make the decision.
//!
//! The notification is edge-triggered (tokio `Notify` semantics): if no one is
//! awaiting at the moment of notify, one wake is buffered; subsequent notifies
//! before the receiver awakes collapse into the single pending wake.

use core::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::Notify;

pub struct RenderSink {
    pub notify: Arc<Notify>,
    /// Last progress value at which `notify.notify_one()` was invoked.
    pub last_notified_progress: AtomicU64,
    /// Minimum progress advancement between intermediate notifications. `0`
    /// disables intermediate notifications (only the completion notify fires).
    pub stride: AtomicU64,
}

impl RenderSink {
    pub fn new(stride: u64) -> Self {
        Self {
            notify: Arc::new(Notify::new()),
            last_notified_progress: AtomicU64::new(0),
            stride: AtomicU64::new(stride),
        }
    }

    /// Build a fresh sink, ready to be embedded in a `Ctx`.
    pub fn with_stride(stride: u64) -> Self {
        Self::new(stride)
    }

    /// Set `stride` at runtime (e.g. when the UI adaptively retunes).
    pub fn set_stride(&self, stride: u64) {
        self.stride
            .store(stride, core::sync::atomic::Ordering::Relaxed);
    }

    /// Subscriber handle: the UI awaits on this `Notify` to learn when the
    /// kernel has made enough visible progress to merit a redraw.
    pub fn notify_handle(&self) -> Arc<Notify> {
        Arc::clone(&self.notify)
    }
}
