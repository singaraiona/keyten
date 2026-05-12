//! A runtime-agnostic `YieldNow` future.
//!
//! Returns `Pending` exactly once (signalling the executor to reschedule) and
//! `Ready` thereafter. Works under any executor that polls futures correctly:
//! tokio, async-std, smol, glommio, etc.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

pub struct YieldNow {
    yielded: bool,
}

impl YieldNow {
    #[inline]
    pub const fn new() -> Self {
        Self { yielded: false }
    }
}

impl Default for YieldNow {
    fn default() -> Self {
        Self::new()
    }
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}
