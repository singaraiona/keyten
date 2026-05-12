//! A minimal in-tree executor for synchronously driving futures that only
//! ever `await` on [`crate::yield_now::YieldNow`].
//!
//! It exists so the library can expose **one** kernel implementation (the
//! async one) and still serve sync callers — tests, batch jobs, library
//! users who don't want a tokio runtime — at essentially zero throughput cost.
//!
//! ## Why this is safe to spin on `Poll::Pending`
//!
//! Our kernels only suspend on [`YieldNow`], which returns `Pending` exactly
//! once and `Ready` thereafter. The poll loop here therefore terminates in
//! `O(chunks)` polls and never spins on a future that needs an external wake.
//!
//! If a future is awaited that requires real wake notification (network I/O,
//! a timer, a notify), polling it via `block_on` will busy-loop. Don't do
//! that — use a real runtime instead.

use core::future::Future;
use core::pin::pin;
use core::ptr;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// Drive a future to completion on the current thread.
///
/// Suitable only for futures that exclusively use [`crate::yield_now::YieldNow`]
/// as their suspension point. See the module docs.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => {
                // YieldNow set the waker eagerly; the next poll yields Ready.
            }
        }
    }
}

const VTABLE: RawWakerVTable = RawWakerVTable::new(
    |_| RawWaker::new(ptr::null(), &VTABLE), // clone
    |_| {},                                  // wake
    |_| {},                                  // wake_by_ref
    |_| {},                                  // drop
);

fn noop_waker() -> Waker {
    // SAFETY: the vtable's operations are all no-ops and the data pointer is
    // unused; this satisfies `Waker`'s contract.
    unsafe { Waker::from_raw(RawWaker::new(ptr::null(), &VTABLE)) }
}
