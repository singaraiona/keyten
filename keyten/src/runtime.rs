//! Process-wide runtime: shared state and configuration.
//!
//! `RUNTIME` is a `const`-initialised static, so no `init()` call is required.
//! The per-thread atom-cell freelist is stored in a native `thread_local!`
//! macro (single TLS slot, ≤3 cycles to access) and exposed via methods on
//! `Runtime`.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::alloc;

use crate::obj::Obj;

pub struct Runtime {
    /// Threshold (bytes) at which vector allocation switches from the system
    /// allocator to anonymous `mmap`. Vectors above this size live on pages
    /// the OS can lazy-fault, evict, and `madvise(DONTNEED)`.
    pub mmap_threshold: AtomicUsize,

    /// Process-wide cancellation flag (e.g. set by a SIGINT handler or REPL
    /// "Ctrl-C"). Checked alongside the per-task `Ctx.cancelled` flag at every
    /// chunk boundary.
    pub global_cancel: AtomicBool,

    /// Master switch for data-parallel kernel execution. When `false` (the
    /// default), kernels run on the calling thread via the existing
    /// single-threaded chunk drivers. When `true`, kernels above
    /// `PARALLEL_THRESHOLD` elements partition their range across the
    /// worker pool (Stage 2). No consumers yet — substrate prep.
    pub parallel: AtomicBool,

    /// Desired worker count when `parallel` is on. `0` means "use
    /// `cpu_count()`". No consumers yet.
    pub worker_count: AtomicUsize,

    /// Cached available parallelism. `0` until first probe; populated lazily
    /// by `cpu_count()` so `Runtime` stays `const`-initialised.
    pub(crate) cpu_count: AtomicUsize,
}

pub static RUNTIME: Runtime = Runtime::const_init();

// 16-byte atom-cell freelist, per thread. Single intrusive linked list whose
// "next" pointer overlays the unused cell.
thread_local! {
    static ATOM_POOL: Cell<*mut Obj> = const { Cell::new(ptr::null_mut()) };
}

impl Runtime {
    pub const fn const_init() -> Self {
        Self {
            mmap_threshold: AtomicUsize::new(1 << 20), // 1 MiB
            global_cancel: AtomicBool::new(false),
            parallel: AtomicBool::new(false),
            worker_count: AtomicUsize::new(0),
            cpu_count: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn mmap_threshold(&self) -> usize {
        self.mmap_threshold.load(Ordering::Relaxed)
    }

    pub fn set_mmap_threshold(&self, bytes: usize) {
        self.mmap_threshold.store(bytes, Ordering::Relaxed);
    }

    pub fn cancel_all(&self) {
        self.global_cancel.store(true, Ordering::Relaxed);
    }

    pub fn clear_cancel(&self) {
        self.global_cancel.store(false, Ordering::Relaxed);
    }

    /// Number of CPUs available to the process. Probed lazily on first call
    /// via `std::thread::available_parallelism`; cached for the rest of the
    /// process. Falls back to `1` on platforms where the probe fails.
    pub fn cpu_count(&self) -> usize {
        let cached = self.cpu_count.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let probed = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        self.cpu_count.store(probed, Ordering::Relaxed);
        probed
    }

    /// Resolved worker count for parallel kernels: explicit override if set,
    /// otherwise `cpu_count()`.
    pub fn worker_count(&self) -> usize {
        match self.worker_count.load(Ordering::Relaxed) {
            0 => self.cpu_count(),
            n => n,
        }
    }

    pub fn set_worker_count(&self, n: usize) {
        self.worker_count.store(n, Ordering::Relaxed);
    }

    #[inline]
    pub fn parallel_enabled(&self) -> bool {
        self.parallel.load(Ordering::Relaxed)
    }

    pub fn set_parallel(&self, on: bool) {
        self.parallel.store(on, Ordering::Relaxed);
    }

    /// Pop a 16-byte cell from the per-thread freelist, allocating fresh from
    /// the system allocator on miss.
    ///
    /// # Safety
    /// Returned pointer is uninitialised; caller must populate the header
    /// before any user can observe it.
    #[inline]
    pub unsafe fn pop_atom_cell(&self) -> *mut Obj {
        ATOM_POOL.with(|slot| {
            let head = slot.get();
            if head.is_null() {
                let layout = Layout::from_size_align_unchecked(16, 8);
                let p = alloc::alloc(layout);
                if p.is_null() {
                    alloc::handle_alloc_error(layout);
                }
                p as *mut Obj
            } else {
                // The first 8 bytes of a free cell hold the "next" pointer.
                let next = *(head as *const *mut Obj);
                slot.set(next);
                head
            }
        })
    }

    /// Return a 16-byte cell to the per-thread freelist.
    ///
    /// # Safety
    /// `c` must be a previously-allocated 16-byte cell with no live references.
    #[inline]
    pub unsafe fn push_atom_cell(&self, c: *mut Obj) {
        ATOM_POOL.with(|slot| {
            // Write the current head into the freed cell's first 8 bytes.
            *(c as *mut *mut Obj) = slot.get();
            slot.set(c);
        });
    }

    /// Drain and free every cell on the current thread's freelist. Useful in
    /// tests and on graceful shutdown; outside of those, leaking the freelist
    /// is correct (the OS reclaims at process exit).
    pub fn drain_atom_pool(&self) {
        ATOM_POOL.with(|slot| {
            let mut head = slot.get();
            while !head.is_null() {
                unsafe {
                    let next = *(head as *const *mut Obj);
                    let layout = Layout::from_size_align_unchecked(16, 8);
                    alloc::dealloc(head as *mut u8, layout);
                    head = next;
                }
            }
            slot.set(ptr::null_mut());
        });
    }
}
