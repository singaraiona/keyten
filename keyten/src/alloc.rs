//! Allocation, deallocation, and the mmap path.
//!
//! Atoms come from the per-thread 16-byte freelist on `Runtime`. Small vectors
//! go through the system allocator. Vectors `≥ Runtime.mmap_threshold` go
//! through anonymous `mmap` so the OS can lazy-fault pages and we can
//! `madvise(DONTNEED)` consumed input chunks.
//!
//! The release path is selected per-cell via flags in `meta`:
//! - atom (kind < 0)                         → freelist
//! - `IS_EXTERNAL`                           → no-op (we don't own it)
//! - `MMAP_BACKED`                           → `munmap`
//! - otherwise (heap-allocated vector)       → `dealloc`

use core::alloc::Layout;
use core::ptr::{self, NonNull};
use core::slice;
use core::sync::atomic::{AtomicU32, Ordering};
use std::alloc;

use crate::ctx::Ctx;
use crate::kind::Kind;
use crate::madvise::madvise_sequential;
use crate::obj::{meta_flags, Obj, RefObj};
use crate::runtime::RUNTIME;

// ---- atoms -------------------------------------------------------------

/// Allocate a fresh atom cell of the given kind, write `v` into the payload,
/// and return its `RefObj` (rc=1).
///
/// # Safety
/// `T` must match the storage type of `kind` (e.g. `T=i64` for `Kind::I64`).
#[inline]
pub unsafe fn alloc_atom<T: Copy>(kind: Kind, v: T) -> RefObj {
    let p = RUNTIME.pop_atom_cell();
    write_header(p, /*meta=*/ 0, /*attr=*/ 0, /*kind=*/ kind.atom(), /*rc=*/ 1);
    ptr::write_unaligned((p as *mut u8).add(8) as *mut T, v);
    RefObj(NonNull::new_unchecked(p))
}

// ---- vectors -----------------------------------------------------------

/// Allocate a fresh vector cell of the given kind and element count.
/// Payload bytes are uninitialised; caller fills them and may then set
/// `attr |= HAS_NULLS` as appropriate.
///
/// The mmap path is taken when total bytes ≥ `Runtime.mmap_threshold`. The
/// resulting cell's `meta` carries `MMAP_BACKED` so release routes through
/// `munmap`.
///
/// # Safety
/// `kind_code` must be a valid positive code for a vector kind whose element
/// size matches `elem_size`.
pub unsafe fn alloc_vec(ctx: &Ctx, kind_code: i8, n: i64, elem_size: usize) -> RefObj {
    let n_usize = n as usize;
    let bytes = 16usize
        .checked_add(
            n_usize
                .checked_mul(elem_size)
                .expect("vector byte size overflow"),
        )
        .expect("vector byte size overflow");

    let threshold = ctx.runtime.mmap_threshold();
    let (p, mmap_backed) = if bytes >= threshold && bytes >= page_size() {
        let m = mmap_anon(bytes);
        (m as *mut Obj, true)
    } else {
        let layout = layout_for_heap(bytes);
        let m = alloc::alloc(layout);
        if m.is_null() {
            alloc::handle_alloc_error(layout);
        }
        (m as *mut Obj, false)
    };

    let meta = if mmap_backed {
        meta_flags::MMAP_BACKED
    } else {
        0
    };
    write_header(p, meta, /*attr=*/ 0, kind_code, /*rc=*/ 1);
    // Vector length at offset 8.
    ptr::write_unaligned((p as *mut u8).add(8) as *mut i64, n);

    if mmap_backed {
        madvise_sequential(p as *mut u8, bytes);
    }

    RefObj(NonNull::new_unchecked(p))
}

/// Release a cell. Routes through the freelist (atoms), `munmap` (mmap-backed
/// vectors), or the system deallocator (heap-backed vectors). External cells
/// short-circuit.
///
/// For composite cells (`List`, `Dict`, `Table`) child `RefObj`s are dropped
/// recursively before the parent is freed.
///
/// # Safety
/// `p` must have been produced by `alloc_atom` or `alloc_vec` in this process,
/// and have no live references.
pub unsafe fn release(p: NonNull<Obj>) {
    let meta = (*p.as_ptr()).meta;
    if meta & meta_flags::IS_EXTERNAL != 0 {
        return; // not ours to free
    }

    let raw = (*p.as_ptr()).kind;

    // Atom: return cell to the freelist.
    if raw < 0 {
        RUNTIME.push_atom_cell(p.as_ptr());
        return;
    }

    // Vector / generic list / dict / table.
    let n = ptr::read_unaligned((p.as_ptr() as *const u8).add(8) as *const i64) as usize;
    let k = Kind::from_raw(raw);

    if matches!(k, Kind::List | Kind::Dict | Kind::Table) {
        let children = slice::from_raw_parts(
            (p.as_ptr() as *const u8).add(16) as *const RefObj,
            n,
        );
        for child in children {
            // Manually run Drop on each child by reading it out.
            ptr::read(child);
        }
    }

    let bytes = 16 + n * k.elem_size();
    if meta & meta_flags::MMAP_BACKED != 0 {
        munmap(p.as_ptr() as *mut u8, bytes);
    } else {
        let layout = layout_for_heap(bytes);
        alloc::dealloc(p.as_ptr() as *mut u8, layout);
    }
}

// ---- RefObj rc / drop -------------------------------------------------

// Refcount discipline follows the canonical `Arc` pattern: `Relaxed` on
// clone (we hold a count, so no synchronization is needed to make the
// increment visible — it is observable via the count itself), and `AcqRel`
// on the decrement so the thread that performs the final drop synchronizes
// with all prior modifications to the cell's payload before `release`
// reclaims it.

impl Clone for RefObj {
    fn clone(&self) -> Self {
        unsafe {
            let p = self.0.as_ptr();
            if (*p).meta & meta_flags::IS_EXTERNAL == 0 {
                (*p).rc.fetch_add(1, Ordering::Relaxed);
            }
            RefObj(self.0)
        }
    }
}

impl Drop for RefObj {
    fn drop(&mut self) {
        unsafe {
            let p = self.0.as_ptr();
            if (*p).meta & meta_flags::IS_EXTERNAL != 0 {
                return;
            }
            if (*p).rc.fetch_sub(1, Ordering::AcqRel) == 1 {
                release(self.0);
            }
        }
    }
}

// ---- helpers ---------------------------------------------------------

#[inline]
unsafe fn write_header(p: *mut Obj, meta: i8, attr: i8, kind: i8, rc: u32) {
    (*p).meta = meta;
    (*p).attr = attr;
    (*p).kind = kind;
    (*p)._resv = 0;
    // The cell's `rc` slot is uninitialised memory at this point (freshly
    // popped from the freelist or just mmap'd). Construct the atomic in
    // place via raw write rather than assigning to it.
    ptr::write(&raw mut (*p).rc, AtomicU32::new(rc));
}

#[inline]
fn layout_for_heap(bytes: usize) -> Layout {
    // 16-byte alignment matches the cell header's natural alignment and keeps
    // typed payload slices well-aligned for SIMD.
    Layout::from_size_align(bytes, 16).expect("invalid heap layout")
}

// ---- mmap/munmap -----------------------------------------------------

#[cfg(unix)]
pub unsafe fn mmap_anon(bytes: usize) -> *mut u8 {
    let p = libc::mmap(
        ptr::null_mut(),
        bytes,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
    );
    if p == libc::MAP_FAILED {
        alloc::handle_alloc_error(Layout::from_size_align_unchecked(bytes, 16));
    }
    p as *mut u8
}

#[cfg(unix)]
pub unsafe fn munmap(addr: *mut u8, bytes: usize) {
    let _ = libc::munmap(addr as *mut libc::c_void, bytes);
}

#[cfg(not(unix))]
pub unsafe fn mmap_anon(bytes: usize) -> *mut u8 {
    let layout = Layout::from_size_align(bytes, 16).expect("invalid heap layout");
    let p = alloc::alloc(layout);
    if p.is_null() {
        alloc::handle_alloc_error(layout);
    }
    p
}

#[cfg(not(unix))]
pub unsafe fn munmap(addr: *mut u8, bytes: usize) {
    let layout = Layout::from_size_align_unchecked(bytes, 16);
    alloc::dealloc(addr, layout);
}

#[cfg(unix)]
fn page_size() -> usize {
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if n <= 0 {
        4096
    } else {
        n as usize
    }
}

#[cfg(not(unix))]
fn page_size() -> usize {
    4096
}

// ---- thin typed constructors used by kernels --------------------------

#[inline]
pub unsafe fn alloc_vec_i64(ctx: &Ctx, n: i64) -> RefObj {
    alloc_vec(ctx, Kind::I64.vec(), n, Kind::I64.elem_size())
}

#[inline]
pub unsafe fn alloc_vec_f64(ctx: &Ctx, n: i64) -> RefObj {
    alloc_vec(ctx, Kind::F64.vec(), n, Kind::F64.elem_size())
}

#[inline]
pub unsafe fn alloc_vec_i32(ctx: &Ctx, n: i64) -> RefObj {
    alloc_vec(ctx, Kind::I32.vec(), n, Kind::I32.elem_size())
}
