//! Memory-residency hints to the kernel.
//!
//! `madvise_sequential` is called once per large mmap'd allocation, to enable
//! readahead and aggressive eviction. `madvise_dontneed` is called per chunk
//! after the kernel has consumed an input range, releasing those pages
//! immediately so working-set growth cannot exceed RAM under streaming
//! workloads.
//!
//! On Windows we currently do nothing; the equivalents (`PrefetchVirtualMemory`,
//! `OfferVirtualMemory`) are deferred to v1.5.

#![allow(clippy::missing_safety_doc)]

#[cfg(unix)]
pub unsafe fn madvise_sequential(addr: *mut u8, len: usize) {
    // libc::MADV_SEQUENTIAL — kernel readahead + aggressive eviction.
    let _ = libc::madvise(addr as *mut libc::c_void, len, libc::MADV_SEQUENTIAL);
}

#[cfg(unix)]
pub unsafe fn madvise_dontneed_raw(addr: *mut u8, len: usize) {
    let _ = libc::madvise(addr as *mut libc::c_void, len, libc::MADV_DONTNEED);
}

/// `madvise(DONTNEED)` over the byte range covered by a typed slice.
///
/// Pages are rounded inward to whole-page boundaries; partial pages at either
/// end are skipped so we never drop bytes we haven't logically consumed.
#[cfg(unix)]
pub unsafe fn madvise_dontneed_slice<T>(s: &[T]) {
    if s.is_empty() {
        return;
    }
    let page = page_size();
    let start = s.as_ptr() as usize;
    let end = start + core::mem::size_of_val(s);

    // Round start UP and end DOWN to page boundaries.
    let aligned_start = (start + page - 1) & !(page - 1);
    let aligned_end = end & !(page - 1);
    if aligned_end <= aligned_start {
        return;
    }
    let len = aligned_end - aligned_start;
    let _ = libc::madvise(
        aligned_start as *mut libc::c_void,
        len,
        libc::MADV_DONTNEED,
    );
}

#[cfg(not(unix))]
pub unsafe fn madvise_sequential(_addr: *mut u8, _len: usize) {}

#[cfg(not(unix))]
pub unsafe fn madvise_dontneed_raw(_addr: *mut u8, _len: usize) {}

#[cfg(not(unix))]
pub unsafe fn madvise_dontneed_slice<T>(_s: &[T]) {}

#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: sysconf is signal-safe and returns -1 on error; we treat that as
    // "unknown page size" and fall back to 4 KiB.
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
