//! Vectors above the mmap threshold are allocated via anonymous mmap and the
//! release path calls munmap. Verified by setting a low threshold and checking
//! the `MMAP_BACKED` meta bit on the resulting cell.

use keyten::alloc::alloc_vec_i64;
use keyten::ctx::Ctx;
use keyten::obj::meta_flags;
use keyten::runtime::RUNTIME;

#[test]
fn large_vec_takes_mmap_path() {
    // Force the mmap path on small allocations by lowering the threshold.
    let prev = RUNTIME.mmap_threshold();
    RUNTIME.set_mmap_threshold(4 * 1024); // 4 KiB
    let ctx = Ctx::quiet();
    unsafe {
        // 16 (header) + 1024*8 (data) = 8208 bytes > 4 KiB
        let v = alloc_vec_i64(&ctx, 1024);
        let m = v.meta();
        assert_ne!(m & meta_flags::MMAP_BACKED, 0, "expected MMAP_BACKED bit set");
        // Vector is writable.
        let mut v = v;
        v.as_mut_slice::<i64>().fill(7);
        assert_eq!(v.as_slice::<i64>()[100], 7);
        // Drop runs munmap via release; not directly observable here other than
        // not crashing.
    }
    RUNTIME.set_mmap_threshold(prev);
}

#[test]
fn small_vec_takes_heap_path() {
    let ctx = Ctx::quiet();
    unsafe {
        let v = alloc_vec_i64(&ctx, 4); // 48 bytes
        let m = v.meta();
        assert_eq!(m & meta_flags::MMAP_BACKED, 0);
    }
}
