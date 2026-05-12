//! Dyadic verb kernels: `,` (concatenate), `#` (take).
//!
//! Both produce a fresh vector whose contents are simple byte/element copies
//! from one or two inputs. Performance-mandated: chunked `ChunkStep` impls
//! and a parallel `parallel_for_each_mut` branch for large inputs.

use crate::alloc::alloc_vec;
use crate::chunk::{drive_async, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::obj::RefObj;
use crate::parallel;

// =======================================================================
// Concatenate `x , y` — produces a vector of length len(x) + len(y).
// =======================================================================

/// `x , y` for same-kind, non-composite operands. The output vector has the
/// same kind as the inputs and length `len(x) + len(y)`.
///
/// For atoms, callers should treat the atom as a 1-element vector (its
/// payload byte block at offset 8 has size `elem_size()`).
///
/// # Safety
/// `x` and `y` must have the same kind, which must be a non-composite
/// vector kind (Bool/U8/I16/I32/I64/F32/F64/Char/Sym/Date/Time*/Dt*).
pub async unsafe fn concat_same_kind_async(
    x: RefObj,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    let kind = x.kind();
    let elem = kind.elem_size();

    // Total output length: 1 if atom else len.
    let nx = if x.is_atom() { 1 } else { x.len() as usize };
    let ny = if y.is_atom() { 1 } else { y.len() as usize };
    let n = nx + ny;

    // Allocate the output. alloc_vec writes the header and len.
    let mut out = alloc_vec(ctx, kind.vec(), n as i64, elem);

    // Get raw source pointers (offset 8 for atoms, offset 16 for vectors)
    // and the output data pointer (offset 16).
    let x_src = if x.is_atom() {
        (x.as_ptr() as *const u8).add(8)
    } else {
        (x.as_ptr() as *const u8).add(16)
    };
    let y_src = if y.is_atom() {
        (y.as_ptr() as *const u8).add(8)
    } else {
        (y.as_ptr() as *const u8).add(16)
    };
    let out_data = (out.as_ptr() as *mut u8).add(16);

    let nx_bytes = nx * elem;
    let ny_bytes = ny * elem;

    // First copy x → out[0..nx_bytes]. For large inputs, do this in a
    // chunked ChunkStep so cancellation/progress observe.
    let mut copy_x = CopyBytes::new(x_src, out_data, nx_bytes, COPY_CHUNK_BYTES);
    let go_parallel_x =
        ctx.runtime.parallel_enabled() && nx_bytes >= parallel::PARALLEL_THRESHOLD * elem;
    if go_parallel_x {
        // Raw `*const u8` isn't Send/Sync, so we pass it across the worker
        // boundary as `usize` and reconstruct on the other side. Safe
        // because (a) `x`/`y` outlive the scope, and (b) workers only read
        // disjoint source slices that match disjoint destination slices.
        let x_src_addr = x_src as usize;
        let out_slice = core::slice::from_raw_parts_mut(out_data, nx_bytes);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            unsafe {
                let src = (x_src_addr as *const u8).add(range.start);
                core::ptr::copy_nonoverlapping(src, my_slice.as_mut_ptr(), my_slice.len());
            }
            Ok(())
        })?;
    } else {
        drive_async(&mut copy_x, ctx).await?;
    }

    // Then copy y → out[nx_bytes..]. Same parallel/sync decision.
    let dst_y = out_data.add(nx_bytes);
    let mut copy_y = CopyBytes::new(y_src, dst_y, ny_bytes, COPY_CHUNK_BYTES);
    let go_parallel_y =
        ctx.runtime.parallel_enabled() && ny_bytes >= parallel::PARALLEL_THRESHOLD * elem;
    if go_parallel_y {
        let y_src_addr = y_src as usize;
        let out_slice = core::slice::from_raw_parts_mut(dst_y, ny_bytes);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            unsafe {
                let src = (y_src_addr as *const u8).add(range.start);
                core::ptr::copy_nonoverlapping(src, my_slice.as_mut_ptr(), my_slice.len());
            }
            Ok(())
        })?;
    } else {
        drive_async(&mut copy_y, ctx).await?;
    }

    let _ = &mut out;
    Ok(out)
}

/// Per-chunk byte size for streaming copies. Sized to keep working set in
/// L2 (~256 KiB) across read+write.
const COPY_CHUNK_BYTES: usize = 64 * 1024;

/// Chunked byte-copy state machine. Treats source and destination as raw
/// byte streams; caller handles kind, alignment, and lifetime.
struct CopyBytes {
    src: *const u8,
    dst: *mut u8,
    total: usize,
    off: usize,
    chunk: usize,
}

impl CopyBytes {
    fn new(src: *const u8, dst: *mut u8, total: usize, chunk: usize) -> Self {
        Self {
            src,
            dst,
            total,
            off: 0,
            chunk: chunk.max(1),
        }
    }
}

impl ChunkStep for CopyBytes {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + self.chunk).min(self.total);
        let n = end - self.off;
        unsafe {
            core::ptr::copy_nonoverlapping(self.src.add(self.off), self.dst.add(self.off), n);
            // Hint that the source pages we just consumed don't need to stay
            // resident — same RSS-bounding pattern as the arithmetic kernels.
            let src_slice = core::slice::from_raw_parts(self.src.add(self.off), n);
            madvise_dontneed_slice(src_slice);
        }
        self.off = end;
        Some(n)
    }
}

// =======================================================================
// Take `n # y` — output is `n` elements of `y`, cycling if `n > len(y)`.
// =======================================================================

/// `n # y` — take the first `n` elements of `y`. If `n` exceeds `len(y)`,
/// the result cycles through `y` (`n # 1 2 3` with `n=7` gives
/// `1 2 3 1 2 3 1`).
///
/// Negative `n` is K-conventional "take from the end" — deferred until
/// the negative case has a parser test. For now `n` must be non-negative.
///
/// # Safety
/// `y` must be a non-composite vector or atom; its kind drives the result.
pub async unsafe fn take_async(
    n: i64,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    if n < 0 {
        // TODO: negative n — take from end (K convention).
        return Err(KernelErr::Type);
    }
    let kind = y.kind();
    let elem = kind.elem_size();
    let n_out = n as usize;

    if n_out == 0 {
        // Empty result vector of the same kind.
        return Ok(alloc_vec(ctx, kind.vec(), 0, elem));
    }

    let src = if y.is_atom() {
        (y.as_ptr() as *const u8).add(8)
    } else {
        (y.as_ptr() as *const u8).add(16)
    };
    let src_len = if y.is_atom() { 1 } else { y.len() as usize };
    if src_len == 0 {
        // Source is empty but n > 0 — can't cycle nothing. Match K and
        // return a vector of N null sentinels. Defer this edge case; for
        // now signal a shape error.
        return Err(KernelErr::Shape);
    }

    let mut out = alloc_vec(ctx, kind.vec(), n_out as i64, elem);
    let dst = (out.as_ptr() as *mut u8).add(16);
    let total_bytes = n_out * elem;
    let src_bytes = src_len * elem;

    // The fast path: n_out is a multiple of src_len (or src_len divides
    // total cleanly). Then we just memcpy src_bytes `n_out/src_len` times
    // back-to-back. The general path uses modulo arithmetic per write.
    //
    // For the parallel path, each worker fills its byte range with the
    // cycled pattern. The cycle modulo is per-byte, so workers compute
    // `(out_byte_off + local_off) % src_bytes` to find the source byte.

    let go_parallel =
        ctx.runtime.parallel_enabled() && total_bytes >= parallel::PARALLEL_THRESHOLD * elem;
    if go_parallel {
        let src_addr = src as usize;
        let out_slice = core::slice::from_raw_parts_mut(dst, total_bytes);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            // Each worker fills `my_slice` starting at global byte offset
            // `range.start`. The source byte for global offset i is
            // src[(i mod src_bytes)].
            let mut g = range.start;
            for byte in my_slice.iter_mut() {
                let s = g % src_bytes;
                *byte = unsafe { *(src_addr as *const u8).add(s) };
                g += 1;
            }
            Ok(())
        })?;
    } else {
        let mut k = CycleBytes::new(src, src_bytes, dst, total_bytes, COPY_CHUNK_BYTES);
        drive_async(&mut k, ctx).await?;
    }

    let _ = &mut out;
    Ok(out)
}

/// Chunked cycle-copy state machine. Source is a fixed-size byte pattern;
/// destination is filled by repeating the pattern modulo its length.
struct CycleBytes {
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    total: usize,
    off: usize,
    chunk: usize,
}

impl CycleBytes {
    fn new(src: *const u8, src_len: usize, dst: *mut u8, total: usize, chunk: usize) -> Self {
        Self {
            src,
            src_len,
            dst,
            total,
            off: 0,
            chunk: chunk.max(1),
        }
    }
}

impl ChunkStep for CycleBytes {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + self.chunk).min(self.total);
        let mut g = self.off;
        for local in 0..(end - self.off) {
            let s = (self.off + local) % self.src_len;
            unsafe {
                *self.dst.add(self.off + local) = *self.src.add(s);
            }
            let _ = g; // keep `g` as a marker for the global offset
            g += 1;
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{alloc_atom, alloc_vec_i64};
    use crate::exec::block_on;

    fn make_i64_vec(data: &[i64]) -> RefObj {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, data.len() as i64) };
        unsafe {
            v.as_mut_slice::<i64>().copy_from_slice(data);
        }
        v
    }

    #[test]
    fn concat_two_vectors_same_kind() {
        let ctx = Ctx::quiet();
        let x = make_i64_vec(&[1, 2, 3]);
        let y = make_i64_vec(&[10, 20, 30, 40]);
        let r = block_on(async {
            unsafe { concat_same_kind_async(x, y, &ctx).await.expect("concat") }
        });
        assert_eq!(r.len(), 7);
        let s = unsafe { r.as_slice::<i64>() };
        assert_eq!(s, &[1, 2, 3, 10, 20, 30, 40]);
    }

    #[test]
    fn concat_atom_plus_vector() {
        let ctx = Ctx::quiet();
        let a = unsafe { alloc_atom(Kind::I64, 99i64) };
        let v = make_i64_vec(&[1, 2, 3]);
        let r = block_on(async {
            unsafe { concat_same_kind_async(a, v, &ctx).await.expect("concat") }
        });
        let s = unsafe { r.as_slice::<i64>() };
        assert_eq!(s, &[99, 1, 2, 3]);
    }

    #[test]
    fn take_exact_length_is_identity() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[10, 20, 30]);
        let r = block_on(async { unsafe { take_async(3, v, &ctx).await.expect("take") } });
        let s = unsafe { r.as_slice::<i64>() };
        assert_eq!(s, &[10, 20, 30]);
    }

    #[test]
    fn take_shorter_truncates() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[10, 20, 30, 40, 50]);
        let r = block_on(async { unsafe { take_async(3, v, &ctx).await.expect("take") } });
        let s = unsafe { r.as_slice::<i64>() };
        assert_eq!(s, &[10, 20, 30]);
    }

    #[test]
    fn take_longer_cycles() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[1, 2, 3]);
        let r = block_on(async { unsafe { take_async(7, v, &ctx).await.expect("take") } });
        let s = unsafe { r.as_slice::<i64>() };
        assert_eq!(s, &[1, 2, 3, 1, 2, 3, 1]);
    }
}
