//! `_` verb kernels.
//!
//! - **Monadic `_x`**: floor. F64 → I64 (atom or vector); I64 passes through
//!   unchanged.
//! - **Dyadic `n _ y`**: drop. Drop the first `n` elements of `y` (or the
//!   last `|n|` if `n < 0`). Same-kind output of length `max(0, len(y)-|n|)`.

use crate::alloc::{alloc_atom, alloc_vec, alloc_vec_i64};
use crate::chunk::{drive_async, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::madvise::madvise_dontneed_slice;
use crate::obj::RefObj;
use crate::parallel;

const FLOOR_CHUNK: usize = 64 * 1024;

/// Monadic `_x` — floor. F64 atom/vector → I64; I64 passes through; other
/// kinds error.
pub async unsafe fn floor_async(x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xk = x.kind();
    if xk == Kind::I64 {
        return Ok(x);
    }
    if xk != Kind::F64 {
        return Err(KernelErr::Type);
    }
    if x.is_atom() {
        return Ok(alloc_atom(Kind::I64, x.atom::<f64>().floor() as i64));
    }
    let xs = x.as_slice::<f64>();
    let n = xs.len();
    let mut out = alloc_vec_i64(ctx, n as i64);

    let go_parallel = ctx.runtime.parallel_enabled() && n >= parallel::PARALLEL_THRESHOLD;
    if go_parallel {
        let xs_addr = xs.as_ptr() as usize;
        let os_addr = (out.as_ptr() as *mut u8).add(16) as usize;
        // Per-element byte stride: 8 bytes for i64 output. Split the byte
        // slice over workers so each touches a contiguous I64 sub-range.
        let out_bytes_slice = core::slice::from_raw_parts_mut(os_addr as *mut u8, n * 8);
        parallel::parallel_for_each_mut(out_bytes_slice, ctx, move |range, my_slice| {
            let elem_start = range.start / 8;
            let elem_end = elem_start + my_slice.len() / 8;
            unsafe {
                let xs = xs_addr as *const f64;
                let os = os_addr as *mut i64;
                for i in elem_start..elem_end {
                    *os.add(i) = (*xs.add(i)).floor() as i64;
                }
            }
            Ok(())
        })?;
    } else {
        let os = out.as_mut_slice::<i64>();
        let mut k = FloorStep::new(xs.as_ptr(), os.as_mut_ptr(), n);
        drive_async(&mut k, ctx).await?;
    }
    Ok(out)
}

/// Dyadic `n _ y` — drop. `n` must be an I64 atom. `y` non-composite.
/// Output is a vector of `y`'s kind, length `max(0, len(y) - |n|)`. The
/// kept bytes are copied verbatim from `y`.
pub async unsafe fn drop_async(
    n: i64,
    y: RefObj,
    ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    let kind = y.kind();
    if matches!(kind, Kind::List | Kind::Dict | Kind::Table) {
        return Err(KernelErr::Type);
    }
    let elem = kind.elem_size();
    let src_len = if y.is_atom() { 1usize } else { y.len() as usize };

    let drop_count = n.unsigned_abs() as usize;
    let n_out = src_len.saturating_sub(drop_count);
    if n_out == 0 {
        return Ok(alloc_vec(ctx, kind.vec(), 0, elem));
    }

    let mut out = alloc_vec(ctx, kind.vec(), n_out as i64, elem);
    let src_base = if y.is_atom() {
        (y.as_ptr() as *const u8).add(8)
    } else {
        (y.as_ptr() as *const u8).add(16)
    };
    let src = if n >= 0 {
        // Drop first n — start reading at offset n*elem.
        src_base.add(drop_count * elem)
    } else {
        // Drop last |n| — read from start, stop at n_out*elem.
        src_base
    };
    let dst = (out.as_ptr() as *mut u8).add(16);
    let bytes = n_out * elem;

    let go_parallel =
        ctx.runtime.parallel_enabled() && bytes >= parallel::PARALLEL_THRESHOLD * elem;
    if go_parallel {
        let src_addr = src as usize;
        let out_slice = core::slice::from_raw_parts_mut(dst, bytes);
        parallel::parallel_for_each_mut(out_slice, ctx, move |range, my_slice| {
            unsafe {
                let s = (src_addr as *const u8).add(range.start);
                core::ptr::copy_nonoverlapping(s, my_slice.as_mut_ptr(), my_slice.len());
            }
            Ok(())
        })?;
    } else {
        // Single memcpy; cheap. Could chunk but the kernel surface for
        // tiny copies isn't worth the overhead.
        core::ptr::copy_nonoverlapping(src, dst, bytes);
    }
    let _ = &mut out;
    Ok(out)
}

struct FloorStep {
    xs: *const f64,
    out: *mut i64,
    total: usize,
    off: usize,
}
impl FloorStep {
    fn new(xs: *const f64, out: *mut i64, total: usize) -> Self {
        Self { xs, out, total, off: 0 }
    }
}
impl ChunkStep for FloorStep {
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.total {
            return None;
        }
        let end = (self.off + FLOOR_CHUNK).min(self.total);
        unsafe {
            for i in self.off..end {
                *self.out.add(i) = (*self.xs.add(i)).floor() as i64;
            }
            let s = core::slice::from_raw_parts(self.xs.add(self.off), end - self.off);
            madvise_dontneed_slice(s);
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
    use crate::exec::block_on;

    #[test]
    fn floor_atom() {
        let ctx = Ctx::quiet();
        let a = unsafe { alloc_atom(Kind::F64, 3.7f64) };
        let r = block_on(async { unsafe { floor_async(a, &ctx).await.unwrap() } });
        assert_eq!(r.kind(), Kind::I64);
        assert_eq!(unsafe { r.atom::<i64>() }, 3);
    }

    #[test]
    fn floor_vec() {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_f64(&ctx, 4) };
        unsafe {
            v.as_mut_slice::<f64>().copy_from_slice(&[1.5, -1.5, 3.0, -0.1]);
        }
        let r = block_on(async { unsafe { floor_async(v, &ctx).await.unwrap() } });
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, -2, 3, -1]);
    }

    #[test]
    fn drop_first_n() {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, 5) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3, 4, 5]); }
        let r = block_on(async { unsafe { drop_async(2, v, &ctx).await.unwrap() } });
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[3, 4, 5]);
    }

    #[test]
    fn drop_negative_drops_from_end() {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, 5) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3, 4, 5]); }
        let r = block_on(async { unsafe { drop_async(-2, v, &ctx).await.unwrap() } });
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3]);
    }

    #[test]
    fn drop_more_than_len_is_empty() {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, 3) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]); }
        let r = block_on(async { unsafe { drop_async(10, v, &ctx).await.unwrap() } });
        assert_eq!(r.len(), 0);
    }
}
