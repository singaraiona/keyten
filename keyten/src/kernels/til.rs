//! `!n` — til. Generates the i64 vector `0 1 2 … n−1`.

use crate::alloc::{alloc_atom, alloc_vec_i64};
use crate::chunk::{drive_async, drive_sync, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kernels::I64_CHUNK;
use crate::kind::Kind;
use crate::obj::RefObj;
use crate::parallel;

pub struct TilI64<'a> {
    /// Output slice — for the parallel path this is each worker's sub-slice.
    out: &'a mut [i64],
    /// Element index of `out[0]` in the global output. The parallel
    /// dispatch hands each worker a sub-range starting at `base`, so the
    /// value written at `out[i]` must be `base + i`, not just `i`.
    base: usize,
    off: usize,
    chunk: usize,
}

impl<'a> TilI64<'a> {
    pub fn new(out: &'a mut [i64], chunk: usize) -> Self {
        TilI64 { out, base: 0, off: 0, chunk: chunk.max(1) }
    }

    /// Constructor for the parallel path: takes the worker's sub-slice and
    /// the global offset of its first element.
    pub fn new_with_base(out: &'a mut [i64], base: usize, chunk: usize) -> Self {
        TilI64 { out, base, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for TilI64<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.out.len() {
            return None;
        }
        let end = (self.off + self.chunk).min(self.out.len());
        let base = self.base;
        for i in self.off..end {
            self.out[i] = (base + i) as i64;
        }
        let n = end - self.off;
        self.off = end;
        Some(n)
    }
}

/// `!n` — generate `0 1 … n−1` as an i64 vector. `n` must be a non-negative
/// I64-kinded atom.
pub async fn til_async(n: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let count = read_count(&n)?;
    if count == 0 {
        let v = unsafe { alloc_vec_i64(ctx, 0) };
        return Ok(v);
    }
    let mut out = unsafe { alloc_vec_i64(ctx, count) };
    let chunk = if ctx.chunk_elems != 0 {
        ctx.chunk_elems
    } else {
        I64_CHUNK
    };
    let n_usize = count as usize;
    let go_parallel = ctx.runtime.parallel_enabled() && n_usize >= parallel::PARALLEL_THRESHOLD;
    {
        let os = unsafe { out.as_mut_slice::<i64>() };
        if go_parallel {
            // Each worker fills its sub-slice with `base..base+len` values.
            // Without `base`, all workers would write [0, 1, …] starting at
            // index 0 in their local slice — producing a vector full of the
            // global sub-range starts instead of the monotonically increasing
            // sequence.
            parallel::parallel_for_each_mut(os, ctx, |range, my_os| {
                let mut k = TilI64::new_with_base(my_os, range.start, chunk);
                drive_sync(&mut k, ctx)
            })?;
        } else {
            let mut k = TilI64::new(os, chunk);
            drive_async(&mut k, ctx).await?;
        }
    }
    let _ = alloc_atom::<i64>;
    Ok(out)
}

pub fn til(n: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(til_async(n, ctx))
}

fn read_count(n: &RefObj) -> Result<i64, KernelErr> {
    if !n.is_atom() || n.kind() != Kind::I64 {
        return Err(KernelErr::Type);
    }
    let v = unsafe { n.atom::<i64>() };
    if v < 0 {
        return Err(KernelErr::Shape);
    }
    Ok(v)
}
