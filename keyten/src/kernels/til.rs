//! `!n` — til. Generates the i64 vector `0 1 2 … n−1`.

use crate::alloc::{alloc_atom, alloc_vec_i64};
use crate::chunk::{drive_async, ChunkStep};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kernels::I64_CHUNK;
use crate::kind::Kind;
use crate::obj::RefObj;

pub struct TilI64<'a> {
    out: &'a mut [i64],
    off: usize,
    chunk: usize,
}

impl<'a> TilI64<'a> {
    pub fn new(out: &'a mut [i64], chunk: usize) -> Self {
        TilI64 { out, off: 0, chunk: chunk.max(1) }
    }
}

impl<'a> ChunkStep for TilI64<'a> {
    #[inline]
    fn step(&mut self) -> Option<usize> {
        if self.off >= self.out.len() {
            return None;
        }
        let end = (self.off + self.chunk).min(self.out.len());
        for i in self.off..end {
            self.out[i] = i as i64;
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
        // Empty I64 vector.
        let v = unsafe { alloc_vec_i64(ctx, 0) };
        return Ok(v);
    }
    let mut out = unsafe { alloc_vec_i64(ctx, count) };
    let chunk = if ctx.chunk_elems != 0 {
        ctx.chunk_elems
    } else {
        I64_CHUNK
    };
    {
        let os = unsafe { out.as_mut_slice::<i64>() };
        let mut k = TilI64::new(os, chunk);
        drive_async(&mut k, ctx).await?;
    }
    let _ = alloc_atom::<i64>; // suppress unused-import lint if alloc_atom not used
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
