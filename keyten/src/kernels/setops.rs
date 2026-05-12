//! Set-ish operations: `^` sort, `?` unique.
//!
//! **Performance note (parallel TODO)**: both kernels are sequential for
//! now. Parallel sort (radix for i64, samplesort for f64) and parallel
//! unique (parallel hash with merging) are tracked as Phase 2.x work —
//! see `docs/plans/2026-05-12-k9-spec-implementation.md`. Sequential
//! pdqsort (`sort_unstable`) handles 100M i64 in well under a second on
//! modern hardware which is acceptable for the v1 surface; the parallel
//! upgrade lands when it's worth ~50 lines of test+bench scaffolding.

use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::obj::RefObj;
use std::collections::HashSet;

/// Monadic `^x` — sort ascending. Atom → identity; vector → new sorted
/// vector (input unchanged).
pub fn sort_asc(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if x.is_atom() {
        return Ok(x);
    }
    match x.kind() {
        Kind::I64 => {
            let xs = unsafe { x.as_slice::<i64>() };
            let n = xs.len();
            let mut out = unsafe { alloc_vec_i64(ctx, n as i64) };
            unsafe {
                let os = out.as_mut_slice::<i64>();
                os.copy_from_slice(xs);
                os.sort_unstable();
            }
            Ok(out)
        }
        Kind::F64 => {
            let xs = unsafe { x.as_slice::<f64>() };
            let n = xs.len();
            let mut out = unsafe { alloc_vec_f64(ctx, n as i64) };
            unsafe {
                let os = out.as_mut_slice::<f64>();
                os.copy_from_slice(xs);
                // f64 sort: total order from `total_cmp` handles NaN
                // deterministically (NaNs sort to the end). K9 nulls are
                // NaN — this matches "nulls last" convention.
                os.sort_unstable_by(|a, b| a.total_cmp(b));
            }
            Ok(out)
        }
        _ => Err(KernelErr::Type),
    }
}

/// Monadic `?x` — distinct values in first-seen order. Atom → 1-element
/// vector containing the atom; vector → vector of distinct values.
pub fn unique(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if x.is_atom() {
        // K convention: `?atom` returns a 1-element vector of the same kind.
        // Reuse the enlist machinery.
        return crate::kernels::monad::enlist(x, ctx);
    }
    match x.kind() {
        Kind::I64 => {
            let xs = unsafe { x.as_slice::<i64>() };
            let mut seen: HashSet<i64> = HashSet::with_capacity(xs.len());
            let mut out_vals: Vec<i64> = Vec::with_capacity(xs.len());
            for &v in xs {
                if seen.insert(v) {
                    out_vals.push(v);
                }
            }
            let mut out = unsafe { alloc_vec_i64(ctx, out_vals.len() as i64) };
            unsafe { out.as_mut_slice::<i64>().copy_from_slice(&out_vals); }
            Ok(out)
        }
        Kind::F64 => {
            let xs = unsafe { x.as_slice::<f64>() };
            // f64 isn't Hash/Eq directly; we hash the bit pattern so that
            // distinct NaNs with different payloads are NOT collapsed
            // (matching K convention for null distinctness when nulls land).
            let mut seen: HashSet<u64> = HashSet::with_capacity(xs.len());
            let mut out_vals: Vec<f64> = Vec::with_capacity(xs.len());
            for &v in xs {
                if seen.insert(v.to_bits()) {
                    out_vals.push(v);
                }
            }
            let mut out = unsafe { alloc_vec_f64(ctx, out_vals.len() as i64) };
            unsafe { out.as_mut_slice::<f64>().copy_from_slice(&out_vals); }
            Ok(out)
        }
        _ => Err(KernelErr::Type),
    }
}

/// Monadic `%x` — square root. F64 atom/vec → F64 atom/vec; I64 promotes
/// to F64 (since sqrt of integers usually isn't integer).
pub fn sqrt(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    match x.kind() {
        Kind::F64 => {
            if x.is_atom() {
                Ok(unsafe { alloc_atom(Kind::F64, x.atom::<f64>().sqrt()) })
            } else {
                let xs = unsafe { x.as_slice::<f64>() };
                let n = xs.len();
                let mut out = unsafe { alloc_vec_f64(ctx, n as i64) };
                unsafe {
                    let os = out.as_mut_slice::<f64>();
                    for i in 0..n {
                        os[i] = xs[i].sqrt();
                    }
                }
                Ok(out)
            }
        }
        Kind::I64 => {
            if x.is_atom() {
                let v = unsafe { x.atom::<i64>() } as f64;
                Ok(unsafe { alloc_atom(Kind::F64, v.sqrt()) })
            } else {
                let xs = unsafe { x.as_slice::<i64>() };
                let n = xs.len();
                let mut out = unsafe { alloc_vec_f64(ctx, n as i64) };
                unsafe {
                    let os = out.as_mut_slice::<f64>();
                    for i in 0..n {
                        os[i] = (xs[i] as f64).sqrt();
                    }
                }
                Ok(out)
            }
        }
        _ => Err(KernelErr::Type),
    }
}

/// Monadic `|x` — reverse. Returns a new vector with elements in reverse
/// order. Atom → identity.
pub fn reverse(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if x.is_atom() {
        return Ok(x);
    }
    let kind = x.kind();
    if matches!(kind, Kind::List | Kind::Dict | Kind::Table) {
        return Err(KernelErr::Type);
    }
    let elem = kind.elem_size();
    let n = x.len() as usize;
    let mut out =
        unsafe { crate::alloc::alloc_vec(ctx, kind.vec(), n as i64, elem) };
    if n == 0 {
        return Ok(out);
    }
    unsafe {
        let src = (x.as_ptr() as *const u8).add(16);
        let dst = (out.as_ptr() as *mut u8).add(16);
        // Reverse copy: dst[i] = src[n-1-i] per element.
        for i in 0..n {
            let s = src.add((n - 1 - i) * elem);
            let d = dst.add(i * elem);
            core::ptr::copy_nonoverlapping(s, d, elem);
        }
    }
    Ok(out)
}

/// Monadic `&x` — where. Returns indices where `x` is non-zero. For Bool
/// input this gives the indices of `true` entries; for I64 input, indices
/// of non-zero entries. Output is I64 vector.
pub fn where_indices(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let kind = x.kind();
    if x.is_atom() {
        return Err(KernelErr::Type); // K convention: `&` on atom is "where" which only makes sense on a counted thing
    }
    match kind {
        Kind::Bool => {
            let xs = unsafe { x.as_slice::<u8>() };
            let mut indices: Vec<i64> = Vec::with_capacity(xs.len());
            for (i, &b) in xs.iter().enumerate() {
                if b != 0 {
                    indices.push(i as i64);
                }
            }
            let mut out = unsafe { alloc_vec_i64(ctx, indices.len() as i64) };
            unsafe { out.as_mut_slice::<i64>().copy_from_slice(&indices); }
            Ok(out)
        }
        Kind::I64 => {
            let xs = unsafe { x.as_slice::<i64>() };
            let mut indices: Vec<i64> = Vec::with_capacity(xs.len());
            for (i, &v) in xs.iter().enumerate() {
                if v != 0 {
                    indices.push(i as i64);
                }
            }
            let mut out = unsafe { alloc_vec_i64(ctx, indices.len() as i64) };
            unsafe { out.as_mut_slice::<i64>().copy_from_slice(&indices); }
            Ok(out)
        }
        _ => Err(KernelErr::Type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::alloc_vec_i64;
    use crate::ctx::Ctx;

    fn make_i64_vec(d: &[i64]) -> RefObj {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, d.len() as i64) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(d); }
        v
    }

    #[test]
    fn sort_int_vec() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[5, 2, 8, 1, 9, 3]);
        let r = sort_asc(v, &ctx).unwrap();
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3, 5, 8, 9]);
    }

    #[test]
    fn sort_empty_vec() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[]);
        let r = sort_asc(v, &ctx).unwrap();
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn unique_int_preserves_first_seen_order() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[3, 1, 2, 1, 3, 4, 2]);
        let r = unique(v, &ctx).unwrap();
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[3, 1, 2, 4]);
    }

    #[test]
    fn sqrt_i64_atom() {
        let ctx = Ctx::quiet();
        let a = unsafe { alloc_atom(Kind::I64, 16i64) };
        let r = sqrt(a, &ctx).unwrap();
        assert_eq!(r.kind(), Kind::F64);
        assert_eq!(unsafe { r.atom::<f64>() }, 4.0);
    }

    #[test]
    fn reverse_int_vec() {
        let ctx = Ctx::quiet();
        let v = make_i64_vec(&[1, 2, 3, 4, 5]);
        let r = reverse(v, &ctx).unwrap();
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[5, 4, 3, 2, 1]);
    }

    #[test]
    fn where_on_bool_vec() {
        let ctx = Ctx::quiet();
        let mut v = unsafe { crate::alloc::alloc_vec(&ctx, Kind::Bool.vec(), 5, 1) };
        unsafe { v.as_mut_slice::<u8>().copy_from_slice(&[0, 1, 0, 1, 1]); }
        let r = where_indices(v, &ctx).unwrap();
        assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 3, 4]);
    }
}
