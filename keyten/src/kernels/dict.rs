//! Dictionary kernel: `keys ! values` constructor and basic dict access.
//!
//! Storage layout (per `kind.rs`):
//!   Dict is a 2-element composite cell. Its payload (offset 16+) is two
//!   `RefObj` slots — `[keys, values]`. Both keys and values are
//!   themselves `RefObj`s pointing to vectors.

use crate::alloc::alloc_vec;
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::obj::RefObj;

/// `keys ! values` — build a dict. Returns a `Kind::Dict` cell whose
/// payload holds the two `RefObj` operands.
///
/// Both operands must be vectors of the same length (atom keys are
/// promoted to 1-element vectors elsewhere; for v1 we require explicit
/// vectors).
pub fn make_dict(keys: RefObj, values: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if keys.is_atom() || values.is_atom() {
        // K convention: dicts hold vector-shaped keys and values. Atom
        // operands should be enlisted first by the caller.
        return Err(KernelErr::Shape);
    }
    if keys.len() != values.len() {
        return Err(KernelErr::Shape);
    }
    // Allocate a 2-slot composite of kind Dict. elem_size for Dict is 8
    // (size_of::<RefObj>()).
    let mut out = unsafe { alloc_vec(ctx, Kind::Dict.vec(), 2, 8) };
    // Write keys, values into the two RefObj slots.
    unsafe {
        let slots = out.as_mut_slice::<RefObj>().as_mut_ptr();
        // Use raw write to avoid Drop on uninitialised slots.
        core::ptr::write(slots, keys);
        core::ptr::write(slots.add(1), values);
    }
    Ok(out)
}

/// Read the keys vector of a dict. Returns a clone (rc bump) of the keys
/// `RefObj` so the caller can manipulate it independently.
pub fn dict_keys(d: &RefObj) -> Result<RefObj, KernelErr> {
    if d.kind() != Kind::Dict {
        return Err(KernelErr::Type);
    }
    unsafe {
        let p = (d.as_ptr() as *const u8).add(16) as *const RefObj;
        Ok((*p).clone())
    }
}

/// Read the values vector of a dict.
pub fn dict_values(d: &RefObj) -> Result<RefObj, KernelErr> {
    if d.kind() != Kind::Dict {
        return Err(KernelErr::Type);
    }
    unsafe {
        let p = (d.as_ptr() as *const u8).add(16) as *const RefObj;
        Ok((*p.add(1)).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::alloc_vec_i64;

    fn make_i64_vec(d: &[i64]) -> RefObj {
        let ctx = Ctx::quiet();
        let mut v = unsafe { alloc_vec_i64(&ctx, d.len() as i64) };
        unsafe { v.as_mut_slice::<i64>().copy_from_slice(d); }
        v
    }

    #[test]
    fn build_simple_dict() {
        let ctx = Ctx::quiet();
        let k = make_i64_vec(&[1, 2, 3]);
        let v = make_i64_vec(&[10, 20, 30]);
        let d = make_dict(k, v, &ctx).unwrap();
        assert_eq!(d.kind(), Kind::Dict);

        let recovered_k = dict_keys(&d).unwrap();
        let recovered_v = dict_values(&d).unwrap();
        assert_eq!(unsafe { recovered_k.as_slice::<i64>() }, &[1, 2, 3]);
        assert_eq!(unsafe { recovered_v.as_slice::<i64>() }, &[10, 20, 30]);
    }

    #[test]
    fn mismatched_length_rejected() {
        let ctx = Ctx::quiet();
        let k = make_i64_vec(&[1, 2]);
        let v = make_i64_vec(&[10, 20, 30]);
        let r = make_dict(k, v, &ctx);
        assert!(matches!(r, Err(KernelErr::Shape)));
    }
}
