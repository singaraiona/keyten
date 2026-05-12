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

/// `vec @ idx` — read element at integer index. Atom idx → element atom;
/// out-of-range returns the kind's null sentinel (for kinds that have one)
/// or KernelErr::Shape (for kinds without nulls).
pub fn vec_index(v: &RefObj, idx: i64) -> Result<RefObj, KernelErr> {
    if !v.is_vec() {
        return Err(KernelErr::Type);
    }
    let n = v.len();
    if idx < 0 || idx >= n {
        return null_of(v.kind());
    }
    let i = idx as usize;
    let kind = v.kind();
    let r = unsafe {
        match kind {
            Kind::I64 => crate::alloc::alloc_atom(Kind::I64, v.as_slice::<i64>()[i]),
            Kind::F64 => crate::alloc::alloc_atom(Kind::F64, v.as_slice::<f64>()[i]),
            Kind::Bool | Kind::U8 | Kind::Char => {
                crate::alloc::alloc_atom(kind, v.as_slice::<u8>()[i])
            }
            Kind::Sym => crate::alloc::alloc_atom(
                Kind::Sym,
                v.as_slice::<crate::sym::Sym>()[i],
            ),
            Kind::Date | Kind::TimeS | Kind::TimeMs | Kind::I32 | Kind::F32 => {
                crate::alloc::alloc_atom(kind, v.as_slice::<i32>()[i])
            }
            Kind::I16 => crate::alloc::alloc_atom(Kind::I16, v.as_slice::<i16>()[i]),
            Kind::TimeUs | Kind::TimeNs | Kind::DtS | Kind::DtMs | Kind::DtUs | Kind::DtNs => {
                crate::alloc::alloc_atom(kind, v.as_slice::<i64>()[i])
            }
            _ => return Err(KernelErr::Type),
        }
    };
    Ok(r)
}

/// `dict @ key` — look up the value for `key`. Linear scan through the
/// dict's keys; on match, return the corresponding value as an atom.
/// On miss, return the value-kind's null sentinel.
pub fn dict_lookup(d: &RefObj, key: &RefObj) -> Result<RefObj, KernelErr> {
    let keys = dict_keys(d)?;
    let values = dict_values(d)?;
    if !key.is_atom() {
        return Err(KernelErr::Type);
    }
    let kk = key.kind();
    if kk != keys.kind() {
        return Err(KernelErr::Type);
    }
    let idx = unsafe {
        match kk {
            Kind::I64 => {
                let k = key.atom::<i64>();
                keys.as_slice::<i64>().iter().position(|&v| v == k)
            }
            Kind::Sym => {
                let k = key.atom::<crate::sym::Sym>();
                keys.as_slice::<crate::sym::Sym>().iter().position(|&v| v == k)
            }
            Kind::F64 => {
                let k = key.atom::<f64>();
                // Use bit-pattern comparison for total order (NaN-safe).
                let kb = k.to_bits();
                keys.as_slice::<f64>()
                    .iter()
                    .position(|&v| v.to_bits() == kb)
            }
            _ => return Err(KernelErr::Type),
        }
    };
    match idx {
        Some(i) => vec_index(&values, i as i64),
        None => null_of(values.kind()),
    }
}

/// `+dict` — flip a dict into a table. In v1 this is a kind-tag change
/// with no payload reshape, since our simple dicts already store
/// `[keys, values]` in the layout a table expects. Real K9 tables
/// require each value column to be a vector and all columns to have
/// the same length — once mixed-list values land, this kernel will
/// validate that invariant.
pub fn flip_dict_to_table(d: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if d.kind() != Kind::Dict {
        return Err(KernelErr::Type);
    }
    let keys = dict_keys(&d)?;
    let values = dict_values(&d)?;
    let mut out = unsafe { alloc_vec(ctx, Kind::Table.vec(), 2, 8) };
    unsafe {
        let slots = out.as_mut_slice::<RefObj>().as_mut_ptr();
        core::ptr::write(slots, keys);
        core::ptr::write(slots.add(1), values);
    }
    Ok(out)
}

fn null_of(kind: Kind) -> Result<RefObj, KernelErr> {
    match kind {
        Kind::I64 => Ok(unsafe { crate::alloc::alloc_atom(Kind::I64, crate::nulls::NULL_I64) }),
        Kind::F64 => Ok(unsafe { crate::alloc::alloc_atom(Kind::F64, crate::nulls::NULL_F64) }),
        Kind::Sym => Ok(unsafe { crate::alloc::alloc_atom(Kind::Sym, crate::nulls::NULL_SYM) }),
        _ => Err(KernelErr::Shape),
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
