//! Monadic verb kernels: `@` type, `#` count, `,` enlist.
//!
//! These three share a file because each is small (≤30 LOC), reads a
//! `RefObj`'s metadata + length, and allocates a single result. Future verbs
//! that are also "metadata read + small result" live here too.

use crate::alloc::{alloc_atom, alloc_vec};
use crate::ctx::{Ctx, KernelErr};
use crate::kind::Kind;
use crate::nulls::NULL_I64;
use crate::obj::RefObj;
use crate::sym::{intern, Sym};

/// `@x` — return the kind of `x` as a one-character symbol.
///
/// K9 convention: atoms get a lowercase letter; uniform vectors get the
/// uppercase form of the same letter; the mixed list gets `` `L ``.
/// Datetime atoms keep their uppercase letter (see `Kind::letter_atom` /
/// `Kind::letter_vec`).
///
/// Result is an atom of kind `Sym`.
pub fn type_of(x: RefObj) -> Result<RefObj, KernelErr> {
    let raw = x.kind_raw();
    let kind = Kind::from_raw(raw);
    let c = if raw < 0 {
        kind.letter_atom()
    } else {
        kind.letter_vec()
    };
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    // Symbols are ASCII-only; the K9 letters are all ASCII.
    debug_assert!(s.len() == 1, "K9 type letters are ASCII");
    let sym = intern(s).expect("ASCII single-char symbol");
    Ok(unsafe { alloc_atom(Kind::Sym, sym) })
}

/// `#x` — count: number of elements in `x`. Atom → `1`; vector → length;
/// dict → number of entries (length of values vector); table → row count.
pub fn count(x: RefObj) -> Result<RefObj, KernelErr> {
    let n: i64 = if x.is_atom() {
        1
    } else if x.kind() == Kind::Dict {
        // Storage len is 2 (keys, values RefObjs). Entry count is the
        // length of the values vector.
        let values = crate::kernels::dict::dict_values(&x)?;
        values.len()
    } else {
        x.len()
    };
    Ok(unsafe { alloc_atom(Kind::I64, n) })
}

/// `,x` — enlist: wrap `x` as a 1-element list of `x`'s kind. For a vector
/// input, this is a no-op semantically (already a list of its elements).
/// For an atom of kind `K`, the result is a uniform vector of kind `K`
/// holding the one element. For a composite (List/Dict/Table) input, the
/// result is a mixed list containing the one entry.
pub fn enlist(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if !x.is_atom() {
        // K idiom: `,` on a vector is identity (it's already a 1-element-of-lists
        // view in some interpretations, but for v1 we mirror K9 — wrap in
        // a mixed list so `count` reports 1 regardless).
        let mut out = unsafe { alloc_vec(ctx, Kind::List.vec(), 1, Kind::List.elem_size()) };
        unsafe {
            let slot = out.as_mut_slice::<RefObj>().as_mut_ptr();
            // Write the RefObj by raw pointer to avoid Drop on uninitialised slot.
            core::ptr::write(slot, x);
        }
        return Ok(out);
    }
    let kind = Kind::from_raw(x.kind_raw());
    // Atom-to-1-element-vector: same kind, length 1, copy the 8-byte payload.
    let mut out = unsafe { alloc_vec(ctx, kind.vec(), 1, kind.elem_size()) };
    unsafe {
        // Source payload is at offset 8 of `x`'s cell; destination is at
        // offset 16 (vector data start) of `out`'s cell. Copy `elem_size`
        // bytes — caller's kind invariant guarantees correctness.
        let src = (x.as_ptr() as *const u8).add(8);
        let dst = (out.as_ptr() as *mut u8).add(16);
        core::ptr::copy_nonoverlapping(src, dst, kind.elem_size());
    }
    let _ = (); // suppress 'unused mut' on `out` if elem_size is small enough
    let _ = ctx;
    let _ = &mut out; // keep `out: mut` shape consistent with vector kernels
    Ok(out)
}

/// Suppress an unused-import warning for `Sym` when callers only use it
/// transitively. `intern` returns a `Sym` already.
#[allow(dead_code)]
fn _sym_marker(s: Sym) -> Sym {
    s
}

/// Monadic `~x` — not. For Bool: invert. For I64/F64: 1 if zero, 0 if
/// non-zero (K convention). Returns Bool atom or Bool vector.
pub fn not(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let kind = x.kind();
    match kind {
        Kind::Bool => {
            if x.is_atom() {
                let b = unsafe { x.atom::<u8>() };
                Ok(unsafe { alloc_atom(Kind::Bool, if b == 0 { 1u8 } else { 0u8 }) })
            } else {
                let xs = unsafe { x.as_slice::<u8>() };
                let n = xs.len();
                let mut out = unsafe { alloc_vec(ctx, Kind::Bool.vec(), n as i64, 1) };
                unsafe {
                    let os = out.as_mut_slice::<u8>();
                    for i in 0..n {
                        os[i] = if xs[i] == 0 { 1 } else { 0 };
                    }
                }
                Ok(out)
            }
        }
        Kind::I64 => {
            if x.is_atom() {
                let v = unsafe { x.atom::<i64>() };
                Ok(unsafe { alloc_atom(Kind::Bool, if v == 0 { 1u8 } else { 0u8 }) })
            } else {
                let xs = unsafe { x.as_slice::<i64>() };
                let n = xs.len();
                let mut out = unsafe { alloc_vec(ctx, Kind::Bool.vec(), n as i64, 1) };
                unsafe {
                    let os = out.as_mut_slice::<u8>();
                    for i in 0..n {
                        os[i] = if xs[i] == 0 { 1 } else { 0 };
                    }
                }
                Ok(out)
            }
        }
        Kind::F64 => {
            if x.is_atom() {
                let v = unsafe { x.atom::<f64>() };
                Ok(unsafe { alloc_atom(Kind::Bool, if v == 0.0 { 1u8 } else { 0u8 }) })
            } else {
                let xs = unsafe { x.as_slice::<f64>() };
                let n = xs.len();
                let mut out = unsafe { alloc_vec(ctx, Kind::Bool.vec(), n as i64, 1) };
                unsafe {
                    let os = out.as_mut_slice::<u8>();
                    for i in 0..n {
                        os[i] = if xs[i] == 0.0 { 1 } else { 0 };
                    }
                }
                Ok(out)
            }
        }
        _ => Err(KernelErr::Type),
    }
}

/// Monadic `$x` — convert an atom to a char vector (string). For now we
/// support I64, F64, Bool, Sym atoms; vectors return KernelErr::Type
/// pending a per-kind "format each element" pass.
pub fn string_of(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    if !x.is_atom() {
        // TODO: dollar on a vector → list of char-vectors per element.
        return Err(KernelErr::Type);
    }
    let kind = x.kind();
    let s: String = unsafe {
        match kind {
            Kind::Bool => {
                let b = x.atom::<u8>();
                if b == 0 { "0b".into() } else { "1b".into() }
            }
            Kind::I64 => {
                let v = x.atom::<i64>();
                if v == NULL_I64 { "0N".into() } else { format!("{v}") }
            }
            Kind::F64 => {
                let v = x.atom::<f64>();
                if v.is_nan() {
                    "0n".into()
                } else if v.is_infinite() {
                    if v > 0.0 { "0w".into() } else { "-0w".into() }
                } else {
                    // K9 prints floats with a trailing `.` when integer-valued
                    // to disambiguate from int. For now use Rust's default.
                    format!("{v}")
                }
            }
            Kind::Sym => {
                let s = x.atom::<Sym>();
                let bytes = s.0.to_le_bytes();
                let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
                std::str::from_utf8(&bytes[..len])
                    .map_err(|_| KernelErr::Type)?
                    .to_string()
            }
            Kind::Char => {
                let c = x.atom::<u8>();
                (c as char).to_string()
            }
            _ => return Err(KernelErr::Type),
        }
    };

    // Build a Kind::Char vector with the string bytes.
    let bytes = s.as_bytes();
    let mut out = unsafe { alloc_vec(ctx, Kind::Char.vec(), bytes.len() as i64, 1) };
    unsafe {
        let dst = (out.as_ptr() as *mut u8).add(16);
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_of_atoms() {
        let i = unsafe { alloc_atom(Kind::I64, 42i64) };
        let r = type_of(i).unwrap();
        let s = unsafe { r.atom::<Sym>() };
        assert_eq!(intern("j").unwrap(), s);
    }

    #[test]
    fn type_of_float_atom() {
        let f = unsafe { alloc_atom(Kind::F64, 3.14f64) };
        let r = type_of(f).unwrap();
        let s = unsafe { r.atom::<Sym>() };
        assert_eq!(intern("f").unwrap(), s);
    }

    #[test]
    fn count_atom_is_one() {
        let i = unsafe { alloc_atom(Kind::I64, 0i64) };
        let r = count(i).unwrap();
        let n = unsafe { r.atom::<i64>() };
        assert_eq!(n, 1);
    }

    #[test]
    fn enlist_atom_produces_unit_vector() {
        let ctx = Ctx::quiet();
        let i = unsafe { alloc_atom(Kind::I64, 42i64) };
        let r = enlist(i, &ctx).unwrap();
        assert!(r.is_vec());
        assert_eq!(r.kind(), Kind::I64);
        assert_eq!(r.len(), 1);
        let xs = unsafe { r.as_slice::<i64>() };
        assert_eq!(xs[0], 42);
    }
}
