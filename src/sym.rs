//! Symbols.
//!
//! A `Sym` is an ASCII string of ≤8 bytes packed into an `i64`. The encoding is
//! bijective for that subset, so no global intern table is needed: equal strings
//! produce equal `i64`s and equality is a single 64-bit compare.

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Sym(pub i64);

#[derive(Debug, PartialEq, Eq)]
pub enum SymErr {
    TooLong,
    NonAscii,
}

/// Pack an ASCII string of ≤8 bytes into a `Sym`.
pub fn intern(s: &str) -> Result<Sym, SymErr> {
    let b = s.as_bytes();
    if b.len() > 8 {
        return Err(SymErr::TooLong);
    }
    if !b.iter().all(|&c| c < 128) {
        return Err(SymErr::NonAscii);
    }
    let mut buf = [0u8; 8];
    buf[..b.len()].copy_from_slice(b);
    Ok(Sym(i64::from_le_bytes(buf)))
}

impl Sym {
    /// Decode the symbol back to its string form (trims trailing NULs).
    pub fn as_str(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_and_compare() {
        let a = intern("abc").unwrap();
        let b = intern("abc").unwrap();
        let c = intern("abcd").unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn too_long_rejected() {
        assert_eq!(intern("123456789"), Err(SymErr::TooLong));
    }

    #[test]
    fn non_ascii_rejected() {
        assert_eq!(intern("café"), Err(SymErr::NonAscii));
    }

    #[test]
    fn empty_is_null() {
        let s = intern("").unwrap();
        assert!(s.is_null());
        assert_eq!(s, crate::nulls::NULL_SYM);
    }
}
