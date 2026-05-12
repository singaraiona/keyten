//! Element kind taxonomy.
//!
//! `Kind` is the positive base code for an element type. `Obj.kind` carries the
//! sign: vector = `+k`, atom = `−k`, same modulus. One kernel per `(|x|, |y|)`.

use core::mem;

#[repr(i8)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Kind {
    List   = 0,    // generic mixed list; payload is &[RefObj]

    Bool   = 1,    // u8 storage, no null
    U8     = 4,    // byte
    I16    = 5,
    I32    = 6,
    I64    = 7,
    F32    = 8,
    F64    = 9,
    Char   = 10,   // u8
    Sym    = 11,   // i64 (packed ≤8 ASCII bytes)
    Date   = 14,   // i32 days from epoch

    TimeS  = 19,   // i32 seconds
    TimeMs = 20,   // i32 milliseconds
    TimeUs = 21,   // i64 microseconds
    TimeNs = 22,   // i64 nanoseconds

    DtS    = 23,
    DtMs   = 24,
    DtUs   = 25,
    DtNs   = 26,

    Table  = 98,   // payload[0] is a Dict obj
    Dict   = 99,   // generic 2-list: [keys, values]
}

impl Kind {
    /// Encoded form when used as an atom: `−code`.
    #[inline]
    pub const fn atom(self) -> i8 {
        -(self as i8)
    }

    /// Encoded form when used as a vector: `+code`.
    #[inline]
    pub const fn vec(self) -> i8 {
        self as i8
    }

    /// Recover `Kind` from `Obj.kind` (atom or vector).
    ///
    /// # Safety
    /// `raw` must correspond to a defined variant (after taking `unsigned_abs`).
    #[inline]
    pub fn from_raw(raw: i8) -> Kind {
        // SAFETY: every Kind variant fits in u8; |raw| ≤ 99 by construction.
        unsafe { mem::transmute::<i8, Kind>(raw.unsigned_abs() as i8) }
    }

    /// Element size in bytes for typed vector payloads. Composite kinds report
    /// the size of a `RefObj` (8 bytes) since their payload is `[RefObj]`.
    #[inline]
    pub const fn elem_size(self) -> usize {
        match self {
            Kind::Bool | Kind::U8 | Kind::Char => 1,
            Kind::I16 => 2,
            Kind::I32 | Kind::F32 | Kind::Date | Kind::TimeS | Kind::TimeMs => 4,
            Kind::I64
            | Kind::F64
            | Kind::Sym
            | Kind::TimeUs
            | Kind::TimeNs
            | Kind::DtS
            | Kind::DtMs
            | Kind::DtUs
            | Kind::DtNs => 8,
            Kind::List | Kind::Dict | Kind::Table => 8, // size_of::<RefObj>()
        }
    }

    /// Whether this kind has a designated null sentinel.
    #[inline]
    pub const fn has_null(self) -> bool {
        !matches!(self, Kind::Bool | Kind::U8 | Kind::Char | Kind::List | Kind::Dict | Kind::Table)
    }
}

#[inline]
pub const fn is_atom_code(raw: i8) -> bool {
    raw < 0 && raw > -90
}

#[inline]
pub const fn is_vec_code(raw: i8) -> bool {
    raw > 0 && raw < 90
}

/// Same-modulus invariant check. Holds for every `Kind` variant by construction.
const _: () = {
    macro_rules! check { ($k:ident) => {
        assert!(Kind::$k.atom() == -(Kind::$k.vec()));
    } }
    check!(Bool);
    check!(U8);
    check!(I16);
    check!(I32);
    check!(I64);
    check!(F32);
    check!(F64);
    check!(Char);
    check!(Sym);
    check!(Date);
    check!(TimeS);
    check!(TimeMs);
    check!(TimeUs);
    check!(TimeNs);
};
