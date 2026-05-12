//! Element kind taxonomy.
//!
//! `Kind` is the positive base code for an element type. `Obj.kind` carries
//! the sign: vector = `+k`, atom = `−k`, same modulus. One kernel per
//! `(|x|, |y|)`.
//!
//! Aligns with the K9 type letter system (Noun.html § 4.1, see
//! `docs/plans/2026-05-12-k9-spec-implementation.md`):
//!
//! ```text
//!  b/B  boolean              1 byte
//!  c/C  character             1 byte (ANSI)
//!  g/G  byte / u8             1 byte
//!  h/H  int16                 2 bytes  (signed; K9 doc claims unsigned but
//!                                       this is unverified — kept signed
//!                                       pending K9 reference binary)
//!  i/I  int32                 4 bytes  (signed; same caveat)
//!  j/J  int64                 8 bytes  (signed; default for K integers)
//!  e/E  float32               4 bytes  (single precision; K9 default for
//!                                       bare-decimal literals — `3.1`)
//!  f/F  float64               8 bytes  (double precision; K9 `3.1f`)
//!  n/N  symbol                8 bytes  (packed ≤8 ASCII chars in i64)
//!  d/D  date                  4 bytes  (i32 days since 2001-01-01)
//!  s/S  time-second           4 bytes  (i32 seconds since midnight)
//!  t/T  time-millisecond      4 bytes  (i32 milliseconds since midnight;
//!                                       fits in i32 — 86_400_000 < 2³¹)
//!  u/U  time-microsecond      8 bytes  (i64 microseconds since midnight)
//!  v/V  time-nanosecond       8 bytes  (i64 nanoseconds since midnight)
//!  S    datetime-second       8 bytes  (i64 seconds since 2001-01-01T00:00Z)
//!  T    datetime-millisecond  8 bytes  (i64 ms since 2001-01-01T00:00Z)
//!  U    datetime-microsecond  8 bytes  (i64 µs since 2001-01-01T00:00Z)
//!  V    datetime-nanosecond   8 bytes  (i64 ns since 2001-01-01T00:00Z)
//!  L    mixed list                     (Vec<RefObj>)
//! ```
//!
//! **Temporal epoch is 2001-01-01 00:00:00 UTC** for `Date` and all `Dt*`
//! kinds. Time-of-day kinds (`TimeS`/Ms/Us/Ns) are anchored at midnight of
//! an unspecified day — they're just durations since midnight.

use core::mem;

#[repr(i8)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Kind {
    List   = 0,    // generic mixed list; payload is &[RefObj]

    Bool   = 1,    // u8 storage, no null
    U8     = 4,    // K9 `g` — byte, 0..255
    I16    = 5,    // K9 `h`
    I32    = 6,    // K9 `i`
    I64    = 7,    // K9 `j` — default int
    F32    = 8,    // K9 `e` — default float literal
    F64    = 9,    // K9 `f` — explicit `3.1f`
    Char   = 10,   // K9 `c`
    Sym    = 11,   // K9 `n` — 8-char packed
    Date   = 14,   // K9 `d` — i32 days since 2001-01-01

    TimeS  = 19,   // K9 `s` — i32 seconds since midnight
    TimeMs = 20,   // K9 `t` — i32 ms since midnight
    TimeUs = 21,   // K9 `u` — i64 µs since midnight
    TimeNs = 22,   // K9 `v` — i64 ns since midnight

    DtS    = 23,   // K9 `S` — i64 sec since 2001-01-01T00:00:00Z
    DtMs   = 24,   // K9 `T` — i64 ms  since 2001-01-01T00:00:00Z
    DtUs   = 25,   // K9 `U` — i64 µs  since 2001-01-01T00:00:00Z
    DtNs   = 26,   // K9 `V` — i64 ns  since 2001-01-01T00:00:00Z

    Table  = 98,   // payload[0] is a Dict obj
    Dict   = 99,   // generic 2-list: [keys, values]

    /// User-defined lambda. Atom-shaped (single value, no vector form).
    /// Payload at offset 8 is a `*mut LambdaInner` — `Box`-allocated outside
    /// the cell, dropped by the release path special-casing this kind.
    Lambda = 30,
}

/// K9 temporal epoch — midnight UTC of 2001-01-01.
///
/// Date stores days since this epoch in i32 (range ≈ ±5.8M years).
/// Datetime stores seconds/ms/µs/ns since this epoch in i64.
/// Time-of-day stores units since midnight (zero of an unspecified day).
pub const K9_EPOCH_YEAR: i32 = 2001;
pub const K9_EPOCH_MONTH: i32 = 1;
pub const K9_EPOCH_DAY: i32 = 1;

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
    /// `raw` must correspond to a defined variant (after taking
    /// `unsigned_abs`).
    #[inline]
    pub fn from_raw(raw: i8) -> Kind {
        // SAFETY: every Kind variant fits in u8; |raw| ≤ 99 by construction.
        unsafe { mem::transmute::<i8, Kind>(raw.unsigned_abs() as i8) }
    }

    /// Element size in bytes for typed vector payloads. Composite kinds
    /// report the size of a `RefObj` (8 bytes) since their payload is
    /// `[RefObj]`.
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
            Kind::Lambda => 8, // *mut LambdaInner
        }
    }

    /// Whether this kind has a designated null sentinel.
    #[inline]
    pub const fn has_null(self) -> bool {
        !matches!(
            self,
            Kind::Bool
                | Kind::U8
                | Kind::Char
                | Kind::List
                | Kind::Dict
                | Kind::Table
                | Kind::Lambda
        )
    }

    /// K9 letter for the atom form of this kind: `b c g h i j e f n d s t u v`
    /// for atoms, plus `S T U V` for datetime atoms (note uppercase here).
    ///
    /// Composite kinds (`List`, `Dict`, `Table`) report `?` since K9
    /// doesn't have a single-letter atom code for them.
    #[inline]
    pub const fn letter_atom(self) -> char {
        match self {
            Kind::Bool => 'b',
            Kind::Char => 'c',
            Kind::U8 => 'g',
            Kind::I16 => 'h',
            Kind::I32 => 'i',
            Kind::I64 => 'j',
            Kind::F32 => 'e',
            Kind::F64 => 'f',
            Kind::Sym => 'n',
            Kind::Date => 'd',
            Kind::TimeS => 's',
            Kind::TimeMs => 't',
            Kind::TimeUs => 'u',
            Kind::TimeNs => 'v',
            Kind::DtS => 'S',
            Kind::DtMs => 'T',
            Kind::DtUs => 'U',
            Kind::DtNs => 'V',
            Kind::List => 'L', // K9 reports `L for mixed list atom (no atom form)
            Kind::Dict => '!', // ad-hoc; K9 has no atom letter
            Kind::Table => '+', // ad-hoc; K9 has no atom letter
            Kind::Lambda => 'F', // K9 reports `F for lambda
        }
    }

    /// K9 letter for the vector form of this kind: uppercase
    /// `B C G H I J E F N D` and `L` for mixed list. **Time and
    /// datetime kinds collide on uppercase** (`S T U V` are datetime
    /// atoms in K9), so vector-time-second prints as itself with no
    /// vector glyph in the K9 spec — we return the same letter as the
    /// atom form for `Time*` kinds and reserve uppercase for `Dt*`.
    #[inline]
    pub const fn letter_vec(self) -> char {
        match self {
            Kind::Bool => 'B',
            Kind::Char => 'C',
            Kind::U8 => 'G',
            Kind::I16 => 'H',
            Kind::I32 => 'I',
            Kind::I64 => 'J',
            Kind::F32 => 'E',
            Kind::F64 => 'F',
            Kind::Sym => 'N',
            Kind::Date => 'D',
            // Time-of-day vectors retain lowercase (K9 ambiguity workaround).
            Kind::TimeS => 's',
            Kind::TimeMs => 't',
            Kind::TimeUs => 'u',
            Kind::TimeNs => 'v',
            Kind::DtS => 'S',
            Kind::DtMs => 'T',
            Kind::DtUs => 'U',
            Kind::DtNs => 'V',
            Kind::List => 'L',
            Kind::Dict => '!',
            Kind::Table => '+',
            Kind::Lambda => 'F',
        }
    }

    /// Look up the K9 atom-letter for a kind. Returns `None` for letters
    /// that don't map cleanly back (composite types).
    pub fn from_letter_atom(c: char) -> Option<Kind> {
        Some(match c {
            'b' => Kind::Bool,
            'c' => Kind::Char,
            'g' => Kind::U8,
            'h' => Kind::I16,
            'i' => Kind::I32,
            'j' => Kind::I64,
            'e' => Kind::F32,
            'f' => Kind::F64,
            'n' => Kind::Sym,
            'd' => Kind::Date,
            's' => Kind::TimeS,
            't' => Kind::TimeMs,
            'u' => Kind::TimeUs,
            'v' => Kind::TimeNs,
            'S' => Kind::DtS,
            'T' => Kind::DtMs,
            'U' => Kind::DtUs,
            'V' => Kind::DtNs,
            'L' => Kind::List,
            _ => return None,
        })
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

/// Same-modulus invariant check. Holds for every `Kind` variant by
/// construction.
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
    check!(DtS);
    check!(DtMs);
    check!(DtUs);
    check!(DtNs);
    check!(Lambda);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_round_trip_all_atom_kinds() {
        // Every kind that has an atom letter round-trips through
        // letter_atom -> from_letter_atom.
        let kinds = [
            Kind::Bool, Kind::Char, Kind::U8, Kind::I16, Kind::I32,
            Kind::I64, Kind::F32, Kind::F64, Kind::Sym, Kind::Date,
            Kind::TimeS, Kind::TimeMs, Kind::TimeUs, Kind::TimeNs,
            Kind::DtS, Kind::DtMs, Kind::DtUs, Kind::DtNs, Kind::List,
        ];
        for k in kinds {
            let c = k.letter_atom();
            let recovered = Kind::from_letter_atom(c)
                .unwrap_or_else(|| panic!("no round-trip for kind {:?} (letter {:?})", k, c));
            assert_eq!(recovered, k, "round-trip mismatch for {:?}", k);
        }
    }

    #[test]
    fn k9_atom_letters_match_spec() {
        // Spot-check against K9 Noun.html § 4.1
        assert_eq!(Kind::Bool.letter_atom(), 'b');
        assert_eq!(Kind::I64.letter_atom(), 'j');
        assert_eq!(Kind::F32.letter_atom(), 'e');
        assert_eq!(Kind::F64.letter_atom(), 'f');
        assert_eq!(Kind::Date.letter_atom(), 'd');
        assert_eq!(Kind::TimeS.letter_atom(), 's');
        assert_eq!(Kind::TimeNs.letter_atom(), 'v');
        assert_eq!(Kind::DtMs.letter_atom(), 'T');
    }

    #[test]
    fn k9_vector_letters_are_uppercase_for_uniform_data() {
        assert_eq!(Kind::Bool.letter_vec(), 'B');
        assert_eq!(Kind::I64.letter_vec(), 'J');
        assert_eq!(Kind::F64.letter_vec(), 'F');
        assert_eq!(Kind::Date.letter_vec(), 'D');
        assert_eq!(Kind::List.letter_vec(), 'L');
    }
}
