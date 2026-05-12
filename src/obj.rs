//! `Obj` and `RefObj` — the heap-allocated cell and its refcounted handle.
//!
//! Header is 8 bytes: `meta, attr, kind, _resv, rc`. Payload starts at offset
//! 8: for atoms it is up to 8 bytes of value; for vectors it is `len: i64` at
//! offset 8 followed by contiguous `data` at offset 16.
//!
//! `Clone` / `Drop` are wired in `alloc.rs` together with `release`.

use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::{ptr, slice};

use crate::kind::Kind;

/// Heap-allocated header. Atom payload begins at offset 8; vector `len` is at
/// offset 8 with contiguous data at offset 16.
#[repr(C)]
pub struct Obj {
    pub meta: i8,
    pub attr: i8,
    pub kind: i8,
    pub _resv: u8,
    pub rc: u32,
    // payload follows at offset 8
}

/// Refcounted handle. Identical in size and layout to a raw cell pointer.
#[repr(transparent)]
pub struct RefObj(pub(crate) NonNull<Obj>);

// ---- meta byte flags ---------------------------------------------------

pub mod meta_flags {
    /// The cell was allocated via anonymous `mmap` (we own the mapping).
    /// `release` must call `munmap` rather than `dealloc`. Refcount semantics
    /// are normal — we own the lifetime.
    pub const MMAP_BACKED: i8 = 1 << 0;

    /// The cell is borrowed from an external memory region we did not allocate
    /// (e.g. a v2 file-backed loader hands us a mapping; the loader, not us,
    /// owns the lifetime). `Clone` and `Drop` are no-ops; `release` returns
    /// immediately. Setting this bit makes the cell immortal from our POV.
    pub const IS_EXTERNAL: i8 = 1 << 1;

    /// The cell is owned by a parent composite (List/Dict/Table) and lives
    /// inline in that parent's payload. `Drop` skips release.
    pub const IS_BORROWED: i8 = 1 << 2;
}

// ---- attr byte flags ---------------------------------------------------

pub mod attr_flags {
    pub const SORTED: i8 = 1 << 0;
    pub const UNIQUE: i8 = 1 << 1;
    pub const PARTED: i8 = 1 << 2;
    pub const GROUPED: i8 = 1 << 3;

    /// Vector is known to contain (or potentially contain) sentinel nulls.
    /// Binary kernels branch on this flag once at entry to skip the null
    /// mask sweep when neither input has it set.
    pub const HAS_NULLS: i8 = -128; // == 1 << 7 as i8
}

// ---- compile-time invariants ------------------------------------------

const _: () = {
    assert!(size_of::<Obj>() == 8);
    assert!(align_of::<Obj>() == 4);
    assert!(size_of::<RefObj>() == 8);
    // offsets within Obj
    // (m=0, a=1, k=2, _resv=3, rc=4)
};

// ---- accessors --------------------------------------------------------

impl RefObj {
    /// SAFETY: caller guarantees the pointer is valid and aligned.
    #[inline]
    pub unsafe fn from_raw(p: NonNull<Obj>) -> Self {
        RefObj(p)
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut Obj {
        self.0.as_ptr()
    }

    #[inline]
    pub unsafe fn kind_raw(&self) -> i8 {
        (*self.0.as_ptr()).kind
    }

    #[inline]
    pub unsafe fn kind(&self) -> Kind {
        Kind::from_raw(self.kind_raw())
    }

    #[inline]
    pub unsafe fn attr(&self) -> i8 {
        (*self.0.as_ptr()).attr
    }

    #[inline]
    pub unsafe fn meta(&self) -> i8 {
        (*self.0.as_ptr()).meta
    }

    #[inline]
    pub unsafe fn rc(&self) -> u32 {
        (*self.0.as_ptr()).rc
    }

    #[inline]
    pub unsafe fn set_attr(&mut self, flags: i8) {
        (*self.0.as_ptr()).attr |= flags;
    }

    #[inline]
    pub unsafe fn clear_attr(&mut self, flags: i8) {
        (*self.0.as_ptr()).attr &= !flags;
    }

    #[inline]
    pub unsafe fn is_atom(&self) -> bool {
        let k = self.kind_raw();
        k < 0 && k > -90
    }

    #[inline]
    pub unsafe fn is_vec(&self) -> bool {
        let k = self.kind_raw();
        k > 0 && k < 90
    }

    /// Read the atom payload at offset 8.
    #[inline]
    pub unsafe fn atom<T: Copy>(&self) -> T {
        let p = self.0.as_ptr() as *const u8;
        ptr::read_unaligned(p.add(8) as *const T)
    }

    /// Read the vector length at offset 8.
    #[inline]
    pub unsafe fn len(&self) -> i64 {
        let p = self.0.as_ptr() as *const u8;
        ptr::read_unaligned(p.add(8) as *const i64)
    }

    /// Whether the vector has zero elements. Defined for vector cells only;
    /// returns `false` for atoms (atoms always carry one value).
    #[inline]
    pub unsafe fn is_empty(&self) -> bool {
        if self.is_atom() {
            false
        } else {
            self.len() == 0
        }
    }

    /// Borrow the vector payload as `&[T]`. Caller must know `T` matches the
    /// kind.
    #[inline]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        let p = self.0.as_ptr() as *const u8;
        slice::from_raw_parts(p.add(16) as *const T, self.len() as usize)
    }

    /// Borrow the vector payload as `&mut [T]`. Caller is responsible for
    /// uniqueness (see `is_unique`).
    #[inline]
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let p = self.0.as_ptr() as *mut u8;
        slice::from_raw_parts_mut(p.add(16) as *mut T, self.len() as usize)
    }

    /// Sole owner ⇒ payload may be mutated in place without violating sharing.
    /// External cells (memory we don't own) are never unique.
    #[inline]
    pub unsafe fn is_unique(&self) -> bool {
        self.meta() & meta_flags::IS_EXTERNAL == 0 && self.rc() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_size_and_layout() {
        assert_eq!(size_of::<Obj>(), 8);
        assert_eq!(size_of::<RefObj>(), 8);
        // Verify offsets within the C-layout struct.
        let mut o = Obj { meta: 1, attr: 2, kind: 3, _resv: 4, rc: 5 };
        let base = &o as *const Obj as usize;
        let m = &o.meta as *const i8 as usize;
        let a = &o.attr as *const i8 as usize;
        let k = &o.kind as *const i8 as usize;
        let u = &o._resv as *const u8 as usize;
        let r = &o.rc as *const u32 as usize;
        assert_eq!(m - base, 0);
        assert_eq!(a - base, 1);
        assert_eq!(k - base, 2);
        assert_eq!(u - base, 3);
        assert_eq!(r - base, 4);
        o.meta = 0;
        let _ = o;
    }
}
