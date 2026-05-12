//! `Obj` and `RefObj` — the heap-allocated cell and its refcounted handle.
//!
//! Header is 8 bytes: `meta, attr, kind, _resv, rc`. Payload starts at offset
//! 8: for atoms it is up to 8 bytes of value; for vectors it is `len: i64` at
//! offset 8 followed by contiguous `data` at offset 16.
//!
//! `Clone` / `Drop` are wired in `alloc.rs` together with `release`.

use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU32, Ordering};
use core::{ptr, slice};

use crate::kind::Kind;

/// Heap-allocated header. Atom payload begins at offset 8; vector `len` is at
/// offset 8 with contiguous data at offset 16.
///
/// `rc` is `AtomicU32` so handles can cross threads (Stage 0 of v2 parallel
/// execution). `AtomicU32` is `#[repr(transparent)]` over `u32`, so the
/// header layout is byte-identical to the pre-v2 form.
#[repr(C)]
pub struct Obj {
    pub meta: i8,
    pub attr: i8,
    pub kind: i8,
    pub _resv: u8,
    pub rc: AtomicU32,
    // payload follows at offset 8
}

/// Refcounted handle. Identical in size and layout to a raw cell pointer.
#[repr(transparent)]
pub struct RefObj(pub(crate) NonNull<Obj>);

// `RefObj` carries shared ownership of an `Obj` whose `rc` is atomic, so it
// can move across threads. It is intentionally **not** `Sync` — kernel-level
// in-place mutation (gated by `is_unique`) assumes one writer at a time.
unsafe impl Send for RefObj {}

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
    /// Wrap a raw pointer.
    ///
    /// # Safety
    /// `p` must point to a properly-initialised `Obj` whose refcount the
    /// caller has just incremented (i.e. the new `RefObj` takes ownership of
    /// one count). The pointer must remain valid for the new `RefObj`'s
    /// lifetime (Drop runs `release` when rc reaches 0).
    #[inline]
    pub unsafe fn from_raw(p: NonNull<Obj>) -> Self {
        RefObj(p)
    }

    /// Raw pointer view. The pointer is non-null and properly aligned by
    /// construction; reading typed fields through it is what carries the
    /// unsafe contract (see `atom`, `as_slice`).
    #[inline]
    pub fn as_ptr(&self) -> *mut Obj {
        self.0.as_ptr()
    }

    // ---- safe untyped accessors -----------------------------------------
    //
    // These read individual bytes / fields of the header. They are safe
    // because the RefObj invariant guarantees a valid, properly-initialised
    // `Obj` lives at `self.0`. Constructors are unsafe; the methods are not.

    #[inline]
    pub fn kind_raw(&self) -> i8 {
        unsafe { (*self.0.as_ptr()).kind }
    }

    #[inline]
    pub fn kind(&self) -> Kind {
        Kind::from_raw(self.kind_raw())
    }

    #[inline]
    pub fn attr(&self) -> i8 {
        unsafe { (*self.0.as_ptr()).attr }
    }

    #[inline]
    pub fn meta(&self) -> i8 {
        unsafe { (*self.0.as_ptr()).meta }
    }

    #[inline]
    pub fn rc(&self) -> u32 {
        unsafe { (*self.0.as_ptr()).rc.load(Ordering::Relaxed) }
    }

    #[inline]
    pub fn set_attr(&mut self, flags: i8) {
        unsafe {
            (*self.0.as_ptr()).attr |= flags;
        }
    }

    #[inline]
    pub fn clear_attr(&mut self, flags: i8) {
        unsafe {
            (*self.0.as_ptr()).attr &= !flags;
        }
    }

    #[inline]
    pub fn is_atom(&self) -> bool {
        let k = self.kind_raw();
        k < 0 && k > -90
    }

    #[inline]
    pub fn is_vec(&self) -> bool {
        let k = self.kind_raw();
        k > 0 && k < 90
    }

    /// Vector length, read from offset 8. For atoms this returns whatever
    /// bytes happen to live at that offset reinterpreted as `i64` — only
    /// meaningful on vector cells.
    #[inline]
    pub fn len(&self) -> i64 {
        unsafe {
            let p = self.0.as_ptr() as *const u8;
            ptr::read_unaligned(p.add(8) as *const i64)
        }
    }

    /// Whether a vector cell has zero elements. Always `false` for atoms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        if self.is_atom() {
            false
        } else {
            self.len() == 0
        }
    }

    /// Sole owner ⇒ payload may be mutated in place without violating sharing.
    /// External cells (memory we don't own) are never reported unique.
    ///
    /// Uses `Acquire` on the atomic load so that any writes performed by a
    /// previously-dropped sharer (whose `fetch_sub(AcqRel)` brought rc to 1)
    /// are visible before we proceed with in-place mutation.
    #[inline]
    pub fn is_unique(&self) -> bool {
        if self.meta() & meta_flags::IS_EXTERNAL != 0 {
            return false;
        }
        unsafe { (*self.0.as_ptr()).rc.load(Ordering::Acquire) == 1 }
    }

    // ---- typed accessors (unsafe — type contract is on the caller) -----

    /// Read the 8-byte atom payload reinterpreted as `T`.
    ///
    /// # Safety
    /// `T` must match the storage type of `self.kind()`. Reading the wrong
    /// type is logical garbage but not memory-unsafe; `as_slice` is the
    /// stricter case.
    #[inline]
    pub unsafe fn atom<T: Copy>(&self) -> T {
        let p = self.0.as_ptr() as *const u8;
        ptr::read_unaligned(p.add(8) as *const T)
    }

    /// Borrow the vector payload as `&[T]`.
    ///
    /// # Safety
    /// `T` must match the storage type of `self.kind()`. A mismatched `T`
    /// changes the slice's element count interpretation and may cause OOB
    /// reads on differently-sized types.
    #[inline]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        let p = self.0.as_ptr() as *const u8;
        slice::from_raw_parts(p.add(16) as *const T, self.len() as usize)
    }

    /// Borrow the vector payload as `&mut [T]`.
    ///
    /// # Safety
    /// As [`as_slice`], plus the caller must ensure exclusive access (i.e.
    /// `is_unique()` was true, or no other `RefObj` to the same cell exists
    /// for the duration of the borrow).
    #[inline]
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let p = self.0.as_ptr() as *mut u8;
        slice::from_raw_parts_mut(p.add(16) as *mut T, self.len() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_size_and_layout() {
        assert_eq!(size_of::<Obj>(), 8);
        assert_eq!(size_of::<RefObj>(), 8);
        // AtomicU32 is #[repr(transparent)] over u32 — header layout is
        // byte-identical to the pre-v2 form.
        assert_eq!(size_of::<AtomicU32>(), 4);
        assert_eq!(align_of::<AtomicU32>(), 4);
        // Verify offsets within the C-layout struct.
        let mut o = Obj {
            meta: 1,
            attr: 2,
            kind: 3,
            _resv: 4,
            rc: AtomicU32::new(5),
        };
        let base = &o as *const Obj as usize;
        let m = &o.meta as *const i8 as usize;
        let a = &o.attr as *const i8 as usize;
        let k = &o.kind as *const i8 as usize;
        let u = &o._resv as *const u8 as usize;
        let r = &o.rc as *const AtomicU32 as usize;
        assert_eq!(m - base, 0);
        assert_eq!(a - base, 1);
        assert_eq!(k - base, 2);
        assert_eq!(u - base, 3);
        assert_eq!(r - base, 4);
        assert_eq!(o.rc.load(Ordering::Relaxed), 5);
        o.meta = 0;
        let _ = o;
    }
}
