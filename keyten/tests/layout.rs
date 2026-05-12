//! Layout invariants: sizes, alignments, field offsets.

use std::mem::{align_of, size_of};

use keyten::obj::{Obj, RefObj};

#[test]
fn obj_size_is_8_bytes() {
    assert_eq!(size_of::<Obj>(), 8);
}

#[test]
fn refobj_size_is_pointer() {
    assert_eq!(size_of::<RefObj>(), size_of::<*mut Obj>());
    assert_eq!(size_of::<RefObj>(), 8);
}

#[test]
fn obj_alignment() {
    // Natural alignment is 4 (u32 refcount); we lay out atom/vec at offset 8
    // which is naturally u64-aligned because Layout::from_size_align(_, 8)
    // is used at allocation time.
    assert_eq!(align_of::<Obj>(), 4);
}

#[test]
fn obj_field_offsets() {
    let o = Obj { meta: 1, attr: 2, kind: 3, _resv: 4, rc: 5 };
    let base = &o as *const Obj as usize;
    assert_eq!(&o.meta as *const _ as usize - base, 0);
    assert_eq!(&o.attr as *const _ as usize - base, 1);
    assert_eq!(&o.kind as *const _ as usize - base, 2);
    assert_eq!(&o._resv as *const _ as usize - base, 3);
    assert_eq!(&o.rc as *const _ as usize - base, 4);
}
