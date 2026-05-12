//! Refcount discipline: Clone bumps rc, Drop dec/free, rc==1 mutate path.

use keyten::alloc::{alloc_atom, alloc_vec_i64};
use keyten::ctx::Ctx;
use keyten::kind::Kind;

#[test]
fn atom_rc_starts_at_1() {
    unsafe {
        let a = alloc_atom(Kind::I64, 42i64);
        assert_eq!(a.rc(), 1);
        assert!(a.is_unique());
    }
}

#[test]
fn clone_increments_drop_decrements() {
    unsafe {
        let a = alloc_atom(Kind::I64, 7i64);
        let b = a.clone();
        assert_eq!(a.rc(), 2);
        assert_eq!(b.rc(), 2);
        drop(b);
        assert_eq!(a.rc(), 1);
        assert!(a.is_unique());
    }
}

#[test]
fn vec_payload_round_trip() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut v = alloc_vec_i64(&ctx, 5);
        let s = v.as_mut_slice::<i64>();
        s.copy_from_slice(&[10, 20, 30, 40, 50]);
        let r = v.as_slice::<i64>();
        assert_eq!(r, &[10, 20, 30, 40, 50]);
    }
}

#[test]
fn no_leak_via_drop() {
    // Allocate and drop many atoms — freelist absorbs and reuses cells. No
    // explicit assertion; failures would show under miri.
    unsafe {
        for i in 0..1000 {
            let _ = alloc_atom(Kind::I64, i as i64);
        }
    }
}
