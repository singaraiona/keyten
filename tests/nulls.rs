//! HAS_NULLS attr bit: null-free hot path stays clean; null-bearing inputs
//! preserve nulls via the cold path, and the output cell carries the bit.

use keyten::alloc::{alloc_vec_f64, alloc_vec_i64};
use keyten::ctx::Ctx;
use keyten::nulls::{NULL_F64, NULL_I64};
use keyten::obj::attr_flags;
use keyten::op::dispatch_plus;

#[test]
fn null_free_vec_op_stays_null_free() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut x = alloc_vec_i64(&ctx, 3);
        x.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]);
        let mut y = alloc_vec_i64(&ctx, 3);
        y.as_mut_slice::<i64>().copy_from_slice(&[10, 20, 30]);
        let r = dispatch_plus(x, y, &ctx).unwrap();
        assert_eq!(r.as_slice::<i64>(), &[11, 22, 33]);
        assert_eq!(r.attr() & attr_flags::HAS_NULLS, 0);
    }
}

#[test]
fn i64_null_preserved_through_plus() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut x = alloc_vec_i64(&ctx, 3);
        x.as_mut_slice::<i64>().copy_from_slice(&[1, NULL_I64, 3]);
        x.set_attr(attr_flags::HAS_NULLS);

        let mut y = alloc_vec_i64(&ctx, 3);
        y.as_mut_slice::<i64>().copy_from_slice(&[10, 20, 30]);

        let r = dispatch_plus(x, y, &ctx).unwrap();
        let rs = r.as_slice::<i64>();
        assert_eq!(rs[0], 11);
        assert_eq!(rs[1], NULL_I64);
        assert_eq!(rs[2], 33);
        assert_ne!(r.attr() & attr_flags::HAS_NULLS, 0);
    }
}

#[test]
fn f64_nan_propagates_naturally() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut x = alloc_vec_f64(&ctx, 3);
        x.as_mut_slice::<f64>().copy_from_slice(&[1.0, NULL_F64, 3.0]);
        x.set_attr(attr_flags::HAS_NULLS);

        let mut y = alloc_vec_f64(&ctx, 3);
        y.as_mut_slice::<f64>().copy_from_slice(&[10.0, 20.0, 30.0]);

        let r = dispatch_plus(x, y, &ctx).unwrap();
        let rs = r.as_slice::<f64>();
        assert_eq!(rs[0], 11.0);
        assert!(rs[1].is_nan());
        assert_eq!(rs[2], 33.0);
        assert_ne!(r.attr() & attr_flags::HAS_NULLS, 0);
    }
}
