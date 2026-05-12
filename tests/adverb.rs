//! `+/` over I64 / F64.

use keyten::adverb::{plus_over_f64, plus_over_i64};
use keyten::alloc::{alloc_vec_f64, alloc_vec_i64};
use keyten::ctx::Ctx;
use keyten::nulls::NULL_I64;
use keyten::obj::attr_flags;

#[test]
fn plus_over_i64_sum() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut v = alloc_vec_i64(&ctx, 5);
        v.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3, 4, 5]);
        let r = plus_over_i64(v, &ctx).unwrap();
        assert_eq!(r.atom::<i64>(), 15);
    }
}

#[test]
fn plus_over_f64_sum() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut v = alloc_vec_f64(&ctx, 4);
        v.as_mut_slice::<f64>().copy_from_slice(&[1.5, 2.5, 3.0, 0.5]);
        let r = plus_over_f64(v, &ctx).unwrap();
        assert!((r.atom::<f64>() - 7.5).abs() < 1e-12);
    }
}

#[test]
fn plus_over_i64_skips_nulls() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut v = alloc_vec_i64(&ctx, 5);
        v.as_mut_slice::<i64>().copy_from_slice(&[1, NULL_I64, 3, NULL_I64, 5]);
        v.set_attr(attr_flags::HAS_NULLS);
        let r = plus_over_i64(v, &ctx).unwrap();
        // 1 + 3 + 5 = 9, ignoring the two nulls.
        assert_eq!(r.atom::<i64>(), 9);
    }
}

#[test]
fn plus_over_spans_multiple_chunks() {
    // Force many chunks via a small chunk_elems override.
    let ctx = Ctx::quiet().with_chunk(32);
    unsafe {
        let n = 1000;
        let mut v = alloc_vec_i64(&ctx, n as i64);
        let s = v.as_mut_slice::<i64>();
        for i in 0..n {
            s[i] = i as i64;
        }
        let r = plus_over_i64(v, &ctx).unwrap();
        // sum 0..1000 = 999*1000/2 = 499_500
        assert_eq!(r.atom::<i64>(), 499_500);
    }
}
