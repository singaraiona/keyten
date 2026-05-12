//! Arithmetic kernels: `+ − × ÷` across atom/vec shapes for I64 and F64.

use keyten::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use keyten::ctx::Ctx;
use keyten::kind::Kind;
use keyten::op::{dispatch_div, dispatch_minus, dispatch_plus, dispatch_times};

fn ctx() -> Ctx<'static> {
    Ctx::quiet()
}

#[track_caller]
unsafe fn vec_i64(xs: &[i64]) -> keyten::RefObj {
    let c = ctx();
    let mut v = alloc_vec_i64(&c, xs.len() as i64);
    v.as_mut_slice::<i64>().copy_from_slice(xs);
    v
}

#[track_caller]
unsafe fn vec_f64(xs: &[f64]) -> keyten::RefObj {
    let c = ctx();
    let mut v = alloc_vec_f64(&c, xs.len() as i64);
    v.as_mut_slice::<f64>().copy_from_slice(xs);
    v
}

#[track_caller]
unsafe fn read_i64(r: &keyten::RefObj) -> Vec<i64> {
    if r.is_atom() {
        vec![r.atom::<i64>()]
    } else {
        r.as_slice::<i64>().to_vec()
    }
}

#[track_caller]
unsafe fn read_f64(r: &keyten::RefObj) -> Vec<f64> {
    if r.is_atom() {
        vec![r.atom::<f64>()]
    } else {
        r.as_slice::<f64>().to_vec()
    }
}

// =================== I64 + ===================

#[test]
fn plus_i64_atom_atom() {
    unsafe {
        let x = alloc_atom(Kind::I64, 7i64);
        let y = alloc_atom(Kind::I64, 3i64);
        let r = dispatch_plus(x, y, &ctx()).unwrap();
        assert_eq!(r.atom::<i64>(), 10);
    }
}

#[test]
fn plus_i64_atom_vec() {
    unsafe {
        let x = alloc_atom(Kind::I64, 10i64);
        let y = vec_i64(&[1, 2, 3]);
        let r = dispatch_plus(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![11, 12, 13]);
    }
}

#[test]
fn plus_i64_vec_atom() {
    unsafe {
        let x = vec_i64(&[1, 2, 3]);
        let y = alloc_atom(Kind::I64, 10i64);
        let r = dispatch_plus(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![11, 12, 13]);
    }
}

#[test]
fn plus_i64_vec_vec() {
    unsafe {
        let x = vec_i64(&[1, 2, 3, 4]);
        let y = vec_i64(&[10, 20, 30, 40]);
        let r = dispatch_plus(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![11, 22, 33, 44]);
    }
}

// =================== I64 - ===================

#[test]
fn minus_i64_atom_atom() {
    unsafe {
        let x = alloc_atom(Kind::I64, 7i64);
        let y = alloc_atom(Kind::I64, 3i64);
        let r = dispatch_minus(x, y, &ctx()).unwrap();
        assert_eq!(r.atom::<i64>(), 4);
    }
}

#[test]
fn minus_i64_vec_atom() {
    unsafe {
        let x = vec_i64(&[10, 20, 30]);
        let y = alloc_atom(Kind::I64, 5i64);
        let r = dispatch_minus(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![5, 15, 25]);
    }
}

#[test]
fn minus_i64_atom_vec() {
    unsafe {
        let x = alloc_atom(Kind::I64, 100i64);
        let y = vec_i64(&[1, 2, 3]);
        let r = dispatch_minus(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![99, 98, 97]);
    }
}

#[test]
fn minus_i64_vec_vec() {
    unsafe {
        let x = vec_i64(&[10, 20, 30]);
        let y = vec_i64(&[1, 2, 3]);
        let r = dispatch_minus(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![9, 18, 27]);
    }
}

// =================== I64 × ===================

#[test]
fn times_i64_vec_vec() {
    unsafe {
        let x = vec_i64(&[1, 2, 3, 4]);
        let y = vec_i64(&[10, 10, 10, 10]);
        let r = dispatch_times(x, y, &ctx()).unwrap();
        assert_eq!(read_i64(&r), vec![10, 20, 30, 40]);
    }
}

// =================== F64 + ===================

#[test]
fn plus_f64_vec_vec() {
    unsafe {
        let x = vec_f64(&[1.5, 2.5, 3.5]);
        let y = vec_f64(&[0.5, 0.5, 0.5]);
        let r = dispatch_plus(x, y, &ctx()).unwrap();
        assert_eq!(read_f64(&r), vec![2.0, 3.0, 4.0]);
    }
}

// =================== mixed (i64 + f64 → f64) ===================

#[test]
fn plus_i64_f64_promotes() {
    unsafe {
        let x = vec_i64(&[1, 2, 3]);
        let y = vec_f64(&[0.5, 0.5, 0.5]);
        let r = dispatch_plus(x, y, &ctx()).unwrap();
        assert_eq!(r.kind(), Kind::F64);
        assert_eq!(read_f64(&r), vec![1.5, 2.5, 3.5]);
    }
}

// =================== div always F64 ===================

#[test]
fn div_i64_i64_yields_f64() {
    unsafe {
        let x = alloc_atom(Kind::I64, 7i64);
        let y = alloc_atom(Kind::I64, 2i64);
        let r = dispatch_div(x, y, &ctx()).unwrap();
        assert_eq!(r.kind(), Kind::F64);
        assert_eq!(r.atom::<f64>(), 3.5);
    }
}

#[test]
fn div_vec_vec_f64() {
    unsafe {
        let x = vec_f64(&[10.0, 20.0, 30.0]);
        let y = vec_f64(&[2.0, 4.0, 5.0]);
        let r = dispatch_div(x, y, &ctx()).unwrap();
        assert_eq!(read_f64(&r), vec![5.0, 5.0, 6.0]);
    }
}
