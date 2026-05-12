//! Arithmetic kernels over slices.
//!
//! These are written as straight scalar loops; the modern LLVM autovectorizer
//! turns each into the equivalent of an explicit `Simd<T, N>` reduction on
//! every supported target. Explicit SIMD via `std::arch` (or `std::simd`) can
//! be added behind a `target_feature` gate if benchmarks show autovec misses
//! the target, but on tight `+= * - /` loops over aligned f64/i64 slices it
//! consistently hits the upper bound on x86-64 and aarch64.

#![allow(clippy::missing_safety_doc)]

// ---- vector-vector arithmetic -----------------------------------------

macro_rules! impl_binop {
    ($name:ident, $T:ty, $op:tt) => {
        #[inline]
        pub fn $name(out: &mut [$T], x: &[$T], y: &[$T]) {
            debug_assert_eq!(out.len(), x.len());
            debug_assert_eq!(out.len(), y.len());
            // Iterate by index — LLVM autovectorizes this pattern reliably.
            let n = out.len();
            let mut i = 0;
            while i < n {
                out[i] = x[i] $op y[i];
                i += 1;
            }
        }
    };
}

// Integer ops use wrapping arithmetic to match K-family overflow behaviour.
#[inline]
pub fn add_i64(out: &mut [i64], x: &[i64], y: &[i64]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i].wrapping_add(y[i]);
        i += 1;
    }
}

#[inline]
pub fn sub_i64(out: &mut [i64], x: &[i64], y: &[i64]) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i].wrapping_sub(y[i]);
        i += 1;
    }
}

#[inline]
pub fn mul_i64(out: &mut [i64], x: &[i64], y: &[i64]) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i].wrapping_mul(y[i]);
        i += 1;
    }
}

impl_binop!(add_f64, f64, +);
impl_binop!(sub_f64, f64, -);
impl_binop!(mul_f64, f64, *);
impl_binop!(div_f64, f64, /);

// ---- vector-scalar (broadcast) ----------------------------------------

#[inline]
pub fn add_scalar_i64(out: &mut [i64], x: &[i64], s: i64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i].wrapping_add(s);
        i += 1;
    }
}

#[inline]
pub fn sub_scalar_i64(out: &mut [i64], x: &[i64], s: i64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i].wrapping_sub(s);
        i += 1;
    }
}

#[inline]
pub fn scalar_sub_vec_i64(out: &mut [i64], s: i64, x: &[i64]) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = s.wrapping_sub(x[i]);
        i += 1;
    }
}

#[inline]
pub fn mul_scalar_i64(out: &mut [i64], x: &[i64], s: i64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i].wrapping_mul(s);
        i += 1;
    }
}

#[inline]
pub fn add_scalar_f64(out: &mut [f64], x: &[f64], s: f64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i] + s;
        i += 1;
    }
}

#[inline]
pub fn sub_scalar_f64(out: &mut [f64], x: &[f64], s: f64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i] - s;
        i += 1;
    }
}

#[inline]
pub fn scalar_sub_vec_f64(out: &mut [f64], s: f64, x: &[f64]) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = s - x[i];
        i += 1;
    }
}

#[inline]
pub fn mul_scalar_f64(out: &mut [f64], x: &[f64], s: f64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i] * s;
        i += 1;
    }
}

#[inline]
pub fn div_scalar_f64(out: &mut [f64], x: &[f64], s: f64) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = x[i] / s;
        i += 1;
    }
}

#[inline]
pub fn scalar_div_vec_f64(out: &mut [f64], s: f64, x: &[f64]) {
    let n = out.len();
    let mut i = 0;
    while i < n {
        out[i] = s / x[i];
        i += 1;
    }
}

// ---- reductions -------------------------------------------------------

#[inline]
pub fn sum_i64(xs: &[i64]) -> i64 {
    let mut acc: i64 = 0;
    for &v in xs {
        acc = acc.wrapping_add(v);
    }
    acc
}

#[inline]
pub fn sum_f64(xs: &[f64]) -> f64 {
    let mut acc: f64 = 0.0;
    for &v in xs {
        acc += v;
    }
    acc
}

#[inline]
pub fn sum_i64_skipping(xs: &[i64], null: i64) -> i64 {
    let mut acc: i64 = 0;
    for &v in xs {
        if v != null {
            acc = acc.wrapping_add(v);
        }
    }
    acc
}
