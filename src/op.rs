//! Operator identifiers and per-op dispatch entry points.
//!
//! Dispatch is a `match` keyed on `(|x.kind|, |y.kind|)`. LLVM lowers the
//! dense u8-pair match to a jump table.

use crate::ctx::{Ctx, KernelErr};
use crate::kernels;
use crate::kind::Kind;
use crate::obj::RefObj;

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum OpId {
    Plus = 0,
    Minus = 1,
    Times = 2,
    Div = 3,
}

pub const OP_COUNT: usize = 4;

pub type DyadicFn = unsafe fn(RefObj, RefObj, &Ctx) -> Result<RefObj, KernelErr>;

pub static DYADIC: [DyadicFn; OP_COUNT] = [
    dispatch_plus,
    dispatch_minus,
    dispatch_times,
    dispatch_div,
];

#[inline]
fn key(x_kind: i8, y_kind: i8) -> (u8, u8) {
    (x_kind.unsigned_abs(), y_kind.unsigned_abs())
}

// ---- dispatch_plus -----------------------------------------------------

pub unsafe fn dispatch_plus(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let (xk, yk) = key(x.kind_raw(), y.kind_raw());
    match (xk, yk) {
        (k, _) | (_, k) if k == Kind::F64 as u8 => kernels::plus::plus_f64_f64(
            promote_to_f64(x, ctx)?,
            promote_to_f64(y, ctx)?,
            ctx,
        ),
        (k1, k2) if k1 == Kind::I64 as u8 && k2 == Kind::I64 as u8 => {
            kernels::plus::plus_i64_i64(x, y, ctx)
        }
        _ => Err(KernelErr::Type),
    }
}

pub unsafe fn dispatch_minus(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let (xk, yk) = key(x.kind_raw(), y.kind_raw());
    match (xk, yk) {
        (k, _) | (_, k) if k == Kind::F64 as u8 => kernels::minus::minus_f64_f64(
            promote_to_f64(x, ctx)?,
            promote_to_f64(y, ctx)?,
            ctx,
        ),
        (k1, k2) if k1 == Kind::I64 as u8 && k2 == Kind::I64 as u8 => {
            kernels::minus::minus_i64_i64(x, y, ctx)
        }
        _ => Err(KernelErr::Type),
    }
}

pub unsafe fn dispatch_times(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let (xk, yk) = key(x.kind_raw(), y.kind_raw());
    match (xk, yk) {
        (k, _) | (_, k) if k == Kind::F64 as u8 => kernels::times::times_f64_f64(
            promote_to_f64(x, ctx)?,
            promote_to_f64(y, ctx)?,
            ctx,
        ),
        (k1, k2) if k1 == Kind::I64 as u8 && k2 == Kind::I64 as u8 => {
            kernels::times::times_i64_i64(x, y, ctx)
        }
        _ => Err(KernelErr::Type),
    }
}

/// Division always promotes to F64 per K convention.
pub unsafe fn dispatch_div(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xf = promote_to_f64(x, ctx)?;
    let yf = promote_to_f64(y, ctx)?;
    kernels::div::div_f64_f64(xf, yf, ctx)
}

// ---- helpers -----------------------------------------------------------

/// Promote any numeric operand to F64. Used by div, and by mixed-type arith
/// pairs (any F64 operand widens the other to F64).
unsafe fn promote_to_f64(x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    let xk = x.kind_raw().unsigned_abs();
    if xk == Kind::F64 as u8 {
        return Ok(x);
    }
    if xk == Kind::I64 as u8 {
        if x.is_atom() {
            return Ok(crate::alloc::alloc_atom(Kind::F64, x.atom::<i64>() as f64));
        }
        let xs = x.as_slice::<i64>();
        let mut out = crate::alloc::alloc_vec_f64(ctx, xs.len() as i64);
        let os = out.as_mut_slice::<f64>();
        for i in 0..xs.len() {
            // Sentinel I64 nulls become NaN F64 nulls.
            os[i] = if xs[i] == crate::nulls::NULL_I64 {
                f64::NAN
            } else {
                xs[i] as f64
            };
        }
        if (x.attr() & crate::obj::attr_flags::HAS_NULLS) != 0 {
            out.set_attr(crate::obj::attr_flags::HAS_NULLS);
        }
        return Ok(out);
    }
    Err(KernelErr::Type)
}

// ---- adverb entry ------------------------------------------------------

pub unsafe fn dispatch_over(op: OpId, x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    crate::adverb::over(op, x, ctx)
}
