//! Operator identifiers and per-op dispatch.
//!
//! Dispatch is the **safe API boundary**: it verifies operand kinds via the
//! match, then enters a small `unsafe` block to call the kernel-pair function
//! that assumes the kinds match. External callers see a safe surface.
//!
//! The match is keyed on `(|x.kind|, |y.kind|)`. LLVM lowers the dense u8-pair
//! match to a jump table.

use crate::alloc::{alloc_atom, alloc_vec_f64};
use crate::ctx::{Ctx, KernelErr};
use crate::exec::block_on;
use crate::kernels;
use crate::kind::Kind;
use crate::obj::{attr_flags, RefObj};

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum OpId {
    Plus = 0,
    Minus = 1,
    Times = 2,
    Div = 3,
    /// `!` — monadic: til (`!n` → `0 1 … n−1`); dyadic: mod (reserved).
    Bang = 4,
}

pub const OP_COUNT: usize = 5;

// Function-pointer table of sync dispatch entries, indexed by `OpId as usize`.
pub type DyadicFn = fn(RefObj, RefObj, &Ctx) -> Result<RefObj, KernelErr>;

pub static DYADIC: [DyadicFn; OP_COUNT] = [
    dispatch_plus,
    dispatch_minus,
    dispatch_times,
    dispatch_div,
    dispatch_bang,
];

#[inline]
fn pair(x_kind: i8, y_kind: i8) -> (u8, u8) {
    (x_kind.unsigned_abs(), y_kind.unsigned_abs())
}

// ---- async dispatch (canonical implementations) ------------------------

pub async fn dispatch_plus_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let (xk, yk) = pair(x.kind_raw(), y.kind_raw());
    match (xk, yk) {
        (k, _) | (_, k) if k == Kind::F64 as u8 => {
            let xf = promote_to_f64_async(x, ctx).await?;
            let yf = promote_to_f64_async(y, ctx).await?;
            // SAFETY: both operands are F64 after promotion.
            unsafe { kernels::plus::plus_f64_f64_async(xf, yf, ctx) }.await
        }
        (k1, k2) if k1 == Kind::I64 as u8 && k2 == Kind::I64 as u8 => {
            // SAFETY: both operands are I64.
            unsafe { kernels::plus::plus_i64_i64_async(x, y, ctx) }.await
        }
        _ => Err(KernelErr::Type),
    }
}

pub async fn dispatch_minus_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let (xk, yk) = pair(x.kind_raw(), y.kind_raw());
    match (xk, yk) {
        (k, _) | (_, k) if k == Kind::F64 as u8 => {
            let xf = promote_to_f64_async(x, ctx).await?;
            let yf = promote_to_f64_async(y, ctx).await?;
            unsafe { kernels::minus::minus_f64_f64_async(xf, yf, ctx) }.await
        }
        (k1, k2) if k1 == Kind::I64 as u8 && k2 == Kind::I64 as u8 => {
            unsafe { kernels::minus::minus_i64_i64_async(x, y, ctx) }.await
        }
        _ => Err(KernelErr::Type),
    }
}

pub async fn dispatch_times_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let (xk, yk) = pair(x.kind_raw(), y.kind_raw());
    match (xk, yk) {
        (k, _) | (_, k) if k == Kind::F64 as u8 => {
            let xf = promote_to_f64_async(x, ctx).await?;
            let yf = promote_to_f64_async(y, ctx).await?;
            unsafe { kernels::times::times_f64_f64_async(xf, yf, ctx) }.await
        }
        (k1, k2) if k1 == Kind::I64 as u8 && k2 == Kind::I64 as u8 => {
            unsafe { kernels::times::times_i64_i64_async(x, y, ctx) }.await
        }
        _ => Err(KernelErr::Type),
    }
}

/// Division always promotes to F64.
pub async fn dispatch_div_async(x: RefObj, y: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xf = promote_to_f64_async(x, ctx).await?;
    let yf = promote_to_f64_async(y, ctx).await?;
    unsafe { kernels::div::div_f64_f64_async(xf, yf, ctx) }.await
}

// ---- sync shims (block_on the async versions) --------------------------

pub fn dispatch_plus(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(dispatch_plus_async(x, y, ctx))
}

pub fn dispatch_minus(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(dispatch_minus_async(x, y, ctx))
}

pub fn dispatch_times(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(dispatch_times_async(x, y, ctx))
}

pub fn dispatch_div(x: RefObj, y: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    block_on(dispatch_div_async(x, y, ctx))
}

/// `x ! y` — modulo. Not implemented in v1; returns a type error.
pub fn dispatch_bang(_x: RefObj, _y: RefObj, _ctx: &Ctx) -> Result<RefObj, KernelErr> {
    Err(KernelErr::Type)
}

pub async fn dispatch_bang_async(
    _x: RefObj,
    _y: RefObj,
    _ctx: &Ctx<'_>,
) -> Result<RefObj, KernelErr> {
    Err(KernelErr::Type)
}

// ---- numeric promotion ------------------------------------------------

/// Widen an I64 operand to F64 for mixed-type arithmetic. F64 operands pass
/// through unchanged. Other kinds return `KernelErr::Type`.
async fn promote_to_f64_async(x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let xk = x.kind_raw().unsigned_abs();
    if xk == Kind::F64 as u8 {
        return Ok(x);
    }
    if xk != Kind::I64 as u8 {
        return Err(KernelErr::Type);
    }
    if x.is_atom() {
        // SAFETY: kind is I64-atom (just verified).
        let v = unsafe { x.atom::<i64>() } as f64;
        return Ok(unsafe { alloc_atom(Kind::F64, v) });
    }
    // SAFETY: kind is I64-vector (just verified).
    let xs = unsafe { x.as_slice::<i64>() };
    let mut out = unsafe { alloc_vec_f64(ctx, xs.len() as i64) };
    {
        let os = unsafe { out.as_mut_slice::<f64>() };
        for i in 0..xs.len() {
            os[i] = if xs[i] == crate::nulls::NULL_I64 {
                f64::NAN
            } else {
                xs[i] as f64
            };
        }
    }
    if (x.attr() & attr_flags::HAS_NULLS) != 0 {
        out.set_attr(attr_flags::HAS_NULLS);
    }
    Ok(out)
}

// ---- adverb entry ------------------------------------------------------

pub fn dispatch_over(op: OpId, x: RefObj, ctx: &Ctx) -> Result<RefObj, KernelErr> {
    crate::adverb::over(op, x, ctx)
}

pub async fn dispatch_over_async(op: OpId, x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    crate::adverb::over_async(op, x, ctx).await
}
