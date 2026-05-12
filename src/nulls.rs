//! Null sentinel constants per element type.
//!
//! Chosen for arithmetical convenience (integer-min, IEEE NaN, empty-packed
//! symbol). `HAS_NULLS` is a flag in `Obj.attr` that lets binary kernels skip
//! the null-preserving mask sweep on null-free inputs.

use crate::sym::Sym;

pub const NULL_I16: i16 = i16::MIN;
pub const NULL_I32: i32 = i32::MIN;
pub const NULL_I64: i64 = i64::MIN;

pub const INF_I64: i64 = i64::MAX;

pub const NULL_F32: f32 = f32::NAN;
pub const NULL_F64: f64 = f64::NAN;

pub const INF_F32: f32 = f32::INFINITY;
pub const INF_F64: f64 = f64::INFINITY;

pub const NULL_SYM: Sym = Sym(0);
