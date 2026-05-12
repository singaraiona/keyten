//! Keyten — an array language runtime.
//!
//! Stage 1: object layout, refcount discipline, streaming chunked kernels,
//! sync + async drivers, dispatch for `+ − × ÷` and `+/`.
//!
//! See the design document for invariants and rationale.

#![allow(clippy::missing_safety_doc)]

pub mod kind;
pub mod nulls;
pub mod sym;

pub mod obj;
pub mod runtime;
pub mod ctx;
pub mod render;
pub mod yield_now;
pub mod chunk;
pub mod exec;

pub mod madvise;
pub mod alloc;

pub mod simd;
pub mod bitvec;

pub mod kernels;
pub mod adverb;
pub mod op;

pub use ctx::{Ctx, KernelErr};
pub use exec::block_on;
pub use kind::Kind;
pub use obj::{Obj, RefObj, attr_flags, meta_flags};
pub use op::OpId;
pub use render::RenderSink;
pub use runtime::{Runtime, RUNTIME};
pub use sym::Sym;
