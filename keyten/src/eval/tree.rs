//! Tree-walking evaluator.
//!
//! `eval_async` is the canonical implementation; it `.await`s on dispatch
//! calls so the kernel's chunk yields propagate to the REPL/UI loop. The sync
//! `eval` is a `block_on` shim.

use core::future::Future;
use core::pin::Pin;

use crate::adverb;
use crate::alloc::{alloc_atom, alloc_vec_f64, alloc_vec_i64};
use crate::ast::{AdvId, AtomLit, Expr, Span};
use crate::ctx::{Ctx, KernelErr};
use crate::eval::env::Env;
use crate::exec::block_on;
use crate::kind::Kind;
use crate::nulls::{INF_F64, INF_I64, NULL_F64, NULL_I64};
use crate::obj::{attr_flags, RefObj};
use crate::op;
use crate::sym::Sym;

#[derive(Debug)]
pub enum EvalErr {
    UndefinedName { name: Sym, span: Span },
    Kernel { err: KernelErr, span: Span },
    Type { msg: String, span: Span },
    Empty,
}

/// Synchronous tree-walking evaluator. Block_on's `eval_async`.
pub fn eval(expr: &Expr, env: &mut Env, ctx: &Ctx) -> Result<RefObj, EvalErr> {
    block_on(eval_async(expr, env, ctx))
}

/// Asynchronous tree-walking evaluator. Awaits dispatch entries so kernel
/// chunk yields propagate up to the calling executor.
pub async fn eval_async<'a, 'r>(
    expr: &'a Expr,
    env: &'a mut Env,
    ctx: &'a Ctx<'r>,
) -> Result<RefObj, EvalErr> {
    eval_boxed(expr, env, ctx).await
}

// Boxed-future recursive evaluator. async fn recursion in stable Rust requires
// either explicit Box::pin or async_recursion; we use Box::pin directly.
fn eval_boxed<'a, 'r>(
    expr: &'a Expr,
    env: &'a mut Env,
    ctx: &'a Ctx<'r>,
) -> Pin<Box<dyn Future<Output = Result<RefObj, EvalErr>> + 'a>> {
    Box::pin(async move {
        match expr {
            Expr::AtomLit { lit, .. } => Ok(make_atom(*lit)),
            Expr::VecLit { kind, items, .. } => Ok(make_vec(*kind, items, ctx)),
            Expr::ListLit { items, span } => {
                // v1 has no generic-list semantics in kernels; we materialise a
                // single-statement list as Seq-equivalent (evaluate each, return
                // last). Composite list values arrive in v2.
                let _ = span;
                if items.is_empty() {
                    return Err(EvalErr::Empty);
                }
                let mut last: Option<RefObj> = None;
                for it in items {
                    last = Some(eval_boxed(it, env, ctx).await?);
                }
                Ok(last.unwrap())
            }
            Expr::Name { sym, span } => env.lookup(*sym).ok_or(EvalErr::UndefinedName {
                name: *sym,
                span: *span,
            }),
            Expr::Assign { name, value, .. } => {
                let v = eval_boxed(value, env, ctx).await?;
                env.bind(*name, v.clone());
                Ok(v)
            }
            Expr::Dyad { verb, lhs, rhs, span } => {
                let x = eval_boxed(lhs, env, ctx).await?;
                let y = eval_boxed(rhs, env, ctx).await?;
                let r = match *verb {
                    op::OpId::Plus => op::dispatch_plus_async(x, y, ctx).await,
                    op::OpId::Minus => op::dispatch_minus_async(x, y, ctx).await,
                    op::OpId::Times => op::dispatch_times_async(x, y, ctx).await,
                    op::OpId::Div => op::dispatch_div_async(x, y, ctx).await,
                    op::OpId::Bang => op::dispatch_bang_async(x, y, ctx).await,
                    op::OpId::At => op::dispatch_at_async(x, y, ctx).await,
                    op::OpId::Hash => op::dispatch_hash_async(x, y, ctx).await,
                    op::OpId::Comma => op::dispatch_comma_async(x, y, ctx).await,
                    op::OpId::Eq => op::dispatch_eq_async(x, y, ctx).await,
                    op::OpId::Lt => op::dispatch_lt_async(x, y, ctx).await,
                    op::OpId::Gt => op::dispatch_gt_async(x, y, ctx).await,
                    op::OpId::Tilde => op::dispatch_tilde_async(x, y, ctx).await,
                    op::OpId::Amp => op::dispatch_amp_async(x, y, ctx).await,
                    op::OpId::Pipe => op::dispatch_pipe_async(x, y, ctx).await,
                    op::OpId::Underscore => op::dispatch_underscore_async(x, y, ctx).await,
                    op::OpId::Dollar => op::dispatch_dollar_async(x, y, ctx).await,
                    op::OpId::Caret => op::dispatch_caret_async(x, y, ctx).await,
                    op::OpId::Question => op::dispatch_question_async(x, y, ctx).await,
                    op::OpId::Dot => op::dispatch_dot_async(x, y, ctx).await,
                };
                r.map_err(|e| EvalErr::Kernel { err: e, span: *span })
            }
            Expr::Monad { verb, arg, span } => {
                let x = eval_boxed(arg, env, ctx).await?;
                match *verb {
                    op::OpId::Plus => {
                        // K9: `+dict` flips dict → table. Other monadic `+`
                        // forms (transpose of nested list) need mixed-list
                        // support — defer.
                        if x.kind() == Kind::Dict {
                            crate::kernels::dict::flip_dict_to_table(x, ctx)
                                .map_err(|e| EvalErr::Kernel { err: e, span: *span })
                        } else {
                            Ok(x)
                        }
                    }
                    op::OpId::Minus => negate_async(x, ctx)
                        .await
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Bang => {
                        // K9: `!dict` returns keys; `!N` (i64 atom) is til.
                        if x.kind() == Kind::Dict {
                            crate::kernels::dict::dict_keys(&x)
                                .map_err(|e| EvalErr::Kernel { err: e, span: *span })
                        } else {
                            crate::kernels::til::til_async(x, ctx)
                                .await
                                .map_err(|e| EvalErr::Kernel { err: e, span: *span })
                        }
                    }
                    op::OpId::At => crate::kernels::monad::type_of(x)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Hash => crate::kernels::monad::count(x)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Comma => crate::kernels::monad::enlist(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Tilde => crate::kernels::monad::not(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Underscore => unsafe { crate::kernels::underscore::floor_async(x, ctx) }
                        .await
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Dollar => crate::kernels::monad::string_of(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Caret => crate::kernels::setops::sort_asc(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Question => crate::kernels::setops::unique(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Div => crate::kernels::setops::sqrt(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Pipe => crate::kernels::setops::reverse(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Amp => crate::kernels::setops::where_indices(x, ctx)
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    op::OpId::Dot => {
                        // K9: `.dict` returns the values vector. Other
                        // monadic-dot forms (value of code, etc.) need
                        // Phase 1 lambdas and an eval-from-string path.
                        if x.kind() == Kind::Dict {
                            crate::kernels::dict::dict_values(&x)
                                .map_err(|e| EvalErr::Kernel { err: e, span: *span })
                        } else {
                            Err(EvalErr::Type {
                                msg: "monadic `.` on non-dict input not implemented in v1".into(),
                                span: *span,
                            })
                        }
                    }
                    _ => Err(EvalErr::Type {
                        msg: "monadic form not supported for this verb in v1".into(),
                        span: *span,
                    }),
                }
            }
            Expr::Adverb {
                adv, verb, arg, span,
            } => {
                let x = eval_boxed(arg, env, ctx).await?;
                match adv {
                    AdvId::Over => adverb::over_async(*verb, x, ctx)
                        .await
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    AdvId::Scan => adverb::scan_async(*verb, x, ctx)
                        .await
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    AdvId::Each => adverb::each_async(*verb, x, ctx)
                        .await
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                    AdvId::EachPrior => adverb::eachprior_async(*verb, x, ctx)
                        .await
                        .map_err(|e| EvalErr::Kernel { err: e, span: *span }),
                }
            }
            Expr::Seq { items, .. } => {
                if items.is_empty() {
                    return Err(EvalErr::Empty);
                }
                let mut last: Option<RefObj> = None;
                for it in items {
                    last = Some(eval_boxed(it, env, ctx).await?);
                }
                Ok(last.unwrap())
            }
            Expr::Cond {
                cond,
                then_branch,
                else_branch,
                span: _,
            } => {
                let c = eval_boxed(cond, env, ctx).await?;
                let truthy = cond_is_truthy(&c);
                if truthy {
                    eval_boxed(then_branch, env, ctx).await
                } else {
                    eval_boxed(else_branch, env, ctx).await
                }
            }
        }
    })
}

/// K convention for the test in `$[c;t;e]`: non-zero / non-empty is true.
/// For atoms: numeric non-zero, bool 1b, non-null sym → true.
/// For vectors: K9 typically takes the first element; we follow that.
fn cond_is_truthy(c: &RefObj) -> bool {
    let k = c.kind();
    if c.is_atom() {
        match k {
            Kind::Bool | Kind::U8 | Kind::Char => (unsafe { c.atom::<u8>() }) != 0,
            Kind::I64 => (unsafe { c.atom::<i64>() }) != 0,
            Kind::F64 => (unsafe { c.atom::<f64>() }) != 0.0,
            Kind::Sym => (unsafe { c.atom::<crate::sym::Sym>() }).0 != 0,
            _ => true,
        }
    } else {
        if c.len() == 0 {
            return false;
        }
        unsafe {
            match k {
                Kind::Bool | Kind::U8 | Kind::Char => c.as_slice::<u8>()[0] != 0,
                Kind::I64 => c.as_slice::<i64>()[0] != 0,
                Kind::F64 => c.as_slice::<f64>()[0] != 0.0,
                _ => true,
            }
        }
    }
}

fn make_atom(lit: AtomLit) -> RefObj {
    unsafe {
        match lit {
            AtomLit::Bool(b) => alloc_atom(Kind::Bool, if b { 1u8 } else { 0u8 }),
            AtomLit::I64(v) => alloc_atom(Kind::I64, v),
            AtomLit::F64(v) => alloc_atom(Kind::F64, v),
            AtomLit::Char(c) => alloc_atom(Kind::Char, c),
            AtomLit::Sym(s) => alloc_atom(Kind::Sym, s.0),
            AtomLit::NullI64 => {
                let mut a = alloc_atom(Kind::I64, NULL_I64);
                a.set_attr(attr_flags::HAS_NULLS);
                a
            }
            AtomLit::NullF64 => {
                let mut a = alloc_atom(Kind::F64, NULL_F64);
                a.set_attr(attr_flags::HAS_NULLS);
                a
            }
            AtomLit::InfI64 => alloc_atom(Kind::I64, INF_I64),
            AtomLit::InfF64 => alloc_atom(Kind::F64, INF_F64),
        }
    }
}

fn make_vec(kind: Kind, items: &[AtomLit], ctx: &Ctx) -> RefObj {
    match kind {
        Kind::I64 => {
            let n = items.len() as i64;
            let mut v = unsafe { alloc_vec_i64(ctx, n) };
            let mut has_nulls = false;
            {
                let s = unsafe { v.as_mut_slice::<i64>() };
                for (i, lit) in items.iter().enumerate() {
                    s[i] = match lit {
                        AtomLit::I64(x) => *x,
                        AtomLit::NullI64 => {
                            has_nulls = true;
                            NULL_I64
                        }
                        AtomLit::InfI64 => INF_I64,
                        _ => 0, // parser guarantees homogeneous kind, but be defensive
                    };
                }
            }
            if has_nulls {
                v.set_attr(attr_flags::HAS_NULLS);
            }
            v
        }
        Kind::F64 => {
            let n = items.len() as i64;
            let mut v = unsafe { alloc_vec_f64(ctx, n) };
            let mut has_nulls = false;
            {
                let s = unsafe { v.as_mut_slice::<f64>() };
                for (i, lit) in items.iter().enumerate() {
                    s[i] = match lit {
                        AtomLit::F64(x) => *x,
                        AtomLit::I64(x) => *x as f64, // promotion: mixed-numeric → F64
                        AtomLit::NullF64 => {
                            has_nulls = true;
                            NULL_F64
                        }
                        AtomLit::NullI64 => {
                            has_nulls = true;
                            NULL_F64
                        }
                        AtomLit::InfF64 | AtomLit::InfI64 => INF_F64,
                        _ => 0.0,
                    };
                }
            }
            if has_nulls {
                v.set_attr(attr_flags::HAS_NULLS);
            }
            v
        }
        Kind::Char => {
            // Char vectors come from string literals; payload bytes are u8.
            let n = items.len() as i64;
            let mut v = unsafe {
                crate::alloc::alloc_vec(ctx, Kind::Char.vec(), n, Kind::Char.elem_size())
            };
            {
                let s = unsafe { v.as_mut_slice::<u8>() };
                for (i, lit) in items.iter().enumerate() {
                    s[i] = match lit {
                        AtomLit::Char(c) => *c,
                        _ => 0,
                    };
                }
            }
            v
        }
        Kind::Sym => {
            let n = items.len() as i64;
            let mut v = unsafe {
                crate::alloc::alloc_vec(ctx, Kind::Sym.vec(), n, Kind::Sym.elem_size())
            };
            {
                let s = unsafe { v.as_mut_slice::<i64>() };
                for (i, lit) in items.iter().enumerate() {
                    s[i] = match lit {
                        AtomLit::Sym(sm) => sm.0,
                        _ => 0,
                    };
                }
            }
            v
        }
        _ => {
            // Fallback: treat as I64 (parser shouldn't produce other vec kinds in v1).
            make_vec(Kind::I64, items, ctx)
        }
    }
}

/// Monadic `-x`: compute 0 - x via the dispatcher to share the chunked path.
pub async fn negate_async(x: RefObj, ctx: &Ctx<'_>) -> Result<RefObj, KernelErr> {
    let zero = match x.kind() {
        Kind::I64 => unsafe { alloc_atom(Kind::I64, 0i64) },
        Kind::F64 => unsafe { alloc_atom(Kind::F64, 0.0f64) },
        _ => return Err(KernelErr::Type),
    };
    op::dispatch_minus_async(zero, x, ctx).await
}
