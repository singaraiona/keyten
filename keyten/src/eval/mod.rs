//! Tree-walking evaluator over the AST.
//!
//! Public surface: [`Env`], [`EvalErr`], [`eval`], [`eval_async`].

mod env;
pub mod tree;

pub use env::Env;
pub use tree::{eval, eval_async, EvalErr};
