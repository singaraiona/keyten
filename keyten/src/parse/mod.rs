//! Surface-syntax parser for the v1 language subset.
//!
//! Public entry: [`parse`]. Returns an [`Expr`] (single statement or `Seq`)
//! or a [`ParseErr`] with a span into the input source.

mod lex;
mod grammar;

use crate::ast::Expr;

pub use grammar::ParseErr;
pub use lex::{Token, TokenKind};

/// Parse a source string into an `Expr`. The result is a `Seq` of one or more
/// statements when the source contains `;` separators; otherwise the single
/// `Expr`.
pub fn parse(source: &str) -> Result<Expr, ParseErr> {
    let tokens = lex::tokenize(source)?;
    grammar::parse_program(source, &tokens)
}
