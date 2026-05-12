//! Surface syntax of the language as an AST.
//!
//! `Span` is a byte range into the original source so eval / parse errors can
//! point back to the offending text.

use crate::kind::Kind;
use crate::op::OpId;
use crate::sym::Sym;
use std::sync::Arc;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }
    pub fn merge(a: Span, b: Span) -> Self {
        Span {
            start: a.start.min(b.start),
            end: a.end.max(b.end),
        }
    }
}

/// Literal atoms reachable from surface syntax.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AtomLit {
    Bool(bool),
    I64(i64),
    F64(f64),
    Char(u8),
    Sym(Sym),
    NullI64,
    NullF64,
    InfI64,
    InfF64,
}

impl AtomLit {
    /// Kind of the atom this literal denotes.
    pub fn kind(self) -> Kind {
        match self {
            AtomLit::Bool(_) => Kind::Bool,
            AtomLit::I64(_) | AtomLit::NullI64 | AtomLit::InfI64 => Kind::I64,
            AtomLit::F64(_) | AtomLit::NullF64 | AtomLit::InfF64 => Kind::F64,
            AtomLit::Char(_) => Kind::Char,
            AtomLit::Sym(_) => Kind::Sym,
        }
    }
}

/// Adverb identifier in source.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum AdvId {
    /// `/` — over: reduce the input with the verb.
    Over,
    /// `\` — scan: running aggregate (same length as input).
    Scan,
    /// `'` — each: apply the monadic verb to each element.
    Each,
    /// `':` — eachprior: apply the dyadic verb to consecutive pairs.
    /// `f':v` produces `[v[0], f(v[1], v[0]), f(v[2], v[1]), ...]`.
    EachPrior,
}

/// Expressions in the source language.
#[derive(Debug)]
pub enum Expr {
    /// Single-atom literal.
    AtomLit { lit: AtomLit, span: Span },
    /// Typed-vector literal (homogeneous atoms after parsing).
    VecLit {
        kind: Kind,
        items: Vec<AtomLit>,
        span: Span,
    },
    /// Generic list `(a; b; c)` of arbitrary sub-expressions.
    ListLit { items: Vec<Expr>, span: Span },
    /// Variable lookup.
    Name { sym: Sym, span: Span },
    /// `name: expr`.
    Assign {
        name: Sym,
        value: Box<Expr>,
        span: Span,
    },
    /// Dyadic verb application: `lhs V rhs`.
    Dyad {
        verb: OpId,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// Monadic verb application: `V x` (currently only `+x` and `-x`).
    Monad {
        verb: OpId,
        arg: Box<Expr>,
        span: Span,
    },
    /// Adverb composition: `V/x`, `V\x`, etc.
    Adverb {
        adv: AdvId,
        verb: OpId,
        arg: Box<Expr>,
        span: Span,
    },
    /// Multiple top-level statements separated by `;`. Returns the last
    /// statement's value.
    Seq { items: Vec<Expr>, span: Span },
    /// `$[c;t;e]` — conditional. Eval `c`; if non-zero/true, eval `t`; else
    /// eval `e`. K9 also has the cond ladder form `$[c1;t1;c2;t2;...;e]`
    /// where odd positions are conditions and even positions are branches,
    /// with the final unpaired element being the default else. v1 supports
    /// the 3-arg form only; the ladder form is a parser extension.
    Cond {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
        span: Span,
    },
    /// Lambda expression: `{[x;y]body}` (explicit param list) or `{body}`
    /// (implicit single arg named `x`). The body is `Arc`d so the lambda
    /// value can be cheaply cloned without copying the AST tree.
    Lambda {
        params: Vec<Sym>,
        body: Arc<Expr>,
        span: Span,
    },
    /// Function application: `func[a;b;c]`. `args` may be empty for `f[]`.
    /// Currently only the bracket form; juxtaposition `f x` is v1.2.
    Apply {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
}

/// Heap-allocated lambda body. Pointed at by the atom payload of a
/// `Kind::Lambda` cell. Cloning the lambda `RefObj` shares this inner
/// (we ref-share via the cell's rc, not the `Arc<Expr>` inside).
pub struct LambdaInner {
    pub params: Vec<Sym>,
    pub body: Arc<Expr>,
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::AtomLit { span, .. } => *span,
            Expr::VecLit { span, .. } => *span,
            Expr::ListLit { span, .. } => *span,
            Expr::Name { span, .. } => *span,
            Expr::Assign { span, .. } => *span,
            Expr::Dyad { span, .. } => *span,
            Expr::Monad { span, .. } => *span,
            Expr::Adverb { span, .. } => *span,
            Expr::Seq { span, .. } => *span,
            Expr::Cond { span, .. } => *span,
            Expr::Lambda { span, .. } => *span,
            Expr::Apply { span, .. } => *span,
        }
    }
}
