//! Recursive-descent parser over the token stream.
//!
//! Grammar (informal):
//!
//! ```text
//! program     ::= statement (";" statement)*
//! statement   ::= assign | expr
//! assign      ::= IDENT ":" expr
//! expr        ::= prefix
//! prefix      ::= ("+" | "-") prefix | infix     ; monadic + / -
//! infix       ::= primary (verb expr)?            ; K-style right-associative dyadic
//! verb        ::= "+" | "-" | "*" | "%"
//! primary     ::= adverb | atom_or_vec | list | name | "(" expr ")"
//! adverb      ::= verb "/" expr                   ; only V/x form supported
//! atom_or_vec ::= number_lit (S number_lit)*      ; sequence of literals → vector
//! list        ::= "(" expr (";" expr)* ")"
//! ```
//!
//! Vector literals are sequences of *homogeneous* number literals separated by
//! whitespace (no commas). If types differ, we promote (int → float) up to F64.

use crate::ast::{AdvId, AtomLit, Expr, Span};
use crate::kind::Kind;
use crate::op::OpId;
use crate::parse::lex::{Token, TokenKind};
use crate::sym::intern;

#[derive(Debug)]
pub struct ParseErr {
    pub msg: String,
    pub span: Span,
}

/// Whether a token kind can appear as the *result* position of an expression
/// (i.e. produces a value). Used to decide if the next verb is dyadic.
fn is_value_producing(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Int(_)
            | TokenKind::Float(_)
            | TokenKind::Str(_)
            | TokenKind::Sym(_)
            | TokenKind::Ident(_)
            | TokenKind::NullI64
            | TokenKind::NullF64
            | TokenKind::InfI64
            | TokenKind::InfF64
            | TokenKind::RParen
    )
}

/// Map a token kind to its `OpId`. Returns `None` if the token isn't a verb.
fn verb_of(kind: &TokenKind) -> Option<OpId> {
    match kind {
        TokenKind::Plus => Some(OpId::Plus),
        TokenKind::Minus => Some(OpId::Minus),
        TokenKind::Times => Some(OpId::Times),
        TokenKind::Div => Some(OpId::Div),
        TokenKind::Bang => Some(OpId::Bang),
        TokenKind::At => Some(OpId::At),
        TokenKind::Hash => Some(OpId::Hash),
        TokenKind::Comma => Some(OpId::Comma),
        TokenKind::Eq => Some(OpId::Eq),
        TokenKind::Lt => Some(OpId::Lt),
        TokenKind::Gt => Some(OpId::Gt),
        TokenKind::Tilde => Some(OpId::Tilde),
        TokenKind::Amp => Some(OpId::Amp),
        TokenKind::Pipe => Some(OpId::Pipe),
        TokenKind::Underscore => Some(OpId::Underscore),
        TokenKind::Dollar => Some(OpId::Dollar),
        TokenKind::Caret => Some(OpId::Caret),
        TokenKind::Question => Some(OpId::Question),
        _ => None,
    }
}

pub fn parse_program(src: &str, tokens: &[Token]) -> Result<Expr, ParseErr> {
    let mut p = Parser { src, tokens, pos: 0 };
    let stmts = p.parse_program()?;
    let _ = src;
    Ok(if stmts.len() == 1 {
        stmts.into_iter().next().unwrap()
    } else {
        let span = match (stmts.first(), stmts.last()) {
            (Some(a), Some(b)) => Span::merge(a.span(), b.span()),
            _ => Span::default(),
        };
        Expr::Seq { items: stmts, span }
    })
}

struct Parser<'a> {
    src: &'a str,
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    fn peek_at(&self, k: usize) -> Option<&Token> {
        self.tokens.get(self.pos + k)
    }
    fn bump(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.peek().map(|t| &t.kind) == Some(kind) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn err<T>(&self, msg: impl Into<String>, span: Span) -> Result<T, ParseErr> {
        Err(ParseErr { msg: msg.into(), span })
    }
    fn span_end(&self) -> Span {
        if let Some(t) = self.tokens.get(self.pos.saturating_sub(1)) {
            Span::new(t.span.end, t.span.end)
        } else {
            Span::new(self.src.len() as u32, self.src.len() as u32)
        }
    }

    // ---- top level ---------------------------------------------------

    fn parse_program(&mut self) -> Result<Vec<Expr>, ParseErr> {
        let mut out = Vec::new();
        // empty source → empty program
        if self.peek().is_none() {
            return Ok(out);
        }
        loop {
            let s = self.parse_statement()?;
            out.push(s);
            if !self.eat(&TokenKind::Semicolon) {
                break;
            }
            if self.peek().is_none() {
                break;
            }
        }
        if let Some(t) = self.peek() {
            return self.err(format!("unexpected token `{:?}`", t.kind), t.span);
        }
        Ok(out)
    }

    fn parse_statement(&mut self) -> Result<Expr, ParseErr> {
        // Assignment: IDENT ":" expr.
        if let Some(t0) = self.peek() {
            if matches!(t0.kind, TokenKind::Ident(_)) {
                if let Some(t1) = self.peek_at(1) {
                    if matches!(t1.kind, TokenKind::Colon) {
                        let (name_str, name_span) = match self.bump().unwrap() {
                            Token { kind: TokenKind::Ident(n), span } => (n.clone(), *span),
                            _ => unreachable!(),
                        };
                        self.bump(); // colon
                        let value = self.parse_expr()?;
                        let span = Span::merge(name_span, value.span());
                        let sym = intern(&name_str).map_err(|_| ParseErr {
                            msg: format!("invalid symbol name `{name_str}`"),
                            span: name_span,
                        })?;
                        return Ok(Expr::Assign {
                            name: sym,
                            value: Box::new(value),
                            span,
                        });
                    }
                }
            }
        }
        self.parse_expr()
    }

    // ---- expression layers -------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseErr> {
        self.parse_prefix()
    }

    /// Monadic verb before any primary. Bows out if the next token is an
    /// adverb (`/`, `\`) — that form is `V/x`, parsed in `parse_primary`.
    fn parse_prefix(&mut self) -> Result<Expr, ParseErr> {
        if let Some(t) = self.peek() {
            if verb_of(&t.kind).is_some() {
                let next = self.peek_at(1).map(|x| &x.kind);
                if matches!(
                    next,
                    Some(TokenKind::Slash) | Some(TokenKind::Backslash) | Some(TokenKind::Tick)
                ) {
                    return self.parse_infix();
                }
                if let Some(verb) = verb_of(&t.kind) {
                    // Monadic-position rule: monadic when the previous token is
                    // *not* a value-producing token (i.e. not an atom/identifier/closing paren).
                    let prev = self.tokens.get(self.pos.wrapping_sub(1));
                    let allow_monadic = self.pos == 0
                        || prev
                            .map(|t| !is_value_producing(&t.kind))
                            .unwrap_or(true);
                    if allow_monadic {
                        let start = t.span;
                        self.bump();
                        let arg = self.parse_prefix()?;
                        let span = Span::merge(start, arg.span());
                        return Ok(Expr::Monad {
                            verb,
                            arg: Box::new(arg),
                            span,
                        });
                    }
                }
            }
        }
        self.parse_infix()
    }

    /// Right-associative dyadic verbs. K convention: `a + b * c` ≡ `a + (b * c)`.
    fn parse_infix(&mut self) -> Result<Expr, ParseErr> {
        let lhs = self.parse_primary()?;
        if let Some(t) = self.peek() {
            if let Some(v) = verb_of(&t.kind) {
                self.bump();
                let rhs = self.parse_expr()?;
                let span = Span::merge(lhs.span(), rhs.span());
                return Ok(Expr::Dyad {
                    verb: v,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span,
                });
            }
        }
        Ok(lhs)
    }

    /// A primary expression — possibly an adverb-prefixed verb.
    fn parse_primary(&mut self) -> Result<Expr, ParseErr> {
        // Adverb form: V/x (or V\x).
        if let Some(t) = self.peek() {
            let verb = verb_of(&t.kind);
            if let Some(v) = verb {
                if let Some(t1) = self.peek_at(1) {
                    if matches!(
                        t1.kind,
                        TokenKind::Slash | TokenKind::Backslash | TokenKind::Tick
                    ) {
                        let start = t.span;
                        self.bump(); // verb
                        let adv_tok = self.bump().unwrap();
                        let adv = match adv_tok.kind {
                            TokenKind::Slash => AdvId::Over,
                            TokenKind::Backslash => AdvId::Scan,
                            TokenKind::Tick => AdvId::Each,
                            _ => unreachable!(),
                        };
                        let arg = self.parse_expr()?;
                        let span = Span::merge(start, arg.span());
                        return Ok(Expr::Adverb {
                            adv,
                            verb: v,
                            arg: Box::new(arg),
                            span,
                        });
                    }
                }
            }
        }

        // Parenthesised expression or generic list literal `(a; b; c)`.
        if let Some(t) = self.peek() {
            if matches!(t.kind, TokenKind::LParen) {
                let lparen = t.span;
                self.bump();
                let first = self.parse_expr()?;
                if self.eat(&TokenKind::RParen) {
                    return Ok(first);
                }
                // List: first ; rest
                if !self.eat(&TokenKind::Semicolon) {
                    let span = self.span_end();
                    return self.err("expected `;` or `)` in parenthesised expression", span);
                }
                let mut items = vec![first];
                loop {
                    let item = self.parse_expr()?;
                    items.push(item);
                    if self.eat(&TokenKind::RParen) {
                        break;
                    }
                    if !self.eat(&TokenKind::Semicolon) {
                        let span = self.span_end();
                        return self.err("expected `;` or `)` inside list", span);
                    }
                }
                let span = Span::merge(lparen, self.span_end());
                return Ok(Expr::ListLit { items, span });
            }
        }

        // Name reference.
        if let Some(t) = self.peek() {
            if matches!(t.kind, TokenKind::Ident(_)) {
                let (name, span) = match self.bump().unwrap() {
                    Token { kind: TokenKind::Ident(n), span } => (n.clone(), *span),
                    _ => unreachable!(),
                };
                let sym = intern(&name).map_err(|_| ParseErr {
                    msg: format!("invalid symbol name `{name}`"),
                    span,
                })?;
                return Ok(Expr::Name { sym, span });
            }
        }

        // Numeric / sym / string atoms — possibly a vector.
        self.parse_atom_or_vec()
    }

    fn parse_atom_or_vec(&mut self) -> Result<Expr, ParseErr> {
        let first = self.parse_one_literal_or_str()?;
        // Try to greedily collect more homogeneous literals separated by
        // whitespace (no comma, no operator). We only do this for numeric/sym
        // tokens — strings stay as a single Char vector.
        match &first {
            Expr::AtomLit { lit, .. } => {
                let initial_kind = lit.kind();
                if matches!(
                    initial_kind,
                    Kind::I64 | Kind::F64 | Kind::Sym
                ) {
                    let mut items = vec![*lit];
                    let mut span = first.span();
                    let mut promoted = initial_kind;
                    while let Some(t) = self.peek() {
                        // Only consume more literals when they are immediately
                        // following whitespace-separated numeric / sym tokens.
                        // A binary operator or any other token terminates.
                        let next_lit = match &t.kind {
                            TokenKind::Int(_)
                            | TokenKind::Float(_)
                            | TokenKind::Sym(_)
                            | TokenKind::NullI64
                            | TokenKind::NullF64
                            | TokenKind::InfI64
                            | TokenKind::InfF64 => true,
                            _ => false,
                        };
                        if !next_lit {
                            break;
                        }
                        let lit = self.parse_one_atom_token()?;
                        // Decide promoted kind: if any literal is F64, vector is F64.
                        let kind = lit.kind();
                        match (promoted, kind) {
                            (Kind::I64, Kind::I64) => {}
                            (Kind::F64, _) => {}
                            (Kind::I64, Kind::F64) => promoted = Kind::F64,
                            (Kind::Sym, Kind::Sym) => {}
                            _ => {
                                let span = first.span();
                                return self.err(
                                    "heterogeneous types in vector literal — use `(a; b; …)` for a generic list",
                                    span,
                                );
                            }
                        }
                        span = Span::merge(span, Span::new(0, 0)); // refined below
                        items.push(lit);
                    }
                    if items.len() > 1 {
                        // Refine span across collected literals.
                        let last_span = match items.last() {
                            Some(_) => self
                                .tokens
                                .get(self.pos.saturating_sub(1))
                                .map(|t| t.span)
                                .unwrap_or(span),
                            None => span,
                        };
                        let span = Span::merge(span, last_span);
                        return Ok(Expr::VecLit {
                            kind: promoted,
                            items,
                            span,
                        });
                    }
                }
            }
            _ => {}
        }
        Ok(first)
    }

    /// One numeric / sym / null-or-inf literal, returned as `AtomLit`.
    fn parse_one_atom_token(&mut self) -> Result<AtomLit, ParseErr> {
        let t = self.bump().cloned().ok_or_else(|| ParseErr {
            msg: "unexpected end of input".into(),
            span: self.span_end(),
        })?;
        match t.kind {
            TokenKind::Int(n) => Ok(AtomLit::I64(n)),
            TokenKind::Float(f) => Ok(AtomLit::F64(f)),
            TokenKind::Sym(s) => {
                let sym = intern(&s).map_err(|_| ParseErr {
                    msg: format!("invalid symbol literal `{s}`"),
                    span: t.span,
                })?;
                Ok(AtomLit::Sym(sym))
            }
            TokenKind::NullI64 => Ok(AtomLit::NullI64),
            TokenKind::NullF64 => Ok(AtomLit::NullF64),
            TokenKind::InfI64 => Ok(AtomLit::InfI64),
            TokenKind::InfF64 => Ok(AtomLit::InfF64),
            _ => self.err(format!("expected a literal, got `{:?}`", t.kind), t.span),
        }
    }

    /// Parse a literal token OR a string (string is a Char vector, not an atom).
    fn parse_one_literal_or_str(&mut self) -> Result<Expr, ParseErr> {
        let t = self.peek().cloned().ok_or_else(|| ParseErr {
            msg: "expected an expression".into(),
            span: self.span_end(),
        })?;
        match &t.kind {
            TokenKind::Str(s) => {
                self.bump();
                if s.chars().count() == 1 {
                    let ch = s.as_bytes()[0];
                    Ok(Expr::AtomLit {
                        lit: AtomLit::Char(ch),
                        span: t.span,
                    })
                } else {
                    let items: Vec<AtomLit> = s.bytes().map(AtomLit::Char).collect();
                    Ok(Expr::VecLit {
                        kind: Kind::Char,
                        items,
                        span: t.span,
                    })
                }
            }
            _ => {
                let lit = self.parse_one_atom_token()?;
                Ok(Expr::AtomLit {
                    lit,
                    span: t.span,
                })
            }
        }
    }
}
