//! Tokeniser. Splits source text into a flat token stream with byte spans.

use crate::ast::Span;
use crate::parse::ParseErr;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Integer literal in source (decimal, optional leading `-`).
    Int(i64),
    /// Float literal in source (contains `.` or `e`).
    Float(f64),
    /// Character literal: `"a"` or `"abc"` (a string is a Char vector).
    Str(String),
    /// Backtick-prefixed symbol: `` `abc ``.
    Sym(String),
    /// Bareword identifier (variable name).
    Ident(String),
    /// Null/inf literals: `0N`, `0n`, `0W`, `0w`.
    NullI64,
    NullF64,
    InfI64,
    InfF64,

    Plus,
    Minus,
    Times,
    Div,
    Bang, // `!` — monadic: til (range); dyadic: mod (reserved)
    At, // `@` — monadic: type accessor; dyadic: index/apply (reserved)
    Hash, // `#` — monadic: count; dyadic: take (reserved)
    Comma, // `,` — monadic: enlist; dyadic: concatenate (reserved)

    Slash, // `/` adverb
    Backslash, // `\` adverb (reserved)

    Colon,         // `:` assignment
    Semicolon,     // `;` statement separator (or list element separator inside parens)
    LParen,
    RParen,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, ParseErr> {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4);
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        // Whitespace.
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
            i += 1;
            continue;
        }
        // Line comment: `/ ...` to end of line. But only after whitespace or
        // at start of input — otherwise `/` is the adverb.
        // The K convention is `/` at column-start = comment. We keep it simple:
        // treat `/ ` (slash followed by space) at column-start or after newline
        // as a comment. Not implemented in v1 for safety; users avoid that
        // form.

        let start = i as u32;

        // String literal: `"..."`.
        if c == b'"' {
            i += 1;
            let s_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i >= bytes.len() {
                return Err(ParseErr {
                    msg: "unterminated string literal".into(),
                    span: Span::new(start, bytes.len() as u32),
                });
            }
            let s = decode_string(&src[s_start..i]);
            i += 1; // consume closing `"`
            out.push(Token {
                kind: TokenKind::Str(s),
                span: Span::new(start, i as u32),
            });
            continue;
        }

        // Symbol literal: `` `ident ``.
        if c == b'`' {
            i += 1;
            let s_start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let s = src[s_start..i].to_string();
            out.push(Token {
                kind: TokenKind::Sym(s),
                span: Span::new(start, i as u32),
            });
            continue;
        }

        // Numbers and null/inf literals.
        // `0N`, `0n`, `0W`, `0w` are atomic null/inf tokens.
        // `0`..`9`(`.`|`e`)? are numeric literals.
        // Leading `-` followed by digit is part of the number (unary minus
        // is handled in the parser; we tokenise `-3` as MINUS INT(3) so that
        // expressions like `x-3` parse correctly).
        if c.is_ascii_digit() {
            // Lookahead for null/inf forms after a `0`.
            if c == b'0' && i + 1 < bytes.len() {
                match bytes[i + 1] {
                    b'N' => {
                        i += 2;
                        out.push(Token {
                            kind: TokenKind::NullI64,
                            span: Span::new(start, i as u32),
                        });
                        continue;
                    }
                    b'n' => {
                        i += 2;
                        out.push(Token {
                            kind: TokenKind::NullF64,
                            span: Span::new(start, i as u32),
                        });
                        continue;
                    }
                    b'W' => {
                        i += 2;
                        out.push(Token {
                            kind: TokenKind::InfI64,
                            span: Span::new(start, i as u32),
                        });
                        continue;
                    }
                    b'w' => {
                        i += 2;
                        out.push(Token {
                            kind: TokenKind::InfF64,
                            span: Span::new(start, i as u32),
                        });
                        continue;
                    }
                    _ => {}
                }
            }
            // Plain number.
            let n_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let mut is_float = false;
            if i < bytes.len() && bytes[i] == b'.' {
                is_float = true;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                is_float = true;
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s = &src[n_start..i];
            let kind = if is_float {
                let v = s.parse::<f64>().map_err(|_| ParseErr {
                    msg: format!("invalid float literal `{s}`"),
                    span: Span::new(start, i as u32),
                })?;
                TokenKind::Float(v)
            } else {
                let v = s.parse::<i64>().map_err(|_| ParseErr {
                    msg: format!("invalid int literal `{s}`"),
                    span: Span::new(start, i as u32),
                })?;
                TokenKind::Int(v)
            };
            out.push(Token {
                kind,
                span: Span::new(start, i as u32),
            });
            continue;
        }

        // Identifier.
        if c.is_ascii_alphabetic() || c == b'_' {
            let s_start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let s = src[s_start..i].to_string();
            out.push(Token {
                kind: TokenKind::Ident(s),
                span: Span::new(start, i as u32),
            });
            continue;
        }

        // Single-character tokens.
        let kind = match c {
            b'+' => TokenKind::Plus,
            b'-' => TokenKind::Minus,
            b'*' => TokenKind::Times,
            b'%' => TokenKind::Div,
            b'!' => TokenKind::Bang,
            b'@' => TokenKind::At,
            b'#' => TokenKind::Hash,
            b',' => TokenKind::Comma,
            b'/' => TokenKind::Slash,
            b'\\' => TokenKind::Backslash,
            b':' => TokenKind::Colon,
            b';' => TokenKind::Semicolon,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            other => {
                return Err(ParseErr {
                    msg: format!("unexpected character `{}`", other as char),
                    span: Span::new(start, (i + 1) as u32),
                });
            }
        };
        i += 1;
        out.push(Token {
            kind,
            span: Span::new(start, i as u32),
        });
    }
    Ok(out)
}

fn decode_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '\\' => out.push('\\'),
                    '"' => out.push('"'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
