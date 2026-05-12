//! K pretty-printer.
//!
//! Renders a `RefObj` as one or more ratatui `Line`s, with colour and
//! truncation.

use keyten::{obj::attr_flags, Kind, RefObj};
use ratatui::style::Style;
use ratatui::text::{Line, Span};

use crate::tui::theme;

#[allow(dead_code)]
pub struct PrintOpts {
    pub max_width: u16,
    pub max_lines: u16,
    pub max_elems_per_line: usize,
}


impl Default for PrintOpts {
    fn default() -> Self {
        PrintOpts {
            max_width: 120,
            max_lines: 8,
            max_elems_per_line: 32,
        }
    }
}

/// Format a value into one or more `Line`s.
pub fn format(r: &RefObj, opts: &PrintOpts) -> Vec<Line<'static>> {
    let has_nulls = (r.attr() & attr_flags::HAS_NULLS) != 0;
    match r.kind() {
        Kind::Bool => format_atom_or_vec_u8(r, opts, "Bool"),
        Kind::U8 | Kind::Char => format_chars(r, opts),
        Kind::I16 => format_atom_or_vec_int::<i16>(r, opts, has_nulls, render_i16),
        Kind::I32 => format_atom_or_vec_int::<i32>(r, opts, has_nulls, render_i32),
        Kind::I64 => format_atom_or_vec_int::<i64>(r, opts, has_nulls, render_i64),
        Kind::F32 => format_atom_or_vec_float::<f32>(r, opts, render_f32),
        Kind::F64 => format_atom_or_vec_float::<f64>(r, opts, render_f64),
        Kind::Sym => format_sym(r, opts),
        Kind::Date | Kind::TimeS | Kind::TimeMs | Kind::TimeUs | Kind::TimeNs
        | Kind::DtS | Kind::DtMs | Kind::DtUs | Kind::DtNs => format_atom_or_vec_int::<i64>(
            r, opts, has_nulls, render_i64,
        ),
        Kind::List | Kind::Dict | Kind::Table => {
            vec![Line::from(Span::styled(
                format!("<{:?}>", r.kind()),
                theme::DIM,
            ))]
        }
    }
}

fn format_atom_or_vec_u8(r: &RefObj, opts: &PrintOpts, label: &str) -> Vec<Line<'static>> {
    if r.is_atom() {
        let v = unsafe { r.atom::<u8>() };
        return vec![Line::from(Span::styled(v.to_string(), theme::RESULT_NUMERIC))];
    }
    let xs = unsafe { r.as_slice::<u8>() };
    let mut spans = Vec::new();
    for (i, v) in xs.iter().take(opts.max_elems_per_line).enumerate() {
        if i > 0 {
            spans.push(Span::raw(" "));
        }
        spans.push(Span::styled(v.to_string(), theme::RESULT_NUMERIC));
    }
    if xs.len() > opts.max_elems_per_line {
        spans.push(Span::styled(
            format!(" \u{2026}({} more)", xs.len() - opts.max_elems_per_line),
            theme::DIM,
        ));
    }
    let _ = label;
    vec![Line::from(spans)]
}

fn format_chars(r: &RefObj, _opts: &PrintOpts) -> Vec<Line<'static>> {
    if r.is_atom() {
        let v = unsafe { r.atom::<u8>() };
        return vec![Line::from(Span::styled(
            format!("\"{}\"", v as char),
            theme::RESULT_STRING,
        ))];
    }
    let xs = unsafe { r.as_slice::<u8>() };
    let s: String = xs.iter().map(|b| *b as char).collect();
    vec![Line::from(Span::styled(
        format!("\"{s}\""),
        theme::RESULT_STRING,
    ))]
}

fn format_sym(r: &RefObj, opts: &PrintOpts) -> Vec<Line<'static>> {
    fn decode(packed: i64) -> String {
        let bytes = packed.to_le_bytes();
        let nz: Vec<u8> = bytes.iter().copied().take_while(|b| *b != 0).collect();
        String::from_utf8_lossy(&nz).into_owned()
    }
    if r.is_atom() {
        let v = unsafe { r.atom::<i64>() };
        let s = decode(v);
        return vec![Line::from(Span::styled(format!("`{s}"), theme::ACCENT))];
    }
    let xs = unsafe { r.as_slice::<i64>() };
    let mut spans = Vec::new();
    for (i, v) in xs.iter().take(opts.max_elems_per_line).enumerate() {
        if i > 0 {
            spans.push(Span::raw(""));
        }
        let s = decode(*v);
        spans.push(Span::styled(format!("`{s}"), theme::ACCENT));
    }
    if xs.len() > opts.max_elems_per_line {
        spans.push(Span::styled(
            format!(" \u{2026}({} more)", xs.len() - opts.max_elems_per_line),
            theme::DIM,
        ));
    }
    vec![Line::from(spans)]
}

fn format_atom_or_vec_int<T: Copy>(
    r: &RefObj,
    opts: &PrintOpts,
    has_nulls: bool,
    render: fn(T) -> (String, Style),
) -> Vec<Line<'static>> {
    let _ = has_nulls;
    if r.is_atom() {
        let v = unsafe { r.atom::<T>() };
        let (s, style) = render(v);
        return vec![Line::from(Span::styled(s, style))];
    }
    let xs = unsafe { r.as_slice::<T>() };
    let mut spans = Vec::new();
    for (i, v) in xs.iter().copied().take(opts.max_elems_per_line).enumerate() {
        if i > 0 {
            spans.push(Span::raw(" "));
        }
        let (s, style) = render(v);
        spans.push(Span::styled(s, style));
    }
    if xs.len() > opts.max_elems_per_line {
        spans.push(Span::styled(
            format!(" \u{2026}({} more)", xs.len() - opts.max_elems_per_line),
            theme::DIM,
        ));
    }
    vec![Line::from(spans)]
}

fn format_atom_or_vec_float<T: Copy>(
    r: &RefObj,
    opts: &PrintOpts,
    render: fn(T) -> (String, Style),
) -> Vec<Line<'static>> {
    format_atom_or_vec_int::<T>(r, opts, false, render)
}

fn render_i16(v: i16) -> (String, Style) {
    if v == keyten::nulls::NULL_I16 {
        ("0Nh".to_string(), theme::RESULT_NULL)
    } else {
        (v.to_string(), theme::RESULT_NUMERIC)
    }
}
fn render_i32(v: i32) -> (String, Style) {
    if v == keyten::nulls::NULL_I32 {
        ("0Ni".to_string(), theme::RESULT_NULL)
    } else {
        (v.to_string(), theme::RESULT_NUMERIC)
    }
}
fn render_i64(v: i64) -> (String, Style) {
    if v == keyten::nulls::NULL_I64 {
        ("0N".to_string(), theme::RESULT_NULL)
    } else if v == keyten::nulls::INF_I64 {
        ("0W".to_string(), theme::RESULT_NULL)
    } else {
        (v.to_string(), theme::RESULT_NUMERIC)
    }
}
fn render_f32(v: f32) -> (String, Style) {
    if v.is_nan() {
        ("0ne".to_string(), theme::RESULT_NULL)
    } else if v.is_infinite() {
        ("0we".to_string(), theme::RESULT_NULL)
    } else {
        (format_float64(v as f64), theme::RESULT_NUMERIC)
    }
}
fn render_f64(v: f64) -> (String, Style) {
    if v.is_nan() {
        ("0n".to_string(), theme::RESULT_NULL)
    } else if v.is_infinite() {
        ("0w".to_string(), theme::RESULT_NULL)
    } else {
        (format_float64(v), theme::RESULT_NUMERIC)
    }
}

fn format_float64(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}
