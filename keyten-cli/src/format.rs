//! K-style pretty-printer that emits a plain `String` with ANSI colour codes.

use nu_ansi_term::{Color, Style};

use keyten::{obj::attr_flags, Kind, RefObj};

pub fn format(r: &RefObj) -> String {
    let has_nulls = (r.attr() & attr_flags::HAS_NULLS) != 0;
    match r.kind() {
        Kind::Bool | Kind::U8 => format_byte_like(r),
        Kind::Char => format_char(r),
        Kind::I16 => format_int_like::<i16>(r, has_nulls, render_i16),
        Kind::I32 => format_int_like::<i32>(r, has_nulls, render_i32),
        Kind::I64 => format_int_like::<i64>(r, has_nulls, render_i64),
        Kind::F32 => format_int_like::<f32>(r, has_nulls, render_f32),
        Kind::F64 => format_int_like::<f64>(r, has_nulls, render_f64),
        Kind::Sym => format_sym(r),
        Kind::Date | Kind::TimeS | Kind::TimeMs | Kind::TimeUs | Kind::TimeNs
        | Kind::DtS | Kind::DtMs | Kind::DtUs | Kind::DtNs => {
            format_int_like::<i64>(r, has_nulls, render_i64)
        }
        Kind::List | Kind::Dict | Kind::Table => {
            Style::new().dimmed().paint(format!("<{:?}>", r.kind())).to_string()
        }
    }
}

fn format_byte_like(r: &RefObj) -> String {
    if r.is_atom() {
        let v = unsafe { r.atom::<u8>() };
        v.to_string()
    } else {
        let xs = unsafe { r.as_slice::<u8>() };
        xs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
    }
}

fn format_char(r: &RefObj) -> String {
    if r.is_atom() {
        let v = unsafe { r.atom::<u8>() };
        Style::new()
            .fg(Color::Green)
            .paint(format!("\"{}\"", v as char))
            .to_string()
    } else {
        let xs = unsafe { r.as_slice::<u8>() };
        let s: String = xs.iter().map(|b| *b as char).collect();
        Style::new()
            .fg(Color::Green)
            .paint(format!("\"{s}\""))
            .to_string()
    }
}

fn format_sym(r: &RefObj) -> String {
    fn decode(packed: i64) -> String {
        let bytes = packed.to_le_bytes();
        bytes
            .iter()
            .copied()
            .take_while(|b| *b != 0)
            .map(|b| b as char)
            .collect()
    }
    if r.is_atom() {
        let v = unsafe { r.atom::<i64>() };
        Style::new().fg(Color::Cyan).paint(format!("`{}", decode(v))).to_string()
    } else {
        let xs = unsafe { r.as_slice::<i64>() };
        xs.iter()
            .map(|v| {
                Style::new()
                    .fg(Color::Cyan)
                    .paint(format!("`{}", decode(*v)))
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

fn format_int_like<T: Copy>(
    r: &RefObj,
    _has_nulls: bool,
    render: fn(T) -> String,
) -> String {
    if r.is_atom() {
        let v = unsafe { r.atom::<T>() };
        render(v)
    } else {
        let xs = unsafe { r.as_slice::<T>() };
        xs.iter().map(|v| render(*v)).collect::<Vec<_>>().join(" ")
    }
}

fn render_i16(v: i16) -> String {
    if v == keyten::nulls::NULL_I16 {
        Style::new().dimmed().italic().paint("0Nh").to_string()
    } else {
        v.to_string()
    }
}
fn render_i32(v: i32) -> String {
    if v == keyten::nulls::NULL_I32 {
        Style::new().dimmed().italic().paint("0Ni").to_string()
    } else {
        v.to_string()
    }
}
fn render_i64(v: i64) -> String {
    if v == keyten::nulls::NULL_I64 {
        Style::new().dimmed().italic().paint("0N").to_string()
    } else if v == keyten::nulls::INF_I64 {
        Style::new().dimmed().italic().paint("0W").to_string()
    } else {
        v.to_string()
    }
}
fn render_f32(v: f32) -> String {
    if v.is_nan() {
        Style::new().dimmed().italic().paint("0ne").to_string()
    } else if v.is_infinite() {
        Style::new().dimmed().italic().paint("0we").to_string()
    } else {
        format_float(v as f64)
    }
}
fn render_f64(v: f64) -> String {
    if v.is_nan() {
        Style::new().dimmed().italic().paint("0n").to_string()
    } else if v.is_infinite() {
        Style::new().dimmed().italic().paint("0w").to_string()
    } else {
        format_float(v)
    }
}

fn format_float(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}
