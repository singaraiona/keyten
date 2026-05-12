//! K-style pretty-printer with terminal-width truncation.
//!
//! Vector results are truncated when they would not fit on a single line.
//! Truncation is signalled with `..` at the end (K convention). Width comes
//! from `TIOCGWINSZ` on Unix, falling back to the `COLUMNS` env var, then 80.

use nu_ansi_term::{Color, Style};

use keyten::{obj::attr_flags, Kind, RefObj};

/// Reserved width for the `..` suffix when we elide.
const ELLIPSIS: &str = "..";

pub fn format(r: &RefObj) -> String {
    format_with_width(r, terminal_width())
}

/// Format `r` aiming to fit within `max_width` columns.
pub fn format_with_width(r: &RefObj, max_width: usize) -> String {
    let has_nulls = (r.attr() & attr_flags::HAS_NULLS) != 0;
    match r.kind() {
        Kind::Bool | Kind::U8 => format_bytes(r, max_width),
        Kind::Char => format_chars(r),
        Kind::I16 => format_int_vec::<i16>(r, has_nulls, render_i16, max_width),
        Kind::I32 => format_int_vec::<i32>(r, has_nulls, render_i32, max_width),
        Kind::I64 => format_int_vec::<i64>(r, has_nulls, render_i64, max_width),
        Kind::F32 => format_int_vec::<f32>(r, has_nulls, render_f32, max_width),
        Kind::F64 => format_int_vec::<f64>(r, has_nulls, render_f64, max_width),
        Kind::Sym => format_sym(r, max_width),
        Kind::Date | Kind::TimeS | Kind::TimeMs | Kind::TimeUs | Kind::TimeNs
        | Kind::DtS | Kind::DtMs | Kind::DtUs | Kind::DtNs => {
            format_int_vec::<i64>(r, has_nulls, render_i64, max_width)
        }
        Kind::List | Kind::Dict | Kind::Table | Kind::Lambda => Style::new()
            .dimmed()
            .paint(format!("<{:?}>", r.kind()))
            .to_string(),
    }
}

// ---- helpers ---------------------------------------------------------

/// Build a space-separated rendering of `xs` that fits within `max_width`
/// columns. If it doesn't fit, emit as many elements as fit and append `..`.
fn render_vec<T: Copy>(
    xs: &[T],
    render: impl Fn(T) -> String,
    max_width: usize,
) -> String {
    let mut out = String::new();
    let n = xs.len();
    // Width budget: reserve room for " .." suffix when we truncate.
    let trunc_budget = max_width.saturating_sub(ELLIPSIS.len() + 1);
    for (i, v) in xs.iter().copied().enumerate() {
        let r = render(v);
        let extra = if out.is_empty() { 0 } else { 1 } + visible_len(&r);
        let remaining = max_width.saturating_sub(visible_len(&out));
        // If there are more elements after this AND adding this one would
        // leave no room for the truncation marker, stop here.
        let last = i == n - 1;
        if !last && visible_len(&out) + extra > trunc_budget {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(ELLIPSIS);
            return out;
        }
        // Even the current element won't fit — emit ellipsis and stop.
        if extra > remaining {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(ELLIPSIS);
            return out;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&r);
    }
    out
}

fn format_bytes(r: &RefObj, max_width: usize) -> String {
    if r.is_atom() {
        return unsafe { r.atom::<u8>() }.to_string();
    }
    let xs = unsafe { r.as_slice::<u8>() };
    render_vec(xs, |v: u8| v.to_string(), max_width)
}

fn format_chars(r: &RefObj) -> String {
    if r.is_atom() {
        let v = unsafe { r.atom::<u8>() };
        return Style::new()
            .fg(Color::Green)
            .paint(format!("\"{}\"", v as char))
            .to_string();
    }
    let xs = unsafe { r.as_slice::<u8>() };
    let s: String = xs.iter().map(|b| *b as char).collect();
    Style::new()
        .fg(Color::Green)
        .paint(format!("\"{s}\""))
        .to_string()
}

fn format_sym(r: &RefObj, max_width: usize) -> String {
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
        return Style::new()
            .fg(Color::Cyan)
            .paint(format!("`{}", decode(v)))
            .to_string();
    }
    let xs = unsafe { r.as_slice::<i64>() };
    // Symbols are written without spaces in K: `a`b`c
    let mut out = String::new();
    let trunc_budget = max_width.saturating_sub(ELLIPSIS.len() + 1);
    let n = xs.len();
    for (i, v) in xs.iter().copied().enumerate() {
        let s = format!("`{}", decode(v));
        let styled = Style::new().fg(Color::Cyan).paint(s.clone()).to_string();
        let last = i == n - 1;
        if !last && visible_len(&out) + visible_len(&styled) > trunc_budget {
            out.push_str(ELLIPSIS);
            return out;
        }
        out.push_str(&styled);
    }
    out
}

fn format_int_vec<T: Copy>(
    r: &RefObj,
    _has_nulls: bool,
    render: fn(T) -> String,
    max_width: usize,
) -> String {
    if r.is_atom() {
        let v = unsafe { r.atom::<T>() };
        return render(v);
    }
    let xs = unsafe { r.as_slice::<T>() };
    render_vec(xs, render, max_width)
}

// ---- per-type renders ------------------------------------------------

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

// ---- width-aware string measurement ---------------------------------

/// Visible width of `s`, ignoring ANSI escape sequences. Counts bytes for
/// ASCII (which is all we currently emit); good enough for our purposes.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // CSI sequence: ESC [ ... letter
            i += 2;
            while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            } // consume final byte
        } else {
            n += 1;
            i += 1;
        }
    }
    n
}

pub fn terminal_width() -> usize {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws as *mut _) == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }
    if let Ok(v) = std::env::var("COLUMNS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    80
}
