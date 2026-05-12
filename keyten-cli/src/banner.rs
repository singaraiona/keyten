//! Rich startup banner: box-drawing frame with logo, build info, and a
//! small system summary.

use nu_ansi_term::{Color, Style};

use crate::format::terminal_width;
use crate::sysinfo::SysInfo;

const MIN_WIDTH: usize = 56;
const MAX_WIDTH: usize = 72;
const INNER_PAD: usize = 3; // " │   …   │ " on each side

pub fn print() {
    let info = SysInfo::probe();
    let version = env!("CARGO_PKG_VERSION");
    let build_date = option_env!("KEYTEN_BUILD_DATE").unwrap_or("unknown");
    let commit = option_env!("KEYTEN_GIT_COMMIT").unwrap_or("unknown");

    let term_w = terminal_width().clamp(MIN_WIDTH, MAX_WIDTH);
    let inner_w = term_w.saturating_sub(2);

    let style_frame = Style::new().fg(Color::DarkGray);
    let style_logo = Style::new().fg(Color::Cyan).bold();
    let style_tag = Style::new().fg(Color::LightCyan);
    let style_label = Style::new().fg(Color::DarkGray);
    let style_value = Style::new().fg(Color::White);
    let style_meta = Style::new().fg(Color::LightGray);
    let style_hint = Style::new().fg(Color::DarkGray);

    let top = format!("\u{256D}{}\u{256E}", "\u{2500}".repeat(inner_w));
    let bottom = format!("\u{2570}{}\u{256F}", "\u{2500}".repeat(inner_w));

    println!("{}", style_frame.paint(top));
    blank(inner_w, &style_frame);

    // Logo + tagline
    line(
        inner_w,
        &style_frame,
        &format!(
            "{logo}  {dot}  {tag}",
            logo = style_logo.paint("\u{1D55C}eyten"),
            dot = style_label.paint("\u{00B7}"),
            tag = style_tag.paint("streaming array language"),
        ),
        // Rendered visible width: "𝕜eyten" is 7 cells (the math-k is wide-ish
        // but usually 1 in modern terminals; we estimate 6 for the word part
        // + 1 for the math-k = 7), plus ascii. Compute conservatively.
        visible_chars("𝕜eyten  ·  streaming array language"),
    );

    blank(inner_w, &style_frame);

    // Version line
    let v = format!(
        "{} {} {} {} {} {} {}",
        style_value.paint(format!("v{version}")),
        style_label.paint("\u{00B7}"),
        style_meta.paint(build_date),
        style_label.paint("\u{00B7}"),
        style_meta.paint(commit),
        style_label.paint("\u{00B7}"),
        style_meta.paint(&info.os_arch),
    );
    line(
        inner_w,
        &style_frame,
        &v,
        visible_chars(&format!(
            "v{version} · {build_date} · {commit} · {}",
            info.os_arch
        )),
    );

    blank(inner_w, &style_frame);

    // CPU
    let cpu_text = clip_to(&info.cpu_model, inner_w.saturating_sub(INNER_PAD * 2 + 6));
    let cpu = format!(
        "{}  {}",
        style_label.paint("CPU  "),
        style_value.paint(&cpu_text),
    );
    line(inner_w, &style_frame, &cpu, visible_chars(&format!("CPU    {cpu_text}")));

    // RAM
    let mem_str = if info.mem_gib > 0.0 {
        format!("{:.1} GiB", info.mem_gib)
    } else {
        "unknown".to_string()
    };
    let ram = format!(
        "{}  {}",
        style_label.paint("RAM  "),
        style_value.paint(&mem_str),
    );
    line(inner_w, &style_frame, &ram, visible_chars(&format!("RAM    {mem_str}")));

    // Cores
    let cores = format!(
        "{}  {}",
        style_label.paint("cores"),
        style_value.paint(info.cores.to_string()),
    );
    line(inner_w, &style_frame, &cores, visible_chars(&format!("cores  {}", info.cores)));

    blank(inner_w, &style_frame);
    println!("{}", style_frame.paint(bottom));
    println!(
        "  {}",
        style_hint.paint("type \\h for help  \u{00B7}  \\\\ to quit  \u{00B7}  Tab completes")
    );
    println!();
}

fn blank(inner_w: usize, frame: &Style) {
    println!(
        "{}{}{}",
        frame.paint("\u{2502}"),
        " ".repeat(inner_w),
        frame.paint("\u{2502}")
    );
}

fn line(inner_w: usize, frame: &Style, content: &str, visible: usize) {
    let pad_left = INNER_PAD;
    let pad_right = inner_w
        .saturating_sub(pad_left)
        .saturating_sub(visible);
    println!(
        "{}{}{}{}{}",
        frame.paint("\u{2502}"),
        " ".repeat(pad_left),
        content,
        " ".repeat(pad_right),
        frame.paint("\u{2502}")
    );
}

/// Visible (printed) character width of an UTF-8 string, counting one column
/// per Rust `char`. ANSI escape sequences are not expected here (we feed in
/// the plain logical string for measurement). This treats every Unicode
/// scalar as one cell, which is right for the BMP punctuation we use.
fn visible_chars(s: &str) -> usize {
    s.chars().count()
}

fn clip_to(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let take = max.saturating_sub(1);
        let mut out: String = s.chars().take(take).collect();
        out.push('\u{2026}'); // ellipsis
        out
    }
}
