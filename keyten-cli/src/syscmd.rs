//! K-style system commands: lines beginning with `\`.
//!
//! Supported:
//!
//! | Command          | Effect                                              |
//! |------------------|-----------------------------------------------------|
//! | `\t expr`        | evaluate `expr`, print value + elapsed milliseconds |
//! | `\v`             | list bound variable names                           |
//! | `\p [0|1]`       | show or set parallel kernel execution               |
//! | `\h` / `\?`      | help                                                |
//! | `\\`             | quit                                                |
//!
//! Anything else is an unknown command.

use std::time::Instant;

use anyhow::Result;
use nu_ansi_term::{Color, Style};

use keyten::{Env, Sym, RUNTIME};

use crate::eval_runner::{run_one, Outcome};
use crate::format::format;

pub enum SysOutcome {
    /// REPL should continue to the next prompt.
    Continue,
    /// REPL should exit.
    Quit,
}

/// Process a line that begins with `\`. The leading `\` has already been
/// observed by the caller.
pub fn dispatch(line: &str, env: &mut Env) -> Result<SysOutcome> {
    debug_assert!(line.starts_with('\\'));
    let rest = line[1..].trim_start();

    // Bare `\\` (or `\` followed by another `\`) exits.
    if rest.is_empty() || rest == "\\" {
        return Ok(SysOutcome::Quit);
    }

    // Split into command word + remainder.
    let (cmd, args) = match rest.find(char::is_whitespace) {
        Some(i) => (&rest[..i], rest[i..].trim()),
        None => (rest, ""),
    };

    match cmd {
        "t" => cmd_time(args, env)?,
        "v" => cmd_vars(env),
        "p" => cmd_parallel(args),
        "h" | "?" => cmd_help(),
        _ => println!(
            "{}",
            Style::new()
                .fg(Color::Red)
                .paint(format!("unknown system command: \\{cmd}"))
        ),
    }
    Ok(SysOutcome::Continue)
}

fn cmd_parallel(arg: &str) {
    let ok = Style::new().fg(Color::Green).bold();
    let dim = Style::new().dimmed();
    if arg.is_empty() {
        let state = if RUNTIME.parallel_enabled() { "on" } else { "off" };
        let nw = RUNTIME.worker_count();
        println!(
            "parallel: {}   workers: {}   {}",
            ok.paint(state),
            ok.paint(nw.to_string()),
            dim.paint("(`\\p 1` to enable, `\\p 0` to disable)"),
        );
        return;
    }
    match arg.trim() {
        "1" | "on" | "true" => {
            RUNTIME.set_parallel(true);
            println!("parallel: {}   workers: {}", ok.paint("on"), ok.paint(RUNTIME.worker_count().to_string()));
        }
        "0" | "off" | "false" => {
            RUNTIME.set_parallel(false);
            println!("parallel: {}", ok.paint("off"));
        }
        other => println!(
            "{}",
            Style::new()
                .fg(Color::Red)
                .paint(format!("\\p: unknown argument {other:?}; expected 0|1|on|off"))
        ),
    }
}

fn cmd_time(expr: &str, env: &mut Env) -> Result<()> {
    if expr.is_empty() {
        println!(
            "{}",
            Style::new()
                .fg(Color::Red)
                .paint("usage: \\t expr   (times the evaluation of expr)")
        );
        return Ok(());
    }
    let start = Instant::now();
    let outcome = run_one(expr, env)?;
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0;
    let timing = Style::new().dimmed().paint(format!("({})", format_ms(ms)));
    match outcome {
        Outcome::Ok(v) => println!("{}   {}", format(&v), timing),
        Outcome::Cancelled => println!(
            "{}   {}",
            Style::new().fg(Color::Yellow).paint("cancelled"),
            timing
        ),
        Outcome::Err(msg) => println!(
            "{}   {}",
            Style::new()
                .fg(Color::Red)
                .bold()
                .paint(format!("error: {msg}")),
            timing
        ),
    }
    Ok(())
}

fn cmd_vars(env: &Env) {
    let mut names: Vec<String> = env
        .iter()
        .filter_map(|(s, _)| decode_sym(*s))
        .collect();
    if names.is_empty() {
        println!(
            "{}",
            Style::new().dimmed().paint("  (no variables bound)")
        );
        return;
    }
    names.sort();
    for name in names {
        println!("  {}", Style::new().fg(Color::LightYellow).bold().paint(name));
    }
}

fn cmd_help() {
    let dim = Style::new().fg(Color::DarkGray);
    println!("System commands (start with `\\`):");
    println!("  \\t expr    time the evaluation of expr (prints value + elapsed)");
    println!("  \\v         list bound variable names");
    println!("  \\p [0|1]   show or toggle parallel kernel execution");
    println!("  \\h, \\?     this help");
    println!("  \\\\         quit");
    println!();
    println!("Language:");
    println!("  Verbs:           +  -  *  %");
    println!("  Monadic verbs:   -x (negate)  !n (til: 0..n-1)");
    println!("  Adverbs:         +/  (over)");
    println!("  Atoms:           42  3.14  \"a\"  `sym  0N  0n  0W  0w");
    println!("  Vectors:         1 2 3   1.5 2.5 3.5   `a`b `c");
    println!("  Assignment:      x: 1 2 3");
    println!(
        "{}",
        dim.paint("Try: \\t +/!1000000   |   Tab completes names   |   Ctrl-C cancels")
    );
}

fn format_ms(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.0} µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.2} ms", ms)
    } else {
        format!("{:.3} s", ms / 1000.0)
    }
}

fn decode_sym(s: Sym) -> Option<String> {
    let bytes = s.0.to_le_bytes();
    let v: Vec<u8> = bytes.iter().copied().take_while(|b| *b != 0).collect();
    if v.is_empty() {
        return None;
    }
    String::from_utf8(v).ok()
}
