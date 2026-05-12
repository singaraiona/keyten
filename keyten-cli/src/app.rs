//! Main REPL loop. reedline handles line editing + history + highlighting +
//! completion; each successful line is handed to `eval_runner::run_one` which
//! spins up a tokio runtime for that submission.

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use std::io::BufRead;

use anyhow::Result;
use nu_ansi_term::{Color, Style};
use reedline::{
    ColumnarMenu, DefaultHinter, Emacs, KeyCode, KeyModifiers, MenuBuilder, Reedline, ReedlineEvent,
    ReedlineMenu, Signal,
};

use keyten::Env;

use crate::completer::KCompleter;
use crate::eval_runner::{run_one, Outcome};
use crate::format::format;
use crate::highlighter::KHighlighter;
use crate::history::open as open_history;
use crate::names::{Names, SharedNames};
use crate::prompt::KPrompt;
use crate::validator::KValidator;

pub fn run() -> Result<()> {
    let env: Rc<RefCell<Env>> = Rc::new(RefCell::new(Env::new()));
    let names: SharedNames = Arc::new(Mutex::new(Names::default()));

    // If stdin is not a TTY, switch to a simple line-by-line mode so scripts
    // can pipe expressions through.
    if !is_stdin_tty() {
        return run_pipe(env);
    }

    let mut line_editor = build_editor(names.clone());

    print_banner();

    loop {
        let sig = line_editor.read_line(&KPrompt);
        match sig {
            Ok(Signal::Success(text)) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with(':') {
                    handle_meta(trimmed);
                    continue;
                }
                let mut env_borrow = env.borrow_mut();
                let outcome = run_one(&text, &mut env_borrow)?;
                // Refresh the names registry so highlighter / completer see new bindings.
                if let Ok(mut n) = names.lock() {
                    n.refresh_from(&env_borrow);
                }
                drop(env_borrow);
                render_outcome(outcome);
            }
            Ok(Signal::CtrlC) => continue,
            Ok(Signal::CtrlD) => {
                println!();
                break;
            }
            Err(err) => {
                eprintln!("\nreedline error: {err}");
                break;
            }
        }
    }

    Ok(())
}

fn build_editor(names: SharedNames) -> Reedline {
    let highlighter = Box::new(KHighlighter::new(names.clone()));
    let completer = Box::new(KCompleter::new(names.clone()));
    let validator = Box::new(KValidator);
    let hinter = Box::new(DefaultHinter::default().with_style(Style::new().fg(Color::DarkGray)));

    let completion_menu = Box::new(
        ColumnarMenu::default()
            .with_name("completion_menu")
            .with_columns(1),
    );

    let mut keybindings = reedline::default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu("completion_menu".to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );

    let edit_mode = Box::new(Emacs::new(keybindings));

    let mut editor = Reedline::create()
        .with_highlighter(highlighter)
        .with_completer(completer)
        .with_validator(validator)
        .with_hinter(hinter)
        .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
        .with_edit_mode(edit_mode);

    if let Some(history) = open_history() {
        editor = editor.with_history(Box::new(history));
    }
    editor
}

fn render_outcome(outcome: Outcome) {
    match outcome {
        Outcome::Ok(value) => {
            let s = format(&value);
            println!("{s}");
        }
        Outcome::Cancelled => {
            println!("{}", Style::new().fg(Color::Yellow).paint("cancelled"));
        }
        Outcome::Err(msg) => {
            println!(
                "{}",
                Style::new()
                    .fg(Color::Red)
                    .bold()
                    .paint(format!("error: {msg}"))
            );
        }
    }
    let _ = io::stdout().flush();
}

fn print_banner() {
    let line1 = Style::new().fg(Color::Cyan).bold().paint("keyten 0.1.0");
    let line2 = Style::new()
        .fg(Color::DarkGray)
        .paint("  type expressions \u{2022} :help for help \u{2022} :q to quit");
    println!("{line1}");
    println!("{line2}");
}

fn handle_meta(line: &str) {
    match line {
        ":q" | ":quit" | ":exit" => {
            println!();
            std::process::exit(0);
        }
        ":h" | ":help" => print_help(),
        _ => println!(
            "{}",
            Style::new()
                .fg(Color::Red)
                .paint(format!("unknown command `{line}`"))
        ),
    }
}

fn is_stdin_tty() -> bool {
    // Best-effort TTY check via libc::isatty on fd 0.
    #[cfg(unix)]
    unsafe {
        extern "C" {
            fn isatty(fd: i32) -> i32;
        }
        isatty(0) != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Non-interactive mode: read lines from stdin, evaluate each, print result.
/// No colours, no prompt, no editing — suitable for `echo … | keyten` or
/// `keyten < file.k`.
fn run_pipe(env: Rc<RefCell<Env>>) -> Result<()> {
    let stdin = std::io::stdin();
    let lock = stdin.lock();
    for line in lock.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('/') {
            continue;
        }
        if trimmed.starts_with(':') {
            match trimmed {
                ":q" | ":quit" | ":exit" => break,
                _ => continue, // :help etc. are silent in pipe mode
            }
        }
        let mut env_borrow = env.borrow_mut();
        let outcome = run_one(trimmed, &mut env_borrow)?;
        drop(env_borrow);
        match outcome {
            Outcome::Ok(v) => println!("{}", format(&v)),
            Outcome::Cancelled => println!("cancelled"),
            Outcome::Err(msg) => eprintln!("error: {msg}"),
        }
    }
    Ok(())
}

fn print_help() {
    let dim = Style::new().fg(Color::DarkGray);
    println!("Built-in verbs:  +  -  *  %");
    println!("Adverbs:         +/  (over)");
    println!("Atoms:           42  3.14  \"a\"  `sym  0N  0n  0W  0w");
    println!("Vectors:         1 2 3   1.5 2.5 3.5   `a`b `c");
    println!("Assignment:      x: 1 2 3");
    println!("Sequence:        a; b; c  (returns last)");
    println!(
        "{}",
        dim.paint("Meta: :help :q  |  Tab completes names  |  Ctrl-C cancels a running op")
    );
}
