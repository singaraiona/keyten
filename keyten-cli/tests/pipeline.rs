//! Non-interactive smoke test of the keyten-cli pipeline:
//! parse → eval (under tokio current_thread) → format → assert.

use keyten::{Ctx, Env};
use keyten_cli::format::{format, PrintOpts};
use ratatui::text::Line;

fn plain(lines: &[Line]) -> String {
    let mut out = String::new();
    for (i, l) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        for s in l.iter() {
            out.push_str(&s.content);
        }
    }
    out
}

#[tokio::test(flavor = "current_thread")]
async fn vec_plus_atom_renders_correctly() {
    let expr = keyten::parse("1 2 3 + 10").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    let lines = format(&r, &PrintOpts::default());
    assert_eq!(plain(&lines), "11 12 13");
}

#[tokio::test(flavor = "current_thread")]
async fn null_renders_as_0n() {
    let expr = keyten::parse("1 0N 3 + 10").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    let lines = format(&r, &PrintOpts::default());
    let s = plain(&lines);
    assert!(s.contains("11"));
    assert!(s.contains("0N"));
    assert!(s.contains("13"));
}

#[tokio::test(flavor = "current_thread")]
async fn float_div_renders_as_decimal() {
    let expr = keyten::parse("7 % 2").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    let lines = format(&r, &PrintOpts::default());
    assert_eq!(plain(&lines), "3.5");
}

#[tokio::test(flavor = "current_thread")]
async fn over_sum_renders_as_atom() {
    let expr = keyten::parse("+/ 1 2 3 4 5").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    let lines = format(&r, &PrintOpts::default());
    assert_eq!(plain(&lines), "15");
}

#[tokio::test(flavor = "current_thread")]
async fn variable_binding_then_lookup() {
    let expr = keyten::parse("x: 1 2 3; +/x").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    let lines = format(&r, &PrintOpts::default());
    assert_eq!(plain(&lines), "6");
}
