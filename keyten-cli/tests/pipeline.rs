//! Non-interactive smoke test: parse + eval + format from the lib API.

use keyten::{Ctx, Env};
use keyten_cli::format::{format, format_with_width};

#[test]
fn vec_plus_atom() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("1 2 3 + 10").unwrap(), &mut env, &ctx).unwrap();
    assert_eq!(format(&r), "11 12 13");
}

#[test]
fn over_sum() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("+/ 1 2 3 4 5").unwrap(), &mut env, &ctx).unwrap();
    assert_eq!(format(&r), "15");
}

#[test]
fn div_promotes_to_float() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("7 % 2").unwrap(), &mut env, &ctx).unwrap();
    assert_eq!(format(&r), "3.5");
}

#[test]
fn null_renders_with_ansi() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("1 0N 3 + 10").unwrap(), &mut env, &ctx).unwrap();
    let s = format(&r);
    assert!(s.contains("11"));
    assert!(s.contains("0N"));
    assert!(s.contains("13"));
}

#[test]
fn assign_then_use() {
    let expr = keyten::parse("x: 1 2 3; +/x").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&expr, &mut env, &ctx).unwrap();
    assert_eq!(format(&r), "6");
}

#[test]
fn long_vector_truncated_with_ellipsis() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("!1000").unwrap(), &mut env, &ctx).unwrap();
    let s = format_with_width(&r, 40);
    assert!(s.ends_with(".."), "expected `..` suffix in {s:?}");
    assert!(s.len() <= 40, "output exceeded width: {} chars", s.len());
    // Should still start with the first few elements.
    assert!(s.starts_with("0 1 2"));
}

#[test]
fn short_vector_not_truncated() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("!10").unwrap(), &mut env, &ctx).unwrap();
    let s = format_with_width(&r, 80);
    assert_eq!(s, "0 1 2 3 4 5 6 7 8 9");
}

#[test]
fn atom_never_truncated() {
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&keyten::parse("+/!1000").unwrap(), &mut env, &ctx).unwrap();
    let s = format(&r);
    assert_eq!(s, "499500");
}

#[tokio::test(flavor = "current_thread")]
async fn eval_async_under_tokio() {
    let expr = keyten::parse("+/ 1 2 3 4 5").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    assert_eq!(format(&r), "15");
}
