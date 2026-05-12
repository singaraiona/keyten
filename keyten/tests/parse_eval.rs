//! End-to-end parse + eval tests for the v1 surface.

use keyten::{eval, parse, Ctx, Env, Kind};

fn run(src: &str) -> keyten::RefObj {
    let expr = parse(src).expect("parse error");
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    eval(&expr, &mut env, &ctx).expect("eval error")
}

#[test]
fn atom_int() {
    let r = run("42");
    assert_eq!(r.kind(), Kind::I64);
    assert!(r.is_atom());
    let v = unsafe { r.atom::<i64>() };
    assert_eq!(v, 42);
}

#[test]
fn atom_float() {
    let r = run("3.5");
    assert_eq!(r.kind(), Kind::F64);
    let v = unsafe { r.atom::<f64>() };
    assert_eq!(v, 3.5);
}

#[test]
fn vec_int() {
    let r = run("1 2 3");
    assert_eq!(r.kind(), Kind::I64);
    assert!(r.is_vec());
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[1, 2, 3]);
}

#[test]
fn vec_float() {
    let r = run("1.5 2.5 3.5");
    assert_eq!(r.kind(), Kind::F64);
    let s = unsafe { r.as_slice::<f64>() };
    assert_eq!(s, &[1.5, 2.5, 3.5]);
}

#[test]
fn dyadic_plus_vec_atom() {
    let r = run("1 2 3 + 10");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[11, 12, 13]);
}

#[test]
fn dyadic_minus() {
    let r = run("10 - 3");
    assert_eq!(unsafe { r.atom::<i64>() }, 7);
}

#[test]
fn dyadic_times() {
    let r = run("2 3 4 * 10");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[20, 30, 40]);
}

#[test]
fn dyadic_div_promotes_to_f64() {
    let r = run("7 % 2");
    assert_eq!(r.kind(), Kind::F64);
    assert_eq!(unsafe { r.atom::<f64>() }, 3.5);
}

#[test]
fn right_associative_arithmetic() {
    // K convention: 2 * 3 + 4 == 2 * (3 + 4) == 14
    let r = run("2 * 3 + 4");
    assert_eq!(unsafe { r.atom::<i64>() }, 14);
}

#[test]
fn parens_grouping() {
    // (2 * 3) + 4 = 10
    let r = run("(2 * 3) + 4");
    assert_eq!(unsafe { r.atom::<i64>() }, 10);
}

#[test]
fn variable_assign_and_use() {
    let expr = parse("x: 1 2 3; x + 10").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = eval(&expr, &mut env, &ctx).unwrap();
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[11, 12, 13]);
}

#[test]
fn over_adverb_sum() {
    let r = run("+/ 1 2 3 4 5");
    assert_eq!(unsafe { r.atom::<i64>() }, 15);
}

#[test]
fn over_adverb_float_sum() {
    let r = run("+/ 1.5 2.5 3.0");
    assert!((unsafe { r.atom::<f64>() } - 7.0).abs() < 1e-12);
}

#[test]
fn monadic_negate() {
    let r = run("- 5");
    assert_eq!(unsafe { r.atom::<i64>() }, -5);
}

#[test]
fn null_literal_preserves_through_plus() {
    let r = run("1 0N 3 + 10");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s[0], 11);
    assert_eq!(s[1], keyten::nulls::NULL_I64);
    assert_eq!(s[2], 13);
}

#[test]
fn til_generates_range() {
    let r = run("!5");
    assert!(r.is_vec());
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[0, 1, 2, 3, 4]);
}

#[test]
fn til_zero_is_empty() {
    let r = run("!0");
    assert!(r.is_vec());
    assert_eq!(r.len(), 0);
}

#[test]
fn sum_til_n() {
    let r = run("+/!100");
    assert_eq!(unsafe { r.atom::<i64>() }, 4950);
}

#[test]
fn til_bound_to_var_then_summed() {
    let expr = keyten::parse("x: !10; +/x").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval(&expr, &mut env, &ctx).unwrap();
    assert_eq!(unsafe { r.atom::<i64>() }, 45);
}

// =======================================================================
// Phase 2 verbs: `@` (type), `#` (count), `,` (enlist)
// =======================================================================

#[test]
fn type_at_returns_letter_atom() {
    use keyten::sym::intern;
    let r = run("@42");
    // I64 atom → `j (the K9 default int letter)
    assert_eq!(r.kind(), Kind::Sym);
    let s = unsafe { r.atom::<keyten::Sym>() };
    assert_eq!(s, intern("j").unwrap());
}

#[test]
fn type_at_vector_returns_uppercase() {
    use keyten::sym::intern;
    let r = run("@1 2 3");
    let s = unsafe { r.atom::<keyten::Sym>() };
    assert_eq!(s, intern("J").unwrap());
}

#[test]
fn type_at_float_atom_and_vec() {
    use keyten::sym::intern;
    let r = run("@3.14");
    let s = unsafe { r.atom::<keyten::Sym>() };
    assert_eq!(s, intern("f").unwrap());
    let r = run("@1.0 2.0");
    let s = unsafe { r.atom::<keyten::Sym>() };
    assert_eq!(s, intern("F").unwrap());
}

#[test]
fn count_hash_atom_is_one() {
    let r = run("#42");
    assert_eq!(unsafe { r.atom::<i64>() }, 1);
}

#[test]
fn count_hash_vector() {
    let r = run("#1 2 3 4 5");
    assert_eq!(unsafe { r.atom::<i64>() }, 5);
}

#[test]
fn count_hash_til_n() {
    let r = run("#!1000");
    assert_eq!(unsafe { r.atom::<i64>() }, 1000);
}

#[test]
fn enlist_comma_atom_makes_unit_vector() {
    let r = run(",42");
    assert!(r.is_vec());
    assert_eq!(r.kind(), Kind::I64);
    assert_eq!(r.len(), 1);
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[42]);
}

#[test]
fn enlist_comma_float_atom() {
    let r = run(",3.5");
    assert!(r.is_vec());
    assert_eq!(r.kind(), Kind::F64);
    assert_eq!(r.len(), 1);
    let s = unsafe { r.as_slice::<f64>() };
    assert_eq!(s, &[3.5]);
}

#[test]
fn comma_concat_two_int_vectors() {
    let r = run("1 2 3 , 10 20");
    assert!(r.is_vec());
    assert_eq!(r.kind(), Kind::I64);
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[1, 2, 3, 10, 20]);
}

#[test]
fn comma_concat_atom_and_vector() {
    let r = run("99 , 1 2 3");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[99, 1, 2, 3]);
}

#[test]
fn hash_take_exact_length() {
    let r = run("3 # 10 20 30");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[10, 20, 30]);
}

#[test]
fn hash_take_truncates() {
    let r = run("3 # 10 20 30 40 50");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[10, 20, 30]);
}

#[test]
fn hash_take_cycles() {
    let r = run("7 # 1 2 3");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[1, 2, 3, 1, 2, 3, 1]);
}

#[test]
fn hash_take_from_til() {
    let r = run("3 # !100");
    let s = unsafe { r.as_slice::<i64>() };
    assert_eq!(s, &[0, 1, 2]);
}

// =======================================================================
// Comparison verbs: = < > ~
// =======================================================================

#[test]
fn eq_atom_atom_true() {
    let r = run("5 = 5");
    assert_eq!(r.kind(), Kind::Bool);
    assert_eq!(unsafe { r.atom::<u8>() }, 1);
}

#[test]
fn eq_atom_atom_false() {
    let r = run("5 = 6");
    assert_eq!(unsafe { r.atom::<u8>() }, 0);
}

#[test]
fn lt_vec_vec() {
    let r = run("1 5 3 < 2 5 1");
    assert_eq!(r.kind(), Kind::Bool);
    let s = unsafe { r.as_slice::<u8>() };
    assert_eq!(s, &[1, 0, 0]);
}

#[test]
fn gt_vec_atom_broadcasts() {
    let r = run("1 5 3 10 > 3");
    let s = unsafe { r.as_slice::<u8>() };
    assert_eq!(s, &[0, 1, 0, 1]);
}

#[test]
fn eq_promotes_int_and_float() {
    let r = run("1 2 3 = 1.0 2.0 4.0");
    let s = unsafe { r.as_slice::<u8>() };
    assert_eq!(s, &[1, 1, 0]);
}

#[test]
fn match_atoms_equal() {
    let r = run("42 ~ 42");
    assert_eq!(r.kind(), Kind::Bool);
    assert_eq!(unsafe { r.atom::<u8>() }, 1);
}

#[test]
fn match_atom_and_vec_differ() {
    let r = run("42 ~ 42 42 42");
    assert_eq!(unsafe { r.atom::<u8>() }, 0);
}

#[test]
fn match_vectors_equal() {
    let r = run("1 2 3 ~ 1 2 3");
    assert_eq!(unsafe { r.atom::<u8>() }, 1);
}

#[test]
fn match_different_kind() {
    let r = run("1 ~ 1.0");
    assert_eq!(unsafe { r.atom::<u8>() }, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn eval_async_runs_under_tokio() {
    let expr = parse("+/ 1 2 3 4 5").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    assert_eq!(unsafe { r.atom::<i64>() }, 15);
}
