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

// =======================================================================
// Min/max verbs: & |
// =======================================================================

#[test]
fn min_atom_atom() {
    let r = run("3 & 5");
    assert_eq!(unsafe { r.atom::<i64>() }, 3);
}

#[test]
fn max_atom_atom() {
    let r = run("3 | 5");
    assert_eq!(unsafe { r.atom::<i64>() }, 5);
}

#[test]
fn min_vec_vec() {
    let r = run("1 5 3 & 2 4 7");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 4, 3]);
}

#[test]
fn max_vec_atom_broadcast() {
    let r = run("1 5 3 10 | 4");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[4, 5, 4, 10]);
}

#[test]
fn min_promotes_to_f64() {
    let r = run("1.5 2.5 3.5 & 2 2 2");
    assert_eq!(unsafe { r.as_slice::<f64>() }, &[1.5, 2.0, 2.0]);
}

// =======================================================================
// Scan adverb: \
// =======================================================================

#[test]
fn scan_plus_i64_small() {
    let r = run("+\\1 2 3 4 5");
    assert_eq!(r.kind(), Kind::I64);
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 3, 6, 10, 15]);
}

#[test]
fn scan_plus_til() {
    // +\!5 = running sum of [0,1,2,3,4] = [0,1,3,6,10]
    let r = run("+\\!5");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[0, 1, 3, 6, 10]);
}

#[test]
fn scan_plus_f64() {
    let r = run("+\\1.0 2.0 3.0");
    assert_eq!(unsafe { r.as_slice::<f64>() }, &[1.0, 3.0, 6.0]);
}

#[test]
fn scan_then_count() {
    // `+\` produces same-length vector; count should match input length.
    let r = run("#+\\!10");
    assert_eq!(unsafe { r.atom::<i64>() }, 10);
}

// =======================================================================
// _ verb: floor (monadic), drop (dyadic)
// =======================================================================

#[test]
fn floor_atom() {
    let r = run("_ 3.7");
    assert_eq!(r.kind(), Kind::I64);
    assert_eq!(unsafe { r.atom::<i64>() }, 3);
}

#[test]
fn floor_negative() {
    let r = run("_ -1.5");
    assert_eq!(unsafe { r.atom::<i64>() }, -2);
}

#[test]
fn floor_vec() {
    let r = run("_ 1.5 2.9 3.0");
    assert_eq!(r.kind(), Kind::I64);
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3]);
}

#[test]
fn drop_first_n() {
    let r = run("2 _ 1 2 3 4 5");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[3, 4, 5]);
}

#[test]
fn drop_last_n() {
    // Parentheses are needed because the parser eats `-2` as `-(2)` if `-`
    // sits at expression-start and the next thing is a number. K9
    // convention writes `(- 2) _ x` or assigns the negative number first.
    let r = run("(-2) _ 1 2 3 4 5");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3]);
}

#[test]
fn drop_more_than_len() {
    let r = run("10 _ 1 2 3");
    assert_eq!(r.len(), 0);
}

// =======================================================================
// $ verb: string-convert (monadic)
// =======================================================================

fn run_str(src: &str) -> String {
    let r = run(src);
    assert!(r.is_vec(), "expected vector, got atom");
    assert_eq!(r.kind(), Kind::Char);
    let bytes = unsafe { r.as_slice::<u8>() };
    std::str::from_utf8(bytes).unwrap().to_string()
}

#[test]
fn dollar_int_atom() {
    assert_eq!(run_str("$42"), "42");
}

#[test]
fn dollar_negative_int() {
    assert_eq!(run_str("$ -1"), "-1");
}

#[test]
fn dollar_float_atom() {
    assert_eq!(run_str("$3.5"), "3.5");
}

#[test]
fn dollar_bool() {
    // `0=0` is `1b`.
    assert_eq!(run_str("$ 0 = 0"), "1b");
    assert_eq!(run_str("$ 0 = 1"), "0b");
}

// =======================================================================
// ~ monadic (not)
// =======================================================================

#[test]
fn not_bool_atom() {
    // ~1b = 0b (using the equality verb to produce a bool)
    let r = run("~ 0 = 0");
    assert_eq!(r.kind(), Kind::Bool);
    assert_eq!(unsafe { r.atom::<u8>() }, 0);
}

#[test]
fn not_int_atom() {
    let r = run("~5");
    assert_eq!(r.kind(), Kind::Bool);
    assert_eq!(unsafe { r.atom::<u8>() }, 0);
    let r = run("~0");
    assert_eq!(unsafe { r.atom::<u8>() }, 1);
}

#[test]
fn not_bool_vector() {
    let r = run("~ 1 2 3 = 1 5 3");
    let s = unsafe { r.as_slice::<u8>() };
    assert_eq!(s, &[0, 1, 0]);
}

// =======================================================================
// ^ sort, ? unique, monadic % sqrt, | reverse, & where
// =======================================================================

#[test]
fn sort_int_vec() {
    let r = run("^5 2 8 1 9 3");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3, 5, 8, 9]);
}

#[test]
fn sort_float_vec() {
    let r = run("^3.5 1.5 2.5");
    assert_eq!(unsafe { r.as_slice::<f64>() }, &[1.5, 2.5, 3.5]);
}

#[test]
fn unique_int_vec() {
    let r = run("?3 1 2 1 3 4 2");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[3, 1, 2, 4]);
}

#[test]
fn sqrt_int_atom() {
    let r = run("%16");
    assert_eq!(r.kind(), Kind::F64);
    assert_eq!(unsafe { r.atom::<f64>() }, 4.0);
}

#[test]
fn sqrt_float_vec() {
    let r = run("%1.0 4.0 9.0 16.0");
    assert_eq!(unsafe { r.as_slice::<f64>() }, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn reverse_int_vec() {
    let r = run("|1 2 3 4 5");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[5, 4, 3, 2, 1]);
}

#[test]
fn where_on_bool_vec() {
    let r = run("& 1 2 3 4 5 > 2");
    // 1>2 0>2 ... so bools are [0,0,1,1,1]; where → [2,3,4].
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[2, 3, 4]);
}

// =======================================================================
// Each adverb: '
// =======================================================================

#[test]
fn each_negate() {
    let r = run("-'1 2 3");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[-1, -2, -3]);
}

#[test]
fn each_count_atoms_passthrough() {
    // `#'5` on an atom = `#5` = 1.
    let r = run("#'5");
    assert_eq!(unsafe { r.atom::<i64>() }, 1);
}

#[test]
fn each_sqrt_int_vec() {
    let r = run("%'1 4 9 16");
    assert_eq!(unsafe { r.as_slice::<f64>() }, &[1.0, 2.0, 3.0, 4.0]);
}

// =======================================================================
// EachPrior adverb: ':
// =======================================================================

#[test]
fn eachprior_plus_pairs() {
    // +':1 2 3 4  →  [1, 3, 5, 7]  (first elem stays; rest x[i]+x[i-1])
    let r = run("+':1 2 3 4");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 3, 5, 7]);
}

#[test]
fn eachprior_minus_first_differences() {
    // -':1 3 6 10 →  [1, 2, 3, 4]  (first differences)
    let r = run("-':1 3 6 10");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3, 4]);
}

#[test]
fn eachprior_minus_inverts_scan() {
    // -':+\v should recover v (running sum then first-differences).
    let r = run("-':+\\1 2 3 4 5");
    assert_eq!(unsafe { r.as_slice::<i64>() }, &[1, 2, 3, 4, 5]);
}

// =======================================================================
// Dict construction via `!` dyadic
// =======================================================================

#[test]
fn dict_construct() {
    let r = run("1 2 3 ! 10 20 30");
    assert_eq!(r.kind(), Kind::Dict);
    // The kind is the result kind; the storage holds [keys, values] as two
    // RefObj slots. Verified by the dict_keys / dict_values helpers in
    // the unit-test module of kernels/dict.rs.
}

#[test]
fn dict_lookup_length() {
    // `#` on a dict reports the number of key-value pairs (which equals
    // the length of the values vector, which is what our `.len()` reads
    // from offset 8). Dict has len = 2 in storage but is logically "3"
    // for a 3-element dict. For now `#` returns the raw stored len.
    // TODO: make `#dict` return the entry count once dict has a proper
    // length convention. For v1 we just check construction works.
    let r = run("@ 1 2 3 ! 10 20 30");
    assert_eq!(r.kind(), Kind::Sym);
}

#[test]
fn scan_at_threshold_parallel_path() {
    // Force the parallel branch by going past PARALLEL_THRESHOLD (256K).
    // For !N, +\!N[i] = i*(i+1)/2.
    let r = run("+\\!300000");
    let s = unsafe { r.as_slice::<i64>() };
    // Spot-check a few positions instead of materialising the whole 300K
    // expected vector here.
    assert_eq!(s[0], 0);
    assert_eq!(s[1], 1);
    assert_eq!(s[2], 3);
    assert_eq!(s[99], 99 * 100 / 2);
    assert_eq!(s[299_999], 299_999i64 * 300_000 / 2);
}

#[tokio::test(flavor = "current_thread")]
async fn eval_async_runs_under_tokio() {
    let expr = parse("+/ 1 2 3 4 5").unwrap();
    let mut env = Env::new();
    let ctx = Ctx::quiet();
    let r = keyten::eval_async(&expr, &mut env, &ctx).await.unwrap();
    assert_eq!(unsafe { r.atom::<i64>() }, 15);
}
