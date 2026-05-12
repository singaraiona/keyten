//! `Validator` — Enter submits only when brackets are balanced; otherwise the
//! line is considered incomplete and reedline inserts a newline.

use reedline::{ValidationResult, Validator};

pub struct KValidator;

impl Validator for KValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        if brackets_balanced(line) {
            ValidationResult::Complete
        } else {
            ValidationResult::Incomplete
        }
    }
}

pub fn brackets_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev = '\0';
    for c in s.chars() {
        if in_str {
            if c == '"' && prev != '\\' {
                in_str = false;
            }
        } else {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                '"' => in_str = true,
                _ => {}
            }
        }
        prev = c;
        if depth < 0 {
            return false;
        }
    }
    depth == 0 && !in_str
}
