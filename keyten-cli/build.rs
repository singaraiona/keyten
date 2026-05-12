//! Build script: captures build date and the current git commit short hash
//! into compile-time environment variables read by the banner module.

use std::process::Command;

fn main() {
    // Build date (UTC, YYYY-MM-DD).
    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=KEYTEN_BUILD_DATE={date}");

    // Short git commit hash.
    let commit = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=KEYTEN_GIT_COMMIT={commit}");

    // Rebuild when HEAD moves (so `git checkout` updates the embedded hash).
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/heads");
}
