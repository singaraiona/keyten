//! Lightweight platform probe: CPU model, total RAM, logical core count.
//!
//! No external dependencies. Falls back to "unknown" / `0` where the platform
//! doesn't expose the info via a cheap interface.

pub struct SysInfo {
    pub cpu_model: String,
    pub mem_gib: f64,
    pub cores: usize,
    pub os_arch: String,
}

impl SysInfo {
    pub fn probe() -> Self {
        SysInfo {
            cpu_model: probe_cpu().unwrap_or_else(|| "unknown".to_string()),
            mem_gib: probe_mem_gib().unwrap_or(0.0),
            cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            os_arch: format!(
                "{}/{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        }
    }
}

#[cfg(target_os = "linux")]
fn probe_cpu() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some(idx) = rest.find(':') {
                return Some(rest[idx + 1..].trim().to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn probe_cpu() -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn probe_cpu() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn probe_mem_gib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024.0 / 1024.0);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn probe_mem_gib() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let bytes: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(bytes / 1024.0 / 1024.0 / 1024.0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn probe_mem_gib() -> Option<f64> {
    None
}
