//! Temporal arithmetic for the K9 date / time / datetime types.
//!
//! All conversions are anchored at the **K9 epoch: 2001-01-01 00:00:00 UTC**.
//!
//! - `Date` (kind `d`): i32 days since the epoch
//! - `TimeS` (`s`):    i32 seconds since midnight (0..86_400)
//! - `TimeMs` (`t`):   i32 ms since midnight   (0..86_400_000)
//! - `TimeUs` (`u`):   i64 µs since midnight   (0..86_400_000_000)
//! - `TimeNs` (`v`):   i64 ns since midnight   (0..86_400_000_000_000)
//! - `DtS` (`S`):      i64 seconds since the epoch
//! - `DtMs` (`T`):     i64 ms      since the epoch
//! - `DtUs` (`U`):     i64 µs      since the epoch
//! - `DtNs` (`V`):     i64 ns      since the epoch
//!
//! Calendar is the proleptic Gregorian: dates before the epoch use negative
//! day counts (`2000-12-31` is `-1`). Leap years follow the standard
//! Gregorian rule (divisible by 4, not by 100, except divisible by 400).

use crate::kind::K9_EPOCH_YEAR;

/// Seconds per day.
pub const SECS_PER_DAY: i64 = 86_400;
/// Milliseconds per day.
pub const MS_PER_DAY: i64 = 86_400_000;
/// Microseconds per day.
pub const US_PER_DAY: i64 = 86_400_000_000;
/// Nanoseconds per day.
pub const NS_PER_DAY: i64 = 86_400_000_000_000;

#[inline]
fn is_leap_year(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days from civil date `(y, m, d)` to **Unix epoch (1970-01-01)**.
/// Negative for dates before 1970. Algorithm: Howard Hinnant's
/// "date" library, civil_from_days inverse. Well-tested across the full
/// i32 year range.
///
/// Reference: <https://howardhinnant.github.io/date_algorithms.html>
fn days_from_civil_unix(y: i32, m: i32, d: i32) -> i64 {
    let y = (y - if m <= 2 { 1 } else { 0 }) as i64;
    let m = m as i64;
    let d = d as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Inverse: civil_from_days(z) returns (y, m, d) for `z = days from Unix
/// epoch`. From the same Hinnant reference.
fn civil_from_unix_days(z: i64) -> (i32, i32, i32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = y + if m <= 2 { 1 } else { 0 };
    (y as i32, m as i32, d as i32)
}

/// Days from 1970-01-01 to 2001-01-01. Computed once at compile time so
/// `days_from_ymd` / `ymd_from_days` are pure arithmetic.
const K9_EPOCH_DAYS_FROM_UNIX: i64 = {
    // Inlined Hinnant for the K9 epoch. 31 years from 1970 to 2001, with
    // leap days in 1972/76/80/84/88/92/96/2000 — 8 leap days.
    let years_since_unix = (K9_EPOCH_YEAR - 1970) as i64;
    years_since_unix * 365 + 8
    // + 0 (epoch is January 1, day-of-year 0)
};

/// Convert (year, month, day) → days since **2001-01-01** (the K9 epoch).
///
/// `month` should be 1..=12 and `day` 1..=N (where N is the days in that
/// month). Other inputs are valid but may name "non-canonical" dates that
/// round-trip to the same canonical (y, m, d) via month/year carrying.
pub fn days_from_ymd(year: i32, month: i32, day: i32) -> i32 {
    let unix_days = days_from_civil_unix(year, month, day);
    (unix_days - K9_EPOCH_DAYS_FROM_UNIX) as i32
}

/// Convert days since 2001-01-01 → (year, month, day).
pub fn ymd_from_days(days: i32) -> (i32, i32, i32) {
    let unix_days = days as i64 + K9_EPOCH_DAYS_FROM_UNIX;
    civil_from_unix_days(unix_days)
}

/// Compose a time-of-day from (h, m, s) at the requested granularity.
/// Returns the unit count since midnight (no normalisation — caller must
/// ensure `0 ≤ h < 24`, `0 ≤ m < 60`, `0 ≤ s < 60` to stay within a day).
#[inline]
pub fn time_s_from_hms(h: i32, m: i32, s: i32) -> i32 {
    h * 3600 + m * 60 + s
}

#[inline]
pub fn time_ms_from_hms_ms(h: i32, m: i32, s: i32, ms: i32) -> i32 {
    time_s_from_hms(h, m, s) * 1000 + ms
}

#[inline]
pub fn time_us_from_hms_us(h: i32, m: i32, s: i32, us: i64) -> i64 {
    (time_s_from_hms(h, m, s) as i64) * 1_000_000 + us
}

#[inline]
pub fn time_ns_from_hms_ns(h: i32, m: i32, s: i32, ns: i64) -> i64 {
    (time_s_from_hms(h, m, s) as i64) * 1_000_000_000 + ns
}

/// Decompose a time-of-day into (h, m, s, fractional_units).
/// `unit_per_second` is 1, 1_000, 1_000_000, or 1_000_000_000.
pub fn hms_from_time(units: i64, unit_per_second: i64) -> (i32, i32, i32, i64) {
    let total_secs = units.div_euclid(unit_per_second);
    let frac = units.rem_euclid(unit_per_second);
    let h = total_secs.div_euclid(3600) as i32;
    let rem_secs = total_secs.rem_euclid(3600);
    let m = (rem_secs / 60) as i32;
    let s = (rem_secs % 60) as i32;
    (h, m, s, frac)
}

/// Compose a datetime at second granularity from a date (days since
/// 2001-01-01) and a time (seconds since midnight).
#[inline]
pub fn dt_s(date_days: i32, time_secs: i32) -> i64 {
    (date_days as i64) * SECS_PER_DAY + time_secs as i64
}

#[inline]
pub fn dt_ms(date_days: i32, time_ms: i32) -> i64 {
    (date_days as i64) * MS_PER_DAY + time_ms as i64
}

#[inline]
pub fn dt_us(date_days: i32, time_us: i64) -> i64 {
    (date_days as i64) * US_PER_DAY + time_us
}

#[inline]
pub fn dt_ns(date_days: i32, time_ns: i64) -> i64 {
    (date_days as i64) * NS_PER_DAY + time_ns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_day_zero() {
        assert_eq!(days_from_ymd(2001, 1, 1), 0);
        assert_eq!(ymd_from_days(0), (2001, 1, 1));
    }

    #[test]
    fn day_before_epoch_is_minus_one() {
        assert_eq!(days_from_ymd(2000, 12, 31), -1);
        assert_eq!(ymd_from_days(-1), (2000, 12, 31));
    }

    #[test]
    fn known_dates_round_trip() {
        // 2020-06-14: 19 full years + a partial year.
        let d1 = days_from_ymd(2020, 6, 14);
        assert_eq!(ymd_from_days(d1), (2020, 6, 14));

        // 1970-01-01 (Unix epoch): well before K9 epoch.
        let unix_epoch = days_from_ymd(1970, 1, 1);
        assert_eq!(ymd_from_days(unix_epoch), (1970, 1, 1));
        // Unix epoch is 31 years + 8 leap days before K9 epoch.
        // 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000 = 8 leap years.
        // 31*365 + 8 = 11315 + 8 = 11323 days.
        assert_eq!(unix_epoch, -11323);

        // 2001-12-31: one short year after epoch.
        let end_2001 = days_from_ymd(2001, 12, 31);
        assert_eq!(end_2001, 364);
        assert_eq!(ymd_from_days(364), (2001, 12, 31));
    }

    #[test]
    fn leap_year_handling() {
        // 2024 is a leap year (divisible by 4, not by 100).
        assert!(is_leap_year(2024));
        // 2100 is NOT a leap year (divisible by 100, not by 400).
        assert!(!is_leap_year(2100));
        // 2000 IS a leap year (divisible by 400).
        assert!(is_leap_year(2000));

        // 2024-02-29 → 2024-03-01 is consecutive.
        let feb29 = days_from_ymd(2024, 2, 29);
        let mar01 = days_from_ymd(2024, 3, 1);
        assert_eq!(mar01 - feb29, 1);
        assert_eq!(ymd_from_days(feb29), (2024, 2, 29));
    }

    #[test]
    fn time_of_day_round_trips() {
        // 12:34:56 in seconds.
        let t = time_s_from_hms(12, 34, 56);
        assert_eq!(t, 12 * 3600 + 34 * 60 + 56);
        let (h, m, s, _) = hms_from_time(t as i64, 1);
        assert_eq!((h, m, s), (12, 34, 56));

        // 12:34:56.789 in milliseconds.
        let t = time_ms_from_hms_ms(12, 34, 56, 789);
        let (h, m, s, ms) = hms_from_time(t as i64, 1_000);
        assert_eq!((h, m, s, ms), (12, 34, 56, 789));

        // 23:59:59.999999999 in nanoseconds — edge of a day.
        let t = time_ns_from_hms_ns(23, 59, 59, 999_999_999);
        let (h, m, s, ns) = hms_from_time(t, 1_000_000_000);
        assert_eq!((h, m, s, ns), (23, 59, 59, 999_999_999));
        assert!(t < NS_PER_DAY);
    }

    #[test]
    fn datetime_composition() {
        // K9 epoch midnight = DtNs 0.
        assert_eq!(dt_ns(0, 0), 0);
        assert_eq!(dt_s(0, 0), 0);

        // 2001-01-01T12:00:00 in DtS.
        let noon = dt_s(0, 12 * 3600);
        assert_eq!(noon, 43_200);

        // 2001-01-02T00:00:00.000 in DtMs.
        let day_two_ms = dt_ms(1, 0);
        assert_eq!(day_two_ms, MS_PER_DAY);

        // 2024-02-29T23:59:59.999_999_999 in DtNs — non-overflow check.
        let d = days_from_ymd(2024, 2, 29);
        let t = time_ns_from_hms_ns(23, 59, 59, 999_999_999);
        let dt = dt_ns(d, t);
        // Days from K9 epoch to 2024-02-29 fit in i32; result fits in i64.
        // i64::MAX / NS_PER_DAY ≈ 106751 days ≈ 292 years from epoch, so we
        // can represent 2001–2293 in DtNs without overflow. 2024 is well
        // inside that range.
        assert!(dt > 0);
        assert!(dt < i64::MAX);
    }
}
