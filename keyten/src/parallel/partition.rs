//! Range partitioning for data-parallel kernels.

use core::ops::Range;

/// Split `[0..n)` into `k` contiguous, roughly-equal ranges.
///
/// Returns at most `k` ranges (fewer if `k > n`) and exactly one empty
/// result for `n == 0`. Ranges always cover `[0..n)` exactly with no gaps
/// or overlaps — this is what makes them safe to split a `&mut [T]` along.
///
/// When `n` is not a multiple of `k`, the first `n % k` ranges are one
/// element longer than the rest. The split is deterministic so kernel
/// runs are reproducible (helpful for testing and for floating-point
/// reduction order under `Runtime.deterministic`).
pub fn balanced(n: usize, k: usize) -> Vec<Range<usize>> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let base = n / k;
    let extra = n % k;
    let mut out = Vec::with_capacity(k);
    let mut start = 0;
    for i in 0..k {
        let len = base + if i < extra { 1 } else { 0 };
        out.push(start..start + len);
        start += len;
    }
    debug_assert_eq!(start, n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn covers(ranges: &[Range<usize>], n: usize) -> bool {
        if ranges.is_empty() {
            return n == 0;
        }
        if ranges[0].start != 0 || ranges.last().unwrap().end != n {
            return false;
        }
        ranges.windows(2).all(|w| w[0].end == w[1].start)
    }

    #[test]
    fn empty_range() {
        assert_eq!(balanced(0, 4), Vec::<Range<usize>>::new());
    }

    #[test]
    fn single_worker() {
        assert_eq!(balanced(10, 1), vec![0..10]);
    }

    #[test]
    fn even_split() {
        let r = balanced(100, 4);
        assert_eq!(r, vec![0..25, 25..50, 50..75, 75..100]);
        assert!(covers(&r, 100));
    }

    #[test]
    fn uneven_split_distributes_extras_to_front() {
        // 10 / 3 = 3 rem 1 → first range gets 4, rest get 3.
        let r = balanced(10, 3);
        assert_eq!(r, vec![0..4, 4..7, 7..10]);
        assert!(covers(&r, 10));
        // 11 / 3 = 3 rem 2 → first two ranges get 4, last gets 3.
        let r = balanced(11, 3);
        assert_eq!(r, vec![0..4, 4..8, 8..11]);
        assert!(covers(&r, 11));
    }

    #[test]
    fn more_workers_than_items() {
        // Capped at n: 3 items, 8 workers requested → 3 ranges of 1.
        let r = balanced(3, 8);
        assert_eq!(r.len(), 3);
        assert_eq!(r, vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn zero_workers_is_empty() {
        assert_eq!(balanced(100, 0), Vec::<Range<usize>>::new());
    }
}
