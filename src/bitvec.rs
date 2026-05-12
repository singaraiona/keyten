//! Minimal heap-allocated bit vector used for per-chunk null masks.
//!
//! Owns a `Vec<u64>` storage. Not exposed in the public API; lives here so the
//! null-preserving kernel cold path can build and consume a mask.

pub struct BitVec {
    words: Vec<u64>,
    len: usize,
}

impl BitVec {
    pub fn zeros(len: usize) -> Self {
        let n_words = len.div_ceil(64);
        BitVec {
            words: vec![0u64; n_words],
            len,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        debug_assert!(i < self.len);
        let w = i >> 6;
        let b = i & 63;
        self.words[w] |= 1u64 << b;
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        let w = i >> 6;
        let b = i & 63;
        (self.words[w] >> b) & 1 == 1
    }

    pub fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }
}
