//! Task-local RNG compatible with Lean 4.34's StdGen and randNat.
//! The recurrence is L'Ecuyer's combined generator; matching the range mapping
//! preserves historical seeded experiments independently of worker scheduling.
#[derive(Clone)]
pub struct Random {
    s1: i64,
    s2: i64,
}
impl Random {
    pub fn new(seed: u64) -> Self {
        Self {
            s1: (seed % 2147483562 + 1) as i64,
            s2: (seed / 2147483562 % 2147483398 + 1) as i64,
        }
    }
    fn next(&mut self) -> u128 {
        let k = self.s1 / 53668;
        self.s1 = 40014 * (self.s1 - k * 53668) - k * 12211;
        if self.s1 < 0 {
            self.s1 += 2147483563;
        }
        let k = self.s2 / 52774;
        self.s2 = 40692 * (self.s2 - k * 52774) - k * 3791;
        if self.s2 < 0 {
            self.s2 += 2147483399;
        }
        let z = self.s1 - self.s2;
        if z < 1 {
            (z + 2147483562) as u128
        } else {
            (z % 2147483562) as u128
        }
    }
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        let k = hi as u128 - lo as u128 + 1;
        let mut r = k * 1000;
        let mut v = 0u128;
        while r != 0 {
            v = v * 2147483562 + self.next().saturating_sub(1);
            r = (r / 2147483562).saturating_sub(1);
        }
        lo + (v % k) as usize
    }
}
