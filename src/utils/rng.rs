// Lightweight random number generator for noise sources.
//
// Uses Xoshiro256+ for uniform generation and Box-Muller for Gaussian.
// No external dependencies — self-contained for zero-overhead noise blocks.

/// Fast PRNG based on xoshiro256+ (Blackman & Vigna, 2018).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create from a 64-bit seed (uses SplitMix64 to expand to full state).
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to generate initial state
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    /// Create from ambient entropy (non-reproducible across runs).
    ///
    /// Mixes the wall clock (nanoseconds since the epoch), ASLR (the address
    /// of a promoted static) and a process-global counter. The clock is the
    /// load-bearing source: the static's address is constant within a process
    /// and, without ASLR (notably WASM/Pyodide), constant across runs too —
    /// address+counter alone made every run draw the same noise there.
    pub fn from_entropy() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let addr = &0u8 as *const u8 as u64;
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Spread the counter across the high bits so consecutive calls within
        // one clock tick still seed distinct SplitMix64 streams.
        Self::new(nanos ^ addr.rotate_left(32) ^ count.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Gaussian (normal) with mean 0, standard deviation 1.
    /// Uses Box-Muller transform.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE); // avoid log(0)
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Gaussian with given mean and standard deviation.
    #[inline]
    pub fn normal_scaled(&mut self, mean: f64, std_dev: f64) -> f64 {
        mean + std_dev * self.normal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.uniform(), rng2.uniform());
        }
    }

    #[test]
    fn test_uniform_range() {
        let mut rng = Rng::new(123);
        for _ in 0..10000 {
            let v = rng.uniform();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_normal_statistics() {
        let mut rng = Rng::new(456);
        let n = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let v = rng.normal();
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean={mean}");
        assert!((var - 1.0).abs() < 0.05, "var={var}");
    }
}
