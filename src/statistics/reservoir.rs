//! Bounded-memory reservoir sampling (Algorithm R, Vitter 1985).
//!
//! Maintains a fixed-size sample from a stream of unknown length.
//! After processing N items, each has equal probability (k/N) of being in the
//! reservoir, where k is the reservoir capacity.
//!
//! Tracks exact min/max separately from the sample (not approximate like
//! the Python MBO-LOB-analyzer's StreamingDistribution — see audit Issue 3).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

fn reconstruct_rng(seed: u64, total_seen: u64) -> StdRng {
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..total_seen {
        let _ = rng.gen::<u64>();
    }
    rng
}

/// Reservoir sampler with deterministic seeding and exact min/max.
///
/// # Performance Note
///
/// Deserialization replays the RNG from seed to restore deterministic state,
/// which is O(total_seen). For streams with millions+ of values, deserialization
/// may take seconds. Consider small reservoir capacities for checkpoint-heavy workflows.
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    capacity: usize,
    reservoir: Vec<f64>,
    total_seen: u64,
    true_min: f64,
    true_max: f64,
    rng: StdRng,
    seed: u64,
}

impl Serialize for ReservoirSampler {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("ReservoirSampler", 5)?;
        s.serialize_field("capacity", &self.capacity)?;
        s.serialize_field("reservoir", &self.reservoir)?;
        s.serialize_field("total_seen", &self.total_seen)?;
        s.serialize_field("true_min", &self.true_min)?;
        s.serialize_field("true_max", &self.true_max)?;
        s.serialize_field("seed", &self.seed)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for ReservoirSampler {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            capacity: usize,
            reservoir: Vec<f64>,
            total_seen: u64,
            true_min: f64,
            true_max: f64,
            seed: u64,
        }
        let h = Helper::deserialize(deserializer)?;
        Ok(Self {
            rng: reconstruct_rng(h.seed, h.total_seen),
            capacity: h.capacity,
            reservoir: h.reservoir,
            total_seen: h.total_seen,
            true_min: h.true_min,
            true_max: h.true_max,
            seed: h.seed,
        })
    }
}

impl ReservoirSampler {
    pub fn new(capacity: usize) -> Self {
        Self::with_seed(capacity, 42)
    }

    pub fn with_seed(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            reservoir: Vec::with_capacity(capacity),
            total_seen: 0,
            true_min: f64::INFINITY,
            true_max: f64::NEG_INFINITY,
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Add a single value to the reservoir.
    #[inline]
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.true_min = self.true_min.min(value);
        self.true_max = self.true_max.max(value);

        if self.reservoir.len() < self.capacity {
            self.reservoir.push(value);
        } else {
            let j = self.rng.gen_range(0..=self.total_seen) as usize;
            if j < self.capacity {
                self.reservoir[j] = value;
            }
        }
        self.total_seen += 1;
    }

    /// Add a batch of values.
    pub fn add_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    pub fn sample_size(&self) -> usize {
        self.reservoir.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Exact minimum across all values seen (not just the reservoir).
    pub fn true_min(&self) -> f64 {
        if self.total_seen == 0 {
            f64::NAN
        } else {
            self.true_min
        }
    }

    /// Exact maximum across all values seen (not just the reservoir).
    pub fn true_max(&self) -> f64 {
        if self.total_seen == 0 {
            f64::NAN
        } else {
            self.true_max
        }
    }

    /// Get a sorted copy of the reservoir for percentile computation.
    pub fn sorted_sample(&self) -> Vec<f64> {
        let mut sorted = self.reservoir.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Compute a percentile from the reservoir sample.
    ///
    /// Uses linear interpolation. `p` must be in [0, 100].
    pub fn percentile(&self, p: f64) -> f64 {
        if self.reservoir.is_empty() {
            return f64::NAN;
        }
        let sorted = self.sorted_sample();
        let idx = (p / 100.0) * (sorted.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi || hi >= sorted.len() {
            sorted[lo.min(sorted.len() - 1)]
        } else {
            let frac = idx - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }

    /// Compute multiple percentiles efficiently (single sort).
    pub fn percentiles(&self, ps: &[f64]) -> Vec<f64> {
        if self.reservoir.is_empty() {
            return ps.iter().map(|_| f64::NAN).collect();
        }
        let sorted = self.sorted_sample();
        ps.iter()
            .map(|&p| {
                let idx = (p / 100.0) * (sorted.len() - 1) as f64;
                let lo = idx.floor() as usize;
                let hi = idx.ceil() as usize;
                if lo == hi || hi >= sorted.len() {
                    sorted[lo.min(sorted.len() - 1)]
                } else {
                    let frac = idx - lo as f64;
                    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
                }
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.reservoir.clear();
        self.total_seen = 0;
        self.true_min = f64::INFINITY;
        self.true_max = f64::NEG_INFINITY;
        self.rng = StdRng::seed_from_u64(self.seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_underfilled_reservoir() {
        let mut rs = ReservoirSampler::new(100);
        for i in 0..50 {
            rs.add(i as f64);
        }
        assert_eq!(rs.sample_size(), 50);
        assert_eq!(rs.total_seen(), 50);
        assert_eq!(rs.true_min(), 0.0);
        assert_eq!(rs.true_max(), 49.0);
    }

    #[test]
    fn test_full_reservoir() {
        let mut rs = ReservoirSampler::new(100);
        for i in 0..10000 {
            rs.add(i as f64);
        }
        assert_eq!(rs.sample_size(), 100);
        assert_eq!(rs.total_seen(), 10000);
        assert_eq!(rs.true_min(), 0.0);
        assert_eq!(rs.true_max(), 9999.0);
    }

    #[test]
    fn test_exact_min_max_not_approximate() {
        let mut rs = ReservoirSampler::new(10);
        rs.add(-999.0);
        for i in 0..100000 {
            rs.add(i as f64);
        }
        rs.add(999999.0);

        assert_eq!(
            rs.true_min(),
            -999.0,
            "min must be exact, not from reservoir"
        );
        assert_eq!(
            rs.true_max(),
            999999.0,
            "max must be exact, not from reservoir"
        );
    }

    #[test]
    fn test_nan_skipped() {
        let mut rs = ReservoirSampler::new(10);
        rs.add(1.0);
        rs.add(f64::NAN);
        rs.add(2.0);
        assert_eq!(rs.total_seen(), 2);
        assert_eq!(rs.sample_size(), 2);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mut a = ReservoirSampler::with_seed(10, 123);
        let mut b = ReservoirSampler::with_seed(10, 123);
        for i in 0..1000 {
            a.add(i as f64);
            b.add(i as f64);
        }
        assert_eq!(a.sorted_sample(), b.sorted_sample());
    }

    #[test]
    fn test_percentile_basic() {
        let mut rs = ReservoirSampler::new(1000);
        for i in 0..=100 {
            rs.add(i as f64);
        }
        assert!((rs.percentile(50.0) - 50.0).abs() < 1.0);
        assert!((rs.percentile(0.0) - 0.0).abs() < 0.1);
        assert!((rs.percentile(100.0) - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_reset() {
        let mut rs = ReservoirSampler::new(10);
        rs.add(42.0);
        rs.reset();
        assert_eq!(rs.total_seen(), 0);
        assert_eq!(rs.sample_size(), 0);
        assert!(rs.true_min().is_nan());
    }
}
