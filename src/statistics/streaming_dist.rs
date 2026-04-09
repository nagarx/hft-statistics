//! Streaming distribution: combines Welford + Reservoir + exact min/max.
//!
//! This is the Rust equivalent of the Python MBO-LOB-analyzer's
//! `StreamingDistribution` but with the audit Issue 3 fix: min/max are exact
//! (tracked by the reservoir sampler), not approximate from the sample.

use serde::{Deserialize, Serialize};

use super::reservoir::ReservoirSampler;
use super::welford::WelfordAccumulator;

/// Combined streaming distribution tracker.
///
/// Provides exact mean/variance (via Welford) plus approximate percentiles,
/// skewness, and kurtosis (via reservoir sampling). Min/max are exact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDistribution {
    welford: WelfordAccumulator,
    reservoir: ReservoirSampler,
}

impl StreamingDistribution {
    pub fn new(reservoir_capacity: usize) -> Self {
        Self {
            welford: WelfordAccumulator::new(),
            reservoir: ReservoirSampler::new(reservoir_capacity),
        }
    }

    pub fn with_seed(reservoir_capacity: usize, seed: u64) -> Self {
        Self {
            welford: WelfordAccumulator::new(),
            reservoir: ReservoirSampler::with_seed(reservoir_capacity, seed),
        }
    }

    /// Add a single value.
    #[inline]
    pub fn add(&mut self, value: f64) {
        self.welford.update(value);
        self.reservoir.add(value);
    }

    /// Add a batch of values.
    pub fn add_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    pub fn count(&self) -> u64 {
        self.welford.count()
    }

    pub fn mean(&self) -> f64 {
        self.welford.mean()
    }

    pub fn variance(&self) -> f64 {
        self.welford.variance()
    }

    pub fn std(&self) -> f64 {
        self.welford.std()
    }

    pub fn min(&self) -> f64 {
        self.reservoir.true_min()
    }

    pub fn max(&self) -> f64 {
        self.reservoir.true_max()
    }

    pub fn percentile(&self, p: f64) -> f64 {
        self.reservoir.percentile(p)
    }

    pub fn percentiles(&self, ps: &[f64]) -> Vec<f64> {
        self.reservoir.percentiles(ps)
    }

    /// Get the sorted reservoir sample for direct access (Hill estimator, CVaR).
    pub fn sorted_sample(&self) -> Vec<f64> {
        self.reservoir.sorted_sample()
    }

    /// Number of values in the reservoir sample.
    pub fn sample_size(&self) -> usize {
        self.reservoir.sample_size()
    }

    /// Compute skewness from the reservoir sample.
    pub fn skewness(&self) -> f64 {
        let sorted = self.reservoir.sorted_sample();
        if sorted.len() < 3 {
            return f64::NAN;
        }
        let n = sorted.len() as f64;
        let mean = sorted.iter().sum::<f64>() / n;
        let m2: f64 = sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = sorted.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
        let std = m2.sqrt();
        if std < 1e-15 {
            return 0.0;
        }
        m3 / std.powi(3)
    }

    /// Compute excess kurtosis from the reservoir sample.
    pub fn kurtosis(&self) -> f64 {
        let sorted = self.reservoir.sorted_sample();
        if sorted.len() < 4 {
            return f64::NAN;
        }
        let n = sorted.len() as f64;
        let mean = sorted.iter().sum::<f64>() / n;
        let m2: f64 = sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4: f64 = sorted.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
        let var = m2;
        if var < 1e-15 {
            return 0.0;
        }
        (m4 / var.powi(2)) - 3.0
    }

    /// Produce a summary as a JSON-serializable map.
    pub fn summary(&self) -> serde_json::Value {
        let ps = vec![1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
        let pvals = self.percentiles(&ps);

        serde_json::json!({
            "count": self.count(),
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            "percentiles": {
                "p1": pvals[0],
                "p5": pvals[1],
                "p10": pvals[2],
                "p25": pvals[3],
                "p50": pvals[4],
                "p75": pvals[5],
                "p90": pvals[6],
                "p95": pvals[7],
                "p99": pvals[8],
            }
        })
    }

    pub fn reset(&mut self) {
        self.welford.reset();
        self.reservoir.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_count() {
        let mut dist = StreamingDistribution::new(100);
        for i in 0..50 {
            dist.add(i as f64);
        }
        assert_eq!(dist.count(), 50);
    }

    #[test]
    fn test_exact_mean_from_welford() {
        let mut dist = StreamingDistribution::new(100);
        for i in 1..=100 {
            dist.add(i as f64);
        }
        assert!((dist.mean() - 50.5).abs() < 1e-10);
    }

    #[test]
    fn test_exact_min_max() {
        let mut dist = StreamingDistribution::new(10);
        dist.add(-100.0);
        for i in 0..100000 {
            dist.add(i as f64);
        }
        dist.add(500000.0);

        assert_eq!(dist.min(), -100.0);
        assert_eq!(dist.max(), 500000.0);
    }

    #[test]
    fn test_summary_has_expected_keys() {
        let mut dist = StreamingDistribution::new(100);
        for i in 0..100 {
            dist.add(i as f64);
        }
        let summary = dist.summary();
        assert!(summary.get("count").is_some());
        assert!(summary.get("mean").is_some());
        assert!(summary.get("std").is_some());
        assert!(summary.get("min").is_some());
        assert!(summary.get("max").is_some());
        assert!(summary.get("skewness").is_some());
        assert!(summary.get("kurtosis").is_some());
        assert!(summary.get("percentiles").is_some());
    }

    // =========================================================================
    // Formula-correctness tests with hand-computed expected values
    // =========================================================================

    #[test]
    fn test_skewness_symmetric_data() {
        // Data: [1, 2, 3, 4, 5]. Mean = 3.0 (symmetric about mean).
        // m3 = ((1-3)^3 + (2-3)^3 + (3-3)^3 + (4-3)^3 + (5-3)^3) / 5
        //    = (-8 + -1 + 0 + 1 + 8) / 5 = 0.0
        // Expected skewness = 0.0
        let mut dist = StreamingDistribution::new(100);
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            dist.add(v);
        }
        assert!(
            dist.skewness().abs() < 1e-10,
            "Symmetric data [1,2,3,4,5] should have skewness=0, got {}",
            dist.skewness()
        );
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Data: [1, 1, 1, 1, 10]. Mean = 14/5 = 2.8
        // This is right-skewed (long right tail). Skewness must be positive.
        // m2 = ((1-2.8)^2*4 + (10-2.8)^2) / 5 = (12.96 + 51.84)/5 = 12.96
        // m3 = ((1-2.8)^3*4 + (10-2.8)^3) / 5 = (-23.328 + 373.248)/5 = 69.984
        // std = sqrt(12.96) = 3.6
        // skewness = 69.984 / 3.6^3 = 69.984 / 46.656 = 1.5
        let mut dist = StreamingDistribution::new(100);
        for &v in &[1.0, 1.0, 1.0, 1.0, 10.0] {
            dist.add(v);
        }
        let skew = dist.skewness();
        assert!(
            (skew - 1.5).abs() < 1e-6,
            "Right-skewed [1,1,1,1,10]: expected skewness=1.5, got {}",
            skew
        );
    }

    #[test]
    fn test_kurtosis_uniform_like() {
        // Data: [1,2,3,4,5,6,7,8,9,10]. Mean = 5.5
        // For discrete uniform {1..10}: excess kurtosis = -6(n+1)/5(n-1) where n=number of outcomes
        // But for sample moments: m2 = 8.25, m4 = 101.8125
        // excess kurtosis = m4/m2^2 - 3 = 101.8125/68.0625 - 3 = 1.4955... - 3 = -1.5045...
        // The population excess kurtosis for discrete uniform(1,10) ≈ -1.224
        // But our formula uses the reservoir sample moments directly.
        // Hand-compute: m2 = sum((x-5.5)^2)/10 = (20.25+12.25+6.25+2.25+0.25+0.25+2.25+6.25+12.25+20.25)/10 = 82.5/10 = 8.25
        // m4 = sum((x-5.5)^4)/10 = (410.0625+150.0625+39.0625+5.0625+0.0625+0.0625+5.0625+39.0625+150.0625+410.0625)/10 = 1208.625/10 = 120.8625
        // excess kurt = 120.8625 / 8.25^2 - 3 = 120.8625/68.0625 - 3 = 1.77597... - 3 = -1.224
        let mut dist = StreamingDistribution::new(100);
        for i in 1..=10 {
            dist.add(i as f64);
        }
        let kurt = dist.kurtosis();
        assert!(
            (kurt - (-1.224)).abs() < 0.01,
            "Uniform [1..10]: expected kurtosis near -1.224, got {}",
            kurt
        );
    }

    #[test]
    fn test_kurtosis_peaked() {
        // Data: [0,0,0,0,10,0,0,0,0]. Mean = 10/9 ≈ 1.111
        // This has a single extreme value → leptokurtic → positive excess kurtosis
        let mut dist = StreamingDistribution::new(100);
        for &v in &[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0] {
            dist.add(v);
        }
        assert!(
            dist.kurtosis() > 0.0,
            "Peaked data should have positive excess kurtosis, got {}",
            dist.kurtosis()
        );
    }

    #[test]
    fn test_skewness_constant_returns_zero() {
        let mut dist = StreamingDistribution::new(100);
        for _ in 0..10 {
            dist.add(5.0);
        }
        assert_eq!(
            dist.skewness(),
            0.0,
            "Constant data: skewness should be 0.0"
        );
    }

    #[test]
    fn test_kurtosis_constant_returns_zero() {
        let mut dist = StreamingDistribution::new(100);
        for _ in 0..10 {
            dist.add(5.0);
        }
        assert_eq!(
            dist.kurtosis(),
            0.0,
            "Constant data: kurtosis should be 0.0"
        );
    }

    #[test]
    fn test_skewness_too_few_returns_nan() {
        let mut dist = StreamingDistribution::new(100);
        dist.add(1.0);
        dist.add(2.0);
        assert!(dist.skewness().is_nan(), "len<3: skewness should be NaN");
    }

    #[test]
    fn test_kurtosis_too_few_returns_nan() {
        let mut dist = StreamingDistribution::new(100);
        for &v in &[1.0, 2.0, 3.0] {
            dist.add(v);
        }
        assert!(dist.kurtosis().is_nan(), "len<4: kurtosis should be NaN");
    }

    #[test]
    fn test_add_batch_equals_sequential() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

        let mut seq = StreamingDistribution::with_seed(1000, 42);
        for &v in &values {
            seq.add(v);
        }

        let mut batch = StreamingDistribution::with_seed(1000, 42);
        batch.add_batch(&values);

        assert_eq!(seq.count(), batch.count());
        assert!((seq.mean() - batch.mean()).abs() < 1e-10);
        assert!((seq.std() - batch.std()).abs() < 1e-10);
        assert_eq!(seq.min(), batch.min());
        assert_eq!(seq.max(), batch.max());
    }

    #[test]
    fn test_sorted_sample_is_sorted() {
        let mut dist = StreamingDistribution::new(100);
        for &v in &[5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0] {
            dist.add(v);
        }
        let sorted = dist.sorted_sample();
        for i in 1..sorted.len() {
            assert!(
                sorted[i] >= sorted[i - 1],
                "sorted_sample not monotonic at index {}: {} < {}",
                i,
                sorted[i],
                sorted[i - 1]
            );
        }
    }

    #[test]
    fn test_sample_size_matches_capacity() {
        let mut dist = StreamingDistribution::new(100);
        for i in 0..50 {
            dist.add(i as f64);
        }
        assert_eq!(
            dist.sample_size(),
            50,
            "Underfilled: sample_size should be 50"
        );

        for i in 50..200 {
            dist.add(i as f64);
        }
        assert_eq!(
            dist.sample_size(),
            100,
            "Overfilled: sample_size should be capacity=100"
        );
    }
}
