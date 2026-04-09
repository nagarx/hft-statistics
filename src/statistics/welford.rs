//! Numerically stable streaming mean/variance via Welford's algorithm.
//!
//! Reference: Welford, B.P. (1962). "Note on a method for calculating corrected
//! sums of squares and products." Technometrics, 4(3), 419-420.

use serde::{Deserialize, Serialize};

/// Streaming mean/variance accumulator using Welford's online algorithm.
///
/// Handles NaN/Inf values by skipping them. Tracks exact min/max.
/// All operations are O(1) per update with O(1) memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelfordAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl WelfordAccumulator {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Update with a single value. NaN/Inf values are skipped.
    #[inline]
    pub fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Update with a batch of values.
    #[inline]
    pub fn update_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.mean
        }
    }

    /// Population variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Sample variance (Bessel's correction).
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn sample_std(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    pub fn min(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.min
        }
    }

    pub fn max(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.max
        }
    }

    /// Merge another accumulator into this one (parallel-friendly).
    pub fn merge(&mut self, other: &WelfordAccumulator) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * (other.count as f64 / combined_count as f64);
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64 / combined_count as f64);

        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.mean(), 0.0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn test_single_value() {
        let mut acc = WelfordAccumulator::new();
        acc.update(5.0);
        assert_eq!(acc.count(), 1);
        assert!((acc.mean() - 5.0).abs() < 1e-10);
        assert_eq!(acc.min(), 5.0);
        assert_eq!(acc.max(), 5.0);
    }

    #[test]
    fn test_known_values() {
        let mut acc = WelfordAccumulator::new();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            acc.update(v);
        }
        assert_eq!(acc.count(), 8);
        assert!((acc.mean() - 5.0).abs() < 1e-10);
        assert!((acc.variance() - 4.0).abs() < 1e-10);
        assert!((acc.sample_variance() - (32.0 / 7.0)).abs() < 1e-10);
        assert_eq!(acc.min(), 2.0);
        assert_eq!(acc.max(), 9.0);
    }

    #[test]
    fn test_nan_skipped() {
        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(f64::NAN);
        acc.update(3.0);
        assert_eq!(acc.count(), 2);
        assert!((acc.mean() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_inf_skipped() {
        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(f64::INFINITY);
        acc.update(f64::NEG_INFINITY);
        assert_eq!(acc.count(), 1);
    }

    #[test]
    fn test_merge() {
        let mut a = WelfordAccumulator::new();
        let mut b = WelfordAccumulator::new();
        for &v in &[1.0, 2.0, 3.0] {
            a.update(v);
        }
        for &v in &[4.0, 5.0, 6.0] {
            b.update(v);
        }
        a.merge(&b);
        assert_eq!(a.count(), 6);
        assert!((a.mean() - 3.5).abs() < 1e-10);
        assert_eq!(a.min(), 1.0);
        assert_eq!(a.max(), 6.0);
    }

    #[test]
    fn test_reset() {
        let mut acc = WelfordAccumulator::new();
        acc.update(42.0);
        acc.reset();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.mean(), 0.0);
    }

    #[test]
    fn test_batch_update() {
        let mut acc = WelfordAccumulator::new();
        acc.update_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(acc.count(), 5);
        assert!((acc.mean() - 3.0).abs() < 1e-10);
    }

    // =========================================================================
    // Formula-correctness: merge variance
    // =========================================================================

    #[test]
    fn test_merge_variance_correctness() {
        // A = [1, 2, 3], B = [4, 5, 6]
        // Combined = [1, 2, 3, 4, 5, 6], mean = 3.5
        // Population variance = sum((x-3.5)^2)/6
        //   = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6
        //   = 17.5 / 6 = 35/12 ≈ 2.91667
        let mut a = WelfordAccumulator::new();
        let mut b = WelfordAccumulator::new();
        for &v in &[1.0, 2.0, 3.0] {
            a.update(v);
        }
        for &v in &[4.0, 5.0, 6.0] {
            b.update(v);
        }
        a.merge(&b);
        let expected_var = 35.0 / 12.0;
        assert!(
            (a.variance() - expected_var).abs() < 1e-10,
            "Merged variance: expected {}, got {}",
            expected_var,
            a.variance()
        );
    }

    #[test]
    fn test_merge_empty_into_nonempty() {
        let mut a = WelfordAccumulator::new();
        for &v in &[1.0, 2.0, 3.0] {
            a.update(v);
        }
        let original_mean = a.mean();
        let original_var = a.variance();

        let empty = WelfordAccumulator::new();
        a.merge(&empty);

        assert_eq!(a.count(), 3);
        assert!((a.mean() - original_mean).abs() < 1e-15);
        assert!((a.variance() - original_var).abs() < 1e-15);
    }

    #[test]
    fn test_merge_nonempty_into_empty() {
        let mut empty = WelfordAccumulator::new();
        let mut b = WelfordAccumulator::new();
        for &v in &[10.0, 20.0, 30.0] {
            b.update(v);
        }

        empty.merge(&b);

        assert_eq!(empty.count(), 3);
        assert!((empty.mean() - 20.0).abs() < 1e-10);
        // var([10,20,30]) = ((10-20)^2 + (20-20)^2 + (30-20)^2)/3 = 200/3
        assert!((empty.variance() - 200.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_std_single_element() {
        let mut acc = WelfordAccumulator::new();
        acc.update(42.0);
        assert_eq!(
            acc.sample_std(),
            0.0,
            "Single element: sample_std should be 0.0"
        );
        assert_eq!(
            acc.sample_variance(),
            0.0,
            "Single element: sample_variance should be 0.0"
        );
    }
}
