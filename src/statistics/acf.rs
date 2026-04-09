//! Autocorrelation function (ACF) computation.
//!
//! Stores values in a ring buffer and computes sample ACF at configurable lags.
//!
//! Formula: ACF(k) = (1/N) * sum_{t=1}^{N-k} (x_t - mean)(x_{t+k} - mean) / var(x)
//!
//! Used by OFI (flow persistence), Return (momentum/mean-reversion),
//! Volatility (clustering/ARCH effects), Spread (persistence).

use serde::{Deserialize, Serialize};

/// Ring buffer for streaming ACF computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcfComputer {
    buffer: Vec<f64>,
    capacity: usize,
    write_pos: usize,
    filled: bool,
    max_lag: usize,
}

impl AcfComputer {
    /// Create a new ACF computer.
    ///
    /// # Arguments
    /// * `capacity` — Maximum number of values to store
    /// * `max_lag` — Maximum lag for ACF computation
    pub fn new(capacity: usize, max_lag: usize) -> Self {
        assert!(capacity > max_lag, "capacity must exceed max_lag");
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            filled: false,
            max_lag,
        }
    }

    /// Default: 10,000 values, 20 lags.
    pub fn default_20lag() -> Self {
        Self::new(10_000, 20)
    }

    /// Add a value to the ring buffer.
    #[inline]
    pub fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.buffer[self.write_pos] = value;
        self.write_pos += 1;
        if self.write_pos >= self.capacity {
            self.write_pos = 0;
            self.filled = true;
        }
    }

    /// Add a batch of values.
    pub fn push_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.push(v);
        }
    }

    /// Number of values currently stored.
    pub fn len(&self) -> usize {
        if self.filled {
            self.capacity
        } else {
            self.write_pos
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compute ACF at lags 1..=max_lag.
    ///
    /// Returns a vector of length `max_lag` where index i corresponds to lag i+1.
    /// Returns empty vec if insufficient data.
    pub fn compute(&self) -> Vec<f64> {
        let n = self.len();
        if n < self.max_lag + 2 {
            return vec![f64::NAN; self.max_lag];
        }

        let data = self.as_slice();
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if variance < 1e-15 {
            return vec![0.0; self.max_lag];
        }

        let mut acf = Vec::with_capacity(self.max_lag);
        for lag in 1..=self.max_lag {
            let cov: f64 = data[..n - lag]
                .iter()
                .zip(data[lag..].iter())
                .map(|(&x, &y)| (x - mean) * (y - mean))
                .sum::<f64>()
                / n as f64;
            acf.push(cov / variance);
        }
        acf
    }

    /// Get values as a contiguous slice (linearized from ring buffer).
    fn as_slice(&self) -> Vec<f64> {
        if self.filled {
            let mut out = Vec::with_capacity(self.capacity);
            out.extend_from_slice(&self.buffer[self.write_pos..]);
            out.extend_from_slice(&self.buffer[..self.write_pos]);
            out
        } else {
            self.buffer[..self.write_pos].to_vec()
        }
    }

    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.filled = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_series_zero_acf() {
        let mut acf = AcfComputer::new(100, 5);
        for _ in 0..100 {
            acf.push(42.0);
        }
        let result = acf.compute();
        assert_eq!(result.len(), 5);
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-10, "Constant series should have ACF=0");
        }
    }

    #[test]
    fn test_alternating_series() {
        let mut acf = AcfComputer::new(200, 5);
        for i in 0..200 {
            acf.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let result = acf.compute();
        assert!(
            result[0] < -0.9,
            "Alternating series should have lag-1 ACF near -1"
        );
        assert!(
            result[1] > 0.9,
            "Alternating series should have lag-2 ACF near +1"
        );
    }

    #[test]
    fn test_insufficient_data() {
        let mut acf = AcfComputer::new(100, 20);
        for i in 0..10 {
            acf.push(i as f64);
        }
        let result = acf.compute();
        assert_eq!(result.len(), 20);
        assert!(result[0].is_nan());
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut acf = AcfComputer::new(50, 5);
        for i in 0..100 {
            acf.push(i as f64);
        }
        assert_eq!(acf.len(), 50);
        assert!(acf.filled);
    }

    #[test]
    fn test_nan_skipped() {
        let mut acf = AcfComputer::new(100, 5);
        acf.push(1.0);
        acf.push(f64::NAN);
        acf.push(2.0);
        assert_eq!(acf.len(), 2);
    }

    #[test]
    fn test_acf_exact_alternating_value() {
        // Values: [-1, 1, -1, 1, ...] (200 values)
        // Mean = 0 (equal count of +1 and -1)
        // Variance = mean(x^2) - mean(x)^2 = 1 - 0 = 1
        // ACF(1) = cov(1) / var
        // cov(1) = (1/N) * sum_{t=0}^{N-2}(x_t * x_{t+1})
        //        = (1/200) * (-1 * 199) = -199/200 = -0.995
        // ACF(1) = -199/200 / 1 = -0.995
        let mut acf = AcfComputer::new(200, 5);
        for i in 0..200 {
            acf.push(if i % 2 == 0 { -1.0 } else { 1.0 });
        }
        let result = acf.compute();
        let expected_lag1 = -199.0 / 200.0;
        assert!(
            (result[0] - expected_lag1).abs() < 0.001,
            "ACF(1) of alternating ±1: expected {}, got {}",
            expected_lag1,
            result[0]
        );
    }
}
