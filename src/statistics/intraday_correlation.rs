//! Intraday bivariate correlation accumulator.
//!
//! Accumulates per-bin Pearson correlation statistics for two variables
//! across multiple days, using canonical wall-clock bins (same binning
//! logic as `IntradayCurveAccumulator`).
//!
//! Designed for answering: "What is the OFI-return Pearson r during each
//! minute of the trading day?"
//!
//! # Usage
//!
//! ```ignore
//! let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
//! // For each aligned (ofi, return) pair:
//! corr.add(timestamp_ns, ofi_val, return_val, utc_offset_hours);
//! // After all days:
//! let bins = corr.finalize(); // Vec<CorrelationBinStats>
//! ```

use serde::{Deserialize, Serialize};

const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i64 = 3600 * NS_PER_SECOND;
const NS_PER_DAY: i64 = 24 * NS_PER_HOUR;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PearsonBin {
    sum_xy: f64,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    count: u64,
}

impl PearsonBin {
    fn new() -> Self {
        Self {
            sum_xy: 0.0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_x2: 0.0,
            sum_y2: 0.0,
            count: 0,
        }
    }

    #[inline]
    fn add(&mut self, x: f64, y: f64) {
        self.sum_xy += x * y;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.count += 1;
    }

    fn pearson_r(&self) -> f64 {
        if self.count < 3 {
            return f64::NAN;
        }
        let nf = self.count as f64;
        let cov = self.sum_xy / nf - (self.sum_x / nf) * (self.sum_y / nf);
        let var_x = self.sum_x2 / nf - (self.sum_x / nf).powi(2);
        let var_y = self.sum_y2 / nf - (self.sum_y / nf).powi(2);
        let denom = (var_x * var_y).sqrt();
        if denom > 1e-15 {
            cov / denom
        } else {
            f64::NAN
        }
    }
}

/// Per-bin correlation statistics for finalized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationBinStats {
    pub bin_index: usize,
    pub minutes_since_open: f64,
    pub pearson_r: f64,
    pub count: u64,
}

/// Streaming intraday bivariate correlation accumulator.
///
/// Each bin accumulates Pearson sums for (x, y) pairs falling in that
/// time-of-day slot. Uses the same wall-clock binning as
/// `IntradayCurveAccumulator`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntradayCorrelationAccumulator {
    bins: Vec<PearsonBin>,
    n_bins: usize,
    bin_width_ns: i64,
    open_ns_local: i64,
    close_ns_local: i64,
}

impl IntradayCorrelationAccumulator {
    /// Create an accumulator with custom parameters.
    pub fn new(
        n_bins: usize,
        open_hour: i64,
        open_minute: i64,
        close_hour: i64,
        close_minute: i64,
    ) -> Self {
        let open_ns = open_hour * NS_PER_HOUR + open_minute * NS_PER_MINUTE;
        let close_ns = close_hour * NS_PER_HOUR + close_minute * NS_PER_MINUTE;
        let bin_width = (close_ns - open_ns) / n_bins as i64;

        Self {
            bins: (0..n_bins).map(|_| PearsonBin::new()).collect(),
            n_bins,
            bin_width_ns: bin_width,
            open_ns_local: open_ns,
            close_ns_local: close_ns,
        }
    }

    /// Standard RTH curve: 390 one-minute bins from 09:30 to 16:00.
    pub fn new_rth_1min() -> Self {
        Self::new(390, 9, 30, 16, 0)
    }

    /// Add an (x, y) pair to the appropriate time bin.
    ///
    /// Events outside the [open, close) window are silently dropped.
    #[inline]
    pub fn add(&mut self, ts_ns: i64, x: f64, y: f64, utc_offset_hours: i32) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let offset_ns = (utc_offset_hours as i64) * NS_PER_HOUR;
        let local_ns = ((ts_ns + offset_ns) % NS_PER_DAY + NS_PER_DAY) % NS_PER_DAY;

        if local_ns < self.open_ns_local || local_ns >= self.close_ns_local {
            return;
        }

        let bin = ((local_ns - self.open_ns_local) / self.bin_width_ns) as usize;
        if bin < self.n_bins {
            self.bins[bin].add(x, y);
        }
    }

    /// Produce finalized per-bin correlation statistics.
    pub fn finalize(&self) -> Vec<CorrelationBinStats> {
        self.bins
            .iter()
            .enumerate()
            .map(|(i, bin)| {
                let minutes = i as f64 * (self.bin_width_ns as f64 / NS_PER_MINUTE as f64);
                CorrelationBinStats {
                    bin_index: i,
                    minutes_since_open: minutes,
                    pearson_r: bin.pearson_r(),
                    count: bin.count,
                }
            })
            .collect()
    }

    /// Total observation pairs across all bins.
    pub fn total_count(&self) -> u64 {
        self.bins.iter().map(|b| b.count).sum()
    }

    /// Reset all bins while preserving structural parameters (n_bins, bin_width, open/close).
    pub fn reset(&mut self) {
        for bin in &mut self.bins {
            bin.count = 0;
            bin.sum_x = 0.0;
            bin.sum_y = 0.0;
            bin.sum_x2 = 0.0;
            bin.sum_y2 = 0.0;
            bin.sum_xy = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rth_has_390_bins() {
        let corr = IntradayCorrelationAccumulator::new_rth_1min();
        assert_eq!(corr.n_bins, 390);
    }

    #[test]
    fn test_perfect_positive_correlation() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let utc_offset = -5; // EST
        let ts_base = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE; // 09:30 ET

        for i in 0..100 {
            let x = i as f64;
            let ts = ts_base + (i as i64) * (NS_PER_MINUTE / 200);
            corr.add(ts, x, x, utc_offset);
        }

        let stats = corr.finalize();
        assert!(
            (stats[0].pearson_r - 1.0).abs() < 1e-10,
            "Perfect positive: expected r=1.0, got {}",
            stats[0].pearson_r
        );
        assert_eq!(stats[0].count, 100);
    }

    #[test]
    fn test_perfect_negative_correlation() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let utc_offset = -5;
        let ts_base = 14 * NS_PER_HOUR + 31 * NS_PER_MINUTE; // 09:31 ET → bin 1

        for i in 0..50 {
            let x = i as f64;
            let ts = ts_base + (i as i64) * (NS_PER_MINUTE / 100);
            corr.add(ts, x, -x, utc_offset);
        }

        let stats = corr.finalize();
        assert!(
            (stats[1].pearson_r - (-1.0)).abs() < 1e-10,
            "Perfect negative: expected r=-1.0, got {}",
            stats[1].pearson_r
        );
    }

    #[test]
    fn test_insufficient_data_returns_nan() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let utc_offset = -5;
        let ts = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        corr.add(ts, 1.0, 2.0, utc_offset);
        corr.add(ts + 1, 3.0, 4.0, utc_offset);

        let stats = corr.finalize();
        assert!(
            stats[0].pearson_r.is_nan(),
            "Fewer than 3 observations should produce NaN, got {}",
            stats[0].pearson_r
        );
        assert_eq!(stats[0].count, 2);
    }

    #[test]
    fn test_pre_market_dropped() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        corr.add(14 * NS_PER_HOUR, 1.0, 2.0, -5); // 09:00 ET
        assert_eq!(corr.total_count(), 0);
    }

    #[test]
    fn test_post_market_dropped() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        corr.add(21 * NS_PER_HOUR + NS_PER_MINUTE, 1.0, 2.0, -5); // 16:01 ET
        assert_eq!(corr.total_count(), 0);
    }

    #[test]
    fn test_non_finite_values_dropped() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let ts = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;
        corr.add(ts, f64::NAN, 1.0, -5);
        corr.add(ts, 1.0, f64::INFINITY, -5);
        corr.add(ts, f64::NEG_INFINITY, 1.0, -5);
        assert_eq!(corr.total_count(), 0);
    }

    #[test]
    fn test_cross_day_accumulation() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let utc_offset = -5;
        let ts = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        for i in 0..200 {
            let x = i as f64;
            let y = 0.5 * x + 10.0;
            corr.add(ts + (i as i64) * 100, x, y, utc_offset);
        }

        let stats = corr.finalize();
        assert_eq!(stats[0].count, 200);
        assert!(
            (stats[0].pearson_r - 1.0).abs() < 1e-10,
            "y = 0.5x + 10 should give r=1.0, got {}",
            stats[0].pearson_r
        );
    }

    #[test]
    fn test_bin_routing_across_minutes() {
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let utc_offset = -5;

        // Add to bin 0 (09:30)
        let ts0 = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;
        for i in 0..10 {
            corr.add(ts0 + (i as i64) * 1000, i as f64, i as f64, utc_offset);
        }

        // Add to bin 5 (09:35)
        let ts5 = 14 * NS_PER_HOUR + 35 * NS_PER_MINUTE;
        for i in 0..20 {
            corr.add(ts5 + (i as i64) * 1000, i as f64, -(i as f64), utc_offset);
        }

        let stats = corr.finalize();
        assert_eq!(stats[0].count, 10);
        assert_eq!(stats[5].count, 20);
        assert_eq!(stats[1].count, 0);
    }

    #[test]
    fn test_minutes_since_open_labels() {
        let corr = IntradayCorrelationAccumulator::new_rth_1min();
        let stats = corr.finalize();
        assert!((stats[0].minutes_since_open - 0.0).abs() < 1e-10);
        assert!((stats[1].minutes_since_open - 1.0).abs() < 1e-10);
        assert!((stats[389].minutes_since_open - 389.0).abs() < 1e-10);
    }

    #[test]
    fn test_known_correlation_value() {
        // x = [1, 2, 3, 4, 5], y = [2, 4, 5, 4, 5]
        // Pearson r = 0.7745966... (calculated by hand)
        let mut corr = IntradayCorrelationAccumulator::new_rth_1min();
        let utc_offset = -5;
        let ts = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 5.0, 4.0, 5.0];
        for (i, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
            corr.add(ts + (i as i64) * 100, x, y, utc_offset);
        }

        let stats = corr.finalize();
        let expected_r = 0.7745966692414834;
        assert!(
            (stats[0].pearson_r - expected_r).abs() < 1e-10,
            "Known r: expected {}, got {}",
            expected_r,
            stats[0].pearson_r
        );
    }
}
