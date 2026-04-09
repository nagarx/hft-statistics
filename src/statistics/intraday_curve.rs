//! Intraday curve accumulator using canonical wall-clock bins.
//!
//! Accumulates per-bin streaming statistics across multiple days for
//! intraday U-shaped curves (spread, volatility, volume, OFI).
//!
//! Bins are aligned to wall-clock time (e.g., 09:30:00, 09:31:00, ...),
//! NOT to data-dependent start times. This fixes Python audit Issue 1.
//!
//! # Usage
//!
//! ```ignore
//! let mut curve = IntradayCurveAccumulator::new_rth_1min(); // 390 bins, 09:30-16:00
//! // For each event on each day:
//! curve.add(timestamp_ns, value, utc_offset_hours);
//! // After all days:
//! let bins = curve.finalize(); // Vec of (bin_start_minutes, mean, std, count)
//! ```

use serde::{Deserialize, Serialize};

use super::welford::WelfordAccumulator;

const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i64 = 3600 * NS_PER_SECOND;

/// Per-bin statistics for an intraday curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinStats {
    pub bin_index: usize,
    pub minutes_since_open: f64,
    pub mean: f64,
    pub std: f64,
    pub count: u64,
}

/// Streaming intraday curve accumulator with canonical wall-clock bins.
///
/// Each bin is a `WelfordAccumulator` that receives values from all days
/// for that specific time-of-day slot. Cross-day aggregation is automatic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntradayCurveAccumulator {
    bins: Vec<WelfordAccumulator>,
    n_bins: usize,
    bin_width_ns: i64,
    open_ns_local: i64,
    close_ns_local: i64,
}

impl IntradayCurveAccumulator {
    /// Create an accumulator with custom parameters.
    ///
    /// # Arguments
    /// * `n_bins` — Number of time bins
    /// * `open_hour` / `open_minute` — Market open in local time
    /// * `close_hour` / `close_minute` — Market close in local time
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
            bins: (0..n_bins).map(|_| WelfordAccumulator::new()).collect(),
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

    /// Add a value to the appropriate time bin.
    ///
    /// Events outside the [open, close) window are silently dropped.
    #[inline]
    pub fn add(&mut self, ts_ns: i64, value: f64, utc_offset_hours: i32) {
        if !value.is_finite() {
            return;
        }
        let offset_ns = (utc_offset_hours as i64) * NS_PER_HOUR;
        let local_ns =
            ((ts_ns + offset_ns) % (24 * NS_PER_HOUR) + 24 * NS_PER_HOUR) % (24 * NS_PER_HOUR);

        if local_ns < self.open_ns_local || local_ns >= self.close_ns_local {
            return;
        }

        let bin = ((local_ns - self.open_ns_local) / self.bin_width_ns) as usize;
        if bin < self.n_bins {
            self.bins[bin].update(value);
        }
    }

    /// Produce finalized bin statistics.
    pub fn finalize(&self) -> Vec<BinStats> {
        self.bins
            .iter()
            .enumerate()
            .map(|(i, acc)| {
                let minutes = i as f64 * (self.bin_width_ns as f64 / NS_PER_MINUTE as f64);
                BinStats {
                    bin_index: i,
                    minutes_since_open: minutes,
                    mean: acc.mean(),
                    std: acc.std(),
                    count: acc.count(),
                }
            })
            .collect()
    }

    /// Get total observations across all bins.
    pub fn total_count(&self) -> u64 {
        self.bins.iter().map(|b| b.count()).sum()
    }

    /// Reset all bins while preserving structural parameters (n_bins, bin_width, open/close).
    pub fn reset(&mut self) {
        for bin in &mut self.bins {
            bin.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rth_has_390_bins() {
        let curve = IntradayCurveAccumulator::new_rth_1min();
        assert_eq!(curve.n_bins, 390);
    }

    #[test]
    fn test_bin_assignment() {
        let mut curve = IntradayCurveAccumulator::new_rth_1min();
        let utc_offset = -5; // EST

        // 09:30:00 ET = 14:30:00 UTC → bin 0
        let ts_open = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;
        curve.add(ts_open, 1.0, utc_offset);

        // 09:31:00 ET = 14:31:00 UTC → bin 1
        let ts_next = ts_open + NS_PER_MINUTE;
        curve.add(ts_next, 2.0, utc_offset);

        let stats = curve.finalize();
        assert_eq!(stats[0].count, 1);
        assert!((stats[0].mean - 1.0).abs() < 1e-10);
        assert_eq!(stats[1].count, 1);
        assert!((stats[1].mean - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pre_market_dropped() {
        let mut curve = IntradayCurveAccumulator::new_rth_1min();
        // 09:00 ET = 14:00 UTC (pre-market, should be dropped)
        curve.add(14 * NS_PER_HOUR, 99.0, -5);
        assert_eq!(curve.total_count(), 0);
    }

    #[test]
    fn test_post_market_dropped() {
        let mut curve = IntradayCurveAccumulator::new_rth_1min();
        // 16:01 ET = 21:01 UTC (post-market, should be dropped)
        curve.add(21 * NS_PER_HOUR + NS_PER_MINUTE, 99.0, -5);
        assert_eq!(curve.total_count(), 0);
    }

    #[test]
    fn test_cross_day_accumulation() {
        let mut curve = IntradayCurveAccumulator::new_rth_1min();
        let ts = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE; // 09:30 ET

        curve.add(ts, 10.0, -5);
        curve.add(ts, 20.0, -5);
        curve.add(ts, 30.0, -5);

        let stats = curve.finalize();
        assert_eq!(stats[0].count, 3);
        assert!((stats[0].mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_minutes_since_open_labels() {
        let curve = IntradayCurveAccumulator::new_rth_1min();
        let stats = curve.finalize();
        assert!((stats[0].minutes_since_open - 0.0).abs() < 1e-10);
        assert!((stats[1].minutes_since_open - 1.0).abs() < 1e-10);
        assert!((stats[389].minutes_since_open - 389.0).abs() < 1e-10);
    }
}
