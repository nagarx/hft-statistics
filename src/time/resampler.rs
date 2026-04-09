//! Canonical-grid time resampler for multi-scale analysis.
//!
//! Bins timestamps + values onto a fixed wall-clock grid (e.g., every 5 seconds
//! from 09:30:00.000). Grid edges are determined by `day_epoch_ns` + `utc_offset` +
//! `bin_width_ns`, NOT by the first data point. This ensures identical bin boundaries
//! across different days — fixing Python audit Issue 1.
//!
//! # Aggregation Modes
//!
//! - `Sum`: sum of values per bin (OFI accumulation)
//! - `Mean`: arithmetic mean (spread, volume)
//! - `Last`: last value in bin (price sampling for returns)
//! - `Count`: number of events per bin (activity)

use serde::{Deserialize, Serialize};

const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i64 = 3600 * NS_PER_SECOND;

/// Aggregation mode for binned values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggMode {
    Sum,
    Mean,
    Last,
    Count,
}

/// Result of resampling a day's data onto a canonical grid.
#[derive(Debug, Clone)]
pub struct ResampledBins {
    /// Bin edge timestamps in nanoseconds (N+1 edges for N bins).
    pub edges_ns: Vec<i64>,
    /// Aggregated values per bin.
    pub values: Vec<f64>,
    /// Event count per bin.
    pub counts: Vec<u64>,
    /// Bin width in nanoseconds.
    pub bin_width_ns: i64,
}

impl ResampledBins {
    /// Get indices of bins that have at least one event.
    pub fn filled_indices(&self) -> Vec<usize> {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.values.len()
    }

    /// Number of non-empty bins.
    pub fn n_filled(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }
}

/// Resample timestamps + values onto a canonical grid.
///
/// # Arguments
/// * `timestamps_ns` — UTC nanosecond timestamps (must be sorted)
/// * `values` — Corresponding values (same length as timestamps)
/// * `bin_width_ns` — Width of each bin in nanoseconds
/// * `day_epoch_ns` — Midnight UTC nanoseconds for the trading day
/// * `utc_offset_hours` — UTC offset for this day (-5 EST, -4 EDT)
/// * `mode` — Aggregation mode
///
/// # Grid Construction
///
/// Grid runs from RTH open (09:30 local) to close (16:00 local):
/// - Edge 0: 09:30:00.000 in UTC nanoseconds
/// - Edge 1: 09:30:00.000 + bin_width_ns
/// - ...
/// - Edge N: 16:00:00.000
pub fn resample_to_grid(
    timestamps_ns: &[i64],
    values: &[f64],
    bin_width_ns: i64,
    day_epoch_ns: i64,
    utc_offset_hours: i32,
    mode: AggMode,
) -> ResampledBins {
    let offset_ns = (utc_offset_hours as i64) * NS_PER_HOUR;

    let open_local_ns = 9 * NS_PER_HOUR + 30 * NS_PER_MINUTE;
    let close_local_ns = 16 * NS_PER_HOUR;

    let open_utc_ns = day_epoch_ns + open_local_ns - offset_ns;
    let close_utc_ns = day_epoch_ns + close_local_ns - offset_ns;

    let rth_duration = close_utc_ns - open_utc_ns;
    let n_bins = (rth_duration / bin_width_ns) as usize;

    let mut edges_ns = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        edges_ns.push(open_utc_ns + i as i64 * bin_width_ns);
    }

    let mut sums = vec![0.0f64; n_bins];
    let mut counts = vec![0u64; n_bins];
    let mut lasts = vec![f64::NAN; n_bins];

    for (&ts, &val) in timestamps_ns.iter().zip(values.iter()) {
        if ts < open_utc_ns || ts >= close_utc_ns || !val.is_finite() {
            continue;
        }
        let bin = ((ts - open_utc_ns) / bin_width_ns) as usize;
        if bin < n_bins {
            sums[bin] += val;
            counts[bin] += 1;
            lasts[bin] = val;
        }
    }

    let values = match mode {
        AggMode::Sum => sums,
        AggMode::Mean => sums
            .iter()
            .zip(counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN })
            .collect(),
        AggMode::Last => lasts,
        AggMode::Count => counts.iter().map(|&c| c as f64).collect(),
    };

    ResampledBins {
        edges_ns,
        values,
        counts,
        bin_width_ns,
    }
}

/// Convenience: compute RTH grid edges for a given bin width.
pub fn rth_grid_edges_ns(day_epoch_ns: i64, bin_width_ns: i64, utc_offset_hours: i32) -> Vec<i64> {
    let offset_ns = (utc_offset_hours as i64) * NS_PER_HOUR;
    let open_utc = day_epoch_ns + 9 * NS_PER_HOUR + 30 * NS_PER_MINUTE - offset_ns;
    let close_utc = day_epoch_ns + 16 * NS_PER_HOUR - offset_ns;
    let n_bins = ((close_utc - open_utc) / bin_width_ns) as usize;

    (0..=n_bins)
        .map(|i| open_utc + i as i64 * bin_width_ns)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_day_epoch() -> i64 {
        0 // epoch midnight
    }

    #[test]
    fn test_grid_has_correct_bin_count() {
        let bins = resample_to_grid(
            &[],
            &[],
            5 * NS_PER_SECOND,
            make_day_epoch(),
            -5,
            AggMode::Sum,
        );
        // 6.5 hours / 5 seconds = 4680 bins
        assert_eq!(bins.n_bins(), 4680);
    }

    #[test]
    fn test_1min_grid_has_390_bins() {
        let bins = resample_to_grid(&[], &[], NS_PER_MINUTE, make_day_epoch(), -5, AggMode::Sum);
        assert_eq!(bins.n_bins(), 390);
    }

    #[test]
    fn test_sum_aggregation() {
        let offset = -5;
        let open_utc = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE; // 09:30 ET

        let ts = vec![open_utc + 1, open_utc + 2, open_utc + 3];
        let vals = vec![10.0, 20.0, 30.0];

        let bins = resample_to_grid(&ts, &vals, 5 * NS_PER_SECOND, 0, offset, AggMode::Sum);
        assert!((bins.values[0] - 60.0).abs() < 1e-10);
        assert_eq!(bins.counts[0], 3);
    }

    #[test]
    fn test_mean_aggregation() {
        let offset = -5;
        let open_utc = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        let ts = vec![open_utc + 1, open_utc + 2];
        let vals = vec![10.0, 30.0];

        let bins = resample_to_grid(&ts, &vals, 5 * NS_PER_SECOND, 0, offset, AggMode::Mean);
        assert!((bins.values[0] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_last_aggregation() {
        let offset = -5;
        let open_utc = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        let ts = vec![open_utc + 1, open_utc + 2, open_utc + 3];
        let vals = vec![10.0, 20.0, 30.0];

        let bins = resample_to_grid(&ts, &vals, 5 * NS_PER_SECOND, 0, offset, AggMode::Last);
        assert!((bins.values[0] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_pre_market_events_excluded() {
        let offset = -5;
        let pre_market = 14 * NS_PER_HOUR; // 09:00 ET

        let ts = vec![pre_market];
        let vals = vec![99.0];

        let bins = resample_to_grid(&ts, &vals, NS_PER_MINUTE, 0, offset, AggMode::Sum);
        assert_eq!(bins.n_filled(), 0);
    }

    #[test]
    fn test_different_bins_separate() {
        let offset = -5;
        let open_utc = 14 * NS_PER_HOUR + 30 * NS_PER_MINUTE;

        let ts = vec![open_utc + 1, open_utc + 5 * NS_PER_SECOND + 1];
        let vals = vec![10.0, 20.0];

        let bins = resample_to_grid(&ts, &vals, 5 * NS_PER_SECOND, 0, offset, AggMode::Sum);
        assert!((bins.values[0] - 10.0).abs() < 1e-10);
        assert!((bins.values[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_bins_are_nan_for_mean() {
        let bins = resample_to_grid(&[], &[], NS_PER_MINUTE, 0, -5, AggMode::Mean);
        assert!(bins.values[0].is_nan());
    }

    #[test]
    fn test_rth_grid_edges() {
        let edges = rth_grid_edges_ns(0, NS_PER_MINUTE, -5);
        assert_eq!(edges.len(), 391); // 390 bins + 1 edge
    }
}
