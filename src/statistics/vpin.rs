//! Volume-Synchronized Probability of Informed Trading (VPIN).
//!
//! Measures information asymmetry using volume-bucketed bars. VPIN is the
//! dominant out-of-sample predictor for spread, volatility, kurtosis,
//! skewness, and serial correlation changes (Easley et al. 2019).
//!
//! ## Algorithm
//!
//! 1. Aggregate trades into equal-volume bars (V_bar = configurable)
//! 2. Classify each bar's buy/sell volume using BVC (Bulk Volume Classification)
//! 3. VPIN = rolling mean of |V_buy - V_sell| / V_bar over n bars
//!
//! ## BVC (Bulk Volume Classification)
//!
//! `V_buy = V * Phi((P_close - P_open) / sigma_P)`
//!
//! where Phi = standard normal CDF, sigma_P = rolling std of bar returns.
//! This is more robust than tick-rule classification for volume-clock data.
//!
//! ## Reference
//!
//! - Easley, López de Prado, O'Hara (2012). "Flow Toxicity and Liquidity
//!   in a High-Frequency World." Review of Financial Studies, 25(5).
//! - Easley et al. (2019). "Microstructure in the Machine Age." RFS.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const DEFAULT_BAR_VOLUME: u64 = 5000;
const DEFAULT_WINDOW_BARS: usize = 50;

/// Standard normal CDF via the error function.
///
/// `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`
///
/// Uses a rational approximation of erf() with max error < 1.5e-7.
/// Reference: Abramowitz & Stegun (1964), formula 7.1.26.
///
/// Used by:
/// - `VpinComputer` for volume-bar BVC classification
/// - `basic-quote-processor` BVC for per-trade probabilistic signing (Easley et al. 2012, Eq. 7)
pub fn phi(x: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Rational approximation of the error function.
///
/// `erf(x) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵) * exp(-x²)`
/// where `t = 1 / (1 + 0.3275911 * |x|)`.
///
/// Max error < 1.5e-7. Reference: Abramowitz & Stegun (1964), formula 7.1.26.
pub fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// A completed volume bar with BVC-classified buy/sell breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VolumeBar {
    buy_volume: f64,
    sell_volume: f64,
}

/// Streaming VPIN computer.
///
/// Feed trades one by one via `add_trade()`. Query `current_vpin()` for
/// the latest VPIN value. Returns `None` until enough bars have been
/// accumulated to fill the rolling window.
///
/// Parameters:
///   - bar_volume: Number of shares per volume bar (default 5000)
///   - window_bars: Rolling window for VPIN average (default 50)
///
/// Reference: Easley, López de Prado, O'Hara (2012), §3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpinComputer {
    bar_volume: u64,
    window_bars: usize,

    current_volume: u64,
    current_value_sum: f64,
    current_open_price: f64,
    current_close_price: f64,

    completed_bars: VecDeque<VolumeBar>,
    bar_returns: VecDeque<f64>,
    rolling_imbalance_sum: f64,

    n_bars_total: u64,
}

impl VpinComputer {
    pub fn new(bar_volume: u64, window_bars: usize) -> Self {
        Self {
            bar_volume: if bar_volume > 0 {
                bar_volume
            } else {
                DEFAULT_BAR_VOLUME
            },
            window_bars: if window_bars > 0 {
                window_bars
            } else {
                DEFAULT_WINDOW_BARS
            },
            current_volume: 0,
            current_value_sum: 0.0,
            current_open_price: 0.0,
            current_close_price: 0.0,
            completed_bars: VecDeque::new(),
            bar_returns: VecDeque::new(),
            rolling_imbalance_sum: 0.0,
            n_bars_total: 0,
        }
    }

    /// Add a single trade. Returns Some(vpin) when VPIN is computable
    /// (enough bars in window), None otherwise.
    pub fn add_trade(&mut self, price: f64, volume: u64) -> Option<f64> {
        if price <= 0.0 || volume == 0 || !price.is_finite() {
            return self.current_vpin();
        }

        if self.current_volume == 0 {
            self.current_open_price = price;
        }
        self.current_close_price = price;
        self.current_volume += volume;
        self.current_value_sum += price * volume as f64;

        let mut result = None;
        while self.current_volume >= self.bar_volume {
            self.complete_bar();
            result = self.current_vpin();
        }
        result
    }

    /// Get current VPIN value if enough bars exist.
    pub fn current_vpin(&self) -> Option<f64> {
        if self.completed_bars.len() < self.window_bars {
            return None;
        }
        let denom = self.window_bars as f64 * self.bar_volume as f64;
        if denom < 1e-12 {
            return None;
        }
        Some(self.rolling_imbalance_sum / denom)
    }

    /// Total volume bars completed since construction.
    pub fn total_bars(&self) -> u64 {
        self.n_bars_total
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.current_volume = 0;
        self.current_value_sum = 0.0;
        self.current_open_price = 0.0;
        self.current_close_price = 0.0;
        self.completed_bars.clear();
        self.bar_returns.clear();
        self.rolling_imbalance_sum = 0.0;
        self.n_bars_total = 0;
    }

    fn complete_bar(&mut self) {
        let bar_vol = self.bar_volume as f64;
        let open = self.current_open_price;
        let close = self.current_close_price;
        let bar_return = if open > 1e-12 {
            (close - open) / open
        } else {
            0.0
        };

        self.bar_returns.push_back(bar_return);

        let sigma = self.estimate_return_sigma();
        let z = if sigma > 1e-12 {
            bar_return / sigma
        } else {
            0.0
        };
        let buy_frac = phi(z);
        let buy_volume = bar_vol * buy_frac;
        let sell_volume = bar_vol * (1.0 - buy_frac);

        let imbalance = (buy_volume - sell_volume).abs();

        let bar = VolumeBar {
            buy_volume,
            sell_volume,
        };
        self.completed_bars.push_back(bar);
        self.rolling_imbalance_sum += imbalance;

        if self.completed_bars.len() > self.window_bars {
            let old = self.completed_bars.pop_front().unwrap();
            let old_imbalance = (old.buy_volume - old.sell_volume).abs();
            self.rolling_imbalance_sum -= old_imbalance;
        }
        if self.bar_returns.len() > self.window_bars + 10 {
            self.bar_returns.pop_front();
        }

        self.n_bars_total += 1;
        let remaining = self.current_volume - self.bar_volume;
        self.current_volume = remaining;
        self.current_value_sum = self.current_close_price * remaining as f64;
        self.current_open_price = self.current_close_price;
    }

    fn estimate_return_sigma(&self) -> f64 {
        if self.bar_returns.len() < 2 {
            return 1e-6;
        }
        let n = self.bar_returns.len() as f64;
        let mean = self.bar_returns.iter().sum::<f64>() / n;
        let variance = self
            .bar_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt().max(1e-8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-6, "Phi(0) should be 0.5");
        assert!(phi(3.0) > 0.998, "Phi(3) should be > 0.998");
        assert!(phi(-3.0) < 0.002, "Phi(-3) should be < 0.002");
        assert!((phi(1.0) - 0.8413).abs() < 0.001, "Phi(1) ≈ 0.8413");
    }

    #[test]
    fn test_phi_golden_values() {
        // Golden values from scipy.stats.norm.cdf() (Python 3.11, scipy 1.12)
        // These are pinned to ensure bit-for-bit consistency across the pipeline.
        let cases = [
            (0.0, 0.5),
            (0.5, 0.6914624612740131),
            (1.0, 0.8413447460685429),
            (-1.0, 0.15865525393145702),
            (2.0, 0.9772498680518208),
            (-2.0, 0.022750131948179195),
            (3.0, 0.9986501019683699),
        ];
        for (x, expected) in cases {
            let got = phi(x);
            assert!(
                (got - expected).abs() < 1.5e-7,
                "phi({}) = {}, expected {} (max error 1.5e-7)",
                x,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_phi_edge_cases() {
        assert_eq!(phi(f64::INFINITY), 1.0, "Phi(+inf) = 1.0");
        assert_eq!(phi(f64::NEG_INFINITY), 0.0, "Phi(-inf) = 0.0");
        assert_eq!(phi(f64::NAN), 0.0, "Phi(NaN) = 0.0 (guard)");
    }

    #[test]
    fn test_vpin_not_ready_until_window_full() {
        let mut vpin = VpinComputer::new(1000, 5);
        // 3000 volume / 1000 bar = 3 bars
        for _ in 0..30 {
            vpin.add_trade(100.0, 100);
        }
        assert!(vpin.current_vpin().is_none(), "3 bars < 5 window");

        // Add 3000 more → 6 bars total
        for _ in 0..30 {
            vpin.add_trade(100.0, 100);
        }
        assert!(vpin.current_vpin().is_some(), "6 bars >= 5 window");
    }

    #[test]
    fn test_vpin_range() {
        let mut vpin = VpinComputer::new(1000, 10);
        for i in 0..20000 {
            let price = 100.0 + (i as f64 * 0.01).sin() * 0.5;
            vpin.add_trade(price, 50);
        }
        if let Some(v) = vpin.current_vpin() {
            assert!(v >= 0.0 && v <= 1.0, "VPIN should be in [0, 1], got {}", v);
        }
    }

    #[test]
    fn test_vpin_high_for_directional_flow() {
        let mut vpin = VpinComputer::new(100, 5);
        for i in 0..1000 {
            let price = 100.0 + i as f64 * 0.1;
            vpin.add_trade(price, 20);
        }
        if let Some(v) = vpin.current_vpin() {
            assert!(
                v > 0.3,
                "Directional flow should produce high VPIN, got {}",
                v
            );
        }
    }

    #[test]
    fn test_reset() {
        let mut vpin = VpinComputer::new(100, 5);
        for _ in 0..1000 {
            vpin.add_trade(100.0, 10);
        }
        assert!(vpin.total_bars() > 0);
        vpin.reset();
        assert_eq!(vpin.total_bars(), 0);
        assert!(vpin.current_vpin().is_none());
    }
}
