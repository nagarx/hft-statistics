//! Per-regime streaming statistics accumulator.
//!
//! Maintains a separate `WelfordAccumulator` for each of the 7 intraday regimes,
//! enabling regime-conditional analysis across all days.

use serde::{Deserialize, Serialize};

use super::welford::WelfordAccumulator;
use crate::time::N_REGIMES;

/// Streaming accumulator that partitions values by intraday regime.
///
/// Maintains one `WelfordAccumulator` per regime (7 regimes for US equities).
///
/// Uses custom serde to handle `WelfordAccumulator` fields that contain
/// `f64::INFINITY` (min/max of empty accumulators), which JSON cannot represent.
#[derive(Debug, Clone, Serialize)]
pub struct RegimeAccumulator {
    accumulators: [WelfordAccumulator; N_REGIMES],
}

impl<'de> Deserialize<'de> for RegimeAccumulator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Match the derive Serialize output: {"accumulators": [...]}
        #[derive(Deserialize)]
        struct Helper {
            accumulators: Vec<serde_json::Value>,
        }
        let helper = Helper::deserialize(deserializer)?;
        let mut arr: [WelfordAccumulator; N_REGIMES] =
            std::array::from_fn(|_| WelfordAccumulator::new());
        for (i, val) in helper.accumulators.into_iter().enumerate().take(N_REGIMES) {
            // Only deserialize if the accumulator has data (count > 0).
            // Empty accumulators have Infinity min/max which serializes as null.
            if let Some(count) = val.get("count").and_then(|c| c.as_u64()) {
                if count > 0 {
                    if let Ok(acc) = serde_json::from_value(val) {
                        arr[i] = acc;
                    }
                }
            }
        }
        Ok(Self { accumulators: arr })
    }
}

impl RegimeAccumulator {
    pub fn new() -> Self {
        Self {
            accumulators: std::array::from_fn(|_| WelfordAccumulator::new()),
        }
    }

    /// Add a value to the accumulator for a given regime.
    #[inline]
    pub fn add(&mut self, regime: u8, value: f64) {
        let idx = regime as usize;
        if idx < N_REGIMES {
            self.accumulators[idx].update(value);
        }
    }

    /// Get the accumulator for a specific regime.
    ///
    /// # Panics
    ///
    /// Panics if `regime >= N_REGIMES` (7). This is consistent with `add()`,
    /// which silently drops values for out-of-range regimes.
    pub fn get(&self, regime: u8) -> &WelfordAccumulator {
        let idx = regime as usize;
        assert!(
            idx < N_REGIMES,
            "regime {} out of range (max {})",
            regime,
            N_REGIMES - 1
        );
        &self.accumulators[idx]
    }

    /// Produce a summary for all regimes.
    pub fn finalize(&self) -> serde_json::Value {
        let regime_labels = crate::time::regime::REGIME_LABELS;
        let mut map = serde_json::Map::new();
        for (i, acc) in self.accumulators.iter().enumerate() {
            if acc.count() > 0 {
                map.insert(
                    regime_labels[i].to_string(),
                    serde_json::json!({
                        "mean": acc.mean(),
                        "std": acc.std(),
                        "count": acc.count(),
                        "min": acc.min(),
                        "max": acc.max(),
                    }),
                );
            }
        }
        serde_json::Value::Object(map)
    }

    pub fn reset(&mut self) {
        for acc in &mut self.accumulators {
            acc.reset();
        }
    }
}

impl Default for RegimeAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_separation() {
        let mut ra = RegimeAccumulator::new();
        ra.add(0, 10.0); // pre-market
        ra.add(0, 20.0);
        ra.add(3, 100.0); // midday
        ra.add(3, 200.0);

        assert_eq!(ra.get(0).count(), 2);
        assert!((ra.get(0).mean() - 15.0).abs() < 1e-10);
        assert_eq!(ra.get(3).count(), 2);
        assert!((ra.get(3).mean() - 150.0).abs() < 1e-10);
        assert_eq!(ra.get(1).count(), 0);
    }

    #[test]
    fn test_finalize_output() {
        let mut ra = RegimeAccumulator::new();
        ra.add(1, 5.0);
        let result = ra.finalize();
        assert!(result.get("open-auction").is_some());
        assert!(result.get("pre-market").is_none());
    }

    #[test]
    fn test_serde_round_trip() {
        let mut ra = RegimeAccumulator::new();
        ra.add(0, 10.0);
        ra.add(0, 20.0);
        ra.add(3, 100.0);

        let json = serde_json::to_string(&ra).unwrap();
        let restored: RegimeAccumulator = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.get(0).count(), 2);
        assert!((restored.get(0).mean() - 15.0).abs() < 1e-10);
        assert_eq!(restored.get(3).count(), 1);
        assert!((restored.get(3).mean() - 100.0).abs() < 1e-10);
        assert_eq!(restored.get(1).count(), 0);
    }

    #[test]
    fn test_out_of_range_regime_dropped() {
        let mut ra = RegimeAccumulator::new();
        ra.add(255, 1.0);
        for i in 0..7 {
            assert_eq!(ra.get(i).count(), 0, "Regime {} should have 0 events", i);
        }
    }
}
