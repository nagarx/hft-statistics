//! Fixed-size transition matrix for event transition probabilities.
//!
//! Counts transitions between N states and computes P(next | current).
//! Used by LifecycleTracker for the 4x4 action transition matrix:
//! P(Cancel | Add), P(Trade | Modify), etc.
//!
//! Zero-alloc: stored as `[[u64; N]; N]` on the stack.
//!
//! ## Serialization
//!
//! Custom `Serialize`/`Deserialize` implementations convert the const-generic
//! `[[u64; N]; N]` to/from `Vec<Vec<u64>>` (serde's derive does not support
//! const-generic array sizes). The `total` field is recomputed on deserialize.

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Fixed-size NxN transition counter.
///
/// `matrix[from][to]` counts transitions from state `from` to state `to`.
#[derive(Debug, Clone)]
pub struct TransitionMatrix<const N: usize> {
    matrix: [[u64; N]; N],
    total: u64,
}

impl<const N: usize> Serialize for TransitionMatrix<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("TransitionMatrix", 2)?;
        state.serialize_field("matrix", &self.count_matrix())?;
        state.serialize_field("total", &self.total)?;
        state.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for TransitionMatrix<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            matrix: Vec<Vec<u64>>,
            #[allow(dead_code)]
            total: u64,
        }
        let helper = Helper::deserialize(deserializer)?;
        let mut matrix = [[0u64; N]; N];
        let mut total = 0u64;
        for (i, row) in helper.matrix.iter().enumerate().take(N) {
            for (j, &val) in row.iter().enumerate().take(N) {
                matrix[i][j] = val;
                total += val;
            }
        }
        Ok(Self { matrix, total })
    }
}

impl<const N: usize> TransitionMatrix<N> {
    pub fn new() -> Self {
        Self {
            matrix: [[0u64; N]; N],
            total: 0,
        }
    }

    /// Record a transition from state `from` to state `to`.
    #[inline]
    pub fn record(&mut self, from: usize, to: usize) {
        if from < N && to < N {
            self.matrix[from][to] += 1;
            self.total += 1;
        }
    }

    /// Get the raw count for a transition.
    pub fn count(&self, from: usize, to: usize) -> u64 {
        if from < N && to < N {
            self.matrix[from][to]
        } else {
            0
        }
    }

    /// Get the row-normalized probability P(to | from).
    pub fn probability(&self, from: usize, to: usize) -> f64 {
        if from >= N || to >= N {
            return 0.0;
        }
        let row_total: u64 = self.matrix[from].iter().sum();
        if row_total == 0 {
            0.0
        } else {
            self.matrix[from][to] as f64 / row_total as f64
        }
    }

    /// Get the total number of transitions recorded.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Produce the full probability matrix as a 2D JSON array.
    pub fn probability_matrix(&self) -> Vec<Vec<f64>> {
        (0..N)
            .map(|from| {
                let row_total: u64 = self.matrix[from].iter().sum();
                (0..N)
                    .map(|to| {
                        if row_total == 0 {
                            0.0
                        } else {
                            self.matrix[from][to] as f64 / row_total as f64
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Produce the raw count matrix as a 2D JSON array.
    pub fn count_matrix(&self) -> Vec<Vec<u64>> {
        self.matrix.iter().map(|row| row.to_vec()).collect()
    }

    pub fn reset(&mut self) {
        self.matrix = [[0u64; N]; N];
        self.total = 0;
    }
}

impl<const N: usize> Default for TransitionMatrix<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_counting() {
        let mut tm = TransitionMatrix::<4>::new();
        tm.record(0, 1); // Add → Cancel
        tm.record(0, 1); // Add → Cancel
        tm.record(0, 3); // Add → Trade
        assert_eq!(tm.count(0, 1), 2);
        assert_eq!(tm.count(0, 3), 1);
        assert_eq!(tm.total(), 3);
    }

    #[test]
    fn test_probability() {
        let mut tm = TransitionMatrix::<4>::new();
        tm.record(0, 1);
        tm.record(0, 1);
        tm.record(0, 3);

        assert!((tm.probability(0, 1) - 2.0 / 3.0).abs() < 1e-10);
        assert!((tm.probability(0, 3) - 1.0 / 3.0).abs() < 1e-10);
        assert!((tm.probability(0, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_row_probability() {
        let tm = TransitionMatrix::<4>::new();
        assert_eq!(tm.probability(0, 1), 0.0);
    }

    #[test]
    fn test_out_of_bounds_ignored() {
        let mut tm = TransitionMatrix::<4>::new();
        tm.record(5, 0);
        tm.record(0, 5);
        assert_eq!(tm.total(), 0);
    }

    #[test]
    fn test_probability_matrix() {
        let mut tm = TransitionMatrix::<3>::new();
        tm.record(0, 1);
        tm.record(0, 2);
        tm.record(1, 0);

        let pm = tm.probability_matrix();
        assert!((pm[0][1] - 0.5).abs() < 1e-10);
        assert!((pm[0][2] - 0.5).abs() < 1e-10);
        assert!((pm[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut tm = TransitionMatrix::<4>::new();
        tm.record(0, 1);
        tm.reset();
        assert_eq!(tm.total(), 0);
        assert_eq!(tm.count(0, 1), 0);
    }

    #[test]
    fn test_row_probabilities_sum_to_one() {
        let mut tm = TransitionMatrix::<4>::new();
        tm.record(0, 0);
        tm.record(0, 1);
        tm.record(0, 2);
        tm.record(0, 3);
        tm.record(1, 0);
        tm.record(1, 1);
        tm.record(2, 3);

        let pm = tm.probability_matrix();
        for (i, row) in pm.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "Row {} probabilities sum to {}, expected 1.0",
                    i,
                    sum
                );
            }
        }
    }
}
