//! Time utilities for intraday analysis.
//!
//! DST-aware regime classification, canonical grid resampling,
//! and US equity market calendar.

pub mod regime;
pub mod resampler;

pub use regime::{infer_day_params, infer_utc_offset, time_regime, utc_offset_for_date, N_REGIMES};
pub use resampler::{resample_to_grid, AggMode, ResampledBins};
