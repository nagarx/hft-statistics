//! Shared streaming statistical primitives and time utilities for HFT pipeline.
//!
//! This is a leaf crate with zero domain dependencies — no LOB, MBO, or options types.
//! It provides bounded-memory streaming accumulators and DST-aware time utilities
//! reusable by any profiler (equity, options, futures, etc.).
//!
//! # Modules
//!
//! - `statistics` — Welford, reservoir sampling, streaming distributions, intraday curves,
//!   ACF, regime accumulator, transition matrix, intraday correlation
//! - `time` — DST-aware regime classification, canonical-grid resampling, US equity calendar

pub mod statistics;
pub mod time;
