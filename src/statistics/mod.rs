//! Statistical primitives for streaming analysis.
//!
//! All primitives are designed for bounded-memory operation over billions of events.

pub mod acf;
pub mod intraday_correlation;
pub mod intraday_curve;
pub mod regime_accumulator;
pub mod reservoir;
pub mod streaming_dist;
pub mod transition_matrix;
pub mod vpin;
pub mod welford;

pub use acf::AcfComputer;
pub use intraday_correlation::IntradayCorrelationAccumulator;
pub use intraday_curve::IntradayCurveAccumulator;
pub use regime_accumulator::RegimeAccumulator;
pub use reservoir::ReservoirSampler;
pub use streaming_dist::StreamingDistribution;
pub use transition_matrix::TransitionMatrix;
pub use vpin::{erf_approx, phi, VpinComputer};
pub use welford::WelfordAccumulator;
