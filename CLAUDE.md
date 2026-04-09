# CLAUDE.md

Guidance for Claude Code when working with hft-statistics.

## Build Commands

```bash
cargo build          # Build
cargo test           # Run all 94 tests
cargo clippy         # Lint (0 warnings policy)
cargo fmt            # Format
cargo doc --no-deps  # Generate API docs
```

## Module Structure

```
src/
  lib.rs                          — Root: pub mod statistics, pub mod time
  statistics/
    mod.rs                        — Re-exports all 9 types + 2 free functions
    welford.rs                    — WelfordAccumulator (streaming mean/variance, Welford 1962)
    streaming_dist.rs             — StreamingDistribution (Welford + reservoir composite)
    reservoir.rs                  — ReservoirSampler (Algorithm R, Vitter 1985)
    vpin.rs                       — VpinComputer (VPIN with BVC, Easley et al. 2012)
    acf.rs                        — AcfComputer (autocorrelation, ring buffer)
    intraday_curve.rs             — IntradayCurveAccumulator (per-bin Welford)
    intraday_correlation.rs       — IntradayCorrelationAccumulator (per-bin Pearson)
    regime_accumulator.rs         — RegimeAccumulator (per-regime Welford, 7 regimes)
    transition_matrix.rs          — TransitionMatrix<N> (NxN stack-allocated counter)
  time/
    mod.rs                        — Re-exports regime + resampler
    regime.rs                     — 7-regime classification, DST, UTC offset inference
    resampler.rs                  — Canonical-grid resampler (Sum/Mean/Last/Count)
```

## Key Types

- `WelfordAccumulator` — Most widely used. Streaming mean/variance with NaN/Inf skip, merge support, exact min/max.
- `StreamingDistribution` — Wraps Welford + ReservoirSampler for percentiles + skewness/kurtosis.
- `VpinComputer` — VPIN via Bulk Volume Classification. Feed trades, get toxicity measure.
- `time_regime(ts_ns, offset) -> u8` — 7 regimes: pre-market(0), open-auction(1), morning(2), midday(3), afternoon(4), close-auction(5), post-market(6).
- `utc_offset_for_date(year, month, day) -> i32` — Returns -4 (EDT) or -5 (EST).

## Critical Constraints

- All accumulators skip NaN/Inf inputs (never corrupt state)
- Welford uses population variance (not sample) by default; use `sample_variance()` for Bessel's correction
- ReservoirSampler deserialization is O(total_seen) — replays RNG for determinism
- TransitionMatrix uses const generics; serde uses custom impl via Vec<Vec<u64>>
- RegimeAccumulator serde handles empty Welford accumulators (Infinity min/max → null in JSON)
- Time regime boundaries are hardcoded for US Eastern time
- DST offset uses full-day granularity (switches at midnight, not 2 AM)

## Dependencies

- `serde` + `serde_json` — Serialization (all types except noted)
- `rand 0.8` — Reservoir sampling RNG
- `chrono 0.4` (minimal) — Date computation for DST only

## Test Coverage

94 tests across 11 source files. All formulas validated against hand-computed golden values.
