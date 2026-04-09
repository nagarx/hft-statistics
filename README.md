# hft-statistics

Shared streaming statistical primitives for the HFT pipeline. Zero domain coupling — this crate provides numerically stable accumulators, samplers, and time utilities used across multiple Rust modules.

## Modules

### `statistics` — Streaming Accumulators

| Type | Purpose | Reference |
|------|---------|-----------|
| `WelfordAccumulator` | Streaming mean/variance (O(1) per update, O(1) memory) | Welford (1962) |
| `StreamingDistribution` | Composite: Welford + reservoir + skewness/kurtosis | — |
| `ReservoirSampler` | Bounded-memory sampling with exact min/max | Vitter (1985), Algorithm R |
| `VpinComputer` | Volume-Synchronized Probability of Informed Trading | Easley, Lopez de Prado, O'Hara (2012) |
| `AcfComputer` | Streaming autocorrelation function (ring buffer) | — |
| `IntradayCurveAccumulator` | Per-bin intraday statistics (canonical wall-clock bins) | — |
| `IntradayCorrelationAccumulator` | Per-bin Pearson correlation for two variables | — |
| `RegimeAccumulator` | Per-regime Welford (7 US equity regimes) | — |
| `TransitionMatrix<N>` | Fixed-size NxN transition counter (stack-allocated) | — |

Free functions: `phi(x)` (standard normal CDF), `erf_approx(x)` (error function, max error < 1.5e-7).

### `time` — Market Time Utilities

| Function/Type | Purpose |
|--------------|---------|
| `time_regime(ts_ns, utc_offset)` | Classify UTC nanosecond timestamp into 7 intraday regimes |
| `utc_offset_for_date(year, month, day)` | DST-aware UTC offset (-4 EDT / -5 EST) |
| `infer_utc_offset(timestamps)` | Infer UTC offset from a set of timestamps |
| `infer_day_params(timestamps)` | Infer (utc_offset, day_epoch_ns) from timestamps |
| `day_epoch_ns(year, month, day, offset)` | Compute midnight local time as UTC nanoseconds |
| `resample_to_grid(...)` | Canonical-grid resampler with Sum/Mean/Last/Count modes |
| `N_REGIMES` / `REGIME_LABELS` | Constants for the 7-regime system |

## Consumers

| Repo | Usage |
|------|-------|
| `feature-extractor-MBO-LOB` | `time_regime`, `utc_offset_for_date` (via hft-feature-core signals + seasonality) |
| `mbo-statistical-profiler` | ALL types — 13 trackers use Welford, StreamingDist, ACF, regime, resampler |
| `basic-quote-processor` | `phi`, `VpinComputer`, `WelfordAccumulator`, `time_regime` |

## Usage

```toml
[dependencies]
hft-statistics = { git = "https://github.com/nagarx/hft-statistics.git" }
```

```rust
use hft_statistics::statistics::WelfordAccumulator;
use hft_statistics::time::time_regime;

let mut acc = WelfordAccumulator::new();
acc.update(100.5);
acc.update(101.2);
println!("mean={:.4}, std={:.4}", acc.mean(), acc.std());

let regime = time_regime(1738588200_000_000_000, -5); // EST
println!("regime={}", regime); // 3 = midday
```

## Build

```bash
cargo test          # 94 tests
cargo clippy        # 0 warnings
cargo doc --no-deps # API documentation
```

## Known Limitations

- `ReservoirSampler` deserialization replays RNG from seed (O(total_seen)). Use small capacities for checkpoint-heavy workflows.
- `IntradayCorrelationAccumulator` uses single-pass Pearson formula (numerically unstable for large values with small variance).
- `day_epoch_ns()` panics on invalid dates (month > 12, day > 31).
- Time regime and DST rules are hardcoded for US Eastern time (NYSE/NASDAQ).

## License

Proprietary. All rights reserved.
