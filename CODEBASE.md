# CODEBASE.md — hft-statistics

> **Purpose**: Deep technical reference for every module, type, formula, and invariant in the `hft-statistics` crate. An LLM coder (or human) should be able to understand the exact mathematical computation, data flow, edge case handling, and serialization strategy for every public type without reading the source code.
>
> **Version**: 0.1.0 | **Tests**: 94 | **Lines**: ~3,100 | **Clippy**: 0 warnings

---

## 1. Crate Architecture

```
hft-statistics (leaf crate, zero domain coupling)
├── statistics/          9 streaming accumulators + 2 free functions
│   ├── welford          Streaming mean/variance (Welford 1962)
│   ├── streaming_dist   Composite: Welford + reservoir + skewness/kurtosis
│   ├── reservoir        Bounded-memory sampling (Vitter 1985, Algorithm R)
│   ├── vpin             Volume-Synchronized PIN (Easley et al. 2012)
│   ├── acf              Autocorrelation function (ring buffer)
│   ├── intraday_curve   Per-bin intraday statistics (wall-clock bins)
│   ├── intraday_correlation  Per-bin Pearson correlation
│   ├── regime_accumulator    Per-regime Welford (7 US equity regimes)
│   └── transition_matrix     NxN stack-allocated transition counter
└── time/                Market time utilities
    ├── regime           7-regime classification, DST, UTC offset inference
    └── resampler        Canonical-grid time resampler (Sum/Mean/Last/Count)
```

**Dependencies**: `serde` (serialization), `serde_json` (JSON output), `rand 0.8` (reservoir RNG), `chrono 0.4` (DST date computation, minimal features).

**Design invariants**:
- All accumulators skip NaN/Inf inputs (never corrupt internal state)
- All accumulators have `reset()` (clears data, preserves structural parameters)
- All public types are `Send` (safe for parallel use in separate threads)
- All types with state have `Serialize`/`Deserialize` for checkpoint/restore
- Time handling uses nanosecond precision throughout (`i64` UTC nanoseconds since epoch)

---

## 2. Statistics Module

### 2.1 WelfordAccumulator (`statistics/welford.rs`)

Numerically stable streaming mean and variance using Welford's online algorithm.

**Reference**: Welford, B.P. (1962). "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419-420.

**Struct** (all fields private):
```rust
pub struct WelfordAccumulator {
    count: u64,            // number of finite values seen
    mean: f64,             // running mean
    m2: f64,               // sum of squared deviations from mean
    min: f64,              // exact minimum (INFINITY when empty)
    max: f64,              // exact maximum (NEG_INFINITY when empty)
}
```

**Serde**: `#[derive(Serialize, Deserialize)]` (derive). Note: empty accumulators serialize `min: null, max: null` in JSON because `f64::INFINITY` becomes `null`.

**Core formulas**:

| Operation | Formula | Code |
|-----------|---------|------|
| Update (per value) | `mean_new = mean + (x - mean) / n` | `self.mean += delta / self.count as f64` |
| M2 update | `M2 += (x - mean_old)(x - mean_new)` | `self.m2 += delta * delta2` |
| Population variance | `σ² = M2 / n` | `self.m2 / self.count as f64` |
| Sample variance | `s² = M2 / (n-1)` | `self.m2 / (self.count - 1) as f64` |
| Parallel merge | Chan-Perlman formula | See below |

**Merge formula** (for combining two accumulators computed in parallel):
```
n_combined = n_a + n_b
δ = mean_b - mean_a
mean_combined = mean_a + δ × (n_b / n_combined)
M2_combined = M2_a + M2_b + δ² × (n_a × n_b / n_combined)
min_combined = min(min_a, min_b)
max_combined = max(max_a, max_b)
```

**Edge cases**:
- `update(NaN)` or `update(±Inf)` → skipped silently, count unchanged
- `mean()` when count=0 → `0.0`
- `variance()` when count < 2 → `0.0`
- `min()`/`max()` when count=0 → `f64::NAN`
- `merge()` with empty other → no-op; merge into empty → clones source

**Tests**: 12 — empty, single value, known values (mean=5.0, var=4.0 for [2,4,4,4,5,5,7,9]), NaN skip, Inf skip, merge (including variance correctness), reset, batch.

---

### 2.2 VpinComputer (`statistics/vpin.rs`)

Volume-Synchronized Probability of Informed Trading using Bulk Volume Classification.

**References**:
- Easley, López de Prado, O'Hara (2012). "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies*, 25(5).
- Easley et al. (2019). "Microstructure in the Machine Age." *RFS*.
- Abramowitz & Stegun (1964), formula 7.1.26 (error function approximation).

**Algorithm**:
1. Aggregate trades into equal-volume bars (`V_bar` shares each)
2. Classify each bar's buy/sell volume using Bulk Volume Classification (BVC)
3. VPIN = rolling mean of `|V_buy - V_sell| / V_bar` over `n` bars

**BVC formula**:
```
V_buy = V × Φ((P_close - P_open) / σ_P)
V_sell = V × (1 - Φ(...))
```
where `Φ` = standard normal CDF, `σ_P` = rolling sample std of bar returns.

**Free functions**:
- `phi(x: f64) -> f64` — Standard normal CDF: `Φ(x) = 0.5 × (1 + erf(x/√2))`
- `erf_approx(x: f64) -> f64` — Error function approximation (Abramowitz & Stegun 7.1.26), max error < 1.5e-7. Coefficients: `a₁=0.254829592, a₂=-0.284496736, a₃=1.421413741, a₄=-1.453152027, a₅=1.061405429, p=0.3275911`.

**Constants**: `DEFAULT_BAR_VOLUME = 5000`, `DEFAULT_WINDOW_BARS = 50`.

**Serde**: `#[derive(Serialize, Deserialize)]` on both `VpinComputer` and private `VolumeBar`.

**Key edge cases**:
- `add_trade()` with price ≤ 0, volume = 0, or non-finite price → skipped
- `current_vpin()` → `None` until `window_bars` completed
- `new(0, 0)` → falls back to defaults (5000 bar volume, 50 window bars)
- Bar return sigma floored at `1e-8`; falls back to `1e-6` with < 2 bar returns

**Tests**: 7 — phi golden values pinned to scipy at 1.5e-7 tolerance, phi edge cases (±∞, NaN), VPIN window fill, VPIN range [0,1], directional flow produces high VPIN, reset.

---

### 2.3 ReservoirSampler (`statistics/reservoir.rs`)

Bounded-memory random sampling with deterministic seeding and exact min/max tracking.

**Reference**: Vitter, J.S. (1985). "Random Sampling with a Reservoir." *ACM Transactions on Mathematical Software*, 11(1).

**Algorithm R** (per value):
```
if reservoir not full:
    reservoir.push(value)
else:
    j = random(0..=total_seen)
    if j < capacity:
        reservoir[j] = value
```

**Struct** (fields private):
```rust
pub struct ReservoirSampler {
    capacity: usize,
    reservoir: Vec<f64>,
    total_seen: u64,
    true_min: f64,          // exact minimum across ALL values (not just reservoir)
    true_max: f64,          // exact maximum across ALL values
    rng: StdRng,            // deterministic RNG (seeded)
    seed: u64,
}
```

**Serde**: CUSTOM. Serializes 6 fields (omits `rng`). Deserialization replays the RNG from `seed` for `total_seen` iterations via `reconstruct_rng()` to restore deterministic state.

**Performance note**: Deserialization is O(total_seen). For streams with millions+ of values, this may take seconds. The RNG must be replayed exactly to maintain deterministic behavior across serialize/deserialize cycles.

**Percentile computation** (linear interpolation):
```
idx = (p / 100) × (n - 1)
result = sorted[floor(idx)] × (1 - frac) + sorted[ceil(idx)] × frac
```

**Edge cases**:
- NaN/Inf → skipped, does not increment `total_seen`
- `true_min()`/`true_max()` with no values → `f64::NAN`
- `percentile()` with empty reservoir → `f64::NAN`
- Default seed: `42`

**Tests**: 7 — underfilled, full, exact min/max, NaN skip, deterministic seed, percentile, reset.

---

### 2.4 StreamingDistribution (`statistics/streaming_dist.rs`)

Composite streaming distribution tracker combining Welford (exact mean/variance) + Reservoir (approximate percentiles, skewness, kurtosis).

**Struct** (fields private):
```rust
pub struct StreamingDistribution {
    welford: WelfordAccumulator,
    reservoir: ReservoirSampler,
}
```

**Serde**: `#[derive(Serialize, Deserialize)]` (delegates to component types).

**Skewness** (computed from reservoir sample, population moment):
```
m₃ = (1/n) Σ(xᵢ - mean)³
skewness = m₃ / σ³
```
Returns `NaN` if < 3 samples. Returns `0.0` if σ < 1e-15 (constant data).

**Excess kurtosis** (computed from reservoir sample):
```
m₄ = (1/n) Σ(xᵢ - mean)⁴
kurtosis = m₄ / σ⁴ - 3
```
Returns `NaN` if < 4 samples. Returns `0.0` if σ² < 1e-15.

**Note**: Skewness and kurtosis are computed from the reservoir *sample*, not from streaming moments. For streams larger than the reservoir capacity, these are approximate. This is a deliberate design choice — exact streaming higher moments require careful numerical handling.

**`summary()` output** (JSON):
```json
{
    "count": u64,
    "mean": f64,
    "std": f64,
    "min": f64,
    "max": f64,
    "skewness": f64,
    "kurtosis": f64,
    "percentiles": { "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99" }
}
```

**Tests**: 15 — count, mean, min/max, summary keys, skewness (symmetric=0, right-skewed=1.5 hand-computed, constant=0, <3→NaN), kurtosis (uniform, peaked, constant, <4→NaN), batch=sequential, sorted, sample size.

---

### 2.5 AcfComputer (`statistics/acf.rs`)

Streaming autocorrelation function using a fixed-size ring buffer.

**Formula**:
```
ACF(k) = (1/N) × Σ_{t=0}^{N-k-1} (xₜ - mean)(x_{t+k} - mean) / var(x)
```
Computed for lags `k = 1, 2, ..., max_lag`.

**Struct** (ring buffer, fields private):
```rust
pub struct AcfComputer {
    buffer: Vec<f64>,        // ring buffer
    capacity: usize,
    write_pos: usize,
    filled: bool,
    max_lag: usize,
}
```

**Serde**: `#[derive(Serialize, Deserialize)]` (derive).

**Construction**: `assert!(capacity > max_lag)` — panics if capacity is too small. Default: 10,000 capacity, 20 lags.

**Computation**: O(N × max_lag) per `compute()` call. Allocates a linearized copy of the ring buffer on each call.

**Edge cases**:
- Insufficient data (n < max_lag + 2) → returns `vec![f64::NAN; max_lag]`
- Constant data (variance < 1e-15) → returns `vec![0.0; max_lag]`
- NaN/Inf → skipped

**Tests**: 6 — constant (ACF=0), alternating (lag-1≈-1, lag-2≈+1), insufficient data, ring wrap, NaN skip, exact alternating value (ACF(1)=-199/200).

---

### 2.6 IntradayCurveAccumulator (`statistics/intraday_curve.rs`)

Per-bin intraday statistics using canonical wall-clock bins. Each bin contains a WelfordAccumulator for streaming mean/std.

**Key design**: Bins are defined by wall-clock time boundaries, NOT by data arrival. This fixes a common audit issue where bins start at the first data point rather than at canonical market boundaries.

**Bin assignment**:
```
local_ns = ((ts_ns + offset_ns) % (24h)) mod (24h)    // double mod for negative offsets
bin_index = (local_ns - open_ns) / bin_width_ns
```

Events outside `[open, close)` are silently dropped.

**Standard constructor**: `new_rth_1min()` → 390 bins, 09:30-16:00 ET (6.5 hours = 390 minutes).

**Output type** (`BinStats`):
```rust
pub struct BinStats {
    pub bin_index: usize,
    pub minutes_since_open: f64,
    pub mean: f64,
    pub std: f64,
    pub count: u64,
}
```

**Tests**: 6 — 390 bins, bin assignment, pre/post market drop, cross-day accumulation, minute labels.

---

### 2.7 IntradayCorrelationAccumulator (`statistics/intraday_correlation.rs`)

Per-bin streaming Pearson correlation for two variables across intraday bins.

**Pearson correlation formula** (single-pass):
```
r = cov(X,Y) / (σ_X × σ_Y)

where:
  cov = E[XY] - E[X]E[Y] = sum_xy/n - (sum_x/n)(sum_y/n)
  σ²_X = E[X²] - E[X]² = sum_x2/n - (sum_x/n)²
  σ²_Y = E[Y²] - E[Y]² = sum_y2/n - (sum_y/n)²
```

**Known limitation**: The single-pass `E[X²] - E[X]²` formula is numerically unstable for large values with small variance (catastrophic cancellation). For market data (prices ~$130, returns ~1e-4), this is acceptable. For extreme values, consider Welford-style two-variable accumulation.

**Minimum samples**: Returns `NaN` for count < 3. Returns `NaN` if denominator < 1e-15.

**Output type** (`CorrelationBinStats`):
```rust
pub struct CorrelationBinStats {
    pub bin_index: usize,
    pub minutes_since_open: f64,
    pub pearson_r: f64,
    pub count: u64,
}
```

**Tests**: 11 — 390 bins, perfect ±1 correlation, insufficient data, pre/post market, non-finite, cross-day, bin routing, labels, known value (hand-computed r=0.7746).

---

### 2.8 RegimeAccumulator (`statistics/regime_accumulator.rs`)

Maintains one `WelfordAccumulator` per intraday regime (7 regimes for US equities).

**Struct**: `accumulators: [WelfordAccumulator; 7]` (7 = `N_REGIMES`).

**Regime indices**: 0=pre-market, 1=open-auction, 2=morning, 3=midday, 4=afternoon, 5=close-auction, 6=post-market.

**`add()` vs `get()` behavior**:
- `add(regime, value)`: silently drops if regime >= 7 (defensive, called in hot paths)
- `get(regime)`: **panics** if regime >= 7 (assertive, called in analysis code)

**Serde**: derive `Serialize` + CUSTOM `Deserialize`. Empty WelfordAccumulators have `min: INFINITY, max: NEG_INFINITY` which serialize as `null` in JSON. The custom deserializer checks `count > 0` before attempting to parse; count=0 entries are replaced with fresh defaults.

**`finalize()` output**: JSON object keyed by regime label, containing mean/std/count/min/max per regime. Only regimes with count > 0 appear.

**Tests**: 4 — regime separation, finalize, serde round-trip, out-of-range dropped.

---

### 2.9 TransitionMatrix<N> (`statistics/transition_matrix.rs`)

Fixed-size NxN transition counter using const generics. Zero heap allocation (stack-based `[[u64; N]; N]`).

**Usage**: P(next_state | current_state) for Markov chain analysis. Primary consumer: LifecycleTracker in mbo-statistical-profiler (4×4 matrix: Add→Cancel, Add→Trade, Modify→Cancel, etc.).

**Probability formula**:
```
P(to | from) = matrix[from][to] / Σⱼ matrix[from][j]
```
Empty rows (no transitions from a state) return `P = 0.0`.

**Serde**: CUSTOM `Serialize` + `Deserialize`. Cannot use derive because serde does not support `[T; N]` with const generic `N`. Serializes as `{"matrix": [[...], ...], "total": u64}`. On deserialize, `total` is recomputed from the matrix (ignores serialized value for consistency). Dimension mismatches (deserialized matrix smaller/larger than N) are handled via `take(N)` clipping.

**Tests**: 7 — counting, probability, empty row, out-of-bounds, probability matrix, reset, row sums to 1.

---

## 3. Time Module

### 3.1 Market Time Regime (`time/regime.rs`)

7-regime intraday classification for US equities with DST-aware UTC offset computation.

**Regime boundaries** (local ET time):

| Regime | Index | Start | End |
|--------|-------|-------|-----|
| pre-market | 0 | 00:00 | 09:30 |
| open-auction | 1 | 09:30 | 09:35 |
| morning | 2 | 09:35 | 10:30 |
| midday | 3 | 10:30 | 15:00 |
| afternoon | 4 | 15:00 | 15:55 |
| close-auction | 5 | 15:55 | 16:00 |
| post-market | 6 | 16:00 | 24:00 |

**Local time computation**:
```
offset_ns = utc_offset_hours × 3,600,000,000,000
local_ns = ((ts_ns + offset_ns) mod 24h + 24h) mod 24h
```
The double-modulo ensures non-negative results even for negative UTC offsets.

**DST rules** (US Eastern):
- DST starts: 2nd Sunday in March (→ EDT, UTC-4)
- DST ends: 1st Sunday in November (→ EST, UTC-5)
- Day-of-week uses Tomohiko Sakamoto's algorithm

**`utc_offset_for_date(year, month, day)`**: Returns -4 (EDT) or -5 (EST). Uses full-day granularity — the offset switches at midnight, not at 2:00 AM. For market hours (09:30-16:00), this is always correct since DST transitions occur at 2 AM.

**`infer_utc_offset(timestamps)`**: Infers offset from a slice of UTC nanosecond timestamps. Extracts the date from the first timestamp, computes offset. Empty input defaults to -5 (EST).

**`day_epoch_ns(year, month, day, utc_offset)`**: Computes midnight local time as UTC nanoseconds. **Panics on invalid dates** (e.g., month=13, Feb 30) via `NaiveDate::from_ymd_opt().unwrap()`.

**Magic number**: `719163` in `infer_utc_offset()` = days from CE epoch (0001-01-01) to Unix epoch (1970-01-01).

**Constants**: `N_REGIMES = 7`, `REGIME_LABELS = ["pre-market", "open-auction", "morning", "midday", "afternoon", "close-auction", "post-market"]`.

**Tests**: 10 — regime classification, DST offsets, label count, infer EST/EDT/empty, day params, epoch ns, DST boundary March 2025, leap year.

---

### 3.2 Canonical Grid Resampler (`time/resampler.rs`)

Time-based resampler for converting irregular event data into canonical wall-clock bins. Fixes the "data-dependent bin start" issue found in Python analysis code.

**Grid construction**:
```
open_utc = day_epoch + 9:30_local - offset
close_utc = day_epoch + 16:00_local - offset
n_bins = (close_utc - open_utc) / bin_width
```
For non-divisible durations, the last partial bin is dropped.

**Bin assignment**: `bin = (ts - open_utc) / bin_width`

**Aggregation modes** (`AggMode` enum):

| Mode | Formula | Empty bin |
|------|---------|-----------|
| `Sum` | Σ values in bin | 0.0 |
| `Mean` | Σ / count | NaN |
| `Last` | Last value written | NaN |
| `Count` | Number of values | 0.0 |

**Output** (`ResampledBins`):
```rust
pub struct ResampledBins {
    pub edges_ns: Vec<i64>,     // n_bins + 1 edge timestamps
    pub values: Vec<f64>,       // aggregated values per bin
    pub counts: Vec<u64>,       // raw counts per bin
    pub bin_width_ns: i64,
}
```

**Edge cases**:
- Events outside RTH [09:30, 16:00) ET → dropped
- Non-finite values → dropped
- Bin index ≥ n_bins → dropped

**Tests**: 9 — bin counts (5s=4680, 1min=390), sum/mean/last aggregation, pre-market excluded, different bins separate, empty bins NaN, grid edges.

---

## 4. Nanosecond Time Constants

Defined independently in 4 files (intraday_curve, intraday_correlation, regime, resampler):

| Constant | Value | Type |
|----------|-------|------|
| `NS_PER_SECOND` | 1,000,000,000 | `i64` |
| `NS_PER_MINUTE` | 60 × NS_PER_SECOND | `i64` |
| `NS_PER_HOUR` | 3,600 × NS_PER_SECOND | `i64` |
| `NS_PER_DAY` | 24 × NS_PER_HOUR | `i64` (intraday_correlation only) |

---

## 5. Serialization Strategy Summary

| Type | Strategy | Notes |
|------|----------|-------|
| `WelfordAccumulator` | derive | Empty: `min`/`max` → `null` in JSON |
| `VpinComputer` | derive | Private `VolumeBar` also derives |
| `ReservoirSampler` | custom | Omits `rng`; replays from `seed` on deser (O(total_seen)) |
| `StreamingDistribution` | derive | Delegates to Welford + Reservoir |
| `AcfComputer` | derive | Ring buffer serialized as Vec |
| `IntradayCurveAccumulator` | derive | Bins = Vec<WelfordAccumulator> |
| `IntradayCorrelationAccumulator` | derive | Bins = Vec<PearsonBin> |
| `RegimeAccumulator` | derive Ser + custom Deser | Skips count=0 entries (Infinity → null) |
| `TransitionMatrix<N>` | custom both | `[[u64; N]; N]` → `Vec<Vec<u64>>`; total recomputed |
| `ResampledBins` | none | Transient output type, not checkpointed |

---

## 6. Consumer Map

| Consumer | Types Used |
|----------|-----------|
| **feature-extractor-MBO-LOB** (hft-feature-core) | `time_regime`, `utc_offset_for_date` |
| **feature-extractor-MBO-LOB** (monolith signals) | `time_regime`, `utc_offset_for_date`, `N_REGIMES`, `REGIME_LABELS`, `day_epoch_ns`, `infer_utc_offset` |
| **mbo-statistical-profiler** (13 trackers) | ALL statistics types + ALL time types |
| **basic-quote-processor** | `phi`, `VpinComputer`, `WelfordAccumulator`, `utc_offset_for_date`, `time_regime` |

**Universally shared** (4 consumers): `time_regime`, `utc_offset_for_date`
**Unused across ecosystem**: `erf_approx` (only used internally by `phi`), `ReservoirSampler` (used only internally by `StreamingDistribution`)

---

## 7. Test Coverage

| Module | Tests | Key validations |
|--------|-------|-----------------|
| welford | 12 | Merge formula, NaN/Inf skip, known values (mean=5.0, var=4.0) |
| vpin | 7 | Phi pinned to scipy golden values (1.5e-7 tolerance), VPIN range [0,1] |
| reservoir | 7 | Algorithm R, exact min/max, deterministic seeding |
| streaming_dist | 15 | Hand-computed skewness (1.5), kurtosis (-1.224), batch=sequential |
| acf | 6 | Exact ACF(1)=-199/200 for alternating series |
| intraday_curve | 6 | 390-bin RTH, pre/post market drop, cross-day accumulation |
| intraday_correlation | 11 | Perfect ±1 r, known value (r=0.7746), non-finite drop |
| regime_accumulator | 4 | Regime separation, serde round-trip, out-of-range |
| transition_matrix | 7 | Row-sum=1 invariant, out-of-bounds, probability matrix |
| regime | 10 | DST boundary March 2025, leap year, infer EST/EDT |
| resampler | 9 | 5s=4680 bins, 1min=390 bins, all 4 agg modes |
| **Total** | **94** | |

---

## 8. Known Limitations

1. **ReservoirSampler deserialization is O(total_seen)** — replays RNG from seed for deterministic state. For billions of events, this can take seconds/minutes. Workaround: use small reservoir capacities for checkpoint-heavy workflows.

2. **Pearson correlation uses E[X²]-E[X]² formula** — numerically unstable for large values with small variance (catastrophic cancellation). Acceptable for HFT data (log-returns ~1e-4), but not for arbitrary inputs.

3. **`day_epoch_ns()` panics on invalid dates** — uses `.unwrap()` on `NaiveDate::from_ymd_opt()`. Pass valid dates only.

4. **DST offset is full-day granularity** — transitions at midnight, not 2:00 AM. Always correct for market hours (09:30-16:00).

5. **Time constants defined in 4 files independently** — `NS_PER_SECOND`, `NS_PER_MINUTE`, `NS_PER_HOUR` are duplicated across intraday_curve, intraday_correlation, regime, resampler. Values are identical but not shared from a single location.

6. **`ResampledBins` lacks serde** — transient output type, not designed for checkpoint/restore.

---

*Last updated: April 9, 2026. Covers hft-statistics v0.1.0 (94 tests, ~3,100 lines).*
