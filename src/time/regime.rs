//! 7-regime intraday classification matching MBO-LOB-analyzer's `time_utils.py`.
//!
//! | Value | Label | Time (ET) |
//! |-------|-------|-----------|
//! | 0 | pre-market | Before 9:30 |
//! | 1 | open-auction | 9:30 - 9:35 |
//! | 2 | morning | 9:35 - 10:30 |
//! | 3 | midday | 10:30 - 15:00 |
//! | 4 | afternoon | 15:00 - 15:55 |
//! | 5 | close-auction | 15:55 - 16:00 |
//! | 6 | post-market | After 16:00 |

use chrono::Datelike;

/// Number of intraday regimes.
pub const N_REGIMES: usize = 7;

const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i64 = 3600 * NS_PER_SECOND;

/// Regime labels for display.
pub const REGIME_LABELS: [&str; N_REGIMES] = [
    "pre-market",
    "open-auction",
    "morning",
    "midday",
    "afternoon",
    "close-auction",
    "post-market",
];

/// Classify a single UTC nanosecond timestamp into one of 7 intraday regimes.
///
/// # Arguments
/// * `ts_ns` — UTC nanoseconds since epoch
/// * `utc_offset_hours` — UTC offset for the trading day (-5 for EST, -4 for EDT)
pub fn time_regime(ts_ns: i64, utc_offset_hours: i32) -> u8 {
    let offset_ns = (utc_offset_hours as i64) * NS_PER_HOUR;
    let local_ns =
        ((ts_ns + offset_ns) % (24 * NS_PER_HOUR) + 24 * NS_PER_HOUR) % (24 * NS_PER_HOUR);

    let open = 9 * NS_PER_HOUR + 30 * NS_PER_MINUTE; // 09:30
    let open_end = 9 * NS_PER_HOUR + 35 * NS_PER_MINUTE; // 09:35
    let morning_end = 10 * NS_PER_HOUR + 30 * NS_PER_MINUTE; // 10:30
    let midday_end = 15 * NS_PER_HOUR; // 15:00
    let afternoon_end = 15 * NS_PER_HOUR + 55 * NS_PER_MINUTE; // 15:55
    let close = 16 * NS_PER_HOUR; // 16:00

    if local_ns < open {
        0 // pre-market
    } else if local_ns < open_end {
        1 // open-auction
    } else if local_ns < morning_end {
        2 // morning
    } else if local_ns < midday_end {
        3 // midday
    } else if local_ns < afternoon_end {
        4 // afternoon
    } else if local_ns < close {
        5 // close-auction
    } else {
        6 // post-market
    }
}

/// Compute UTC offset for a given date using US Eastern DST rules.
///
/// DST starts the 2nd Sunday in March, ends the 1st Sunday in November.
/// Returns -4 (EDT) or -5 (EST).
pub fn utc_offset_for_date(year: i32, month: u32, day: u32) -> i32 {
    let dst_start = nth_weekday_of_month(year, 3, 0, 2); // 2nd Sunday in March
    let dst_end = nth_weekday_of_month(year, 11, 0, 1); // 1st Sunday in November

    let ordinal = day_of_year(year, month, day);
    let dst_start_ordinal = day_of_year(year, 3, dst_start);
    let dst_end_ordinal = day_of_year(year, 11, dst_end);

    if ordinal >= dst_start_ordinal && ordinal < dst_end_ordinal {
        -4 // EDT
    } else {
        -5 // EST
    }
}

/// Day of year (1-indexed).
fn day_of_year(year: i32, month: u32, day: u32) -> u32 {
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_months: [u32; 12] = [
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut doy = day;
    for &days in days_in_months.iter().take(month as usize - 1) {
        doy += days;
    }
    doy
}

/// Find the nth occurrence of a weekday in a month.
/// weekday: 0=Sunday, 1=Monday, ..., 6=Saturday
/// n: 1-indexed (1=first, 2=second, etc.)
fn nth_weekday_of_month(year: i32, month: u32, weekday: u32, n: u32) -> u32 {
    // Tomohiko Sakamoto's algorithm for day-of-week
    let t = [0u32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year } as u32;
    let dow_first = (y + y / 4 - y / 100 + y / 400 + t[(month - 1) as usize] + 1) % 7;

    let offset = (7 + weekday - dow_first) % 7;
    1 + offset + (n - 1) * 7
}

/// Infer UTC offset and day epoch from a slice of timestamps.
///
/// Uses the first timestamp to determine the date and DST offset.
/// Returns the UTC offset in hours (-5 for EST, -4 for EDT).
pub fn infer_utc_offset(timestamps: &[i64]) -> i32 {
    if timestamps.is_empty() {
        return -5;
    }
    let ts = timestamps[0];
    let secs = ts / NS_PER_SECOND;
    let days = secs / 86400;
    let date = chrono::NaiveDate::from_num_days_from_ce_opt((days + 719163) as i32);
    if let Some(d) = date {
        utc_offset_for_date(d.year(), d.month(), d.day())
    } else {
        -5
    }
}

/// Infer both UTC offset and day epoch nanoseconds from timestamps.
pub fn infer_day_params(timestamps: &[i64]) -> (i32, i64) {
    let offset = infer_utc_offset(timestamps);
    let epoch = if !timestamps.is_empty() {
        let secs = timestamps[0] / NS_PER_SECOND;
        (secs / 86400) * 86400 * NS_PER_SECOND
    } else {
        0
    };
    (offset, epoch)
}

/// Compute midnight local time (specified by utc_offset) as UTC nanoseconds since epoch.
///
/// # Panics
///
/// Panics if the date is invalid (e.g., month > 12, day > 31, Feb 30).
pub fn day_epoch_ns(year: i32, month: u32, day: u32, utc_offset_hours: i32) -> i64 {
    let days_since_epoch = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .unwrap()
        .signed_duration_since(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
        .num_days();
    let midnight_utc_ns = days_since_epoch * 24 * 3600 * NS_PER_SECOND;
    midnight_utc_ns - (utc_offset_hours as i64) * NS_PER_HOUR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_classification() {
        let utc_offset = -5; // EST
        let base = 14 * NS_PER_HOUR; // 14:00 UTC = 9:00 ET (pre-market)

        assert_eq!(time_regime(base, utc_offset), 0); // 9:00 → pre-market

        let open = base + 30 * NS_PER_MINUTE; // 9:30 ET
        assert_eq!(time_regime(open, utc_offset), 1); // open-auction

        let morning = base + 40 * NS_PER_MINUTE; // 9:40 ET
        assert_eq!(time_regime(morning, utc_offset), 2); // morning

        let midday = base + 2 * NS_PER_HOUR; // 11:00 ET
        assert_eq!(time_regime(midday, utc_offset), 3); // midday

        let afternoon = 20 * NS_PER_HOUR; // 15:00 ET = 20:00 UTC
        assert_eq!(time_regime(afternoon, utc_offset), 4); // afternoon

        let close_auction = 20 * NS_PER_HOUR + 55 * NS_PER_MINUTE;
        assert_eq!(time_regime(close_auction, utc_offset), 5); // close-auction

        let post = 21 * NS_PER_HOUR + 1; // 16:00 ET + 1ns
        assert_eq!(time_regime(post, utc_offset), 6); // post-market
    }

    #[test]
    fn test_dst_offset() {
        assert_eq!(utc_offset_for_date(2025, 1, 15), -5); // January = EST
        assert_eq!(utc_offset_for_date(2025, 7, 15), -4); // July = EDT
        assert_eq!(utc_offset_for_date(2025, 12, 15), -5); // December = EST
    }

    #[test]
    fn test_regime_labels_count() {
        assert_eq!(REGIME_LABELS.len(), N_REGIMES);
    }

    // =========================================================================
    // Formula-correctness tests for untested functions
    // =========================================================================

    #[test]
    fn test_infer_utc_offset_est() {
        // 2025-01-15 12:00:00 UTC = January (EST, -5)
        // Jan 15 2025: days since epoch = 20103
        // 20103 * 86400 + 12*3600 = 1736942400 seconds = 1736942400_000_000_000 ns
        let ts = 1736942400_000_000_000i64;
        assert_eq!(
            infer_utc_offset(&[ts]),
            -5,
            "January 2025 should be EST (-5)"
        );
    }

    #[test]
    fn test_infer_utc_offset_edt() {
        // 2025-07-15 12:00:00 UTC = July (EDT, -4)
        // Jul 15 2025: days since epoch = 20284
        // 20284 * 86400 + 12*3600 = 1752580800 seconds
        let ts = 1752580800_000_000_000i64;
        assert_eq!(infer_utc_offset(&[ts]), -4, "July 2025 should be EDT (-4)");
    }

    #[test]
    fn test_infer_utc_offset_empty() {
        assert_eq!(
            infer_utc_offset(&[]),
            -5,
            "Empty should default to EST (-5)"
        );
    }

    #[test]
    fn test_infer_day_params_known_date() {
        // 2025-02-03 14:30:00 UTC
        // Feb 3: days since epoch = 20122
        // ts = 20122 * 86400 + 14*3600 + 30*60 = 1738588200 seconds
        let ts = 1738588200_000_000_000i64;
        let (offset, epoch) = infer_day_params(&[ts]);
        assert_eq!(offset, -5, "Feb 2025 = EST");
        // epoch should be midnight UTC of that day = 20122 * 86400 * 1e9
        let expected_epoch = 20122i64 * 86400 * NS_PER_SECOND;
        assert_eq!(epoch, expected_epoch, "day_epoch should be midnight UTC");
    }

    #[test]
    fn test_day_epoch_ns_known_value() {
        // day_epoch_ns(2025, 2, 3, -5) = midnight EST as UTC ns
        // 2025-02-03 00:00 EST = 2025-02-03 05:00 UTC
        // days from 1970-01-01 to 2025-02-03 = 20122
        // midnight UTC = 20122 * 86400 = 1738540800 seconds
        // + 5 hours = 1738540800 + 18000 = 1738558800 seconds
        let result = day_epoch_ns(2025, 2, 3, -5);
        let expected = 1738558800_000_000_000i64;
        assert_eq!(
            result, expected,
            "day_epoch_ns(2025,2,3,-5) should be midnight EST as UTC ns"
        );
    }

    #[test]
    fn test_dst_boundary_march_2025() {
        // 2025: DST starts 2nd Sunday in March = March 9
        // March 8 should be EST (-5), March 9 should be EDT (-4)
        assert_eq!(
            utc_offset_for_date(2025, 3, 8),
            -5,
            "March 8 2025 should be EST"
        );
        assert_eq!(
            utc_offset_for_date(2025, 3, 9),
            -4,
            "March 9 2025 should be EDT (DST starts)"
        );
    }

    #[test]
    fn test_leap_year_day_of_year() {
        // 2024 is a leap year. March 1 = day 31(Jan) + 29(Feb) + 1 = 61
        assert_eq!(day_of_year(2024, 3, 1), 61, "2024 (leap): Mar 1 = day 61");
        // 2025 is NOT a leap year. March 1 = 31 + 28 + 1 = 60
        assert_eq!(
            day_of_year(2025, 3, 1),
            60,
            "2025 (non-leap): Mar 1 = day 60"
        );
    }
}
