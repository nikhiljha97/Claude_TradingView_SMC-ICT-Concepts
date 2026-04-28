#!/usr/bin/env python3
"""Build model features from OHLCV bars.

The neural core gets three families of inputs:
  1. Price/volume sequence features.
  2. SMC/ICT structure features from the existing strategy vocabulary.
  3. Auction-market and geopolitical-risk context.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import PROCESSED_DIR, clamp, ensure_dirs, parse_time, pct_change, read_bars, safe_div, write_rows


FEATURE_FIELDS = (
    "timestamp", "close", "ret_1", "ret_3", "ret_12", "range_pct", "body_pct",
    "upper_wick_pct", "lower_wick_pct", "atr_14_pct", "vol_z_20",
    "trend_20", "compression_20", "range_pos_50",
    "smc_trend_score", "smc_bos_bull", "smc_bos_bear", "ict_sweep_bull",
    "ict_sweep_bear", "ict_equal_highs", "ict_equal_lows", "ict_fvg_bull",
    "ict_fvg_bear", "ict_premium", "ict_discount", "ict_killzone_london",
    "ict_killzone_ny", "ict_killzone_asia", "auction_failed_up",
    "auction_failed_down", "auction_initiative_buy", "auction_initiative_sell",
    "auction_balance", "choch_bull", "choch_bear", "choch_strength",
    "choch_age_norm", "mss_bull", "mss_bear", "pivot_regularity_score",
    "swing_sequence_quality", "fib_retrace_pct", "fib_in_50_618_zone",
    "fib_in_618_705_zone", "fib_in_705_79_zone", "fib_distance_to_618",
    "fib_distance_to_705", "fib_distance_to_79", "fib_rr_proxy",
    "fib_leg_size_atr", "fib_ote_quality", "ms_level0_trend",
    "ms_level1_trend", "ms_level2_trend", "ms_alignment_score",
    "ms_last_high_prominence", "ms_last_low_prominence",
    "ms_distance_to_major_high", "ms_distance_to_major_low",
    "ms_structure_confidence", "gpr_ai", "gpr_aer", "gpr_oil", "gpr_nonoil",
    "gpr_ai_change_7", "gpr_ai_z_90", "news_geo_score", "news_geo_count",
    "news_conflict_score", "news_energy_score", "news_us_score",
    "news_europe_score", "news_russia_score", "news_china_score",
    "news_middle_east_score", "news_global_score",
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def true_range(prev_close: float, bar: dict) -> float:
    return max(bar["high"] - bar["low"], abs(bar["high"] - prev_close), abs(bar["low"] - prev_close))


def bar_datetime(bar: dict) -> dt.datetime | None:
    return parse_time(bar.get("timestamp"))


def load_gpr(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            date = row.get("Date") or row.get("date")
            if not date:
                continue
            try:
                rows.append({
                    "date": date,
                    "GPR_AI": float(row.get("GPR_AI", 0) or 0),
                    "GPR_AER": float(row.get("GPR_AER", 0) or 0),
                    "GPR_OIL": float(row.get("GPR_OIL", 0) or 0),
                    "GPR_NONOIL": float(row.get("GPR_NONOIL", 0) or 0),
                })
            except ValueError:
                continue

    by_date = {}
    ai_values = []
    for row in rows:
        ai_values.append(row["GPR_AI"])
        trailing = ai_values[-90:]
        m = mean(trailing)
        s = stdev(trailing) or 1.0
        prev7 = ai_values[-8] if len(ai_values) >= 8 else row["GPR_AI"]
        by_date[row["date"]] = {
            "gpr_ai": row["GPR_AI"] / 100.0,
            "gpr_aer": row["GPR_AER"] / 100.0,
            "gpr_oil": row["GPR_OIL"] / 100.0,
            "gpr_nonoil": row["GPR_NONOIL"] / 100.0,
            "gpr_ai_change_7": safe_div(row["GPR_AI"] - prev7, 100.0),
            "gpr_ai_z_90": safe_div(row["GPR_AI"] - m, s),
        }
    return by_date


def load_news(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    rows = {}
    try:
        f = open(path, newline="")
    except FileNotFoundError:
        return {}
    with f:
        for row in csv.DictReader(f):
            date = row.get("date") or row.get("Date")
            if not date:
                continue
            try:
                rows[date] = {
                    "news_geo_score": float(row.get("geo_score", 0) or 0),
                    "news_geo_count": float(row.get("geo_count", 0) or 0),
                    "news_conflict_score": float(row.get("conflict_score", 0) or 0),
                    "news_energy_score": float(row.get("energy_score", 0) or 0),
                    "news_us_score": float(row.get("us_score", 0) or 0),
                    "news_europe_score": float(row.get("europe_score", 0) or 0),
                    "news_russia_score": float(row.get("russia_score", 0) or 0),
                    "news_china_score": float(row.get("china_score", 0) or 0),
                    "news_middle_east_score": float(row.get("middle_east_score", 0) or 0),
                    "news_global_score": float(row.get("global_score", 0) or 0),
                }
            except ValueError:
                continue
    return rows


def default_news_features() -> dict:
    return {
        "news_geo_score": 0.0,
        "news_geo_count": 0.0,
        "news_conflict_score": 0.0,
        "news_energy_score": 0.0,
        "news_us_score": 0.0,
        "news_europe_score": 0.0,
        "news_russia_score": 0.0,
        "news_china_score": 0.0,
        "news_middle_east_score": 0.0,
        "news_global_score": 0.0,
    }


def swing_flags(window: list[dict]) -> tuple[float, float, float]:
    if len(window) < 8:
        return 0.0, 0.0, 0.0
    highs = []
    lows = []
    for i in range(2, len(window) - 2):
        if window[i]["high"] > max(window[i - 2]["high"], window[i - 1]["high"], window[i + 1]["high"], window[i + 2]["high"]):
            highs.append((i, window[i]["high"]))
        if window[i]["low"] < min(window[i - 2]["low"], window[i - 1]["low"], window[i + 1]["low"], window[i + 2]["low"]):
            lows.append((i, window[i]["low"]))
    if len(highs) < 2 or len(lows) < 2:
        return 0.0, 0.0, 0.0
    trend = 0.0
    if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
        trend = 1.0
    elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
        trend = -1.0
    return trend, highs[-1][1], lows[-1][1]


def pivot_points(window: list[dict], lookback: int = 3) -> list[dict]:
    pivots = []
    if len(window) < lookback * 2 + 1:
        return pivots
    for i in range(lookback, len(window) - lookback):
        left = window[i - lookback:i]
        right = window[i + 1:i + lookback + 1]
        high = window[i]["high"]
        low = window[i]["low"]
        if all(high > b["high"] for b in left + right):
            pivots.append({"idx": i, "price": high, "type": "high", "bar": window[i]})
        if all(low < b["low"] for b in left + right):
            pivots.append({"idx": i, "price": low, "type": "low", "bar": window[i]})
    return pivots


def pivot_trend(pivots: list[dict]) -> float:
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return 0.0
    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"] > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"] < lows[-2]["price"]
    if hh and hl:
        return 1.0
    if lh and ll:
        return -1.0
    return 0.0


def pivot_regularity(pivots: list[dict]) -> float:
    if len(pivots) < 4:
        return 0.0
    gaps = [pivots[i]["idx"] - pivots[i - 1]["idx"] for i in range(1, len(pivots))]
    avg_gap = mean(gaps)
    if avg_gap <= 0:
        return 0.0
    return clamp(1.0 - safe_div(stdev(gaps), avg_gap), 0.0, 1.0)


def swing_sequence_quality(pivots: list[dict]) -> float:
    if len(pivots) < 4:
        return 0.0
    recent = pivots[-6:]
    alternations = sum(1 for i in range(1, len(recent)) if recent[i]["type"] != recent[i - 1]["type"])
    return safe_div(alternations, max(1, len(recent) - 1))


def choch_features(window: list[dict], atr_14: float) -> dict:
    pivots = pivot_points(window, 3)
    trend = pivot_trend(pivots[:-1])
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    current = window[-1]["close"]
    result = {
        "choch_bull": 0.0,
        "choch_bear": 0.0,
        "choch_strength": 0.0,
        "choch_age_norm": 0.0,
        "mss_bull": 0.0,
        "mss_bear": 0.0,
        "pivot_regularity_score": pivot_regularity(pivots),
        "swing_sequence_quality": swing_sequence_quality(pivots),
    }
    if not highs or not lows:
        return result

    last_high = highs[-1]
    last_low = lows[-1]
    if trend <= 0 and current > last_high["price"]:
        distance = current - last_high["price"]
        result["choch_bull"] = 1.0
        result["choch_strength"] = clamp(safe_div(distance, atr_14), 0.0, 3.0) / 3.0
        result["choch_age_norm"] = clamp(safe_div(len(window) - 1 - last_high["idx"], 50.0), 0.0, 1.0)
    elif trend >= 0 and current < last_low["price"]:
        distance = last_low["price"] - current
        result["choch_bear"] = 1.0
        result["choch_strength"] = clamp(safe_div(distance, atr_14), 0.0, 3.0) / 3.0
        result["choch_age_norm"] = clamp(safe_div(len(window) - 1 - last_low["idx"], 50.0), 0.0, 1.0)

    recent = window[-20:]
    prior_low = min(b["low"] for b in recent[:-1])
    prior_high = max(b["high"] for b in recent[:-1])
    swept_low = window[-1]["low"] < prior_low and current > prior_low
    swept_high = window[-1]["high"] > prior_high and current < prior_high
    result["mss_bull"] = 1.0 if swept_low and result["choch_bull"] else 0.0
    result["mss_bear"] = 1.0 if swept_high and result["choch_bear"] else 0.0
    return result


def fib_features(window: list[dict], atr_14: float) -> dict:
    pivots = pivot_points(window, 3)
    result = {
        "fib_retrace_pct": 0.0,
        "fib_in_50_618_zone": 0.0,
        "fib_in_618_705_zone": 0.0,
        "fib_in_705_79_zone": 0.0,
        "fib_distance_to_618": 1.0,
        "fib_distance_to_705": 1.0,
        "fib_distance_to_79": 1.0,
        "fib_rr_proxy": 0.0,
        "fib_leg_size_atr": 0.0,
        "fib_ote_quality": 0.0,
    }
    if len(pivots) < 2:
        return result

    recent = pivots[-8:]
    leg = None
    for j in range(len(recent) - 1, 0, -1):
        a = recent[j - 1]
        b = recent[j]
        if a["type"] != b["type"] and abs(b["price"] - a["price"]) > 0:
            leg = (a, b)
            break
    if not leg:
        return result

    start, end = leg
    leg_size = abs(end["price"] - start["price"])
    if leg_size <= 0:
        return result
    current = window[-1]["close"]
    if start["type"] == "low" and end["type"] == "high":
        retrace = safe_div(end["price"] - current, leg_size)
        risk_anchor = end["price"] - leg_size * 0.79
        reward_anchor = end["price"] + leg_size * 0.272
        rr_proxy = safe_div(reward_anchor - current, current - risk_anchor)
    else:
        retrace = safe_div(current - end["price"], leg_size)
        risk_anchor = end["price"] + leg_size * 0.79
        reward_anchor = end["price"] - leg_size * 0.272
        rr_proxy = safe_div(current - reward_anchor, risk_anchor - current)

    retrace = clamp(retrace, 0.0, 1.5)
    dist_618 = abs(retrace - 0.618)
    dist_705 = abs(retrace - 0.705)
    dist_79 = abs(retrace - 0.79)
    result.update({
        "fib_retrace_pct": retrace,
        "fib_in_50_618_zone": 1.0 if 0.50 <= retrace < 0.618 else 0.0,
        "fib_in_618_705_zone": 1.0 if 0.618 <= retrace < 0.705 else 0.0,
        "fib_in_705_79_zone": 1.0 if 0.705 <= retrace <= 0.79 else 0.0,
        "fib_distance_to_618": clamp(dist_618, 0.0, 1.0),
        "fib_distance_to_705": clamp(dist_705, 0.0, 1.0),
        "fib_distance_to_79": clamp(dist_79, 0.0, 1.0),
        "fib_rr_proxy": clamp(rr_proxy / 4.0, 0.0, 2.0),
        "fib_leg_size_atr": clamp(safe_div(leg_size, atr_14), 0.0, 10.0) / 10.0,
        "fib_ote_quality": clamp(1.0 - min(dist_618, dist_705, dist_79) / 0.20, 0.0, 1.0),
    })
    return result


def multiscale_structure_features(bars: list[dict], i: int, atr_14: float) -> dict:
    result = {
        "ms_level0_trend": 0.0,
        "ms_level1_trend": 0.0,
        "ms_level2_trend": 0.0,
        "ms_alignment_score": 0.0,
        "ms_last_high_prominence": 0.0,
        "ms_last_low_prominence": 0.0,
        "ms_distance_to_major_high": 1.0,
        "ms_distance_to_major_low": 1.0,
        "ms_structure_confidence": 0.0,
    }
    windows = [bars[max(0, i - 20):i + 1], bars[max(0, i - 50):i + 1], bars[max(0, i - 100):i + 1]]
    trends = []
    for window in windows:
        pivots = pivot_points(window, 3)
        trends.append(pivot_trend(pivots))
    result["ms_level0_trend"], result["ms_level1_trend"], result["ms_level2_trend"] = trends
    nonzero = [t for t in trends if t != 0]
    if nonzero:
        result["ms_alignment_score"] = abs(sum(nonzero)) / len(nonzero)

    major = windows[-1]
    pivots = pivot_points(major, 3)
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    current = bars[i]["close"]
    major_range = max(b["high"] for b in major) - min(b["low"] for b in major)
    if highs:
        last_high = highs[-1]
        local_lows = [p["price"] for p in lows if abs(p["idx"] - last_high["idx"]) <= 20]
        base = min(local_lows) if local_lows else min(b["low"] for b in major)
        result["ms_last_high_prominence"] = clamp(safe_div(last_high["price"] - base, atr_14), 0.0, 5.0) / 5.0
        result["ms_distance_to_major_high"] = clamp(abs(current - last_high["price"]) / max(major_range, 1e-12), 0.0, 1.0)
    if lows:
        last_low = lows[-1]
        local_highs = [p["price"] for p in highs if abs(p["idx"] - last_low["idx"]) <= 20]
        base = max(local_highs) if local_highs else max(b["high"] for b in major)
        result["ms_last_low_prominence"] = clamp(safe_div(base - last_low["price"], atr_14), 0.0, 5.0) / 5.0
        result["ms_distance_to_major_low"] = clamp(abs(current - last_low["price"]) / max(major_range, 1e-12), 0.0, 1.0)
    result["ms_structure_confidence"] = clamp(
        (result["ms_alignment_score"] + max(result["ms_last_high_prominence"], result["ms_last_low_prominence"])) / 2,
        0.0,
        1.0,
    )
    return result


def equal_level_counts(window: list[dict], tolerance: float = 0.001) -> tuple[float, float]:
    equal_highs = 0
    equal_lows = 0
    for i in range(len(window) - 1):
        for j in range(i + 2, len(window)):
            if abs(window[i]["high"] - window[j]["high"]) / window[i]["high"] < tolerance:
                equal_highs += 1
            if abs(window[i]["low"] - window[j]["low"]) / window[i]["low"] < tolerance:
                equal_lows += 1
    scale = max(1, len(window))
    return min(1.0, equal_highs / scale), min(1.0, equal_lows / scale)


def killzone_flags(bar: dict) -> tuple[float, float, float]:
    parsed = bar_datetime(bar)
    if not parsed:
        return 0.0, 0.0, 0.0
    hour = parsed.hour + parsed.minute / 60
    asia = 1.0 if 1 <= hour < 5 else 0.0
    london = 1.0 if 7 <= hour < 10 else 0.0
    ny = 1.0 if 12 <= hour < 16 else 0.0
    return london, ny, asia


def smc_ict_auction_features(bars: list[dict], i: int, atr_14: float, vol_z: float) -> dict:
    window = bars[max(0, i - 50):i + 1]
    recent = bars[max(0, i - 20):i + 1]
    prev = bars[i - 1]
    bar = bars[i]
    current = bar["close"]

    trend, last_high, last_low = swing_flags(window)
    prev_high = max(b["high"] for b in recent[:-1])
    prev_low = min(b["low"] for b in recent[:-1])
    bos_bull = 1.0 if current > prev_high else 0.0
    bos_bear = 1.0 if current < prev_low else 0.0
    sweep_bull = 1.0 if bar["low"] < prev_low and current > prev_low else 0.0
    sweep_bear = 1.0 if bar["high"] > prev_high and current < prev_high else 0.0
    equal_highs, equal_lows = equal_level_counts(recent)

    fvg_bull = 0.0
    fvg_bear = 0.0
    if i >= 2:
        prev2 = bars[i - 2]
        fvg_bull = 1.0 if prev2["high"] < bar["low"] else 0.0
        fvg_bear = 1.0 if prev2["low"] > bar["high"] else 0.0

    high_50 = max(b["high"] for b in window)
    low_50 = min(b["low"] for b in window)
    range_pos = safe_div(current - low_50, high_50 - low_50, 0.5)
    premium = 1.0 if range_pos > 0.55 else 0.0
    discount = 1.0 if range_pos < 0.45 else 0.0
    london, ny, asia = killzone_flags(bar)

    range_size = bar["high"] - bar["low"]
    close_pos = safe_div(current - bar["low"], range_size, 0.5)
    failed_up = 1.0 if bar["high"] > prev_high and current < prev_high else 0.0
    failed_down = 1.0 if bar["low"] < prev_low and current > prev_low else 0.0
    initiative_buy = 1.0 if range_size > atr_14 * 1.2 and close_pos > 0.75 and vol_z > 0.5 else 0.0
    initiative_sell = 1.0 if range_size > atr_14 * 1.2 and close_pos < 0.25 and vol_z > 0.5 else 0.0
    balance = 1.0 if range_size < atr_14 * 0.75 and 0.35 <= close_pos <= 0.65 else 0.0

    return {
        "smc_trend_score": trend,
        "smc_bos_bull": bos_bull,
        "smc_bos_bear": bos_bear,
        "ict_sweep_bull": sweep_bull,
        "ict_sweep_bear": sweep_bear,
        "ict_equal_highs": equal_highs,
        "ict_equal_lows": equal_lows,
        "ict_fvg_bull": fvg_bull,
        "ict_fvg_bear": fvg_bear,
        "ict_premium": premium,
        "ict_discount": discount,
        "ict_killzone_london": london,
        "ict_killzone_ny": ny,
        "ict_killzone_asia": asia,
        "auction_failed_up": failed_up,
        "auction_failed_down": failed_down,
        "auction_initiative_buy": initiative_buy,
        "auction_initiative_sell": initiative_sell,
        "auction_balance": balance,
    }


def build_features(bars: list[dict], gpr: dict[str, dict] | None = None, news: dict[str, dict] | None = None) -> list[dict]:
    rows = []
    trs = []
    closes = []
    volumes = []
    gpr = gpr or {}
    news = news or {}

    for i, bar in enumerate(bars):
        close = bar["close"]
        prev_close = bars[i - 1]["close"] if i > 0 else close
        closes.append(close)
        volumes.append(bar["volume"])
        trs.append(true_range(prev_close, bar))

        if i < 50:
            continue

        atr_14 = mean(trs[-14:])
        vol_window = volumes[-20:]
        close_20 = closes[-20:]
        high_50 = max(b["high"] for b in bars[i - 49:i + 1])
        low_50 = min(b["low"] for b in bars[i - 49:i + 1])
        range_20 = [safe_div(b["high"] - b["low"], b["close"]) for b in bars[i - 19:i + 1]]
        body = abs(bar["close"] - bar["open"])
        full_range = max(bar["high"] - bar["low"], 1e-12)

        vol_z = safe_div(bar["volume"] - mean(vol_window), stdev(vol_window))
        parsed = bar_datetime(bar)
        date_key = parsed.date().isoformat() if parsed else ""
        gpr_features = gpr.get(date_key, {
            "gpr_ai": 1.0,
            "gpr_aer": 1.0,
            "gpr_oil": 0.0,
            "gpr_nonoil": 1.0,
            "gpr_ai_change_7": 0.0,
            "gpr_ai_z_90": 0.0,
        })
        news_features = news.get(date_key, default_news_features())
        structure_window = bars[max(0, i - 100):i + 1]
        structure = smc_ict_auction_features(bars, i, atr_14, vol_z)
        choch = choch_features(structure_window, atr_14)
        fib = fib_features(structure_window, atr_14)
        multiscale = multiscale_structure_features(bars, i, atr_14)

        rows.append({
            "timestamp": bar["timestamp"],
            "close": close,
            "ret_1": pct_change(bars[i - 1]["close"], close),
            "ret_3": pct_change(bars[i - 3]["close"], close),
            "ret_12": pct_change(bars[i - 12]["close"], close),
            "range_pct": safe_div(bar["high"] - bar["low"], close),
            "body_pct": safe_div(body, close),
            "upper_wick_pct": safe_div(bar["high"] - max(bar["open"], bar["close"]), full_range),
            "lower_wick_pct": safe_div(min(bar["open"], bar["close"]) - bar["low"], full_range),
            "atr_14_pct": safe_div(atr_14, close),
            "vol_z_20": vol_z,
            "trend_20": safe_div(close_20[-1] - close_20[0], close_20[0]),
            "compression_20": safe_div(mean(range_20[-5:]), mean(range_20)),
            "range_pos_50": safe_div(close - low_50, high_50 - low_50, 0.5),
            **structure,
            **choch,
            **fib,
            **multiscale,
            **gpr_features,
            **news_features,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gpr", default=None)
    parser.add_argument("--news", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    rows = build_features(read_bars(args.input), load_gpr(args.gpr), load_news(args.news))
    out = Path(args.out) if args.out else PROCESSED_DIR / (Path(args.input).stem + "_features.csv")
    write_rows(out, rows, FEATURE_FIELDS)
    print(out)


if __name__ == "__main__":
    main()
