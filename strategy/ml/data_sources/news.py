#!/usr/bin/env python3
"""Fetch near-real-time geopolitical headline pressure features.

This is deliberately lightweight and auditable. It combines:
  - GDELT DOC 2.0 article search for broad global coverage.
  - Public RSS feeds for source diversity.

The output is a daily feature CSV consumed by the neural model alongside the
official AI-GPR dataset.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import email.utils
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import RAW_DIR, ensure_dirs


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERIES = {
    "geopolitical": (
        "geopolitical risk OR war OR conflict OR missile OR sanctions OR invasion "
        "OR troops OR ceasefire OR oil supply OR terrorism OR military"
    ),
    "macro_fx": (
        "interest rate OR FOMC OR rate decision OR inflation OR CPI OR payrolls "
        "OR GDP OR PMI OR unemployment OR central bank OR Powell OR Lagarde "
        "OR Bailey OR Ueda OR Macklem OR hawkish OR dovish OR tapering OR forex "
        "OR currency OR exchange rate OR yen intervention OR dollar"
    ),
}

RSS_FEEDS = {
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera_all": "https://www.aljazeera.com/xml/rss/all.xml",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "dw_top": "https://rss.dw.com/rdf/rss-en-all",
    "npr_world": "https://feeds.npr.org/1004/rss.xml",
    "cnbc_world": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "fxstreet": "https://www.fxstreet.com/rss/news",
    "investinglive": "https://www.investinglive.com/rss",
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_monetary_policy": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
    "boe_news": "https://www.bankofengland.co.uk/rss/news",
    "boc_press": "https://www.bankofcanada.ca/content_type/press-releases/feed/",
}

FIELDNAMES = [
    "date", "updated_at", "geo_count", "geo_score", "conflict_score",
    "energy_score", "us_score", "europe_score", "russia_score",
    "china_score", "middle_east_score", "global_score", "fed_policy_score",
    "ecb_policy_score", "boe_policy_score", "boj_policy_score",
    "boc_policy_score", "inflation_score", "employment_score",
    "growth_score", "fx_intervention_score", "dollar_strength_score",
    "risk_sentiment_score", "macro_score",
]

CATEGORY_KEYWORDS = {
    "conflict": [
        "war", "conflict", "missile", "attack", "attacks", "strike", "strikes",
        "invasion", "troops", "military", "ceasefire", "terror", "terrorism",
        "hostage", "nuclear", "drone", "border", "escalation",
    ],
    "energy": [
        "oil", "crude", "gas", "lng", "pipeline", "opec", "energy",
        "supply", "shipping", "red sea", "hormuz", "sanction", "sanctions",
    ],
    "us": ["united states", "u.s.", "us ", "america", "american", "washington", "trump", "congress", "fed"],
    "europe": [
        "europe", "european", "eu ", "e.u.", "ukraine", "germany", "france",
        "britain", "uk ", "nato", "poland", "baltic",
    ],
    "russia": ["russia", "russian", "moscow", "kremlin", "putin"],
    "china": ["china", "chinese", "beijing", "taiwan", "xi jinping", "south china sea"],
    "middle_east": [
        "middle east", "israel", "iran", "gaza", "hamas", "hezbollah",
        "lebanon", "syria", "iraq", "yemen", "houthi", "saudi", "qatar",
    ],
    "global": [
        "global", "world", "geopolitical", "election", "inflation", "central bank",
        "trade", "tariff", "supply chain", "risk",
    ],
    "fed_policy": [
        "fomc", "federal reserve", "fed ", "powell", "fed funds", "dot plot",
        "rate hike", "rate cut", "interest rate", "monetary policy",
    ],
    "ecb_policy": [
        "ecb", "european central bank", "lagarde", "eurozone rate",
        "deposit rate", "euro area inflation",
    ],
    "boe_policy": [
        "bank of england", "boe", "bailey", "mpc", "uk rate",
        "british inflation", "gilts",
    ],
    "boj_policy": [
        "bank of japan", "boj", "ueda", "yen", "japanese rate",
        "yield curve control", "jgb",
    ],
    "boc_policy": [
        "bank of canada", "boc", "macklem", "canadian rate",
        "canada inflation",
    ],
    "inflation": [
        "cpi", "inflation", "consumer price", "core pce", "pce inflation",
        "producer price", "ppi", "price pressures",
    ],
    "employment": [
        "nfp", "nonfarm", "payrolls", "unemployment", "jobless claims",
        "labor market", "labour market", "wages", "earnings",
    ],
    "growth": [
        "gdp", "economic growth", "recession", "soft landing", "hard landing",
        "pmi", "ism", "retail sales", "industrial production",
    ],
    "fx_intervention": [
        "fx intervention", "yen intervention", "currency intervention",
        "currency defense", "moj warning", "ministry of finance",
    ],
    "dollar_strength": [
        "dxy", "dollar index", "dollar rally", "dollar strength",
        "greenback", "us dollar", "u.s. dollar",
    ],
    "risk_sentiment": [
        "risk-off", "risk on", "risk-off", "safe haven", "safe-haven",
        "market turmoil", "equity selloff", "volatility",
    ],
}

REGIONAL_CATEGORIES = ("us", "europe", "russia", "china", "middle_east", "global")
MACRO_CATEGORIES = (
    "fed_policy", "ecb_policy", "boe_policy", "boj_policy", "boc_policy",
    "inflation", "employment", "growth", "fx_intervention",
    "dollar_strength", "risk_sentiment",
)


RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def fetch_url(url: str, timeout: int = 8, retries: int = 0, backoff: float = 1.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tradingview-mcp-geonews/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else backoff * (2 ** attempt)
            except ValueError:
                delay = backoff * (2 ** attempt)
            time.sleep(min(delay, 10.0))
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(backoff * (2 ** attempt), 10.0))
    raise RuntimeError("unreachable fetch retry state")


def parse_date(value: str | None) -> dt.date:
    if not value:
        return dt.datetime.now(dt.timezone.utc).date()
    value = value.strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value[:len(fmt)], fmt).date()
        except ValueError:
            pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).date()
    except Exception:
        return dt.datetime.now(dt.timezone.utc).date()


def collect_gdelt(max_records: int, timeout: int, retries: int = 0, timespan: str = "1d") -> list[dict]:
    if max_records <= 0:
        return []
    rows = []
    per_query = max(10, max_records // max(1, len(GDELT_QUERIES)))
    for query_name, query in GDELT_QUERIES.items():
        rows.extend(collect_gdelt_query(query, per_query, timeout, query_name, retries, timespan))
        time.sleep(0.75)
    return rows


def collect_gdelt_query(
    query: str,
    max_records: int,
    timeout: int,
    query_name: str,
    retries: int,
    timespan: str,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "hybridrel",
        "timespan": timespan,
    }
    url = f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"
    try:
        payload = json.loads(fetch_url(url, timeout=timeout, retries=retries, backoff=2.0).decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"warning: gdelt fetch failed query={query_name}: {exc}", file=sys.stderr)
        return []
    rows = []
    for article in payload.get("articles", []):
        title = article.get("title") or ""
        description = article.get("seendate") or ""
        rows.append({
            "source": article.get("sourceCountry") or article.get("domain") or "gdelt",
            "title": title,
            "summary": description,
            "date": parse_date(article.get("seendate")),
            "url": article.get("url") or "",
        })
    return rows


def rss_items(xml_bytes: bytes, source: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        summary = item.findtext("description") or ""
        pub_date = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
        link = item.findtext("link") or ""
        items.append({
            "source": source,
            "title": title,
            "summary": summary,
            "date": parse_date(pub_date),
            "url": link,
        })
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
        summary = (
            entry.findtext("{http://www.w3.org/2005/Atom}summary")
            or entry.findtext("{http://www.w3.org/2005/Atom}content")
            or ""
        )
        pub_date = entry.findtext("{http://www.w3.org/2005/Atom}updated") or entry.findtext("{http://www.w3.org/2005/Atom}published")
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        items.append({
            "source": source,
            "title": title,
            "summary": summary,
            "date": parse_date(pub_date),
            "url": link,
        })
    return items


def collect_rss(timeout: int) -> list[dict]:
    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            rows.extend(rss_items(fetch_url(url, timeout=timeout), source))
        except Exception as exc:
            print(f"warning: rss fetch failed source={source}: {exc}", file=sys.stderr)
    return rows


def keyword_score(text: str, keywords: list[str]) -> float:
    lower = f" {text.lower()} "
    hits = sum(1 for keyword in keywords if keyword in lower)
    return min(1.0, hits / 3.0)


def article_score(article: dict) -> dict:
    text = f"{article.get('title', '')} {article.get('summary', '')}"
    conflict = keyword_score(text, CATEGORY_KEYWORDS["conflict"])
    energy = keyword_score(text, CATEGORY_KEYWORDS["energy"])
    regional = {
        name: keyword_score(text, CATEGORY_KEYWORDS[name])
        for name in REGIONAL_CATEGORIES
    }
    macro = {
        name: keyword_score(text, CATEGORY_KEYWORDS[name])
        for name in MACRO_CATEGORIES
    }
    geo = max(conflict, energy, *regional.values())
    macro_score = max(macro.values()) if macro else 0.0
    return {"geo": geo, "conflict": conflict, "energy": energy, "macro": macro_score, **regional, **macro}


def aggregate(articles: list[dict]) -> list[dict]:
    today = dt.datetime.now(dt.timezone.utc).date()
    deduped = {}
    for article in articles:
        date = article.get("date") or today
        if (today - date).days > 3:
            continue
        key = hashlib.sha256(f"{article.get('title')}|{article.get('url')}".encode()).hexdigest()
        deduped[key] = article

    by_date: dict[str, list[dict]] = {}
    for article in deduped.values():
        by_date.setdefault(article["date"].isoformat(), []).append(article)

    rows = []
    for date, items in sorted(by_date.items()):
        scores = [article_score(item) for item in items]
        total = max(1, len(scores))
        geo_sum = sum(score["geo"] for score in scores)
        rows.append({
            "date": date,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "geo_count": len(items),
            "geo_score": geo_sum / total,
            "conflict_score": sum(score["conflict"] for score in scores) / total,
            "energy_score": sum(score["energy"] for score in scores) / total,
            **{f"{name}_score": sum(score[name] for score in scores) / total for name in REGIONAL_CATEGORIES},
            **{f"{name}_score": sum(score[name] for score in scores) / total for name in MACRO_CATEGORIES},
            "macro_score": sum(score["macro"] for score in scores) / total,
        })
    return rows


def merge_existing(out: Path, rows: list[dict]) -> list[dict]:
    merged = {}
    if out.exists():
        with out.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("date"):
                    merged[row["date"]] = row
    for row in rows:
        merged[row["date"]] = row
    return [merged[key] for key in sorted(merged)]


def write_rows(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RAW_DIR / "news" / "geopolitical_news_daily.csv"))
    parser.add_argument("--max-gdelt", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--gdelt-retries", type=int, default=0)
    parser.add_argument("--gdelt-timespan", default="1d")
    args = parser.parse_args()

    ensure_dirs()
    out = Path(args.out)
    articles = collect_gdelt(args.max_gdelt, args.timeout, args.gdelt_retries, args.gdelt_timespan)
    articles.extend(collect_rss(args.timeout))
    rows = merge_existing(out, aggregate(articles))
    write_rows(out, rows)
    print(out)
    print({"articles": len(articles), "days": len(rows)})


if __name__ == "__main__":
    main()
