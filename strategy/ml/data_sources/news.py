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
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import RAW_DIR, ensure_dirs


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = (
    "geopolitical risk OR war OR conflict OR missile OR sanctions OR invasion "
    "OR troops OR ceasefire OR oil supply OR terrorism OR military"
)

RSS_FEEDS = {
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera_all": "https://www.aljazeera.com/xml/rss/all.xml",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "dw_top": "https://rss.dw.com/rdf/rss-en-all",
    "npr_world": "https://feeds.npr.org/1004/rss.xml",
    "cnbc_world": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
}

FIELDNAMES = [
    "date", "updated_at", "geo_count", "geo_score", "conflict_score",
    "energy_score", "us_score", "europe_score", "russia_score",
    "china_score", "middle_east_score", "global_score",
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
}


def fetch_url(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tradingview-mcp-geonews/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


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


def collect_gdelt(max_records: int) -> list[dict]:
    params = {
        "query": GDELT_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "hybridrel",
    }
    url = f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"
    try:
        payload = json.loads(fetch_url(url).decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"warning: gdelt fetch failed: {exc}", file=sys.stderr)
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
    return items


def collect_rss() -> list[dict]:
    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            rows.extend(rss_items(fetch_url(url), source))
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
        for name in ("us", "europe", "russia", "china", "middle_east", "global")
    }
    geo = max(conflict, energy, *regional.values())
    return {"geo": geo, "conflict": conflict, "energy": energy, **regional}


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
            "us_score": sum(score["us"] for score in scores) / total,
            "europe_score": sum(score["europe"] for score in scores) / total,
            "russia_score": sum(score["russia"] for score in scores) / total,
            "china_score": sum(score["china"] for score in scores) / total,
            "middle_east_score": sum(score["middle_east"] for score in scores) / total,
            "global_score": sum(score["global"] for score in scores) / total,
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
    parser.add_argument("--max-gdelt", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()
    out = Path(args.out)
    articles = collect_gdelt(args.max_gdelt)
    articles.extend(collect_rss())
    rows = merge_existing(out, aggregate(articles))
    write_rows(out, rows)
    print(out)
    print({"articles": len(articles), "days": len(rows)})


if __name__ == "__main__":
    main()
