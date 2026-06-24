"""Compute a per-engineer impact score from the cached PostHog GitHub data.

Reads the JSON files produced by ``fetch_data.py`` (prs.json, reviews.json,
review_comments.json, commits.json) and scores each engineer across four
signals, each normalized to 0-1 and combined into a weighted composite:

  1. Velocity x Quality  - PRs merged per week, damped by how many review
                           rounds (CHANGES_REQUESTED) their PRs needed.
  2. Team Multiplier     - PRs reviewed for others x a fast-turnaround bonus.
  3. Leverage            - how central (widely-touched) the files they work on
                           are; working on high-centrality files scores higher.
  4. Consistency         - how evenly their activity is spread over the window
                           (weekly buckets; bursty work is penalized).

Bots (logins ending in ``[bot]``) are excluded. Output is written to
``data/impact_scores.csv`` and also printed to stdout.

Usage:
    python analyze.py                 # uses the standard 90-day window
    python analyze.py --days 30
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# Composite weights (must sum to 1.0). Velocity x Quality is weighted highest
# as the most direct measure of shipped work; the others capture force
# multiplication, leverage, and sustainability.
WEIGHTS = {
    "velocity_quality": 0.35,
    "team_multiplier": 0.25,
    "leverage": 0.20,
    "consistency": 0.20,
}


# --------------------------------------------------------------------------- #
# Loading / small helpers
# --------------------------------------------------------------------------- #
def load_json(name: str) -> list[dict]:
    """Load a cached JSON list, tolerating a missing or empty file."""
    path = DATA_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        print(f"  warning: {name} is missing or empty; treating as no data")
        return []
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_bot(login: Optional[str]) -> bool:
    return bool(login) and login.endswith("[bot]")


def week_index(dt: datetime, start: datetime) -> int:
    return int((dt - start).days // 7)


def minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize to 0-1. Flat series -> 0.0 (all-zero) or 0.5."""
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        fill = 0.0 if hi <= 1e-12 else 0.5
        return pd.Series(fill, index=series.index)
    return (series - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #
def resolve_window(prs, reviews, commits, days: int) -> tuple[datetime, datetime, int]:
    """Anchor the window on the latest activity seen, looking back `days`."""
    stamps: list[datetime] = []
    for pr in prs:
        stamps += [parse_dt(pr.get("created_at")), parse_dt(pr.get("merged_at"))]
    for r in reviews:
        stamps.append(parse_dt(r.get("submitted_at")))
    for c in commits:
        stamps.append(parse_dt(c.get("authored_at")))
    stamps = [s for s in stamps if s is not None]

    end = max(stamps) if stamps else datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    num_weeks = max(1, math.ceil(days / 7))
    return start, end, num_weeks


# --------------------------------------------------------------------------- #
# Signal 1: Velocity x Quality
# --------------------------------------------------------------------------- #
def signal_velocity_quality(prs, reviews, num_weeks):
    """Merged PRs/week, damped by avg CHANGES_REQUESTED rounds on those PRs."""
    # review rounds per PR number
    rounds_by_pr: dict[int, int] = {}
    for r in reviews:
        if r.get("state") == "CHANGES_REQUESTED":
            rounds_by_pr[r["pr_number"]] = rounds_by_pr.get(r["pr_number"], 0) + 1

    merged_count: dict[str, int] = {}
    total_rounds: dict[str, int] = {}
    for pr in prs:
        author = pr.get("author")
        if is_bot(author) or not author or not pr.get("merged"):
            continue
        merged_count[author] = merged_count.get(author, 0) + 1
        total_rounds[author] = total_rounds.get(author, 0) + rounds_by_pr.get(
            pr["number"], 0
        )

    rows = {}
    for author, merged in merged_count.items():
        per_week = merged / num_weeks
        avg_rounds = total_rounds[author] / merged
        # Quality multiplier in (0, 1]: 1.0 when no changes were ever requested,
        # decaying as the average number of rework rounds grows.
        quality = 1.0 / (1.0 + avg_rounds)
        rows[author] = {
            "prs_merged": merged,
            "merged_per_week": round(per_week, 4),
            "avg_review_rounds": round(avg_rounds, 4),
            "vq_raw": round(per_week * quality, 6),
        }
    return rows


# --------------------------------------------------------------------------- #
# Signal 2: Team Multiplier
# --------------------------------------------------------------------------- #
def signal_team_multiplier(prs, reviews):
    """PRs reviewed for others x average fast-turnaround bonus."""
    pr_author = {pr["number"]: pr.get("author") for pr in prs}
    pr_created = {pr["number"]: parse_dt(pr.get("created_at")) for pr in prs}

    # earliest review timestamp per (reviewer, pr)
    first_review: dict[tuple[str, int], datetime] = {}
    for r in reviews:
        reviewer = r.get("author")
        pr_num = r.get("pr_number")
        submitted = parse_dt(r.get("submitted_at"))
        if is_bot(reviewer) or not reviewer or submitted is None:
            continue
        if pr_author.get(pr_num) == reviewer:
            continue  # don't credit reviewing your own PR
        key = (reviewer, pr_num)
        if key not in first_review or submitted < first_review[key]:
            first_review[key] = submitted

    agg: dict[str, dict] = {}
    for (reviewer, pr_num), submitted in first_review.items():
        created = pr_created.get(pr_num)
        bucket = agg.setdefault(reviewer, {"count": 0, "bonus_sum": 0.0, "hours": []})
        bucket["count"] += 1
        if created is not None:
            hours = max(0.0, (submitted - created).total_seconds() / 3600.0)
            # Speed bonus in (0, 1]: ~1 for an instant review, 0.5 at one day,
            # decaying smoothly for slower turnaround.
            bucket["bonus_sum"] += 1.0 / (1.0 + hours / 24.0)
            bucket["hours"].append(hours)

    rows = {}
    for reviewer, b in agg.items():
        avg_bonus = b["bonus_sum"] / b["count"] if b["count"] else 0.0
        avg_hours = sum(b["hours"]) / len(b["hours"]) if b["hours"] else float("nan")
        rows[reviewer] = {
            "prs_reviewed": b["count"],
            "avg_turnaround_hours": round(avg_hours, 2),
            "team_raw": round(b["count"] * avg_bonus, 6),
        }
    return rows


# --------------------------------------------------------------------------- #
# Signal 3: Leverage
# --------------------------------------------------------------------------- #
def signal_leverage(prs):
    """Reward working on files touched by many distinct engineers."""
    # file -> set of engineers who touched it
    file_engineers: dict[str, set[str]] = {}
    author_files: dict[str, set[str]] = {}
    missing_files = True
    for pr in prs:
        author = pr.get("author")
        if is_bot(author) or not author:
            continue
        files = pr.get("files") or []
        if files:
            missing_files = False
        for path in files:
            file_engineers.setdefault(path, set()).add(author)
            author_files.setdefault(author, set()).add(path)

    if missing_files:
        print(
            "  warning: no 'files' field in prs.json; leverage will be 0. "
            "Re-run fetch_data.py to capture PR file paths."
        )

    # centrality(file) = number of distinct engineers who touched it
    centrality = {f: len(engs) for f, engs in file_engineers.items()}

    rows = {}
    for author, files in author_files.items():
        mean_centrality = (
            sum(centrality[f] for f in files) / len(files) if files else 0.0
        )
        rows[author] = {
            "files_touched": len(files),
            "leverage_raw": round(mean_centrality, 6),
        }
    return rows


# --------------------------------------------------------------------------- #
# Signal 4: Consistency
# --------------------------------------------------------------------------- #
def signal_consistency(prs, reviews, commits, start, num_weeks):
    """Normalized entropy of weekly activity; 1.0 = perfectly even spread."""
    buckets: dict[str, list[int]] = {}

    def record(login, dt):
        if is_bot(login) or not login or dt is None:
            return
        idx = week_index(dt, start)
        if idx < 0 or idx >= num_weeks:
            return
        buckets.setdefault(login, [0] * num_weeks)[idx] += 1

    for pr in prs:
        record(pr.get("author"), parse_dt(pr.get("created_at")))
    for r in reviews:
        record(r.get("author"), parse_dt(r.get("submitted_at")))
    for c in commits:
        record(c.get("author_login"), parse_dt(c.get("authored_at")))

    rows = {}
    for login, weeks in buckets.items():
        total = sum(weeks)
        active = sum(1 for w in weeks if w > 0)
        if total == 0:
            entropy = 0.0
        else:
            probs = [w / total for w in weeks if w > 0]
            h = -sum(p * math.log(p) for p in probs)
            # Normalize by log(num_weeks) so a perfectly even spread -> 1.0.
            entropy = h / math.log(num_weeks) if num_weeks > 1 else 0.0
        rows[login] = {
            "active_weeks": active,
            "total_events": total,
            "consistency_raw": round(entropy, 6),
        }
    return rows


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_dataframe(prs, reviews, commits, days) -> pd.DataFrame:
    start, end, num_weeks = resolve_window(prs, reviews, commits, days)
    print(f"  window: {start.date()} -> {end.date()} ({num_weeks} weeks)")

    vq = signal_velocity_quality(prs, reviews, num_weeks)
    team = signal_team_multiplier(prs, reviews)
    lev = signal_leverage(prs)
    cons = signal_consistency(prs, reviews, commits, start, num_weeks)

    engineers = sorted(set(vq) | set(team) | set(lev) | set(cons))
    if not engineers:
        return pd.DataFrame()

    records = []
    for eng in engineers:
        row = {"engineer": eng}
        # raw signal blocks; default each metric for engineers absent from it
        row.update(vq.get(eng, {"prs_merged": 0, "merged_per_week": 0.0,
                                 "avg_review_rounds": 0.0, "vq_raw": 0.0}))
        row.update(team.get(eng, {"prs_reviewed": 0,
                                  "avg_turnaround_hours": float("nan"),
                                  "team_raw": 0.0}))
        row.update(lev.get(eng, {"files_touched": 0, "leverage_raw": 0.0}))
        row.update(cons.get(eng, {"active_weeks": 0, "total_events": 0,
                                  "consistency_raw": 0.0}))
        records.append(row)

    df = pd.DataFrame(records).set_index("engineer")

    # normalize each raw signal to 0-1
    df["velocity_quality_norm"] = minmax(df["vq_raw"])
    df["team_multiplier_norm"] = minmax(df["team_raw"])
    df["leverage_norm"] = minmax(df["leverage_raw"])
    df["consistency_norm"] = minmax(df["consistency_raw"])

    df["impact_score"] = (
        df["velocity_quality_norm"] * WEIGHTS["velocity_quality"]
        + df["team_multiplier_norm"] * WEIGHTS["team_multiplier"]
        + df["leverage_norm"] * WEIGHTS["leverage"]
        + df["consistency_norm"] * WEIGHTS["consistency"]
    ).round(6)

    # tidy column order: raw blocks, then norms, then composite
    ordered = [
        "prs_merged", "merged_per_week", "avg_review_rounds", "vq_raw",
        "prs_reviewed", "avg_turnaround_hours", "team_raw",
        "files_touched", "leverage_raw",
        "active_weeks", "total_events", "consistency_raw",
        "velocity_quality_norm", "team_multiplier_norm",
        "leverage_norm", "consistency_norm",
        "impact_score",
    ]
    df = df[ordered].sort_values("impact_score", ascending=False)
    return df


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=90, help="analysis window in days (default 90)"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("Loading cached data...")
    prs = load_json("prs.json")
    reviews = load_json("reviews.json")
    review_comments = load_json("review_comments.json")  # loaded for completeness
    commits = load_json("commits.json")
    print(
        f"  {len(prs)} PRs, {len(reviews)} reviews, "
        f"{len(review_comments)} review comments, {len(commits)} commits"
    )

    df = build_dataframe(prs, reviews, commits, args.days)
    if df.empty:
        print("No engineer activity found. Did you run fetch_data.py first?")
        return 1

    out_path = DATA_DIR / "impact_scores.csv"
    df.to_csv(out_path)
    print(f"\nWrote {out_path} ({len(df)} engineers)")
    with pd.option_context("display.max_rows", 20, "display.width", 200):
        print("\nTop engineers by impact score:")
        print(df[["impact_score", "velocity_quality_norm", "team_multiplier_norm",
                  "leverage_norm", "consistency_norm"]].head(15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
