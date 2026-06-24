"""Streamlit dashboard: PostHog engineering impact.

Reads ``data/impact_scores.csv`` (produced by analyze.py) and ``data/prs.json``
(produced by fetch_data.py) and presents the top 5 most impactful engineers,
why they rank where they do, and the methodology behind the score.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    # Single source of truth for the weights, shared with the scorer.
    from analyze import WEIGHTS
except Exception:  # pragma: no cover - fallback if analyze isn't importable
    WEIGHTS = {
        "velocity_quality": 0.35,
        "team_multiplier": 0.25,
        "leverage": 0.20,
        "consistency": 0.20,
    }

DATA_DIR = Path(__file__).resolve().parent / "data"
TOP_N = 5

# Signal display metadata: (key prefix in CSV, label, one-line definition).
SIGNALS = [
    ("velocity_quality", "Velocity × Quality",
     "PRs merged per week, damped by review rework"),
    ("team_multiplier", "Team Multiplier",
     "PRs reviewed for others × review-speed bonus"),
    ("leverage", "Leverage",
     "Working on files many engineers touch"),
    ("consistency", "Consistency",
     "Activity spread evenly across the window"),
]
SIGNAL_COLORS = ["#f54e00", "#1d4aff", "#30abc6", "#c278cf"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame | None:
    path = DATA_DIR / "impact_scores.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_prs() -> list[dict]:
    path = DATA_DIR / "prs.json"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def pr_context(prs: list[dict]) -> dict:
    authors = {p.get("author") for p in prs
               if p.get("author") and not str(p["author"]).endswith("[bot]")}
    dates = sorted(p["created_at"][:10] for p in prs if p.get("created_at"))
    return {
        "pr_count": len(prs),
        "contributors": len(authors),
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
    }


# --------------------------------------------------------------------------- #
# "Why" bullets — built from RAW columns, not normalized scores
# --------------------------------------------------------------------------- #
def rework_label(rounds: float) -> str:
    if rounds < 0.5:
        return "low rework"
    if rounds < 1.5:
        return "moderate rework"
    return "heavy rework"


def turnaround_phrase(hours: float) -> str:
    if hours is None or (isinstance(hours, float) and math.isnan(hours)):
        return "varied turnaround"
    if hours < 1:
        return "near-instant turnaround (<1h)"
    if hours < 24:
        return f"fast turnaround (~{hours:.0f}h)"
    return f"~{hours / 24:.1f}-day turnaround"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def why_bullets(row: pd.Series) -> list[tuple[float, str]]:
    """Return (signal_norm, bullet) pairs; caller picks the strongest few."""
    bullets: list[tuple[float, str]] = []

    # Velocity × Quality
    bullets.append((
        row["velocity_quality_norm"],
        f"Merged **{plural(int(row['prs_merged']), 'PR')}** "
        f"(**{row['merged_per_week']:.1f}/week**) with "
        f"{rework_label(row['avg_review_rounds'])} "
        f"({row['avg_review_rounds']:.1f} change-request rounds/PR)",
    ))

    # Team Multiplier — only if they actually reviewed others' work
    if int(row["prs_reviewed"]) > 0:
        bullets.append((
            row["team_multiplier_norm"],
            f"Reviewed **{plural(int(row['prs_reviewed']), 'PR')}** for "
            f"teammates with {turnaround_phrase(row['avg_turnaround_hours'])}",
        ))

    # Leverage
    if int(row["files_touched"]) > 0:
        bullets.append((
            row["leverage_norm"],
            f"Touched **{plural(int(row['files_touched']), 'file')}**, often in "
            f"high-traffic areas of the codebase shared across the team",
        ))

    # Consistency
    bullets.append((
        row["consistency_norm"],
        f"Active across **{plural(int(row['active_weeks']), 'week')}** "
        f"({plural(int(row['total_events']), 'contribution')}) — "
        f"{'steady, not bursty' if row['consistency_norm'] >= 0.5 else 'concentrated activity'}",
    ))

    return bullets


def top_bullets(row: pd.Series, n: int = 3) -> list[str]:
    ranked = sorted(why_bullets(row), key=lambda x: x[0], reverse=True)
    return [text for _, text in ranked[:n]]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def signal_chart(row: pd.Series) -> alt.Chart:
    data = pd.DataFrame({
        "Signal": [label for _, label, _ in SIGNALS],
        "Score": [row[f"{key}_norm"] for key, _, _ in SIGNALS],
    })
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", tickCount=3, title=None)),
            y=alt.Y("Signal:N", sort=[label for _, label, _ in SIGNALS],
                    axis=alt.Axis(title=None, labelLimit=120)),
            color=alt.Color("Signal:N",
                            scale=alt.Scale(
                                domain=[label for _, label, _ in SIGNALS],
                                range=SIGNAL_COLORS),
                            legend=None),
            tooltip=[alt.Tooltip("Signal:N"),
                     alt.Tooltip("Score:Q", format=".0%")],
        )
        .properties(height=130)
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1400px;}
          [data-testid="stMetricValue"] {font-size: 1.2rem;}
          .lead-row {display:flex; align-items:center; gap:10px; margin:2px 0;}
          .lead-rank {color:#8b949e; width:1.6rem; font-weight:700;}
          .lead-name {flex:0 0 9rem; font-weight:600;}
          .lead-bar {flex:1; background:#2a2d31; border-radius:6px; height:14px; overflow:hidden;}
          .lead-fill {height:100%; background:linear-gradient(90deg,#f54e00,#ff8a4d);}
          .lead-score {width:3.2rem; text-align:right; color:#e7e9ea; font-variant-numeric:tabular-nums;}
          .why-card {background:#1d1f23; border:1px solid #2a2d31; border-radius:10px;
                     padding:10px 12px; height:100%;}
          .why-card h4 {margin:0 0 .2rem 0; font-size:1rem;}
          .why-card .rank {color:#f54e00; font-weight:700;}
          .sig-chip {color:#8b949e; font-size:.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(ctx: dict) -> None:
    st.title("⚡ PostHog Engineering Impact")
    sub = "Top engineers over the last 90 days, scored across four signals."
    if ctx["start"] and ctx["end"]:
        sub += (f"  ·  {ctx['pr_count']:,} PRs from {ctx['contributors']} "
                f"contributors ({ctx['start']} → {ctx['end']})")
    st.caption(sub)

    cols = st.columns(4)
    for col, (key, label, desc) in zip(cols, SIGNALS):
        col.markdown(
            f"**{label}** &nbsp;`{WEIGHTS[key]:.0%}`<br>"
            f"<span class='sig-chip'>{desc}</span>",
            unsafe_allow_html=True,
        )


def render_leaderboard(top: pd.DataFrame) -> None:
    st.subheader("🏆 Top 5 leaderboard")
    best = top["impact_score"].max() or 1.0
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        pct = 100 * row["impact_score"] / best
        st.markdown(
            f"<div class='lead-row'>"
            f"<span class='lead-rank'>{rank}</span>"
            f"<span class='lead-name'>{row['engineer']}</span>"
            f"<span class='lead-bar'><span class='lead-fill' "
            f"style='width:{pct:.1f}%'></span></span>"
            f"<span class='lead-score'>{row['impact_score']:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_cards(top: pd.DataFrame) -> None:
    st.subheader("📊 Signal breakdown & why")
    cols = st.columns(len(top))
    for rank, (col, (_, row)) in enumerate(zip(cols, top.iterrows()), start=1):
        with col:
            st.markdown(
                f"<div class='why-card'><h4><span class='rank'>#{rank}</span> "
                f"{row['engineer']}</h4>"
                f"<div class='sig-chip'>impact {row['impact_score']:.3f}</div></div>",
                unsafe_allow_html=True,
            )
            st.altair_chart(signal_chart(row), use_container_width=True)
            for bullet in top_bullets(row):
                st.markdown(f"- {bullet}")


def render_methodology() -> None:
    with st.expander("📐 Methodology — how each signal is computed"):
        st.markdown(
            f"""
Every engineer is scored on four signals. Each raw signal is **min-max
normalized to 0–1 across all engineers**, then combined into the composite:

> **impact = {WEIGHTS['velocity_quality']:.0%} · Velocity×Quality
> + {WEIGHTS['team_multiplier']:.0%} · Team Multiplier
> + {WEIGHTS['leverage']:.0%} · Leverage
> + {WEIGHTS['consistency']:.0%} · Consistency**

**1. Velocity × Quality** — PRs merged per week, multiplied by a quality factor
`1 / (1 + avg_review_rounds)`, where a *round* is a `CHANGES_REQUESTED` review on
the author's PRs. Shipping fast with little rework scores highest; lots of
requested changes drags the score down.

**2. Team Multiplier** — the number of *other people's* PRs an engineer reviewed
(self-reviews excluded), multiplied by an average speed bonus
`1 / (1 + hours/24)` measured from PR creation to that reviewer's first review.
Rewards people who unblock teammates quickly.

**3. Leverage** — each file's *centrality* is the number of distinct engineers
who touch it. An engineer's leverage is the mean centrality of the files in
their PRs, so working on widely-shared, high-traffic code scores higher than
working in an isolated corner.

**4. Consistency** — the normalized Shannon entropy of the engineer's weekly
activity (PRs, reviews, and commits) across the 90-day window split into weekly
buckets. An even spread approaches **1.0**; a single burst of activity
approaches **0**.

*Scores are **relative to this cohort** — they rank impact among these
engineers, not on an absolute scale. Bots (`*[bot]`) are excluded throughout.
The "why" bullets above are generated from raw counts (PRs merged, review
rounds, PRs reviewed, turnaround hours, active weeks), so you can trace each
ranking back to the underlying activity.*
            """
        )


def main() -> None:
    st.set_page_config(page_title="PostHog Engineering Impact",
                       page_icon="⚡", layout="wide")
    inject_css()

    scores = load_scores()
    if scores is None or scores.empty:
        st.title("⚡ PostHog Engineering Impact")
        st.error(
            "No `data/impact_scores.csv` found. Run the pipeline first:\n\n"
            "```\npython fetch_data.py\npython analyze.py\n```"
        )
        return

    ctx = pr_context(load_prs())
    top = scores.sort_values("impact_score", ascending=False).head(TOP_N)

    render_header(ctx)
    st.divider()
    left, right = st.columns([1, 2], gap="large")
    with left:
        render_leaderboard(top)
    with right:
        render_cards(top)
    st.divider()
    render_methodology()


if __name__ == "__main__":
    main()
