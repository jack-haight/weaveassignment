"""Fetch engineering activity from the PostHog/posthog GitHub repo.

Pulls pull requests, PR reviews, PR review comments, and repo commits from the
last 90 days using PyGithub, caching each dataset to a JSON file under ./data.

The script is rate-limit aware: it proactively sleeps before the core API quota
is exhausted and retries individual calls when GitHub returns a primary or
secondary (abuse) rate-limit error.

Usage:
    python fetch_data.py                 # use cache if present, else fetch
    python fetch_data.py --refresh       # ignore cache, re-fetch everything
    python fetch_data.py --days 30       # look back 30 days instead of 90
    python fetch_data.py --max-prs 200   # cap PRs fetched (useful for testing)
    python fetch_data.py --commit-stats  # also fetch per-commit line stats (slow)

Requires a GitHub token in the GITHUB_TOKEN (or GH_TOKEN) environment variable,
which may be supplied via a .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from dotenv import load_dotenv

try:
    from github import Github, GithubException, RateLimitExceededException
    try:
        # PyGithub >= 1.55 exposes the Auth helper; fall back to the legacy
        # token-as-first-arg constructor if it is unavailable.
        from github import Auth
    except ImportError:  # pragma: no cover - depends on PyGithub version
        Auth = None
except ImportError:  # pragma: no cover - dependency missing
    print(
        "PyGithub is not installed. Install dependencies with:\n"
        "    pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise


REPO_NAME = "PostHog/posthog"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Sleep buffer (seconds) added on top of the reported reset time so we never
# wake up a hair too early and immediately trip the limit again.
RESET_BUFFER_SECONDS = 5
# Refill the quota whenever fewer than this many core requests remain.
MIN_CORE_REMAINING = 75
# Retry settings for transient errors / secondary rate limits.
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def iso(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to an ISO-8601 string, tolerating None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def login_of(user: Any) -> Optional[str]:
    """Safely pull the login from a (possibly None) NamedUser."""
    return getattr(user, "login", None) if user is not None else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)  # atomic-ish: never leave a half-written cache file
    print(f"  wrote {path.name} ({path.stat().st_size:,} bytes)")


def cache_is_fresh(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Keeps the core GitHub quota healthy and retries rate-limited calls."""

    def __init__(self, gh: Github):
        self.gh = gh

    def _seconds_until_reset(self, reset: datetime) -> float:
        if reset.tzinfo is None:
            reset = reset.replace(tzinfo=timezone.utc)
        return (reset - datetime.now(timezone.utc)).total_seconds()

    def _core_rate(self):
        """Return the core ``Rate``, compatible with PyGithub 1.x and 2.x.

        PyGithub 2.x returns a ``RateLimitOverview`` whose individual rates live
        under ``.resources`` (mirroring the GitHub API's response shape); older
        versions expose ``.core`` directly on the returned object.
        """
        overview = self.gh.get_rate_limit()
        resources = getattr(overview, "resources", overview)
        return resources.core

    def preflight(self) -> None:
        """Sleep until the quota refills if we are running low."""
        try:
            core = self._core_rate()
        except GithubException:
            # If even the rate-limit endpoint fails, just continue; the call
            # wrappers below will catch a real RateLimitExceededException.
            return
        if core.remaining > MIN_CORE_REMAINING:
            return
        wait = self._seconds_until_reset(core.reset) + RESET_BUFFER_SECONDS
        if wait > 0:
            print(
                f"  rate limit low ({core.remaining} left); "
                f"sleeping {wait:.0f}s until reset...",
                flush=True,
            )
            time.sleep(wait)

    def call(self, fn: Callable[[], Any], label: str = "") -> Any:
        """Run fn(), transparently handling rate-limit and transient errors."""
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            self.preflight()
            try:
                return fn()
            except RateLimitExceededException:
                # Primary rate limit: wait until the live reset time.
                try:
                    wait = self._seconds_until_reset(
                        self._core_rate().reset
                    ) + RESET_BUFFER_SECONDS
                except GithubException:
                    wait = backoff
                print(
                    f"  rate limit exceeded{f' on {label}' if label else ''}; "
                    f"sleeping {max(wait, 0):.0f}s (attempt {attempt})",
                    flush=True,
                )
                time.sleep(max(wait, backoff))
            except GithubException as exc:
                # 403/429 with a Retry-After is a secondary/abuse limit; 5xx is
                # transient. Back off and retry; re-raise anything else.
                status = getattr(exc, "status", None)
                if status not in (403, 429, 500, 502, 503):
                    raise
                retry_after = self._retry_after(exc)
                wait = retry_after if retry_after is not None else backoff
                print(
                    f"  transient GitHub error {status}"
                    f"{f' on {label}' if label else ''}; "
                    f"retrying in {wait:.0f}s (attempt {attempt})",
                    flush=True,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, 300)
        raise RuntimeError(f"giving up after {MAX_RETRIES} retries on {label!r}")

    @staticmethod
    def _retry_after(exc: GithubException) -> Optional[float]:
        headers = getattr(exc, "headers", None) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def iterate(self, paginated: Iterable, label: str = "") -> Iterable:
        """Yield items from a PaginatedList, handling per-page rate limits.

        PyGithub fetches pages lazily as the iterator advances, so a long list
        can trip the limit mid-iteration. We drive the iterator manually and
        retry each ``next()`` through ``call``.
        """
        iterator = iter(paginated)
        while True:
            try:
                item = self.call(lambda: next(iterator), label=label)
            except StopIteration:
                return
            yield item


# --------------------------------------------------------------------------- #
# Extraction (GitHub objects -> plain dicts)
# --------------------------------------------------------------------------- #
def extract_pr(pr: Any) -> dict:
    return {
        "number": pr.number,
        "title": pr.title,
        "state": pr.state,
        "draft": pr.draft,
        "author": login_of(pr.user),
        "created_at": iso(pr.created_at),
        "updated_at": iso(pr.updated_at),
        "closed_at": iso(pr.closed_at),
        "merged_at": iso(pr.merged_at),
        "merged": pr.merged,
        "merged_by": login_of(pr.merged_by),
        "additions": pr.additions,
        "deletions": pr.deletions,
        "changed_files": pr.changed_files,
        "commit_count": pr.commits,
        "comment_count": pr.comments,
        "review_comment_count": pr.review_comments,
        "base_ref": pr.base.ref if pr.base else None,
        "head_ref": pr.head.ref if pr.head else None,
        "labels": [label.name for label in pr.labels],
        "html_url": pr.html_url,
    }


def extract_review(pr_number: int, review: Any) -> dict:
    return {
        "id": review.id,
        "pr_number": pr_number,
        "author": login_of(review.user),
        "state": review.state,  # APPROVED / CHANGES_REQUESTED / COMMENTED / ...
        "submitted_at": iso(review.submitted_at),
        "body": review.body or "",
        "commit_id": review.commit_id,
        "html_url": review.html_url,
    }


def extract_review_comment(pr_number: int, comment: Any) -> dict:
    return {
        "id": comment.id,
        "pr_number": pr_number,
        "review_id": getattr(comment, "pull_request_review_id", None),
        "author": login_of(comment.user),
        "created_at": iso(comment.created_at),
        "updated_at": iso(comment.updated_at),
        "path": comment.path,
        "body": comment.body or "",
        "html_url": comment.html_url,
    }


def extract_commit(commit: Any, with_stats: bool) -> dict:
    data = commit.commit  # the underlying git commit metadata
    record = {
        "sha": commit.sha,
        # `commit.author` is the GitHub account (may be None); `data.author`
        # is the git author identity embedded in the commit itself.
        "author_login": login_of(commit.author),
        "author_name": data.author.name if data.author else None,
        "authored_at": iso(data.author.date) if data.author else None,
        "committer_login": login_of(commit.committer),
        "committed_at": iso(data.committer.date) if data.committer else None,
        "message": (data.message or "").splitlines()[0] if data.message else "",
        "html_url": commit.html_url,
    }
    if with_stats:
        stats = commit.stats  # triggers an extra GET per commit
        record.update(
            additions=stats.additions,
            deletions=stats.deletions,
            total=stats.total,
        )
    return record


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #
def fetch_prs_and_reviews(
    repo: Any,
    limiter: RateLimiter,
    cutoff: datetime,
    max_prs: Optional[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk PRs newest-first until we pass the cutoff, collecting reviews too."""
    prs: list[dict] = []
    reviews: list[dict] = []
    review_comments: list[dict] = []

    pulls = repo.get_pulls(state="all", sort="created", direction="desc")
    for pr in limiter.iterate(pulls, label="get_pulls"):
        created = pr.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created < cutoff:
            break  # list is desc by created date, so everything else is older

        record = extract_pr(pr)
        # Capture the file paths touched so analyze.py can compute file
        # centrality / leverage. This is an extra (paginated) call per PR.
        record["files"] = [
            f.filename
            for f in limiter.iterate(
                pr.get_files(), label=f"PR #{pr.number} files"
            )
        ]
        prs.append(record)

        for review in limiter.iterate(
            pr.get_reviews(), label=f"PR #{pr.number} reviews"
        ):
            reviews.append(extract_review(pr.number, review))

        for comment in limiter.iterate(
            pr.get_review_comments(), label=f"PR #{pr.number} review comments"
        ):
            review_comments.append(extract_review_comment(pr.number, comment))

        if len(prs) % 25 == 0:
            print(
                f"  ...{len(prs)} PRs, {len(reviews)} reviews, "
                f"{len(review_comments)} review comments so far",
                flush=True,
            )

        if max_prs is not None and len(prs) >= max_prs:
            print(f"  reached --max-prs limit ({max_prs}); stopping PR walk")
            break

    return prs, reviews, review_comments


def fetch_commits(
    repo: Any,
    limiter: RateLimiter,
    cutoff: datetime,
    with_stats: bool,
) -> list[dict]:
    commits: list[dict] = []
    paged = repo.get_commits(since=cutoff)
    for commit in limiter.iterate(paged, label="get_commits"):
        commits.append(extract_commit(commit, with_stats))
        if len(commits) % 100 == 0:
            print(f"  ...{len(commits)} commits so far", flush=True)
    return commits


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def get_client(token: str) -> Github:
    kwargs = dict(per_page=100, retry=3)
    if Auth is not None:
        return Github(auth=Auth.Token(token), **kwargs)
    return Github(token, **kwargs)  # legacy constructor


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=90, help="look-back window in days (default 90)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore existing cache files and re-fetch",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=None,
        help="cap the number of PRs fetched (handy for testing)",
    )
    parser.add_argument(
        "--commit-stats",
        action="store_true",
        help="fetch per-commit line stats (one extra API call per commit; slow)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv()

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        print(
            "No GitHub token found. Set GITHUB_TOKEN (or GH_TOKEN) in your "
            "environment or a .env file. A token raises the rate limit from "
            "60 to 5,000 requests/hour and is required to fetch this much data.",
            file=sys.stderr,
        )
        return 1

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    print(f"Fetching {REPO_NAME} activity since {cutoff.date()} ({args.days} days)")

    gh = get_client(token)
    limiter = RateLimiter(gh)
    repo = limiter.call(lambda: gh.get_repo(REPO_NAME), label="get_repo")

    prs_path = DATA_DIR / "prs.json"
    reviews_path = DATA_DIR / "reviews.json"
    review_comments_path = DATA_DIR / "review_comments.json"
    commits_path = DATA_DIR / "commits.json"

    pr_caches = (prs_path, reviews_path, review_comments_path)
    if not args.refresh and all(cache_is_fresh(p) for p in pr_caches):
        print("PR/review caches present; skipping (use --refresh to re-fetch).")
    else:
        print("Fetching pull requests, reviews, and review comments...")
        prs, reviews, review_comments = fetch_prs_and_reviews(
            repo, limiter, cutoff, args.max_prs
        )
        write_json(prs_path, prs)
        write_json(reviews_path, reviews)
        write_json(review_comments_path, review_comments)
        print(
            f"  totals: {len(prs)} PRs, {len(reviews)} reviews, "
            f"{len(review_comments)} review comments"
        )

    if not args.refresh and cache_is_fresh(commits_path):
        print("Commit cache present; skipping (use --refresh to re-fetch).")
    else:
        print("Fetching commits...")
        commits = fetch_commits(repo, limiter, cutoff, args.commit_stats)
        write_json(commits_path, commits)
        print(f"  totals: {len(commits)} commits")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
