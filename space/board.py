"""Turning stored submissions into what people are shown.

Everything here is a pure function of submissions and the competition config.
Two rules run through all of it: participants see a dataset's codename, never
its name, and they never see the organizer-only parts of a result.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from datetime import datetime, timedelta

from competition import Competition
from space.state import FAILED, QUEUED, SCORED, Submission

PIPELINE_ERROR = "the leaderboard could not process this submission; it does not count against your limit"


def _labels(competition: Competition) -> dict[str, str]:
    return {d.name: d.label for d in competition.datasets.dev + competition.datasets.eval}


def _relabel(text: str | None, competition: Competition) -> str | None:
    """A message written for a participant, with each dataset's name replaced by its label.
    The evaluator knows datasets by name only, and names them in what it says."""
    for name, label in sorted(_labels(competition).items(), key=lambda pair: -len(pair[0])):
        if text and label != name:
            text = re.sub(rf"\b{re.escape(name)}\b", label, text)
    return text


def declared_fields(submission: Submission, competition: Competition) -> dict[str, str]:
    """A submission's categories, limited to the ones the competition declares
    today: a stored submission can carry categories that have since been dropped."""
    names = {field.name for field in competition.submission.fields}
    return {name: value for name, value in submission.fields.items() if name in names}


def public_results(results: dict, competition: Competition) -> dict:
    """A results envelope as a participant may see it."""
    labels = _labels(competition)
    return {
        "metrics": results["metrics"],
        "breakdown": [{"name": labels.get(row["name"], row["name"]), "metrics": row["metrics"]} for row in results["breakdown"]],
        "message": _relabel(results.get("message"), competition),
    }


def failure(submission: Submission, competition: Competition) -> str | None:
    """Why a failed submission failed, in words meant for its author."""
    if submission.status != FAILED:
        return None
    why = _why(submission, competition)
    if submission.excused:
        return f"{why}\n\nThe organizers have excused this attempt, so it does not count against your limit: {submission.excused}"
    return why


def _why(submission: Submission, competition: Competition) -> str:
    if submission.results and not submission.results["ok"]:
        return _relabel(submission.results["message"], competition)
    if submission.report and not submission.report["ok"]:
        failed = submission.report["datasets"][-1]
        text = f"{_labels(competition).get(failed['name'], failed['name'])}: {failed['reason']}"
        if failed.get("log_tail"):
            return text + "\n\n" + failed["log_tail"]
        return text + ". Run `python submit.py --local` to see the full error."
    return PIPELINE_ERROR


def counts_against_limit(submission: Submission) -> bool:
    """A submission uses up an attempt unless the pipeline itself broke or an
    organizer excused its failure."""
    return not (submission.status == FAILED and (submission.error or submission.excused))


def describe(submission: Submission, competition: Competition, *, progress: dict | None = None, queued_ahead: int = 0) -> dict:
    """One submission as its own team may see it."""
    if progress:
        progress = {**progress, "dataset": _labels(competition).get(progress["dataset"], progress["dataset"])}
    return {
        "id": submission.id,
        "team": submission.team,
        "status": submission.status,
        "submitted_at": submission.submitted_at,
        "fields": declared_fields(submission, competition),
        "queued_ahead": queued_ahead if submission.status == QUEUED else 0,
        "progress": progress,
        "seconds": submission.report["seconds"] if submission.report else None,
        "results": public_results(submission.results, competition) if submission.status == SCORED else None,
        "error": failure(submission, competition),
        "flagged": submission.flagged is not None,
        "flag_note": submission.flagged or None,  # the organizers' note, which only the team is shown
    }


def rate_limit(mine: list[Submission], team: str, competition: Competition, now: datetime) -> dict:
    """How much of its allowance a team has used. ``mine`` is every submission
    the team has made, oldest first; the window is the last ``window_hours``."""
    limit = competition.submission.rate_limit
    cutoff = (now - timedelta(hours=limit.window_hours)).isoformat(timespec="seconds")
    used = [s for s in mine if s.submitted_at >= cutoff and counts_against_limit(s)]
    status = {"max": limit.max, "window_hours": limit.window_hours, "used": len(used), "remaining": None, "retry_at": None}
    if limit.max <= 0 or team in limit.exempt:
        return status
    status["remaining"] = max(0, limit.max - len(used))
    if status["remaining"] == 0:
        # The window slides: an attempt comes back when the oldest one ages out.
        oldest = datetime.fromisoformat(used[-limit.max].submitted_at)
        status["retry_at"] = (oldest + timedelta(hours=limit.window_hours)).isoformat(timespec="seconds")
    return status


def by_phase(submissions: list[Submission], starts: list[str]) -> list[list[Submission]]:
    """Submissions grouped by the phase each was made in, phase 1 first.
    ``starts`` is when each phase after the first began, oldest first: a
    submission belongs to the last phase that had begun when it was made."""
    phases: list[list[Submission]] = [[] for _ in range(len(starts) + 1)]
    for submission in submissions:
        phases[bisect_right(starts, submission.submitted_at)].append(submission)
    return phases


def _rank_key(metrics: dict, competition: Competition) -> tuple:
    """Sorts better results first. A missing metric sorts after every value."""
    key = []
    for name in competition.leaderboard.rank_by:
        value = metrics.get(name)
        if value is None:
            key.append((1, 0.0))
        else:
            key.append((0, value if name in competition.leaderboard.lower_is_better else -value))
    return tuple(key)


def leaderboard(submissions: list[Submission], competition: Competition) -> list[dict]:
    """Every scored submission, best first. ``best`` marks each team's top row;
    ``rank`` numbers the best rows of competing teams (baselines are shown in
    place but not ranked)."""
    scored = [s for s in submissions if s.status == SCORED]
    scored.sort(key=lambda s: (_rank_key(s.results["metrics"], competition), s.submitted_at))

    rows, seen, rank = [], set(), 0
    for submission in scored:
        best = submission.team not in seen
        seen.add(submission.team)
        baseline = submission.team in competition.leaderboard.baselines
        if best and not baseline:
            rank += 1
        rows.append(
            {
                "id": submission.id,
                "team": submission.team,
                "submitted_at": submission.submitted_at,
                "fields": declared_fields(submission, competition),
                "seconds": submission.report["seconds"] if submission.report else None,
                "baseline": baseline,
                "best": best,
                "flagged": submission.flagged is not None,
                "rank": rank if best and not baseline else None,
                **public_results(submission.results, competition),
            }
        )
    return rows
