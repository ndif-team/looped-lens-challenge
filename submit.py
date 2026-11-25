#!/usr/bin/env python3
"""Run your method locally, submit it, and follow its progress.

    python submit.py --local                 run on the public dev datasets, on this machine
    python submit.py --team NAME --wait      submit submission/ and wait for the scores
    python submit.py --status                your team's submissions and remaining attempts

Your method is everything under ``submission/``: an entry file (``main.py`` or
``main.ipynb``) plus whatever it needs beside it. ``--local`` runs it exactly as
the leaderboard does, one process per dataset, and scores it with the same
evaluator, so a method that works locally works when submitted.

A submission is identified by your NDIF API key (``--ndif-api-key``,
``$NDIF_API_KEY``, or the key saved in nnsight). Your team is the NDIF account
the key belongs to; ``--team`` names it the first time you submit.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import competition as comp
import evaluator
import runner

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".leaderboard"  # local runs keep their data and outputs here
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
POLL_SECONDS = 10


# ------------------------------------------------------------ credentials --


def _clean(value: str | None) -> str | None:
    """A credential, or None when it is missing, blank, or the text "None"
    (which is what an unset shell substitution tends to leave behind)."""
    value = (value or "").strip()
    return None if value.lower() in ("", "none", "null") else value


def ndif_api_key(given: str | None) -> str | None:
    if key := _clean(given) or _clean(os.environ.get("NDIF_API_KEY")):
        return key
    try:
        from nnsight import CONFIG

        return _clean(CONFIG.API.APIKEY)
    except Exception:
        return None


def hf_token(given: str | None) -> str | None:
    if token := _clean(given) or _clean(os.environ.get("HF_TOKEN")):
        return token
    try:
        from huggingface_hub import get_token

        return _clean(get_token())
    except Exception:
        return None


# ---------------------------------------------------------------- display --


def _format(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def show_results(results: dict, columns: list[str]) -> None:
    """Print a results envelope: headline metrics, then the breakdown rows."""
    names = [c for c in columns if c in results["metrics"]] + [m for m in results["metrics"] if m not in columns]
    if not results["ok"]:
        print(f"\n  not scored: {results['message']}")
        return
    width = max([len("overall")] + [len(row["name"]) for row in results["breakdown"]])
    print("\n  " + " " * width + "".join(f"  {name:>10}" for name in names))
    print("  " + "overall".ljust(width) + "".join(f"  {_format(results['metrics'].get(name)):>10}" for name in names))
    for row in results["breakdown"]:
        print("  " + row["name"].ljust(width) + "".join(f"  {_format(row['metrics'].get(name)):>10}" for name in names))
    if results.get("message"):
        print(f"\n  {results['message']}")


# ------------------------------------------------------------------ local --


def run_local(competition: comp.Competition, submission: Path, only: list[str], args) -> int:
    """Run the submission on the dev datasets and score it, all on this machine."""
    manifest = comp.prepare(competition, "dev", WORK / "data", private=True, token=hf_token(args.hf_token))
    datasets, limits = runner.load_manifest(manifest)
    if only:
        if unknown := sorted(set(only) - {d.name for d in datasets}):
            sys.exit(f"no dev dataset named {unknown}; there are: {[d.name for d in datasets]}")
        datasets = [d for d in datasets if d.name in only]
    if not datasets:
        sys.exit("competition.yaml lists no dev datasets to run on")

    outputs = WORK / "outputs"
    shutil.rmtree(outputs, ignore_errors=True)

    # The method's process inherits this shell's environment, so a key saved in
    # nnsight or exported here already works; a key passed as a flag is added.
    env = {"NDIF_HOST": competition.ndif.host}
    if key := _clean(args.ndif_api_key):
        env["NDIF_API_KEY"] = key
    if token := _clean(args.hf_token):
        env["HF_TOKEN"] = token

    def announce(event: dict) -> None:
        if event["state"] == "running":
            print(f"\n── {event['dataset']}  ({event['index']}/{event['total']})", flush=True)

    limits["reveal_errors"] = False  # the method's output is already on the screen
    report = runner.run_submission(submission, datasets, outputs, env=env, on_progress=announce, **limits)
    if not report.ok:
        failed = report.datasets[-1]
        print(f"\n  {failed.name}: {failed.reason}")
        print(f"  full log: {outputs / (failed.name + '.log')}")
        return 1

    scored = [d for d in evaluator.load_manifest(manifest, outputs) if d.name in {r.name for r in report.datasets}]
    results = evaluator.score(scored)
    show_results(results, competition.leaderboard.columns)
    if trace := results["details"].get("traceback"):
        print("\n" + trace)  # a bug in evaluator/evaluate.py; on the leaderboard only organizers see this
    print(f"\n  outputs: {outputs}   ({report.seconds:.0f}s)")
    return 0 if results["ok"] else 1


# ----------------------------------------------------------------- submit --


def package(submission: Path, max_mb: int) -> bytes:
    """Zip the submission directory, entry file at the top level of the archive."""
    try:
        runner.find_entry(submission)  # fail here, not after the upload
    except ValueError as error:
        sys.exit(str(error))
    buffer = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission.rglob("*")):
            relative = path.relative_to(submission)
            if not path.is_file() or SKIP_DIRS & set(relative.parts) or path.suffix in SKIP_SUFFIXES:
                continue
            archive.write(path, relative.as_posix())
            count += 1
    data = buffer.getvalue()
    megabytes = len(data) / 1e6
    if megabytes > max_mb:
        sys.exit(f"the package is {megabytes:.0f} MB; the limit is {max_mb} MB")
    print(f"  packaged {count} files, {megabytes:.1f} MB")
    return data


class Client:
    """The leaderboard Space's API, authenticated by an NDIF key."""

    def __init__(self, url: str, key: str, token: str | None):
        import requests

        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-NDIF-API-Key"] = key
        if token:
            # Hugging Face checks this before the request reaches the Space; it
            # is what lets a member of the Space's organization use it while it
            # is still private. A public Space ignores it.
            self.session.headers["Authorization"] = f"Bearer {token}"

    def call(self, method: str, path: str, **kwargs) -> dict:
        import requests

        try:
            response = self.session.request(method, self.url + path, timeout=300, **kwargs)
        except requests.RequestException as error:
            sys.exit(f"could not reach the leaderboard at {self.url}: {error}")
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            sys.exit(f"the leaderboard said no ({response.status_code}): {detail}")
        return response.json()


def describe(submission: dict) -> str:
    """One line saying where a submission is."""
    line = status = submission["status"]
    if status == "queued":
        ahead = submission.get("queued_ahead") or 0
        line = f"queued ({ahead} ahead)" if ahead else "queued (next up)"
    elif status == "running" and (progress := submission.get("progress")):
        line = f"running {progress['dataset']} ({progress['index']}/{progress['total']})"
    elif status == "failed":
        line = f"failed: {submission.get('error') or 'no reason recorded'}"
    if submission.get("flagged"):  # the organizers are reviewing it
        line += "  [potentially invalid: flagged for review" + (f": {note}]" if (note := submission.get("flag_note")) else "]")
    return line


def wait(client: Client, submission_id: str, columns: list[str]) -> int:
    """Poll a submission until it is scored or has failed. Interrupting this
    only stops the watching; the submission carries on."""
    last = None
    while True:
        submission = client.call("GET", f"/api/submissions/{submission_id}")
        if (line := describe(submission)) != last:
            print(f"  {time.strftime('%H:%M:%S')}  {line}", flush=True)
            last = line
        if submission["status"] == "scored":
            show_results({"ok": True, **submission["results"]}, columns)
            return 0
        if submission["status"] == "failed":
            return 1
        time.sleep(POLL_SECONDS)


def submit(competition: comp.Competition, submission: Path, args) -> int:
    key = ndif_api_key(args.ndif_api_key)
    if not key:
        sys.exit("an NDIF API key is needed: --ndif-api-key, $NDIF_API_KEY, or a key saved in nnsight")
    fields = {f.name: getattr(args, f.name) for f in competition.submission.fields}
    if missing := [name for name, value in fields.items() if value is None]:
        sys.exit("this competition asks every submission to declare: " + ", ".join(f"--{name}" for name in missing))

    token = hf_token(args.hf_token)
    client = Client(args.space_url or competition.space.url, key, token)
    headers = {"X-HF-Token": token} if token else {}  # forwarded into the run, for gated model repos
    created = client.call(
        "POST",
        "/api/submissions",
        files={"package": ("submission.zip", package(submission, competition.submission.max_size_mb), "application/zip")},
        data={"team": args.team or "", "fields": json.dumps(fields)},
        headers=headers,
    )
    print(f"  submitted as {created['team']}: {created['id']}")
    if args.wait:
        return wait(client, created["id"], competition.leaderboard.columns)
    print(f"  follow it with: python submit.py --status {created['id']}")
    return 0


def status(competition: comp.Competition, args) -> int:
    key = ndif_api_key(args.ndif_api_key)
    if not key:
        sys.exit("an NDIF API key is needed: --ndif-api-key, $NDIF_API_KEY, or a key saved in nnsight")
    client = Client(args.space_url or competition.space.url, key, hf_token(args.hf_token))
    if args.status:
        return wait(client, args.status, competition.leaderboard.columns)

    me = client.call("GET", "/api/me")
    if not me["team"]:
        print("  this NDIF account has not submitted yet")
        return 0
    limit = me["rate_limit"]
    allowance = "unlimited" if limit["remaining"] is None else f"{limit['remaining']} of {limit['max']} left in this {limit['window_hours']:g}h window"
    print(f"  team {me['team']}   submissions: {allowance}")
    primary = competition.leaderboard.rank_by[0]
    for submission in me["submissions"]:
        score = _format(submission["results"]["metrics"].get(primary)) if submission.get("results") else ""
        print(f"  {submission['submitted_at'][:19]}  {submission['id']}  {describe(submission)}  {score}")
    return 0


def main() -> int:
    competition = comp.load()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run on the dev datasets on this machine instead of submitting")
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME", help="with --local: run only this dev dataset (repeatable)")
    parser.add_argument("--wait", action="store_true", help="after submitting, stay and print progress and scores")
    parser.add_argument("--status", nargs="?", const="", metavar="ID", help="list your submissions, or follow one by id")
    parser.add_argument("--team", help="your team's display name; needed the first time you submit")
    for field in competition.submission.fields:
        parser.add_argument(f"--{field.name}", choices=field.choices, help=field.help or f"this submission's {field.name}")
    parser.add_argument("--submission", type=Path, default=ROOT / "submission", help="the directory to run or submit (default: submission/)")
    parser.add_argument("--ndif-api-key", help="default: $NDIF_API_KEY, else the key saved in nnsight")
    parser.add_argument("--hf-token", help="for gated model repos; default: $HF_TOKEN, else your saved Hugging Face login")
    parser.add_argument("--space-url", help=f"default: {competition.space.url}")
    args = parser.parse_args()

    print(f"\n  {competition.title}")
    submission = args.submission.resolve()
    if args.status is not None:
        return status(competition, args)
    if args.local:
        return run_local(competition, submission, args.dataset, args)
    return submit(competition, submission, args)


if __name__ == "__main__":
    sys.exit(main())
