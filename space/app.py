"""The HTTP API and the page.

    GET  /                       the leaderboard page
    GET  /api/config             what the page needs to know about the competition
    GET  /api/leaderboard        a phase's scored submissions, ranked
    POST /api/submissions        submit a package            (NDIF key)
    GET  /api/submissions/{id}   one of your submissions     (NDIF key)
    DELETE /api/submissions/{id} withdraw one of them        (NDIF key)
    GET  /api/me                 your team and its history   (NDIF key)

A caller is identified by the NDIF API key in ``X-NDIF-API-Key``: NDIF says
which account the key belongs to, and that account's email is the team. The
email is never sent back to anyone; teams are shown by the name they chose.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

import competition as comp
from space import board, ndif
from space.state import IN_FLIGHT, QUEUED, NameTaken, State, team_name, utcnow

STATIC = Path(__file__).parent / "static"
DISPATCH_SECONDS = 20  # how often to look for a free slot
MIRROR_SECONDS = 300
BOARD_CACHE_SECONDS = 15
IDENTITY_CACHE_SECONDS = 300


def create_app(competition: comp.Competition, state: State, *, whoami=ndif.whoami, mirror=None) -> FastAPI:
    identities: dict[str, tuple[float, str]] = {}  # hash of an accepted key -> (when checked, its email)
    board_cache: dict = {"at": 0.0, "phases": [[]]}  # each phase's board, phase 1 first

    # ---------------------------------------------------------- background --

    async def every(seconds: float, action) -> None:
        while True:
            try:
                await run_in_threadpool(action)
            except Exception as error:  # a failed tick must not end the loop
                print(f"[space] {action.__name__} failed: {error!r}", flush=True)
            await asyncio.sleep(seconds)

    def sync_mirror() -> None:
        mirror.sync(state)

    @contextlib.asynccontextmanager
    async def background(_: FastAPI):
        # The dispatch timer is what notices a slot freed by a finished run.
        tasks = [asyncio.create_task(every(DISPATCH_SECONDS, state.dispatch))]
        if mirror is not None:
            tasks.append(asyncio.create_task(every(MIRROR_SECONDS, sync_mirror)))
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(title=competition.title, lifespan=background)

    # ------------------------------------------------------------ identity --

    def identify(api_key: str | None) -> str:
        """The email of the NDIF account behind a key, after checking the key
        may take part. Raises the HTTP error to send back otherwise."""
        if not api_key:
            raise HTTPException(401, "send your NDIF API key in the X-NDIF-API-Key header")
        digest = hashlib.sha256(api_key.encode()).hexdigest()
        cached = identities.get(digest)
        if cached and time.monotonic() - cached[0] < IDENTITY_CACHE_SECONDS:
            return cached[1]

        try:
            account = whoami(competition.ndif.host, api_key)
        except ndif.NdifUnavailable:
            raise HTTPException(503, "could not reach NDIF to check your API key; try again shortly")
        if not account["email"]:
            raise HTTPException(401, "NDIF does not recognise this API key")
        required = competition.ndif.required_tags
        if required and not set(required) & set(account["tags"]):
            raise HTTPException(403, f"this NDIF account is not registered for {competition.title}")

        # Only an accepted key is remembered. A refusal is asked again every
        # time, so a key that has just been tagged works at once.
        identities[digest] = (time.monotonic(), account["email"])
        return account["email"]

    def queued_ahead(submission, everyone) -> int:
        return sum(s.status == QUEUED and s.id < submission.id for s in everyone)

    # -------------------------------------------------------------- public --

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config() -> dict:
        return {
            "title": competition.title,
            "intro": competition.leaderboard.intro,
            "accent": competition.leaderboard.accent,
            "links": competition.leaderboard.links,
            "columns": competition.leaderboard.columns,
            "rank_by": competition.leaderboard.rank_by,
            "lower_is_better": competition.leaderboard.lower_is_better,
            "fields": [f.model_dump() for f in competition.submission.fields],
        }

    @app.get("/api/leaderboard")
    def leaderboard(phase: int | None = None) -> dict:
        """One phase's board: the current phase, unless an earlier one is asked for."""
        if time.monotonic() - board_cache["at"] > BOARD_CACHE_SECONDS:
            made = board.by_phase(state.submissions(), state.phase_starts())
            board_cache.update(at=time.monotonic(), phases=[board.leaderboard(submissions, competition) for submissions in made])
        boards = board_cache["phases"]
        phase = len(boards) if phase is None else phase
        if not 1 <= phase <= len(boards):
            raise HTTPException(404, f"there is no phase {phase}")
        return {"phase": phase, "phases": len(boards), "rows": boards[phase - 1]}

    # --------------------------------------------------------- participant --

    @app.post("/api/submissions")
    async def submit(
        request: Request,
        package: UploadFile = File(...),
        team: str = Form(""),
        fields: str = Form("{}"),
        api_key: str | None = Header(None, alias="X-NDIF-API-Key"),
        hf_token: str | None = Header(None, alias="X-HF-Token"),
    ) -> dict:
        # Everything that can refuse a submission happens before anything is
        # stored, so a refusal never uses up an attempt.
        limit_bytes = competition.submission.max_size_mb * 1_000_000
        if int(request.headers.get("content-length", 0)) > limit_bytes + 1_000_000:
            raise HTTPException(413, f"the package is over the {competition.submission.max_size_mb} MB limit")
        email = await run_in_threadpool(identify, api_key)

        try:
            chosen = json.loads(fields)
        except ValueError:
            raise HTTPException(400, "`fields` must be a JSON object")
        for declared in competition.submission.fields:
            if chosen.get(declared.name) not in declared.choices:
                raise HTTPException(400, f"`{declared.name}` must be one of {declared.choices}")
        chosen = {f.name: chosen[f.name] for f in competition.submission.fields}

        data = await package.read()
        if len(data) > limit_bytes:
            raise HTTPException(413, f"the package is over the {competition.submission.max_size_mb} MB limit")
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise HTTPException(400, "the package is not a zip file")

        def accept() -> dict:
            name = state.team_of(email)
            if name is None:
                name = team_name(team)
                if name is None:
                    raise HTTPException(400, "this is your first submission: choose a team name of 2 to 40 characters with --team")
                try:
                    state.register_team(email, name)
                except NameTaken:
                    raise HTTPException(409, f"the team name {name!r} is taken; choose another with --team")

            mine = [s for s in state.submissions() if s.email == email]
            if any(s.status in IN_FLIGHT for s in mine):
                raise HTTPException(409, "you already have a submission queued or running; wait for it to finish")
            allowance = board.rate_limit(mine, name, competition, utcnow())
            if allowance["remaining"] == 0:
                raise HTTPException(429, f"submission limit reached ({allowance['max']} per {allowance['window_hours']:g}h); the next one opens at {allowance['retry_at']} UTC")

            credentials = {"NDIF_API_KEY": api_key, "NDIF_HOST": competition.ndif.host, "HF_TOKEN": hf_token}
            submission = state.create(email, name, chosen, data, credentials)
            state.dispatch()
            return {"id": submission.id, "team": name}

        return await run_in_threadpool(accept)

    def own(submission_id: str, api_key: str | None):
        """One of the caller's submissions that it has not withdrawn. Raises
        the HTTP error to send back otherwise."""
        email = identify(api_key)
        submission = state.submission(submission_id)
        if submission is None or submission.email != email or submission.withdrawn:
            raise HTTPException(404, "no such submission for this NDIF account")
        return submission

    @app.get("/api/submissions/{submission_id}")
    def one(submission_id: str, api_key: str | None = Header(None, alias="X-NDIF-API-Key")) -> dict:
        submission = own(submission_id, api_key)
        return board.describe(
            submission,
            competition,
            progress=state.progress(submission.id) if submission.status == "running" else None,
            queued_ahead=queued_ahead(submission, state.submissions()) if submission.status == QUEUED else 0,
        )

    @app.delete("/api/submissions/{submission_id}")
    def withdraw(submission_id: str, api_key: str | None = Header(None, alias="X-NDIF-API-Key")) -> dict:
        """Take one of the caller's finished submissions off the board and out
        of its own list, for good. It is kept, and still counts against the limit."""
        submission = own(submission_id, api_key)
        if submission.status in IN_FLIGHT:
            raise HTTPException(409, f"this submission is {submission.status}; it can be withdrawn once it has finished")
        state.withdraw(submission.id)
        board_cache["at"] = 0.0  # the board is worked out again on its next request
        return {"id": submission.id}

    @app.get("/api/me")
    def me(api_key: str | None = Header(None, alias="X-NDIF-API-Key")) -> dict:
        email = identify(api_key)
        name = state.team_of(email)
        everyone = state.submissions()
        mine = [s for s in everyone if s.email == email]
        return {
            "team": name,
            "rate_limit": board.rate_limit(mine, name or "", competition, utcnow()),
            "submissions": [
                board.describe(
                    s,
                    competition,
                    progress=state.progress(s.id) if s.status == "running" else None,
                    queued_ahead=queued_ahead(s, everyone),
                )
                for s in reversed(mine)
                if not s.withdrawn
            ],
        }

    return app


def from_environment() -> FastAPI:
    """The app as the Space runs it (``uvicorn space.app:from_environment --factory``).

    ``manage.py deploy`` sets these on the Space: ``LEADERBOARD_COMPETITION``, the
    organizers' configuration as YAML text (it names the scored datasets, so it is
    a secret and not a file in the Space's repository); from the deployed stack,
    ``LEADERBOARD_TABLE``, ``LEADERBOARD_BUCKET`` and ``LEADERBOARD_STATE_MACHINE``;
    AWS credentials for the Space's own IAM user; and ``HF_TOKEN`` for the mirror.
    """
    competition = comp.parse(os.environ["LEADERBOARD_COMPETITION"])
    state = State(
        os.environ["LEADERBOARD_TABLE"],
        os.environ["LEADERBOARD_BUCKET"],
        os.environ["LEADERBOARD_STATE_MACHINE"],
        competition.aws.region,
        max_concurrent=competition.submission.max_concurrent,
        url_seconds=competition.runner.timeout_total + competition.evaluator.timeout + 3600,
    )
    mirror = None
    if competition.space.mirror and os.environ.get("HF_TOKEN"):
        from space.mirror import Mirror

        mirror = Mirror(competition.space.mirror, os.environ["HF_TOKEN"])
    return create_app(competition, state, mirror=mirror)
