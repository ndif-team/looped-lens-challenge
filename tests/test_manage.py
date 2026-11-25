"""The organizers' commands that change stored submissions and teams, against
moto's DynamoDB and S3. Each is run the way `python manage.py <command>` runs
it, with what the organizer would type at its question."""

from argparse import Namespace

import pytest

import manage
from space import board
from space.state import FAILED
from test_space import client, competition, pipeline_wrote, report, score, state, submit  # noqa: F401  (fixtures)

ADA = {"X-NDIF-API-Key": "key-ada"}


@pytest.fixture
def operate(monkeypatch, capsys, competition, state):
    """Run one command on the test state. Returns its exit code and what it printed."""
    monkeypatch.setattr(manage, "deployed_state", lambda competition: (state, {}))

    def operate(command, typed: str = "", **args) -> tuple[int, str]:
        monkeypatch.setattr("builtins.input", lambda question: typed)
        try:
            code, said = command(competition, Namespace(**args)), ""
        except SystemExit as stop:
            code, said = 1, str(stop.code)
        return code, capsys.readouterr().out + said

    return operate


def test_an_excused_failure_does_not_count_and_says_why(operate, client, state, competition):
    competition.submission.rate_limit.max = 1
    created = submit(client).json()
    pipeline_wrote(state, created["id"], FAILED, report=report(ok=False, reason="exited with code 1"))
    assert submit(client).status_code == 429

    code, said = operate(manage.excuse, id=created["id"], reason="the proxy restarted during this run")
    assert code == 0 and "no longer counts against the limit of Ada's Team" in said

    error = client.get(f"/api/submissions/{created['id']}", headers=ADA).json()["error"]
    assert error.startswith("Dataset A: exited with code 1.") and error.endswith("it does not count against your limit: the proxy restarted during this run")
    assert "(excused)" in manage.one_line(state.submission(created["id"]), competition)
    assert submit(client).status_code == 200


def test_only_a_failed_submission_can_be_excused(operate, client, state):
    created = submit(client).json()
    score(state, created["id"], 0.9)
    code, said = operate(manage.excuse, id=created["id"], reason="no reason")
    assert code == 1 and "is scored: only a failed submission can be excused" in said
    assert state.submission(created["id"]).excused is None

    code, said = operate(manage.excuse, id="20260101T000000-00000000", reason="no reason")
    assert code == 1 and "there is no submission" in said


def test_flag_marks_a_submission_and_unflag_clears_it(operate, client, state, competition):
    created = submit(client).json()
    score(state, created["id"], 0.9)
    listed = lambda: manage.one_line(state.submission(created["id"]), competition)

    code, said = operate(manage.flag, id=created["id"], note="it reads no activations")
    assert code == 0 and "of Ada's Team is flagged for review" in said
    assert state.submission(created["id"]).flagged == "it reads no activations" and listed().endswith("(flagged: it reads no activations)")
    code, _ = operate(manage.flag, id=created["id"], note="")  # flagged again, this time with no note
    assert code == 0 and state.submission(created["id"]).flagged == "" and listed().endswith("(flagged)")

    code, said = operate(manage.unflag, id=created["id"])
    assert code == 0 and "is no longer flagged" in said and state.submission(created["id"]).flagged is None and "flagged" not in listed()
    code, said = operate(manage.unflag, id=created["id"])
    assert code == 1 and "is not flagged" in said
    code, said = operate(manage.flag, id="20260101T000000-00000000", note="")
    assert code == 1 and "there is no submission" in said and state.submissions() == [state.submission(created["id"])]


def test_remove_deletes_the_record_and_every_file(operate, client, state, competition):
    kept, gone = submit(client, key="key-bob", team="Bob").json(), submit(client).json()
    for created in (kept, gone):
        score(state, created["id"], 0.9)
        state.s3.put_object(Bucket="bucket", Key=f"runs/{created['id']}/outputs.tar.gz", Body=b"outputs")
    before = state.files(gone["id"])
    assert f"submissions/{gone['id']}.zip" in before and f"credentials/{gone['id']}.json" in before and len(before) == 5

    # Anything but the id leaves it alone.
    code, said = operate(manage.remove, typed="yes", id=gone["id"])
    assert code == 1 and "Nothing was deleted." in said and state.files(gone["id"]) == before

    code, said = operate(manage.remove, typed=gone["id"], id=gone["id"])
    assert code == 0 and f"{gone['id']} removed" in said and "Ada's Team" in said
    assert state.submission(gone["id"]) is None and state.files(gone["id"]) == []
    assert [row["team"] for row in board.leaderboard(state.submissions(), competition)] == ["Bob"]
    assert len(state.files(kept["id"])) == 5
    # The team is still registered, and the removed attempt no longer counts.
    me = client.get("/api/me", headers=ADA).json()
    assert me["team"] == "Ada's Team" and me["rate_limit"]["used"] == 0


def test_a_submission_in_flight_is_not_removed(operate, client, state):
    created = submit(client).json()
    code, said = operate(manage.remove, typed=created["id"], id=created["id"])
    assert code == 1 and f"is running. Stop it first: python manage.py cancel {created['id']}" in said
    assert state.submission(created["id"]) is not None


def test_rename_team_reaches_past_submissions_and_frees_the_old_name(operate, client, state, competition):
    competition.leaderboard.baselines = ["Ada's Team"]
    created = submit(client).json()
    score(state, created["id"], 0.9)
    submit(client, key="key-bob", team="Bob")

    code, said = operate(manage.rename_team, name="ada's team", new="  The   Lovelaces ")
    assert code == 0 and "\"Ada's Team\" is now 'The Lovelaces'" in said
    assert "listed in leaderboard.baselines: change it in competition.prod.yaml" in said
    assert state.submission(created["id"]).team == "The Lovelaces"
    assert client.get("/api/me", headers=ADA).json()["team"] == "The Lovelaces"
    assert state.team_account("Ada's Team") is None and state.team_account("the lovelaces") == "ada@example.org"

    # A name another account holds is refused, and so is one that cannot be a name.
    code, said = operate(manage.rename_team, name="The Lovelaces", new="bob")
    assert code == 1 and "the team name 'bob' is taken" in said
    code, said = operate(manage.rename_team, name="The Lovelaces", new="x")
    assert code == 1 and "2 to 40 printable characters" in said
    code, said = operate(manage.rename_team, name="Nobody", new="Somebody")
    assert code == 1 and "there is no team named 'Nobody'" in said
    assert state.team_of("ada@example.org") == "The Lovelaces"

    # Only the casing changes: the name stays the team's.
    code, _ = operate(manage.rename_team, name="The Lovelaces", new="the lovelaces")
    assert code == 0 and state.team_of("ada@example.org") == "the lovelaces" and state.team_account("The Lovelaces") == "ada@example.org"


def test_remove_team_deletes_its_submissions_and_frees_its_name(operate, client, state, competition):
    first = submit(client).json()
    score(state, first["id"], 0.9)
    second = submit(client).json()
    pipeline_wrote(state, second["id"], FAILED, report=report(ok=False, reason="exited with code 1"))
    other = submit(client, key="key-bob", team="Bob").json()
    score(state, other["id"], 0.5)

    code, said = operate(manage.remove_team, typed="Ada", name="Ada's Team")
    assert code == 1 and "Nothing was deleted." in said
    assert state.team_of("ada@example.org") == "Ada's Team" and len(state.submissions()) == 3

    code, said = operate(manage.remove_team, typed="Ada's Team", name="ada's team")
    assert code == 0 and first["id"] in said and second["id"] in said and "removed, with 2 submissions" in said
    assert [s.id for s in state.submissions()] == [other["id"]]
    assert state.files(first["id"]) == state.files(second["id"]) == [] and state.files(other["id"])
    assert state.team_of("ada@example.org") is None and state.team_account("Ada's Team") is None

    # The account is not barred: its next submission registers a team again, under any free name.
    again = submit(client, team="Ada's Team")
    assert again.status_code == 200 and again.json()["team"] == "Ada's Team"


def test_a_team_with_a_submission_in_flight_is_not_removed(operate, client, state):
    created = submit(client).json()
    code, said = operate(manage.remove_team, typed="Ada's Team", name="Ada's Team")
    assert code == 1 and f"Stop it first: python manage.py cancel {created['id']}" in said
    assert state.team_of("ada@example.org") == "Ada's Team"


def test_new_phase_asks_first_and_then_starts_one(operate, state):
    code, said = operate(manage.new_phase, typed="yes")
    assert code == 1 and "Nothing was changed" in said and state.phase_starts() == []

    code, said = operate(manage.new_phase, typed="2")
    assert code == 0 and "Phase 2 has begun" in said and len(state.phase_starts()) == 1
