"""The Space's API against moto's DynamoDB, S3 and Step Functions. NDIF is a
dict of keys; the state machine's writes are made by hand with `pipeline_wrote`."""

import io
import json
import zipfile
from datetime import datetime, timezone

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

import competition as comp
from space import app as space_app
from space import board, ndif
from space.state import EVALUATING, FAILED, QUEUED, RUNNING, SCORED, State, Submission

REGION = "us-east-1"
ACCOUNTS = {
    "key-ada": {"email": "ada@example.org", "tags": ["example-competition"]},
    "key-bob": {"email": "bob@example.org", "tags": ["example-competition", "other"]},
    "key-cy": {"email": "cy@example.org", "tags": ["example-competition"]},
    "key-untagged": {"email": "dee@example.org", "tags": ["other"]},
}


def fake_whoami(host: str, key: str) -> dict:
    if key == "key-outage":
        raise ndif.NdifUnavailable("timed out")
    return ACCOUNTS.get(key, {"email": None, "tags": []})


@pytest.fixture
def competition() -> comp.Competition:
    competition = comp.load()
    competition.submission.max_concurrent = 2
    competition.submission.rate_limit.max, competition.submission.rate_limit.window_hours = 3, 24
    # These tests are about the Space, not this competition, so they set up what they
    # exercise: a required tag, a category, two ranking metrics and codenames for two datasets.
    competition.ndif.required_tags = ["example-competition"]
    competition.submission.fields = [comp.SubmissionField(name="method", choices=["white-box", "black-box"])]
    competition.leaderboard.rank_by = competition.leaderboard.columns = ["accuracy", "f1"]
    competition.datasets.eval = scored_datasets()
    return competition


def scored_datasets() -> list[comp.Dataset]:
    """Scored datasets as the organizers configure them: a name an entry is told, and for two
    of the three a codename the leaderboard shows instead."""
    here = comp.Source(dir=".")
    return [comp.Dataset(name="dataset_a", codename="Dataset A", inputs=here), comp.Dataset(name="dataset_b", codename="Dataset B", inputs=here), comp.Dataset(name="dataset_c", inputs=here)]


@pytest.fixture
def state(monkeypatch, competition):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "testing")
    with mock_aws():
        boto3.client("dynamodb", region_name=REGION).create_table(
            TableName="table",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        boto3.client("s3", region_name=REGION).create_bucket(Bucket="bucket")
        machine = boto3.client("stepfunctions", region_name=REGION).create_state_machine(
            name="pipeline",
            definition=json.dumps({"StartAt": "Done", "States": {"Done": {"Type": "Succeed"}}}),
            roleArn="arn:aws:iam::123456789012:role/pipeline",
        )["stateMachineArn"]
        yield State("table", "bucket", machine, REGION, max_concurrent=competition.submission.max_concurrent, url_seconds=3600)


@pytest.fixture
def client(competition, state) -> TestClient:
    return TestClient(space_app.create_app(competition, state, whoami=fake_whoami))


def package(files: dict[str, str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in (files or {"main.py": "print('hi')"}).items():
            archive.writestr(name, text)
    return buffer.getvalue()


def submit(client, key="key-ada", team="Ada's Team", method="white-box", data=None, hf_token=None):
    headers = {"X-NDIF-API-Key": key} | ({"X-HF-Token": hf_token} if hf_token else {})
    return client.post(
        "/api/submissions",
        headers=headers,
        files={"package": ("submission.zip", data if data is not None else package(), "application/zip")},
        data={"team": team, "fields": json.dumps({"method": method})},
    )


def pipeline_wrote(state: State, submission_id: str, status: str, **attributes) -> None:
    """Write to a submission the way the state machine does."""
    names = {"#status": "status"} | {f"#{k}": k for k in attributes}
    values = {":status": status} | {f":{k}": v if isinstance(v, str) else json.dumps(v) for k, v in attributes.items()}
    state.table.update_item(
        Key={"pk": "submission", "sk": submission_id},
        UpdateExpression="SET " + ", ".join(f"{name} = :{name[1:]}" for name in names),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def report(ok=True, reason=None, log_tail=None) -> dict:
    datasets = [{"name": "dataset_a", "ok": ok, "seconds": 4.0, "reason": reason, "log_tail": log_tail}]
    return {"ok": ok, "seconds": 9.0, "datasets": datasets}


def results(accuracy: float, f1: float = 0.5) -> dict:
    return {
        "ok": True,
        "metrics": {"accuracy": accuracy, "f1": f1},
        "breakdown": [{"name": "dataset_a", "metrics": {"accuracy": accuracy}}, {"name": "dataset_b", "metrics": {"accuracy": accuracy}}],
        "message": None,
        "details": {"per_row": "organizers only"},
    }


def score(state, submission_id, accuracy, f1=0.5):
    pipeline_wrote(state, submission_id, SCORED, report=report(), results=results(accuracy, f1))


# ------------------------------------------------------------------ identity --


@pytest.mark.parametrize(
    "key, code, words",
    [
        (None, 401, "X-NDIF-API-Key"),
        ("key-nobody", 401, "does not recognise"),
        ("key-untagged", 403, "not registered for The Looped Lens Challenge"),
        ("key-outage", 503, "could not reach NDIF"),
    ],
)
def test_who_may_submit(client, state, key, code, words):
    response = client.post("/api/submissions", headers={"X-NDIF-API-Key": key} if key else {}, files={"package": ("s.zip", package())})
    assert response.status_code == code and words in response.json()["detail"]
    assert state.submissions() == []


def test_a_key_that_gains_the_tag_is_accepted_at_once(client, monkeypatch):
    refused = submit(client, key="key-untagged", team="Dee")
    assert refused.status_code == 403
    monkeypatch.setitem(ACCOUNTS, "key-untagged", {"email": "dee@example.org", "tags": ["example-competition"]})
    assert submit(client, key="key-untagged", team="Dee").status_code == 200


def test_any_recognised_key_may_submit_when_no_tag_is_required(competition, state):
    competition.ndif.required_tags = []
    client = TestClient(space_app.create_app(competition, state, whoami=fake_whoami))
    assert submit(client, key="key-untagged", team="Dee").status_code == 200


def test_team_is_the_account_and_is_named_once(client, state):
    assert "choose a team name" in submit(client, team="").json()["detail"]
    first = submit(client, team="Ada's Team").json()
    assert first["team"] == "Ada's Team"
    assert "is taken" in submit(client, key="key-bob", team="ada's team").json()["detail"]

    score(state, first["id"], 0.8)
    # Later submissions need no name, and a different one is ignored.
    assert submit(client, team="").json()["team"] == "Ada's Team"
    assert state.team_of("ada@example.org") == "Ada's Team"


# -------------------------------------------------------------- acceptance --


def test_a_submission_is_stored_and_started(client, state):
    uploaded = package()
    created = submit(client, hf_token="hf_abc", data=uploaded).json()
    stored = state.submission(created["id"])

    assert stored.status == RUNNING and stored.fields == {"method": "white-box"} and stored.email == "ada@example.org"
    assert state.s3.get_object(Bucket="bucket", Key=f"submissions/{created['id']}.zip")["Body"].read() == uploaded
    credentials = json.loads(state.s3.get_object(Bucket="bucket", Key=f"credentials/{created['id']}.json")["Body"].read())
    assert credentials == {"NDIF_API_KEY": "key-ada", "NDIF_HOST": "https://api.ndif.us", "HF_TOKEN": "hf_abc"}

    executions = state.stepfunctions.list_executions(stateMachineArn=state.state_machine)["executions"]
    assert [e["name"] for e in executions] == [created["id"]]
    given = json.loads(state.stepfunctions.describe_execution(executionArn=executions[0]["executionArn"])["input"])
    assert "key-ada" not in json.dumps(given)  # the key travels by link, not in the execution's history
    assert len(json.dumps(given)) < 8192  # what ECS accepts as a container override, with room to spare

    def links(task: str) -> dict:
        return json.loads(state.s3.get_object(Bucket="bucket", Key=f"runs/{created['id']}/{task}-links.json")["Body"].read())

    assert set(links("runner")) == {"submission", "credentials", "output", "report", "progress"}
    assert set(links("evaluator")) == {"output", "results"}
    assert given["runner"].split("?")[0].endswith(f"/runs/{created['id']}/runner-links.json")


def test_a_competition_declares_no_categories_by_default(state):
    competition = comp.load()
    assert competition.submission.fields == []
    competition.ndif.required_tags = ["example-competition"]  # the tag the made-up accounts carry
    client = TestClient(space_app.create_app(competition, state, whoami=fake_whoami))

    created = client.post("/api/submissions", headers={"X-NDIF-API-Key": "key-ada"}, files={"package": ("s.zip", package())}, data={"team": "Ada"})
    assert created.status_code == 200 and state.submission(created.json()["id"]).fields == {}
    assert client.get("/api/config").json()["fields"] == []


def test_a_category_that_is_no_longer_declared_is_not_shown(competition, client, state):
    created = submit(client, method="black-box").json()
    score(state, created["id"], 0.8)
    headers = {"X-NDIF-API-Key": "key-ada"}
    assert client.get(f"/api/submissions/{created['id']}", headers=headers).json()["fields"] == {"method": "black-box"}

    competition.submission.fields = []  # the competition drops its categories
    assert client.get(f"/api/submissions/{created['id']}", headers=headers).json()["fields"] == {}
    assert client.get("/api/leaderboard").json()["rows"][0]["fields"] == {}
    assert state.submission(created["id"]).fields == {"method": "black-box"}  # the record is untouched


@pytest.mark.parametrize(
    "kwargs, code, words",
    [
        ({"method": "grey-box"}, 400, "`method` must be one of"),
        ({"data": b"not a zip"}, 400, "not a zip file"),
    ],
)
def test_refused_submissions_store_nothing(client, state, kwargs, code, words):
    response = submit(client, **kwargs)
    assert response.status_code == code and words in response.json()["detail"]
    assert state.submissions() == []


def test_oversized_package_is_refused(competition, state):
    competition.submission.max_size_mb = 0
    client = TestClient(space_app.create_app(competition, state, whoami=fake_whoami))
    assert submit(client).status_code == 413


def test_one_submission_in_flight_per_team(client, state):
    first = submit(client).json()
    assert "already have a submission queued or running" in submit(client).json()["detail"]
    pipeline_wrote(state, first["id"], EVALUATING, report=report())
    assert submit(client).status_code == 409
    score(state, first["id"], 0.8)
    assert submit(client).status_code == 200


# ------------------------------------------------------------------- queue --


def test_submissions_wait_for_a_free_slot_in_order(client, state):
    ada = submit(client, key="key-ada", team="Ada")
    bob = submit(client, key="key-bob", team="Bob")
    cy = submit(client, key="key-cy", team="Cy")
    ids = [r.json()["id"] for r in (ada, bob, cy)]
    assert [state.submission(i).status for i in ids] == [RUNNING, RUNNING, QUEUED]
    assert client.get(f"/api/submissions/{ids[2]}", headers={"X-NDIF-API-Key": "key-cy"}).json()["queued_ahead"] == 0

    assert state.dispatch() == []  # still full
    score(state, ids[0], 0.7)
    assert state.dispatch() == [ids[2]]
    assert state.submission(ids[2]).status == RUNNING
    assert state.dispatch() == []  # and starting is not repeated


# -------------------------------------------------------------- rate limit --


def test_rate_limit_counts_attempts_but_not_pipeline_failures(client, state):
    for _ in range(3):
        created = submit(client).json()
        pipeline_wrote(state, created["id"], FAILED, report=report(ok=False, reason="exited with code 1"))
    refused = submit(client)
    assert refused.status_code == 429 and "3 per 24h" in refused.json()["detail"]

    me = client.get("/api/me", headers={"X-NDIF-API-Key": "key-ada"}).json()
    assert me["rate_limit"]["remaining"] == 0 and me["rate_limit"]["retry_at"]

    # One of them turns out to have been the leaderboard's fault.
    pipeline_wrote(state, created["id"], FAILED, error="States.TaskFailed")
    assert submit(client).status_code == 200


def test_exempt_teams_are_not_limited(competition, state):
    competition.submission.rate_limit.max = 1
    competition.submission.rate_limit.exempt = ["Organizers"]
    client = TestClient(space_app.create_app(competition, state, whoami=fake_whoami))
    for _ in range(3):
        created = submit(client, team="Organizers")
        assert created.status_code == 200
        score(state, created.json()["id"], 0.5)


# ----------------------------------------------------- what participants see --


def test_own_submission_view(client, state):
    created = submit(client).json()
    state.s3.put_object(Bucket="bucket", Key=f"runs/{created['id']}/progress.json", Body=json.dumps({"dataset": "dataset_b", "index": 2, "total": 2, "state": "running"}))
    headers = {"X-NDIF-API-Key": "key-ada"}

    running = client.get(f"/api/submissions/{created['id']}", headers=headers).json()
    assert running["status"] == "running" and running["progress"] == {"dataset": "Dataset B", "index": 2, "total": 2, "state": "running"}

    score(state, created["id"], 0.75)
    scored = client.get(f"/api/submissions/{created['id']}", headers=headers).json()
    assert scored["results"]["metrics"]["accuracy"] == 0.75 and scored["seconds"] == 9.0
    assert [row["name"] for row in scored["results"]["breakdown"]] == ["Dataset A", "Dataset B"]

    assert client.get(f"/api/submissions/{created['id']}", headers={"X-NDIF-API-Key": "key-bob"}).status_code == 404


def test_failures_are_explained_without_leaking(client, state):
    headers = {"X-NDIF-API-Key": "key-ada"}
    created = submit(client).json()

    pipeline_wrote(state, created["id"], FAILED, report=report(ok=False, reason="exited with code 1"))
    error = client.get(f"/api/submissions/{created['id']}", headers=headers).json()["error"]
    assert error == "Dataset A: exited with code 1. Run `python submit.py --local` to see the full error."

    pipeline_wrote(state, created["id"], FAILED, report=report(ok=False, reason="exited with code 1", log_tail="ValueError: nope"))
    assert "ValueError: nope" in client.get(f"/api/submissions/{created['id']}", headers=headers).json()["error"]

    # The evaluator names datasets by name in what it says; the participant is shown codenames.
    unusable = {"ok": False, "metrics": {}, "breakdown": [], "message": "dataset_a: no readouts.jsonl in $OUTPUT_DIR", "details": {}}
    pipeline_wrote(state, created["id"], FAILED, report=report(), results=unusable)
    assert client.get(f"/api/submissions/{created['id']}", headers=headers).json()["error"] == "Dataset A: no readouts.jsonl in $OUTPUT_DIR"

    noted = {**results(0.5), "message": "Passed 3 of 6 questions. dataset_b: 1 question without a readout. dataset_a: 2 lines for ids that are not this dataset's questions."}
    pipeline_wrote(state, created["id"], SCORED, report=report(), results=noted)
    shown = client.get(f"/api/submissions/{created['id']}", headers=headers).json()["results"]["message"]
    assert shown == "Passed 3 of 6 questions. Dataset B: 1 question without a readout. Dataset A: 2 lines for ids that are not this dataset's questions."


def test_leaderboard_ranks_best_per_team_and_keeps_secrets(competition, state):
    competition.leaderboard.baselines = ["Baseline"]
    competition.submission.rate_limit.max = 0
    client = TestClient(space_app.create_app(competition, state, whoami=fake_whoami))

    def scored(key, team, accuracy, f1=0.5, method="white-box"):
        created = submit(client, key=key, team=team, method=method).json()
        score(state, created["id"], accuracy, f1)
        return created["id"]

    ada_old = scored("key-ada", "Ada", 0.70)
    ada_new = scored("key-ada", "Ada", 0.90, f1=0.2)
    bob = scored("key-bob", "Bob", 0.90, f1=0.8, method="black-box")
    base = scored("key-cy", "Baseline", 0.95)
    unfinished = submit(client, key="key-cy", team="Baseline").json()["id"]

    body = client.get("/api/leaderboard").text
    rows = json.loads(body)["rows"]
    assert [(r["id"], r["rank"], r["best"], r["baseline"]) for r in rows] == [
        (base, None, True, True),  # shown in place, not ranked
        (bob, 1, True, False),  # ties on accuracy, wins on f1
        (ada_new, 2, True, False),
        (ada_old, None, False, False),
    ]
    assert unfinished not in body
    assert rows[1]["fields"] == {"method": "black-box"}
    assert [row["name"] for row in rows[0]["breakdown"]] == ["Dataset A", "Dataset B"]
    for private in ("example.org", "organizers only", "dataset_a", "dataset_b", "key-"):
        assert private not in body


def test_a_submission_belongs_to_the_phase_open_when_it_was_made():
    made = lambda at: Submission("id", "ada@example.org", "Ada", at, SCORED)
    starts = ["2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"]
    before, at_the_start, after = made("2026-01-31T23:59:59+00:00"), made("2026-02-01T00:00:00+00:00"), made("2026-03-05T12:00:00+00:00")
    assert board.by_phase([before, at_the_start, after], starts) == [[before], [at_the_start], [after]]
    assert board.by_phase([before, after], []) == [[before, after]]  # no phase was started: everything is phase 1


def test_a_new_phase_has_a_board_of_its_own(client, state, monkeypatch):
    monkeypatch.setattr(space_app, "BOARD_CACHE_SECONDS", -1)  # every request works the boards out again
    days = iter(datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 9))
    monkeypatch.setattr("space.state.utcnow", lambda: next(days))  # submissions and phases a day apart

    def scored(accuracy):
        created = submit(client).json()
        score(state, created["id"], accuracy)
        return created["id"]

    def shown(query=""):
        body = client.get("/api/leaderboard" + query).json()
        return body["phase"], body["phases"], [(row["id"], row["rank"]) for row in body["rows"]]

    first = scored(0.9)
    assert shown() == (1, 1, [(first, 1)])

    state.start_phase()
    assert shown() == (2, 2, [])  # the new board starts empty
    second = scored(0.4)
    assert shown() == shown("?phase=2") == (2, 2, [(second, 1)])  # below the team's phase 1 score, and its best here
    assert shown("?phase=1") == (1, 2, [(first, 1)])
    assert client.get("/api/leaderboard?phase=3").status_code == 404


def test_a_flagged_submission_keeps_its_place_and_only_its_team_reads_the_note(client, state, monkeypatch):
    monkeypatch.setattr(space_app, "BOARD_CACHE_SECONDS", -1)  # every request works the board out again
    created = submit(client).json()
    score(state, created["id"], 0.9)
    mine = lambda: client.get(f"/api/submissions/{created['id']}", headers={"X-NDIF-API-Key": "key-ada"}).json()
    assert client.get("/api/leaderboard").json()["rows"][0]["flagged"] is False and mine()["flagged"] is False

    state.flag(created["id"], "it reads no activations")
    public = client.get("/api/leaderboard")
    assert [(row["rank"], row["flagged"]) for row in public.json()["rows"]] == [(1, True)]
    assert "it reads no activations" not in public.text
    assert (mine()["flagged"], mine()["flag_note"]) == (True, "it reads no activations")

    state.flag(created["id"])  # a flag needs no note
    assert (mine()["flagged"], mine()["flag_note"]) == (True, None)
    state.unflag(created["id"])
    assert client.get("/api/leaderboard").json()["rows"][0]["flagged"] is False and mine()["flagged"] is False


def test_rank_key_handles_direction_and_missing_metrics(competition):
    competition.leaderboard.rank_by = ["error", "accuracy"]
    competition.leaderboard.lower_is_better = ["error"]
    key = lambda **metrics: board._rank_key(metrics, competition)
    assert key(error=0.1, accuracy=0.5) < key(error=0.2, accuracy=0.9)
    assert key(error=0.1, accuracy=0.9) < key(error=0.1, accuracy=0.5)
    assert key(error=0.9, accuracy=0.1) < key(error=None, accuracy=0.99)


def test_me_before_any_submission(client):
    me = client.get("/api/me", headers={"X-NDIF-API-Key": "key-ada"}).json()
    assert me == {"team": None, "rate_limit": {"max": 3, "window_hours": 24.0, "used": 0, "remaining": 3, "retry_at": None}, "submissions": []}


def test_config_is_what_the_page_needs(client):
    config = client.get("/api/config").json()
    assert config["columns"] == ["accuracy", "f1"] and config["fields"][0]["choices"] == ["white-box", "black-box"]
