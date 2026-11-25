"""Submissions and teams: one DynamoDB table and one S3 bucket.

The table holds four kinds of item, told apart by their partition key::

    pk             sk               what
    "submission"   <submission id>  one per submission; ids sort by time
    "team"         <email>          the display name an NDIF account chose
    "team-name"    <name, folded>   reserves a display name for one account
    "phase"        <start time>     one per phase after the first: when it began

The bucket holds the files of each submission::

    submissions/<id>.zip          the uploaded package, kept
    credentials/<id>.json         the submitter's keys for the run; the state
                                  machine deletes it when the run ends
    runs/<id>/*-links.json        what each task may read and write, as links
    runs/<id>/outputs.tar.gz      what the run wrote
    runs/<id>/report.json         the runner's report
    runs/<id>/progress.json       the dataset the run is on
    runs/<id>/results.json        the evaluator's results envelope

A submission moves through ``queued → running → evaluating → scored``, or ends
at ``failed``. The Space writes ``queued`` and ``running``; the state machine in
infra/ writes everything after that, storing the runner's report and the
evaluator's results on the item as JSON text.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config
from botocore.exceptions import ClientError

QUEUED, RUNNING, EVALUATING, SCORED, FAILED = "queued", "running", "evaluating", "scored", "failed"
IN_FLIGHT = (QUEUED, RUNNING, EVALUATING)

SUBMISSION, TEAM, TEAM_NAME, PHASE = "submission", "team", "team-name", "phase"

# What each task is allowed to touch: a name, the HTTP method its link permits,
# and the object the link points at. A task is started with one environment
# variable, LINKS_VARIABLE, holding a link to a JSON object of these names and
# their links. (One variable, because ECS caps how much a state machine can pass
# to a container and a handful of signed links does not fit.)
LINKS_VARIABLE = "LEADERBOARD_LINKS_URL"
RUNNER_LINKS = {
    "submission": ("GET", "submissions/{id}.zip"),
    "credentials": ("GET", "credentials/{id}.json"),
    "output": ("PUT", "runs/{id}/outputs.tar.gz"),
    "report": ("PUT", "runs/{id}/report.json"),
    "progress": ("PUT", "runs/{id}/progress.json"),
}
EVALUATOR_LINKS = {
    "output": ("GET", "runs/{id}/outputs.tar.gz"),
    "results": ("PUT", "runs/{id}/results.json"),
}


class NameTaken(Exception):
    """Another NDIF account already uses this team name."""


@dataclass
class Submission:
    id: str
    email: str
    team: str
    submitted_at: str  # ISO 8601, UTC
    status: str
    fields: dict[str, str] = field(default_factory=dict)
    report: dict | None = None  # runner.RunReport, once the run has ended
    results: dict | None = None  # the evaluator's envelope, once it has been scored
    error: str | None = None  # set when the pipeline itself broke
    excused: str | None = None  # an organizer's reason why this failure does not count against the team
    flagged: str | None = None  # set while the organizers review it: their note to its team, "" when they left none

    @classmethod
    def from_item(cls, item: dict) -> "Submission":
        return cls(
            id=item["sk"],
            email=item["email"],
            team=item["team"],
            submitted_at=item["submitted_at"],
            status=item["status"],
            fields=dict(item.get("fields", {})),
            report=json.loads(item["report"]) if item.get("report") else None,
            results=json.loads(item["results"]) if item.get("results") else None,
            error=item.get("error"),
            excused=item.get("excused"),
            flagged=item.get("flagged"),
        )


def team_name(given: str) -> str | None:
    """A team name as it is stored, with runs of spaces made one. None when the
    text cannot be a name: it has to be 2 to 40 printable characters."""
    name = " ".join(given.split())
    return name if 2 <= len(name) <= 40 and name.isprintable() else None


def new_id(now: datetime) -> str:
    """Sorts by submission time and is a valid Step Functions execution name."""
    return f"{now:%Y%m%dT%H%M%S}-{secrets.token_hex(4)}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class State:
    def __init__(self, table: str, bucket: str, state_machine: str, region: str, *, max_concurrent: int, url_seconds: int):
        self.table = boto3.resource("dynamodb", region_name=region).Table(table)
        # Links are signed for the bucket's regional address, which is the one
        # host the runner task may reach without going through the proxy.
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
        self.stepfunctions = boto3.client("stepfunctions", region_name=region)
        self.bucket = bucket
        self.state_machine = state_machine
        self.max_concurrent = max_concurrent
        # How long the links handed to the two tasks stay valid. It has to cover
        # a whole run and its evaluation.
        self.url_seconds = url_seconds

    # ---------------------------------------------------------------- teams --

    def team_of(self, email: str) -> str | None:
        item = self.table.get_item(Key={"pk": TEAM, "sk": email}).get("Item")
        return item["name"] if item else None

    def register_team(self, email: str, name: str) -> None:
        """Bind a display name to an NDIF account. Raises ``NameTaken`` when a
        different account holds the name."""
        try:
            self.table.put_item(
                Item={"pk": TEAM_NAME, "sk": name.casefold(), "email": email},
                ConditionExpression="attribute_not_exists(pk) OR email = :email",
                ExpressionAttributeValues={":email": email},
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NameTaken(name) from None
            raise
        self.table.put_item(Item={"pk": TEAM, "sk": email, "name": name})

    def team_account(self, name: str) -> str | None:
        """The email of the NDIF account that holds a team name, in any casing."""
        item = self.table.get_item(Key={"pk": TEAM_NAME, "sk": name.casefold()}).get("Item")
        return item["email"] if item else None

    def rename_team(self, email: str, name: str) -> None:
        """Give an account's team another name, on its past submissions too. Raises
        ``NameTaken`` when a different account holds the name."""
        old = self.team_of(email)
        self.register_team(email, name)
        if old.casefold() != name.casefold():
            self.table.delete_item(Key={"pk": TEAM_NAME, "sk": old.casefold()})
        for submission in self.submissions():
            if submission.email == email:
                self.table.update_item(
                    Key={"pk": SUBMISSION, "sk": submission.id},
                    UpdateExpression="SET #team = :name",
                    ExpressionAttributeNames={"#team": "team"},
                    ExpressionAttributeValues={":name": name},
                )

    def remove_team(self, email: str) -> None:
        """Forget an account's team and free its name. Its submissions are not
        touched: remove them first."""
        if name := self.team_of(email):
            self.table.delete_item(Key={"pk": TEAM_NAME, "sk": name.casefold()})
        self.table.delete_item(Key={"pk": TEAM, "sk": email})

    # ---------------------------------------------------------- submissions --

    def submissions(self) -> list[Submission]:
        """Every submission, oldest first."""
        items, query = [], {"KeyConditionExpression": Key("pk").eq(SUBMISSION)}
        while True:
            page = self.table.query(**query)
            items += page["Items"]
            if "LastEvaluatedKey" not in page:
                return [Submission.from_item(item) for item in items]
            query["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def submission(self, submission_id: str) -> Submission | None:
        item = self.table.get_item(Key={"pk": SUBMISSION, "sk": submission_id}).get("Item")
        return Submission.from_item(item) if item else None

    def create(self, email: str, team: str, fields: dict[str, str], package: bytes, credentials: dict) -> Submission:
        """Store a new submission and its files. It starts out queued."""
        now = utcnow()
        submission = Submission(new_id(now), email, team, now.isoformat(timespec="seconds"), QUEUED, fields)
        self.s3.put_object(Bucket=self.bucket, Key=f"submissions/{submission.id}.zip", Body=package)
        self.s3.put_object(Bucket=self.bucket, Key=f"credentials/{submission.id}.json", Body=json.dumps(credentials).encode())
        self.table.put_item(
            Item={
                "pk": SUBMISSION,
                "sk": submission.id,
                "email": email,
                "team": team,
                "submitted_at": submission.submitted_at,
                "status": QUEUED,
                "fields": fields,
            }
        )
        return submission

    def files(self, submission_id: str) -> list[str]:
        """The key of every object stored for a submission: its package, its
        credentials while they exist, and everything its run wrote."""
        listing = self.s3.get_paginator("list_objects_v2")
        prefixes = (f"submissions/{submission_id}.zip", f"credentials/{submission_id}.json", f"runs/{submission_id}/")
        return [obj["Key"] for prefix in prefixes for page in listing.paginate(Bucket=self.bucket, Prefix=prefix) for obj in page.get("Contents", [])]

    def remove(self, submission_id: str) -> None:
        """Delete a submission for good: its files, then its record. The record
        goes last, so a removal that stops halfway can be run again."""
        for key in self.files(submission_id):
            self.s3.delete_object(Bucket=self.bucket, Key=key)
        self.table.delete_item(Key={"pk": SUBMISSION, "sk": submission_id})

    def excuse(self, submission_id: str, reason: str) -> bool:
        """Record that a failed submission was not its team's doing. Returns
        False, changing nothing, when the submission has not failed."""
        try:
            self.table.update_item(
                Key={"pk": SUBMISSION, "sk": submission_id},
                UpdateExpression="SET excused = :reason",
                ConditionExpression="#status = :failed",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":reason": reason, ":failed": FAILED},
            )
        except ClientError as error:
            if error.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
            return False
        return True

    def flag(self, submission_id: str, note: str = "") -> None:
        """Mark a submission as under review, with a note for its team if there
        is one. It keeps its place on the board, where it carries a warning."""
        self.table.update_item(
            Key={"pk": SUBMISSION, "sk": submission_id},
            UpdateExpression="SET flagged = :note",
            ConditionExpression="attribute_exists(pk)",
            ExpressionAttributeValues={":note": note},
        )

    def unflag(self, submission_id: str) -> None:
        """Take the review mark off a submission."""
        self.table.update_item(Key={"pk": SUBMISSION, "sk": submission_id}, UpdateExpression="REMOVE flagged", ConditionExpression="attribute_exists(pk)")

    def progress(self, submission_id: str) -> dict | None:
        """The latest progress event a running submission reported, if any."""
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=f"runs/{submission_id}/progress.json")["Body"].read()
            return json.loads(body)
        except (ClientError, ValueError):
            return None

    # --------------------------------------------------------------- phases --

    def phase_starts(self) -> list[str]:
        """When each phase after the first began, oldest first. A competition is
        in phase 1 until a new one is started, so by default there are none."""
        return [item["sk"] for item in self.table.query(KeyConditionExpression=Key("pk").eq(PHASE))["Items"]]

    def start_phase(self) -> None:
        """Begin a new phase now. Submissions made from this moment belong to it;
        nothing stored about the earlier ones changes."""
        self.table.put_item(Item={"pk": PHASE, "sk": utcnow().isoformat(timespec="seconds")})

    # ------------------------------------------------------------- dispatch --

    def dispatch(self) -> list[str]:
        """Start queued submissions, oldest first, while fewer than
        ``max_concurrent`` are in the pipeline. Returns the ids it started.

        This is the whole queue: a submission waits as a ``queued`` item until
        this is called and finds a free slot. It is called after every new
        submission and on a timer, so a slot freed by a finished run is noticed.
        """
        submissions = self.submissions()
        busy = sum(s.status in (RUNNING, EVALUATING) for s in submissions)
        started = []
        for submission in submissions:
            if busy >= self.max_concurrent:
                break
            if submission.status == QUEUED and self._start(submission.id):
                started.append(submission.id)
                busy += 1
        return started

    def _link(self, method: str, key: str) -> str:
        return self.s3.generate_presigned_url(
            {"GET": "get_object", "PUT": "put_object"}[method],
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.url_seconds,
        )

    def _hand_over(self, task: str, links: dict, submission_id: str) -> str:
        """Write the links one task may use to an object of its own, and return
        a link to that object: the single thing the task is started with."""
        key = f"runs/{submission_id}/{task}-links.json"
        body = {name: self._link(method, target.format(id=submission_id)) for name, (method, target) in links.items()}
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(body).encode())
        return self._link("GET", key)

    def _start(self, submission_id: str) -> bool:
        """Start the pipeline for one queued submission. The two tasks are given
        links to exactly the objects they may touch and nothing else."""
        pipeline_input = {
            "id": submission_id,
            "credentials_key": f"credentials/{submission_id}.json",
            "report_key": f"runs/{submission_id}/report.json",
            "results_key": f"runs/{submission_id}/results.json",
            "runner": self._hand_over("runner", RUNNER_LINKS, submission_id),
            "evaluator": self._hand_over("evaluator", EVALUATOR_LINKS, submission_id),
        }
        try:
            # The execution is named after the submission, so starting it twice
            # is refused rather than run twice.
            self.stepfunctions.start_execution(stateMachineArn=self.state_machine, name=submission_id, input=json.dumps(pipeline_input))
        except ClientError as error:
            if error.response["Error"]["Code"] != "ExecutionAlreadyExists":
                raise
        try:
            self.table.update_item(
                Key={"pk": SUBMISSION, "sk": submission_id},
                UpdateExpression="SET #status = :running",
                ConditionExpression="#status = :queued",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":running": RUNNING, ":queued": QUEUED},
            )
        except ClientError as error:
            if error.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
            return False  # already past queued; nothing to do
        return True
