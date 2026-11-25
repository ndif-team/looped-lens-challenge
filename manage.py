#!/usr/bin/env python3
"""Operate a competition: check it, build it, deploy it, and look after it.

    python manage.py verify                compare this environment with the NDIF server
    python manage.py build                 assemble the image build contexts in build/
    python manage.py deploy                build, deploy the AWS stack, publish the Space
    python manage.py submissions           list every submission, newest first
    python manage.py logs ID               print a submission's runner and evaluator logs
    python manage.py cancel ID             stop a submission and free its slot
    python manage.py excuse ID WHY         stop a failed submission counting against its team's limit
    python manage.py flag ID [NOTE]        mark a submission as possibly invalid, under review
    python manage.py unflag ID             take that mark off
    python manage.py remove ID             delete a submission: its record and its files
    python manage.py rename-team NAME NEW  rename a team, on its past submissions too
    python manage.py remove-team NAME      delete a team and every submission it made
    python manage.py new-phase             start the next phase, with a board of its own
    python manage.py destroy               delete the AWS stack

Everything but `verify` and `build` needs AWS credentials for the account the
competition is deployed in. Install the tools with `uv sync --group ops`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import competition as comp

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "Dockerfile", "README.md")
# What the evaluator image needs of final-evals/: the judge's code and its prompts. The folder
# can also hold question files, saved runs and a .env with an API key, which stay out of images.
JUDGE_PARTS = ("lens_eval", "judge_prompts")


def stack_name(competition: comp.Competition) -> str:
    return f"{competition.name}-leaderboard"


# ------------------------------------------------------------------ verify --


def verify(competition: comp.Competition, args) -> int:
    """Compare the packages installed here with the ones the NDIF server runs.
    nnsight marks the mismatches that make a server reject requests."""
    import nnsight
    from nnsight import CONFIG

    CONFIG.API.HOST = competition.ndif.host
    report = str(nnsight.compare())

    def differs(line: str) -> bool:
        # A row reads: package, local version, server version, status. A build
        # tag after "+" (torch's "+cpu" here against "+cu126" there) is the
        # same release built for different hardware, not a mismatch.
        _, local, server, *_ = line.split()
        return local.split("+")[0] != server.split("+")[0]

    critical = [line for line in report.splitlines() if "CRITICAL" in line and differs(line)]
    print(report if args.all else "\n".join(critical) or f"this environment matches {competition.ndif.host} where it matters")
    if critical:
        print("\nPin these in pyproject.toml to the server's versions, run `uv lock`, and rebuild.")
        return 1
    print("\nNext: `python submit.py --local` runs the starter method against the server.")
    return 0


# ------------------------------------------------------------------- build --


def build_info() -> dict:
    """What an image is made of, so a submission's report can say what it ran on."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return {
        "commit": (commit or "uncommitted") + ("+changes" if commit and dirty else ""),
        "lock": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()[:12],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build(competition: comp.Competition, args=None) -> int:
    """Assemble build/runner, build/evaluator and build/space: each is the
    complete Docker build context of one image, so what an image contains is
    exactly what its directory contains. The runner's has the eval datasets'
    inputs; only the evaluator's has their private directories."""
    token = os.environ.get("HF_TOKEN")
    info = build_info()
    for image, private in (("runner", False), ("evaluator", True)):
        context = BUILD / image
        shutil.rmtree(context, ignore_errors=True)
        context.mkdir(parents=True)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy(ROOT / name, context / name)
        shutil.copytree(ROOT / image, context / image, ignore=IGNORE)
        if image == "evaluator":
            for part in JUDGE_PARTS:
                shutil.copytree(ROOT / "final-evals" / part, context / "final-evals" / part, ignore=IGNORE)
        shutil.copy(ROOT / "infra" / f"{image}.Dockerfile", context / "Dockerfile")
        comp.prepare(competition, "eval", context / "data", private=private, token=token, build=info)
        with_private = private and any(d.private is not None for d in competition.datasets.eval)
        print(f"  {context.relative_to(ROOT)}: {len(competition.datasets.eval)} datasets{' with private data' if with_private else ''}")
    shutil.copy(ROOT / "infra" / "proxy" / "trust.py", BUILD / "runner" / "trust_proxy.py")
    print(f"  {build_space(competition).relative_to(ROOT)}: the Space")

    leaked = [p for p in (BUILD / "runner").rglob("*") if "private" in p.relative_to(BUILD / "runner").parts]
    assert not leaked, f"private data in the runner's build context: {leaked}"
    print(f"  built from {info['commit']} with lockfile {info['lock']}")
    return 0


def build_space(competition: comp.Competition) -> Path:
    """Assemble build/space: the files the Space is made of, which is also a
    Docker build context (`docker build build/space` builds what Hugging Face
    will)."""
    context = BUILD / "space"
    shutil.rmtree(context, ignore_errors=True)
    shutil.copytree(ROOT / "space", context / "space", ignore=IGNORE)
    # No configuration file goes into the Space's repository, which anyone can read
    # once the Space is public. The Space is handed the configuration as a secret.
    for name in ("competition.py", "pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, context / name)
    shutil.copy(ROOT / "space" / "Dockerfile", context / "Dockerfile")
    readme = (ROOT / "space" / "README.md").read_text().replace("title: Leaderboard", f"title: {competition.title}")
    (context / "README.md").write_text(readme)
    return context


# ------------------------------------------------------------------ deploy --


# The CDK command line has to be at least as new as the aws-cdk-lib in the
# lockfile, so the two are pinned as a pair. Raise them together.
CDK_CLI = "aws-cdk@2.1148.1"


def cdk(command: str, *extra: str) -> None:
    """Run the CDK command line on the stack, fetched with npx at the pinned
    version rather than taken from whatever the machine has installed."""
    app = f"{sys.executable} -m infra.app"
    subprocess.run(["npx", "--yes", CDK_CLI, command, "--app", app, *extra], cwd=ROOT, check=True)


def outputs(competition: comp.Competition) -> dict[str, str]:
    """The deployed stack's outputs: the names of the things it created."""
    import boto3

    stacks = boto3.client("cloudformation", region_name=competition.aws.region).describe_stacks(StackName=stack_name(competition))
    return {o["OutputKey"]: o["OutputValue"] for o in stacks["Stacks"][0]["Outputs"]}


def new_space_key(competition: comp.Competition, out: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Mint an access key for the Space's own AWS user (the stack created the
    user, with no key). Returns the new key and the ids of the keys it replaces.

    The key is made here and handed straight to the Space, so it is never
    stored anywhere else, and every publish rotates it."""
    import boto3

    iam, user = boto3.client("iam"), out["SpaceUserName"]
    existing = sorted(iam.list_access_keys(UserName=user)["AccessKeyMetadata"], key=lambda k: k["CreateDate"])
    if len(existing) == 2:  # IAM allows two; make room by dropping the older
        iam.delete_access_key(UserName=user, AccessKeyId=existing.pop(0)["AccessKeyId"])
    key = iam.create_access_key(UserName=user)["AccessKey"]
    fresh = {"AWS_ACCESS_KEY_ID": key["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": key["SecretAccessKey"]}
    return fresh, [k["AccessKeyId"] for k in existing]


def publish_space(competition: comp.Competition, out: dict[str, str]) -> None:
    """Upload the Space's files and point it at the deployed stack."""
    from huggingface_hub import HfApi

    api, repo, space = HfApi(), competition.space.repo, competition.space
    api.create_repo(repo, repo_type="space", space_sdk="docker", space_hardware=space.hardware, private=space.private, exist_ok=True)
    # For a Space that already existed, bring its settings in line with the config.
    if api.get_space_runtime(repo).requested_hardware != space.hardware:
        api.request_space_hardware(repo, space.hardware)

    staging = build_space(competition)
    secrets, replaced = new_space_key(competition, out)
    secrets["LEADERBOARD_COMPETITION"] = comp.PROD_FILE.read_text()
    if mirror_token := os.environ.get("LEADERBOARD_MIRROR_TOKEN"):
        secrets["HF_TOKEN"] = mirror_token
    elif competition.space.mirror:
        print("  LEADERBOARD_MIRROR_TOKEN is not set: the Space will run without mirroring to", competition.space.mirror)
    # A public Space shows its variables to everyone, so the stack's resource names, one of
    # which holds the AWS account's number, are secrets too.
    secrets |= {"LEADERBOARD_TABLE": out["Table"], "LEADERBOARD_BUCKET": out["Bucket"], "LEADERBOARD_STATE_MACHINE": out["StateMachine"]}
    for name in api.get_space_variables(repo):  # set as variables by an earlier deploy
        if name in secrets:
            api.delete_space_variable(repo, name)
    for name, value in secrets.items():
        api.add_space_secret(repo, name, value)

    api.upload_folder(folder_path=str(staging), repo_id=repo, repo_type="space", commit_message="Deploy", delete_patterns=["*"])
    # Who can reach the Space changes last, once nothing it shows could be a variable.
    api.update_repo_settings(repo, repo_type="space", private=space.private)

    # The Space restarts with the new key; the ones it replaces can go.
    import boto3

    for key_id in replaced:
        boto3.client("iam").delete_access_key(UserName=out["SpaceUserName"], AccessKeyId=key_id)
    print(f"  Space: https://huggingface.co/spaces/{repo}")


def deploy(competition: comp.Competition, args) -> int:
    """Deploy the AWS stack, then the Space. `--stack` or `--space` does one of
    the two: a change to the page or to the limits only needs `--space`."""
    everything = not (args.stack or args.space)
    if args.stack or everything:
        build(competition)
        cdk("deploy", "--require-approval", "never")
    if args.space or everything:
        publish_space(competition, outputs(competition))
    return 0


def destroy(competition: comp.Competition, args) -> int:
    """Delete the AWS stack. The Space is left alone: pause or delete it on Hugging Face."""
    import boto3

    name = stack_name(competition)
    # Asked here, before anything is changed, rather than by the CDK after the keys are gone.
    if input(f"Delete the stack {name}? Its bucket, table and logs are kept. Type its name to go on: ").strip() != name:
        print("  Nothing was deleted.")
        return 1
    # The Space's access keys were made outside the stack (new_space_key), and AWS will not
    # delete a user that still has one.
    iam, user = boto3.client("iam"), outputs(competition)["SpaceUserName"]
    for key in iam.list_access_keys(UserName=user)["AccessKeyMetadata"]:
        iam.delete_access_key(UserName=user, AccessKeyId=key["AccessKeyId"])
    cdk("destroy", "--force")
    print("  The bucket, the table and the logs are kept. Delete them by hand once you no longer need the submissions.")
    return 0


# ----------------------------------------------------------------- operate --


def deployed_state(competition: comp.Competition):
    """The deployed competition's state, and the stack outputs it was found through."""
    from space.state import State

    out = outputs(competition)
    store = State(out["Table"], out["Bucket"], out["StateMachine"], competition.aws.region, max_concurrent=competition.submission.max_concurrent, url_seconds=3600)
    return store, out


def one_line(s, competition: comp.Competition) -> str:
    """A submission as `submissions` lists it."""
    from space import board

    primary = competition.leaderboard.rank_by[0]
    if s.results and s.results["ok"] and s.results["metrics"].get(primary) is not None:
        note = f"{primary}={s.results['metrics'][primary]:.4f}"
    else:
        note = (board.failure(s, competition) or "").split("\n")[0] + (" (excused)" if s.excused else "")
    flag = "" if s.flagged is None else f"  (flagged: {s.flagged})" if s.flagged else "  (flagged)"
    return f"{s.submitted_at[:19]}  {s.id}  {s.status:<10}  {s.team}  <{s.email}>  {note}{flag}"


def submissions(competition: comp.Competition, args) -> int:
    store, _ = deployed_state(competition)
    for s in reversed(store.submissions()):
        print(one_line(s, competition))
    return 0


def execution_arn(state_machine: str, submission_id: str) -> str:
    return state_machine.replace(":stateMachine:", ":execution:") + ":" + submission_id


def logs(competition: comp.Competition, args) -> int:
    """Print what the runner and evaluator tasks of one submission wrote. This
    is where a method's full output is, private inputs and all."""
    import boto3

    _, out = deployed_state(competition)
    region = competition.aws.region
    history = boto3.client("stepfunctions", region_name=region).get_paginator("get_execution_history")
    cloudwatch = boto3.client("logs", region_name=region)
    for page in history.paginate(executionArn=execution_arn(out["StateMachine"], args.id)):
        for event in page["events"]:
            if event["type"] != "TaskSubmitted":
                continue
            started = json.loads(event["taskSubmittedEventDetails"]["output"])
            if "Tasks" not in started:
                continue
            task = started["Tasks"][0]
            container = task["Containers"][0]["Name"]  # "runner" or "evaluator"
            stream = f"{container}/{container}/{task['TaskArn'].split('/')[-1]}"
            print(f"\n══ {container}  ({stream})")
            pages = cloudwatch.get_paginator("filter_log_events").paginate(logGroupName=out["LogGroup"], logStreamNames=[stream])
            for log_page in pages:
                for line in log_page["events"]:
                    print(line["message"])
    return 0


def cancel(competition: comp.Competition, args) -> int:
    """Stop a submission's pipeline and mark it failed. Marked as the
    leaderboard's doing, it does not count against the team's limit."""
    import boto3
    from botocore.exceptions import ClientError

    from space.state import FAILED, SUBMISSION

    store, out = deployed_state(competition)
    try:
        boto3.client("stepfunctions", region_name=competition.aws.region).stop_execution(
            executionArn=execution_arn(out["StateMachine"], args.id), cause="cancelled by the organizers"
        )
    except ClientError as error:
        print(f"  no running pipeline to stop ({error.response['Error']['Code']})")
    store.table.update_item(
        Key={"pk": SUBMISSION, "sk": args.id},
        UpdateExpression="SET #status = :failed, #error = :error",
        ConditionExpression="attribute_exists(pk)",
        ExpressionAttributeNames={"#status": "status", "#error": "error"},
        ExpressionAttributeValues={":failed": FAILED, ":error": "cancelled by the organizers"},
    )
    print(f"  {args.id} cancelled")
    return 0


def found(store, submission_id: str):
    """The submission with this id. Stops the command when there is none."""
    submission = store.submission(submission_id)
    if submission is None:
        sys.exit(f"there is no submission {submission_id}")
    return submission


def excuse(competition: comp.Competition, args) -> int:
    """Record that a failed submission was not its team's doing: the leaderboard
    broke under it, or NDIF did. It stops counting against the team's limit, and
    the team is shown the reason beside the failure."""
    store, _ = deployed_state(competition)
    submission = found(store, args.id)
    if not store.excuse(args.id, args.reason):
        sys.exit(f"{args.id} is {submission.status}: only a failed submission can be excused")
    print(f"  {args.id} excused: it no longer counts against the limit of {submission.team}")
    return 0


def flag(competition: comp.Competition, args) -> int:
    """Mark a submission as possibly invalid while the organizers review it. It
    keeps its place on the board, where everyone sees a warning beside it. Its
    team is shown the note as well, when one is given."""
    store, _ = deployed_state(competition)
    submission = found(store, args.id)
    store.flag(args.id, args.note)
    print(f"  {args.id} of {submission.team} is flagged for review")
    return 0


def unflag(competition: comp.Competition, args) -> int:
    """Take the review mark off a submission."""
    store, _ = deployed_state(competition)
    submission = found(store, args.id)
    if submission.flagged is None:
        sys.exit(f"{args.id} is not flagged")
    store.unflag(args.id)
    print(f"  {args.id} of {submission.team} is no longer flagged")
    return 0


def remove(competition: comp.Competition, args) -> int:
    """Delete one submission for good: its package, everything its run wrote and
    its record. It leaves the board and no longer counts against its team's limit.
    The team keeps its name."""
    from space.state import IN_FLIGHT

    store, _ = deployed_state(competition)
    submission = found(store, args.id)
    if submission.status in IN_FLIGHT:
        sys.exit(f"{args.id} is {submission.status}. Stop it first: python manage.py cancel {args.id}")
    print(one_line(submission, competition))
    if input(f"Delete this submission and its {len(store.files(args.id))} files? This cannot be undone. Type its id to go on: ").strip() != args.id:
        print("  Nothing was deleted.")
        return 1
    store.remove(args.id)
    print(f"  {args.id} removed")
    return 0


def team_found(store, name: str) -> tuple[str, str]:
    """The account that holds a team name, and the name as it is stored. Stops
    the command when no team has the name."""
    email = store.team_account(name)
    if email is None:
        sys.exit(f"there is no team named {name!r}")
    return email, store.team_of(email)


def rename_team(competition: comp.Competition, args) -> int:
    """Give a team another name, on the board and on its past submissions."""
    from space.state import NameTaken, team_name

    store, _ = deployed_state(competition)
    email, old = team_found(store, args.name)
    new = team_name(args.new)
    if new is None:
        sys.exit("a team name is 2 to 40 printable characters")
    try:
        store.rename_team(email, new)
    except NameTaken:
        sys.exit(f"the team name {new!r} is taken")
    print(f"  {old!r} is now {new!r}")
    lists = {"leaderboard.baselines": competition.leaderboard.baselines, "submission.rate_limit.exempt": competition.submission.rate_limit.exempt}
    if listed := [where for where, names in lists.items() if old in names]:
        print(f"  {old!r} is listed in {' and '.join(listed)}: change it in {comp.PROD_FILE.name}, then run `python manage.py deploy --space`.")
    return 0


def remove_team(competition: comp.Competition, args) -> int:
    """Delete a team for good: every submission it made, with their files, and
    its registration. Its name becomes free. The NDIF account behind it can
    still submit and would register a new team: to keep an account out, take
    the competition's tag off it at NDIF."""
    from space.state import IN_FLIGHT

    store, _ = deployed_state(competition)
    email, name = team_found(store, args.name)
    theirs = [s for s in store.submissions() if s.email == email]
    if busy := [s for s in theirs if s.status in IN_FLIGHT]:
        sys.exit(f"{busy[0].id} is {busy[0].status}. Stop it first: python manage.py cancel {busy[0].id}")
    for s in theirs:
        print(one_line(s, competition))
    asked = f"Delete the team {name!r} and its {len(theirs)} submissions with their files? This cannot be undone. Type the team's name to go on: "
    if input(asked).strip() != name:
        print("  Nothing was deleted.")
        return 1
    for s in theirs:
        store.remove(s.id)
    store.remove_team(email)
    print(f"  {name!r} removed, with {len(theirs)} submissions")
    return 0


def new_phase(competition: comp.Competition, args) -> int:
    """Start the next phase. Submissions made from now on are ranked on a board
    of their own, which starts empty; the boards of earlier phases stay as they
    are, and the page keeps a button for each. Nothing else changes: the
    datasets, the limits and the teams are whatever is deployed."""
    store, _ = deployed_state(competition)
    following = len(store.phase_starts()) + 2
    if input(f"The competition is in phase {following - 1}. Start phase {following} now, with an empty board? Type {following} to go on: ").strip() != str(following):
        print("  Nothing was changed.")
        return 1
    store.start_phase()
    print(f"  Phase {following} has begun")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify").add_argument("--all", action="store_true", help="print the whole comparison, not just the mismatches that matter")
    for name in ("build", "destroy", "submissions", "new-phase"):
        commands.add_parser(name)
    deploying = commands.add_parser("deploy")
    deploying.add_argument("--stack", action="store_true", help="only build the images and deploy the AWS stack")
    deploying.add_argument("--space", action="store_true", help="only publish the Space")
    for name in ("logs", "cancel", "remove", "unflag"):
        commands.add_parser(name).add_argument("id", help="a submission id")
    excusing = commands.add_parser("excuse")
    excusing.add_argument("id", help="a failed submission's id")
    excusing.add_argument("reason", help="why it was not the team's doing; the team is shown this")
    flagging = commands.add_parser("flag")
    flagging.add_argument("id", help="a submission id")
    flagging.add_argument("note", nargs="?", default="", help="what its team is told about why; optional")
    renaming = commands.add_parser("rename-team")
    renaming.add_argument("name", help="the team's name now")
    renaming.add_argument("new", help="the name to give it")
    commands.add_parser("remove-team").add_argument("name", help="the team's name")
    args = parser.parse_args()
    if not comp.PROD_FILE.is_file():
        sys.exit(f"{comp.PROD_FILE.name} is not here. It is the organizers' configuration, which names the scored datasets and is not in the repository.")
    return globals()[args.command.replace("-", "_")](comp.load(comp.PROD_FILE), args)


if __name__ == "__main__":
    sys.exit(main())
