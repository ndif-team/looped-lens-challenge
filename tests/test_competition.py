import json

import pytest
from conftest import FAMILIES
from pydantic import ValidationError

import competition as comp


def test_the_repository_holds_the_participants_config():
    competition = comp.load()
    # A dataset's name is what an entry is told, so it says nothing about the questions in it.
    assert [d.name for d in competition.datasets.dev] == ["practice"]
    assert {d.inputs.repo for d in competition.datasets.dev} == {d.private.repo for d in competition.datasets.dev} == {"NDIF/looped-lens-challenge-test"}
    assert competition.datasets.dev[0].env == {"MODEL_ID": "thinkingmachines/Inkling-2-Mini"}
    # The scored datasets are named only in the organizers' configuration, which is not committed.
    assert competition.datasets.eval == []
    assert competition.leaderboard.columns == ["pass_rate"]
    assert competition.space.url == "https://ndif-looped-lens-challenge.hf.space"


@pytest.mark.skipif(not comp.PROD_FILE.is_file(), reason="the organizers' configuration is not in the repository")
def test_the_organizers_config():
    competition, public = comp.load(comp.PROD_FILE), comp.load()
    assert [d.name for d in competition.datasets.eval] == ["dataset_a", "dataset_b", "dataset_c", "dataset_d"]
    assert [d.label for d in competition.datasets.eval] == ["Dataset A", "Dataset B", "Dataset C", "Dataset D"]  # what the leaderboard shows
    assert {d.inputs.repo for d in competition.datasets.eval} == {d.private.repo for d in competition.datasets.eval} == {"NDIF/looped-lens-challenge-eval"}
    assert competition.runner.reveal_errors is False  # the scored questions are private
    assert competition.evaluator.secrets == ["OPENAI_API_KEY"]
    assert competition.ndif.required_tags == ["looped-lens-challenge"]  # only registered accounts may submit
    assert (competition.submission.rate_limit.max, competition.submission.rate_limit.window_hours) == (1, 6)  # one every six hours
    assert competition.submission.max_concurrent == 6
    # What the participants' file repeats has to say the same thing.
    for part in (lambda c: c.title, lambda c: c.ndif.host, lambda c: c.datasets.dev, lambda c: c.space.url, lambda c: c.leaderboard.columns):
        assert part(competition) == part(public)
    for limit in ("timeout_per_dataset", "timeout_total", "max_output_mb"):
        assert getattr(competition.runner, limit) == getattr(public.runner, limit)
    assert competition.submission.max_size_mb == public.submission.max_size_mb


def test_unknown_key_is_rejected():
    raw = comp.load().model_dump()
    raw["runner"]["timout_total"] = 5
    with pytest.raises(ValidationError, match="timout_total"):
        comp.Competition.model_validate(raw)


def test_dataset_names_are_unique_within_a_list():
    raw = comp.load().model_dump()
    raw["datasets"]["dev"].append(raw["datasets"]["dev"][0])
    with pytest.raises(ValidationError, match="unique"):
        comp.Competition.model_validate(raw)


def test_only_the_evaluators_copy_of_the_data_has_the_answers(tmp_path, made_up_competition):
    runner = json.loads(comp.prepare(made_up_competition, "eval", tmp_path / "runner", private=False).read_text())
    evaluator = json.loads(comp.prepare(made_up_competition, "eval", tmp_path / "evaluator", private=True).read_text())
    assert [d["name"] for d in runner["datasets"]] == [d["name"] for d in evaluator["datasets"]] == FAMILIES
    assert "private" not in runner["datasets"][0] and evaluator["datasets"][0]["private"] == "adding/private"
    assert not list((tmp_path / "runner").rglob("private"))
    given = (tmp_path / "runner/linking/inputs/questions.jsonl").read_text()
    assert "model_input" in given and "intermediate" not in given and "options" not in given
    assert "intermediate" in (tmp_path / "evaluator/linking/private/questions.jsonl").read_text()
    assert runner["runner"]["reveal_errors"] is False
