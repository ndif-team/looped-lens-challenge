<p align="center">
  <img src="assets/flower.svg" alt="The flower mark of the Looped Lens Challenge" width="120">
</p>

# The Looped Lens Challenge

Build the lens that best reads [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
A lens looks at the model's internal state at one token of a prompt and says, in
tokens, what the model is computing there. This repository holds the starter
lens, the tools to run and submit one, and the leaderboard's own code. The
[challenge page](https://ndif.us/looped-lens-challenge/) has the background and
the dates, and the [leaderboard](https://ndif-looped-lens-challenge.hf.space/)
shows the standings.

Your lens runs on [NDIF](https://ndif.us), which hosts the model, so you do not
need a GPU.

## Setup

1. Get an NDIF API key at [login.ndif.us](https://login.ndif.us).
2. Register for the challenge with the [registration and waitlist form](https://forms.gle/enBNFNUBDj5SWvfC9).
   Use the email your key was created with. Compute is limited, so later
   registrations go on a waitlist. Until your account is admitted you can run
   your lens locally but not submit it.
3. Install this repository with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ndif-team/looped-lens-challenge && cd looped-lens-challenge
uv sync                      # the competition environment, in .venv
source .venv/bin/activate
export NDIF_API_KEY=...      # or save it in nnsight
```

If you work with a coding agent, install NDIF's [agent skills](https://github.com/ndif-team/skills)
for nnsight. [`AGENTS.md`](AGENTS.md) tells the agent how to work in this
repository.

## Taking part

```bash
python submit.py --local                  # run submission/ on the practice questions and score it
python submit.py --team "My Team" --wait  # submit; the team name is needed the first time only
python submit.py --status                 # your submissions
```

`submission/` is what you edit and what gets uploaded. It ships with a logit
lens, so these commands work before you change anything.

A team is one NDIF account. The [leaderboard](https://ndif-looped-lens-challenge.hf.space/)
shows each team's best submission.

On the leaderboard's page, under "Your submissions", the × beside a finished
submission takes it off the leaderboard and out of your list for good. It
still counts against your submission limit.

## The questions

A question is a prompt on which the model has to work something out that it
never writes down. That something is the question's *intermediate*, and a lens
passes the question by surfacing it. One of the practice questions:

| Prompt | Intermediate |
|---|---|
| What sport does the athlete Sidney Crosby play? Answer in one word. | `hockey`, the answer, read before the model has started to write it. |

There are two sets:

- **Practice:** 10 questions with their answers, in the dataset
  [`NDIF/looped-lens-challenge-test`](https://huggingface.co/datasets/NDIF/looped-lens-challenge-test).
  `python submit.py --local` runs on these, as one dataset named `practice`.
- **Scored:** 290 questions, which you cannot see, in three datasets. The
  leaderboard runs your lens on each of them and shows them as Dataset A,
  Dataset B and Dataset C. They include kinds of question that the practice
  set does not have.

These are the first phase's questions: simple tasks whose intermediate is a
single token. Later phases add harder ones, and the final ranking mixes the
early and the later questions.

Each question your lens is given has these fields:

| Field | What it is |
|---|---|
| `id` | Which question this is. |
| `model_input` | The exact string to run the model on. Tokenize it with no special tokens added. |
| `read_position` | The index of the token to read at. It is always the last token. |
| `read_token` | That token's text, as a check that you tokenized the same way. |
| `prompt` | The question in plain words. `model_input` is this, as it is or wrapped in the model's chat format. |

## Writing a lens

The leaderboard starts `submission/main.py` once per dataset, with the dataset's
name in `$DATASET_NAME` (`dataset_a`, `dataset_b` or `dataset_c`; `practice` in
a local run). It reads the dataset's questions from
`$DATASET_PATH/questions.jsonl`, runs the model on NDIF, and writes
`$OUTPUT_DIR/readouts.jsonl`.

| File | What it is |
|---|---|
| [`submission/main.py`](submission/main.py) | The entry point: loads the questions, runs the lens over them in NDIF requests, writes the readouts. |
| [`submission/lenses.py`](submission/lenses.py) | The lens, as functions that execute on NDIF beside the model. Yours goes here. |
| [`submission/questions.py`](submission/questions.py) | Plumbing: turns each question into token ids and a read position, and writes the readouts file. |

**The model is looped.** It has 24 layers, and layers 6 to 17 run three times in
every forward pass. A token therefore passes through 48 *layer visits*, numbered
0 to 47 in the order they run. The challenge page calls them effective layers. `lenses.visit_address` says which layer and which
of its runs a visit is, and `main.py` reads one with nnsight's `tracer.iter`.

**What you choose:**

- which visits to read, up to 10 per question. They need not be the same for
  every question;
- where to read. The question's `read_position` is the recommended place, and
  the starter reads there, but nothing requires it;
- what the lens computes from the state it reads.

**What you hand back**, one line of `readouts.jsonl` per question:

```json
{"id": "athlete-05", "readouts": [{"layer": 36, "tokens": [" hockey", " ice", " sport"]}, {"layer": 47, "tokens": [" Hockey", " hockey"]}]}
```

- at most 10 readouts per question, each labelled with the layer it was read
  at, and no layer twice;
- 10 tokens per readout, as text, best first. Any after the tenth are not read.
  A token is at most 64 characters;
- a question with no line, with no readouts or more than 10, with a layer twice
  or with a token that is too long counts as failed;
- a file with two lines for one question, or with a line that is not a JSON
  object, is not scored at all.

**Trained weights** reach NDIF through the Hugging Face Hub. Publish them from
your own machine as a safetensors file and fetch them at the top of the NDIF
session, before the first trace, with `lenses.fetch`. They cannot travel in the
request: NDIF accepts requests up to about 1 MB. If the repository is private,
pass `os.environ.get("HF_TOKEN")` to `lenses.fetch`. The token then travels to
NDIF in each request, so use a read-only one.

**Your credentials.** `submit.py` sends your NDIF key with a submission, and
your Hugging Face token if it finds one: from `--hf-token`, from `$HF_TOKEN`, or
from a saved Hugging Face login. The leaderboard hands both to your entry as
`NDIF_API_KEY` and `HF_TOKEN` and deletes its copy when the run ends.

**Limits of the environment:**

- your code on NDIF cannot read the model's weights. It can call the model's own
  modules, which is how the starter unembeds (`model.lm_head(x)`);
- on the leaderboard your code reaches NDIF and, read-only, Hugging Face.
  Nothing else;
- packages cannot be added. A lens runs in the environment `uv sync` installs;
- a submission is at most 200 MB;
- a submission's run is at most 45 minutes, all datasets together. One that
  takes longer fails.

## Scoring

1. A judge model is shown one question's readout and five options: the
   intermediate and four distractors. It is not shown the question, or which
   option is right. For each layer it lists the options that a token on that
   layer names.
2. A token names an option if it is the option itself, in any case, inflection,
   spelling or language, or a word formed from it ("Turkish" names Turkey). A
   clue does not count ("Istanbul" does not name Turkey), and a number option is
   named only by a token that writes the whole number.
3. Each layer votes for the named option that ranks highest on it.
4. The question passes if the intermediate has more votes than every other
   option. A tie fails.

A dataset's score is the share of its questions that pass. The leaderboard ranks
on `pass_rate`, the mean of the three datasets' scores.

The judge's prompt and the scoring code are in [`final-evals/`](final-evals/).
Two short texts in that prompt come with each question and are not in the
repository for the scored ones: a line describing the task, and the rules for
naming an option of the question's kind. The practice questions show theirs.

**Scoring a local run.** With no key for the judge model, `submit.py --local`
scores by exact matching: a token names an option only if it equals it. The
judge also accepts inflections and translations, so a local score and the
leaderboard's can differ in either direction. To use the same judge, run
`uv sync --group evaluator` and set `OPENAI_API_KEY` to an OpenAI API key; the
practice set is 10 calls and costs well under a cent.

## Rules

The [challenge page](https://ndif.us/looped-lens-challenge/#competition) states
the rules in full. In short:

- **An entry has to be a real attempt at a lens for Inkling 2 Mini.** The goal
  is to read the model, not to game the scoring. An entry is invalid if it:
  - does not read the model's activations;
  - calls another model to produce its readouts. This may be relaxed in a
    later phase, for lenses such as NLAs and activation oracles;
  - works its readouts out from the text of the prompt;
  - carries answers to the questions, or tries to get the hidden questions out.

  Winning submissions are audited, and one that does not fit the spirit of the
  competition is disqualified.
- **A submission under review is marked.** A warning sign beside it on the
  leaderboard, "Potentially invalid: flagged for review", means the organizers
  are checking whether it is valid. It keeps its place meanwhile. If they left
  a note, its team reads it under the submission in "Your submissions" on the
  leaderboard's page and in `python submit.py --status`.
- **Submissions are made public.** By submitting you agree that your
  submission, its code and its associated materials are published under the MIT
  license, during and after the competition, and that the organizers may
  reproduce them in the competition report and the public repository. You keep
  authorship credit.
- **The scored questions are hidden.** A submission that fails on the
  leaderboard is told which dataset failed and not shown its log, because a log
  could quote them. Find errors with `--local`, which shows everything.
- **Only admitted accounts can submit.** A submission made with the key of an
  account that has not registered, or that is still on the waitlist, is refused.
- **A team can submit once every 2 hours,** and can have one submission queued
  or running at a time. A submission that fails because the leaderboard itself
  broke does not count. `python submit.py --status` shows when the next one
  opens.
- **The evaluation can change.** The organizers may change the scored
  questions, the judge's prompts and the scoring code while the competition
  runs. Scores and rankings can move when they do.
- **Eligibility.** Participants are 18 or older, each person is on one team,
  and the organizers and anyone who knows how the questions were built may not
  take part. The page's [eligibility terms](https://ndif.us/looped-lens-challenge/#eligibility)
  also say who can receive a prize.

## Running the competition

This repository is built on NDIF's leaderboard framework. `python manage.py
--help` lists what an organizer runs: building, deploying, and looking after a
live competition. What is specific to this one:

| To change | Edit |
|---|---|
| The scored datasets, limits, the model, who may submit | `competition.prod.yaml` |
| The practice dataset, and what a local run is held to | `competition.yaml`, and the same in `competition.prod.yaml` |
| How readouts are checked and scored | `evaluator/evaluate.py`, and the judge in `final-evals/` |
| The judge model | `JUDGE_MODEL` in `final-evals/lens_eval/config.py` |
| Package versions | `pyproject.toml`, then `uv lock` |

- **The leaderboard's configuration is not in the repository.** `competition.yaml`
  is what a participant's tools read. `python manage.py` reads
  `competition.prod.yaml`, the organizers' full configuration, which names the
  scored datasets and is never committed: get it from an organizer and put it
  beside `competition.yaml`. `manage.py deploy` hands it to the Space as the
  secret `LEADERBOARD_COMPETITION`, so it is not in the Space's files either.
- **The questions are on the Hub.** The practice set is the public dataset
  named above. The scored set is the private dataset
  `NDIF/looped-lens-challenge-eval`; `python manage.py build` fetches it, so it
  needs a Hugging Face login that can read it. Only the evaluator image gets
  the answers. Each private question also carries what the judge is told about
  it (`task_type`, `rules`); the dataset's `reference/judge/` folder has the
  files those are filled from and the script that fills them.
- **Judging costs money.** The evaluator reaches the judge model at OpenAI
  with the key in the AWS Secrets Manager secret `OPENAI_API_KEY`, which has
  to exist in the stack's region before a deploy. A submission is 290 calls,
  roughly 10 cents.
- **Versions follow NDIF.** nnsight, torch and transformers are pinned to what
  NDIF runs; `python manage.py verify` checks them.
- **Looking after submissions.** `python manage.py excuse ID "why"` gives a
  team its attempt back when a run failed through no fault of theirs, and
  shows them the reason. `remove` and `remove-team` delete for good: each
  lists what it will delete and asks for the id or the name to be typed.
- **A team can withdraw its own finished submissions** on the page. A
  withdrawn submission leaves the board and the team's list and still counts
  against the team's limit. It is kept: `python manage.py submissions` lists
  it as withdrawn, and deleting the `withdrawn` attribute from its item puts
  it back.
- **Reviewing a submission.** `python manage.py flag ID "note"` marks one as
  possibly invalid while it is looked into, as the rules above describe. The
  note is optional and only its team is shown it. `unflag` clears the mark.
- **Phases.** `python manage.py new-phase` starts the next phase. Submissions
  made from then on are ranked on a new board, which starts empty, the
  reference row included; the page keeps a button for each earlier phase. A
  submission belongs to the phase it was made in. Nothing else changes: to
  give a phase new datasets, deploy them while no submission is queued and
  start the phase straight after.

## Tests

```bash
uv sync --all-groups
python -m pytest
```

They call neither NDIF nor a judge model, and they use made-up questions. They
need the model's tokenizer, which Hugging Face caches after the first download.
