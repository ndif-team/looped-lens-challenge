# For agents working in this repository

This is the Looped Lens Challenge: a competition to build the lens that best
reads Inkling 2 Mini. You are most likely helping someone enter it.
[`README.md`](README.md) is the participant's guide and has the rules and the
file formats. Read it first. This file adds what an agent needs to do the work
well.

## Load the nnsight skill before writing lens code

A lens here is [nnsight](https://nnsight.net) 0.8 code that runs on NDIF. Most
nnsight code online was written for earlier versions and does not run as
written. NDIF publishes agent skills that teach the current one:
<https://github.com/ndif-team/skills>.

- **If a skill named `nnsight` is available to you, load it** before you write
  or change anything in `submission/`. This challenge uses its guides on remote
  execution, the logit lens, and modules that run several times in one forward
  pass.
- **If it is not available, ask the person to install it,** then carry on:

  | Agent | Commands |
  |---|---|
  | Claude Code | `/plugin marketplace add ndif-team/skills`, then `/plugin install nnsight@ndif-team` |
  | Codex | `codex plugin marketplace add ndif-team/skills`, then `codex plugin add nnsight@ndif-team` |
  | Others | `npx skills add ndif-team/skills` |

- **If they would rather not install it,** read
  `plugins/nnsight/skills/nnsight/SKILL.md` in that repository, and the files
  under `references/` that it points to, before writing code.

The starter uses nnsight directly. The `nnterp` and `nnspec` skills from the
same repository are not needed for it.

## What to change and what to leave alone

| Path | What to do with it |
|---|---|
| `submission/` | The entry. Everything in it is uploaded. This is what you change. |
| `submit.py` | Runs an entry locally and submits it. Use it; do not change it. |
| `runner/`, `evaluator/`, `final-evals/`, `space/`, `infra/`, `competition.yaml` | The leaderboard's own code. The leaderboard runs its own copy, so a change here only makes local runs disagree with it. |

Never write an API key or token into `submission/`: everything in it is
uploaded, and submissions are made public. Keys come from the environment
(`NDIF_API_KEY`, `HF_TOKEN`).

## How to work

Run every command with the repository's environment, `.venv/bin/python`. A
shell that has not activated `.venv` has a different `python`.

1. **Run locally:** `.venv/bin/python submit.py --local`. It runs the entry on
   the practice set, 10 questions, and scores it, with full errors.
2. **Run it on questions of your own.** The scored datasets include kinds of
   question that the practice set does not have. See the next section.
3. **Submit:** `.venv/bin/python submit.py --wait`. Ask the person before each
   submission. Before the first one, tell them that submitting agrees to the
   challenge's terms: the submission, its code and its associated materials are
   published under the MIT license (the README's Rules section). The first one
   also needs `--team "<name>"`: ask them for the name, which is 2 to 40
   characters, must not be taken, and stays with their NDIF account.

**Running questions of your own.** The leaderboard runs the entry on four
datasets, named `dataset_a`, `dataset_b`, `dataset_c` and `dataset_d`. What kind of question
each holds is not published, and they are not all of the practice set's kind. An
entry that crashes on one of them fails the whole submission, with no log.
Before submitting, write a few questions that differ from the practice ones, put
them in a directory as `questions.jsonl`, and start the entry on it the way the
leaderboard does:

```bash
cd submission
DATASET_NAME=mine DATASET_PATH=/path/to/that/directory OUTPUT_DIR=/path/to/an/output/directory \
  MODEL_ID=thinkingmachines/Inkling-2-Mini NDIF_HOST=https://api.ndif.us ../.venv/bin/python main.py
```

A question needs `id`, `model_input`, `read_position` and `read_token`.
`model_input` is either the prompt as it is or the prompt inside the model's
chat format with the reply started, as in the practice questions: try both.
`read_position` is the index of the last token and `read_token` is that token's
text: the starter checks both and stops if they disagree. An entry must not
depend on `$DATASET_NAME` being a name it knows.

**Reading a local run.** It prints pass rates only. To see which questions
failed, compare what the lens wrote with the answers:

| File | Contents |
|---|---|
| `.leaderboard/outputs/practice/readouts.jsonl` | What the entry wrote. |
| `.leaderboard/outputs/practice.log` | The entry's own output. |
| `.leaderboard/data/practice/private/questions.jsonl` | The practice questions with `intermediate` and `options`. |

**Things to know before trusting or repeating a run:**

- **Do not use the leaderboard to find errors.** A run that fails there returns
  no log, because a log could quote the hidden questions. Submit only an entry
  that passes `--local` and has run on questions of your own.
- **A local run sends fewer requests than the leaderboard.** The practice set is
  10 questions and a scored dataset is 90 or 100, so with the starter's 25
  questions per request a local run never sends a second request. Before
  submitting, run once with the batch size below 10, so that the code between
  requests runs too.
- **A team can submit once every 6 hours.** A submission that fails on the
  entry's own error uses the attempt up; `submit.py --status` shows when the
  next one opens. A submission takes about 12 minutes, its judging is paid for
  by the organizers, and a team can have one queued or running at a time.
- **Only an admitted account can submit.** The registration form linked in the
  README is also a waitlist. If the leaderboard answers that the account is not
  registered, the person has either not filled in the form, with the email
  their NDIF key was created with, or is still waiting for a place. Local runs
  work in the meantime.
- **`--wait` only watches.** Stopping it does not stop the submission.
  `submit.py --status <id>` follows it again.
- **The practice score is coarse.** It is 10 questions of one kind, so one
  question moves it by 0.10, and it says nothing about the kinds of question
  the practice set does not have.
  A local run scores by exact matching: a token names an option only if it
  equals it. The leaderboard's judge also accepts inflections and translations,
  so the two can differ in either direction. To score locally with that judge,
  run `uv sync --group evaluator` once and set `OPENAI_API_KEY`. The judge
  itself changes its verdict on one or two questions in a few hundred from run
  to run. Do not read a one-question change as an improvement.
- **Do not edit files in `submission/` while a run is going.** nnsight reads the
  lens's source from disk when it builds a request, and a changed file breaks
  the requests that follow.

## What an entry must do

The leaderboard starts `submission/main.py` (or `main.ipynb`) once per dataset,
in the `submission/` directory, with `DATASET_NAME`, `DATASET_PATH`,
`OUTPUT_DIR`, `MODEL_ID` and the participant's credentials in the environment.
It reads `$DATASET_PATH/questions.jsonl` and writes `$OUTPUT_DIR/readouts.jsonl`,
one line per question. The README gives both formats.

| Rule | What breaking it costs |
|---|---|
| A line for every question | A question with no line fails. |
| 1 to 10 readouts on a line, each with a different layer label | The question fails. |
| Each token at most 64 characters | The question fails. |
| At most 10 tokens per readout, as text, best first | Tokens after the tenth are ignored. |
| One line per question, every line a JSON object | The whole submission is not scored. |
| The run exits with code 0 for every dataset | The whole submission fails. |
| 60 minutes for the whole run, all datasets together, and 200 MB for the entry | The submission fails or is refused. |

A readout's entries are tokens of the model's vocabulary, written as text. The
judge is told that each one is a single token read from one activation.

The environment is fixed:

- **Packages:** only what plain `uv sync` installs. An entry cannot add any.
- **Network:** on the leaderboard the entry reaches NDIF and, read-only,
  Hugging Face. Nothing else.
- **Weights of the model:** code on NDIF cannot read them. It can call the
  model's modules (`model.lm_head(x)`).
- **Request size:** NDIF accepts requests up to about 1 MB, and a tensor used
  inside a trace travels in the request. Trained weights go to the Hugging Face
  Hub and are fetched on NDIF with `lenses.fetch`, which returns them on the
  CPU: move each to the device and dtype of the activation it meets.

## How the score is computed

The README states the rules. These are the consequences for a lens:

- **Each readout casts one vote,** for the option named by the highest-ranked
  token on it that names one. Tokens after that one change nothing. A readout
  that names no option casts no vote.
- **A question passes only if the intermediate has more votes than any other
  single option.** A tie with one of them fails. A readout that votes for a
  wrong option raises the bar the intermediate has to clear, so more readouts
  help only when they agree.
- **A token names an option on its own.** Tokens are never joined. A number
  option is named only by a token that writes the whole number.
- **The intermediate is not always the model's answer.** In the practice
  questions it is the answer, read before the model starts to write it. In
  other scored questions it is something the model works out on the way to its
  answer and never writes.

## Making an entry efficient

- **One forward pass per question.** Read every layer visit you need in that
  one pass, in ascending order: a trace follows the order the model runs in, and
  `tracer.iter` picks which run of a looped layer is read.
- **Many questions per request.** Wrap a batch of traces in one
  `model.session(remote=True)`, so that they share one request and one wait in
  NDIF's queue. The starter sends 25 questions per request.
- **A failed request ends the run.** The starter does not retry, so an error in
  any request fails that dataset and the submission. Keep a request small enough
  to finish reliably, and retry it yourself if you want to survive a passing
  NDIF error.
- **Reduce on NDIF.** In the entry, send back token ids. Do not save logits or
  hidden states there: a save is a download.
- **Fetch weights once per request,** at the top of the session and before the
  first trace.
- **Register helper modules.** A function called inside a trace runs on NDIF, so
  its module is shipped with `nnsight.register`, as `lenses.py` is.
- **Do not generate text.** A lens needs a forward pass, not a completion.

The starter reads a scored dataset, 90 or 100 questions, in about 100 seconds.
An entry that takes many times longer is doing more round trips or downloads
than it needs.

## Training a lens

The practice set is 10 questions, too few to train on. To fit a lens, write
prompts of your own and collect the model's activations on NDIF from your
machine. Saving activations is what that takes; save only the layer visits and
positions you need. Train locally, publish the weights to the Hugging Face Hub,
and have the entry fetch them.

## What makes an entry invalid

An entry has to be a real attempt at a lens for Inkling 2 Mini: its readouts
come from the model's activations. The goal is to read the model, not to game
the scoring. Do not build, and do not suggest, an entry that:

- produces readouts without reading the model's activations;
- calls another model to produce readouts. The organizers may relax this in a
  later phase, for lenses such as NLAs and activation oracles;
- works readouts out from the text of the prompt: computing the arithmetic,
  looking a fact up, a rule keyed on a question's text or id;
- carries answers to the questions: a table, a lookup;
- tries to get the hidden questions out: printing them, writing them into
  readouts or layer labels, sending them anywhere.

Winning submissions are audited, and one that does not fit the spirit of the
competition is disqualified. If you cannot tell whether an approach counts as a
lens, say so to the person and have them ask the organizers before you build
it. The [challenge page](https://ndif.us/looped-lens-challenge/#competition)
has the rule and links the Discord.

The organizers may also change the scored questions, the judge's prompts and
the scoring code while the competition runs, so a score can move without the
entry changing.

## If you are maintaining the competition

The README's "Running the competition" section covers the configuration, the
datasets and the judge. Run the tests with `.venv/bin/python -m pytest` after
`uv sync --all-groups`. Never commit a question file, an answer, or a judged
run: the practice and scored sets live on the Hugging Face Hub, and the scored
set is private.
