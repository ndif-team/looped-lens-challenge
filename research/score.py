"""Score a readouts file against labelled questions, by exact matching, the way a local run does.

    .venv/bin/python research/score.py QUESTIONS.jsonl READOUTS.jsonl [--details]

A token names an option when the two are equal after the official normalization (lower case,
spaces and punctuation removed: final-evals/lens_eval/scoring.py ``_norm``). Each readout votes for
the option its best-ranked naming token names; a question passes when the intermediate has
strictly more votes than every other option. The leaderboard's judge also accepts inflections
and translations, so it can only score a question higher than this.
"""

import argparse
import json
import re
from collections import Counter, defaultdict


def norm(s) -> str:
    return re.sub(r"[\s\.\,\:\;\!\?\"'“”‘’\(\)\[\]\*_]+", "", str(s)).lower()


def votes_of(question: dict, readouts: list[dict]) -> dict:
    """{layer: option} for each readout that names exactly one option at its best rank."""
    options = {norm(o): o for o in question["options"]}
    votes = {}
    for readout in readouts:
        for token in readout["tokens"][:10]:
            if (t := norm(token)) in options:
                votes[readout["layer"]] = options[t]
                break
    return votes


def passes(question: dict, votes: dict) -> bool:
    count = Counter(votes.values())
    gold = question["intermediate"]
    return count[gold] > max((n for o, n in count.items() if o != gold), default=0)


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score(questions: list[dict], readouts: dict) -> dict:
    """{family: (passed, total)}, plus an "overall" entry: the mean of the family scores, as the
    leaderboard takes the mean of its datasets."""
    by_family = defaultdict(lambda: [0, 0])
    for q in questions:
        tally = by_family[q.get("family", "all")]
        tally[1] += 1
        if q["id"] in readouts and passes(q, votes_of(q, readouts[q["id"]])):
            tally[0] += 1
    result = {family: tuple(t) for family, t in sorted(by_family.items())}
    result["overall"] = sum(p / n for p, n in result.values()) / len(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions")
    parser.add_argument("readouts")
    parser.add_argument("--details", action="store_true", help="print each question's votes")
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    readouts = {r["id"]: r["readouts"] for r in read_jsonl(args.readouts)}
    if args.details:
        for q in questions:
            votes = votes_of(q, readouts.get(q["id"], []))
            mark = "PASS" if passes(q, votes) else "fail"
            print(f"{mark}  {q['id']:<22} {q['intermediate']:<12} " + " ".join(f"{l}:{v}" for l, v in sorted(votes.items())))
    result = score(questions, readouts)
    for family, value in result.items():
        if family != "overall":
            print(f"  {family:<20} {value[0]:>4} / {value[1]:<4} {value[0] / value[1]:.3f}")
    print(f"  {'overall':<20} {result['overall']:.3f}")


if __name__ == "__main__":
    main()
