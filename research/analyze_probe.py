"""Read the probe's results: which readout, visit and position carry each family's intermediate.

    .venv/bin/python research/analyze_probe.py [--probe research/data/probe] [--questions ...]

A *config* is (visit, readout, position from the end; -1 is the read position). For each question
and config the probe's ranks say which option that readout would vote for under the official rule
(the best-ranked option among the readout's top ten, if any). From that this prints:

1. the starter's lens replayed, as a check against the real runs;
2. per family: the share of questions where each readout at the read position votes for the
   intermediate, at the best visits;
3. the same over positions: which position carries the intermediate best, per family;
4. a greedy choice of up to ten configs with distinct visits (a question's readouts need distinct
   layer labels), fitted on half the prompts and scored on the other half (a prompt's chat and plain
   forms fall in the same half), with the pass rate as the leaderboard takes it: the mean of the
   family pass rates.
"""

import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

READOUTS = ("active", "mixed", "mlp", "out")
STARTER = [(v, 0, -1) for v in (12, 24, 34, 35, 36, 43, 44, 45, 46, 47)]


def load(probe: Path, questions_path: Path):
    questions = {q["id"]: q for q in map(json.loads, open(questions_path))}
    rows = [r for path in sorted(probe.glob("batch-*.pt")) for r in torch.load(path)]
    return questions, rows


def votes_array(rows, questions, positions):
    """votes [questions, 48, readouts, positions]: the option index each config votes for, or -1;
    gold [questions]: the intermediate's option index."""
    votes = np.full((len(rows), 48, len(READOUTS), positions), -1, dtype=np.int8)
    gold = np.zeros(len(rows), dtype=np.int8)
    for n, row in enumerate(rows):
        q = questions[row["id"]]
        gold[n] = q["options"].index(q["intermediate"])
        ranks = row["ranks"].numpy().astype(np.int32)  # [48, R, P, V]
        owners = np.array(row["owners"])
        per_option = np.full(ranks.shape[:3] + (len(q["options"]),), 10**6, dtype=np.int32)
        for o in range(len(q["options"])):
            if (owners == o).any():
                per_option[..., o] = ranks[..., owners == o].min(-1)
        best = per_option.min(-1)
        unique = (per_option == best[..., None]).sum(-1) == 1
        choice = np.where((best < 10) & unique, per_option.argmin(-1), -1)
        p = choice.shape[2]
        votes[n, :, :, positions - p :] = choice  # right-align: the last position is the read position
    return votes, gold


def pass_rate(votes, gold, configs, families, mask=None):
    """Mean over families of the share of questions that pass with ``configs`` [(visit, readout, pos)]."""
    mask = np.ones(len(gold), bool) if mask is None else mask
    counts = np.zeros((len(gold), 5), dtype=np.int16)
    for v, r, p in configs:
        chosen = votes[:, v, r, p]
        ok = chosen >= 0
        counts[np.nonzero(ok)[0], chosen[ok]] += 1
    g = counts[np.arange(len(gold)), gold]
    other = counts.copy()
    other[np.arange(len(gold)), gold] = -1
    passed = g > other.max(1)
    rates = {f: passed[(families == f) & mask].mean() for f in np.unique(families[mask])}
    return float(np.mean(list(rates.values()))), rates


def greedy(votes, gold, families, mask, max_configs=10):
    chosen, used, best_score = [], set(), -1.0
    candidates = [(v, r, p) for v in range(48) for r in range(len(READOUTS)) for p in range(votes.shape[3])]
    for _ in range(max_configs):
        scored = [(pass_rate(votes, gold, chosen + [c], families, mask)[0], c) for c in candidates if c[0] not in used]
        score, config = max(scored)
        if score <= best_score:
            break
        chosen.append(config)
        used.add(config[0])
        best_score = score
    return sorted(chosen), best_score


def name(config, positions):
    v, r, p = config
    return f"v{v}/{READOUTS[r]}/pos{p - positions}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--probe", type=Path, default=here / "data" / "probe")
    parser.add_argument("--questions", type=Path, default=here / "data" / "dev" / "questions.jsonl")
    args = parser.parse_args()
    questions, rows = load(args.probe, args.questions)
    positions = max(len(r["positions"]) for r in rows)
    votes, gold = votes_array(rows, questions, positions)
    families = np.array([questions[r["id"]]["family"] for r in rows])
    styles = np.array([questions[r["id"]]["style"] for r in rows])
    read = positions - 1
    print(f"{len(rows)} questions, {positions} positions, families: {dict(zip(*np.unique(families, return_counts=True)))}\n")

    print("1. The starter's lens, replayed")
    for style in ("chat", "plain"):
        score, rates = pass_rate(votes, gold, STARTER, families, styles == style)
        print(f"   {style:<6} {score:.3f}   " + "  ".join(f"{f} {x:.2f}" for f, x in rates.items()))

    print("\n2. Share of chat questions where one readout at the read position votes for the intermediate (best 3 visits)")
    chat = styles == "chat"
    for f in np.unique(families):
        m = chat & (families == f)
        cells = []
        for r, readout in enumerate(READOUTS):
            hit = (votes[m, :, r, read] == gold[m, None]).mean(0)  # [48]
            top = np.argsort(-hit)[:3]
            cells.append(f"{readout}: " + ",".join(f"v{v}={hit[v]:.2f}" for v in top))
        print(f"   {f:<12} " + " | ".join(cells))

    print("\n3. The best single config per family, over all positions (chat / plain)")
    for f in np.unique(families):
        line = []
        for style in ("chat", "plain"):
            m = (styles == style) & (families == f)
            hit = (votes[m] == gold[m, None, None, None]).mean(0)  # [48, R, P]
            v, r, p = np.unravel_index(hit.argmax(), hit.shape)
            at_read = hit[:, :, read].max()
            line.append(f"{style}: {name((v, r, p), positions)}={hit[v, r, p]:.2f} (read pos best {at_read:.2f})")
        print(f"   {f:<12} " + " | ".join(line))

    print("\n4. Greedy choice of up to 10 readouts (distinct visits), fitted on half the prompts")
    prompt_ids = np.array([questions[r["id"]]["id"].removesuffix("-plain") for r in rows])
    fit = np.array([zlib.crc32(p.encode()) % 2 == 0 for p in prompt_ids])  # stable across runs
    for label, allowed in (("read position only", [read]), ("any of the last positions", list(range(positions)))):
        sub = np.full_like(votes, -1)
        sub[..., allowed] = votes[..., allowed]
        configs, fitted = greedy(sub, gold, families, fit)
        held, rates = pass_rate(votes, gold, configs, families, ~fit)
        print(f"   {label}: fitted {fitted:.3f}, held out {held:.3f}")
        print(f"     configs: {', '.join(name(c, positions) for c in configs)}")
        print("     held out by family: " + "  ".join(f"{f} {x:.2f}" for f, x in rates.items()))
    starter, rates = pass_rate(votes, gold, STARTER, families, ~fit)
    print(f"   starter on the same held-out half: {starter:.3f}   " + "  ".join(f"{f} {x:.2f}" for f, x in rates.items()))


if __name__ == "__main__":
    main()
