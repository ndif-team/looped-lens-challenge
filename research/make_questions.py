"""Write a labelled dev set of our own, in the leaderboard's question format.

    .venv/bin/python research/make_questions.py [--out research/data/dev] [--per-family 30]

Writes ``questions.jsonl`` (all fields, as the evaluator sees them) and ``inputs/questions.jsonl``
(only what an entry is given) under ``--out``. Every question appears twice: chat-wrapped as in
the practice set, and as the plain prompt (``-plain`` ids).

Families:
- single-hop, the intermediate is the answer (as in the practice set): sport, country,
  instrument, one-step arithmetic, day and month shifts;
- two-hop, the intermediate is a step the model never writes: the country of a landmark when
  asked for its capital or language, the inner sum of ``(a + b) × c``, the day or month in
  between for a shift of two.

Answers and intermediates are known by construction. These questions are for measuring lenses,
never for an entry to read: they stay in research/data/, which is git-ignored.
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
CHAT = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{}<|end_message|><|message_model|>"
SUFFIX = " Answer in one word."

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

ATHLETES = {
    "Sidney Crosby": "hockey", "Wayne Gretzky": "hockey", "Alex Ovechkin": "hockey", "Connor McDavid": "hockey",
    "Tiger Woods": "golf", "Phil Mickelson": "golf", "Rory McIlroy": "golf", "Jack Nicklaus": "golf",
    "Roger Federer": "tennis", "Rafael Nadal": "tennis", "Novak Djokovic": "tennis", "Serena Williams": "tennis",
    "LeBron James": "basketball", "Stephen Curry": "basketball", "Kobe Bryant": "basketball", "Shaquille O'Neal": "basketball",
    "Babe Ruth": "baseball", "Derek Jeter": "baseball", "Shohei Ohtani": "baseball", "Hank Aaron": "baseball",
    "Muhammad Ali": "boxing", "Mike Tyson": "boxing", "Floyd Mayweather": "boxing", "Manny Pacquiao": "boxing",
    "Sachin Tendulkar": "cricket", "Virat Kohli": "cricket", "Don Bradman": "cricket",
    "Michael Phelps": "swimming", "Katie Ledecky": "swimming", "Mark Spitz": "swimming",
    "Magnus Carlsen": "chess", "Garry Kasparov": "chess", "Bobby Fischer": "chess",
    "Simone Biles": "gymnastics", "Nadia Comaneci": "gymnastics",
    "Eddy Merckx": "cycling", "Chris Froome": "cycling",
    "Kelly Slater": "surfing", "Tony Hawk": "skateboarding",
}
SPORT_DISTRACTORS = ["tennis", "hockey", "golf", "basketball", "baseball", "boxing", "cricket", "swimming", "chess",
                     "gymnastics", "cycling", "surfing", "rugby", "sumo", "karate", "volleyball", "fencing", "rowing"]

MUSICIANS = {
    "Jimi Hendrix": "guitar", "Eric Clapton": "guitar", "B.B. King": "guitar", "Jimmy Page": "guitar",
    "Yo-Yo Ma": "cello", "Jacqueline du Pré": "cello", "Pablo Casals": "cello",
    "Itzhak Perlman": "violin", "Niccolò Paganini": "violin", "Joshua Bell": "violin",
    "Louis Armstrong": "trumpet", "Miles Davis": "trumpet", "Dizzy Gillespie": "trumpet",
    "John Coltrane": "saxophone", "Charlie Parker": "saxophone", "Kenny G": "saxophone",
    "Lang Lang": "piano", "Glenn Gould": "piano", "Vladimir Horowitz": "piano",
    "Ringo Starr": "drums", "Neil Peart": "drums", "John Bonham": "drums",
    "Earl Scruggs": "banjo", "Béla Fleck": "banjo",
    "Benny Goodman": "clarinet", "Ravi Shankar": "sitar", "James Galway": "flute",
}
INSTRUMENT_DISTRACTORS = ["guitar", "cello", "violin", "trumpet", "saxophone", "piano", "drums", "banjo", "clarinet",
                          "sitar", "flute", "harp", "accordion", "trombone", "organ", "ukulele", "oboe", "tuba"]

# landmark: (country, capital or None, language or None)
LANDMARKS = {
    "the Eiffel Tower": ("France", "Paris", "French"), "the Louvre": ("France", "Paris", "French"),
    "Mont Saint-Michel": ("France", "Paris", "French"),
    "the Colosseum": ("Italy", "Rome", "Italian"), "the Leaning Tower of Pisa": ("Italy", "Rome", "Italian"),
    "the Trevi Fountain": ("Italy", "Rome", "Italian"),
    "the Sagrada Familia": ("Spain", "Madrid", "Spanish"), "the Alhambra": ("Spain", "Madrid", "Spanish"),
    "the Acropolis": ("Greece", "Athens", "Greek"), "the Parthenon": ("Greece", "Athens", "Greek"),
    "the Taj Mahal": ("India", None, "Hindi"),
    "the Great Wall": ("China", "Beijing", "Chinese"), "the Forbidden City": ("China", "Beijing", "Chinese"),
    "Machu Picchu": ("Peru", "Lima", "Spanish"),
    "Christ the Redeemer": ("Brazil", None, "Portuguese"),
    "the Sydney Opera House": ("Australia", "Canberra", "English"), "Uluru": ("Australia", "Canberra", "English"),
    "Bondi Beach": ("Australia", "Canberra", "English"),
    "Petra": ("Jordan", "Amman", "Arabic"),
    "Angkor Wat": ("Cambodia", None, "Khmer"),
    "the Brandenburg Gate": ("Germany", "Berlin", "German"), "Neuschwanstein Castle": ("Germany", "Berlin", "German"),
    "the Kremlin": ("Russia", "Moscow", "Russian"), "Red Square": ("Russia", "Moscow", "Russian"),
    "Mount Fuji": ("Japan", "Tokyo", "Japanese"), "Kinkaku-ji": ("Japan", "Tokyo", "Japanese"),
    "the Pyramids of Giza": ("Egypt", "Cairo", "Arabic"), "the Sphinx": ("Egypt", "Cairo", "Arabic"),
    "Chichen Itza": ("Mexico", None, "Spanish"),
    "the Hagia Sophia": ("Turkey", "Ankara", "Turkish"),
    "the CN Tower": ("Canada", "Ottawa", None),
    "Wawel Castle": ("Poland", "Warsaw", "Polish"),
    "the Atomium": ("Belgium", "Brussels", None),
    "the Rijksmuseum": ("Netherlands", "Amsterdam", "Dutch"),
    "the Little Mermaid statue": ("Denmark", "Copenhagen", "Danish"),
    "Hallgrímskirkja": ("Iceland", "Reykjavik", "Icelandic"),
    "Schönbrunn Palace": ("Austria", "Vienna", "German"),
    "the Moai statues of Easter Island": ("Chile", "Santiago", "Spanish"),
    "Ha Long Bay": ("Vietnam", "Hanoi", "Vietnamese"),
    "Lake Bled": ("Slovenia", "Ljubljana", "Slovenian"),
    "the Cliffs of Moher": ("Ireland", "Dublin", None),
}
COUNTRY_DISTRACTORS = sorted({c for c, _, _ in LANDMARKS.values()} | {"Sweden", "Norway", "Portugal", "Argentina", "Kenya", "Thailand"})


def single_token(tokenizer, word: str) -> bool:
    """Whether some casing of the word, with or without a leading space, is one token."""
    return any(len(tokenizer(v, add_special_tokens=False)["input_ids"]) == 1
               for w in {word, word.lower(), word.capitalize()} for v in (w, " " + w))


def options_for(rng, answer, pool, extra=()):
    """The intermediate plus four distractors: the ``extra`` ones first (hard distractors such as
    the final answer of a two-hop question), then random ones from ``pool``."""
    distractors = [e for e in dict.fromkeys(extra) if e != answer][:4]
    rest = [p for p in pool if p != answer and p not in distractors]
    distractors += rng.sample(rest, 4 - len(distractors))
    options = [answer] + distractors
    rng.shuffle(options)
    return options


def build(rng, per_family, ok):
    """``ok(word)``: whether a word can be an option (it has a single-token form)."""
    global SPORT_DISTRACTORS, INSTRUMENT_DISTRACTORS, COUNTRY_DISTRACTORS
    SPORT_DISTRACTORS = [w for w in SPORT_DISTRACTORS if ok(w)]
    INSTRUMENT_DISTRACTORS = [w for w in INSTRUMENT_DISTRACTORS if ok(w)]
    COUNTRY_DISTRACTORS = [w for w in COUNTRY_DISTRACTORS if ok(w)]
    rows = []  # (family, prompt, intermediate, answer, options)

    def add(family, items):
        items = list({item[0]: item for item in items}.values())  # one question per prompt
        for prompt, intermediate, answer, options in items[:per_family]:
            rows.append((family, prompt, intermediate, answer, options))

    athletes = [(a, s) for a, s in ATHLETES.items() if ok(s)]; rng.shuffle(athletes)
    add("sport", [(f"What sport does the athlete {a} play?", s, s, options_for(rng, s, SPORT_DISTRACTORS)) for a, s in athletes])
    musicians = [(m, i) for m, i in MUSICIANS.items() if ok(i)]; rng.shuffle(musicians)
    add("instrument", [(f"Which instrument is the musician {m} known for playing?", i, i, options_for(rng, i, INSTRUMENT_DISTRACTORS)) for m, i in musicians])
    landmarks = [(l, v) for l, v in LANDMARKS.items() if ok(v[0])]; rng.shuffle(landmarks)
    add("country", [(f"In which country is {l}?", c, c, options_for(rng, c, COUNTRY_DISTRACTORS)) for l, (c, _, _) in landmarks])

    arith = []
    for _ in range(per_family * 3):
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        op = rng.choice("+-×")
        if op == "-" and b > a:
            a, b = b, a
        value = a + b if op == "+" else a - b if op == "-" else a * b
        if value <= 20:
            arith.append((f"What is {a} {op} {b}?", str(value), str(value), options_for(rng, str(value), [str(n) for n in range(0, 21)])))
    add("arith1", arith)

    shifts = []
    for names, unit in ((DAYS, "day"), (MONTHS, "month")):
        for i, name in enumerate(names):
            for word, step in (("after", 1), ("before", -1)):
                target = names[(i + step) % len(names)]
                shifts.append((f"What is the {unit} {word} {name}?", target, target, options_for(rng, target, names, extra=[name])))
    rng.shuffle(shifts)
    add("shift1", shifts)

    # Two-hop: the intermediate is never written.
    hop = []
    for landmark, (country, capital, language) in landmarks:
        if capital:
            hop.append((f"What is the capital of the country where {landmark} is?", country, capital, options_for(rng, country, COUNTRY_DISTRACTORS)))
        if language:
            hop.append((f"What language is spoken in the country where {landmark} is?", country, language, options_for(rng, country, COUNTRY_DISTRACTORS)))
    rng.shuffle(hop)
    add("hop_country", hop)

    arith2 = []
    for _ in range(per_family * 4):
        a, b, c = rng.randint(1, 6), rng.randint(1, 6), rng.randint(2, 5)
        inner = a + b
        answer = inner * c
        # The final answer is a distractor: a lens that shows it votes against the intermediate.
        arith2.append((f"What is ({a} + {b}) × {c}?", str(inner), str(answer),
                       options_for(rng, str(inner), [str(n) for n in range(0, 13)], extra=[str(answer), str(c)])))
    add("arith2", arith2)

    shift2 = []
    for names, unit in ((DAYS, "day"), (MONTHS, "month")):
        for i, name in enumerate(names):
            for word, step in (("after", 1), ("before", -1)):
                middle = names[(i + step) % len(names)]
                answer = names[(i + 2 * step) % len(names)]
                shift2.append((f"What is the {unit} two {unit}s {word} {name}?", middle, answer, options_for(rng, middle, names, extra=[answer, name])))
    rng.shuffle(shift2)
    add("shift2", shift2)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "dev")
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rng = random.Random(args.seed)

    questions, skipped = [], 0
    for n, (family, prompt, intermediate, answer, options) in enumerate(build(rng, args.per_family, lambda w: single_token(tokenizer, w))):
        if not all(single_token(tokenizer, o) for o in options):
            skipped += 1
            continue
        prompt = prompt + SUFFIX
        for style, model_input in (("chat", CHAT.format(prompt)), ("plain", prompt)):
            ids = tokenizer(model_input, add_special_tokens=False)["input_ids"]
            questions.append({
                "id": f"{family}-{n:03d}" + ("-plain" if style == "plain" else ""),
                "family": family, "style": style, "prompt": prompt, "model_input": model_input,
                "read_position": len(ids) - 1, "read_token": tokenizer.decode([ids[-1]]),
                "intermediate": intermediate, "answer": answer, "options": options,
            })

    (args.out / "inputs").mkdir(parents=True, exist_ok=True)
    with open(args.out / "questions.jsonl", "w") as full, open(args.out / "inputs" / "questions.jsonl", "w") as given:
        for q in questions:
            full.write(json.dumps(q, ensure_ascii=False) + "\n")
            given.write(json.dumps({k: q[k] for k in ("id", "prompt", "model_input", "read_position", "read_token")}, ensure_ascii=False) + "\n")
    families = {}
    for q in questions:
        families[q["family"]] = families.get(q["family"], 0) + 1
    print(f"wrote {len(questions)} questions to {args.out} ({skipped} skipped for multi-token options): {families}")


if __name__ == "__main__":
    main()
