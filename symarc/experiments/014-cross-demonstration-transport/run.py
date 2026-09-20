"""014: exact local transport with whole-demonstration cross-validation.

Three separate commands quarantine query answers: prepare, predict, score.
Prediction learns exact lookup tables; unseen keys abstain, never copy by default.
Only Python's standard library is used. No SymArc solver is reimplemented.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

RADII = (0, 1, 2, 3)
MODES = ("local", "positioned", "memorise")
UNKNOWN = -1
PAD = 10


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def shape(grid: list[list[int]]) -> tuple[int, int]:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("A grid must be nonempty and rectangular")
    if any(type(v) is not int or not 0 <= v <= 9 for row in grid for v in row):
        raise ValueError("Grid colours must be integers 0..9")
    if len(grid) > 30 or len(grid[0]) > 30:
        raise ValueError("This experiment uses ARC's 30x30 grid bound")
    return len(grid), len(grid[0])


def signature(grid: list[list[int]]) -> bytes:
    h, w = shape(grid)
    return bytes((h, w)) + bytes(v for row in grid for v in row)


def project(task: dict[str, Any], task_id: str, split: str) -> dict[str, Any]:
    # Deliberately never accesses task['test'][i]['output'].
    return {"id": task_id, "split": split, "train": task["train"],
            "query_inputs": [pair["input"] for pair in task["test"]]}


def prepare(data: Path, out: Path) -> None:
    problems, answers, hashes = [], {}, {}
    for split in ("training", "evaluation"):
        for path in sorted((data / split).glob("*.json")):
            raw = path.read_bytes()
            task = json.loads(raw)
            key = f"{split}/{path.stem}"
            problems.append(project(task, path.stem, split))
            answers[key] = [pair["output"] for pair in task["test"]]
            hashes[key] = hashlib.sha256(raw).hexdigest()
    if not problems:
        raise ValueError("No task JSON files found")
    write_json(out / "problems.json", problems)
    write_json(out / "answers.json", answers)
    write_json(out / "data.json", {"files": hashes, "files_sha256": digest(hashes),
                                   "tasks": len(problems)})


def features(grid: list[list[int]], mode: str, radius: int) -> list[bytes | tuple]:
    h, w = shape(grid)
    if mode == "memorise":
        s = signature(grid)
        return [(s, r, c) for r in range(h) for c in range(w)]
    keys = []
    for r in range(h):
        for c in range(w):
            patch = bytes(grid[rr][cc] if 0 <= rr < h and 0 <= cc < w else PAD
                          for rr in range(r-radius, r+radius+1)
                          for cc in range(c-radius, c+radius+1))
            keys.append((bytes((h, w, r, c)) + patch) if mode == "positioned" else patch)
    return keys


def fit(train: list[dict], mode: str, cache: dict) -> tuple[int | None, dict | None]:
    for radius in ((0,) if mode == "memorise" else RADII):
        table = {}
        conflict = False
        for pair in train:
            grid = pair["input"]
            ck = (signature(grid), mode, radius)
            if ck not in cache:
                cache[ck] = features(grid, mode, radius)
            labels = [v for row in pair["output"] for v in row]
            for key, label in zip(cache[ck], labels, strict=True):
                previous = table.setdefault(key, label)
                if previous != label:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            return radius, table
    return None, None


def predict_grid(grid: list[list[int]], mode: str, radius: int | None,
                 table: dict | None, cache: dict) -> list[list[int]] | None:
    if table is None:
        return None
    ck = (signature(grid), mode, radius)
    if ck not in cache:
        cache[ck] = features(grid, mode, radius)
    values = [table.get(key, UNKNOWN) for key in cache[ck]]
    w = len(grid[0])
    return [values[k:k+w] for k in range(0, len(values), w)]


def metrics(pred: list[list[int]] | None, target: list[list[int]],
            input_grid: list[list[int]]) -> dict[str, Any]:
    h, w = shape(target)
    total = h*w
    same_shape = shape(input_grid) == (h, w)
    changed = sum(a != b for rowa, rowb in zip(input_grid, target)
                  for a, b in zip(rowa, rowb)) if same_shape else None
    complete = pred is not None and all(v != UNKNOWN for row in pred for v in row)
    aligned = pred is not None and len(pred) == h and all(len(row) == w for row in pred)
    known = correct = changed_known = changed_correct = 0
    if aligned:
        for r in range(h):
            for c in range(w):
                p, y = pred[r][c], target[r][c]
                if p != UNKNOWN:
                    known += 1
                    correct += p == y
                    if same_shape and input_grid[r][c] != y:
                        changed_known += 1
                        changed_correct += p == y
    return dict(complete=complete, exact=pred == target, cells=total, known=known,
                correct=correct, wrong=known-correct, changed=changed,
                changed_known=changed_known, changed_correct=changed_correct,
                shape_changed=not same_shape)


def infer(problem: dict[str, Any]) -> dict[str, Any]:
    train = problem["train"]
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, pair in enumerate(train):
        groups[signature(pair["input"])].append(i)
    eligible = len(groups) >= 2 and all(shape(e["input"]) == shape(e["output"]) for e in train)
    result = {"id": problem["id"], "split": problem["split"], "eligible": eligible,
              "n_demonstrations": len(train), "n_distinct_inputs": len(groups),
              "query_inputs": problem["query_inputs"], "pipelines": {}}
    if not eligible:
        return result
    cache: dict = {}
    for mode in MODES:
        radius, table = fit(train, mode, cache)
        folds = []
        for excluded in groups.values():
            reduced = [e for i, e in enumerate(train) if i not in excluded]
            fold_radius, fold_table = fit(reduced, mode, cache)
            scores = [metrics(predict_grid(train[i]["input"], mode, fold_radius, fold_table, cache),
                              train[i]["output"], train[i]["input"]) for i in excluded]
            folds.append({"excluded": excluded, "radius": fold_radius, "scores": scores})
        all_cv = all(s["exact"] for fold in folds for s in fold["scores"])
        result["pipelines"][mode] = {
            "radius": radius, "compatible": table is not None,
            "table_keys": len(table) if table is not None else 0,
            "cv_exact": all_cv, "folds": folds,
            "predictions": [predict_grid(g, mode, radius, table, cache) for g in problem["query_inputs"]],
        }
    return result


def predict(problems_path: Path, out: Path, workers: int) -> None:
    # This process reads only the projected problems file, not the answers file.
    problems = json.loads(problems_path.read_text())
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(infer, problems))
    write_json(out / "predictions.json", results)
    write_json(out / "prediction-run.json", {
        "workers": workers, "seconds": time.perf_counter()-start,
        "seed": None, "python": sys.version, "platform": platform.platform(),
        "radii": RADII, "modes": MODES, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "problems_sha256": hashlib.sha256(problems_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256((out / "predictions.json").read_bytes()).hexdigest(),
    })


def score(predictions_path: Path, answers_path: Path, out: Path) -> None:
    rows = json.loads(predictions_path.read_text())
    answers = json.loads(answers_path.read_text())
    summaries: dict[tuple[str, str], Counter] = defaultdict(Counter)
    task_scores = []
    for row in rows:
        key = f"{row['split']}/{row['id']}"
        ys = answers[key]
        task_score = {"id": row["id"], "split": row["split"], "eligible": row["eligible"], "policies": {}}
        for mode in (*MODES, "identity"):
            model = row["pipelines"].get(mode)
            for gated in ((False,) if mode == "identity" else (False, True)):
                policy = mode + ("_cv_gate" if gated else "")
                eligible = row["eligible"]
                compatible = model is not None and model["compatible"]
                supported = model is not None and model["cv_exact"]
                if mode == "identity":
                    ps = row["query_inputs"] if eligible else [None]*len(ys)
                elif compatible and (not gated or supported):
                    ps = model["predictions"]
                else:
                    ps = [None]*len(ys)
                ss = [metrics(p, y, x) for p, y, x in zip(ps, ys, row["query_inputs"], strict=True)]
                counts = summaries[(row["split"], policy)]
                counts["tasks"] += 1
                counts["eligible_tasks"] += eligible
                counts["compatible_tasks"] += compatible
                counts["cv_supported_tasks"] += supported
                complete = all(s["complete"] for s in ss)
                exact = all(s["exact"] for s in ss)
                counts["complete_tasks"] += complete
                counts["correct_tasks"] += exact
                counts["wrong_complete_tasks"] += complete and not exact
                counts["tasks_with_wrong_known_cells"] += any(s["wrong"] for s in ss)
                counts["test_grids"] += len(ss)
                for s in ss:
                    for metric in ("complete", "exact", "cells", "known", "correct", "wrong",
                                   "changed", "changed_known", "changed_correct", "shape_changed"):
                        if s[metric] is not None:
                            counts[metric] += s[metric]
                task_score["policies"][policy] = {"radius": model["radius"] if model else None,
                    "compatible": compatible, "cv_supported": supported,
                    "complete": complete, "correct": exact, "grids": ss}
        task_scores.append(task_score)
    summary = {split: {policy: dict(value) for (sp, policy), value in sorted(summaries.items()) if sp == split}
               for split in ("training", "evaluation")}
    write_json(out / "summary.json", summary)
    write_json(out / "scores.json", task_scores)
    write_json(out / "score-run.json", {
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "answers_sha256": hashlib.sha256(answers_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest(),
    })
    print("split\tpolicy\teligible\tfit\tcv_pass\tcomplete\tcorrect\twrong_complete")
    for split, policies in summary.items():
        for policy, counts in policies.items():
            print("\t".join(map(str, [split, policy] + [counts[k] for k in
                  ("eligible_tasks", "compatible_tasks", "cv_supported_tasks", "complete_tasks", "correct_tasks", "wrong_complete_tasks") ])))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--data", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict"); p.add_argument("--problems", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("score"); p.add_argument("--predictions", type=Path, required=True); p.add_argument("--answers", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "prepare": prepare(a.data, a.out)
    elif a.command == "predict": predict(a.problems, a.out, a.workers)
    else: score(a.predictions, a.answers, a.out)


if __name__ == "__main__":
    main()
