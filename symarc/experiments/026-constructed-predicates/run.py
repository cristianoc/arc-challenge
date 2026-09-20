"""Construct symbolic predicates using operation-relative class consistency.

Separate prepare, construct and score commands. No query labels or query inputs
are used to synthesize or choose a predicate. Standard library only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import language as L


def write(p: Path, x) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, sort_keys=True, separators=(",", ":"))+"\n")


def read(p: Path): return json.loads(p.read_text())
def sha(p: Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def direct(xs, labels, candidates):
    """Reference enumeration of if/constant/constant programs, all minima."""
    palette = sorted(set(labels)); result = []; metrics = Counter()
    for pred in candidates:
        if result and L.cost(pred) > result[0]["predicate_cost"]: break
        sig = [L.evaluate(pred, x) for x in xs]
        metrics["predicate_observation_evaluations"] += len(xs)
        metrics["predicates_considered"] += 1
        for false, true in itertools.product(palette, repeat=2):
            metrics["branch_assignments_considered"] += 1
            fits = True
            for b, y in zip(sig, labels, strict=True):
                metrics["label_checks"] += 1
                if (true if b else false) != y: fits = False; break
            if fits:
                result.append({"predicate": pred, "false": false, "true": true,
                               "predicate_cost": L.cost(pred), "expanded_program_cost": L.cost(pred)+3})
    return result, dict(metrics)


def construct(xs, labels, candidates):
    """For each constructed predicate, solve its class/action intersections."""
    palette = sorted(set(labels)); result = []; rejected = []; metrics = Counter()
    for pred in candidates:
        if result and L.cost(pred) > result[0]["predicate_cost"]: break
        sig = [L.evaluate(pred, x) for x in xs]
        metrics["predicate_observation_evaluations"] += len(xs)
        metrics["predicates_considered"] += 1
        classes = {}; conflict = None
        for i, (b, y) in enumerate(zip(sig, labels, strict=True)):
            metrics["label_checks"] += 1
            if b in classes and classes[b][0] != y:
                conflict = [classes[b][1], i]; break
            classes.setdefault(b, (y, i))
        if conflict is not None:
            rejected.append({"predicate": pred, "observations": conflict})
            continue
        options = [[classes[b][0]] if b in classes else palette for b in (False, True)]
        for false, true in itertools.product(*options):
            metrics["branch_assignments_constructed"] += 1
            result.append({"predicate": pred, "false": false, "true": true,
                           "predicate_cost": L.cost(pred), "expanded_program_cost": L.cost(pred)+3})
    return result, dict(metrics), rejected


def teacher_rectangle(points):
    # A row/interval interpreter, deliberately not the count * span formula.
    rows = {}
    for r, c in points: rows.setdefault(r, set()).add(c)
    if not rows: return False
    rs = sorted(rows)
    if rs != list(range(rs[0], rs[-1]+1)): return False
    first = sorted(rows[rs[0]])
    if first != list(range(first[0], first[-1]+1)): return False
    return all(rows[r] == set(first) for r in rs)


def teacher_frame(points):
    ps = set(map(tuple, points)); rs = [r for r,c in ps]; cs = [c for r,c in ps]
    lo, hi, left, right = min(rs), max(rs), min(cs), max(cs)
    expected = {(r,c) for r in range(lo,hi+1) for c in range(left,right+1)
                if r in (lo,hi) or c in (left,right)}
    return ps == expected


def shape(h, w, mask):
    return [[i//w, i%w] for i in range(h*w) if mask & (1 << i)]


def prepare(out: Path):
    train = [{"points": shape(3,3,m), "label": int(teacher_rectangle(shape(3,3,m)))} for m in range(1,512)]
    primary = [shape(4,4,m) for m in range(1,65536)]
    rng = random.Random(2601); shapes = set()
    for h,w in ((5,7),(7,5),(8,8),(11,13)):
        full = {(r,c) for r in range(h) for c in range(w)}
        masks = [full] + [full-{p} for p in sorted(full)]
        for _ in range(64):
            s = {p for p in sorted(full) if rng.randrange(2)}
            if s: masks.append(s)
        for s in masks:
            for dr,dc in ((0,0),(-9,4),(12,-6)):
                shifted = {(r+dr,c+dc) for r,c in s}
                shapes.add(tuple(sorted(shifted)))
                shapes.add(tuple(sorted((c,r) for r,c in shifted)))
    secondary = [[list(p) for p in s] for s in sorted(shapes)]
    positive = shape(2,3,63); negative = [[0,0],[0,2],[1,0],[1,2]]
    transfer = [{"id":i, "train":[{"points":positive,"label":a}, {"points":negative,"label":b}]}
                for i,(a,b) in enumerate(((2,7),(8,4),(6,3),(9,1)))]
    problem = {"train": train, "queries": {"primary": primary,"secondary": secondary}, "transfer": transfer}
    answers = {bank: [int(teacher_rectangle(s)) for s in ss] for bank,ss in problem["queries"].items()}
    write(out/"problem.json", problem); write(out/"answers.json", answers)
    write(out/"frame-control.json", [{"points":e["points"], "label":int(teacher_frame(e["points"]))} for e in train])
    write(out/"data-manifest.json", {"seed":2601,"training":len(train),"query_banks":{b:len(v) for b,v in answers.items()},
          "problem_sha256":sha(out/"problem.json"),"answers_sha256":sha(out/"answers.json")})


def vocabulary_conflict(examples):
    seen = {}
    for i,e in enumerate(examples):
        key = L.observations(e["points"])
        if key in seen and seen[key][0] != e["label"]:
            return {"observations":[seen[key][1],i],"scalar_values":list(key)}
        seen.setdefault(key,(e["label"],i))
    return None


def sorted_programs(models):
    return sorted(models, key=lambda m: (m["expanded_program_cost"],L.code(m["predicate"]),m["false"],m["true"]))


def main_construct(problem_path: Path, frame_path: Path, out: Path):
    problem = read(problem_path)
    xs = [L.observations(e["points"]) for e in problem["train"]]; labels = [e["label"] for e in problem["train"]]
    # Only labelled construction data enter these calls.
    t0 = time.perf_counter(); candidates = L.grammar(5)
    cold, cm = direct(xs,labels,candidates)
    learnt, lm, rejections = construct(xs,labels,candidates)
    assert sorted_programs(cold) == sorted_programs(learnt)
    if not learnt: raise RuntimeError("No abstraction constructed within the frozen bound")
    selected = sorted_programs(learnt)[0]
    predicates = sorted({m["predicate"] for m in learnt},key=L.code)
    library = {"body": selected["predicate"], "body_cost":L.cost(selected["predicate"]),
               "program": selected, "all_minimal_programs":sorted_programs(learnt),
               "source_inputs_sha256":digest(problem["train"])}
    # Freeze the declaration before accessing any transfer examples or query inputs.
    write(out/"library.json",library)
    transfer_results = []
    for task in problem["transfer"]:
        tx = [L.observations(e["points"]) for e in task["train"]]; ty = [e["label"] for e in task["train"]]
        # Same source evidence for cold direct resynthesis; no withheld labels.
        again, dm = direct(xs,labels,candidates)
        candidate_bodies = sorted({m["predicate"] for m in again},key=lambda p:(L.cost(p),L.code(p)))
        target_cold, tdm = direct(tx,ty,candidate_bodies)
        target_reuse, trm, _ = construct(tx,ty,sorted(predicates,key=lambda p:(L.cost(p),L.code(p))))
        assert sorted_programs(target_cold) == sorted_programs(target_reuse)
        transfer_results.append({"id":task["id"], "program":sorted_programs(target_reuse)[0],
           "all_programs":sorted_programs(target_reuse),"cold_source_work":dm,"cold_target_work":tdm,"reuse_target_work":trm})
    # Aliases are presentation changes. The macro body is expanded before costing.
    aliases = {"N":("count",),"N_again":("count",),"R":("span_r",),"C":("span_c",),
               "constructed_body":selected["predicate"],"nested_alias":("alias","constructed_body")}
    alternate = L.grammar(5,aliases)
    assert candidates == alternate
    alt, am, _ = construct(xs,labels,alternate)
    assert sorted_programs(alt) == sorted_programs(learnt) and am == lm
    negative = read(frame_path); collision = vocabulary_conflict(negative)
    predictions = {}
    for bank,points in problem["queries"].items():
        observations = [L.observations(s) for s in points]
        predictions[bank] = {"source":[selected["true"] if L.evaluate(selected["predicate"],x) else selected["false"] for x in observations],
            "all_minimal":[[m["true"] if L.evaluate(m["predicate"],x) else m["false"] for x in observations] for m in sorted_programs(learnt)],
            "transfer":[[r["program"]["true"] if L.evaluate(r["program"]["predicate"],x) else r["program"]["false"] for x in observations] for r in transfer_results]}
    write(out/"predictions.json",predictions)
    result = {"candidate_predicates":len(candidates),"candidate_hash":digest(candidates),
              "selected":selected,"selected_readable":L.render(selected["predicate"]),
              "minimum_programs":sorted_programs(learnt),"direct_work":cm,"constructive_work":lm,
              "rejections":rejections,"transfers":transfer_results,"alias_invariance":True,
              "frame_vocabulary_conflict":collision}
    write(out/"construction.json",result)
    write(out/"run.json",{"seconds":time.perf_counter()-t0,"source_sha256":sha(Path(__file__)),
       "language_sha256":sha(Path(L.__file__)),"problem_sha256":sha(problem_path),
       "library_sha256":sha(out/"library.json"),"predictions_sha256":sha(out/"predictions.json"),
       "query_inputs_used_in_synthesis":False,"workers":1,"timing_comparison":False})
    print(json.dumps({k:v for k,v in result.items() if k not in ("rejections","transfers")},indent=2))


def score(out: Path, answers_path: Path):
    assert sha(out/"predictions.json") == read(out/"run.json")["predictions_sha256"]
    predictions = read(out/"predictions.json"); answers = read(answers_path); result = {}
    pairs = ((2,7),(8,4),(6,3),(9,1))
    for bank,y in answers.items():
        p = predictions[bank]
        result[bank] = {"examples":len(y),"correct":sum(a==b for a,b in zip(y,p["source"],strict=True)),
                       "all_minimal_correct":[sum(a==b for a,b in zip(y,z,strict=True)) for z in p["all_minimal"]],
                       "transfer_correct":[sum(pred==(pairs[i][0] if label else pairs[i][1]) for pred,label in zip(z,y,strict=True)) for i,z in enumerate(p["transfer"])]}
    write(out/"scores.json",result)
    write(out/"score-run.json",{"answers_sha256":sha(answers_path),"predictions_sha256":sha(out/"predictions.json"),"scores_sha256":sha(out/"scores.json")})
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest="cmd",required=True)
    q=s.add_parser("prepare");q.add_argument("--out",type=Path,required=True)
    q=s.add_parser("construct");q.add_argument("--problem",type=Path,required=True);q.add_argument("--frame",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q=s.add_parser("score");q.add_argument("--out",type=Path,required=True);q.add_argument("--answers",type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="prepare":prepare(a.out)
    elif a.cmd=="construct":main_construct(a.problem,a.frame,a.out)
    else:score(a.out,a.answers)

if __name__=="__main__":main()
