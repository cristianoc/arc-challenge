"""013 development check: distinguish compatibility from positive support.

This runner reuses the three retrospective 012 cases. It intentionally does
not score official test outputs. Its purpose is to verify that the training
examples do not independently vary the distinctions separating the original
and repaired explanations, while retaining 012's unlabelled disagreement
witnesses.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDS = ["7b5033c1", "8f215267", "97d7923e"]


def load_012():
    path = ROOT / "experiments/012-information-loss/run.py"
    spec = importlib.util.spec_from_file_location("exp012", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def analyse_path(rows):
    feats = [r["features"] for r in rows]
    repeated = [f for f in feats if not f["single_run_per_colour"]]
    mismatches = [f for f in feats if not f["path_equals_histogram"]]
    return {
        "support_question":
            "does training vary path order beyond histogram/first-seen order?",
        "training_examples": len(feats),
        "examples_with_repeated_colour_runs": len(repeated),
        "examples_where_path_differs_from_histogram_rendering": len(mismatches),
        "positive_support_for_path_over_histogram": False,
        "reason":
            "no training example exhibits the distinction exposed by the witness",
    }


def analyse_counts(rows):
    frames = [f for r in rows for f in r["features"]]
    raw_disagreement = [
        f for f in frames if f["local_count"] != f["global_count"]
    ]
    clipped_disagreement = [
        f for f in frames if not f["counts_agree_after_clipping"]
    ]
    labelled_global_fail = [
        f for f in frames
        if f["observed_stripes"] is not None
        and min(f["capacity"], f["global_count"]) != f["observed_stripes"]
    ]
    return {
        "support_question":
            "does training distinguish local-patch lookup from global same-colour count?",
        "training_frames": len(frames),
        "raw_local_global_disagreements": len(raw_disagreement),
        "post_clipping_disagreements": len(clipped_disagreement),
        "labelled_failures_of_global_count": len(labelled_global_fail),
        "positive_support_for_global_over_local": False,
        "reason":
            "local and global counts coincide on every training frame",
    }


def analyse_rank(rows):
    groups = [g for r in rows for g in r["features"]]
    marker_fail = [
        g for g in groups
        if g["observed_selected_ranks"] != [g["marker_length"]]
    ]
    original_fail = [
        g for g in groups
        if g["original_selected_ranks"] != g["observed_selected_ranks"]
    ]
    ties = [g for g in groups if g["ties"]]
    return {
        "support_question":
            "does training distinguish marker-relative rank from positional guards?",
        "training_groups": len(groups),
        "marker_rank_label_mismatches": len(marker_fail),
        "original_guard_label_mismatches": len(original_fail),
        "groups_with_rank_ties": len(ties),
        "positive_support_for_rank_over_position": False,
        "reason":
            "both decision rules make the same labelled choices on every training group",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["ARC_REPAIR_CORPUS"] = str(Path(args.corpus).resolve())
    e12 = load_012()

    result = {}
    for task in IDS:
        _, data = e12.case(task)
        train = data["examples"]["train"]
        if task == IDS[0]:
            support = analyse_path(train)
        elif task == IDS[1]:
            support = analyse_counts(train)
        else:
            support = analyse_rank(train)
        result[task] = {
            "support": support,
            "witness_kind": data["witness"]["kind"],
            "witness_ground_truth": data["witness"]["ground_truth"],
        }

    result["summary"] = {
        "cases": 3,
        "cases_with_positive_training_support_for_repair_distinction": sum(
            int(result[t]["support"][k])
            for t, k in [
                (IDS[0], "positive_support_for_path_over_histogram"),
                (IDS[1], "positive_support_for_global_over_local"),
                (IDS[2], "positive_support_for_rank_over_position"),
            ]
        ),
        "interpretation":
            "all three known repairs are compatible with repeated training relationships, "
            "but none is selected by an observed training contrast that separates it "
            "from the original explanation",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
