"""Synthetic controls only; no ARC labels are used in these tests."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("transport014", Path(__file__).with_name("run.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pair(grid):
    return {"input": grid, "output": [[2 if v == 1 else v for v in row] for row in grid]}


def problem():
    return {"id": "synthetic", "split": "training",
            "train": [pair([[0, 1], [1, 0]]), pair([[1, 1, 0], [0, 1, 0]])],
            "query_inputs": [[[1, 0, 1]]]}


class Controls(unittest.TestCase):
    def test_positive_transport(self):
        r = m.infer(problem())["pipelines"]["local"]
        self.assertTrue(r["compatible"])
        self.assertTrue(r["cv_exact"])
        self.assertEqual(r["radius"], 0)
        self.assertEqual(r["predictions"], [[[2, 0, 2]]])

    def test_no_information_loss_control(self):
        r = m.infer(problem())["pipelines"]["memorise"]
        self.assertTrue(r["compatible"])
        self.assertFalse(r["cv_exact"])
        self.assertEqual(r["predictions"], [[[-1, -1, -1]]])

    def test_unseen_key_abstains(self):
        p = problem(); p["query_inputs"] = [[[3]]]
        self.assertEqual(m.infer(p)["pipelines"]["local"]["predictions"], [[[-1]]])

    def test_conflict_refines_and_cv_relearns(self):
        p = problem()
        p["train"] = [{"input": [[1, 0]], "output": [[2, 0]]},
                      {"input": [[1, 3]], "output": [[4, 3]]}]
        r = m.infer(p)["pipelines"]["local"]
        self.assertEqual(r["radius"], 1)
        self.assertFalse(r["cv_exact"])
        self.assertTrue(all(f["radius"] == 0 for f in r["folds"]))
        self.assertTrue(all(any(s["wrong"] for s in f["scores"]) for f in r["folds"]))

    def test_query_answers_are_not_read(self):
        class NoOutput(dict):
            def __getitem__(self, key):
                if key == "output":
                    raise AssertionError("test answer read")
                return super().__getitem__(key)
        p = problem()
        t = {"train": p["train"], "test": [NoOutput(input=p["query_inputs"][0])]}
        self.assertEqual(m.infer(m.project(t, p["id"], p["split"])), m.infer(p))

    def test_duplicate_demos_are_held_out_together(self):
        p = problem(); p["train"].append(copy.deepcopy(p["train"][0]))
        r = m.infer(p)
        self.assertEqual(r["n_distinct_inputs"], 2)
        self.assertEqual(r["pipelines"]["local"]["folds"][0]["excluded"], [0, 2])
        self.assertFalse(r["pipelines"]["memorise"]["cv_exact"])

    def test_dimension_change_is_not_silently_accepted(self):
        p = problem(); p["train"][1]["output"] = [[2]]
        self.assertFalse(m.infer(p)["eligible"])
        s = m.metrics([[2]], [[2, 2]], [[1]])
        self.assertTrue(s["complete"])
        self.assertFalse(s["exact"])

    def test_no_default_copy(self):
        s = m.metrics([[-1, 0]], [[2, 0]], [[1, 0]])
        self.assertEqual((s["known"], s["changed_known"]), (1, 0))
        self.assertFalse(s["complete"])

    def test_known_wrong_partial_predictions_are_counted(self):
        s = m.metrics([[3, -1]], [[2, 0]], [[1, 0]])
        self.assertEqual((s["wrong"], s["changed_correct"]), (1, 0))
        self.assertFalse(s["complete"])


if __name__ == "__main__":
    unittest.main()
