import dataclasses
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_r34_causal_chunks as m
import summarize_r34_causal_chunks as stats


class CausalChunkTest(unittest.TestCase):
    def fixture(self):
        tasks = {"T": dict(priority=1, atoms=[dict(volume=900)])}
        jobs = [dict(id="J", task="T", sat="S", ready=0, deadline=1800,
                     priority=1, volume=900)]
        c = m.PublicContact("A", "S", "G", 0, 1800, 1.0, (1.0,) * 12, (1.0,) * 12)
        return tasks, jobs, c

    def test_outcome_not_in_policy_view(self):
        fields = {f.name for f in dataclasses.fields(m.PublicContact)}
        self.assertNotIn("onset", fields)
        self.assertNotIn("event", fields)
        self.assertNotIn("target", fields)

    def test_no_failure_preserves_bytes_and_deadline(self):
        t, j, c = self.fixture()
        r = m.simulate(t, j, {}, [c], {"A": float("inf")}, {"S": 1000}, "edf", overhead=0)
        self.assertEqual(r["priority"], 1)
        self.assertAlmostEqual(r["goodput"], 1)

    def test_splitting_does_not_reset_outcome(self):
        t, j, c = self.fixture()
        r = m.simulate(t, j, {}, [c], {"A": 300}, {"S": 1000}, "edf_5min", overhead=0)
        self.assertAlmostEqual(r["goodput"], 1 / 3)
        self.assertEqual(r["priority"], 0)

    def test_future_outcome_cannot_change_prior_actions(self):
        t, j, c = self.fixture()
        a = m.simulate(t, j, {}, [c], {"A": 900}, {"S": 1000}, "isotonic", overhead=0, capture_actions=True)
        b = m.simulate(t, j, {}, [c], {"A": 1200}, {"S": 1000}, "isotonic", overhead=0, capture_actions=True)
        self.assertEqual([x for x in a["actions"] if x[0] < 900],
                         [x for x in b["actions"] if x[0] < 900])

    def test_detection_delay_and_switch(self):
        t, j, c = self.fixture()
        # First transfer stops at onset 300 but is not aborted until 900.
        other = dataclasses.replace(c, key="B", gs="H", start=900, end=1800)
        r = m.simulate(t, j, {}, [c, other], {"A": 300, "B": float("inf")},
                       {"S": 1000}, "edf", overhead=0, capture_actions=True)
        recovered = [a for a in r["actions"] if a[2] == "B"]
        self.assertTrue(recovered)
        self.assertTrue(all(a[0] >= 910 for a in recovered))

    def test_inference_sanity(self):
        self.assertEqual(stats.exact_signflip([0, 0]), 1.0)
        self.assertAlmostEqual(stats.exact_signflip([1, 1, 1, 1]), 0.125)
        self.assertEqual(list(stats.holm([0.01, 0.04, 0.2])), [0.03, 0.08, 0.2])

    def test_storage_admission(self):
        t, j, c = self.fixture()
        r = m.simulate(t, j, {}, [c], {"A": float("inf")}, {"S": 100}, "edf")
        self.assertEqual(r["storage_rejections"], 1)
        self.assertEqual(r["chunks"], 0)

    def test_fixed_resources_are_not_overwritten(self):
        t, j, c = self.fixture()
        r = m.simulate(t, j, {"sat:S": [(0, 1200)]}, [c], {"A": float("inf")},
                       {"S": 1000}, "edf", overhead=0, capture_actions=True)
        self.assertTrue(all(a[0] >= 1200 for a in r["actions"]))


if __name__ == "__main__":
    unittest.main()
