"""Checks for the pieces that are easy to break without noticing. No network needed.

Run: .venv/bin/python -m unittest discover tests
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import build  # noqa: E402
import features as F  # noqa: E402
import train  # noqa: E402


class Names(unittest.TestCase):
    def test_norm_name(self):
        self.assertEqual(F.norm_name("Kenneth Walker III"), "kenneth walker")
        self.assertEqual(F.norm_name("Ja'Marr Chase"), "jamarr chase")
        self.assertEqual(F.norm_name("Amon-Ra St. Brown"), "amonra st brown")
        self.assertEqual(F.norm_name("José  Núñez Jr."), "jose nunez")

    def test_status_for(self):
        inj = {"DET|jahmyr gibbs": "Questionable", "DET|a st brown": "Out"}
        self.assertEqual(build.status_for("DET", "jahmyr gibbs", inj, ["jahmyr gibbs"]), "Questionable")
        # A surname stands in only when it's unique on both lists.
        self.assertEqual(build.status_for("DET", "amonra st brown", inj, ["amonra st brown"]), "Out")
        self.assertEqual(build.status_for("DET", "amonra st brown", inj, ["amonra st brown", "equanimeous brown"]), "")
        self.assertEqual(build.status_for("NE", "jahmyr gibbs", inj, ["jahmyr gibbs"]), "")


class Kickoff(unittest.TestCase):
    def test_eastern_time_with_and_without_daylight_saving(self):
        self.assertEqual(build.stamp(build.kickoff("2026-09-13", "13:00")), "2026-09-13T17:00:00Z")
        self.assertEqual(build.stamp(build.kickoff("2026-11-15", "13:00")), "2026-11-15T18:00:00Z")


class Rates(unittest.TestCase):
    TIERS = [{"min": 0, "avgP": .1, "hit": .08}, {"min": .5, "avgP": .6, "hit": .58}, {"min": .8, "avgP": .9, "hit": .82}]

    def test_rate_is_flat_past_the_ends_and_a_line_between(self):
        self.assertEqual(build.rate_for(self.TIERS, 0.01), .08)
        self.assertEqual(build.rate_for(self.TIERS, 0.99), .82)
        self.assertAlmostEqual(build.rate_for(self.TIERS, 0.35), .33)
        self.assertAlmostEqual(build.rate_for(self.TIERS, 0.75), .70)

    def test_rate_never_falls_as_the_score_rises(self):
        rates = [build.rate_for(self.TIERS, p / 100) for p in range(101)]
        self.assertEqual(rates, sorted(rates))

    def test_tier_for(self):
        self.assertEqual(build.tier_for(self.TIERS, 0.79)["min"], .5)
        self.assertEqual(build.tier_for(self.TIERS[1:], 0.2)["min"], .5)  # Under every tier: the lowest one.

    def test_predict_matches_the_logistic_formula(self):
        coef = {"intercept": -1.0, "a": 0.5, "b": -0.25}
        self.assertAlmostEqual(build.predict(coef, {"a": 4, "b": None}), 1 / (1 + np.exp(-1.0)))


class Tiers(unittest.TestCase):
    def test_tiers_are_big_enough_and_never_run_backwards(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 6000)
        y = (rng.uniform(0, 1, 6000) < p * .9).astype(int)
        tiers = train.tiers(p, y, np.full(6000, 2025))
        self.assertEqual(sum(t["n"] for t in tiers), 6000)
        self.assertTrue(all(t["n"] >= train.MIN_TIER for t in tiers))
        hits = [t["hit"] for t in tiers]
        self.assertEqual(hits, sorted(hits))
        self.assertTrue(all(t["lo"] <= t["hit"] <= t["hi"] for t in tiers))

    def test_wilson(self):
        lo, hi = train.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=3)
        self.assertAlmostEqual(hi, 0.596, places=3)
        self.assertEqual(train.wilson(0, 10)[0], 0)


class Model(unittest.TestCase):
    """model.json has to match features.py, or the build would quietly score with missing inputs."""

    def setUp(self):
        with open(os.path.join(os.path.dirname(HERE), "model.json"), encoding="utf-8") as f:
            self.model = json.load(f)

    def test_inputs_match_features(self):
        self.assertEqual(build.stale_inputs(self.model), [])

    def test_every_milestone_has_usable_tiers(self):
        for key, m in F.MARKETS.items():
            for line in m["lines"]:
                tiers = self.model["markets"][key]["lines"][str(line)]["tiers"]
                self.assertEqual(tiers[0]["min"], 0, (key, line))
                avg = [t["avgP"] for t in tiers]
                self.assertEqual(avg, sorted(set(avg)), (key, line))


if __name__ == "__main__":
    unittest.main()
