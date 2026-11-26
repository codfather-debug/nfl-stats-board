"""Checks that discord_post.py posts picks once per new week and results once a week is fully graded."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discord_post as d  # noqa: E402


def play(week, team, graded):
    e = {"season": 2026, "week": week, "team": team, "opp": "BUF", "player": f"{team} Back", "pos": "RB",
         "status": "Active", "market": "rush_yds", "line": 40, "p": 0.7, "rate": 0.65, "proj": 55.0,
         "anyRate": 0.55, "firstRate": 0.25, "gameRate": 0.12, "isPlay": True, "isPick": True}
    if graded:
        e.update(actual=61.0, hit=True, played=True, anyTd=True, teamFirst=True, gameFirst=False)
    return e


class Messages(unittest.TestCase):
    def test_unchanged_posts_nothing(self):
        plays = [play(11, "DET", True), play(12, "DET", False)]
        self.assertEqual(d.messages(plays, plays), [])

    def test_new_week_posts_picks_once(self):
        old = [play(11, "DET", True)]
        new = old + [play(12, "DET", False), play(12, "CHI", False)]
        out = d.messages(old, new)
        self.assertEqual(len(out), 1)
        self.assertIn("Week 12", out[0])

    def test_results_wait_for_the_last_game(self):
        old = [play(12, "DET", False), play(12, "CHI", False)]
        partly = [play(12, "DET", True), play(12, "CHI", False)]
        self.assertEqual(d.messages(old, partly), [])
        out = d.messages(partly, [play(12, "DET", True), play(12, "CHI", True)])
        self.assertEqual(len(out), 1)
        self.assertIn("results", out[0])

    def test_messages_fit_discord(self):
        new = [play(12, t, True) for t in ("DET", "CHI", "GB", "MIN") * 40]
        self.assertTrue(all(len(m) <= 2000 for m in d.messages([], new)))


if __name__ == "__main__":
    unittest.main()
