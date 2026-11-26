#!/usr/bin/env python3
"""Posts the week's top plays and, once graded, its results to a Discord channel.

Compares results/history.json with the version at an earlier commit (the one the
refresh started from), so it only posts when something new was published:
  - plays, the first time a week's plays are recorded
  - results, when the last of a week's plays is graded

    python3 discord_post.py                 # compare with HEAD, post if DISCORD_WEBHOOK_URL is set
    python3 discord_post.py --since <sha>   # compare with that commit
    python3 discord_post.py --dry-run       # print the messages instead of posting
    python3 discord_post.py --test          # post the latest week's picks, marked as a test
    python3 discord_post.py --preview 11    # print week 11's plays and results messages

Standard library only. Posting never fails the refresh: errors are printed, not raised.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = "results/history.json"
SITE = "https://codfather-debug.github.io/nfl-stats-board/"
TOP = 10
# Keep in step with MARKETS in features.py (that file needs pandas; this one doesn't).
LABELS = {"rush_yds": "rushing yds", "carries": "rush attempts", "rec_yds": "receiving yds",
          "receptions": "receptions", "scrimmage": "rush + rec yds", "td": "anytime TD", "long_rush": "longest rush",
          "long_rec": "longest catch", "pass_yds": "passing yds", "pass_tds": "passing TDs",
          "completions": "completions"}


def pct(x):
    return f"{round(x * 100)}%"


def label(e):
    if e["market"] == "td":
        return "anytime TD"
    return f"{e['line']}+ {LABELS.get(e['market'], e['market'].replace('_', ' '))}"


def key(e):
    return e["season"], e["week"]


def week_done(plays, wk):
    mine = [e for e in plays if key(e) == wk]
    return bool(mine) and all("actual" in e for e in mine)


def plays_message(plays, wk):
    mine = [e for e in plays if key(e) == wk]
    top = sorted(mine, key=lambda e: (-e["rate"], -e["p"]))[:TOP]
    lines = [f"📊 **NFL Stat Board · Week {wk[1]} top plays**", "",
             f"Highest tested hit rates of {len(mine)} recorded plays:"]
    for i, e in enumerate(top, 1):
        proj = f", proj {e['proj']:g}" if e.get("proj") is not None and e["market"] != "td" else ""
        lines.append(f"{i}. {e['player']} ({e['team']} {e['pos']} vs {e['opp']}): {label(e)}, **{pct(e['rate'])}**{proj}")
    lines += ["", f"Plays can change with late injury news: {SITE}"]
    return "\n".join(lines)


def results_message(plays, wk):
    mine = [e for e in plays if key(e) == wk and e.get("hit") is not None]
    lines = [f"📊 **NFL Stat Board · Week {wk[1]} results**", ""]
    if not mine:
        return "\n".join(lines + ["No graded plays this week.", "", SITE])
    hits = sum(e["hit"] for e in mine)
    expected = sum(e["rate"] for e in mine) / len(mine)
    lines.append(f"**All plays:** {hits} of {len(mine)} hit ({pct(hits / len(mine))}; tested rate said {pct(expected)})")
    top = sorted(mine, key=lambda e: (-e["rate"], -e["p"]))[:TOP]
    lines.append(f"**Top {len(top)} plays:** {sum(e['hit'] for e in top)} of {len(top)} hit")
    by = collections.defaultdict(list)
    for e in mine:
        by[e["market"]].append(e["hit"])
    rows = sorted(by.items(), key=lambda kv: -len(kv[1]))
    lines += ["", "**By stat**"] + [f"• {LABELS.get(m, m)}: {sum(h)}/{len(h)} ({pct(sum(h) / len(h))})" for m, h in rows]
    void = sum(1 for e in plays if key(e) == wk and e.get("hit") is None)
    if void:
        lines.append(f"({void} void: player didn't play)")
    lines += ["", f"Full results: {SITE}"]
    return "\n".join(lines)


def messages(old, new):
    """The posts the change from old to new plays calls for, oldest week first."""
    old_weeks = {key(e) for e in old}
    out = []
    for wk in sorted({key(e) for e in new}):
        if wk not in old_weeks:
            out.append(plays_message(new, wk))
        if week_done(new, wk) and not week_done(old, wk):
            out.append(results_message(new, wk))
    return out


TEST_NOTE = "🧪 *Test post: this channel is connected. Real posts look like this.*\n\n"


def latest_picks(plays):
    return messages([], [e for e in plays if key(e) == max(key(e) for e in plays)])[0]


def load_at(commit):
    try:
        text = subprocess.run(["git", "show", f"{commit}:{HISTORY}"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout
        return json.loads(text)["plays"]
    except (subprocess.CalledProcessError, ValueError, KeyError):
        return []


def post(url, text):
    body = json.dumps({"content": text[:2000], "allowed_mentions": {"parse": []}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "User-Agent": "DiscordBot (nfl-stats-board, 1.0)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="HEAD")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", type=int)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(HERE, HISTORY), encoding="utf-8") as f:
        new = json.load(f)["plays"]
    if args.preview is not None:
        season = max(e["season"] for e in new)
        wk = (season, args.preview)
        print(plays_message(new, wk), results_message(new, wk), sep="\n\n---\n\n")
        return
    todo = messages(load_at(args.since), new)
    if args.test:
        todo = [TEST_NOTE + latest_picks(new)]
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not todo:
        print("Discord: nothing new to post.")
    for text in todo:
        if args.dry_run or not url:
            print(("" if args.dry_run else "Discord: no DISCORD_WEBHOOK_URL, would post:\n") + text + "\n")
            continue
        try:
            print(f"Discord: posted ({post(url, text)}).")
        except Exception as e:  # A Discord outage must not stop the board from publishing.
            print(f"Discord: could not post: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
