#!/usr/bin/env python3
"""Rebuild index.html for the next NFL week straight from nflverse, ESPN injuries and model.json.

Usage: python3 build.py [data_dir]
data_dir is optional and holds snapsYYYY.csv, statsYYYY.csv and games.csv for offline runs.
Also records this week's plays before kickoff and grades past plays in results/history.json.
"""
import datetime
import json
import math
import os
import sys
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
SEASON = 2026
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
HISTORY_PATH = os.path.join(HERE, "results", "history.json")
UNAVAILABLE = {"Out", "Doubtful", "Injured Reserve", "Suspension"}
RECORD_MIN = 0.5  # Plays at or above this tested hit rate are recorded and graded.
EASTERN = ZoneInfo("America/New_York")
# ESPN team names end with these; keep in step with TEAMS in template.html.
TEAMS = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI",
         "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
         "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC", "Rams": "LA", "Chargers": "LAC",
         "Raiders": "LV", "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
         "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA", "49ers": "SF", "Buccaneers": "TB",
         "Titans": "TEN", "Commanders": "WAS"}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def kickoff(gameday, gametime):
    """games.csv lists Eastern time."""
    t = datetime.datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=EASTERN)
    return t.astimezone(datetime.timezone.utc)


def injuries():
    """Latest ESPN status by team|name. Keep in step with parseInjuries in template.html."""
    with urllib.request.urlopen(INJURIES_URL, timeout=60) as resp:
        data = json.loads(resp.read())
    latest = {}
    for team in data.get("injuries", []):
        abbr = next((a for n, a in TEAMS.items() if team.get("displayName", "").endswith(n)), None)
        for item in team.get("injuries", []):
            key = f"{abbr}|{F.norm_name(item['athlete']['displayName'])}"
            if key not in latest or item.get("date", "") > latest[key][0]:
                latest[key] = (item.get("date", ""), item.get("status", ""))
    return {k: v[1] for k, v in latest.items()}


def status_for(team, key, inj, teammates):
    """Exact name, else a surname that's unique on both the roster and the team's ESPN list."""
    if f"{team}|{key}" in inj:
        return inj[f"{team}|{key}"]
    sur = F.surname(key)
    listed = [k for k in inj if k.startswith(team + "|") and F.surname(k.split("|")[1]) == sur]
    if len(listed) == 1 and sum(F.surname(t) == sur for t in teammates) == 1:
        return inj[listed[0]]
    return ""


def upcoming_week(games):
    g = games[(games.season == SEASON) & (games.game_type == "REG")]
    open_weeks = g[g.result.isna()].week
    return int(open_weeks.min()) if len(open_weeks) else None


def slate(games, week):
    g = games[(games.season == SEASON) & (games.game_type == "REG") & (games.week == week)]
    out = []
    for r in g.itertuples():
        ko = kickoff(r.gameday, r.gametime)
        indoor, turf = int(r.roof in ("dome", "closed")), int(isinstance(r.surface, str) and r.surface != "grass")
        for team, opp, home in ((r.home_team, r.away_team, 1), (r.away_team, r.home_team, 0)):
            out.append({"team": team, "opp": opp, "home": home, "indoor": indoor, "turf": turf,
                        "kickoff": stamp(ko), "final": bool(pd.notna(r.result))})
    return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def linear(coef, x):
    return coef["intercept"] + sum(c * (x.get(k) or 0) for k, c in coef.items() if k != "intercept")


def predict(coef, x):
    """Keep in step with predict in template.html."""
    return sigmoid(linear(coef, x))


def project(coef, x):
    """Median projection. Keep in step with project in template.html."""
    return max(0.0, linear(coef, x))


def tier_for(tiers, p):
    return [t for t in tiers if p >= t["min"]][-1]


def roster_url(season):
    return f"{F.NFLVERSE}/weekly_rosters/roster_weekly_{season}.csv"


def candidates(pg, teams, week, games, src):
    """This week's row for every active skill player on a team on the slate.

    Uses nflverse's weekly roster (latest week published): status ACT only, so IR and practice squad
    are out, and a traded player shows with the new team. If the roster can't be read, falls back to
    players who took a snap for the team in one of its last four games.
    """
    names = pg.sort_values(["season", "week"]).groupby("pid").player.last()
    try:
        r = F.read_csv(f"{src}/roster{SEASON}.csv" if src else roster_url(SEASON))
        r = r[r.week == r.week[r.week <= week].max()]
        r = r[(r.status == "ACT") & r.team.isin(teams) & r.pfr_id.notna()].copy()
        r["pos"] = r.position.replace(F.POS_ALIASES)
        r = r[r.pos.isin(F.POSITIONS)]
        up = pd.DataFrame({"season": SEASON, "team": r.team.values, "pid": r.pfr_id.values, "pos": r.pos.values,
                           "player": [names.get(p, n) for p, n in zip(r.pfr_id, r.full_name)]})
        print(f"Roster: week {int(r.week.max())}, {len(up)} active skill players")
    except Exception as e:
        print(f"Roster unavailable ({e}); using recent snaps")
        cur = pg[pg.season == SEASON]
        recent_weeks = cur.groupby("team").week.apply(lambda w: set(sorted(set(w))[-4:])).to_dict()
        cur = cur[[w in recent_weeks.get(t, ()) for t, w in zip(cur.team, cur.week)]]
        last = cur.sort_values("week").groupby("pid").tail(1)
        up = last[last.team.isin(teams)][["season", "team", "pid", "player", "pos"]].copy()
    up["key"] = up.player.map(F.norm_name)
    up = up.drop_duplicates("pid").assign(week=week)
    lines = F.team_lines(games, SEASON)
    return up.merge(lines[["week", "team", "opp", "home", "indoor", "turf", "game_id"]], how="left", on=["week", "team"])


def ruled_out(pg, up, week):
    """Regulars (a snap in one of the team's last two games) who won't play: off the active roster or
    listed Out, Doubtful, IR or suspended. Feeds the teammates-out inputs."""
    cur = pg[pg.season == SEASON]
    active = {p for p, s in zip(up.pid, up.status) if s not in UNAVAILABLE}
    out = set()
    for team in set(up.team):
        weeks = sorted(set(cur[cur.team == team].week))[-2:]
        for p in set(cur[(cur.team == team) & cur.week.isin(weeks)].pid):
            if p not in active:
                out.add((SEASON, week, team, p))
    return out


# Shown on the page but not model inputs (or inputs some models skip).
SHOWN = ["gp", "prev_gp", "missed", "snap_g", "snap_l3", "targets_sh", "targets_sh3", "carries_sh", "carries_sh3", "rz_opp_g", "i10_opp_g",
         "rz_opp_sh", "vac_carries", "vac_targets"]
# Each player's game log, for hit history. Keep in step with LOG in template.html.
LOG = ["week", "opp", "home", "indoor", "turf", "snap", "carries", "rushing_yards", "long_rush", "targets", "receptions", "receiving_yards",
       "long_rec", "rr_tds", "rz_opp", "completions", "attempts", "passing_yards", "passing_tds"]


def game_logs(pg):
    out = {}
    for r in pg.sort_values("week").to_dict("records"):
        out.setdefault(r["pid"], []).append([r[c] if isinstance(r[c], str) else round(float(r[c]), 2) for c in LOG])
    return out


def tier_splits(played):
    """Per player and stat: [average, games] against soft, mid and tough defenses (by rank going in)."""
    out = {}
    for c in F.SOS_STATS:
        g = played[played["tier_" + c] != ""].groupby(["pid", "tier_" + c])[c].agg(["mean", "size"])
        for (pid, tier), r in g.iterrows():
            out.setdefault(pid, {}).setdefault(c, {})[tier] = [round(float(r["mean"]), 1), int(r["size"])]
    return out


def score(rows, model, slate_by_team, logs, tiers):
    """One record per player on the slate with inputs, probabilities, tested rates and game log."""
    inputs = sorted({f for m in model["markets"].values() for f in m["inputs"]} | set(SHOWN) |
                    {f"prev_{m['stat']}" for m in model["markets"].values()} |
                    {f"{k}_{c}" for k in ("sos", "adj") for c in F.SOS_STATS} |
                    {f"{m['stat']}_g" for m in model["markets"].values()} | {f"{m['stat']}_l3" for m in model["markets"].values()})
    players = []
    for r in rows.itertuples(index=False):
        x = {k: (None if pd.isna(getattr(r, k)) else round(float(getattr(r, k)), 4)) for k in inputs}
        game = slate_by_team[r.team]
        p = {"player": r.player, "key": r.key, "team": r.team, "opp": game["opp"], "pos": r.pos, "x": x, "plays": {},
             "regular": bool(r.recent), "status": r.status, "log": logs.get(r.pid, []), "proj": {},
             "tiers": tiers.get(r.pid, {}), "tierNow": {c: getattr(r, "tier_" + c) for c in F.SOS_STATS}}
        for mk, m in model["markets"].items():
            if r.pos not in m["pos"] or (r.gp < model["minGames"] and r.prev_gp < model["minPrevGames"]):
                continue
            if "proj" in m:
                p["proj"][mk] = round(project(m["proj"]["coef"], x), 1)
            for line, spec in m["lines"].items():
                prob = predict(spec["coef"], x)
                p["plays"][f"{mk}|{line}"] = {"p": round(prob, 4), "rate": tier_for(spec["tiers"], prob)["hit"]}
        if p["plays"]:
            players.append(p)
    return players


def defenses(pg, week):
    """Allowed per game to each position before this week, with league average and rank (1 = allows most)."""
    t = F.defense_table(pg, SEASON, week)
    a = F.allowed_by_game(pg)
    a = a[(a.season == SEASON) & (a.week < week)]
    # Share of all targets against each defense that went to each position.
    tg = a.groupby(["defense", "pos"]).targets.sum()
    tshare = tg / tg.groupby(level="defense").transform("sum").replace(0, float("nan"))
    league = a.groupby("pos").targets.sum() / a.targets.sum()
    trank = tshare.groupby(level="pos").rank(ascending=False, method="min")
    venue = a.groupby(["defense", "pos", "dhome"])[F.ALLOWED].mean()
    venue_n = a.groupby(["defense", "pos", "dhome"]).size()
    out = []
    for r in t.to_dict("records"):
        d, p = r["defense"], r["pos"]
        row = {"team": d, "pos": p, "games": int(r["games"]),
               "gamesHome": int(venue_n.get((d, p, 1), 0)), "gamesRoad": int(venue_n.get((d, p, 0), 0))}
        for c in F.ALLOWED:
            at = lambda h: round(float(venue.loc[(d, p, h), c]), 1) if (d, p, h) in venue.index else None
            # [per game, league avg, rank, at home, on the road]
            row[c] = [round(r[c], 1), round(r[c + "_lg"], 1), int(r[c + "_rank"]), at(1), at(0)]
        if (d, p) in tshare.index and tshare[(d, p)] == tshare[(d, p)]:
            # [share, league share, rank (1 = most)]
            row["tshare"] = [round(float(tshare[(d, p)]), 3), round(float(league[p]), 3), int(trank[(d, p)])]
        out.append(row)
    return out


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"plays": []}


def snapshot(history, players, slate_by_team, week, now):
    """Replace recorded plays for games that haven't kicked off. Started games stay locked."""
    open_teams = {t for t, g in slate_by_team.items()
                  if not g["final"] and now < datetime.datetime.fromisoformat(g["kickoff"].replace("Z", "+00:00"))}
    history["plays"] = [e for e in history["plays"] if not (e["season"] == SEASON and e["week"] == week and e["team"] in open_teams)]
    locked = stamp(now)
    for p in players:
        if p["team"] not in open_teams or p["status"] in UNAVAILABLE:
            continue
        for k, v in p["plays"].items():
            if v["rate"] >= RECORD_MIN:
                mk, line = k.split("|")
                history["plays"].append({"season": SEASON, "week": week, "team": p["team"], "opp": p["opp"],
                                         "player": p["player"], "pos": p["pos"], "status": p["status"], "market": mk,
                                         "line": int(line), "p": v["p"], "rate": v["rate"], "proj": p["proj"].get(mk),
                                         "lockedAt": locked})


def grade(history, pg, games, model):
    """Fill in the stat once the game is final and its box score is published. Did-not-play is void."""
    finals = {(r.season, r.week, t) for r in games[games.result.notna()].itertuples() for t in (r.home_team, r.away_team)}
    published = set(zip(pg.season, pg.week, pg.team))
    box = {k: r for k, r in zip(zip(pg.season, pg.week, pg.team, pg.key), pg.to_dict("records"))}
    for e in history["plays"]:
        game = (e["season"], e["week"], e["team"])
        if "actual" in e or game not in finals or game not in published:
            continue
        r = box.get((*game, F.norm_name(e["player"])))
        if r:
            e["actual"] = float(r[model["markets"][e["market"]]["stat"]])
            e["hit"] = e["actual"] >= e["line"]
        else:
            e["actual"], e["hit"] = None, None  # Did not take an offensive snap: void.


def main(src):
    now = now_utc()
    games = F.read_csv(f"{src}/games.csv" if src else F.GAMES_URL)
    with open(os.path.join(HERE, "model.json"), encoding="utf-8") as f:
        model = json.load(f)
    week = upcoming_week(games)
    # Last season too: it feeds the carried-over averages.
    pg = pd.concat([F.load_season(s, games, src) for s in (SEASON - 1, SEASON)], ignore_index=True)
    history = load_history()
    data = {"season": SEASON, "week": week, "builtAt": stamp(now), "model": model, "slate": [], "players": []}
    if week:
        games_week = slate(games, week)
        slate_by_team = {g["team"]: g for g in games_week}
        up = candidates(pg, set(slate_by_team), week, games, src)
        try:
            inj = injuries()
            data["injuriesAt"] = stamp(now)
        except Exception as e:
            print(f"Injury data unavailable: {e}")
            inj, data["injuriesAt"] = {}, ""
        roster = up.groupby("team").key.apply(list).to_dict()
        up["status"] = [status_for(t, k, inj, roster[t]) for t, k in zip(up.team, up.key)]
        # Ruled-out teammates free up carries and targets for the rest.
        rows = F.add_features(pg, up, ruled_out(pg, up, week))
        tiers = tier_splits(rows[rows.played & (rows.season == SEASON)])
        rows = rows[~rows.played & (rows.season == SEASON) & (rows.week == week)]
        players = score(rows, model, slate_by_team, game_logs(pg[pg.season == SEASON]), tiers)
        if len(players) < 50:
            sys.exit(f"Only {len(players)} players scored for week {week}; leaving the last page live.")
        data["slate"], data["players"] = games_week, players
        data["defenses"] = defenses(pg[pg.season == SEASON], week)
        data["logCols"] = LOG
        if data["injuriesAt"]:
            snapshot(history, players, slate_by_team, week, now)
    try:
        grade(history, pg, games, model)
    except Exception as e:
        print(f"Could not grade results: {e}")
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, separators=(",", ":"), ensure_ascii=False)
    data["results"] = history["plays"]
    # Inputs go out as arrays against one key list, which keeps the page about half the size.
    data["xCols"] = sorted({k for p in data["players"] for k in p["x"]})
    for p in data["players"]:
        p["x"] = [None if p["x"].get(k) is None else round(p["x"][k], 3) for k in data["xCols"]]
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n')
        f.write(html.replace("/*__DATA__*/null", payload))
    graded = sum(1 for e in history["plays"] if e.get("hit") is not None)
    print(f"Week {week}: {len(data['players'])} players, {len(history['plays'])} recorded plays, {graded} graded.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
