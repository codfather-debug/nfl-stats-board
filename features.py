"""Pre-game player features from nflverse. Shared by train.py and build.py so the two never drift.

Every feature for a game uses only that player's earlier games in the same season, plus the
game's betting line. A row with no outcome (an upcoming game) goes through the same code.
"""
import io
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
POSITIONS = ("QB", "RB", "WR", "TE")
# Outcome columns taken from the weekly stats file.
STATS = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards", "rushing_tds",
         "receptions", "targets", "receiving_yards", "receiving_tds"]
# Per-game usage taken from play-by-play: red zone (inside the 20) and goal line (inside the 10)
# carries plus targets, and the longest rush and catch.
PBP = ["rz_opp", "i10_opp", "long_rush", "long_rec"]
PBP_COLS = ["season_type", "week", "posteam", "play_type", "rusher_player_id", "receiver_player_id",
            "yards_gained", "complete_pass", "yardline_100", "two_point_attempt"]


def snaps_url(season):
    return f"{NFLVERSE}/snap_counts/snap_counts_{season}.csv"


def stats_url(season):
    return f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv"


def pbp_url(season):
    return f"{NFLVERSE}/pbp/play_by_play_{season}.csv.gz"


def read_csv(src, **kw):
    compression = "gzip" if src.endswith(".gz") else "infer"
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=300) as resp:
            src = io.BytesIO(resp.read())
    return pd.read_csv(src, low_memory=False, compression=compression, **kw)


def pbp_usage(pbp):
    """Red zone and goal line opportunities and longest plays per player per game, keyed by GSIS id."""
    p = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["run", "pass"]) & (pbp.two_point_attempt != 1)].copy()
    p["pid"] = np.where(p.play_type == "run", p.rusher_player_id, p.receiver_player_id)
    p = p[p.pid.notna()]
    p["rz_opp"] = (p.yardline_100 <= 20).astype(int)
    p["i10_opp"] = (p.yardline_100 <= 10).astype(int)
    p["long_rush"] = np.where(p.play_type == "run", p.yards_gained, 0)
    p["long_rec"] = np.where((p.play_type == "pass") & (p.complete_pass == 1), p.yards_gained, 0)
    out = p.groupby(["week", "posteam", "pid"]).agg(rz_opp=("rz_opp", "sum"), i10_opp=("i10_opp", "sum"),
                                                    long_rush=("long_rush", "max"), long_rec=("long_rec", "max"))
    return out.reset_index().rename(columns={"posteam": "team", "pid": "player_id"})


def norm_name(name):
    """Lowercase, no accents, punctuation or suffix. Keep in step with normName in template.html."""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c)).lower()
    name = re.sub(r"[.'’\-]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", name.strip())
    return re.sub(r"\s+", " ", name)


def surname(key):
    return key.split()[-1] if key else ""


def team_lines(games, season):
    """One row per team per game: implied points, margin (positive = favored), kickoff, final."""
    g = games[(games.season == season) & (games.game_type == "REG")]
    rows = []
    for r in g.itertuples():
        spread = r.spread_line if pd.notna(r.spread_line) else np.nan
        total = r.total_line if pd.notna(r.total_line) else np.nan
        base = {"season": season, "week": int(r.week), "game_id": r.game_id, "total": total,
                "kickoff": f"{r.gameday}T{r.gametime}:00-05:00", "final": pd.notna(r.result)}
        rows.append({**base, "team": r.home_team, "opp": r.away_team, "home": 1,
                     "imp": (total + spread) / 2, "margin": spread})
        rows.append({**base, "team": r.away_team, "opp": r.home_team, "home": 0,
                     "imp": (total - spread) / 2, "margin": -spread})
    return pd.DataFrame(rows)


def match_stats(snaps, stats):
    """Attach each snap row's box score. Snap and stat files name players differently
    (suffixes, accents, Bam vs Zonovan), so fall back to a unique same-team surname."""
    st = stats[stats.season_type == "REG"].copy()
    st["key"] = st.player_display_name.map(norm_name)
    st = st.groupby(["week", "team", "key"], as_index=False).agg({**{c: "sum" for c in STATS}, "player_id": "first"})
    out = snaps.merge(st, how="left", on=["week", "team", "key"], indicator=True)
    miss = out["_merge"] == "left_only"
    st["sur"] = st.key.map(surname)
    by_sur = st.groupby(["week", "team", "sur"])
    unique = by_sur.filter(lambda x: len(x) == 1).set_index(["week", "team", "sur"])
    taken = set(zip(out.loc[~miss, "week"], out.loc[~miss, "team"], out.loc[~miss, "key"]))
    missing = list(zip(out.loc[miss, "week"], out.loc[miss, "team"], out.loc[miss, "key"].map(surname)))
    for i, k in zip(out.index[miss], missing):
        if missing.count(k) == 1 and k in unique.index and (k[0], k[1], unique.loc[k, "key"]) not in taken:
            out.loc[i, STATS + ["player_id"]] = unique.loc[k, STATS + ["player_id"]].values
    out[STATS] = out[STATS].fillna(0)
    # Snap counts can post before box scores; leave out games whose box score isn't published yet.
    boxed = set(zip(st.week, st.team))
    return out[[k in boxed for k in zip(out.week, out.team)]].drop(columns="_merge")


def player_games(season, snaps, stats, pbp):
    """Every game an offensive skill player took a snap in, with the box score and play-by-play usage."""
    s = snaps[(snaps.game_type == "REG") & (snaps.offense_snaps > 0)].copy()
    s["position"] = s.position.replace({"FB": "RB"})
    s = s[s.position.isin(POSITIONS)]
    s["key"] = s.player.map(norm_name)
    s = s[["week", "team", "player", "pfr_player_id", "position", "key", "offense_pct"]]
    s = match_stats(s, stats)
    s = s.merge(pbp_usage(pbp), how="left", on=["week", "team", "player_id"])
    s[PBP] = s[PBP].fillna(0).clip(lower=0)
    s["season"] = season
    s["scrimmage_yards"] = s.rushing_yards + s.receiving_yards
    s["rr_tds"] = s.rushing_tds + s.receiving_tds
    return s.rename(columns={"pfr_player_id": "pid", "position": "pos", "offense_pct": "snap"})


# Per-game volume and production averaged over a player's earlier games.
ROLLING = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
           "receptions", "targets", "receiving_yards", "scrimmage_yards", "snap", "rr_tds",
           "rz_opp", "i10_opp", "long_rush", "long_rec"]
SHARES = ["carries", "targets", "rz_opp", "i10_opp"]


def add_features(pg, upcoming=None, out=()):
    """Features for every played game, plus `upcoming` rows (season, week, team, pid, ...) with no box score.

    `out` is (season, week, team, pid) for upcoming-week players ruled out, which feeds the vacated-role inputs.
    """
    rows = pg.copy()
    rows["played"] = True
    if upcoming is not None and len(upcoming):
        rows = pd.concat([rows, upcoming.assign(played=False)], ignore_index=True)
    rows = rows.sort_values(["season", "pid", "week"]).reset_index(drop=True)
    grp = rows.groupby(["season", "pid"])
    # Only earlier games count: shift by one, and the upcoming row's own stats are NaN anyway.
    played = rows.where(rows.played)
    for c in ROLLING:
        prev = played[c].groupby([rows.season, rows.pid]).shift()
        rows[c + "_g"] = prev.groupby([rows.season, rows.pid]).transform(lambda x: x.expanding().mean())
        rows[c + "_l3"] = prev.groupby([rows.season, rows.pid]).transform(lambda x: x.rolling(3, min_periods=1).mean())
    rows["gp"] = grp.played.cumsum() - rows.played.astype(int)
    rows["snap_last"] = played.snap.groupby([rows.season, rows.pid]).shift()
    rows["snap_last"] = rows.groupby(["season", "pid"]).snap_last.ffill()

    # Team volume shares: the player's share of the team's carries and targets in games played.
    # _sh3 is the same over the last three games, to show a role growing or shrinking.
    team = pg.groupby(["season", "week", "team"])[SHARES + ["attempts"]].sum().add_prefix("t_").reset_index()
    rows = rows.merge(team, how="left", on=["season", "week", "team"])
    by = [rows.season, rows.pid]
    for c in SHARES:
        num, den = rows[c].where(rows.played).groupby(by).shift(), rows["t_" + c].where(rows.played).groupby(by).shift()
        rows[c + "_sh"] = (num.groupby(by).cumsum() / den.groupby(by).cumsum().replace(0, np.nan)).fillna(0)
        n3 = num.groupby(by).transform(lambda x: x.rolling(3, min_periods=1).sum())
        d3 = den.groupby(by).transform(lambda x: x.rolling(3, min_periods=1).sum())
        rows[c + "_sh3"] = (n3 / d3.replace(0, np.nan)).fillna(0)
    rows = add_vacated(rows, pg, set(out))

    # Played in one of the team's two previous games: a proxy for still having a role.
    team_weeks = pg.groupby(["season", "team"]).week.apply(lambda w: sorted(set(w))).to_dict()
    seen = set(zip(pg.season, pg.team, pg.pid, pg.week))
    recent = []
    for r in rows[["season", "team", "pid", "week"]].itertuples(index=False):
        prior = [w for w in team_weeks.get((r.season, r.team), []) if w < r.week][-2:]
        recent.append(any((r.season, r.team, r.pid, w) in seen for w in prior))
    rows["recent"] = recent
    for p in POSITIONS:
        rows["is" + p] = (rows.pos == p).astype(int)
    return add_matchup(rows, pg)


def add_vacated(rows, pg, out):
    """vac_<c>: season share of team carries/targets held by regulars missing this game.

    A regular played for the team in one of its two previous games. In past weeks "missing" means
    no offensive snap; for the upcoming week it means ruled out (Out, Doubtful, IR, suspended).
    """
    g = pg.sort_values("week").join(pg.groupby(["season", "week", "team"])[["carries", "targets"]].sum().add_prefix("t_"),
                                    on=["season", "week", "team"])
    by = [g.season, g.pid]
    for c in ("carries", "targets"):
        g[c + "_thru"] = (g.groupby(by)[c].cumsum() / g.groupby(by)["t_" + c].cumsum().replace(0, np.nan)).fillna(0)
    thru = {(r.season, r.team, r.pid, r.week): (r.carries_thru, r.targets_thru) for r in g.itertuples(index=False)}
    on_field = {}
    for r in g.itertuples(index=False):
        on_field.setdefault((r.season, r.team, r.week), set()).add(r.pid)
    weeks_of = {}
    for s, t, w in set(zip(rows.season, rows.team, rows.week)) | set(on_field):
        weeks_of.setdefault((s, t), set()).add(w)
    vac = {}
    for (s, t), weeks in weeks_of.items():
        weeks = sorted(weeks)
        for i, w in enumerate(weeks):
            prev = [pw for pw in weeks[:i] if (s, t, pw) in on_field]
            regulars = set().union(*[on_field[(s, t, pw)] for pw in prev[-2:]]) if prev else set()
            here = on_field.get((s, t, w))
            missing = [p for p in regulars if (here is not None and p not in here) or (s, w, t, p) in out]
            vac[(s, w, t)] = {}
            for p in missing:
                last = [pw for pw in prev if (s, t, p, pw) in thru]
                if last:
                    vac[(s, w, t)][p] = thru[(s, t, p, last[-1])]
    for k, c in ((0, "vac_carries"), (1, "vac_targets")):
        rows[c] = [min(1, sum(v[k] for p, v in vac.get((s, w, t), {}).items() if p != me))
                   for s, w, t, me in zip(rows.season, rows.week, rows.team, rows.pid)]
    return rows


# What each defense has allowed per game to each position.
ALLOWED = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
           "receptions", "targets", "receiving_yards", "scrimmage_yards", "rr_tds"]


def allowed_by_game(pg):
    """One row per defense, game and position: totals that position put up against it."""
    games = pg[["season", "week", "opp"]].drop_duplicates()
    out = pg.groupby(["season", "week", "opp", "pos"])[ALLOWED].sum()
    # A position with no snaps against a defense in a game it played allowed zero, not missing.
    full = pd.MultiIndex.from_tuples([(s, w, d, p) for s, w, d in games.itertuples(index=False) for p in POSITIONS],
                                     names=["season", "week", "opp", "pos"])
    return out.reindex(full, fill_value=0).reset_index().rename(columns={"opp": "defense"})


def defense_table(pg, season, week):
    """Per-game allowed by each defense to each position in games before `week`, with league average and rank.

    Rank 1 allows the most, so 1 is the best matchup for the offense.
    """
    a = allowed_by_game(pg)
    a = a[(a.season == season) & (a.week < week)]
    per = a.groupby(["defense", "pos"])[ALLOWED].mean()
    per["games"] = a.groupby(["defense", "pos"]).size()
    per = per.reset_index()
    for c in ALLOWED:
        per[c + "_lg"] = per.groupby("pos")[c].transform("mean")
        per[c + "_rank"] = per.groupby("pos")[c].rank(ascending=False, method="min")
    return per


def add_matchup(rows, pg):
    """dvp_<stat>: the opponent's allowed per game to the player's position, minus league average, before this week."""
    a = allowed_by_game(pg)
    keys = rows[["season", "week"]].drop_duplicates()
    parts = []
    for s, w in keys.itertuples(index=False):
        prior = a[(a.season == s) & (a.week < w)]
        if not len(prior):
            continue
        per = prior.groupby(["defense", "pos"])[ALLOWED].mean()
        lg = prior.groupby("pos")[ALLOWED].mean()
        diff = (per - lg.reindex(per.index.get_level_values("pos")).values).add_prefix("dvp_")
        diff["dgp"] = prior.groupby(["defense", "pos"]).size()
        parts.append(diff.reset_index().assign(season=s, week=w))
    dvp = pd.concat(parts, ignore_index=True).rename(columns={"defense": "opp"})
    return rows.merge(dvp, how="left", on=["season", "week", "opp", "pos"])


def load_season(season, games, src=None):
    """Player games for one season; src is a local folder holding snapsYYYY.csv and statsYYYY.csv."""
    snaps = read_csv(f"{src}/snaps{season}.csv" if src else snaps_url(season))
    stats = read_csv(f"{src}/stats{season}.csv" if src else stats_url(season))
    pbp = read_csv(f"{src}/pbp{season}.csv.gz" if src else pbp_url(season), usecols=PBP_COLS)
    pg = player_games(season, snaps, stats, pbp)
    lines = team_lines(games, season)
    return pg.merge(lines[["season", "week", "team", "opp", "imp", "margin", "home", "game_id"]],
                    how="left", on=["season", "week", "team"])


# Milestones the board shows. "feats" are the model inputs for that stat; position flags are added
# for stats that cover more than one position. "matchup" adds the opponent's allowed-vs-position,
# kept only where it improved held-out accuracy at every milestone (receiving, receptions,
# rush + rec and passing TDs saw no gain or a loss). vac_ inputs went in the same way; they hurt
# the QB stats. Share trends (_sh3) and red zone usage didn't help yardage, so they're display only.
MARKETS = {
    "rush_yds": {"label": "Rushing yards", "stat": "rushing_yards", "pos": ["RB", "QB"], "lines": [40, 60, 80],
                 "matchup": True,
                 "feats": ["rushing_yards_g", "rushing_yards_l3", "carries_g", "carries_sh", "vac_carries"]},
    "carries": {"label": "Rush attempts", "stat": "carries", "pos": ["RB", "QB"], "lines": [12, 15, 18],
                "matchup": True,
                 "feats": ["carries_g", "carries_l3", "carries_sh", "vac_carries"]},
    "rec_yds": {"label": "Receiving yards", "stat": "receiving_yards", "pos": ["WR", "TE", "RB"], "lines": [40, 60, 80],
                "feats": ["receiving_yards_g", "receiving_yards_l3", "targets_g", "targets_sh", "vac_targets"]},
    "receptions": {"label": "Receptions", "stat": "receptions", "pos": ["WR", "TE", "RB"], "lines": [4, 5, 6],
                   "feats": ["receptions_g", "receptions_l3", "targets_g", "targets_sh", "vac_targets"]},
    "scrimmage": {"label": "Rush + rec yards", "stat": "scrimmage_yards", "pos": ["RB", "WR", "TE"], "lines": [60, 80, 100],
                  "feats": ["scrimmage_yards_g", "scrimmage_yards_l3", "carries_g", "targets_g", "vac_carries", "vac_targets"]},
    "td": {"label": "Anytime TD", "stat": "rr_tds", "pos": ["RB", "WR", "TE"], "lines": [1],
           "feats": ["rr_tds_g", "rz_opp_g", "i10_opp_g", "rz_opp_sh", "i10_opp_sh", "carries_g", "targets_g"]},
    "long_rush": {"label": "Longest rush", "stat": "long_rush", "pos": ["RB", "QB"], "lines": [10, 15, 20],
                  "feats": ["long_rush_g", "long_rush_l3", "carries_g", "carries_sh"]},
    "long_rec": {"label": "Longest catch", "stat": "long_rec", "pos": ["WR", "TE", "RB"], "lines": [15, 20, 25],
                 "feats": ["long_rec_g", "long_rec_l3", "targets_g", "receptions_g", "targets_sh", "vac_targets"]},
    "pass_yds": {"label": "Passing yards", "stat": "passing_yards", "pos": ["QB"], "lines": [225, 250],
                 "matchup": True,
                 "feats": ["passing_yards_g", "passing_yards_l3", "attempts_g"]},
    "pass_tds": {"label": "Passing TDs", "stat": "passing_tds", "pos": ["QB"], "lines": [2],
                 "feats": ["passing_tds_g", "passing_yards_g", "attempts_g"]},
    "completions": {"label": "Completions", "stat": "completions", "pos": ["QB"], "lines": [20, 22],
                    "matchup": True,
                 "feats": ["completions_g", "completions_l3", "attempts_g"]},
}
GAME = ["imp", "margin"]
MIN_GAMES = 2


def model_inputs(market):
    m = MARKETS[market]
    flags = ["is" + p for p in m["pos"][1:]]
    return m["feats"] + ["snap_last"] + GAME + flags + (["dvp_" + m["stat"]] if m.get("matchup") else [])


def eligible(rows, market):
    """Players the board would list: enough earlier games, a recent role, a betting line."""
    m = MARKETS[market]
    return rows[rows.pos.isin(m["pos"]) & (rows.gp >= MIN_GAMES) & rows.recent & rows.imp.notna()]
