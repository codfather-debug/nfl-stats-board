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
STATS = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
         "receptions", "targets", "receiving_yards"]


def snaps_url(season):
    return f"{NFLVERSE}/snap_counts/snap_counts_{season}.csv"


def stats_url(season):
    return f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv"


def read_csv(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=120) as resp:
            src = io.BytesIO(resp.read())
    return pd.read_csv(src, low_memory=False)


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
    st = st.groupby(["week", "team", "key"], as_index=False)[STATS].sum()
    out = snaps.merge(st, how="left", on=["week", "team", "key"], indicator=True)
    miss = out["_merge"] == "left_only"
    st["sur"] = st.key.map(surname)
    by_sur = st.groupby(["week", "team", "sur"])
    unique = by_sur.filter(lambda x: len(x) == 1).set_index(["week", "team", "sur"])
    taken = set(zip(out.loc[~miss, "week"], out.loc[~miss, "team"], out.loc[~miss, "key"]))
    missing = list(zip(out.loc[miss, "week"], out.loc[miss, "team"], out.loc[miss, "key"].map(surname)))
    for i, k in zip(out.index[miss], missing):
        if missing.count(k) == 1 and k in unique.index and (k[0], k[1], unique.loc[k, "key"]) not in taken:
            out.loc[i, STATS] = unique.loc[k, STATS].values
    out[STATS] = out[STATS].fillna(0)
    # Snap counts can post before box scores; leave out games whose box score isn't published yet.
    boxed = set(zip(st.week, st.team))
    return out[[k in boxed for k in zip(out.week, out.team)]].drop(columns="_merge")


def player_games(season, snaps, stats):
    """Every game an offensive skill player took a snap in, with the box score."""
    s = snaps[(snaps.game_type == "REG") & (snaps.offense_snaps > 0)].copy()
    s["position"] = s.position.replace({"FB": "RB"})
    s = s[s.position.isin(POSITIONS)]
    s["key"] = s.player.map(norm_name)
    s = s[["week", "team", "player", "pfr_player_id", "position", "key", "offense_pct"]]
    s = match_stats(s, stats)
    s["season"] = season
    s["scrimmage_yards"] = s.rushing_yards + s.receiving_yards
    return s.rename(columns={"pfr_player_id": "pid", "position": "pos", "offense_pct": "snap"})


# Per-game volume and production averaged over a player's earlier games.
ROLLING = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
           "receptions", "targets", "receiving_yards", "scrimmage_yards", "snap"]


def add_features(pg, upcoming=None):
    """Features for every played game, plus `upcoming` rows (season, week, team, pid, ...) with no box score.

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
    team = pg.groupby(["season", "week", "team"])[["carries", "targets", "attempts"]].sum().add_prefix("t_").reset_index()
    rows = rows.merge(team, how="left", on=["season", "week", "team"])
    for c in ("carries", "targets"):
        num = rows[c].where(rows.played).groupby([rows.season, rows.pid]).transform(lambda x: x.shift().expanding().sum())
        den = rows["t_" + c].where(rows.played).groupby([rows.season, rows.pid]).transform(lambda x: x.shift().expanding().sum())
        rows[c + "_sh"] = (num / den.replace(0, np.nan)).fillna(0)

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
    return rows


def load_season(season, games, src=None):
    """Player games for one season; src is a local folder holding snapsYYYY.csv and statsYYYY.csv."""
    snaps = read_csv(f"{src}/snaps{season}.csv" if src else snaps_url(season))
    stats = read_csv(f"{src}/stats{season}.csv" if src else stats_url(season))
    pg = player_games(season, snaps, stats)
    lines = team_lines(games, season)
    return pg.merge(lines[["season", "week", "team", "opp", "imp", "margin", "home", "game_id"]],
                    how="left", on=["season", "week", "team"])


# Milestones the board shows. "feats" are the model inputs for that stat; position flags are added
# for stats that cover more than one position.
MARKETS = {
    "rush_yds": {"label": "Rushing yards", "stat": "rushing_yards", "pos": ["RB", "QB"], "lines": [40, 60, 80],
                 "feats": ["rushing_yards_g", "rushing_yards_l3", "carries_g", "carries_sh"]},
    "carries": {"label": "Rush attempts", "stat": "carries", "pos": ["RB", "QB"], "lines": [12, 15, 18],
                "feats": ["carries_g", "carries_l3", "carries_sh"]},
    "rec_yds": {"label": "Receiving yards", "stat": "receiving_yards", "pos": ["WR", "TE", "RB"], "lines": [40, 60, 80],
                "feats": ["receiving_yards_g", "receiving_yards_l3", "targets_g", "targets_sh"]},
    "receptions": {"label": "Receptions", "stat": "receptions", "pos": ["WR", "TE", "RB"], "lines": [4, 5, 6],
                   "feats": ["receptions_g", "receptions_l3", "targets_g", "targets_sh"]},
    "scrimmage": {"label": "Rush + rec yards", "stat": "scrimmage_yards", "pos": ["RB", "WR", "TE"], "lines": [60, 80, 100],
                  "feats": ["scrimmage_yards_g", "scrimmage_yards_l3", "carries_g", "targets_g"]},
    "pass_yds": {"label": "Passing yards", "stat": "passing_yards", "pos": ["QB"], "lines": [225, 250],
                 "feats": ["passing_yards_g", "passing_yards_l3", "attempts_g"]},
    "pass_tds": {"label": "Passing TDs", "stat": "passing_tds", "pos": ["QB"], "lines": [2],
                 "feats": ["passing_tds_g", "passing_yards_g", "attempts_g"]},
    "completions": {"label": "Completions", "stat": "completions", "pos": ["QB"], "lines": [20, 22],
                    "feats": ["completions_g", "completions_l3", "attempts_g"]},
}
GAME = ["imp", "margin"]
MIN_GAMES = 2


def model_inputs(market):
    m = MARKETS[market]
    flags = ["is" + p for p in m["pos"][1:]]
    return m["feats"] + ["snap_last"] + GAME + flags


def eligible(rows, market):
    """Players the board would list: enough earlier games, a recent role, a betting line."""
    m = MARKETS[market]
    return rows[rows.pos.isin(m["pos"]) & (rows.gp >= MIN_GAMES) & rows.recent & rows.imp.notna()]
