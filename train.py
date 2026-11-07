#!/usr/bin/env python3
"""Fit one model per milestone and write model.json. Run by hand when adding a season, not daily.

Usage: train.py [data_dir]   (data_dir holds snapsYYYY.csv, statsYYYY.csv, pbpYYYY.csv.gz and games.csv; else downloads)
Needs numpy, pandas and scikit-learn.

Tier hit rates are graded walk-forward only: each season from 2022 on is predicted by a model trained
on the seasons before it (2020 through the year before). Inputs were chosen on the 2022-24 folds, so
2025-26 are the cleanest check. The live model is refit on every season, but the board shows these
held-out rates, not its raw output.
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
SEASONS = tuple(range(2020, 2027))
FOLDS = tuple((tuple(range(2020, y)), y) for y in range(2022, 2027))
EDGES = [0, .2, .3, .4, .5, .6, .7, .8, .9, 1]
MIN_TIER = 40  # Smaller bins merge into the one below, so every tier rate rests on 40+ games.


def fit(df, inputs, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=2000)).fit(df[inputs].fillna(0), y)


def fit_median(df, inputs, y):
    return make_pipeline(StandardScaler(), QuantileRegressor(quantile=.5, alpha=0, solver="highs")).fit(df[inputs].fillna(0), y)


def median_coefficients(model, inputs):
    scaler, reg = model[0], model[1]
    coef = reg.coef_ / scaler.scale_
    return {"intercept": float(reg.intercept_ - (coef * scaler.mean_).sum()), **{f: float(c) for f, c in zip(inputs, coef)}}


def projection(e, stat, inputs):
    """Median projection: the value the player should land above half the time. Graded walk-forward too."""
    p, y, avg = [], [], []
    for train, test in FOLDS:
        tr, te = e[e.season.isin(train)], e[e.season == test]
        p.append(np.clip(fit_median(tr, inputs, tr[stat]).predict(te[inputs].fillna(0)), 0, None))
        y.append(te[stat].values)
        avg.append(te[stat + "_g"].fillna(0).values)
    p, y, avg = map(np.concatenate, (p, y, avg))
    return {"coef": median_coefficients(fit_median(e, inputs, e[stat]), inputs),
            "heldOut": {"n": int(len(y)), "mae": round(float(np.abs(p - y).mean()), 2),
                        "maeAvg": round(float(np.abs(avg - y).mean()), 2),
                        "over": round(float((y > p).mean()), 3), "under": round(float((y < p).mean()), 3)}}


def coefficients(model, inputs):
    scaler, logit = model[0], model[1]
    coef = logit.coef_[0] / scaler.scale_
    return {"intercept": float(logit.intercept_[0] - (coef * scaler.mean_).sum()),
            **{f: float(c) for f, c in zip(inputs, coef)}}


def tiers(p, y, season):
    """Held-out hit rate by predicted-probability band, pooled so higher bands never rate lower."""
    frame = pd.DataFrame({"p": p, "y": y, "season": season})
    frame["b"] = pd.cut(frame.p, EDGES, labels=False, include_lowest=True)
    bins = []
    for i in range(len(EDGES) - 1):
        part = frame[frame.b == i]
        if not len(part):
            continue
        if bins and (len(part) < MIN_TIER or bins[-1]["n"] < MIN_TIER):
            bins[-1]["parts"].append(part)
            bins[-1]["n"] += len(part)
        else:
            bins.append({"min": EDGES[i], "parts": [part], "n": len(part)})
    # Pool adjacent bands whose rates run backwards.
    merged = []
    for b in bins:
        merged.append(b)
        while len(merged) > 1 and rate(merged[-1]) < rate(merged[-2]):
            top = merged.pop()
            merged[-1]["parts"] += top["parts"]
            merged[-1]["n"] += top["n"]
    out = []
    for b in merged:
        part = pd.concat(b["parts"])
        out.append({"min": b["min"], "n": len(part), "hit": round(float(part.y.mean()), 3),
                    "avgP": round(float(part.p.mean()), 3),
                    "bySeason": {str(s): [int(g.y.sum()), len(g)] for s, g in part.groupby("season")}})
    return out


def rate(b):
    return pd.concat(b["parts"]).y.mean()


def main(src):
    games = F.read_csv(f"{src}/games.csv" if src else F.GAMES_URL)
    pg = pd.concat([F.load_season(s, games, src) for s in SEASONS], ignore_index=True)
    rows = F.add_features(pg)
    rows = rows[rows.played]
    model = {"trained": datetime.date.today().isoformat(), "seasons": list(SEASONS), "minGames": F.MIN_GAMES, "minPrevGames": F.MIN_PREV_GAMES, "markets": {}}
    for key, m in F.MARKETS.items():
        e = F.eligible(rows, key)
        inputs = F.model_inputs(key)
        market = {"label": m["label"], "stat": m["stat"], "pos": m["pos"], "inputs": inputs, "lines": {}}
        for line in m["lines"]:
            ps, ys, ss = [], [], []
            for train, test in FOLDS:
                tr, te = e[e.season.isin(train)], e[e.season == test]
                ps.append(fit(tr, inputs, tr[m["stat"]] >= line).predict_proba(te[inputs].fillna(0))[:, 1])
                ys.append((te[m["stat"]] >= line).astype(int).values)
                ss.append(te.season.values)
            p, y, s = np.concatenate(ps), np.concatenate(ys), np.concatenate(ss)
            market["lines"][str(line)] = {
                "coef": coefficients(fit(e, inputs, e[m["stat"]] >= line), inputs),
                "tiers": tiers(p, y, s),
                "heldOut": {"n": int(len(y)), "base": round(float(y.mean()), 3)},
            }
            t = market["lines"][str(line)]["tiers"]
            print(f"{key} {line}+  n={len(y)} base={y.mean():.2f}  " +
                  "  ".join(f"{x['min']:.0%}+:{x['hit']:.0%}({x['n']})" for x in t))
        if key != "td":  # A median of zero TDs says nothing; the TD page shows the chance instead.
            market["proj"] = projection(e, m["stat"], inputs)
            h = market["proj"]["heldOut"]
            print(f"{key} projection: MAE {h['mae']} vs season avg {h['maeAvg']}, over {h['over']:.0%} under {h['under']:.0%}")
        model["markets"][key] = market
    with open(os.path.join(HERE, "model.json"), "w") as f:
        json.dump(model, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
