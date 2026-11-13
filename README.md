# NFL Stat Board

Players most likely to clear a stat milestone this week (rushing, receiving, receptions, rush + rec yards,
rush attempts, anytime TD, longest rush, longest catch, passing yards, passing TDs, completions), with hit rates tested on seasons the model never saw.
Player stats only (no betting lines). Also shows a projected line per stat, home/away, indoor/outdoor and turf/grass splits, defense vs. position, hit history, role trends and teammates out. Built straight from nflverse data (weekly stats, snaps, play-by-play, weekly rosters), nflverse/nfldata betting lines and ESPN injuries.

- `features.py`: pre-game features, shared by training and the daily build. Also holds the milestone list.
- `train.py`: fits one logistic model and one median projection per stat on 2020–2026, graded walk-forward (each season from 2022 predicted by a model trained only on earlier seasons). Writes `model.json`. Run by hand, for example after adding a season or mid-season to fold in new weeks, and always after changing the model inputs in `features.py` (the build refuses to run on a `model.json` that doesn't match).
- `build.py`: the daily refresh. Scores the next week's players, removes Out/Doubtful/IR players, records plays (tested rate 50%+) before kickoff in `results/history.json`, grades past plays and writes `index.html`.
- `template.html`: the page. `build.py` drops the week's data into it to make `index.html`.
- `tests/`: quick checks of name matching, kickoff times, the tested-rate lookup and that `model.json` matches `features.py`.
- `.github/workflows/refresh.yml`: runs `build.py` every morning and before each kickoff window. `tests.yml` runs the tests when code changes.

Local run (the Mac's Python needs the cert path for HTTPS):

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt
SSL_CERT_FILE=/etc/ssl/cert.pem .venv/bin/python train.py
SSL_CERT_FILE=/etc/ssl/cert.pem .venv/bin/python build.py
.venv/bin/python -m unittest discover tests
```

`requirements.txt` is what the daily build needs; `requirements-train.txt` adds scikit-learn for training and the tests. Both scripts take an optional folder of downloaded files (`train.py data`), which saves re-downloading seven seasons while experimenting; `data/` is git-ignored.

Anytime TD is the exception: its chance comes from the NFL TD Board (`chances.json` on that site), so both sites show one number. `build.py` falls back to this board's own TD model for a player the TD Board doesn't list, or when its file is missing or for another week.

The chance shown for a play is not the model's raw score. Held-out games are sorted into 5-point bands by score, and the chance is read off the line through the bands' real hit rates.

Changing `SEASON` in `build.py` and `SEASONS`/`FOLDS` in `train.py` rolls it to a new season.
