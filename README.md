# NFL Stat Board

Players most likely to clear a stat milestone this week (rushing, receiving, receptions, rush + rec yards,
rush attempts, anytime TD, longest rush, longest catch, passing yards, passing TDs, completions), with hit rates tested on seasons the model never saw.
Player stats only (no betting lines). Also shows a projected line per stat, home/away, indoor/outdoor and turf/grass splits, defense vs. position, hit history, role trends and teammates out. Built straight from nflverse data (weekly stats, snaps, play-by-play, weekly rosters), nflverse/nfldata betting lines and ESPN injuries.

- `features.py`: pre-game features, shared by training and the daily build. Also holds the milestone list.
- `train.py`: fits one logistic model and one median projection per stat on 2020–2026, graded walk-forward (each season from 2022 predicted by a model trained only on earlier seasons). Writes `model.json`. Run by hand, for example after adding a season or mid-season to fold in new weeks.
- `build.py`: the daily refresh. Scores the next week's players, removes Out/Doubtful/IR players, records plays (tested rate 50%+) before kickoff in `results/history.json`, grades past plays and writes `index.html`.
- `.github/workflows/refresh.yml`: runs `build.py` every morning and before each kickoff window.

Local run (the Mac's Python needs the cert path for HTTPS):

```
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn
SSL_CERT_FILE=/etc/ssl/cert.pem .venv/bin/python train.py
SSL_CERT_FILE=/etc/ssl/cert.pem .venv/bin/python build.py
```

Changing `SEASON` in `build.py` and `SEASONS`/`FOLDS` in `train.py` rolls it to a new season.
