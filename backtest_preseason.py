"""Compare ways of picking preseason games on past preseasons.

Run by the "backtest" workflow (manual only). For each season it carries the
previous season's Elo forward exactly as predict.py does, pulls that preseason's
results and betting lines from ESPN, and scores several strategies. Nothing is
written; the table in the log is the output.
"""
import os
import numpy as np
import pandas as pd

import predict as P
from nba_source import _get_espn, _parse_espn

SEASONS = [int(s) for s in os.environ.get("SEASONS", "2023,2024,2025,2026").split(",")]
SHRINK = [1.0, 0.5, 0.25, 0.0]


def p_home(diff, s):
    x = np.array([diff * s, 3, 3, 0, 0])          # preseason: no rest data, as live
    z = np.dot((x - P.MU) / P.SD, P.W) + P.B
    return 1 / (1 + np.exp(-z))


rows = []
for season in SEASONS:
    prev = P.fetch_season(season - 1)
    elo = P.update_elo({}, prev.dropna(subset=["home_pts", "away_pts"]))
    elo = {t: P.BASE + P.CARRY * (v - P.BASE) for t, v in elo.items()}

    data = _get_espn(f"{season - 1}0915", f"{season - 1}1115")
    games = pd.DataFrame(_parse_espn(data, season, set()))
    games = games.dropna(subset=["home_pts", "away_pts"])
    games = games[games["home_pts"] != games["away_pts"]]
    print(f"{season}: {len(games)} finished preseason games, "
          f"{games['p_market'].notna().sum()} with a betting line")

    for g in games.itertuples():
        diff = elo.get(g.home, P.BASE) - elo.get(g.away, P.BASE)
        r = {"season": season, "home_won": int(g.home_pts > g.away_pts),
             "p_market": g.p_market}
        for s in SHRINK:
            r[f"elo x{s}"] = p_home(diff, s)
        rows.append(r)

d = pd.DataFrame(rows)
strategies = [f"elo x{s}" for s in SHRINK]


def score(df, col):
    df = df.dropna(subset=[col])
    if not len(df):
        return np.nan, np.nan, 0
    p = df[col].astype(float)
    acc = ((p > 0.5).astype(int) == df["home_won"]).mean()
    brier = ((p - df["home_won"]) ** 2).mean()
    return acc, brier, len(df)


print()
print("Home teams won", f"{d['home_won'].mean():.1%}", "of", len(d), "games")
print()
print(f"{'strategy':<14}{'acc':>8}{'brier':>8}{'n':>6}   per season acc")
for col in strategies + ["p_market"]:
    acc, brier, n = score(d, col)
    per = "  ".join(f"{s}:{score(d[d.season == s], col)[0]:.0%}" for s in SEASONS
                    if score(d[d.season == s], col)[2])
    print(f"{col:<14}{acc:>8.1%}{brier:>8.3f}{n:>6}   {per}")

# Like-for-like on the games that have a line.
m = d.dropna(subset=["p_market"])
if len(m):
    print()
    print(f"On the {len(m)} games with a betting line:")
    for col in strategies + ["p_market"]:
        acc, brier, n = score(m, col)
        print(f"  {col:<12}{acc:>8.1%}{brier:>8.3f}")
