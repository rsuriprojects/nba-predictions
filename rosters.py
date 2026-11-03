"""Roster strength, used to set each team's rating at the start of a season.

Each player counts for the first team he plays for this season, valued by his
VORP (value over a replacement player) from the last two seasons, 70/30. A
team's total is turned into a z-score across the league. Rookies count as 0,
which is roughly replacement level.

Before opening night nobody has played yet, so each player counts for the team
he finished last season with instead.

Backtested on 2023-24 to 2026-27 (3,919 games): starting each season from the
roster alone beat carrying over last season's Elo, 67.5% vs 67.1%, and 67.1% vs
66.0% over each team's first 15 games.
"""
import time

import numpy as np
import pandas as pd

from nba_source import TRICODE

# Basketball Reference codes that differ from the NBA's own.
BR_CODES = {"BRK": "Brooklyn Nets", "CHO": "Charlotte Hornets", "PHO": "Phoenix Suns"}
MULTI = ("TOT", "2TM", "3TM", "4TM", "5TM")      # season totals for traded players
LAST_WEIGHT = 0.7


def _team_name(code):
    return BR_CODES.get(code) or TRICODE.get(code)


def fetch_players(year):
    """Basketball Reference advanced stats for one season, or an empty table."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    try:
        df = pd.read_html(url)[0]
    except Exception as e:
        print(f"  no player stats for {year}: {e}")
        return pd.DataFrame(columns=["Player", "Team", "VORP"])
    finally:
        time.sleep(3)
    df = df.rename(columns={"Tm": "Team"})
    return df[df["Player"] != "Player"]


def _vorp(df):
    """{player: VORP}; a traded player's season total comes first on the page."""
    df = df.drop_duplicates("Player", keep="first")
    return dict(zip(df["Player"], pd.to_numeric(df["VORP"], errors="coerce").fillna(0)))


def _first_team(df, keep):
    df = df[~df["Team"].isin(MULTI)].drop_duplicates("Player", keep=keep)
    return {p: _team_name(t) for p, t in zip(df["Player"], df["Team"]) if _team_name(t)}


def roster_z(season):
    """{team: roster z-score} for `season`, or {} if the data isn't there."""
    now, last, before = (fetch_players(season - k) for k in (0, 1, 2))
    if len(last) == 0:
        return {}
    if len(now):
        teams = _first_team(now, keep="first")
    else:
        teams = _first_team(last, keep="last")
        print("  no games played yet; using where each player finished last season")
    v1, v2 = _vorp(last), _vorp(before)
    total = {}
    for player, team in teams.items():
        value = LAST_WEIGHT * v1.get(player, 0) + (1 - LAST_WEIGHT) * v2.get(player, 0)
        total[team] = total.get(team, 0) + value
    if len(total) < 25:
        return {}
    xs = np.array(list(total.values()))
    return {t: float((x - xs.mean()) / xs.std()) for t, x in total.items()}
