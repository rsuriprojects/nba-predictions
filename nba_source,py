"""Preseason schedule + results from NBA.com's public schedule JSON.

Basketball Reference does not publish preseason games at all, so this is a
separate data source used ONLY for preseason. The regular season still comes
from Basketball Reference.

This endpoint is unofficial and could change or disappear without notice, so
every function here fails softly: on any problem it prints a warning and
returns an empty table, letting the rest of the pipeline carry on.
"""
import json
import urllib.request
import pandas as pd
import numpy as np

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

COLS = ["date", "away", "home", "away_pts", "home_pts"]

# NBA.com identifies teams by three-letter code; Basketball Reference uses full
# names, and our Elo ratings are keyed by the Basketball Reference name. This
# map is the bridge. Anything not in here (international clubs that show up in
# preseason, for example) gets skipped.
TRICODE = {
    "ATL": "Atlanta Hawks",          "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",          "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",          "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",       "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",        "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",        "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",   "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",      "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",        "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",   "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",  "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",     "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",      "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",              "WAS": "Washington Wizards",
}


def empty():
    """An empty table with the right columns, so callers can treat it normally."""
    return pd.DataFrame(columns=COLS)


def fetch_preseason(season):
    """Preseason games for a season. season=2027 means the 2026-27 season.

    Returns a table with the same columns as predict.fetch_season(). Points are
    filled in for games that are final and left blank (NaN) for games that
    haven't been played yet.
    """
    try:
        req = urllib.request.Request(SCHEDULE_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"WARNING: could not fetch NBA.com schedule ({e}); no preseason")
        return empty()

    league = data.get("leagueSchedule", {})

    # Make sure we got the season we asked for. 2027 -> "2026-27".
    want = f"{season - 1}-{str(season)[-2:]}"
    got = league.get("seasonYear", "")
    if got != want:
        print(f"WARNING: NBA.com is serving {got!r}, wanted {want!r}; no preseason")
        return empty()

    rows = []
    skipped = set()
    for game_date in league.get("gameDates", []):
        raw_date = game_date.get("gameDate", "")
        try:
            date = pd.to_datetime(raw_date.split(" ")[0], format="%m/%d/%Y")
        except Exception:
            continue

        for g in game_date.get("games", []):
            gid = g.get("gameId", "")
            # Game IDs look like 0012600001. The third character is the game
            # type: 1 = preseason, 2 = regular season, 4 = playoffs.
            if len(gid) < 3 or gid[2] != "1":
                continue

            home_code = g.get("homeTeam", {}).get("teamTricode", "")
            away_code = g.get("awayTeam", {}).get("teamTricode", "")
            home = TRICODE.get(home_code)
            away = TRICODE.get(away_code)
            if not home or not away:
                skipped.update(c for c in (home_code, away_code)
                               if c and c not in TRICODE)
                continue

            # gameStatus 3 means Final. Before that the score fields are zeros,
            # which we must not mistake for a real 0-0 result.
            if g.get("gameStatus") == 3:
                home_pts = g.get("homeTeam", {}).get("score")
                away_pts = g.get("awayTeam", {}).get("score")
            else:
                home_pts = away_pts = np.nan

            rows.append({"date": date, "away": away, "home": home,
                         "away_pts": away_pts, "home_pts": home_pts})

    if skipped:
        print(f"skipped non-NBA preseason opponents: {', '.join(sorted(skipped))}")

    if not rows:
        print("no preseason games found on NBA.com yet")
        return empty()

    df = pd.DataFrame(rows, columns=COLS)
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
