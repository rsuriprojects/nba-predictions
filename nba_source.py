"""Preseason schedule + results from NBA.com's public schedule JSON.

Basketball Reference does not publish preseason games at all, so this is a
separate data source used ONLY for preseason. The regular season still comes
from Basketball Reference.

Two endpoints are tried in order, because NBA.com blocks some cloud IPs and
the two are served by different infrastructure. Both carry the same games in
different shapes. If both fail we print a warning and return nothing, so the
regular-season pipeline is never broken by a preseason problem.
"""
import json
import urllib.request
import pandas as pd
import numpy as np

CDN_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
DATA_URL = ("https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/"
            "{year}/league/00_full_schedule.json")

# NBA.com returns 403 to requests that don't look like a browser. A User-Agent
# alone is not enough; it also wants to see that we came from nba.com.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

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


def _get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_preseason(game_id):
    # Game IDs look like 0012600001. The third character is the game type:
    # 1 = preseason, 2 = regular season, 4 = playoffs.
    return len(game_id) >= 3 and game_id[2] == "1"


def _parse_cdn(data, season, skipped):
    """Parse the cdn.nba.com shape."""
    league = data.get("leagueSchedule", {})

    want = f"{season - 1}-{str(season)[-2:]}"          # 2027 -> "2026-27"
    got = league.get("seasonYear", "")
    if got != want:
        print(f"WARNING: cdn.nba.com is serving {got!r}, wanted {want!r}")
        return []

    rows = []
    for game_date in league.get("gameDates", []):
        try:
            date = pd.to_datetime(game_date.get("gameDate", "").split(" ")[0],
                                  format="%m/%d/%Y")
        except Exception:
            continue

        for g in game_date.get("games", []):
            if not _is_preseason(g.get("gameId", "")):
                continue

            home_code = g.get("homeTeam", {}).get("teamTricode", "")
            away_code = g.get("awayTeam", {}).get("teamTricode", "")
            home, away = TRICODE.get(home_code), TRICODE.get(away_code)
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
    return rows


def _parse_data(data, skipped):
    """Parse the older data.nba.com shape (months -> games)."""
    rows = []
    for block in data.get("lscd", []):
        for g in block.get("mscd", {}).get("g", []):
            if not _is_preseason(g.get("gid", "")):
                continue

            try:
                date = pd.to_datetime(g.get("gdte", ""), format="%Y-%m-%d")
            except Exception:
                continue

            home_code = g.get("h", {}).get("ta", "")
            away_code = g.get("v", {}).get("ta", "")
            home, away = TRICODE.get(home_code), TRICODE.get(away_code)
            if not home or not away:
                skipped.update(c for c in (home_code, away_code)
                               if c and c not in TRICODE)
                continue

            # Scores are strings here, empty until the game is played.
            def score(side):
                s = str(g.get(side, {}).get("s", "")).strip()
                return int(s) if s.isdigit() else np.nan

            rows.append({"date": date, "away": away, "home": home,
                         "away_pts": score("v"), "home_pts": score("h")})
    return rows


def fetch_preseason(season):
    """Preseason games for a season. season=2027 means the 2026-27 season.

    Returns a table with the same columns as predict.fetch_season(). Points are
    filled in for games that are final and left blank (NaN) for games that
    haven't been played yet.
    """
    skipped = set()
    rows = []

    # data.nba.com indexes by the year the season starts: 2026-27 -> 2026.
    attempts = [
        ("cdn.nba.com", CDN_URL, _parse_cdn),
        ("data.nba.com", DATA_URL.format(year=season - 1), _parse_data),
    ]

    for name, url, parse in attempts:
        try:
            data = _get_json(url)
        except Exception as e:
            print(f"could not reach {name} ({e})")
            continue

        try:
            rows = parse(data, season, skipped) if parse is _parse_cdn \
                   else parse(data, skipped)
        except Exception as e:
            print(f"could not parse {name} ({e})")
            continue

        if rows:
            print(f"preseason source: {name}")
            break

    if skipped:
        print(f"skipped non-NBA preseason opponents: {', '.join(sorted(skipped))}")

    if not rows:
        print("WARNING: no preseason games available from NBA.com")
        return empty()

    df = pd.DataFrame(rows, columns=COLS)
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
