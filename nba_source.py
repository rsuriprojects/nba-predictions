"""Preseason schedule + results.

Basketball Reference does not publish preseason games at all, so this is a
separate data source used ONLY for preseason. The regular season still comes
from Basketball Reference.

Three endpoints are tried in order. ESPN comes first because NBA.com blocks
GitHub Actions IP addresses; the two NBA.com URLs are kept as fallbacks in
case that ever changes or ESPN goes down. If all three fail we print a warning
and return nothing, so the regular-season pipeline is never broken by a
preseason problem.
"""
import json
import urllib.request
import pandas as pd
import numpy as np

ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
            "scoreboard?dates={start}-{end}&seasontype=1&limit=1000")
CDN_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
DATA_URL = ("https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/"
            "{year}/league/00_full_schedule.json")

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

# Our Elo ratings are keyed by the Basketball Reference team name, so every
# other source has to be translated into those names.
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

# ESPN spells a few teams differently from Basketball Reference.
ESPN_ALIAS = {
    "LA Clippers": "Los Angeles Clippers",
}

# Every valid Basketball Reference name, for checking ESPN's full team names.
BR_NAMES = set(TRICODE.values())


def empty():
    """An empty table with the right columns, so callers can treat it normally."""
    return pd.DataFrame(columns=COLS)


def _get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_preseason(game_id):
    # NBA.com game IDs look like 0012600001. The third character is the game
    # type: 1 = preseason, 2 = regular season, 4 = playoffs.
    return len(game_id) >= 3 and game_id[2] == "1"


def _eastern_date(iso):
    """ESPN timestamps are UTC. Convert to the Eastern calendar date."""
    ts = pd.to_datetime(iso)
    try:
        return ts.tz_convert("America/New_York").tz_localize(None).normalize()
    except Exception:
        # If timezone data isn't installed, shift by hand. Any offset in the
        # 4-7 hour range lands evening games on the right Eastern date.
        return (ts.tz_localize(None) - pd.Timedelta(hours=6)).normalize()


def _parse_espn(data, season, skipped):
    rows = []
    for event in data.get("events", []):
        if event.get("season", {}).get("type") != 1:   # 1 = preseason
            continue

        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        try:
            date = _eastern_date(event.get("date"))
        except Exception:
            continue

        done = comp.get("status", {}).get("type", {}).get("completed", False)

        side = {}
        for c in comp.get("competitors", []):
            team = c.get("team", {})
            name = team.get("displayName", "")
            name = ESPN_ALIAS.get(name, name)
            if name not in BR_NAMES:
                name = TRICODE.get(team.get("abbreviation", ""))
            if not name:
                skipped.add(team.get("displayName") or team.get("abbreviation"))
                continue

            raw = str(c.get("score", "")).strip()
            pts = int(raw) if (done and raw.isdigit()) else np.nan
            side[c.get("homeAway")] = (name, pts)

        if "home" not in side or "away" not in side:
            continue

        rows.append({"date": date,
                     "away": side["away"][0], "home": side["home"][0],
                     "away_pts": side["away"][1], "home_pts": side["home"][1]})
    return rows


def _parse_cdn(data, season, skipped):
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


def _parse_data(data, season, skipped):
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
            def score(sd):
                s = str(g.get(sd, {}).get("s", "")).strip()
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

    # Preseason always falls in the autumn of the season's first year. This
    # window is deliberately wide; season type is what actually filters.
    start = f"{season - 1}0915"
    end = f"{season - 1}1115"

    attempts = [
        ("ESPN", ESPN_URL.format(start=start, end=end), _parse_espn),
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
            rows = parse(data, season, skipped)
        except Exception as e:
            print(f"could not parse {name} ({e})")
            continue

        if rows:
            print(f"preseason source: {name}")
            break
        print(f"{name} returned no preseason games")

    if skipped:
        print(f"skipped non-NBA preseason opponents: "
              f"{', '.join(sorted(str(s) for s in skipped))}")

    if not rows:
        print("WARNING: no preseason games available")
        return empty()

    df = pd.DataFrame(rows, columns=COLS)
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    return df.drop_duplicates(subset=["date", "home", "away"]) \
             .sort_values("date").reset_index(drop=True)
