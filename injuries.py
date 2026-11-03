"""Players ruled out today, from ESPN's injury report, as Elo points per team.

A player listed Out (or suspended) costs his team his value per game: VORP per
82 games from the last two seasons (70/30), which is roughly the points per
game he adds over a replacement player. In this model one point of margin is
~20 Elo (the regression's 0.0065 logit per Elo vs ~0.13 logit per point, from
NBA margins' SD of ~13). Day-to-day players count for nothing: most play.

Elo already reflects a player who has been missing for weeks, so his cost is
scaled by the share of his team's games he has played this season. A star out
since opening night costs ~0; a star who has played every game costs full value.

This could not be backtested (old injury reports aren't kept), so every pick
also saves p_home_no_inj, and score.py compares the two.
"""
import re
import unicodedata

from nba_source import BR_NAMES, ESPN_ALIAS, _get_json
from rosters import LAST_WEIGHT, MULTI, _team_name, fetch_players

INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
OUT = {"Out", "Suspension"}
ELO_PER_VORP = 20       # per 82 games; see above
MIN_GAMES = 20          # fewer games than this and a per-game rate is noise
MAX_COST = 200          # many players out: the replacement-level math stops holding


def _key(name):
    """Name for matching across sites: no accents, punctuation or Jr./III."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv"))


def _rates(df):
    """{player key: VORP per 82 games}, from one season's stats."""
    df = df.drop_duplicates("Player", keep="first")      # traded: season total first
    g = df["G"].astype(float).clip(lower=MIN_GAMES)
    v = df["VORP"].astype(float).fillna(0) * 82 / g
    return {_key(p): x for p, x in zip(df["Player"], v)}


def injury_costs(season, team_games):
    """{team: {"elo": points lost, "out": [names]}} for today. {} on any failure.

    `team_games` is each team's regular-season games played so far.
    """
    try:
        report = _get_json(INJURY_URL).get("injuries", [])
        r1, r2 = _rates(fetch_players(season - 1)), _rates(fetch_players(season - 2))
        value = {p: LAST_WEIGHT * r1.get(p, 0) + (1 - LAST_WEIGHT) * r2.get(p, 0)
                 for p in set(r1) | set(r2)}
        now = fetch_players(season)
        played = {}                       # (player, team) -> games for that team
        for p, t, g in zip(now.get("Player", []), now.get("Team", []), now.get("G", [])):
            if t not in MULTI and _team_name(t):
                played[(_key(p), _team_name(t))] = float(g)
    except Exception as e:
        print(f"  injuries skipped: {e}")
        return {}

    out = {}
    for team in report:
        name = ESPN_ALIAS.get(team.get("displayName"), team.get("displayName"))
        if name not in BR_NAMES:
            continue
        for inj in team.get("injuries", []):
            if inj.get("status") not in OUT:
                continue
            player = inj.get("athlete", {}).get("displayName", "")
            v = max(value.get(_key(player), 0), 0)
            n = team_games.get(name, 0)
            share = min(played.get((_key(player), name), 0) / n, 1) if n else 1
            cost = ELO_PER_VORP * v * share
            if cost >= 1:
                t = out.setdefault(name, {"elo": 0.0, "out": []})
                t["elo"] += cost
                t["out"].append(player)
    for t in out.values():
        t["elo"] = round(min(t["elo"], MAX_COST), 1)
    print(f"  injuries: {sum(len(t['out']) for t in out.values())} players out "
          f"who matter, across {len(out)} teams")
    return out
