"""Pregame moneylines from The Odds API, for the "favorite" baseline.

The baseline always picks the betting favorite. It gives the model something
honest to be measured against: beating 50% is easy, beating the market is not.

Credits: one call with markets=h2h and regions=us costs 1 credit, and it is made
at most once per Eastern day. The summary of each call is saved to
odds/YYYY-MM-DD.json, and if that file already exists (a manual re-run of the
workflow, say) it is read back instead of calling the API again. ~31 credits a
month against the free plan's 500.

The API key comes only from the ODDS_API_KEY environment variable (a GitHub
Actions secret). It is never printed, never written to disk, and scrubbed from
any error message, because everything this pipeline writes is committed to a
public repo.
"""
import json
import os
import statistics
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from nba_source import BR_NAMES, ESPN_ALIAS, _eastern_date, _ml_prob

ODDS_URL = ("https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            "?apiKey={key}&regions=us&markets=h2h&oddsFormat=american")


def _team(name):
    """Translate an Odds API team name to the Basketball Reference one."""
    name = ESPN_ALIAS.get(name, name)
    return name if name in BR_NAMES else None


def _summarize(event):
    """Consensus favorite for one game across every US bookmaker, or None.

    Each book's two moneylines are turned into probabilities with its margin
    removed, then averaged across books. The favorite is the team with the
    higher average probability, which is the same as the lower (more negative)
    moneyline. Its median moneyline is kept for display.
    """
    home, away = _team(event.get("home_team", "")), _team(event.get("away_team", ""))
    if not home or not away:
        return None

    probs, lines = {home: [], away: []}, {home: [], away: []}
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            prices = {_team(o.get("name", "")): o.get("price")
                      for o in market.get("outcomes", [])}
            if prices.get(home) is None or prices.get(away) is None:
                continue
            ph, pa = _ml_prob(prices[home]), _ml_prob(prices[away])
            probs[home].append(ph / (ph + pa))
            probs[away].append(pa / (ph + pa))
            lines[home].append(prices[home])
            lines[away].append(prices[away])

    if not probs[home]:
        return None
    p_home = statistics.mean(probs[home])
    if p_home == 0.5:
        fav = None      # a true pick'em: there is no favorite to grade
    else:
        fav = home if p_home > 0.5 else away
    return {
        "game_date": str(_eastern_date(event["commence_time"]).date()),
        "commence_time": event["commence_time"],
        "away": away, "home": home,
        "fav": fav,
        "fav_ml": round(statistics.median(lines[fav])) if fav else None,
        "fav_prob": round(max(p_home, 1 - p_home), 4),
        "books": len(probs[home]),
    }


def _call_api(key):
    """One request to The Odds API. Returns the parsed event list."""
    try:
        with urllib.request.urlopen(ODDS_URL.format(key=key), timeout=60) as resp:
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
            print(f"  Odds API credits: {used} used, {remaining} left this month")
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # Some errors echo the request URL, which contains the key.
        raise RuntimeError(str(e).replace(key, "***")) from None


def favorites_for(day, now=None):
    """{(home, away): odds summary} for pregame games on `day`, or {}.

    Never raises: a problem with the odds must not stop the model's picks.
    """
    path = f"odds/{day.date()}.json"
    if os.path.exists(path):
        print(f"odds already fetched today; reading {path}")
        with open(path) as f:
            games = json.load(f)["games"]
        return {(g["home"], g["away"]): g for g in games}

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("WARNING: ODDS_API_KEY not set; skipping the favorite baseline")
        return {}

    try:
        events = _call_api(key)
    except Exception as e:
        print(f"WARNING: could not fetch odds ({e}); skipping the favorite baseline")
        return {}

    # Only games that haven't started: a live line already knows the score.
    now = now or datetime.now(timezone.utc)
    games = []
    for event in events:
        try:
            if pd.Timestamp(event["commence_time"]) <= now:
                continue
            g = _summarize(event)
        except Exception:
            continue
        if g and g["game_date"] == str(day.date()):
            games.append(g)

    os.makedirs("odds", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                   "games": games}, f, indent=2)
    print(f"saved odds for {len(games)} games to {path}")
    return {(g["home"], g["away"]): g for g in games}
