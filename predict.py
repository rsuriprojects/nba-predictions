"""Daily NBA game predictions. Runs standalone, no notebook state."""
import json, os, time
from datetime import datetime, timezone, timedelta
import pandas as pd, numpy as np

from nba_source import fetch_preseason

SEASON = int(os.environ.get("SEASON", 2027))
TEST_DATE = os.environ.get("TEST_DATE")
MONTHS = ["october","november","december","january","february","march","april"]
BASE, K, HCA, CARRY = 1500, 20, 65, 0.75

def fetch_season(year):
    frames = []
    for m in MONTHS:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{m}.html"
        try:
            frames.append(pd.read_html(url)[0])
        except Exception:
            pass
        time.sleep(3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Visitor/Neutral":"away","PTS":"away_pts",
                            "Home/Neutral":"home","PTS.1":"home_pts"})
    df = df[df["Date"] != "Date"]
    df["date"] = pd.to_datetime(df["Date"], format="%a, %b %d, %Y", errors="coerce")
    # A schedule with no results yet may not have PTS columns at all.
    # Create them as empty so the rest of the code has something to work with.
    for col in ("away_pts", "home_pts"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def update_elo(r, played):
    for g in played.itertuples():
        rh, ra = r.get(g.home, BASE), r.get(g.away, BASE)
        exp = 1/(1+10**(-((rh+HCA)-ra)/400))
        act = 1.0 if g.home_pts > g.away_pts else 0.0
        r[g.home] = rh + K*(act-exp)
        r[g.away] = ra - K*(act-exp)
    return r

def rest_days(played, today):
    last = {}
    for g in played.itertuples():
        last[g.home] = g.date
        last[g.away] = g.date
    return {t: min((today - d).days, 7) for t, d in last.items()}

def games_on(sched, today):
    """Rows of sched happening on `today`. Safe on an empty table."""
    if len(sched) == 0:
        return sched
    return sched[sched["date"] == today]

W  = np.array([0.6109, 0.0868, -0.0450, -0.0933, 0.0850])
MU = np.array([0.0, 2.2, 2.2, 0.22, 0.22])
SD = np.array([124.3, 1.3, 1.3, 0.41, 0.41])
B  = 0.18

# Preseason: last season's ratings say much less (stars rest, rosters changed).
# Backtested on the 2022-23 to 2025-26 preseasons (263 games), counting the
# Elo gap at half strength gave the best-calibrated probabilities; accuracy
# was ~53% whatever we did. When ESPN posts a betting line, that is used instead.
PRESEASON_SHRINK = 0.5

def win_prob(elo, rest, home, away, shrink=1.0):
    rh, ra = elo.get(home, BASE), elo.get(away, BASE)
    hr = rest.get(home, 3); ar = rest.get(away, 3)
    x = np.array([(rh-ra)*shrink, hr, ar, int(hr<=1), int(ar<=1)])
    z = np.dot((x-MU)/SD, W) + B
    return rh, ra, float(1/(1+np.exp(-z)))

def game_prob(elo, rest, g, season_type):
    """(home elo, away elo, p_home, source) for one scheduled game."""
    if season_type == "preseason":
        market = getattr(g, "p_market", np.nan)
        rh, ra, p = win_prob(elo, rest, g.home, g.away, PRESEASON_SHRINK)
        if pd.notna(market):
            return rh, ra, float(market), "market"
        return rh, ra, p, "elo"
    rh, ra, p = win_prob(elo, rest, g.home, g.away)
    return rh, ra, p, "elo"

def state_before(base_elo, sched, day):
    """Elo and rest days using ONLY regular-season results from before `day`.

    Preseason outcomes never move Elo: starters play ~15 minutes, and letting
    them in would corrupt the regular-season record this project exists to keep.
    """
    played = sched.dropna(subset=["home_pts","away_pts"]) if len(sched) else sched
    if len(played):
        played = played[played["date"] < day]
    return update_elo(dict(base_elo), played), rest_days(played, day)

def predict_day(base_elo, sched, pre, day, backfilled=False):
    elo, rest = state_before(base_elo, sched, day)
    out = []
    for slate, season_type in ((games_on(sched, day), "regular"),
                               (games_on(pre, day), "preseason")):
        for g in slate.itertuples():
            rh, ra, p, source = game_prob(elo, rest, g, season_type)
            row = {
                "logged_at_utc": datetime.now(timezone.utc).isoformat(),
                "game_date": str(day.date()),
                "season_type": season_type,
                "away": g.away, "home": g.home,
                "home_elo": round(rh,1), "away_elo": round(ra,1),
                "p_home": round(p,4),
                "pick": g.home if p > 0.5 else g.away,
                "source": source,
            }
            if backfilled:
                # Logged after the game, but computed only from data that
                # existed before it. Flagged so the site can say so.
                row["backfilled"] = True
            out.append(row)
    return out

def write_day(rows, day):
    os.makedirs("predictions", exist_ok=True)
    path = f"predictions/{day.date()}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path

def write_upcoming(base_elo, sched, pre, today):
    """Every scheduled game from today on, with a projected pick.

    Projections use today's ratings for every future game and are rebuilt each
    run, so they sharpen as the season goes. Rest days come from the schedule
    itself, which is fixed in advance.
    """
    elo, _ = state_before(base_elo, sched, today)
    games = []
    for df, season_type in ((sched, "regular"), (pre, "preseason")):
        if not len(df):
            continue
        for g in df[df["date"] >= today].itertuples():
            prior = sched[(sched["date"] < g.date)] if len(sched) else sched
            rest = rest_days(prior, g.date)
            rh, ra, p, source = game_prob(elo, rest, g, season_type)
            games.append({
                "game_date": str(g.date.date()),
                "season_type": season_type,
                "away": g.away, "home": g.home,
                "home_elo": round(rh,1), "away_elo": round(ra,1),
                "p_home": round(p,4),
                "pick": g.home if p > 0.5 else g.away,
                "source": source,
            })
    games.sort(key=lambda r: (r["game_date"], r["home"]))
    with open("upcoming.json", "w") as f:
        json.dump({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "today": str(today.date()),
                   "elo": {t: round(v,1) for t, v in sorted(elo.items(), key=lambda kv: -kv[1])},
                   "games": games}, f, indent=1)
    print(f"wrote {len(games)} upcoming games to upcoming.json")

def main():
    # Both branches must produce a timezone-NAIVE timestamp, because the dates
    # scraped from Basketball Reference are naive and pandas refuses to compare
    # naive against aware.
    if TEST_DATE:
        today = pd.Timestamp(TEST_DATE).normalize()
    else:
        et_now = datetime.now(timezone.utc) - timedelta(hours=5)   # shift to US Eastern
        today = pd.Timestamp(et_now).tz_localize(None).normalize()  # drop the UTC label

    # --- carry over last season's ratings ---
    prev = fetch_season(SEASON - 1)
    elo = {}
    if len(prev):
        elo = update_elo({}, prev.dropna(subset=["home_pts","away_pts"]))
        elo = {t: BASE + CARRY*(v - BASE) for t, v in elo.items()}
        print(f"carried {len(elo)} ratings from {SEASON-1}")
    else:
        print(f"no {SEASON-1} data; starting everyone at {BASE}")

    # Regular season comes from Basketball Reference; preseason from ESPN,
    # because Basketball Reference does not publish preseason games.
    sched = fetch_season(SEASON)
    pre   = fetch_preseason(SEASON)
    print(f"{len(sched)} regular-season games scheduled, {len(pre)} preseason")

    # Fill in any earlier game day that has no predictions file (a missed run,
    # or the start of the season before the pipeline was working).
    days = set()
    for df in (sched, pre):
        if len(df):
            days |= set(df.loc[df["date"] < today, "date"])
    for day in sorted(days):
        if os.path.exists(f"predictions/{day.date()}.json"):
            continue
        rows = predict_day(elo, sched, pre, day, backfilled=True)
        if rows:
            print(f"backfilled {len(rows)} predictions for {day.date()}")
            write_day(rows, day)

    rows = predict_day(elo, sched, pre, today)
    if rows:
        path = write_day(rows, today)
        print(f"wrote {len(rows)} predictions to {path}")
        for r in rows:
            tag = "" if r["season_type"] == "regular" else " [preseason]"
            print(f"  {r['away']} @ {r['home']}: {r['p_home']:.1%} -> {r['pick']}{tag}")
    else:
        print(f"no games on {today.date()}")

    write_upcoming(elo, sched, pre, today)

if __name__ == "__main__":
    main()
