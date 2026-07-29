"""Daily NBA game predictions. Runs standalone, no notebook state."""
import json, os, time
from datetime import datetime, timezone, timedelta
import pandas as pd, numpy as np

SEASON = int(os.environ.get("SEASON", 2027))
TEST_DATE = os.environ.get("TEST_DATE")
MONTHS = ["october","november","december","january","february","march","april"]
BASE, K, HCA = 1500, 20, 65

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
        print(f"no schedule pages for {year} yet")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Visitor/Neutral":"away","PTS":"away_pts",
                            "Home/Neutral":"home","PTS.1":"home_pts"})
    df = df[df["Date"] != "Date"]
    df["date"] = pd.to_datetime(df["Date"], format="%a, %b %d, %Y", errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def elo_from(played):
    r = {}
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

def main():
    today = pd.Timestamp(TEST_DATE) if TEST_DATE else \
            pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=5)).normalize()

    sched = fetch_season(SEASON)
    if len(sched) == 0:
        print("nothing to predict")
        return

    played = sched.dropna(subset=["home_pts","away_pts"])
    played = played[played["date"] < today]
    slate  = sched[sched["date"] == today]

    if len(slate) == 0:
        print(f"no games on {today.date()}")
        return

    elo  = elo_from(played)
    rest = rest_days(played, today)

    W  = np.array([0.6109, 0.0868, -0.0450, -0.0933, 0.0850])
    MU = np.array([0.0, 2.2, 2.2, 0.22, 0.22])
    SD = np.array([124.3, 1.3, 1.3, 0.41, 0.41])
    B  = 0.18

    rows = []
    for g in slate.itertuples():
        rh, ra = elo.get(g.home, BASE), elo.get(g.away, BASE)
        hr = rest.get(g.home, 3); ar = rest.get(g.away, 3)
        x = np.array([rh-ra, hr, ar, int(hr<=1), int(ar<=1)])
        z = np.dot((x-MU)/SD, W) + B
        p = 1/(1+np.exp(-z))
        rows.append({
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "game_date": str(today.date()),
            "away": g.away, "home": g.home,
            "home_elo": round(rh,1), "away_elo": round(ra,1),
            "p_home": round(float(p),4),
            "pick": g.home if p > 0.5 else g.away,
        })

    os.makedirs("predictions", exist_ok=True)
    path = f"predictions/{today.date()}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} predictions to {path}")
    for r in rows:
        print(f"  {r['away']} @ {r['home']}: {r['p_home']:.1%} -> {r['pick']}")

if __name__ == "__main__":
    main()
