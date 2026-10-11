"""Grade past predictions against actual results.

Regular season and preseason are graded into the same results.csv but kept
apart by the `season_type` column, and reported as two separate records.
"""
import json, glob, time, os
import pandas as pd

from nba_source import fetch_preseason

SEASON = int(os.environ.get("SEASON", 2027))
MONTHS = ["october","november","december","january","february","march","april"]

def fetch(year):
    frames = []
    for m in MONTHS:
        try:
            frames.append(pd.read_html(
                f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{m}.html")[0])
        except Exception:
            pass
        time.sleep(3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "Visitor/Neutral":"away","PTS":"away_pts","Home/Neutral":"home","PTS.1":"home_pts"})
    df = df[df["Date"] != "Date"]
    df["date"] = pd.to_datetime(df["Date"], format="%a, %b %d, %Y", errors="coerce")
    for col in ("away_pts", "home_pts"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date","home_pts","away_pts"])

# Finished games from both sources.
res = fetch(SEASON)
pre = fetch_preseason(SEASON)
if len(pre):
    pre = pre.dropna(subset=["home_pts","away_pts"])

if len(res) == 0 and len(pre) == 0:
    print("no finished games yet")
    raise SystemExit

# One lookup table, with each game remembering which kind it was.
actual = {}
for df, season_type in ((res, "regular"), (pre, "preseason")):
    for r in df.itertuples():
        actual[(str(r.date.date()), r.home, r.away)] = {
            "home_pts": int(r.home_pts),
            "away_pts": int(r.away_pts),
            "winner": r.home if r.home_pts > r.away_pts else r.away,
            "season_type": season_type,
        }

rows = []
for f in sorted(glob.glob("predictions/*.json")):
    for p in json.load(open(f)):
        a = actual.get((p["game_date"], p["home"], p["away"]))
        if not a:
            continue
        rows.append({
            "game_date": p["game_date"],
            # Predictions logged before preseason existed have no tag; they are
            # all regular season.
            "season_type": p.get("season_type", a["season_type"]),
            "away": p["away"], "home": p["home"],
            "pick": p["pick"],
            "p_home": p["p_home"],
            "conf": round(max(p["p_home"], 1 - p["p_home"]), 4),
            "away_pts": a["away_pts"], "home_pts": a["home_pts"],
            "margin": a["home_pts"] - a["away_pts"],
            "winner": a["winner"],
            "correct": p["pick"] == a["winner"],
            "backfilled": bool(p.get("backfilled", False)),
        })

if not rows:
    print("no graded games yet")
    raise SystemExit

d = pd.DataFrame(rows).sort_values(["game_date", "home"])

def report(sub, label):
    if not len(sub):
        return
    print(f"=== {label} ===")
    print(f"{sub['correct'].sum()} / {len(sub)} = {sub['correct'].mean():.1%}")
    bins = pd.cut(sub["conf"], [0.5,0.6,0.7,1.0])
    print(sub.groupby(bins, observed=True)["correct"].agg(["mean","count"]).round(3))
    print()

report(d[d["season_type"] == "regular"], "REGULAR SEASON (the record)")
report(d[d["season_type"] == "preseason"], "PRESEASON (logged separately)")

d.to_csv("results.csv", index=False)
print("wrote results.csv")
