"""Grade past predictions against actual results."""
import json, glob, time, os
import pandas as pd

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
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    return df.dropna(subset=["date","home_pts","away_pts"])

res = fetch(SEASON)
if len(res) == 0:
    print("no finished games yet")
    raise SystemExit

actual = {(str(r.date.date()), r.home, r.away):
          (r.home if r.home_pts > r.away_pts else r.away) for r in res.itertuples()}

rows = []
for f in sorted(glob.glob("predictions/*.json")):
    for p in json.load(open(f)):
        win = actual.get((p["game_date"], p["home"], p["away"]))
        if win:
            rows.append({**p, "actual": win, "correct": p["pick"] == win})

if not rows:
    print("no graded games yet")
    raise SystemExit

d = pd.DataFrame(rows)
d["conf"] = d["p_home"].apply(lambda x: max(x, 1-x))

print(f"{d['correct'].sum()} / {len(d)} = {d['correct'].mean():.1%}")
print()
bins = pd.cut(d["conf"], [0.5,0.6,0.7,1.0])
print(d.groupby(bins, observed=True)["correct"].agg(["mean","count"]).round(3))

d.to_csv("results.csv", index=False)
print("\nwrote results.csv")
