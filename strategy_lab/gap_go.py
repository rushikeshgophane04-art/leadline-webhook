"""
Gap-and-go lab: trade WITH big opening gaps instead of fading them.

Gaps of 1%-4% (today's open vs yesterday's close). Rules fixed up front:

Daily data, 2005-today (train < 2018 <= test). No targets, so stop/target
order never matters:
  day_fullstop  - enter at the open with the gap, stop = yesterday's close
                  (gap filled = idea wrong), exit at the close
  day_halfstop  - same, stop = halfway back into the gap
  5day_fullstop - same entry/stop, hold up to 5 days, exit at the 5th close

Hourly data, last ~2 years (too short to split):
  orb_2R        - wait for the first hour; if it closes in the gap's
                  direction, enter at the next hour's open, stop = first
                  hour's low (high for gap down), target 2R, else exit at
                  the close. Stop assumed first if both hit in one hour.
  orb_close     - same entry and stop, no target, exit at the close

1% risk per trade; costs per side in bps.
"""
import os

import numpy as np
import pandas as pd

from gap_lab import prep, yahoo

START, SPLIT = "2005-01-01", "2018-01-01"
ASSETS = {"SPY": ("S&P 500", 1.0), "QQQ": ("Nasdaq 100", 1.0),
          "^NSEI": ("Nifty 50", 2.0), "^NSEBANK": ("Bank Nifty", 2.0)}
MIN_GAP, MAX_GAP = 0.01, 0.04


def daily_trades(df, stop_frac, days, cost_bps):
    """stop_frac = share of the gap the stop sits behind the open (1 = full gap)."""
    out = []
    idx = np.where((df.gap.abs() >= MIN_GAP) & (df.gap.abs() <= MAX_GAP))[0]
    O, H, L, C, PC = (df[c].values for c in ("Open", "High", "Low", "Close", "pc"))
    busy_until = -1
    for i in idx:
        if i <= busy_until:
            continue
        side = 1 if O[i] > PC[i] else -1
        entry = O[i]
        risk = stop_frac * abs(entry - PC[i])
        stop = entry - side * risk
        exit_px, j = None, i
        for j in range(i, min(i + days, len(df))):
            if j > i and (O[j] <= stop if side == 1 else O[j] >= stop):
                exit_px = O[j]  # gapped through the stop
                break
            if (L[j] <= stop) if side == 1 else (H[j] >= stop):
                exit_px = stop
                break
        if exit_px is None:
            exit_px = C[j]
        busy_until = j
        R = side * (exit_px - entry) / risk - 2 * cost_bps / 1e4 * entry / risk
        out.append((df.index[i], R))
    return out


def orb_trades(ticker, daily, cost_bps, target_r):
    try:
        h = yahoo(ticker, period="730d", interval="1h")
    except Exception:
        return []
    if h.empty:
        return []
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    out = []
    for d, g in h.groupby(h.index.normalize()):
        if d not in daily.index or len(g) < 3:
            continue
        gap = daily.loc[d, "gap"]
        if not (MIN_GAP <= abs(gap) <= MAX_GAP):
            continue
        side = 1 if gap > 0 else -1
        first = g.iloc[0]
        if side * (first.Close - first.Open) <= 0:
            continue  # first hour did not hold the gap's direction
        entry = g.iloc[1].Open
        stop = first.Low if side == 1 else first.High
        risk = side * (entry - stop)
        if risk <= 0:
            continue
        target = entry + side * target_r * risk if target_r else None
        exit_px = None
        for _, b in g.iloc[1:].iterrows():
            if (b.Low <= stop) if side == 1 else (b.High >= stop):
                exit_px = stop
                break
            if target is not None and ((b.High >= target) if side == 1 else (b.Low <= target)):
                exit_px = target
                break
        if exit_px is None:
            exit_px = g.iloc[-1].Close
        R = side * (exit_px - entry) / risk - 2 * cost_bps / 1e4 * entry / risk
        out.append((d, R))
    return out


def stats(rs):
    r = np.array(rs)
    if not len(r):
        return {"trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    eq = np.cumprod(1 + 0.01 * r)
    return {"trades": len(r), "win%": round(100 * len(wins) / len(r), 1),
            "avg_win_R": round(wins.mean(), 2) if len(wins) else 0,
            "avg_loss_R": round(losses.mean(), 2) if len(losses) else 0,
            "expect_R": round(r.mean(), 3),
            "pf": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.nan,
            "total_R": round(r.sum(), 1),
            "maxDD%_1%risk": round(100 * (eq / np.maximum.accumulate(eq) - 1).min(), 1)}


def main():
    out = ["# Gap-and-go lab (trade with gaps of 1%-4%)",
           "R = profit in units of the amount risked; maxDD at 1% risk per trade.", ""]
    for t, (name, cost) in ASSETS.items():
        try:
            df = prep(yahoo(t, start=START, interval="1d"))
        except Exception as e:
            out += [f"## {name}: data error {e!r}", ""]
            continue
        rows = []
        for label, frac, days in (("day_fullstop", 1.0, 1), ("day_halfstop", 0.5, 1),
                                  ("5day_fullstop", 1.0, 5)):
            trs = daily_trades(df, frac, days, cost)
            for part, f in (("2005-2017", lambda d: d < pd.Timestamp(SPLIT)),
                            ("2018-today", lambda d: d >= pd.Timestamp(SPLIT))):
                rows.append({"system": label, "period": part,
                             **stats([r for d, r in trs if f(d)])})
        for label, tr in (("orb_2R", 2.0), ("orb_close", None)):
            rows.append({"system": label, "period": "last 2y (hourly)",
                         **stats([r for _, r in orb_trades(t, df, cost, tr)])})
        out += [f"## {name}", pd.DataFrame(rows).to_markdown(index=False), ""]
    text = "\n".join(out)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
