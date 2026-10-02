"""
Gap-fill lab: do opening gaps fill, and can fading them be traded?

Gap = today's open vs yesterday's close. "Filled" = price trades back to
yesterday's close during the same day.

Trade (fade the gap):
  enter at the open against the gap, target = yesterday's close,
  stop = open +/- k x gap size (k = 1 -> reward:risk 1:1, k = 2 -> 1:2 against),
  otherwise exit at the close. Costs per side in bps.

Daily bars cannot show whether stop or target came first on days that touch
both, so results are given as worst case (stop first) and best case (target
first). For the last ~2 years, hourly bars settle the order ("hourly").
Train 2005-2017, test 2018-today. Gap size filter is fixed up front.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

START, SPLIT = "2005-01-01", "2018-01-01"
ASSETS = {  # ticker -> (name, cost bps per side)
    "SPY": ("S&P 500", 1.0),
    "QQQ": ("Nasdaq 100", 1.0),
    "^NSEI": ("Nifty 50", 2.0),
    "^NSEBANK": ("Bank Nifty", 2.0),
    "GC=F": ("Gold", 2.0),
}
BUCKETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 1.0]
MIN_GAP, MAX_GAP = 0.0025, 0.015  # trade filter: 0.25% - 1.5%


def yahoo(ticker, **kw):
    df = yf.download(ticker, progress=False, auto_adjust=False, **kw)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def prep(df):
    df = df[(df.Open > 0) & (df.High >= df.Low)].copy()
    df["pc"] = df.Close.shift()
    df["gap"] = df.Open / df.pc - 1
    # drop bad prints: open outside the day's range, or zero-range days
    df = df[(df.Open <= df.High) & (df.Open >= df.Low) & (df.High > df.Low)]
    up, dn = df.gap > 0, df.gap < 0
    df["filled"] = (up & (df.Low <= df.pc)) | (dn & (df.High >= df.pc))
    return df.dropna()


def fill_table(df):
    rows = []
    g = df.gap.abs()
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        for side, mask in (("up", df.gap > 0), ("down", df.gap < 0)):
            sel = df[mask & (g >= lo) & (g < hi)]
            if len(sel):
                rows.append({"gap size": f"{lo*100:.2g}-{hi*100:.2g}%" if hi < 1 else f">{lo*100:.2g}%",
                             "dir": side, "days": len(sel),
                             "filled same day %": round(100 * sel.filled.mean(), 1)})
    return rows


def fade_trades(df, k, cost_bps, order=None):
    """R per trade. order: dict date -> 'stop'/'target' for days touching both."""
    out = []
    g = df.gap.abs()
    sel = df[(g >= MIN_GAP) & (g <= MAX_GAP)]
    for d, r in sel.iterrows():
        side = -1 if r.gap > 0 else 1
        dist = abs(r.Open - r.pc)
        stop = r.Open - side * k * dist
        risk = k * dist
        hit_stop = r.High >= stop if side == -1 else r.Low <= stop
        hit_tgt = r.filled
        cost = 2 * cost_bps / 1e4 * r.Open / risk
        res = {}
        for case in ("worst", "best"):
            if hit_stop and hit_tgt:
                first = (order or {}).get(d, "stop" if case == "worst" else "target")
                R = -1.0 if first == "stop" else 1 / k
            elif hit_stop:
                R = -1.0
            elif hit_tgt:
                R = 1 / k
            else:
                R = side * (r.Close - r.Open) / risk
            res[case] = R - cost
        out.append((d, res["worst"], res["best"], hit_stop and hit_tgt))
    return out


def hourly_order(ticker, daily):
    """For days touching both stop and target, which came first (hourly)."""
    try:
        h = yahoo(ticker, period="730d", interval="1h")
    except Exception:
        return {}, None
    if h.empty:
        return {}, None
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)  # exchange-local wall time
    order, k_orders = {}, {}
    for k in (1, 2):
        o = {}
        for d, g in h.groupby(h.index.normalize()):
            if d not in daily.index:
                continue
            r = daily.loc[d]
            gap = r.gap
            if not (MIN_GAP <= abs(gap) <= MAX_GAP):
                continue
            side = -1 if gap > 0 else 1
            dist = abs(r.Open - r.pc)
            stop = r.Open - side * k * dist
            for _, b in g.iterrows():
                s_hit = b.High >= stop if side == -1 else b.Low <= stop
                t_hit = b.Low <= r.pc if side == -1 else b.High >= r.pc
                if s_hit:  # same hour -> assume stop first
                    o[d] = "stop"
                    break
                if t_hit:
                    o[d] = "target"
                    break
        k_orders[k] = o
    return k_orders, h.index.normalize().min()


def stats(trs, col):
    r = np.array([t[col] for t in trs])
    if not len(r):
        return {}
    wins, losses = r[r > 0], r[r <= 0]
    return {"trades": len(r), "win%": round(100 * len(wins) / len(r), 1),
            "expect_R": round(r.mean(), 3),
            "pf": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.nan}


def main():
    out = ["# Gap-fill lab (daily, 2005-today)",
           f"Fade trades only on gaps of {MIN_GAP*100:.2g}%-{MAX_GAP*100:.2g}%. "
           "R = profit in units of the amount risked. worst/best = order of stop vs "
           "target on days touching both; hourly = order settled with hourly bars "
           "(last ~2 years only).", ""]
    for t, (name, cost) in ASSETS.items():
        try:
            df = prep(yahoo(t, start=START, interval="1d"))
        except Exception as e:
            out += [f"## {name}: data error {e!r}", ""]
            continue
        if len(df) < 500:
            out += [f"## {name}: not enough data", ""]
            continue
        out += [f"## {name} ({df.index[0].date()} - {df.index[-1].date()})",
                "### How often gaps fill the same day",
                pd.DataFrame(fill_table(df)).to_markdown(index=False), "",
                "### Fade-the-gap trades"]
        k_orders, h_start = hourly_order(t, df)
        rows = []
        for k in (1, 2):
            trs = fade_trades(df, k, cost)
            for part, f in (("2005-2017", lambda d: d < pd.Timestamp(SPLIT)),
                            ("2018-today", lambda d: d >= pd.Timestamp(SPLIT))):
                p = [x for x in trs if f(x[0])]
                both = round(100 * np.mean([x[3] for x in p]), 1) if p else np.nan
                w, b = stats(p, 1), stats(p, 2)
                rows.append({"stop": f"{k}x gap (1:{1/k:g})", "period": part,
                             "trades": w.get("trades"), "win% worst": w.get("win%"),
                             "expect_R worst": w.get("expect_R"), "expect_R best": b.get("expect_R"),
                             "pf worst": w.get("pf"), "days hitting both %": both})
            if h_start is not None:
                recent = df[df.index >= h_start]
                p = fade_trades(recent, k, cost, k_orders.get(k, {}))
                s = stats(p, 1)
                rows.append({"stop": f"{k}x gap (1:{1/k:g})", "period": "last 2y hourly",
                             "trades": s.get("trades"), "win% worst": s.get("win%"),
                             "expect_R worst": s.get("expect_R"), "expect_R best": s.get("expect_R"),
                             "pf worst": s.get("pf"), "days hitting both %": None})
        out += [pd.DataFrame(rows).to_markdown(index=False), ""]
    text = "\n".join(out)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
