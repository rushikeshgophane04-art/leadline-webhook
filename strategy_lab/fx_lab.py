"""
Forex lab: two strategies with an economic or market-structure reason to work.
Rules are fixed up front (no parameter search) to avoid curve fitting.

1) Carry + trend (daily, 2005-today, 8 currencies)
   Each month-end, rank USD EUR GBP JPY AUD CAD CHF NZD by 3-month interest
   rate (FRED / OECD, lagged one month so nothing is used before release).
   Long the top 3, short the bottom 3. Daily P&L includes the interest
   differential earned. Variants:
     carry          - pure carry
     carry+trend    - keep a leg only while its 3-month return agrees
     momentum       - long top 3 / short bottom 3 by 3-month return
     carry+momentum - rank on the sum of carry rank and momentum rank
   Train 2005-2017, test 2018-today. Results are scaled to 10% yearly
   volatility using the TRAIN period's volatility only.

2) London breakout (1h, last ~2 years, 10 pairs)
   Asian range = high/low of 00:00-06:59 UTC. From 07:00 to 11:59 UTC the
   first break of the range enters (stop order). Stop = other side of the
   range, target = 2R, flat at 20:00 UTC. Skip days whose range is wider
   than 1.5x its 20-day median (news days). One trade per pair per day.
   1% risk per trade, ~1 pip cost per round trip.
"""
import io
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

START, SPLIT = "2005-01-01", "2018-01-01"
RATE_SERIES = {  # OECD 3-month interbank rates, % per year, monthly
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N",
}
# currency -> (yahoo pair, True if price is USD per unit of currency)
USD_PAIRS = {"EUR": ("EURUSD=X", True), "GBP": ("GBPUSD=X", True),
             "AUD": ("AUDUSD=X", True), "NZD": ("NZDUSD=X", True),
             "JPY": ("USDJPY=X", False), "CAD": ("USDCAD=X", False),
             "CHF": ("USDCHF=X", False)}
COST_BPS = 2.0  # per unit of turnover


def yahoo(ticker, **kw):
    df = yf.download(ticker, progress=False, auto_adjust=True, **kw)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()


# ---------------------------------------------------------------- carry
def carry_lab():
    px = {}
    for ccy, (t, direct) in USD_PAIRS.items():
        c = yahoo(t, start=START, interval="1d").Close
        px[ccy] = c if direct else 1 / c
    px = pd.DataFrame(px).dropna(how="all").ffill()
    px["USD"] = 1.0
    ret = px.pct_change().fillna(0)

    rates, notes = {}, []
    for ccy, sid in RATE_SERIES.items():
        s = fred(sid)
        notes.append(f"{ccy}: rates to {s.index[-1].date()}")
        # monthly value for month M becomes usable at the end of month M+1
        s.index = s.index + pd.offsets.MonthEnd(1)
        rates[ccy] = s
    rates = pd.DataFrame(rates).reindex(px.index, method="ffill").ffill() / 100

    month_ends = px.groupby(px.index.to_period("M")).tail(1).index
    mom = (px / px.shift(63) - 1)  # 3-month spot return vs USD (USD = 0)
    ccys = list(px.columns)

    def weights(kind):
        w = pd.DataFrame(0.0, index=px.index, columns=ccys)
        for d in month_ends:
            r, m = rates.loc[d], mom.loc[d]
            if r.isna().any() or m.isna().any():
                continue
            if kind == "momentum":
                score = m.rank()
            elif kind == "carry+momentum":
                score = r.rank() + m.rank()
            else:
                score = r.rank()
            order = score.sort_values()
            longs, shorts = order.index[-3:], order.index[:3]
            row = pd.Series(0.0, index=ccys)
            row[longs], row[shorts] = 1 / 3, -1 / 3
            if kind == "carry+trend":
                for c in longs:
                    if m[c] < 0 and c != "USD":
                        row[c] = 0
                for c in shorts:
                    if m[c] > 0 and c != "USD":
                        row[c] = 0
                row["USD"] -= row.drop("USD").sum() + row["USD"]  # stay dollar-neutral
            w.loc[d] = row
        # hold from the day after each rebalance
        w = w.loc[month_ends].reindex(w.index).ffill().shift().fillna(0)
        return w

    rows = []
    for kind in ["carry", "carry+trend", "momentum", "carry+momentum"]:
        w = weights(kind)
        daily = (w * (ret + rates.shift() / 252)).sum(axis=1)
        daily -= w.diff().abs().sum(axis=1) * COST_BPS / 1e4
        carry_part = (w * rates.shift() / 252).sum(axis=1)
        train = daily[daily.index < SPLIT]
        scale = 0.10 / (train.std() * np.sqrt(252))  # set on train only
        for part, mask in (("train", daily.index < SPLIT), ("test", daily.index >= SPLIT)):
            d = daily[mask] * scale
            eq = (1 + d).cumprod()
            yrs = len(d) / 252
            yearly = d.groupby(d.index.year).sum()
            rows.append({
                "strategy": kind, "part": part,
                "sharpe": round(np.sqrt(252) * d.mean() / d.std(), 2),
                "cagr%_at_10%vol": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
                "maxDD%": round(100 * (eq / eq.cummax() - 1).min(), 1),
                "carry_earned%/yr": round(100 * carry_part[mask].mean() * 252 * scale, 1),
                "up_years": f"{(yearly > 0).sum()}/{len(yearly)}",
            })
    return pd.DataFrame(rows), notes


# ---------------------------------------------------------------- london
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
         "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"]


def london_pair(pair, rr=2.0):
    df = yahoo(f"{pair}=X", period="730d", interval="1h")
    df.index = df.index.tz_convert("UTC")
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    cost = 1.0 * pip
    trades = []
    days = df.groupby(df.index.date)
    ranges = {}
    for day, g in days:
        asia = g[g.index.hour < 7]
        if len(asia) >= 5 and g.index[0].weekday() < 5:
            ranges[day] = (asia.High.max(), asia.Low.min())
    widths = pd.Series({d: h - l for d, (h, l) in ranges.items()})
    med = widths.rolling(20).median().shift()
    for day, g in days:
        if day not in ranges or pd.isna(med.get(day)) or widths[day] > 1.5 * med[day]:
            continue
        hi, lo = ranges[day]
        if hi - lo < 5 * pip:
            continue
        sess = g[(g.index.hour >= 7) & (g.index.hour < 20)]
        pos = None
        for ts, r in sess.iterrows():
            if pos is None:
                if ts.hour >= 12:
                    break
                up, dn = r.High > hi, r.Low < lo
                if up and dn:
                    continue  # ambiguous bar, skip the day
                if up or dn:
                    side = 1 if up else -1
                    entry = max(r.Open, hi) if up else min(r.Open, lo)
                    stop = lo if up else hi
                    risk = abs(entry - stop)
                    pos = dict(side=side, entry=entry, stop=stop, risk=risk,
                               target=entry + side * rr * risk)
                    # same-bar stop/target check (stop first)
            if pos:
                s = pos["side"]
                hit_stop = r.Low <= pos["stop"] if s == 1 else r.High >= pos["stop"]
                hit_tgt = r.High >= pos["target"] if s == 1 else r.Low <= pos["target"]
                if hit_stop:
                    exit_px = pos["stop"]
                elif hit_tgt:
                    exit_px = pos["target"]
                elif ts.hour >= 19:
                    exit_px = r.Close
                else:
                    continue
                trades.append((pd.Timestamp(day), (s * (exit_px - pos["entry"]) - cost) / pos["risk"]))
                break
    return trades


def r_stats(rs):
    rs = np.array(rs)
    if not len(rs):
        return {}
    wins, losses = rs[rs > 0], rs[rs <= 0]
    eq = np.cumprod(1 + 0.01 * rs)
    return {"trades": len(rs), "win%": round(100 * len(wins) / len(rs), 1),
            "reward:risk": round(wins.mean() / -losses.mean(), 2) if len(losses) and len(wins) else np.nan,
            "expect_R": round(rs.mean(), 3),
            "pf": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.nan,
            "total_R": round(rs.sum(), 1),
            "maxDD%_at_1%risk": round(100 * (eq / np.maximum.accumulate(eq) - 1).min(), 1)}


def london_lab():
    rows, all_tr = [], []
    for p in PAIRS:
        tr = london_pair(p)
        all_tr += tr
        rows.append({"pair": p, **r_stats([t[1] for t in tr])})
    all_tr.sort()
    mid = all_tr[len(all_tr) // 2][0] if all_tr else None
    first = [t[1] for t in all_tr if t[0] < mid]
    second = [t[1] for t in all_tr if t[0] >= mid]
    rows.append({"pair": "ALL - first half", **r_stats(first)})
    rows.append({"pair": "ALL - second half", **r_stats(second)})
    return pd.DataFrame(rows)


def main():
    out = []
    try:
        carry, notes = carry_lab()
        out += ["# Forex carry & momentum (8 currencies, monthly rebalance)",
                "Scaled to 10% yearly volatility using train-period volatility only. "
                "Train 2005-2017, test 2018-today.", ", ".join(notes), "",
                carry.to_markdown(index=False), ""]
    except Exception as e:
        out += [f"Carry lab failed: {e!r}", ""]
    london = london_lab()
    out += ["# London breakout (1h, ~2 years, 2R target, 1% risk)",
            london.to_markdown(index=False)]
    text = "\n".join(out)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
