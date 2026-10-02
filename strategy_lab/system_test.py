"""
Full trading-system backtest: stop-loss, targets, 1% risk per trade.

Rules common to every system:
  - signal on the daily close, enter at the NEXT day's open
  - stop-loss = entry -/+ k x ATR(14); a gap through the stop fills at the open
  - optional target at RR x risk; if stop and target are hit the same day,
    the stop is assumed first (worst case)
  - position size = 1% of equity / stop distance, capped at MAX_LEV x equity
  - costs per side in bps; results split into train (< 2018) and test (>= 2018)

Systems:
  rsi2      - Connors RSI(2) dip buy above SMA200 (long only)
  turtle    - Donchian 20-day breakout, long and short
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

START, SPLIT = "2005-01-01", "2018-01-01"
RISK = 0.01
SIZE_ATR = 2  # sizing distance when a system has no stop-loss
FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
         "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"]
# ticker -> (market, cost bps per side, max leverage)
ASSETS = {f"{p}=X": ("Forex", 1.0, 10.0) for p in FOREX}
ASSETS.update({"SPY": ("S&P 500", 1.0, 1.0), "QQQ": ("Nasdaq 100", 1.0, 1.0),
               "GC=F": ("Gold", 2.0, 1.0)})


def load(ticker):
    df = yf.download(ticker, start=START, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df[(df.Close > 0) & (df.Open > 0)]


def prep(df):
    df = df.copy()
    c = df.Close
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    df["rsi2"] = 100 - 100 / (1 + up / dn)
    df["sma5"], df["sma200"] = c.rolling(5).mean(), c.rolling(200).mean()
    tr = pd.concat([df.High - df.Low, (df.High - c.shift()).abs(),
                    (df.Low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["hi20"], df["lo20"] = df.High.rolling(20).max().shift(), df.Low.rolling(20).min().shift()
    df["hi10"], df["lo10"] = df.High.rolling(10).max().shift(), df.Low.rolling(10).min().shift()
    return df


# Each system gives: entry side at bar i (+1/-1/0) and whether an open
# position of `side` should exit on the signal at bar i's close.
def rsi2_entry(r):
    return 1 if (r.rsi2 < 10 and r.Close > r.sma200) else 0


def rsi2_exit(r, side):
    return r.Close > r.sma5


def turtle_entry(r):
    return 1 if r.Close > r.hi20 else -1 if r.Close < r.lo20 else 0


def turtle_exit(r, side):
    return r.Close < r.lo10 if side == 1 else r.Close > r.hi10


SYSTEMS = {"rsi2": (rsi2_entry, rsi2_exit), "turtle": (turtle_entry, turtle_exit)}


def run(df, system, stop_atr, rr, use_signal_exit, cost_bps, max_lev, max_days=None):
    """Bar-by-bar simulation. Returns (trades list, daily equity series).

    stop_atr=None means no stop-loss: size as if risking SIZE_ATR x ATR, but
    never exit on price. Targets and R are measured in that same distance.
    """
    entry_fn, exit_fn = SYSTEMS[system]
    rows = list(df.itertuples())
    eq, equity = 1.0, []
    pos = None   # dict(side, entry, stop, target, units, i0)
    pending = 0  # side to enter at next open
    pending_exit = False
    trades = []
    cost = cost_bps / 1e4
    for i, r in enumerate(rows):
        # 1) open: execute pending exit / entry
        if pos and pending_exit:
            pnl = pos["side"] * (r.Open - pos["entry"]) * pos["units"] - cost * r.Open * pos["units"]
            eq += pnl
            trades.append((rows[pos["i0"]].Index, pnl / pos["risk_cash"], i - pos["i0"]))
            pos, pending_exit = None, False
        if not pos and pending and not np.isnan(r.atr):
            entry = r.Open
            dist = (stop_atr or SIZE_ATR) * r.atr
            units = min(RISK * eq / dist, max_lev * eq / entry)
            eq -= cost * entry * units
            pos = dict(side=pending, entry=entry, units=units, i0=i,
                       stop=entry - pending * dist if stop_atr else None,
                       target=entry + pending * rr * dist if rr else None,
                       risk_cash=dist * units)
        pending = 0
        # 2) intraday: stop / target
        if pos:
            s, exit_px = pos["side"], None
            if pos["stop"] is not None:
                hit = r.Low <= pos["stop"] if s == 1 else r.High >= pos["stop"]
                if hit:
                    exit_px = min(r.Open, pos["stop"]) if s == 1 else max(r.Open, pos["stop"])
            if exit_px is None and pos["target"] is not None:
                hit = r.High >= pos["target"] if s == 1 else r.Low <= pos["target"]
                if hit:
                    exit_px = max(r.Open, pos["target"]) if s == 1 else min(r.Open, pos["target"])
            if exit_px is not None:
                pnl = s * (exit_px - pos["entry"]) * pos["units"] - cost * exit_px * pos["units"]
                eq += pnl
                trades.append((rows[pos["i0"]].Index, pnl / pos["risk_cash"], i - pos["i0"] + 1))
                pos = None
        # 3) close: mark to market, then signals for tomorrow's open
        mtm = eq + (pos["side"] * (r.Close - pos["entry"]) * pos["units"] if pos else 0)
        equity.append(mtm)
        if pos:
            too_long = max_days and i - pos["i0"] + 1 >= max_days
            if (use_signal_exit and exit_fn(r, pos["side"])) or too_long:
                pending_exit = True
        elif not np.isnan(r.sma200 if system == "rsi2" else r.hi20):
            pending = entry_fn(r)
    return trades, pd.Series(equity, index=df.index)


def stats(trades, eq):
    r = np.array([t[1] for t in trades])
    if len(r) == 0:
        return {}
    wins, losses = r[r > 0], r[r <= 0]
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    ret = eq.pct_change().dropna()
    return {
        "trades": len(r),
        "win%": round(100 * len(wins) / len(r), 1),
        "avg_win_R": round(wins.mean(), 2) if len(wins) else 0,
        "avg_loss_R": round(losses.mean(), 2) if len(losses) else 0,
        "reward:risk": round(wins.mean() / -losses.mean(), 2) if len(wins) and len(losses) else np.nan,
        "expect_R": round(r.mean(), 3),
        "pf": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.nan,
        "cagr%": round(100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1), 1),
        "maxDD%": round(100 * (eq / eq.cummax() - 1).min(), 1),
        "sharpe": round(np.sqrt(252) * ret.mean() / ret.std(), 2) if ret.std() > 0 else np.nan,
        "loss_streak": best,
        "avg_days": round(np.mean([t[2] for t in trades]), 1),
    }


# (label, system, stop_atr, rr, signal_exit, max_days)
VARIANTS = [
    ("RSI2 no stop, signal exit", "rsi2", None, None, True, None),
    ("RSI2 stop 2ATR, signal exit", "rsi2", 2, None, True, None),
    ("RSI2 stop 3ATR, signal exit", "rsi2", 3, None, True, None),
    ("RSI2 stop 2ATR, 1:1 target", "rsi2", 2, 1, False, 10),
    ("RSI2 stop 2ATR, 1:2 target", "rsi2", 2, 2, False, 20),
    ("RSI2 stop 1ATR, 1:2 target", "rsi2", 1, 2, False, 20),
    ("RSI2 stop 2ATR, 1:3 target", "rsi2", 2, 3, False, 30),
    ("Turtle stop 2ATR, 10d exit", "turtle", 2, None, True, None),
    ("Turtle stop 2ATR, 1:2 target", "turtle", 2, 2, False, None),
    ("Turtle stop 2ATR, 1:3 target", "turtle", 2, 3, False, None),
]


def main():
    data = {}
    for t in ASSETS:
        df = load(t)
        print(f"{t}: {len(df)} days")
        if len(df) > 400:
            data[t] = prep(df)
    markets = sorted({ASSETS[t][0] for t in data})
    rows = []
    for label, system, stop_atr, rr, sig, max_days in VARIANTS:
        for m in markets:
            per_part = {"train": ([], []), "test": ([], [])}
            for t in [t for t in data if ASSETS[t][0] == m]:
                _, cost, lev = ASSETS[t]
                for part, sl in (("train", data[t].loc[:SPLIT]), ("test", data[t].loc[SPLIT:])):
                    # each part is its own fresh account with warm-up indicators
                    tr, eq = run(sl, system, stop_atr, rr, sig, cost, lev, max_days)
                    per_part[part][0].extend(tr)
                    per_part[part][1].append(eq.pct_change().fillna(0))
            for part, (trades, rets) in per_part.items():
                port = (1 + pd.concat(rets, axis=1).fillna(0).mean(axis=1)).cumprod()
                rows.append({"system": label, "market": m, "part": part, **stats(trades, port)})
    res = pd.DataFrame(rows)
    res.to_csv("system_test_results.csv", index=False)

    out = ["# Full trading-system test: next-open entry, ATR stop, 1% risk per trade",
           "R = profit in units of the amount risked. reward:risk = avg win / avg loss.",
           "Forex sized up to 10x leverage; stocks and gold unlevered.", ""]
    show = ["system", "trades", "win%", "avg_win_R", "avg_loss_R", "reward:risk",
            "expect_R", "pf", "cagr%", "maxDD%", "loss_streak", "avg_days"]
    for m in markets:
        for part in ("test", "train"):
            t = res[(res.market == m) & (res.part == part)][show]
            out += [f"## {m} - {part} ({'2018-today, unseen' if part == 'test' else '2005-2017'})",
                    t.to_markdown(index=False), ""]
    text = "\n".join(out)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
