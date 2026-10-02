"""
Backtest the compression-breakout rules on the last ~60 days of 15m data.

Tests a grid of rule settings on the first 70% of the data (train), picks the
best by total R, then checks those picks on the last 30% (test) which they
never saw. Results are printed and written to the GitHub job summary.
"""
import itertools
import os

import pandas as pd
import yfinance as yf

from scan_once import ATR_LEN, MA_LEN, PAIRS, add_indicators

GRID = {
    "n": [3, 4, 6],                 # compression bars
    "small": [0.6, 0.8, 1.0],       # small candle: range < x * ATR
    "near": [0.5, 0.8, 1.2],        # close within x * ATR of MA
    "big": [1.0, 1.3, 1.6],         # breakout: range >= x * ATR
    "rr": [1.0, 1.5, 2.0],          # target in R
}
INTERVAL = os.getenv("BT_INTERVAL", "15m")
PERIOD = {"15m": "60d", "1h": "730d"}[INTERVAL]
BODY_PCT = 0.6
MAX_HOLD = 96          # bars; exit at close after that
TRAIN_FRAC = 0.7
MIN_TRADES = 15


def spread(pair):
    return 0.02 if pair.endswith("JPY") else 0.0002  # ~2 pips round trip


def load(pair):
    df = yf.download(f"{pair}=X", period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return add_indicators(df)


def signals(df, n, small, near, big):
    """Vectorised version of scan_once.evaluate's SETUP rule."""
    atr_ref = df["atr"].shift(n + 1)
    dist = (df["Close"] - df["ma"]).abs()
    box_ok = ((df["range"].rolling(n).max().shift(1) < small * atr_ref)
              & (dist.rolling(n).max().shift(1) < near * atr_ref))
    box_hi = df["High"].rolling(n).max().shift(1)
    box_lo = df["Low"].rolling(n).min().shift(1)
    is_big = (df["range"] >= big * atr_ref) & (df["body"] >= BODY_PCT * df["range"])
    long_ = box_ok & is_big & (df["Close"] > box_hi) & (df["Close"] > df["ma"])
    short = box_ok & is_big & (df["Close"] < box_lo) & (df["Close"] < df["ma"])
    return long_, short, box_hi, box_lo


def simulate(pair, df, params):
    n, small, near, big, rr = params
    long_, short, box_hi, box_lo = signals(df, n, small, near, big)
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    cost = spread(pair)
    trades, i, L = [], 0, len(df)
    while i < L:
        side = 1 if long_.iat[i] else -1 if short.iat[i] else 0
        if not side:
            i += 1
            continue
        entry = cl[i]
        stop = box_lo.iat[i] if side == 1 else box_hi.iat[i]
        risk = abs(entry - stop)
        if risk <= 0:
            i += 1
            continue
        target = entry + side * rr * risk
        exit_r, j = None, i + 1
        while j < L and j <= i + MAX_HOLD:
            hit_stop = lo[j] <= stop if side == 1 else hi[j] >= stop
            hit_tgt = hi[j] >= target if side == 1 else lo[j] <= target
            if hit_stop:          # same-bar stop+target counts as a loss
                exit_r = -1.0
                break
            if hit_tgt:
                exit_r = rr
                break
            j += 1
        if exit_r is None:
            j = min(j, L - 1)
            exit_r = side * (cl[j] - entry) / risk
        exit_r -= cost / risk
        trades.append((df.index[i], exit_r))
        i = j + 1
    return trades


def stats(trades):
    if not trades:
        return {"trades": 0, "win%": 0.0, "avgR": 0.0, "totalR": 0.0}
    r = pd.Series([t[1] for t in trades])
    return {"trades": len(r), "win%": round(100 * (r > 0).mean(), 1),
            "avgR": round(r.mean(), 3), "totalR": round(r.sum(), 2)}


def main():
    data = {}
    for p in PAIRS:
        df = load(p)
        if len(df) > MA_LEN + ATR_LEN + 20:
            data[p] = df
        print(f"{p}: {len(df)} bars")
    if not data:
        raise SystemExit("No data downloaded")

    rows = []
    combos = list(itertools.product(*GRID.values()))
    for params in combos:
        train, test = [], []
        for p, df in data.items():
            cut = df.index[int(len(df) * TRAIN_FRAC)]
            for t in simulate(p, df, params):
                (train if t[0] < cut else test).append(t)
        rows.append({"params": params, **{f"train_{k}": v for k, v in stats(train).items()},
                     **{f"test_{k}": v for k, v in stats(test).items()}})

    res = pd.DataFrame(rows)
    res[list(GRID)] = pd.DataFrame(res.pop("params").tolist(), index=res.index)
    cols = list(GRID) + [c for c in res.columns if c not in GRID]
    res = res[cols]

    default = res[(res.n == 4) & (res.small == 0.8) & (res.near == 1.2)
                  & (res.big == 1.6) & (res.rr == 2.0)]
    min_trades = MIN_TRADES if (res.train_trades >= MIN_TRADES).any() else 5
    best = (res[res.train_trades >= min_trades]
            .sort_values("train_totalR", ascending=False).head(10))

    out = [f"# Backtest {INTERVAL}: {len(combos)} rule combos, {len(data)} pairs, {PERIOD}",
           f"Train = first {int(TRAIN_FRAC*100)}%, test = last {100-int(TRAIN_FRAC*100)}% "
           f"(unseen). R = multiples of risk, after ~2 pip spread.",
           "", "## Top 10 by train total R (min %d trades)" % min_trades,
           best.to_markdown(index=False) if not best.empty else "_none_",
           "", "## Scanner's current settings (n=4 small=0.8 near=1.2 big=1.6 rr=2)",
           default.to_markdown(index=False)]
    text = "\n".join(out)
    print(text)
    res.to_csv(f"backtest_results_{INTERVAL}.csv", index=False)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
