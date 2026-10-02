"""
Strategy lab: backtest well-known trading strategies on ~20 years of daily data.

Each strategy variant is scored on 2005-2017 (train) and then checked on
2018-today (test), which is never used for choosing. Positions are decided at
the close and earn the next day's return, after trading costs.

Strategies (all classic, published rules):
  ma_cross  - long when fast SMA > slow SMA, short when below (trend)
  donchian  - Turtle breakout: enter on N-day high/low, exit on M-day opposite
  tsmom     - time-series momentum: hold the sign of the past N-day return
  rsi2      - Connors RSI(2): buy deep dips in an uptrend (above SMA200),
              sell rips in a downtrend, exit when price crosses SMA5
  bollinger - fade closes outside 20-day 2-sigma bands, exit at the mean
  buy_hold  - benchmark
  "long-only" variants never go short (stocks drift up over time)
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

START = "2005-01-01"
SPLIT = "2018-01-01"

FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
         "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"]
ASSETS = {f"{p}=X": ("Forex", 1.0) for p in FOREX}     # cost in bps per side
ASSETS.update({
    "SPY": ("S&P 500", 1.0),
    "QQQ": ("Nasdaq 100", 1.0),
    "GC=F": ("Gold", 2.0),
    "BTC-USD": ("Bitcoin", 10.0),
})


# ---------------- indicators ----------------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def hold(entry_long, exit_long, entry_short, exit_short):
    """Turn entry/exit signals into a -1/0/+1 position series."""
    pos, out = 0, []
    for el, xl, es, xs in zip(entry_long, exit_long, entry_short, exit_short):
        if pos == 1 and xl:
            pos = 0
        elif pos == -1 and xs:
            pos = 0
        if pos == 0:
            pos = 1 if el else -1 if es else 0
        out.append(pos)
    return np.array(out)


# ---------------- strategies ----------------
def ma_cross(df, fast, slow):
    f, s = df.Close.rolling(fast).mean(), df.Close.rolling(slow).mean()
    return np.sign(f - s).fillna(0).values


def donchian(df, n_in, n_out):
    hi_in, lo_in = df.High.rolling(n_in).max().shift(), df.Low.rolling(n_in).min().shift()
    hi_out, lo_out = df.High.rolling(n_out).max().shift(), df.Low.rolling(n_out).min().shift()
    c = df.Close
    return hold(c > hi_in, c < lo_out, c < lo_in, c > hi_out)


def tsmom(df, lookback):
    return np.sign(df.Close.pct_change(lookback)).fillna(0).values


def rsi2(df, lo, hi):
    r, sma200, sma5 = rsi(df.Close, 2), df.Close.rolling(200).mean(), df.Close.rolling(5).mean()
    c = df.Close
    return hold((r < lo) & (c > sma200), c > sma5,
                (r > hi) & (c < sma200), c < sma5)


def bollinger(df, n, k):
    m, sd = df.Close.rolling(n).mean(), df.Close.rolling(n).std()
    c = df.Close
    return hold(c < m - k * sd, c >= m, c > m + k * sd, c <= m)


def buy_hold(df):
    return np.ones(len(df))


def long_only(fn):
    return lambda df, **kw: np.clip(fn(df, **kw), 0, None)


VARIANTS = [
    ("buy_hold", buy_hold, {}),
    ("ma_cross 20/100", ma_cross, {"fast": 20, "slow": 100}),
    ("ma_cross 50/200", ma_cross, {"fast": 50, "slow": 200}),
    ("donchian 20/10", donchian, {"n_in": 20, "n_out": 10}),
    ("donchian 55/20", donchian, {"n_in": 55, "n_out": 20}),
    ("tsmom 63d", tsmom, {"lookback": 63}),
    ("tsmom 126d", tsmom, {"lookback": 126}),
    ("tsmom 252d", tsmom, {"lookback": 252}),
    ("rsi2 10/90", rsi2, {"lo": 10, "hi": 90}),
    ("rsi2 5/95", rsi2, {"lo": 5, "hi": 95}),
    ("bollinger 20/2", bollinger, {"n": 20, "k": 2.0}),
    ("bollinger 20/2.5", bollinger, {"n": 20, "k": 2.5}),
    ("ma_cross 50/200 long-only", long_only(ma_cross), {"fast": 50, "slow": 200}),
    ("tsmom 252d long-only", long_only(tsmom), {"lookback": 252}),
    ("rsi2 10/90 long-only", long_only(rsi2), {"lo": 10, "hi": 90}),
    ("rsi2 5/95 long-only", long_only(rsi2), {"lo": 5, "hi": 95}),
]


# ---------------- backtest ----------------
def load(ticker):
    df = yf.download(ticker, start=START, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df[df.Close > 0]


def daily_pnl(df, pos, cost_bps):
    """Strategy daily returns: yesterday's position x today's return - costs."""
    pos = pd.Series(pos, index=df.index).shift().fillna(0)
    ret = df.Close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * ret - turnover * cost_bps / 1e4, pos


def trade_returns(pnl, pos):
    """Compound pnl over each run of the same non-zero position."""
    run_id = (pos != pos.shift()).cumsum()
    out = []
    for _, g in pnl[pos != 0].groupby(run_id[pos != 0]):
        out.append((1 + g).prod() - 1)
    return np.array(out)


def metrics(pnl, trades, periods=252):
    if len(pnl) < 50 or pnl.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan, win=np.nan,
                    trades=len(trades), pf=np.nan)
    eq = (1 + pnl).cumprod()
    years = len(pnl) / periods
    wins, losses = trades[trades > 0], trades[trades <= 0]
    return dict(
        sharpe=round(np.sqrt(periods) * pnl.mean() / pnl.std(), 2),
        cagr=round(100 * (eq.iloc[-1] ** (1 / years) - 1), 1),
        maxdd=round(100 * (eq / eq.cummax() - 1).min(), 1),
        win=round(100 * len(wins) / len(trades), 1) if len(trades) else np.nan,
        trades=len(trades),
        pf=round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.nan,
    )


def main():
    data = {}
    for t in ASSETS:
        df = load(t)
        print(f"{t}: {len(df)} days from {df.index[0].date() if len(df) else '-'}")
        if len(df) > 300:
            data[t] = df

    rows = []
    groups = sorted({ASSETS[t][0] for t in data})
    for name, fn, params in VARIANTS:
        for group in groups:
            tickers = [t for t in data if ASSETS[t][0] == group]
            pnls, trades = {}, {"train": [], "test": []}
            for t in tickers:
                df = data[t]
                pnl, pos = daily_pnl(df, fn(df, **params), ASSETS[t][1])
                pnls[t] = pnl
                for part, mask in (("train", pnl.index < SPLIT), ("test", pnl.index >= SPLIT)):
                    trades[part].extend(trade_returns(pnl[mask], pos[mask]))
            # equal-weight portfolio across the group's assets
            port = pd.DataFrame(pnls).fillna(0).mean(axis=1)
            periods = 365 if group == "Bitcoin" else 252
            for part, mask in (("train", port.index < SPLIT), ("test", port.index >= SPLIT)):
                m = metrics(port[mask], np.array(trades[part]), periods)
                rows.append({"strategy": name, "market": group, "part": part, **m})

    res = pd.DataFrame(rows)
    wide = res.pivot_table(index=["market", "strategy"], columns="part",
                           values=["sharpe", "cagr", "maxdd", "win", "trades", "pf"])
    wide.columns = [f"{p}_{m}" for m, p in wide.columns]
    wide = wide.reset_index()
    cols = ["market", "strategy",
            "train_sharpe", "test_sharpe", "test_cagr", "test_maxdd",
            "test_win", "test_pf", "test_trades", "train_win", "train_cagr"]
    wide = wide[cols]

    out = [f"# Strategy lab: daily data {START[:4]}-today, train < {SPLIT[:4]} <= test",
           "Sharpe > 0.5 is decent, > 1 is strong. CAGR/maxDD in %, unlevered, after costs.",
           "Forex = equal-weight portfolio of 10 pairs.", ""]
    for g in groups:
        t = wide[wide.market == g].sort_values("train_sharpe", ascending=False)
        out += [f"## {g} (sorted by train Sharpe)", t.drop(columns="market").to_markdown(index=False), ""]

    # The honest pick: best train Sharpe per market (excluding buy & hold),
    # reported with its test numbers.
    picks = (wide[wide.strategy != "buy_hold"].sort_values("train_sharpe", ascending=False)
             .groupby("market").head(1))
    bh = wide[wide.strategy == "buy_hold"].set_index("market")
    out += ["## Pick per market (chosen on train only) vs buy & hold on test"]
    lines = []
    for _, r in picks.iterrows():
        lines.append({"market": r.market, "pick": r.strategy,
                      "test_sharpe": r.test_sharpe, "test_cagr": r.test_cagr,
                      "test_maxdd": r.test_maxdd, "test_win": r.test_win,
                      "bh_test_sharpe": bh.loc[r.market, "test_sharpe"],
                      "bh_test_cagr": bh.loc[r.market, "test_cagr"]})
    out.append(pd.DataFrame(lines).to_markdown(index=False))

    text = "\n".join(out)
    print(text)
    wide.to_csv("strategy_lab_results.csv", index=False)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
