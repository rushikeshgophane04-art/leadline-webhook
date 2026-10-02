"""
One-shot forex compression scanner (run every 15 min by GitHub Actions).

Setup: a run of small candles hugging the 52 MA (compression), then one big
candle that breaks out of the compression box.

  WATCH -> compression is forming right now (no breakout yet)
  SETUP -> the last closed candle broke out; sends entry / stop / target

Alerts go to Telegram. Duplicate alerts are suppressed with a small JSON
state file that the workflow carries between runs via actions/cache.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

# -------- CONFIG --------
PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
]
INTERVAL = "15m"
INTERVAL_MIN = 15
MA_LEN = 52
ATR_LEN = 14

COMPRESSION_BARS = int(os.getenv("COMPRESSION_BARS", "6"))   # small candles in a row
SMALL_RANGE_ATR = float(os.getenv("SMALL_RANGE_ATR", "0.6"))  # small = range < x * ATR
NEAR_MA_ATR = float(os.getenv("NEAR_MA_ATR", "0.5"))          # close within x * ATR of MA
BIG_RANGE_ATR = float(os.getenv("BIG_RANGE_ATR", "1.5"))      # big = range >= x * ATR
BIG_BODY_PCT = float(os.getenv("BIG_BODY_PCT", "0.6"))        # body >= x of range
RR = float(os.getenv("RR", "2.0"))                            # target = RR * risk

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = os.getenv("STATE_FILE", "scanner_state.json")
DRY_RUN = os.getenv("DRY_RUN") == "1"
# ------------------------


def fetch(pair: str) -> pd.DataFrame:
    """Download recent candles and drop the still-forming last bar."""
    df = yf.download(f"{pair}=X", period="5d", interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    now = datetime.now(timezone.utc)
    bar_end = df.index.tz_convert("UTC") + timedelta(minutes=INTERVAL_MIN)
    return df[bar_end <= now]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma"] = df["Close"].rolling(MA_LEN).mean()
    prev_close = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    df["range"] = df["High"] - df["Low"]
    df["body"] = (df["Close"] - df["Open"]).abs()
    return df


def is_compression(window: pd.DataFrame, atr: float) -> bool:
    small = (window["range"] < SMALL_RANGE_ATR * atr).all()
    near_ma = ((window["Close"] - window["ma"]).abs() < NEAR_MA_ATR * atr).all()
    return bool(small and near_ma)


def evaluate(df: pd.DataFrame):
    """Return (kind, info) where kind is 'SETUP', 'WATCH' or None."""
    n = COMPRESSION_BARS
    if len(df) < MA_LEN + n + 2:
        return None, None
    last = df.iloc[-1]

    # SETUP: compression on the n bars before the last one, last bar breaks out.
    box = df.iloc[-n - 1:-1]
    atr = df["atr"].iloc[-n - 2]  # ATR measured before compression began
    if pd.notna(atr) and is_compression(box, atr):
        box_hi, box_lo = box["High"].max(), box["Low"].min()
        big = (last["range"] >= BIG_RANGE_ATR * atr
               and last["body"] >= BIG_BODY_PCT * last["range"])
        if big and last["Close"] > box_hi and last["Close"] > last["ma"]:
            entry, stop = last["Close"], box_lo
            return "SETUP", _levels("LONG", entry, stop, df.index[-1])
        if big and last["Close"] < box_lo and last["Close"] < last["ma"]:
            entry, stop = last["Close"], box_hi
            return "SETUP", _levels("SHORT", entry, stop, df.index[-1])

    # WATCH: the latest n bars are compressed.
    atr = df["atr"].iloc[-n - 1]
    box = df.iloc[-n:]
    if pd.notna(atr) and is_compression(box, atr):
        return "WATCH", {
            "bar": str(df.index[-1]),
            "box_hi": box["High"].max(),
            "box_lo": box["Low"].min(),
            "ma": last["ma"],
        }
    return None, None


def _levels(side, entry, stop, bar):
    risk = abs(entry - stop)
    target = entry + RR * risk if side == "LONG" else entry - RR * risk
    return {"bar": str(bar), "side": side, "entry": entry,
            "stop": stop, "target": target}


def fmt(pair: str, x: float) -> str:
    return f"{x:.3f}" if pair.endswith("JPY") else f"{x:.5f}"


def message(pair, kind, info) -> str:
    if kind == "SETUP":
        return (f"🚨 SETUP {pair} {info['side']} ({INTERVAL})\n"
                f"Entry:  {fmt(pair, info['entry'])}\n"
                f"Stop:   {fmt(pair, info['stop'])}\n"
                f"Target: {fmt(pair, info['target'])} ({RR:g}R)\n"
                f"Bar: {info['bar']}")
    return (f"👀 WATCH {pair} ({INTERVAL}) - compression near 52 MA\n"
            f"Box: {fmt(pair, info['box_lo'])} - {fmt(pair, info['box_hi'])}\n"
            f"MA:  {fmt(pair, info['ma'])}\n"
            f"Bar: {info['bar']}")


def send_telegram(text: str):
    if DRY_RUN or not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        print("[telegram skipped]\n" + text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main() -> int:
    state = load_state()
    errors = 0
    for pair in PAIRS:
        try:
            df = fetch(pair)
            if df.empty:
                print(f"{pair}: no data")
                continue
            kind, info = evaluate(add_indicators(df))
            print(f"{pair}: {kind or '-'}")
            if not kind:
                state[f"{pair}:watching"] = False
                continue
            # One WATCH per compression run, one SETUP per breakout bar.
            watching = state.get(f"{pair}:watching", False)
            state[f"{pair}:watching"] = kind == "WATCH"
            if kind == "WATCH" and watching:
                continue
            if kind == "SETUP" and state.get(f"{pair}:setup_bar") == info["bar"]:
                continue
            if kind == "SETUP":
                state[f"{pair}:setup_bar"] = info["bar"]
            send_telegram(message(pair, kind, info))
        except Exception as e:  # keep scanning the other pairs
            errors += 1
            print(f"{pair}: ERROR {e}", file=sys.stderr)
    save_state(state)
    return 1 if errors == len(PAIRS) else 0


if __name__ == "__main__":
    sys.exit(main())
