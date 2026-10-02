# Forex compression scanner

Checks 10 forex pairs on 15-minute candles (free Yahoo Finance data) and sends
Telegram alerts:

- **WATCH**: the last 6 candles are small (range < 0.6×ATR) and closing within
  0.5×ATR of the 52 MA. You get one alert per compression run.
- **SETUP**: compression followed by one big candle (range ≥ 1.5×ATR, body ≥ 60%
  of its range) closing outside the box and on the matching side of the MA.
  Entry is the close, the stop is the far side of the box and the target is 2R.

The thresholds are defaults, not backtested results. Override them with env
vars (`COMPRESSION_BARS`, `SMALL_RANGE_ATR`, `NEAR_MA_ATR`, `BIG_RANGE_ATR`,
`BIG_BODY_PCT`, `RR`) once `backtest_compression.py` finds better values.

## Setup

1. In Telegram, message @BotFather, send `/newbot`, copy the token, then send
   your new bot any message (e.g. "hi").
2. Go to repo **Settings → Secrets and variables → Actions → New repository
   secret**, name it `TELEGRAM_BOT_TOKEN` and paste the token. The chat is
   found automatically (set `TELEGRAM_CHAT_ID` only to send to a different chat).
3. Go to **Actions → Forex compression scan → Run workflow** to test it.

Backtest: **Actions → Forex compression backtest → Run workflow**; results
appear on the run's summary page.

Local dry run: `DRY_RUN=1 python scanner/scan_once.py`
