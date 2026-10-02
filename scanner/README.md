# Forex compression scanner

Checks 10 forex pairs on the **1-hour chart** (free Yahoo Finance data) every
hour and sends Telegram alerts:

- **WATCH**: 4 small candles in a row (range < 0.8×ATR) closing near the
  52 MA (within 1.2×ATR). You get one alert per compression run.
- **SETUP**: compression followed by one big candle (range ≥ 1.6×ATR, body ≥ 60%
  of its range) closing outside the box and on the matching side of the MA.
  Entry is the close, the stop is the far side of the box and the target is 2R.

## Backtest results (2 Oct 2026)

| Chart | Data | Result |
|---|---|---|
| 15m | 60 days | **No edge.** Every top setting lost money on the unseen period. |
| 1h | 2 years | **Small edge.** Training years: 115 trades, 41% wins, +0.09R/trade. Unseen last 30%: 25 trades, 44% wins, +0.22R/trade. |

A 41–44% win rate with 2R targets is a thin edge built on few trades. Paper
trade it for a few weeks before risking real money, and keep risk per trade small.

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
