#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, math, re, threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn

import numpy as np
import pandas as pd

BINANCE_BASE = "https://api.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["15m", "1h", "4h", "1d"]
KLINES_LIMIT = int(os.getenv("KLINES_LIMIT", "500"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", "900"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
USER_AGENT = "Mozilla/5.0 (compatible; OpenAI-DataFetcher/1.0; +https://openai.com)"

cache_lock = threading.Lock()
state: Dict[str, Any] = {
    "last_snapshot": None,
    "last_snapshot_ts": 0.0,
    "last_errors": [],
    "alerts": [],
    "last_run_info": {},
}

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def http_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> requests.Response:
    h = {"User-Agent": USER_AGENT}
    if headers: h.update(headers)
    resp = requests.get(url, params=params or {}, headers=h, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp

def pct_change(a: float, b: float) -> Optional[float]:
    try:
        if b == 0 or b is None or a is None:
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, period=20, num_std=2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_val = dx.rolling(period).mean()
    return plus_di, minus_di, adx_val

def obv(close: pd.Series, volume: pd.Series):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def cvd(close: pd.Series, volume: pd.Series):
    signed_vol = np.sign(close.diff().fillna(0)) * volume
    return signed_vol.cumsum()

def volume_change_pct(volume: pd.Series, lookback: int = 1):
    return volume.pct_change(lookback) * 100.0

def fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    resp = http_get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    arr = resp.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(arr, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.rolling(14).mean(); ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    df["rsi14"] = 100 - (100 / (1 + rs))
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line; df["macd_signal"] = signal_line; df["macd_hist"] = macd_line - signal_line
    mid = close.rolling(20).mean(); std = close.rolling(20).std()
    df["bb_upper"] = mid + 2.0 * std; df["bb_mid"] = mid; df["bb_lower"] = mid - 2.0 * std
    prev_close = close.shift(1)
    tr1 = high - low; tr2 = (high - prev_close).abs(); tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index); minus_dm = pd.Series(minus_dm, index=high.index)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df["adx14"] = dx.rolling(14).mean()
    direction = np.sign(close.diff().fillna(0))
    df["obv"] = (direction * vol).cumsum()
    df["cvd"] = (np.sign(close.diff().fillna(0)) * vol).cumsum()
    df["vol_chg_pct_1"] = vol.pct_change(1) * 100.0
    return df

def pack_chain_under(symbol: str) -> Dict[str, Any]:
    result = {}
    for iv in ["15m", "1h", "4h", "1d"]:
        df = fetch_binance_klines(symbol, iv, int(os.getenv("KLINES_LIMIT", "500")))
        df = compute_indicators(df)
        last = df.iloc[-1].to_dict()
        result[iv] = {
            "last": {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in last.items() if k in [
                        "open","high","low","close","volume",
                        "ema20","ema50","ema200","rsi14",
                        "macd","macd_signal","macd_hist",
                        "bb_upper","bb_mid","bb_lower","adx14",
                        "obv","cvd","vol_chg_pct_1"
                    ]},
            "close_time": df.iloc[-1]["close_time"].isoformat(),
        }
    return result

def parse_next_data_json(html_text: str) -> Optional[dict]:
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', html_text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def scrape_coinglass(btc_symbol="BTC", eth_symbol="ETH", sol_symbol="SOL") -> Dict[str, Any]:
    base = "https://www.coinglass.com/currencies"
    out = {"source": "coinglass", "ts": now_iso(), "data": {}, "warnings": []}
    for s, key in [(btc_symbol, "BTC"), (eth_symbol, "ETH"), (sol_symbol, "SOL")]:
        url = f"{base}/{s}?type=spot"
        try:
            html = http_get(url).text
            nxt = parse_next_data_json(html) or {}
            raw_str = json.dumps(nxt)[:500000]
            def find_num(p):
                mm = re.search(p, raw_str)
                return float(mm.group(1)) if mm else None
            sym_out = {
                "funding_rate": find_num(r'"fundingRate"\s*:\s*([-+]?\d+\.?\d*)'),
                "long_short_ratio": find_num(r'"longShortRatio"\s*:\s*([-+]?\d+\.?\d*)'),
                "open_interest_usd": find_num(r'"openInterestUsd"\s*:\s*([-+]?\d+\.?\d*)'),
                "liquidations_usd_24h": find_num(r'"liquidationsUsd24h"\s*:\s*([-+]?\d+\.?\d*)'),
            }
            out["data"][key] = sym_out
            if all(v is None for v in sym_out.values()):
                out["warnings"].append(f"{key}: 未找到结构化字段，可能页面结构变更")
        except Exception as e:
            out["warnings"].append(f"{key}: 抓取失败 - {e}")
    return out

def scrape_oklink() -> Dict[str, Any]:
    base = "https://www.oklink.com/zh-hans"
    out = {"source": "oklink", "ts": now_iso(), "data": {}, "warnings": []}
    try:
        html = http_get(f"{base}/bitcoin").text
        nxt = parse_next_data_json(html) or {}
        raw_str = json.dumps(nxt)[:500000]
        def find_num(p):
            mm = re.search(p, raw_str)
            return float(mm.group(1)) if mm else None
        out["data"]["BTC"] = {
            "active_addresses": find_num(r'"activeAddresses"\s*:\s*([-+]?\d+\.?\d*)'),
            "new_addresses": find_num(r'"newAddresses"\s*:\s*([-+]?\d+\.?\d*)'),
            "exchange_netflow_usd": find_num(r'"exchangeNetflowUsd"\s*:\s*([-+]?\d+\.?\d*)'),
        }
        if all(v is None for v in out["data"]["BTC"].values()):
            out["warnings"].append("BTC: OKLink 页面结构可能变化，未解析到目标字段")
    except Exception as e:
        out["warnings"].append(f"BTC: 抓取失败 - {e}")

    try:
        html = http_get(f"{base}/ethereum").text
        nxt = parse_next_data_json(html) or {}
        raw_str = json.dumps(nxt)[:500000]
        def find_num(p):
            mm = re.search(p, raw_str)
            return float(mm.group(1)) if mm else None
        out["data"]["ETH"] = {
            "active_addresses": find_num(r'"activeAddresses"\s*:\s*([-+]?\d+\.?\d*)'),
            "new_addresses": find_num(r'"newAddresses"\s*:\s*([-+]?\d+\.?\d*)'),
            "exchange_netflow_usd": find_num(r'"exchangeNetflowUsd"\s*:\s*([-+]?\d+\.?\d*)'),
        }
        if all(v is None for v in out["data"]["ETH"].values()):
            out["warnings"].append("ETH: OKLink 页面结构可能变化，未解析到目标字段")
    except Exception as e:
        out["warnings"].append(f"ETH: 抓取失败 - {e}")

    out["data"]["SOL"] = out["data"].get("SOL", {})
    return out

def aggregate_all(force: bool = False) -> Dict[str, Any]:
    with cache_lock:
        fresh = (time.time() - state["last_snapshot_ts"] < CACHE_TTL_SECONDS)
        if (not force) and fresh and state["last_snapshot"] is not None:
            return {"cached": True, **state["last_snapshot"]}

    chain_under = {}
    errors = []
    for sym in SYMBOLS:
        try:
            chain_under[sym] = pack_chain_under(sym)
        except Exception as e:
            errors.append(f"{sym} binance失败: {e}")

    try:
        coinglass = scrape_coinglass()
    except Exception as e:
        coinglass = {"source":"coinglass","ts":now_iso(),"data":{},"warnings":[f"抓取失败: {e}"]}
        errors.append(f"coinglass: {e}")
    try:
        oklink = scrape_oklink()
    except Exception as e:
        oklink = {"source":"oklink","ts":now_iso(),"data":{},"warnings":[f"抓取失败: {e}"]}
        errors.append(f"oklink: {e}")

    snapshot = {
        "timestamp": now_iso(),
        "symbols": SYMBOLS,
        "intervals": INTERVALS,
        "chain_under": chain_under,
        "onchain": {"coinglass": coinglass, "oklink": oklink},
        "errors": errors
    }

    alerts = []
    try:
        prev = state["last_snapshot"] or {}
        prev_btc_close = (((prev.get("chain_under") or {}).get("BTCUSDT") or {}).get("1h") or {}).get("last", {}).get("close")
        curr_btc_close = chain_under.get("BTCUSDT", {}).get("1h", {}).get("last", {}).get("close")
        if prev_btc_close and curr_btc_close:
            chg = pct_change(curr_btc_close, prev_btc_close)
            if chg is not None and abs(chg) > 5.0:
                alerts.append(f"BTC 1h close 变动 {chg:.2f}% (>5%)")

        def get_oi(snap):
            try:
                return float(((snap.get("onchain") or {}).get("coinglass") or {}).get("data", {}).get("BTC", {}).get("open_interest_usd") or 0)
            except Exception:
                return 0
        prev_oi = get_oi(prev)
        curr_oi = get_oi(snapshot)
        if prev_oi and curr_oi:
            chg_oi = pct_change(curr_oi, prev_oi)
            if chg_oi is not None and abs(chg_oi) > 5.0:
                alerts.append(f"BTC OI 变动 {chg_oi:.2f}% (>5%)")

        for w in coinglass.get("warnings", []):
            alerts.append(f"[结构] Coinglass: {w}")
        for w in oklink.get("warnings", []):
            alerts.append(f"[结构] OKLink: {w}")
    except Exception as e:
        alerts.append(f"告警逻辑异常: {e}")

    snapshot["alerts"] = alerts

    with cache_lock:
        state["last_snapshot"] = snapshot
        state["last_snapshot_ts"] = time.time()
        state["last_errors"] = errors
        state["alerts"] = alerts
        state["last_run_info"] = {
            "force": force,
            "finished_at": now_iso(),
            "errors": errors,
            "warnings": (coinglass.get("warnings", []) + oklink.get("warnings", []))
        }
    return {"cached": False, **snapshot}

def background_refresher():
    while True:
        try:
            aggregate_all(force=False)
        except Exception as e:
            with cache_lock:
                state["last_errors"].append(f"后台刷新失败: {e}")
        time.sleep(REFRESH_INTERVAL_SECONDS)

bg_thread = threading.Thread(target=background_refresher, daemon=True)
bg_thread.start()

app = FastAPI(title="OpenAI 开盘中转抓取器", version="1.0.0")

class FetchResponse(BaseModel):
    cached: bool
    timestamp: str
    symbols: List[str]
    intervals: List[str]
    chain_under: Dict[str, Any]
    onchain: Dict[str, Any]
    errors: List[str]
    alerts: List[str]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "server_time": now_iso(),
        "cache_age_seconds": (time.time() - state["last_snapshot_ts"]) if state["last_snapshot_ts"] else None,
        "last_errors": state["last_errors"],
        "alerts": state["alerts"]
    }

@app.get("/fetch", response_model=FetchResponse)
def fetch(force: bool = Query(False, description="True=忽略缓存，强制实时抓取")):
    snap = aggregate_all(force=force)
    return snap

@app.get("/cache")
def cache_info():
    with cache_lock:
        return {
            "has_snapshot": state["last_snapshot"] is not None,
            "snapshot_ts": state["last_snapshot_ts"],
            "cache_age_seconds": (time.time() - state["last_snapshot_ts"]) if state["last_snapshot_ts"] else None,
            "last_run_info": state["last_run_info"],
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
