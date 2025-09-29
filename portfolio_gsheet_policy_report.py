#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio & Policy-aware Daily Report (Google Sheets + Email)
============================================================
- Reads holdings / watchlist / settings from Google Sheets
- Builds a bilingual (EN | KR) HTML email report with:
  * Portfolio overview, weights, and cash ratio (EN|KR headers)
  * Risk management (beta-aware stop losses, portfolio beta)
  * Dividend mode (forward yield, annual dividends)
  * Indices snapshot & 7‑day news summary with GPT opinions (optional)
  * U.S. policy focus (topic→tickers), weekly news mapping and GPT summary
  * Enhanced charts (allocation by ticker/sector, P&L by ticker, cash vs. equity)

Scheduling: run via cron / GitHub Actions / Cloud Run. For Google Apps Script users,
this script can be hosted anywhere and just run on a schedule; it is independent
from GAS. Keep your API keys in environment variables.

Environment Variables
---------------------
OPENAI_API_KEY                (optional) for GPT summaries
NEWS_API_KEY                  (optional) for NewsAPI.org
EMAIL_SENDER, EMAIL_PASSWORD  (required to send mail)
EMAIL_RECEIVER                (required to send mail)
SMTP_HOST, SMTP_PORT          (defaults gmail: smtp.gmail.com:587)
GSHEET_ID                     (required) the spreadsheet ID
GOOGLE_APPLICATION_CREDENTIALS (optional) path to service-account JSON for gspread
BASE_CURRENCY                 (optional) 'USD' or 'CAD' (default USD; FX to CAD included)

Google Sheet Structure (recommended)
------------------------------------
Worksheet "Holdings" columns (header row at A1):
  Ticker | Shares | AvgPrice

Worksheet "Watchlist" columns:
  Ticker

Worksheet "Settings" (key/value pairs):
  Key                | Value
  -------------------|-----------------
  CashUSD            | 15000
  BaseCurrency       | USD
  RiskMaxPosPct      | 25
  RiskMaxTickerPct   | 15
  RiskStopLossFloor  | 0.90   # -10% fallback

Dependences
-----------
  pip install gspread oauth2client yfinance pandas matplotlib requests openai

Notes
-----
- If OPENAI/NEWS keys are missing, the script still runs (skips GPT/news parts).
- Charts are embedded as base64 <img> tags; no attachments required.
- All tables render with bilingual headers (English | 한국어).
"""

import os, io, base64, time, html, json, math, smtplib, re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Optional: Google Sheets ---
GSHEET_ID = os.getenv("GSHEET_ID", "")
USE_SHEETS = bool(GSHEET_ID)

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:
    gspread = None

# --- Optional: GPT ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        _openai_client = None
except Exception:
    _openai_client = None

# --- Mail / Config ---
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
BASE_CCY       = (os.getenv("BASE_CURRENCY", "USD") or "USD").upper()
TODAY_STR      = datetime.now().strftime("%Y-%m-%d")

# --- Plot / Font (Korean capable) ---
def setup_matplotlib_korean_font():
    try:
        fonts = [f.name for f in fm.fontManager.ttflist]
        if not any("NanumGothic" in f for f in fonts):
            # Best-effort install if running on Debian-like env
            try:
                import subprocess
                subprocess.run(["sudo","apt-get","update"], check=False)
                subprocess.run(["sudo","apt-get","install","-y","fonts-nanum"], check=False)
                matplotlib.font_manager._rebuild()
            except Exception:
                pass
        matplotlib.rcParams["font.family"] = "NanumGothic" if any("NanumGothic" in f for f in [f.name for f in fm.fontManager.ttflist]) else "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

setup_matplotlib_korean_font()

# --- Helpers ---
def usd_to_cad_rate() -> float:
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=CAD", timeout=10)
        return float(r.json().get("rates", {}).get("CAD", 1.38))
    except Exception:
        return 1.38

FX_USD_CAD = usd_to_cad_rate()

# Bilingual column name helper
BK = lambda en, kr: f"{en} | {kr}"

# --- Google Sheets Loaders ---
def _open_gsheet(gs_id: str):
    if not USE_SHEETS:
        return None
    if gspread is None:
        raise RuntimeError("gspread not installed. pip install gspread oauth2client")
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path or not os.path.exists(creds_path):
        # Try default credentials if running in GCP environment
        client = gspread.oauth()
    else:
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
    return client.open_by_key(gs_id)


def load_holdings_watchlist_settings() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (holdings_df, watchlist_df, settings_dict) from Google Sheets.
    Fallback: if Sheets disabled, build small demo from env/json.
    """
    if USE_SHEETS:
        sh = _open_gsheet(GSHEET_ID)
        ws_holdings = sh.worksheet("Holdings")
        ws_watch    = sh.worksheet("Watchlist")
        ws_settings = sh.worksheet("Settings")
        df_holdings = pd.DataFrame(ws_holdings.get_all_records())
        df_watch    = pd.DataFrame(ws_watch.get_all_records())
        settings_rows = ws_settings.get_all_records()
        settings = {r.get('Key'): r.get('Value') for r in settings_rows}
    else:
        # Fallback demo
        df_holdings = pd.DataFrame([
            {"Ticker":"NVDA","Shares":50,"AvgPrice":123.97},
            {"Ticker":"TSLA","Shares":10,"AvgPrice":320.745},
            {"Ticker":"SCHD","Shares":2146,"AvgPrice":24.3851},
        ])
        df_watch = pd.DataFrame([{"Ticker":"AAPL"},{"Ticker":"MSFT"},{"Ticker":"AMZN"}])
        settings = {
            "CashUSD":"13925.60",
            "BaseCurrency": BASE_CCY,
            "RiskMaxPosPct":"25",
            "RiskMaxTickerPct":"15",
            "RiskStopLossFloor":"0.90"
        }
    # Clean
    for c in ["Ticker"]:
        if c in df_holdings.columns:
            df_holdings[c] = df_holdings[c].astype(str).str.upper().str.strip()
    for c in ["Ticker"]:
        if c in df_watch.columns:
            df_watch[c] = df_watch[c].astype(str).str.upper().str.strip()
    return df_holdings, df_watch, settings


# --- Market / Prices ---
def fetch_last_prices(ticker: str, days: int = 2) -> Tuple[float, float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")["Close"].dropna()
        if len(hist) == 0:
            return (np.nan, np.nan)
        last = float(hist.iloc[-1])
        prev = float(hist.iloc[-2]) if len(hist) >= 2 else np.nan
        return (last, prev)
    except Exception:
        return (np.nan, np.nan)


def compute_rsi_macd(ticker: str, period="365d") -> Tuple[float, float]:
    try:
        df = yf.Ticker(ticker).history(period=period)
        close = df["Close"].dropna()
        if len(close) < 20:
            return (np.nan, np.nan)
        delta = close.diff()
        gain, loss = delta.where(delta>0,0), -delta.where(delta<0,0)
        avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100/(1+rs))
        ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return float(rsi.dropna().iloc[-1]), float(hist.dropna().iloc[-1])
    except Exception:
        return (np.nan, np.nan)


def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


# --- Risk Management (Proposal #1) ---
def stop_loss_by_beta(beta: float, floor: float = 0.90) -> float:
    """Return stop-loss multiplier from current price (e.g., 0.95 = -5%)."""
    if beta is None or not isinstance(beta, (int,float)) or math.isnan(beta):
        return float(floor)
    if beta < 0.8:
        return 0.95  # -5%
    if beta <= 1.2:
        return 0.93  # -7%
    return 0.90      # -10%


def portfolio_beta_weighted(df_holdings: pd.DataFrame, last_prices: Dict[str, float], infos: Dict[str, dict]) -> float:
    values, betas = [], []
    for _, r in df_holdings.iterrows():
        t = r["Ticker"]
        sh = float(r.get("Shares",0))
        p  = float(last_prices.get(t, np.nan))
        if math.isnan(p) or sh<=0:
            continue
        v = sh*p
        beta = infos.get(t,{}).get("beta")
        if beta is not None and isinstance(beta,(int,float)):
            values.append(v)
            betas.append(beta)
    if not values:
        return float('nan')
    w = np.array(values)/np.sum(values)
    return float(np.sum(w*np.array(betas)))


# --- Dividend Mode (Proposal #2) ---
def dividend_snapshot(df_holdings: pd.DataFrame, infos: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, r in df_holdings.iterrows():
        t = str(r["Ticker"]).upper()
        sh = float(r.get("Shares",0))
        info = infos.get(t, {})
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        yld  = info.get("dividendYield")  # e.g., 0.028 == 2.8%
        if not price:
            # Fallback price
            price, _ = fetch_last_prices(t)
        ann_div = sh * price * (yld or 0.0)
        rows.append({
            BK("Ticker","종목"): f"<b>{html.escape(t)}</b>",
            BK("Shares","수량"): f"{sh:.2f}",
            BK("Price","현재가"): f"{(price or 0):.2f}",
            BK("Fwd Yield","배당수익률"): f"{(yld*100.0 if yld else 0):.2f}%",
            BK("Est. Annual Dividends","연간 예상 배당금"): f"{ann_div:.2f}"
        })
    return pd.DataFrame(rows)


# --- Charts (Proposal #4) ---
def _img_tag_from_fig() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return f"<img style='max-width:100%;height:auto' src='data:image/png;base64,{img}'/>"


def chart_allocation_by_ticker(values: Dict[str,float]) -> str:
    if not values:
        return ""
    labels = list(values.keys())
    sizes  = list(values.values())
    plt.figure(figsize=(6,4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Allocation by Ticker | 종목별 비중")
    return _img_tag_from_fig()


def chart_cash_vs_equity(cash: float, equity: float) -> str:
    plt.figure(figsize=(5,4))
    labels = ["Equity | 주식", "Cash | 현금"]
    sizes  = [max(equity,0.0), max(cash,0.0)]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Cash vs Equity | 현금 대비 주식 비중")
    return _img_tag_from_fig()


def chart_pnl_by_ticker(pnls: Dict[str,float]) -> str:
    if not pnls:
        return ""
    labels = list(pnls.keys())
    vals   = list(pnls.values())
    plt.figure(figsize=(7,4))
    bars = plt.bar(labels, vals)
    for b, v in zip(bars, vals):
        b.set_color('green' if v>=0 else 'red')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("P&L by Ticker | 종목별 손익")
    return _img_tag_from_fig()


# --- News / GPT ---
POLICY_TOPICS = [
    "White House policy","President remarks","executive order","tariffs","defense budget","NATO",
    "healthcare policy","drug pricing","energy policy","OPEC","semiconductor subsidies","CHIPS Act",
    "AI regulation","crypto ETF flows","bitcoin inflows","ethereum ETF"
]

KEYWORD_TO_TICKERS = {
    "defense": ["LMT","RTX","NOC","GD","BA"],
    "pentagon": ["LMT","RTX","NOC","GD","BA"],
    "semiconductor": ["NVDA","AMD","QCOM","INTC","AVGO","TXN","SMH","SOXX"],
    "chip": ["NVDA","AMD","QCOM","INTC","AVGO","TXN","SMH","SOXX"],
    "tariff": ["AAPL","CAT","DE","GM","F"],
    "drug": ["LLY","PFE","MRK","BMY","JNJ","IBB"],
    "medicare": ["UNH","CI","HUM","CVS"],
    "energy": ["XOM","CVX","COP","XLE"],
    "opec": ["XOM","CVX","COP","XLE"],
    "ai": ["NVDA","MSFT","GOOGL","META","ADBE","CRM","NOW","ARKK"],
    "executive order": ["NVDA","MSFT","GOOGL","META","LMT","RTX"],
    "crypto": ["IBIT","FBTC","ARKB","BITO","BRRR","EETH","ETHE"],
    "bitcoin": ["IBIT","FBTC","ARKB","BITO","BRRR"],
    "ethereum": ["EETH","ETHE"]
}


def news_fetch(query: str, from_days=7, page_size=20) -> List[dict]:
    if not NEWS_API_KEY:
        return []
    try:
        from_date = (datetime.utcnow() - timedelta(days=from_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "apiKey": NEWS_API_KEY
        }
        r = requests.get(url, params=params, timeout=12)
        return r.json().get("articles", [])
    except Exception:
        return []


def gpt_summary(text: str, system_hint: str = "") -> str:
    if not _openai_client:
        return ""
    try:
        msgs = []
        if system_hint:
            msgs.append({"role":"system","content":system_hint})
        msgs.append({"role":"user","content":text})
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.3,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# --- Report Sections ---
def build_portfolio_tables(df_holdings: pd.DataFrame, settings: dict) -> Tuple[str, Dict[str,float], Dict[str,float]]:
    cash_usd = float(settings.get("CashUSD", 0) or 0)

    # fetch prices and compute P&L
    rows = []
    total_cost = 0.0
    total_value = 0.0
    last_prices = {}
    pnls_by_ticker = {}
    values_by_ticker = {}

    for _, r in df_holdings.iterrows():
        t = r["Ticker"]
        sh = float(r.get("Shares", 0))
        avg= float(r.get("AvgPrice", 0))
        last, prev = fetch_last_prices(t)
        if math.isnan(last):
            last = avg
        daily_pl = ((last - (prev if not math.isnan(prev) else last)) * sh) if sh>0 else 0.0
        cost = avg * sh
        value = last * sh
        pnl = value - cost
        rate = (pnl / cost * 100) if cost>0 else 0.0

        rows.append({
            BK("Ticker","종목"): f"<b>{html.escape(t)}</b>",
            BK("Shares","수량"): f"{sh:.2f}",
            BK("Last Price","현재가"): f"{last:.2f}",
            BK("Avg Price","평단"): f"{avg:.2f}",
            BK("Daily P/L","일일손익"): f"{daily_pl:+.2f}",
            BK("P/L","누적손익"): f"{pnl:+.2f}",
            BK("Return","수익률"): f"{rate:+.2f}%",
        })
        total_cost += cost
        total_value += value
        last_prices[t] = last
        pnls_by_ticker[t] = pnl
        values_by_ticker[t] = value

    df = pd.DataFrame(rows)
    summary_html = ""
    if not df.empty:
        summary_html += df.to_html(index=False, escape=False, border=1, justify='center')

    equity_total = total_value
    grand_total  = equity_total + cash_usd

    cash_ratio = (cash_usd / grand_total * 100) if grand_total>0 else 0
    eq_ratio   = 100 - cash_ratio

    totals_html = f"""
    <div style='margin-top:8px'>
      <p>💰 {BK('Cash Balance','현금 보유액')}: {cash_usd:,.2f} USD ({cash_usd*FX_USD_CAD:,.2f} CAD)</p>
      <p>💼 {BK('Equity Value','주식 평가금액')}: {equity_total:,.2f} USD</p>
      <p>📊 {BK('Total Value','총 평가금액')}: {grand_total:,.2f} USD ({grand_total*FX_USD_CAD:,.2f} CAD) — {BK('Cash Ratio','현금 비중')}: {cash_ratio:.2f}%</p>
    </div>
    """

    return summary_html + totals_html, values_by_ticker, pnls_by_ticker


def build_indicators_and_risk(df_holdings: pd.DataFrame) -> Tuple[str, Dict[str,dict]]:
    rows1, rows2 = [], []
    infos = {}
    last_prices = {}

    for _, r in df_holdings.iterrows():
        t = r["Ticker"]
        last, _ = fetch_last_prices(t)
        rsi, macd_h = compute_rsi_macd(t)
        info = fetch_info(t)
        price = last if not math.isnan(last) else info.get("currentPrice") or info.get("regularMarketPrice") or 0
        per = info.get("trailingPE")
        pbr = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        eps = info.get("trailingEps")
        fpe = info.get("forwardPE")
        beta= info.get("beta")
        sl_mult = stop_loss_by_beta(beta, float(os.getenv("RISK_STOP_FLOOR", "0.90")))
        sell_1 = price * 1.03
        sell_2 = price * 1.10
        stop   = price * sl_mult

        rows1.append({
            BK("Ticker","종목"): f"<b>{html.escape(t)}</b>",
            BK("RSI14","RSI14"): "-" if math.isnan(rsi) else f"{rsi:.2f}",
            BK("MACD Hist","MACD 히스토그램"): "-" if math.isnan(macd_h) else f"{macd_h:.2f}",
            BK("PER","PER"): "-" if per is None else f"{per:.2f}",
            BK("PBR","PBR"): "-" if pbr is None else f"{pbr:.2f}",
            BK("ROE","ROE"): "-" if roe is None else f"{roe*100:.2f}%",
            BK("EPS","EPS"): "-" if eps is None else f"{eps:.2f}",
            BK("Fwd PER","선행 PER"): "-" if fpe is None else f"{fpe:.2f}",
            BK("Beta","베타"): "-" if beta is None else f"{beta:.2f}",
        })

        rows2.append({
            BK("Ticker","종목"): f"<b>{html.escape(t)}</b>",
            BK("Take Profit 1","1차 매도"): f"${sell_1:.2f}",
            BK("Take Profit 2","2차 매도"): f"${sell_2:.2f}",
            BK("Stop Loss","손절"): f"${stop:.2f}",
        })
        infos[t] = info
        last_prices[t] = price

    html1 = pd.DataFrame(rows1).to_html(index=False, escape=False, border=1, justify='center') if rows1 else ""
    html2 = pd.DataFrame(rows2).to_html(index=False, escape=False, border=1, justify='center') if rows2 else ""

    return f"<h4>📊 {BK('Indicators by Ticker','종목별 판단 지표')}</h4>"+html1 + "<br>" + \
           f"<h4>📈 {BK('Trade Plan','종목별 매매 전략')}</h4>"+html2, infos


def build_dividend_section(df_holdings: pd.DataFrame, infos: Dict[str,dict]) -> str:
    df_div = dividend_snapshot(df_holdings, infos)
    if df_div.empty:
        return ""
    total = 0.0
    try:
        # parse floats from formatted strings
        total = df_div[BK("Est. Annual Dividends","연간 예상 배당금")].str.replace(',','').astype(float).sum()
    except Exception:
        pass
    return f"<h4>💵 {BK('Dividend Snapshot','배당 요약')}</h4>" + \
           df_div.to_html(index=False, escape=False, border=1, justify='center') + \
           f"<p><b>{BK('Estimated Annual Dividends Total','연간 예상 배당 합계')}</b>: {total:.2f} USD ({total*FX_USD_CAD:.2f} CAD)</p>"


# --- Indices & Market Outlook (reuse minimal) ---
INDEX_TICKERS = {
    "S&P 500":"^GSPC",
    "Nasdaq":"^IXIC",
    "Dow Jones":"^DJI",
    "VIX":"^VIX",
    "US 10Y":"^TNX",
    "Gold":"GC=F",
}


def indices_table() -> str:
    rows = []
    for name, sym in INDEX_TICKERS.items():
        try:
            hist = yf.Ticker(sym).history(period="2d")["Close"].dropna()
            if len(hist) < 1: continue
            last = float(hist.iloc[-1])
            prev = float(hist.iloc[-2]) if len(hist)>=2 else last
            chg = (last-prev)/prev*100 if prev else 0
            rows.append({
                BK("Index","지수"): name,
                BK("Last","현재"): f"{last:,.2f}",
                BK("Chg%","변동률"): f"{chg:+.2f}%",
            })
        except Exception as e:
            rows.append({BK("Index","지수"):name, BK("Last","현재"):'N/A', BK("Chg%","변동률"):'N/A'})
    if not rows:
        return ""
    return pd.DataFrame(rows).to_html(index=False, escape=False, border=1, justify='center')


def news_summary_for_portfolio(tickers: List[str]) -> str:
    if not NEWS_API_KEY:
        return ""
    html_parts = [f"<h4>📰 {BK('News (7d) for Holdings','보유 종목 최근 7일 뉴스')}</h4>"]
    for t in tickers:
        arts = news_fetch(t, from_days=7, page_size=6)
        if not arts:
            continue
        block = [f"<h5>📌 <b>{html.escape(t)}</b></h5>"]
        text_blob = []
        for i, a in enumerate(arts[:3], 1):
            title = a.get("title") or ""
            url   = a.get("url") or "#"
            published = (a.get("publishedAt") or "")[:10]
            desc = a.get("description") or ""
            block.append(f"<p><b>{i}. <a href='{html.escape(url)}'>{html.escape(title)}</a></b> <span style='color:gray;font-size:12px;'>({published})</span></p>")
            if desc:
                block.append(f"<p style='margin-left:18px;color:#555'>{html.escape(desc)}</p>")
            text_blob.append(f"[{i}] {title} ({published}) - {desc}")
        # GPT short summary
        if _openai_client and text_blob:
            summ = gpt_summary(
                "\n".join(text_blob)+"\n\n"+
                "뉴스 요약을 한국어로 3~4줄로 작성하고, 투자 관점으로 한 줄 의견을 덧붙여줘.")
            if summ:
                lines = ''.join([f"<p>{html.escape(l)}</p>" for l in summ.splitlines() if l.strip()])
                block.append(f"<div style='background:#eef;padding:8px;border-radius:8px'>{lines}</div>")
        html_parts.append(''.join(block))
    return ''.join(html_parts)


def policy_focus_section() -> str:
    if not NEWS_API_KEY:
        return ""
    # Aggregate articles across topics, map keywords→tickers counts
    scores = {}
    buckets = {}
    for q in POLICY_TOPICS:
        arts = news_fetch(q, from_days=7, page_size=25)
        for a in arts:
            content = (a.get("title","") + " " + a.get("description",""))
            content_l = content.lower()
            for key, tks in KEYWORD_TO_TICKERS.items():
                if key in content_l:
                    for tk in tks:
                        scores[tk] = scores.get(tk,0.0) + 1.0
                        buckets.setdefault(tk,[]).append(a.get("title") or a.get("description") or "")
    if not scores:
        return ""
    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:12]
    rows = []
    for t, _ in ranked:
        snippets = buckets.get(t, [])[:6]
        summ = gpt_summary(
            "- " + "\n- ".join(snippets) + "\n\n" +
            f"티커 {t} 관련 최근 정책/정부 이슈 헤드라인을 2문장으로 한국어 요약하고, Sentiment(긍정/부정/중립)를 한 단어로 알려줘.") if _openai_client and snippets else ""
        icon = "⚫"
        if '긍정' in summ: icon = '🟢'
        elif '부정' in summ: icon = '🔴'
        rows.append({
            BK("Ticker","종목"): f"<b>{html.escape(t)}</b>",
            BK("Policy-driven view","정책 관점 요약"): html.escape(summ) if summ else "최근 정책 키워드 노출 증가",
            BK("Sentiment","심리"): icon
        })
    df = pd.DataFrame(rows)
    return df.to_html(index=False, escape=False, border=1, justify='center')


# --- Email ---
def send_email_html(subject: str, html_body: str):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("⚠️ Missing email settings — skip sending")
        print(subject)
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html", _charset='utf-8'))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ Email sent to:", EMAIL_RECEIVER)
    except Exception as e:
        print("❌ Email send failed:", e)


# --- Main Report Builder ---
def build_report_html() -> str:
    df_hold, df_watch, settings = load_holdings_watchlist_settings()

    # 1) Portfolio tables
    port_html, values_by_ticker, pnls_by_ticker = build_portfolio_tables(df_hold, settings)

    # 2) Indicators + Risk
    indicators_html, infos = build_indicators_and_risk(df_hold)

    # 3) Dividend mode
    div_html = build_dividend_section(df_hold, infos)

    # 4) Charts
    alloc_img = chart_allocation_by_ticker({k: v for k,v in values_by_ticker.items() if v>0})
    cash_usd = float(settings.get("CashUSD", 0) or 0)
    equity_val = sum(values_by_ticker.values())
    cash_vs_eq_img = chart_cash_vs_equity(cash_usd, equity_val)
    pnl_img = chart_pnl_by_ticker(pnls_by_ticker)

    # 5) Indices + news + policy
    idx_html = indices_table()
    news_html= news_summary_for_portfolio(df_hold["Ticker"].tolist()) if not df_hold.empty else ""
    policy_html = policy_focus_section()

    # 6) GPT: Overall opinion (optional)
    overall_op = ""
    if _openai_client:
        brief = ("포트폴리오 현황 요약:\n" +
                 f"현금비중 약 { (cash_usd/(cash_usd+equity_val)*100 if (cash_usd+equity_val)>0 else 0):.1f}%\n" +
                 "지수 표를 참고하여 단기/중기 의견을 4줄 이내 한국어로 제시하고, 포트폴리오 관점의 리밸런싱 제안 1줄을 덧붙여줘.")
        overall_op = gpt_summary(brief, system_hint="You are a concise investment assistant. Use neutral tone.")
        if overall_op:
            overall_op = ''.join([f"<p>{html.escape(l)}</p>" for l in overall_op.splitlines() if l.strip()])

    # Assemble HTML
    h = []
    h.append(f"<h2 style='text-align:center'>📊 {BK('Daily Portfolio Report','일일 포트폴리오 리포트')} - {TODAY_STR}</h2>")
    h.append("<hr>")

    h.append(f"<h3>💼 {BK('Portfolio Overview','보유자산 요약')}</h3>")
    h.append(port_html)

    h.append("<br>")
    h.append("<div style='display:flex;gap:16px;flex-wrap:wrap'>")
    h.append(f"<div style='flex:1;min-width:280px'>{alloc_img}</div>")
    h.append(f"<div style='flex:1;min-width:280px'>{cash_vs_eq_img}</div>")
    h.append(f"<div style='flex:1;min-width:280px'>{pnl_img}</div>")
    h.append("</div>")

    h.append("<br>")
    h.append(indicators_html)

    if div_html:
        h.append("<br>")
        h.append(div_html)

    if idx_html:
        h.append("<h3>📈 "+BK("Major Indices (Change)","주요 지수 (변동)")+"</h3>")
        h.append(idx_html)

    if news_html:
        h.append("<br>")
        h.append(news_html)

    if policy_html:
        h.append("<h3>🏛️ "+BK("Policy Focus (U.S.)","미 정부 정책 포커스")+"</h3>")
        h.append(policy_html)

    if overall_op:
        h.append("<h3>💡 "+BK("GPT Overall View","GPT 종합 의견")+"</h3>")
        h.append(f"<div style='background:#f7f7ff;padding:10px;border-radius:8px'>{overall_op}</div>")

    return f"<html><head><meta charset='utf-8'><style>table {{ border-collapse: collapse; font-size: 14px; }} th,td {{ padding: 6px 10px; text-align: center; }} th {{ background:#f2f2f2; }}</style></head><body>{''.join(h)}</body></html>"


def main():
    html_doc = build_report_html()
    out_dir = os.getenv("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"portfolio_gsheet_policy_report_{TODAY_STR}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("📄 Report saved:", html_path)
    send_email_html(f"📊 Portfolio Report - {TODAY_STR}", html_doc)


if __name__ == "__main__":
    main()
