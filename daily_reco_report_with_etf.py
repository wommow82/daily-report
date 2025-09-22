
#!/usr/bin/env python3
# coding: utf-8

"""
Daily Recommendation Report with ETF & Crypto ETF
-------------------------------------------------
This version extends the daily stock recommendation report to include:
- Stocks (default S&P100-like universe)
- Major ETFs (broad market, sector ETFs, thematic ETFs)
- Crypto-related ETFs (Bitcoin & Ethereum ETFs)

Outputs HTML and CSV. Optionally emails the HTML if EMAIL_* env vars are set.
"""

import os, smtplib, math
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------
# Config
# ------------------------
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
REPORT_DATE    = datetime.now().strftime("%Y-%m-%d")

# ------------------------
# Universe
# ------------------------
DEFAULT_STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","MA","UNH","HD","PG","XOM","CVX","LLY","JNJ","MRK"
]
DEFAULT_ETFS = [
    # Broad market
    "SPY","VOO","IVV","VTI","QQQ","DIA","IWM",
    # Sector ETFs
    "XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC",
    # Thematic ETFs
    "ARKK","ARKW","SMH","SOXX","IBB","TAN"
]
CRYPTO_ETFS = [
    # Bitcoin ETFs
    "BITO","IBIT","FBTC","ARKB","BRRR",
    # Ethereum ETFs
    "EETH","ETHE"
]
UNIVERSE = [t.strip().upper() for t in os.getenv("UNIVERSE_TICKERS","").split(",") if t.strip()] \
            or DEFAULT_STOCKS + DEFAULT_ETFS + CRYPTO_ETFS

# ------------------------
# Helpers
# ------------------------
def fetch_prices(ticker, period="1y", interval="1d"):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def compute_technicals(df):
    out = {}
    if df.empty or "Close" not in df:
        return out
    close = df["Close"].copy()
    out["Price"] = float(close.iloc[-1])
    # SMAs
    out["SMA50"] = close.rolling(50).mean().iloc[-1] if len(close)>=50 else None
    out["SMA200"] = close.rolling(200).mean().iloc[-1] if len(close)>=200 else None
    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    out["RSI14"] = float(rsi.dropna().iloc[-1]) if rsi.dropna().size else None
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["MACD_Hist"] = float(hist.dropna().iloc[-1]) if hist.dropna().size else None
    # 52w
    out["High52w"] = float(close.max())
    out["Low52w"]  = float(close.min())
    return out

def timing_score(tech):
    if not tech: return 0.0
    score = 0.0
    p,s50,s200,rsi,macdh,h52 = tech.get("Price"), tech.get("SMA50"), tech.get("SMA200"), tech.get("RSI14"), tech.get("MACD_Hist"), tech.get("High52w")
    if None in (p,s50,s200,rsi,macdh,h52): return 0.0
    if p > s200: score += 1
    if p > s50:  score += 1
    if 35 <= rsi <= 60: score += 1
    if macdh > 0: score += 1
    if p >= 0.85*h52: score += 1
    return score

def fetch_fundamentals(ticker):
    info = {}
    try:
        y = yf.Ticker(ticker)
        i = y.info or {}
        per = i.get("trailingPE") or i.get("forwardPE")
        pbr = i.get("priceToBook")
        roe = i.get("returnOnEquity")
        eps = i.get("trailingEps") or i.get("forwardEps")
        total_debt  = i.get("totalDebt")
        total_assets = i.get("totalAssets")
        debt_ratio = (total_debt/total_assets) if (total_debt and total_assets) else None
        rev_growth = i.get("revenueGrowth")
        info.update({
            "PER": float(per) if per else None,
            "PBR": float(pbr) if pbr else None,
            "ROE": float(roe) if roe else None,
            "EPS": float(eps) if eps else None,
            "DebtRatio": float(debt_ratio) if debt_ratio else None,
            "RevGrowth": float(rev_growth) if rev_growth else None,
        })
    except Exception:
        pass
    return info

def fundamental_pass(f):
    if not f: return False
    conditions = [
        (f.get("PER") is not None and 5 <= f["PER"] <= 40),
        (f.get("PBR") is not None and f["PBR"] < 10),
        (f.get("ROE") is not None and f["ROE"] > 0.08),
        (f.get("EPS") is not None and f["EPS"] > 0),
    ]
    return all(conditions)

# ------------------------
# Report Building
# ------------------------
def build_report():
    timing_rows = []
    fund_rows   = []
    for t in UNIVERSE:
        df = fetch_prices(t, "1y", "1d")
        tech = compute_technicals(df)
        f = fetch_fundamentals(t)
        ts = timing_score(tech)
        if ts >= 4:
            timing_rows.append({"Ticker":t, **tech, "TimingScore":ts})
        if fundamental_pass(f):
            fund_rows.append({"Ticker":t, **f})
    return pd.DataFrame(timing_rows), pd.DataFrame(fund_rows)

def to_html_table(df, title):
    if df is None or df.empty:
        return f"<h3>{title}</h3><p style='color:gray;'>조건 충족 없음</p>"
    return f"<h3>{title}</h3>" + df.to_html(index=False, border=1, justify="center")

def send_email_html(subject, html_body):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("메일 설정 없음 → 메일 전송 생략")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_SENDER, EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("메일 발송 완료")
    except Exception as e:
        print(f"메일 전송 실패: {e}")

def main():
    df_timing, df_fund = build_report()
    html = f"""
    <html><body>
    <h2 style='text-align:center;'>📌 Daily Recommendation Report with ETFs ({REPORT_DATE})</h2>
    <hr>
    {to_html_table(df_timing.head(30), "① 매수 타이밍 양호 종목/ETF (상위 30)")}
    <br>
    {to_html_table(df_fund.head(30), "② 재무 우량 종목/ETF (상위 30)")}
    <hr>
    <p style='color:#888;'>포함된 자산군: 개별주식 + 주요 ETF + 코인 ETF</p>
    </body></html>
    """
    out_dir = os.getenv("OUT_DIR",".")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"daily_reco_report_with_etf_{REPORT_DATE}.html")
    with open(html_path,"w",encoding="utf-8") as f:
        f.write(html)
    send_email_html(f"Daily Stock/ETF/CryptoETF Recommendation - {REPORT_DATE}", html)
    print("✅ Report ready:", html_path)

if __name__ == "__main__":
    main()
