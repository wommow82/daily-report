
#!/usr/bin/env python3
# coding: utf-8

"""
Daily Recommendation Report with ETF & Crypto ETF (v2)
-----------------------------------------------------
- STOCK/ETF 분리 출력 (타이밍/재무)
- 섹터/카테고리 라벨 추가 (소비재/반도체/에너지/금융/테크/헬스케어/코인 등)
- 모든 수치 소수점 2자리 표기
- 최근 뉴스 기반 추천 섹션(종목 | 설명) 추가
- 이메일 발송 유지 (SMTP)

환경변수(Secrets):
- EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
- NEWS_API_KEY (선택, 뉴스 기반 추천 활성화)
- UNIVERSE_TICKERS (선택, 콤마 구분 사용자 유니버스)
"""

import os, smtplib, math, html
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
# Universe & Classification
# ------------------------
DEFAULT_STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","MA",
    "UNH","HD","PG","XOM","CVX","LLY","JNJ","MRK","KO","PEP","DIS",
    "INTC","AMD","QCOM","AVGO","TXN","IBM","ORCL","ADBE","CRM","NOW",
    "PFE","BMY","ABT","TMO","DHR","ISRG","GE","CAT","DE","NKE","WMT","COST",
    "BAC","GS","MS","WFC","C","NFLX","SHOP","PYPL","NEE","DUK","SO","LMT","RTX","NOC","BA","GD"
]
DEFAULT_ETFS = [
    # Broad market
    "SPY","VOO","IVV","VTI","QQQ","DIA","IWM",
    # Sector ETFs
    "XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC",
    # Thematic / Industry
    "ARKK","ARKW","SMH","SOXX","IBB","TAN","HACK","CIBR"
]
CRYPTO_ETFS = [
    # Bitcoin ETFs
    "BITO","IBIT","FBTC","ARKB","BRRR",
    # Ethereum ETFs
    "EETH","ETHE"
]

# ETF group labels
ETF_GROUPS = {
    "Broad Market": {"SPY","VOO","IVV","VTI","QQQ","DIA","IWM"},
    "Sector ETF": {"XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC"},
    "Thematic ETF": {"ARKK","ARKW","SMH","SOXX","IBB","TAN","HACK","CIBR"},
    "Crypto ETF": set(CRYPTO_ETFS),
}

UNIVERSE = [t.strip().upper() for t in os.getenv("UNIVERSE_TICKERS","").split(",") if t.strip()] \
            or DEFAULT_STOCKS + DEFAULT_ETFS + CRYPTO_ETFS

def is_etf(ticker: str) -> bool:
    return ticker in (DEFAULT_ETFS + CRYPTO_ETFS)

def classify_stock_category(sector: str, industry: str, ticker: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    t = ticker.upper()
    # Explicit keywords
    if "semiconductor" in i or t in {"NVDA","AMD","INTC","QCOM","AVGO","TXN","TSM"}:
        return "반도체"
    if "software" in i or t in {"MSFT","ADBE","CRM","NOW","ORCL","IBM","GOOGL"}:
        return "테크"
    if "internet" in i or t in {"META","GOOGL","AMZN","NFLX","SHOP"}:
        return "테크"
    if "aerospace" in i or "defense" in i or t in {"LMT","RTX","NOC","BA","GD"}:
        return "방위산업"
    if "oil" in i or "gas" in i or s == "energy" or t in {"XOM","CVX","COP"}:
        return "에너지"
    if s in {"healthcare"} or t in {"LLY","PFE","JNJ","MRK","ABT","TMO","DHR","ISRG","BMY"}:
        return "헬스케어"
    if s in {"financial services","financial","banks"} or t in {"JPM","BAC","GS","MS","WFC","C"}:
        return "금융"
    if s in {"consumer cyclical","consumer defensive"} or t in {"NKE","WMT","HD","COST","KO","PEP","DIS"}:
        return "소비재"
    if s in {"industrials"} or t in {"GE","CAT","DE"}:
        return "산업재"
    if s in {"utilities"} or t in {"NEE","DUK","SO"}:
        return "유틸리티"
    if s in {"real estate"}:
        return "리츠/부동산"
    return "기타"

def classify_etf_group(ticker: str) -> str:
    u = ticker.upper()
    for g, tickers in ETF_GROUPS.items():
        if u in tickers:
            return g
    if u in CRYPTO_ETFS or "BTC" in u or "ETH" in u:
        return "Crypto ETF"
    if u in {"SMH","SOXX"}:
        return "Thematic ETF"
    if u in {"XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC"}:
        return "Sector ETF"
    return "ETF"

# ------------------------
# News-based Recommendation
# ------------------------
def fetch_news(query, from_days=2, page_size=25):
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
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []

def build_news_recos():
    topics = [
        "White House policy","President remarks","executive order","tariffs",
        "defense budget","NATO","healthcare policy","drug pricing",
        "energy policy","OPEC","semiconductor subsidies","CHIPS Act",
        "AI regulation","crypto ETF flows","bitcoin inflows","ethereum ETF"
    ]
    keyword_map = {
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

    articles = []
    for q in topics:
        arts = fetch_news(q, from_days=2, page_size=20)
        for a in arts:
            a["_topic"] = q
        articles.extend(arts)

    scores = {}
    reasons = {}
    now = datetime.utcnow()

    for a in articles:
        title = (a.get("title") or "")
        desc  = (a.get("description") or "")
        content = f"{title} {desc}".lower()
        try:
            published = datetime.fromisoformat(a.get("publishedAt").replace("Z","+00:00"))
            hours = max((now - published).total_seconds() / 3600.0, 1.0)
            recency_w = 1.0 / math.log(hours + 2.0)
        except Exception:
            recency_w = 1.0

        for key, tickers in keyword_map.items():
            if key in content:
                for t in tickers:
                    scores[t] = scores.get(t, 0.0) + recency_w
                    if t not in reasons:
                        topic = a.get("_topic","")
                        reasons[t] = f"{topic}: {title[:140]}"

    rows = [{"Ticker":k, "Score":v, "Reason":reasons.get(k,"")} for k,v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["종목","설명"])
    df = df.head(10).copy()
    df["종목"] = df["Ticker"]
    df["설명"] = df["Reason"].apply(lambda s: html.escape(s))
    return df[["종목","설명"]]

# ------------------------
# Data Fetch & Metrics
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
    out["SMA50"] = float(close.rolling(50).mean().iloc[-1]) if len(close)>=50 else None
    out["SMA200"] = float(close.rolling(200).mean().iloc[-1]) if len(close)>=200 else None
    delta = close.diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    out["RSI14"] = float(rsi.dropna().iloc[-1]) if rsi.dropna().size else None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["MACD_Hist"] = float(hist.dropna().iloc[-1]) if hist.dropna().size else None
    out["High52w"] = float(close.max())
    out["Low52w"]  = float(close.min())
    return out

def timing_score(tech):
    if not tech: return 0.0
    p,s50,s200,rsi,macdh,h52 = tech.get("Price"), tech.get("SMA50"), tech.get("SMA200"), tech.get("RSI14"), tech.get("MACD_Hist"), tech.get("High52w")
    if None in (p,s50,s200,rsi,macdh,h52): return 0.0
    score = 0.0
    if p > s200: score += 1
    if p > s50:  score += 1
    if 35 <= rsi <= 60: score += 1
    if macdh > 0: score += 1
    if p >= 0.85*h52: score += 1
    return float(score)

def fetch_fundamentals(ticker):
    info = {}
    try:
        y = yf.Ticker(ticker)
        i = y.info or {}
        per = i.get("trailingPE") or i.get("forwardPE")
        pbr = i.get("priceToBook")
        roe = i.get("returnOnEquity")
        eps = i.get("trailingEps") or i.get("forwardEps")
        info.update({
            "PER": float(per) if per is not None else None,
            "PBR": float(pbr) if pbr is not None else None,
            "ROE": float(roe) if roe is not None else None,
            "EPS": float(eps) if eps is not None else None,
        })
        sector = i.get("sector") or ""
        industry = i.get("industry") or ""
        info["Sector"] = sector
        info["Industry"] = industry
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
# Formatting Helpers
# ------------------------
def format_two_decimals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return df

def to_html_table(df, title):
    if df is None or df.empty:
        return f"<h3>{title}</h3><p style='color:gray;'>조건 충족 없음</p>"
    styled = df.copy()
    html_table = styled.to_html(index=False, escape=False, border=1, justify="center")
    return f"<h3>{title}</h3><div style='overflow:auto'>{html_table}</div>"

def bold_ticker(t):
    return f"<b>{html.escape(str(t))}</b>"

# ------------------------
# Build Report
# ------------------------
def build_report():
    timing_rows_stock, timing_rows_etf = [], []
    fund_rows_stock, fund_rows_etf = [], []

    for t in UNIVERSE:
        tech = compute_technicals(fetch_prices(t, "1y", "1d"))
        base = {"Ticker": t, **tech}
        f = fetch_fundamentals(t)
        ts = timing_score(tech)

        if is_etf(t):
            base["Group"] = classify_etf_group(t)
        else:
            base["Category"] = classify_stock_category(f.get("Sector"), f.get("Industry"), t)

        # Timing
        if ts >= 4:
            base_ts = {**base, "TimingScore": ts}
            if is_etf(t):
                timing_rows_etf.append(base_ts)
            else:
                timing_rows_stock.append(base_ts)

        # Fundamentals (ETF는 대개 미통과)
        if (not is_etf(t)) and fundamental_pass(f):
            fund_rows_stock.append({
                "Ticker": t,
                "Category": classify_stock_category(f.get("Sector"), f.get("Industry"), t),
                "PER": f.get("PER"),
                "PBR": f.get("PBR"),
                "ROE": f.get("ROE"),
                "EPS": f.get("EPS"),
            })

    # DataFrames
    df_time_stock = pd.DataFrame(timing_rows_stock).sort_values(["TimingScore","Price"], ascending=[False,False]) if timing_rows_stock else pd.DataFrame()
    df_time_etf   = pd.DataFrame(timing_rows_etf).sort_values(["TimingScore","Price"], ascending=[False,False]) if timing_rows_etf else pd.DataFrame()
    df_fund_stock = pd.DataFrame(fund_rows_stock).sort_values(["ROE","PER"], ascending=[False,True]) if fund_rows_stock else pd.DataFrame()

    # Reorder & format
    def reorder(cols, df):
        return df[[c for c in cols if c in df.columns]] if not df.empty else df

    df_time_stock = reorder(["Ticker","Category","Price","SMA50","SMA200","RSI14","MACD_Hist","High52w","TimingScore"], df_time_stock)
    df_time_etf   = reorder(["Ticker","Group","Price","SMA50","SMA200","RSI14","MACD_Hist","High52w","TimingScore"], df_time_etf)
    df_fund_stock = reorder(["Ticker","Category","PER","PBR","ROE","EPS"], df_fund_stock)

    # Two decimals
    df_time_stock = format_two_decimals(df_time_stock)
    df_time_etf   = format_two_decimals(df_time_etf)
    df_fund_stock = format_two_decimals(df_fund_stock)

    # Bold ticker
    for df in [df_time_stock, df_time_etf, df_fund_stock]:
        if not df.empty and "Ticker" in df.columns:
            df["Ticker"] = df["Ticker"].apply(bold_ticker)

    # News Recommendations
    df_news = build_news_recos() if NEWS_API_KEY else pd.DataFrame(columns=["종목","설명"])

    return df_news, df_time_stock, df_time_etf, df_fund_stock

# ------------------------
# Email
# ------------------------
def send_email_html(subject, html_body):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("⚠️ 메일 설정 없음 → 메일 전송 생략")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_SENDER, EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ 메일 발송 완료:", EMAIL_RECEIVER)
    except Exception as e:
        print(f"❌ 메일 전송 실패: {e}")

# ------------------------
# Main
# ------------------------
def main():
    df_news, df_time_stock, df_time_etf, df_fund_stock = build_report()

    sections = []
    sections.append("<h2 style='text-align:center'>📌 Daily Recommendation Report (Stocks & ETFs) - {}</h2>".format(REPORT_DATE))
    sections.append("<p style='text-align:center;color:#666'>분리: STOCK vs ETF ｜ 카테고리 라벨 ｜ 모든 수치 소수점 2자리</p>")
    sections.append("<hr>")

    if df_news is not None and not df_news.empty:
        news_html = df_news.to_html(index=False, escape=False, border=1, justify="center")
    else:
        news_html = "<p style='color:gray;'>최근 뉴스 기반 추천 결과가 없습니다.</p>"
    sections.append(f"<h3>① 최근 뉴스 기반 추천 (종목 | 설명)</h3><div style='overflow:auto'>{news_html}</div>")

    sections.append("<br>")
    sections.append(to_html_table(df_time_stock.head(40), "② 매수 타이밍 양호 - STOCK"))
    sections.append(to_html_table(df_time_etf.head(40), "③ 매수 타이밍 양호 - ETF"))

    sections.append("<br>")
    sections.append(to_html_table(df_fund_stock.head(40), "④ 재무 우량 - STOCK"))
    sections.append("<p style='color:#888;'>참고: ETF는 재무 필드가 비어있는 경우가 많아 재무 우량 표에 포함되지 않을 수 있습니다.</p>")

    html_doc = f"""
    <html><head>
    <meta charset="utf-8">
    <style>
      table {{ border-collapse: collapse; font-size: 14px; }}
      th, td {{ padding: 6px 10px; text-align: center; }}
      th {{ background: #f2f2f2; }}
      h3 {{ margin-top: 16px; }}
    </style>
    </head><body>
    {''.join(sections)}
    </body></html>
    """

    out_dir = os.getenv("OUT_DIR",".")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"daily_reco_report_with_etf_{REPORT_DATE}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("📄 Report saved:", html_path)

    send_email_html(f"Daily Stocks/ETFs Recommendation - {REPORT_DATE}", html_doc)

if __name__ == "__main__":
    main()
