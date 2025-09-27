#!/usr/bin/env python3
# coding: utf-8

import os, smtplib, html, math
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        _openai_client = None
except Exception:
    _openai_client = None

NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
REPORT_DATE    = datetime.now().strftime("%Y-%m-%d")

DEFAULT_STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","MA","UNH","HD","PG","XOM","CVX",
    "LLY","JNJ","MRK","KO","PEP","DIS","INTC","AMD","QCOM","AVGO","TXN","IBM","ORCL","ADBE","CRM",
    "NOW","PFE","BMY","ABT","TMO","DHR","ISRG","GE","CAT","DE","NKE","WMT","COST","BAC","GS","MS",
    "WFC","C","NFLX","SHOP","PYPL","NEE","DUK","SO","LMT","RTX","NOC","BA","GD"
]
DEFAULT_ETFS = [
    "SPY","VOO","IVV","VTI","QQQ","DIA","IWM",
    "XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC",
    "ARKK","ARKW","SMH","SOXX","IBB","TAN","HACK","CIBR"
]
CRYPTO_ETFS = ["BITO","IBIT","FBTC","ARKB","BRRR","EETH","ETHE"]

UNIVERSE = [t.strip().upper() for t in os.getenv("UNIVERSE_TICKERS","").split(",") if t.strip()] \
            or DEFAULT_STOCKS + DEFAULT_ETFS + CRYPTO_ETFS

ETF_GROUPS = {
    "Broad Market": {"SPY","VOO","IVV","VTI","QQQ","DIA","IWM"},
    "Sector ETF": {"XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC"},
    "Thematic ETF": {"ARKK","ARKW","SMH","SOXX","IBB","TAN","HACK","CIBR"},
    "Crypto ETF": set(CRYPTO_ETFS),
}

DOW30 = {"AAPL","MSFT","AMZN","GS","JPM","V","MA","NKE","DIS","IBM","INTC","CAT","BA","HD","WMT","UNH","CVX","JNJ","KO","PG","MCD","TRV","MRK","AXP","CSCO","CRM","VZ","WBA","MMM","DOW"}
NASDAQ100_SAMPLE = {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","PEP","COST","NFLX","AVGO","ADBE","CSCO","AMD","INTC","QCOM","AMAT","PYPL","BKNG","MRVL","GILD","CHTR","VRTX","REGN","LRCX","MU","ADI","ABNB","MELI","PANW","FTNT"}
SP500_SAMPLE = set(DEFAULT_STOCKS)

def is_etf(ticker: str) -> bool:
    return ticker in (DEFAULT_ETFS + CRYPTO_ETFS)

def classify_stock_category(sector: str, industry: str, ticker: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    t = ticker.upper()
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
    return "ETF"

TOPIC_QUERIES = [
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

def fetch_news(query, from_days=7, page_size=25):
    api = NEWS_API_KEY
    if not api:
        return []
    try:
        from_date = (datetime.utcnow() - timedelta(days=from_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "from": from_date, "sortBy": "publishedAt", "language": "en",
                  "pageSize": page_size, "apiKey": api}
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []

def gpt_summarize_reason(ticker, snippets):
    if not OPENAI_API_KEY or not snippets:
        return f"{ticker}: 최근 7일 정책/섹터 뉴스 이슈로 관심."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "당신은 투자 애널리스트입니다. 아래 헤드라인/요약을 바탕으로 "
            f"티커 {ticker}에 대한 투자 관점 한글 요약을 1~2문장으로 작성하세요. "
            "핵심 트리거와 기대/리스크를 간결하게. 과도한 확정적 표현 금지.\n\n- "
            + "\n- ".join(snippets[:6])
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"{ticker}: 최근 7일 관련 뉴스 모니터링 결과 관심 증가."

def build_news_recos():
    articles = []
    for q in TOPIC_QUERIES:
        arts = fetch_news(q, from_days=7, page_size=25)
        for a in arts:
            a["_topic"] = q
        articles.extend(arts)

    scores, buckets = {}, {}
    for a in articles:
        title = a.get("title") or ""
        desc  = a.get("description") or ""
        content = (title + " " + desc).lower()
        snippet = title if title else desc
        for key, tickers in KEYWORD_TO_TICKERS.items():
            if key in content:
                for t in tickers:
                    scores[t] = scores.get(t, 0.0) + 1.0
                    buckets.setdefault(t, []).append(snippet)

    if not scores:
        return pd.DataFrame(columns=["종목","설명"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    rows = []
    for t,_ in ranked:
        reason = gpt_summarize_reason(t, buckets.get(t, []) )
        rows.append({"종목": t, "설명": html.escape(reason)})
    return pd.DataFrame(rows)

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
    rs = (gain.rolling(14).mean()) / (loss.rolling(14).mean())
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
    last7 = close.tail(7)
    if len(last7) >= 3:
        change = (last7.iloc[-1] - last7.iloc[0]) / last7.iloc[0]
        out["Trend30d"] = "▲" if change > 0.02 else ("▼" if change < -0.02 else "▬")
    else:
        out["Trend30d"] = "▬"
    return out

def timing_score(tech):
    if not tech: return 0.0
    p,s50,s200,rsi,macdh,h52,l52 = tech.get("Price"), tech.get("SMA50"), tech.get("SMA200"), tech.get("RSI14"), tech.get("MACD_Hist"), tech.get("High52w"), tech.get("Low52w")
    if None in (p,s50,s200,rsi,macdh,h52,l52): return 0.0
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
        info["Sector"] = i.get("sector") or ""
        info["Industry"] = i.get("industry") or ""
    except Exception:
        pass
    return info

def near_bottom_candidates(universe):
    rows = []
    for t in universe:
        if is_etf(t):
            continue
        df = fetch_prices(t, "1y", "1d")
        tech = compute_technicals(df)
        if not tech:
            continue
        price, low = tech.get("Price"), tech.get("Low52w")
        if not price or not low or low <= 0:
            continue
        dist = (price - low) / low
        if dist <= 0.03 or (dist <= 0.06 and tech.get("Trend30d") == "▼"):
            info = fetch_fundamentals(t)
            cat = classify_stock_category(info.get("Sector"), info.get("Industry"), t)
            idx_tag = []
            if t in DOW30: idx_tag.append("DOW30")
            if t in NASDAQ100_SAMPLE: idx_tag.append("NASDAQ100")
            if t in SP500_SAMPLE: idx_tag.append("S&P500")
            if not idx_tag:
                continue
            rows.append({
                "Ticker": t,
                "Index": ",".join(idx_tag),
                "Category": cat,
                "Price": tech.get("Price"),
                "Low52w": tech.get("Low52w"),
                "DistFromLow(%)": dist*100.0,
                "Trend30d": tech.get("Trend30d")
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["DistFromLow(%)","Price"], ascending=[True,True]).head(40)
    for col in ["Price","Low52w","DistFromLow(%)"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
    df["Ticker"] = df["Ticker"].apply(lambda t: f"<b>{html.escape(t)}</b>")
    return df

def build_report():
    timing_rows_stock, timing_rows_etf = [], []
    for t in UNIVERSE:
        df = fetch_prices(t, "1y", "1d")
        tech = compute_technicals(df)
        if not tech: continue
        base = {"Ticker": t, **tech}
        finfo = fetch_fundamentals(t)
        if is_etf(t):
            base["Group"] = classify_etf_group(t)
        else:
            base["Category"] = classify_stock_category(finfo.get("Sector"), finfo.get("Industry"), t)
        ts = timing_score(tech)
        if ts >= 4:
            base["TimingScore"] = ts
            if is_etf(t):
                timing_rows_etf.append(base)
            else:
                timing_rows_stock.append(base)

    df_time_stock = pd.DataFrame(timing_rows_stock)
    df_time_etf   = pd.DataFrame(timing_rows_etf)

    def reorder(cols, df):
        return df[[c for c in cols if c in df.columns]] if not df.empty else df

    df_time_stock = reorder(["Ticker","Category","Trend30d","Price","SMA50","SMA200","RSI14","MACD_Hist","High52w","TimingScore"], df_time_stock)
    df_time_etf   = reorder(["Ticker","Group","Trend30d","Price","SMA50","SMA200","RSI14","MACD_Hist","High52w","TimingScore"], df_time_etf)

    def fmt2(df):
        if df is None or df.empty: return df
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: f"{x:.2f}")
        df["Ticker"] = df["Ticker"].apply(lambda t: f"<b>{html.escape(str(t))}</b>")
        return df

    df_time_stock = fmt2(df_time_stock).sort_values(["TimingScore","Price"], ascending=[False,False]).head(40)
    df_time_etf   = fmt2(df_time_etf).sort_values(["TimingScore","Price"], ascending=[False,False]).head(40)

    df_news = build_news_recos() if NEWS_API_KEY else pd.DataFrame(columns=["종목","설명"])
    df_bottom = near_bottom_candidates(set(UNIVERSE))

    return df_news, df_time_stock, df_time_etf, df_bottom

def to_html_table(df, title):
    if df is None or df.empty:
        return f"<h3>{title}</h3><p style='color:gray;'>조건 충족 없음</p>"
    return f"<h3>{title}</h3>" + df.to_html(index=False, escape=False, border=1, justify='center')

def send_email_html(subject, html_body):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("⚠️ 메일 설정 없음 → 메일 전송 생략")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ 메일 발송 완료:", EMAIL_RECEIVER)
    except Exception as e:
        print("❌ 메일 전송 실패:", e)

def main():
    df_news, df_time_stock, df_time_etf, df_bottom = build_report()

    sections = []
    sections.append(f"<h2 style='text-align:center'>📌 Daily Recommendation Report - {REPORT_DATE}</h2>")
    sections.append("<p style='text-align:center;color:#666'>뉴스 요약은 GPT(옵션), 추세 아이콘(▲/▼/▬), TimingScore=5조건 합계</p>")
    sections.append("<hr>")

    news_html = df_news.to_html(index=False, escape=False, border=1, justify='center') if not df_news.empty else "<p style='color:gray;'>최근 7일 뉴스 기반 추천 없음</p>"
    sections.append(f"<h3>① 최근 뉴스 기반 추천 (종목 | 설명)</h3>{news_html}")

    sections.append(to_html_table(df_time_stock, "② 매수 타이밍 양호 - STOCK"))
    sections.append(to_html_table(df_time_etf,   "③ 매수 타이밍 양호 - ETF"))

    sections.append("""
    <p style='color:#777;font-size:12px'>
    <b>TimingScore</b> = (1) Price>SMA200, (2) Price>SMA50, (3) RSI 35~60, (4) MACD 히스토그램>0, (5) 52주 고가의 85% 이상. 합계 0~5.
    </p>
    """)

    sections.append(to_html_table(df_bottom, "④ 지수 구성종목 중 52주 저점 인근 후보 (NASDAQ/S&P500/DOW 교집합)"))

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

    send_email_html(f"📊 주식·ETF 추천 리포트 - {REPORT_DATE}", html_doc)

if __name__ == "__main__":
    main()
