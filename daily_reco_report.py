
#!/usr/bin/env python3
# coding: utf-8

"""
Daily Recommendation Report
---------------------------
Creates a *separate* daily recommendations report based on three criteria:

1) US policy-sensitive picks (based on President/White House news sentiment & sector mapping)
2) Timing picks among blue chips (technical momentum/RSI/MACD/SMA filters)
3) Fundamental-quality picks (PER, PBR, ROE, Debt ratio, EPS & revenue growth, Cash Flow)

Outputs HTML and CSV. Optionally emails the HTML if EMAIL_* env vars are set.
Environment Variables (optional but recommended):
- NEWS_API_KEY           (NewsAPI.org key for news scanning)
- EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER (for SMTP via Gmail)
- UNIVERSE_TICKERS       (comma-separated tickers to screen; default: S&P-100 style set below)
"""

import os, io, base64, smtplib, time, json, math, sys
import requests
import pandas as pd
import yfinance as yf
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

# Universe: override via UNIVERSE_TICKERS env (comma-separated)
DEFAULT_UNIVERSE = [
    # Large-cap, policy-sensitive or quality names
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","JPM","V","MA","UNH","HD","ABBV","PG",
    "XOM","CVX","COP","NEE","DUK","SO","LMT","RTX","NOC","BA","GD","HON","CAT","DE","MMM",
    "PFE","JNJ","MRK","LLY","AZN","BMY","CI","HUM","CVS","ANTM","COST","WMT","TGT","LOW",
    "PEP","KO","MCD","SBUX","DIS","NFLX","CMCSA","T","VZ","BAC","C","WFC","GS","MS",
    "ADBE","CRM","ORCL","INTC","AMD","QCOM","TXN","IBM","NOW","SNPS","PANW","SHOP","PYPL",
    "UPS","FDX","CSX","UNP","NSC","AAL","DAL","UAL","MAR","HLT","BKNG",
    "GE","GM","F","TSM","TM","NKE","EL","LULU","ETN","PH","LIN","SCHW","BK",
    "PLTR","ABNB","UBER","LYFT","SQ","ROKU","SPGI","ICE","MSCI","BLK","TMO","DHR","ISRG"
]
UNIVERSE = [t.strip().upper() for t in os.getenv("UNIVERSE_TICKERS","").split(",") if t.strip()] or DEFAULT_UNIVERSE

# ------------------------
# Helpers
# ------------------------
def safe_pct(a, b):
    try:
        if b == 0 or b is None or a is None:
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None

def fetch_news(query, from_days=2, page_size=50):
    """Use NewsAPI to fetch recent articles for a query."""
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

def extract_policy_signals():
    """
    Look for White House / President news in last 2 days and map keywords to sectors/tickers.
    """
    topics = [
        "White House policy", "President remarks", "Biden remarks", "POTUS comments",
        "executive order", "tariffs", "infrastructure bill", "defense budget", "healthcare policy",
        "energy policy", "drug pricing", "semiconductor subsidies", "AI executive order"
    ]
    articles = []
    for q in topics:
        arts = fetch_news(q, from_days=2, page_size=20)
        for a in arts:
            a["_topic"] = q
        articles.extend(arts)

    # Keyword → sector mapping
    keyword_map = {
        "defense": ["LMT","RTX","NOC","GD","BA"],
        "pentagon": ["LMT","RTX","NOC","GD","BA"],
        "military": ["LMT","RTX","NOC","GD","BA"],
        "semiconductor": ["NVDA","AMD","QCOM","TSM","INTC","AVGO","TXN"],
        "chip": ["NVDA","AMD","QCOM","TSM","INTC","AVGO","TXN"],
        "tariff": ["CAT","DE","GM","F","AAPL"],
        "infrastructure": ["CAT","DE","ETN","PH","VMC","HOLX"],
        "drug": ["LLY","PFE","MRK","BMY","JNJ","ABBV","AZN"],
        "medicare": ["UNH","CI","HUM","CVS"],
        "energy": ["XOM","CVX","COP","NEE","DUK","SO"],
        "oil": ["XOM","CVX","COP"],
        "renewable": ["NEE","ENPH","SEDG"],
        "ai": ["NVDA","MSFT","GOOGL","META","AMZN","ADBE","CRM","NOW","PLTR"],
        "executive order": ["NVDA","MSFT","GOOGL","META","LMT","RTX","XOM","NEE"],
    }

    # Score tickers by article counts weighted by recency and topic relevance
    scores = {t: 0.0 for t in UNIVERSE}
    keep_samples = {t: [] for t in UNIVERSE}
    now = datetime.utcnow()

    for a in articles:
        title = (a.get("title") or "").lower()
        desc  = (a.get("description") or "").lower()
        content = f"{title} {desc}"
        # recency weight
        try:
            published = datetime.fromisoformat(a.get("publishedAt").replace("Z","+00:00"))
            hours = max((now - published).total_seconds() / 3600.0, 1.0)
            recency_w = 1.0 / math.log(hours + 2.0)  # slowly decays
        except Exception:
            recency_w = 1.0

        for key, tickers in keyword_map.items():
            if key in content:
                for t in tickers:
                    if t in UNIVERSE:
                        scores[t] = scores.get(t, 0.0) + 1.0 * recency_w
                        if len(keep_samples[t]) < 3:
                            keep_samples[t].append(a.get("title") or a.get("url"))
    # Build DataFrame
    rows = []
    for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if s <= 0: 
            continue
        rows.append({"Ticker": t, "PolicyScore": round(s,3), "SampleNews": " | ".join(keep_samples[t])})
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
    # SMAs
    out["SMA50"]  = close.rolling(50).mean().iloc[-1] if len(close)>=50 else None
    out["SMA200"] = close.rolling(200).mean().iloc[-1] if len(close)>=200 else None
    out["Price"]  = float(close.iloc[-1])
    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    out["RSI14"] = float(rsi.dropna().iloc[-1]) if rsi.dropna().size else None
    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["MACD_Hist"] = float(hist.dropna().iloc[-1]) if hist.dropna().size else None
    # 52-week metrics
    if len(close) >= 252:
        out["High52w"] = float(close[-252:].max())
        out["Low52w"]  = float(close[-252:].min())
    else:
        out["High52w"] = float(close.max())
        out["Low52w"]  = float(close.min())
    return out

def fetch_fundamentals(ticker):
    """
    Pulls key ratios from yfinance. Robust to missing fields.
    Returns dict with PER, PBR, ROE, DebtRatio, EPS, RevGrowthYoY, CFO, FCF, etc.
    """
    info = {}
    try:
        y = yf.Ticker(ticker)

        # info
        i = y.info or {}
        fin = y.financials if hasattr(y, "financials") else None
        qfin = y.quarterly_financials if hasattr(y, "quarterly_financials") else None
        bs = y.balance_sheet if hasattr(y, "balance_sheet") else None
        qbs = y.quarterly_balance_sheet if hasattr(y, "quarterly_balance_sheet") else None
        cf = y.cashflow if hasattr(y, "cashflow") else None
        qcf = y.quarterly_cashflow if hasattr(y, "quarterly_cashflow") else None

        # Basic
        per = i.get("trailingPE") or i.get("forwardPE")
        pbr = i.get("priceToBook")
        roe = i.get("returnOnEquity")
        eps = i.get("trailingEps") or i.get("forwardEps")

        # Debt ratio = TotalDebt / TotalAssets
        total_debt  = i.get("totalDebt")
        total_assets = i.get("totalAssets")
        debt_ratio = (total_debt/total_assets) if (total_debt and total_assets) else None

        # Revenue growth YoY from annual financials if available
        rev_growth = None
        try:
            if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                rev = fin.loc["Total Revenue"].dropna().astype(float)
                if len(rev) >= 2:
                    rev_growth = (rev.iloc[0] - rev.iloc[1]) / rev.iloc[1]
        except Exception:
            pass

        # Cash flow: CFO and FCF (approx: operatingCashFlow - capitalExpenditures)
        CFO = None
        FCF = None
        try:
            if cf is not None and not cf.empty:
                ocf = cf.loc["Total Cash From Operating Activities"].dropna().astype(float)
                capex = None
                # yfinance uses various labels; try both
                for label in ["Capital Expenditures", "Capital Expenditure"]:
                    if label in cf.index:
                        capex = cf.loc[label].dropna().astype(float)
                        break
                if ocf.size:
                    CFO = float(ocf.iloc[0])
                if ocf.size and capex is not None and capex.size:
                    FCF = float(ocf.iloc[0] - capex.iloc[0])
        except Exception:
            pass

        info.update({
            "PER": float(per) if per else None,
            "PBR": float(pbr) if pbr else None,
            "ROE": float(roe) if roe else None,
            "EPS": float(eps) if eps else None,
            "DebtRatio": float(debt_ratio) if debt_ratio else None,  # fraction
            "RevGrowthYoY": float(rev_growth) if rev_growth is not None else None,
            "CFO": CFO,
            "FCF": FCF,
        })
    except Exception:
        pass
    return info

def timing_score(tech):
    """
    Score for buy timing:
      - Price above SMA200 and SMA50 (trend)
      - RSI between 35~60 (not overbought)
      - Positive MACD Histogram
      - Within 15% of 52w high (strength)
    """
    if not tech: 
        return 0.0
    score = 0.0
    p = tech.get("Price")
    s50 = tech.get("SMA50")
    s200 = tech.get("SMA200")
    rsi = tech.get("RSI14")
    macdh = tech.get("MACD_Hist")
    h52 = tech.get("High52w")
    if None in (p, s50, s200, rsi, macdh, h52):
        return 0.0

    if p > s200: score += 1.0
    if p > s50:  score += 1.0
    if 35 <= rsi <= 60: score += 1.0
    if macdh > 0: score += 1.0
    if h52 and p >= 0.85 * h52: score += 1.0
    return score

def fundamental_pass(f):
    """
    Basic quality filters (customize thresholds as needed):
      PER 5~35, PBR < 8, ROE > 10%, DebtRatio < 0.7, EPS > 0
      RevGrowthYoY > 5%, CFO > 0, FCF >= 0
    """
    if not f:
        return False
    conditions = [
        (f.get("PER") is not None and 5 <= f["PER"] <= 35),
        (f.get("PBR") is not None and f["PBR"] < 8),
        (f.get("ROE") is not None and f["ROE"] > 0.10),
        (f.get("DebtRatio") is not None and f["DebtRatio"] < 0.70),
        (f.get("EPS") is not None and f["EPS"] > 0),
        (f.get("RevGrowthYoY") is not None and f["RevGrowthYoY"] > 0.05),
        (f.get("CFO") is not None and f["CFO"] > 0),
        (f.get("FCF") is not None and f["FCF"] >= 0),
    ]
    return all(conditions)

def build_report():
    rows_policy = []
    rows_timing = []
    rows_fund = []

    # (1) Policy-sensitive
    policy_df = extract_policy_signals() if NEWS_API_KEY else pd.DataFrame()
    if not policy_df.empty:
        # Enrich with latest price
        prices = []
        for t in policy_df["Ticker"]:
            dfp = fetch_prices(t, "1mo", "1d")
            price = float(dfp["Close"].iloc[-1]) if not dfp.empty else None
            prices.append(price)
        policy_df["Price"] = prices
        rows_policy = policy_df.sort_values("PolicyScore", ascending=False).head(15).to_dict("records")

    # (2) Timing & (3) Fundamentals
    for t in UNIVERSE:
        df = fetch_prices(t, "1y", "1d")
        tech = compute_technicals(df)
        f = fetch_fundamentals(t)

        # Timing
        tscore = timing_score(tech)
        if tscore >= 4:  # strong setup
            rows_timing.append({
                "Ticker": t,
                "Price": tech.get("Price"),
                "SMA50": tech.get("SMA50"),
                "SMA200": tech.get("SMA200"),
                "RSI14": tech.get("RSI14"),
                "MACD_Hist": tech.get("MACD_Hist"),
                "High52w": tech.get("High52w"),
                "TimingScore(0-5)": tscore
            })

        # Fundamentals
        if fundamental_pass(f):
            rows_fund.append({"Ticker": t, **f})

    df_policy = pd.DataFrame(rows_policy)
    df_timing = pd.DataFrame(rows_timing).sort_values("TimingScore(0-5)", ascending=False) if rows_timing else pd.DataFrame()
    df_fund   = pd.DataFrame(rows_fund).sort_values("ROE", ascending=False) if rows_fund else pd.DataFrame()

    return df_policy, df_timing, df_fund

def to_html_table(df, title):
    if df is None or df.empty:
        return f"<h3>{title}</h3><p style='color:gray;'>해당 조건을 만족하는 결과가 없습니다.</p>"
    # format some columns
    df = df.copy()
    for col in df.columns:
        if col.lower().endswith("ratio") or col in ["ROE","RevGrowthYoY"]:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        if col in ["PER","PBR","RSI14","MACD_Hist","TimingScore(0-5)"]:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        if col in ["Price","SMA50","SMA200","High52w","EPS","CFO","FCF"]:
            df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    html = df.to_html(index=False, escape=False, border=1, justify="center")
    return f"<h3>{title}</h3>{html}"

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
    df_policy, df_timing, df_fund = build_report()

    # Save CSVs
    out_dir = os.getenv("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    csv_policy = os.path.join(out_dir, f"policy_picks_{REPORT_DATE}.csv")
    csv_timing = os.path.join(out_dir, f"timing_picks_{REPORT_DATE}.csv")
    csv_fund   = os.path.join(out_dir, f"fundamental_picks_{REPORT_DATE}.csv")
    if df_policy is not None and not df_policy.empty: df_policy.to_csv(csv_policy, index=False)
    if df_timing is not None and not df_timing.empty: df_timing.to_csv(csv_timing, index=False)
    if df_fund   is not None and not df_fund.empty: df_fund.to_csv(csv_fund,   index=False)

    # HTML assemble
    html = f"""
    <html><body style="font-family:Arial, sans-serif; line-height:1.6;">
    <h2 style="text-align:center;">📌 Daily Recommendation Report ({REPORT_DATE})</h2>
    <p style="text-align:center;color:#666;">조건: (1) 정책 민감 (2) 타이밍 우량 (3) 재무 우량</p>
    <hr>
    {to_html_table(df_policy.head(15) if df_policy is not None else None, "① 미국 정책/대통령 발언 수혜 가능 종목 (상위 15)")}
    <br>
    {to_html_table(df_timing.head(20) if df_timing is not None else None, "② 우량 종목 중 매수 타이밍 양호 (상위 20)")}
    <br>
    {to_html_table(df_fund.head(30) if df_fund is not None else None, "③ 재무 건전 & 성장성 양호 (상위 30)")}
    <hr>
    <p style="color:#888;">Tip: UNIVERSE_TICKERS 환경변수로 유니버스 변경, NEWS_API_KEY로 뉴스 기반 (①) 강화.</p>
    </body></html>
    """
    # Save HTML
    html_path = os.path.join(out_dir, f"daily_reco_report_{REPORT_DATE}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Optional email
    subject = f"Daily Stock Recommendation - {REPORT_DATE}"
    send_email_html(subject, html)

    print("✅ Report ready:", html_path)
    if df_policy is not None and not df_policy.empty: print("   Policy picks:", csv_policy)
    if df_timing is not None and not df_timing.empty: print("   Timing picks:", csv_timing)
    if df_fund   is not None and not df_fund.empty: print("   Fundamental picks:", csv_fund)

if __name__ == "__main__":
    main()
