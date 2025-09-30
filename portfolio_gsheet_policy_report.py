import os
import gspread
import time
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import requests
import feedparser
import pickle
from fredapi import Fred
from openai import OpenAI
from oauth2client.service_account import ServiceAccountCredentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# FRED 클라이언트
fred = Fred(api_key=os.environ.get("FRED_API_KEY"))

# 캐시 파일 경로
CACHE_FILE = "fred_cache.pkl"

# FRED 지표 매핑
FRED_TICKERS = {
    "CPI (소비자물가지수)": "CPIAUCSL",
    "Unemployment (실업률)": "UNRATE",
    "Fed Funds Rate (연방기금금리)": "FEDFUNDS",
    "PCE (개인소비지출)": "PCE",
    "M2 (통화량)": "M2SL"
}

def fmt_money_2(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def fmt_2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def fmt_1(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return str(x)

def emoji_from_change_pct(pct):
    try:
        v = float(pct)
    except Exception:
        return "🟡"
    if v > 0.3:
        return "🟢"
    if v < -0.3:
        return "🔴"
    return "🟡"

# def get_gspread_client():
#     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
#     creds = ServiceAccountCredentials.from_json_keyfile_name(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), scope)
#     return gspread.authorize(creds)

# def open_gsheet(gs_id):
#     return get_gspread_client().open_by_key(gs_id)

def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), scope
    )
    return gspread.authorize(creds)

def open_gsheet(gs_id, retries=3, delay=5):
    """
    Google Sheet 열기 (503 오류 대비 재시도 포함)

    Args:
        gs_id (str): 구글 시트 ID
        retries (int): 최대 재시도 횟수
        delay (int): 재시도 간 대기 시간 (초)

    Returns:
        gspread.Spreadsheet: 열린 스프레드시트 객체
    """
    for i in range(retries):
        try:
            return get_gspread_client().open_by_key(gs_id)
        except gspread.exceptions.APIError as e:
            if "503" in str(e) and i < retries - 1:
                print(f"⚠️ Google API 503 오류 발생, {delay}초 후 재시도... ({i+1}/{retries})")
                time.sleep(delay)
                continue
            raise

def load_holdings_watchlist_settings():
    gs_id = os.environ.get("GSHEET_ID")
    sh = open_gsheet(gs_id)
    df_hold = pd.DataFrame(sh.worksheet("Holdings").get_all_records())
    df_watch = pd.DataFrame(sh.worksheet("Watchlist").get_all_records())
    df_set = pd.DataFrame(sh.worksheet("Settings").get_all_records())
    settings = dict(zip(df_set["Key"], df_set["Value"]))
    return df_hold, df_watch, settings

def get_last_and_prev_close(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
        return last, prev
    except Exception:
        return None, None

def compute_portfolio_values(df_hold, cash_usd):
    total_val = cash_usd
    total_prev = cash_usd
    for idx, row in df_hold.iterrows():
        t = row["Ticker"]
        sh = float(row.get("Shares", 0) or 0)
        last, prev = get_last_and_prev_close(t)
        if last is None:
            last = float(row.get("AvgPrice", 0) or 0)
        if prev is None:
            prev = last
        df_hold.loc[idx, "LastPrice"] = round(last, 2)
        df_hold.loc[idx, "PrevClose"] = round(prev, 2)
        val = sh * last
        pval = sh * prev
        df_hold.loc[idx, "Value"] = round(val, 2)
        df_hold.loc[idx, "PrevValue"] = round(pval, 2)
        total_val += val
        total_prev += pval
    return round(total_val, 2), round(total_prev, 2)

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def fundamentals_yf(ticker):
    out = {"PE": None, "ROE": None, "EPS": None}
    try:
        info = yf.Ticker(ticker).get_info()
        out["PE"] = info.get("trailingPE")
        out["ROE"] = info.get("returnOnEquity")
        out["EPS"] = info.get("trailingEps")
    except Exception:
        pass
    return out

def build_signals_table(tickers):
    rows = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="6mo")
            close = hist["Close"].dropna()

            rsi_val = rsi(close).iloc[-1] if len(close) >= 15 else np.nan
            rsi_emo = "🟡" if pd.isna(rsi_val) else ("🟢" if rsi_val >= 60 else ("🔴" if rsi_val <= 40 else "🟡"))

            macd_line, sig_line, _ = macd(close)
            macd_val = (macd_line.iloc[-1] - sig_line.iloc[-1]) if len(close) >= 30 else np.nan
            macd_emo = "🟡" if pd.isna(macd_val) else ("🟢" if macd_val > 0 else ("🔴" if macd_val < 0 else "🟡"))

            f = fundamentals_yf(t)
            pe, roe, eps = f["PE"], f["ROE"], f["EPS"]

            rows.append({
                "Ticker (종목)": t,
                "RSI (상대강도지수)": fmt_2(rsi_val) if pd.notna(rsi_val) else "N/A",
                "RSI Signal (신호)": rsi_emo,
                "MACD (지표)": fmt_2(macd_val) if pd.notna(macd_val) else "N/A",
                "MACD Signal (신호)": macd_emo,
                "P/E (주가수익비율)": fmt_2(pe) if pe is not None else "N/A",
                "ROE% (자기자본이익률)": fmt_2(roe*100) if isinstance(roe, (float, int)) else "N/A",
                "EPS (주당순이익)": fmt_2(eps) if eps is not None else "N/A"
            })
        except Exception:
            rows.append({
                "Ticker (종목)": t,
                "RSI (상대강도지수)": "N/A",
                "RSI Signal (신호)": "🟡",
                "MACD (지표)": "N/A",
                "MACD Signal (신호)": "🟡",
                "P/E (주가수익비율)": "N/A",
                "ROE% (자기자본이익률)": "N/A",
                "EPS (주당순이익)": "N/A"
            })
    return pd.DataFrame(rows)

def build_strategy_table(tickers, last_prices, settings):
    sl_floor = float(settings.get("RiskStopLossFloor", 0.90))
    tp1 = float(settings.get("TakeProfit1", 1.08))
    tp2 = float(settings.get("TakeProfit2", 1.15))
    rows = []
    for t in tickers:
        p = last_prices.get(t)
        if p is None:
            last = s1 = t1 = t2 = None
        else:
            last = round(p, 2)
            s1 = round(last * sl_floor, 2)
            t1 = round(last * tp1, 2)
            t2 = round(last * tp2, 2)
        rows.append({
            "Ticker (종목)": t,
            "Price (현재가)": fmt_2(last) if last is not None else "N/A",
            "Stop (손절)": fmt_2(s1) if s1 is not None else "N/A",
            "TP1 (1차 매도)": fmt_2(t1) if t1 is not None else "N/A",
            "TP2 (2차 매도)": fmt_2(t2) if t2 is not None else "N/A"
        })
    return pd.DataFrame(rows)

def gpt_strategy_summary(ticker_rows):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "<p>OpenAI API key missing → skip summary.</p>"
    try:
        client = OpenAI(api_key=api_key)
        csv_text = pd.DataFrame(ticker_rows).to_csv(index=False)
        prompt = (
            "다음 표의 RSI, MACD, P/E, ROE, EPS, 손절/목표가를 바탕으로 "
            "각 종목의 매매 전략을 종목별로 1줄씩 한국어로 요약해줘.\n\n"
            "출력 형식은 반드시 아래처럼 해줘:\n"
            "종목명: (매수|매도|관망) - 간단 설명\n\n"
            "예시:\n"
            "NVDA: 매수 - 기술적 지표 긍정적\n"
            "AAPL: 관망 - 실적 발표 대기\n"
            "TSLA: 매도 - 단기 과열\n\n"
            + csv_text
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=600
        )
        raw_text = resp.choices[0].message.content.strip()

        # ✅ 서식 변환
        lines = []
        for line in raw_text.splitlines():
            if ":" not in line:
                continue
            ticker, desc = line.split(":", 1)
            ticker = ticker.strip()
            desc = desc.strip()

            # 신호 아이콘 매핑
            if "매수" in desc:
                icon = "🟢"
            elif "매도" in desc:
                icon = "🔴"
            elif "관망" in desc:
                icon = "🟡"
            else:
                icon = "🔵"  # fallback

            lines.append(f"{icon} <b>{ticker}</b>: {desc}")

        formatted_html = "<br>".join(lines)
        return f"<div class='card'>{formatted_html}</div>"
    except Exception as e:
        return f"<p>GPT summary error: {e}</p>"

def translate_ko(text):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not text:
        return ""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"다음 영어 뉴스를 한국어로 자연스럽게 한 문단으로 요약해줘:\n{text}"}],
            max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception:
        return ""

# def fetch_news_for_ticker(ticker, api_key, page_size=3):
#     url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
#     r = requests.get(url, timeout=20)
#     if r.status_code != 200:
#         return []
#     return r.json().get("articles", [])

def fetch_news_for_ticker(ticker, api_key, page_size=3, days=7):
    """
    종목 뉴스 가져오기: NewsAPI → 실패/한도 초과 시 Google News RSS fallback

    Args:
        ticker (str): 종목 티커 (예: 'AAPL')
        api_key (str): NewsAPI 키
        page_size (int): 가져올 뉴스 개수
        days (int): 최근 n일 동안의 뉴스만 가져오기 (NewsAPI 전용)

    Returns:
        list of dict: [{title, url, source, published}, ...]
    """
    articles = []

    # 1️⃣ NewsAPI 시도
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "from": (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "source": a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", "")[:10],
                })
    except Exception as e:
        print(f"⚠️ NewsAPI 오류: {e}")

    # 2️⃣ fallback → Google News RSS
    if not articles:
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:page_size]:
                articles.append({
                    "title": entry.title,
                    "url": entry.link,
                    "source": getattr(entry, "source", {}).get("title", "Google News") if hasattr(entry, "source") else "Google News",
                    "published": entry.published[:16] if hasattr(entry, "published") else "",
                })
        except Exception as e:
            print(f"⚠️ Google News RSS 오류: {e}")

    return articles

def holdings_news_section(tickers):
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>🗞 Holdings News (보유 종목 뉴스)</h2><p>NEWS_API_KEY missing.</p>"
    html = "<h2>🗞 Holdings News (보유 종목 뉴스)</h2>"
    for t in tickers:
        arts = fetch_news_for_ticker(t, api_key)
        if not arts:
            continue
        cards = []
        for a in arts:
            title = a.get("title") or ""
            url = a.get("url") or "#"
            desc = a.get("description") or ""
            date = (a.get("publishedAt") or "")[:10]
            ko = translate_ko(f"{title}\n{desc}")
            cards.append(f"<div class='card'><b><a href='{url}' target='_blank'>{title}</a></b> <small>({date})</small><br><small>{desc}</small><br><i>{ko}</i></div>")
        html += f"<h3>{t}</h3>" + "".join(cards)
    return html

def watchlist_news_section(tickers):
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>👀 Watchlist News (관심 종목 뉴스)</h2><p>NEWS_API_KEY missing.</p>"
    html = "<h2>👀 Watchlist News (관심 종목 뉴스)</h2>"
    for t in tickers:
        arts = fetch_news_for_ticker(t, api_key)
        if not arts:
            continue
        cards = []
        for a in arts:
            title = a.get("title") or ""
            url = a.get("url") or "#"
            desc = a.get("description") or ""
            date = (a.get("publishedAt") or "")[:10]
            ko = translate_ko(f"{title}\n{desc}")
            cards.append(f"<div class='card'><b><a href='{url}' target='_blank'>{title}</a></b> <small>({date})</small><br><small>{desc}</small><br><i>{ko}</i></div>")
        html += f"<h3>{t}</h3>" + "".join(cards)
    return html

def market_news_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>NEWS_API_KEY missing.</p>"
    url = f"https://newsapi.org/v2/top-headlines?language=en&category=business&pageSize=6&apiKey={api_key}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>Load failed.</p>"
    arts = r.json().get("articles", [])
    cards = []
    for a in arts:
        title = a.get("title") or ""
        url = a.get("url") or "#"
        desc = a.get("description") or ""
        date = (a.get("publishedAt") or "")[:10]
        ko = translate_ko(f"{title}\n{desc}")
        cards.append(f"<div class='card'><b><a href='{url}' target='_blank'>{title}</a></b> <small>({date})</small><br><small>{desc}</small><br><i>{ko}</i></div>")
    return "<h2>📰 Market News (시장 뉴스)</h2>" + "".join(cards)

def policy_focus_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>🏛 Policy Focus (정책 포커스)</h2><p>NEWS_API_KEY missing.</p>"
    url = f"https://newsapi.org/v2/everything?q=Trump+policy+economy&language=en&sortBy=publishedAt&pageSize=6&apiKey={api_key}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return "<h2>🏛 Policy Focus (정책 포커스)</h2><p>Load failed.</p>"
    arts = r.json().get("articles", [])
    cards = []
    for a in arts:
        title = a.get("title") or ""
        url = a.get("url") or "#"
        desc = a.get("description") or ""
        date = (a.get("publishedAt") or "")[:10]
        ko = translate_ko(f"{title}\n{desc}")
        cards.append(f"<div class='card'><b><a href='{url}' target='_blank'>{title}</a></b> <small>({date})</small><br><small>{desc}</small><br><i>{ko}</i></div>")
    return "<h2>🏛 Policy Focus (정책 포커스)</h2>" + "".join(cards)

def load_fred_cache():
    """캐시 파일 불러오기"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("date") == datetime.today().date():
            return cache.get("data", {})
    return {}

def save_fred_cache(data):
    """캐시 파일 저장"""
    cache = {"date": datetime.today().date(), "data": data}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def fetch_economic_data():
    """
    FRED 지표 한 번만 불러와서 캐싱
    Returns: dict {지표명: {YYYY-MM: value, ...}}
    """
    cached = load_fred_cache()
    if cached:
        return cached

    result = {}
    for name, tick in FRED_TICKERS.items():
        try:
            series = fred.get_series(tick, observation_start=f"{datetime.today().year}-01-01")
            if series is None or series.empty:
                result[name] = {}
                continue

            monthly_vals = series.resample("M").last().dropna()
            monthly_dict = {d.strftime("%Y-%m"): float(v) for d, v in monthly_vals.items()}
            result[name] = monthly_dict
        except Exception:
            result[name] = {}

    save_fred_cache(result)
    return result

def econ_section():
    """📊 Economic Indicators (경제 지표) - 월별 테이블 + 최근 증감률"""
    econ_data = fetch_economic_data()

    # 1월부터 현재월까지 열 생성
    months = pd.date_range(
        start=f"{datetime.today().year}-01-01",
        end=datetime.today(),
        freq="M"
    ).strftime("%Y-%m").tolist()

    # 테이블 구조
    table_data = {"Indicator (지표)": []}
    for m in months:
        table_data[m] = []
    table_data["Change (변화)"] = []  # 새 열 추가

    for name in FRED_TICKERS.keys():
        monthly_dict = econ_data.get(name, {})

        # 값 채우기
        row_vals = []
        for m in months:
            if m in monthly_dict:
                row_vals.append(f"{monthly_dict[m]:,.2f}")
            else:
                row_vals.append("N/A")

        # 최근 값 변화 계산
        if len(monthly_dict) >= 2:
            vals = list(monthly_dict.values())
            prev_val, last_val = vals[-2], vals[-1]
            change = last_val - prev_val
            pct = (change / prev_val) * 100 if prev_val != 0 else 0

            if change > 0:
                color, arrow = "green", "🟢"
            elif change < 0:
                color, arrow = "red", "🔴"
            else:
                color, arrow = "black", "⚫"

            change_html = f"<span style='color:{color}'>{last_val:,.2f} ({change:+.2f}, {pct:+.2f}%) {arrow}</span>"
        else:
            change_html = "N/A"

        # 테이블에 넣기
        table_data["Indicator (지표)"].append(name)
        for idx, m in enumerate(months):
            table_data[m].append(row_vals[idx])
        table_data["Change (변화)"].append(change_html)

    df_out = pd.DataFrame(table_data)
    return "<h2>📊 Economic Indicators (경제 지표)</h2>" + df_out.to_html(index=False, escape=False)

def indices_section():
    import yfinance as yf
    import pandas as pd
    from fredapi import Fred
    import os

    # FRED API 클라이언트
    # fred = Fred(api_key=os.environ.get("FRED_API_KEY"))

    # 주요 지수 (Yahoo Finance)
    INDEX_MAP = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
        "Gold": "GC=F",
        "Crude Oil": "CL=F",
    }

    rows = []

    # Yahoo Finance 기반 주요 지수
    for name, ticker in INDEX_MAP.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if data.empty:
                rows.append(f"<tr><td>{name}</td><td colspan='3'>데이터 없음</td></tr>")
                continue

            last_price = float(data["Close"].iloc[-1])
            prev_close = float(data["Close"].iloc[-2])

            change = last_price - prev_close
            pct_change = (change / prev_close) * 100

            if change > 0:
                color, arrow = "green", "🟢"
            elif change < 0:
                color, arrow = "red", "🔴"
            else:
                color, arrow = "black", "⚫"

            rows.append(
                f"<tr>"
                f"<td>{name}</td>"
                f"<td>{prev_close:,.2f}</td>"
                f"<td><span style='color:{color}'>{last_price:,.2f} "
                f"({change:+.2f}, {pct_change:+.2f}%) {arrow}</span></td>"
                f"</tr>"
            )
        except Exception as e:
            rows.append(f"<tr><td>{name}</td><td colspan='3'>Error: {str(e)}</td></tr>")

    # FRED: M2 통화량
    try:
        series = fred.get_series("M2SL")  # 미국 M2 통화량
        monthly_vals = series.resample("M").last().dropna()
        last_val = float(monthly_vals.iloc[-1])
        prev_val = float(monthly_vals.iloc[-2])

        change = last_val - prev_val
        pct_change = (change / prev_val) * 100

        if change > 0:
            color, arrow = "green", "🟢"
        elif change < 0:
            color, arrow = "red", "🔴"
        else:
            color, arrow = "black", "⚫"

        rows.append(
            f"<tr>"
            f"<td>M2 통화량 (억 달러)</td>"
            f"<td>{prev_val:,.2f}</td>"
            f"<td><span style='color:{color}'>{last_val:,.2f} "
            f"({change:+,.2f}, {pct_change:+.2f}%) {arrow}</span></td>"
            f"</tr>"
        )
    except Exception as e:
        rows.append(f"<tr><td>M2 통화량</td><td colspan='3'>Error: {str(e)}</td></tr>")

    # 최종 HTML
    html = """
    <h2>📊 주요 지수 및 경제 지표</h2>
    <table border="1" cellspacing="0" cellpadding="4">
      <tr>
        <th>지수</th>
        <th>전월/전일 종가</th>
        <th>현재값 (변화)</th>
      </tr>
      {}
    </table>
    """.format(
        "\n".join(rows)
    )

    return html

def send_email_html(subject, html_body):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    if not (sender and password and receiver):
        print("⚠️ Missing email settings → Email not sent")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    msg.attach(MIMEText(html_body, "html", _charset="utf-8"))
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("✅ Email sent to:", receiver)
    except Exception as e:
        print("❌ Email send failed:", e)

def build_report_html():
    df_hold, df_watch, settings = load_holdings_watchlist_settings()

    # --- Holdings 데이터 전처리 ---
    if "Shares" in df_hold.columns:
        df_hold["Shares"] = pd.to_numeric(df_hold["Shares"], errors="coerce").fillna(0.0)
    if "AvgPrice" in df_hold.columns:
        df_hold["AvgPrice"] = pd.to_numeric(df_hold["AvgPrice"], errors="coerce").fillna(0.0)

    cash_usd = float(settings.get("CashUSD", 0) or 0)

    # ✅ 원본 df_hold 사용
    total_today, total_yday = compute_portfolio_values(df_hold, cash_usd)

    # 총자산 증감
    diff = total_today - total_yday
    total_change_pct = round(((diff) / total_yday * 100.0), 2) if total_yday != 0 else 0.0
    emo = emoji_from_change_pct(total_change_pct)

    # 현금 행 추가
    cash_row = {
        "Ticker": "CASH", "Shares": np.nan, "AvgPrice": np.nan,
        "LastPrice": 1.00, "PrevClose": 1.00,
        "Value": cash_usd, "PrevValue": cash_usd
    }
    df_disp = pd.concat([df_hold, pd.DataFrame([cash_row])], ignore_index=True)

    # ---- Profit/Loss 계산 (⚠️ 포맷팅 전에 먼저 계산) ----
    def calc_profit_loss(row):
        try:
            sh = float(row.get("Shares", 0) or 0)
            avg = float(row.get("AvgPrice", 0) or 0)
            val = float(row.get("Value", 0) or 0)
            cost = sh * avg
            profit = val - cost
            if profit > 0:
                return f"<span style='color:green'>+{fmt_money_2(profit)}</span>"
            elif profit < 0:
                return f"<span style='color:red'>{fmt_money_2(profit)}</span>"
            else:
                return f"<span style='color:black'>{fmt_money_2(profit)}</span>"
        except Exception:
            return "-"
    df_disp["Profit/Loss (수익/손실)"] = df_disp.apply(calc_profit_loss, axis=1)

    # ---- 포맷팅 함수 ----
    def fmt_price_with_change(row):
        try:
            last = float(row["LastPrice"])
            prev = float(row["PrevClose"])
            if prev == 0:
                return fmt_money_2(last)
            pct = round((last - prev) / prev * 100, 2)
            color = "green" if pct > 0 else ("red" if pct < 0 else "black")
            return f"<span style='color:{color}'>{fmt_money_2(last)} ({pct}%)</span>"
        except Exception:
            return "-"

    def fmt_value_with_change(row):
        try:
            val = float(row["Value"])
            prev = float(row["PrevValue"])
            if prev == 0:
                return fmt_money_2(val)
            pct = round((val - prev) / prev * 100, 2)
            color = "green" if pct > 0 else ("red" if pct < 0 else "black")
            return f"<span style='color:{color}'>{fmt_money_2(val)} ({pct}%)</span>"
        except Exception:
            return "-"

    # ---- 표 컬럼 포맷팅 ----
    if "Shares" in df_disp.columns:
        df_disp["Shares"] = df_disp["Shares"].apply(lambda x: fmt_2(x) if pd.notna(x) else "-")
    if "AvgPrice" in df_disp.columns:
        df_disp["AvgPrice"] = df_disp["AvgPrice"].apply(lambda x: fmt_money_2(x) if pd.notna(x) else "-")

    df_disp["LastPrice"] = df_disp.apply(fmt_price_with_change, axis=1)
    df_disp["Value"] = df_disp.apply(fmt_value_with_change, axis=1)
    df_disp["PrevClose"] = df_disp["PrevClose"].apply(fmt_money_2)
    df_disp["PrevValue"] = df_disp["PrevValue"].apply(fmt_money_2)

    # 컬럼 이름 변경
    df_disp = df_disp.rename(columns={
        "Ticker": "Ticker (종목)",
        "Shares": "Shares (수량)",
        "AvgPrice": "Avg Price (평단가)",
        "LastPrice": "Last Price (현재가)",
        "PrevClose": "Prev Close (전일종가)",
        "Value": "Value (자산가치)",
        "PrevValue": "Prev Value (전일자산)",
        "Profit/Loss (수익/손실)": "Profit/Loss (수익/손실)"
    })

    # 총자산 변화 표시
    total_color = "green" if diff > 0 else ("red" if diff < 0 else "black")
    holdings_html = f"""
    <h2>📂 Holdings (보유 종목)</h2>
    <p><b>Total Assets (총 자산):</b> {fmt_money_2(total_today)} &nbsp;&nbsp;
       <b>Δ vs. Yesterday (전일 대비 변화):</b>
       <span style='color:{total_color}'>{fmt_money_2(diff)} ({fmt_2(total_change_pct)}%)</span> {emo}</p>
    {df_disp.to_html(index=False, escape=False)}
    """

    # -------- Signals Section --------
    tickers_hold = [t for t in df_hold["Ticker"].tolist() if isinstance(t, str)]
    tickers_watch = [t for t in df_watch["Ticker"].tolist() if isinstance(t, str)]

    # 보유 종목 신호
    signals_df_hold = build_signals_table(tickers_hold)
    signals_html_hold = f"<h2>📈 Signals – Holdings (보유 종목)</h2>{signals_df_hold.to_html(index=False)}"

    # 관심 종목 신호
    signals_html_watch = ""
    if tickers_watch:
        signals_df_watch = build_signals_table(tickers_watch)
        signals_html_watch = f"<h2>📊 Signals – Watchlist (관심 종목)</h2>{signals_df_watch.to_html(index=False)}"

    signals_html = signals_html_hold + signals_html_watch

    # -------- Strategies Section --------
    last_prices = {}
    for t in tickers_hold:
        lp, _ = get_last_and_prev_close(t)
        last_prices[t] = lp
    strat_df = build_strategy_table(tickers_hold, last_prices, settings)
    merged_for_gpt = pd.merge(signals_df_hold, strat_df, on="Ticker (종목)", how="left")
    strategy_html = f"<h2>🧭 Strategies (종목별 매매 전략)</h2>{strat_df.to_html(index=False)}"
    strategy_summary_html = f"<h3>📝 Strategy Summary (전략 요약)</h3>{gpt_strategy_summary(merged_for_gpt.to_dict(orient='records'))}"

    # -------- News Section --------
    hold_news_html = holdings_news_section(tickers_hold)
    watch_news_html = watchlist_news_section(tickers_watch) if tickers_watch else ""
    market_html = market_news_section()
    policy_html = policy_focus_section()

    # -------- Econ / Indices / GPT Opinion --------
    econ_html = econ_section()
    indices_html = indices_section()
    gpt_html = f"<h2>🤖 GPT Opinion (투자의견)</h2>{gpt_strategy_summary(merged_for_gpt.to_dict(orient='records'))}"

    # -------- HTML 최종 출력 --------
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background:#fafafa; }
    h1 { text-align:center; }
    h2 { margin-top:30px; color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:5px; }
    table { border-collapse: collapse; width:100%; margin:10px 0; }
    th, td { border:1px solid #ddd; padding:8px; text-align:center; }
    th { background:#f4f6f6; }
    .card { background:white; border:1px solid #ddd; border-radius:8px; padding:10px; margin:10px 0; }
    .gpt-box { background:#f0f4ff; padding:15px; border-radius:10px; }
    .muted { color:#666; }
    </style>
    """
    html = f"""
    <html><head><meta charset='utf-8'>{style}</head><body>
    <h1>📊 Portfolio Report (포트폴리오 리포트)</h1>
    <p style='text-align:center' class='muted'>Generated at {now}</p>

    {holdings_html}
    {signals_html}
    {strategy_html}
    {strategy_summary_html}
    {hold_news_html}
    {watch_news_html}
    {market_html}
    {policy_html}
    {econ_html}
    {indices_html}
    {gpt_html}

    </body></html>
    """
    return html

def main():
    html_doc = build_report_html()
    outname = f"portfolio_gsheet_policy_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")
    send_email_html(f"📊 Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}", html_doc)

if __name__ == "__main__":
    main()
