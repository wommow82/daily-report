#!/usr/bin/env python3
# coding: utf-8

import os
import time
import requests
import yfinance as yf
import openai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import matplotlib.font_manager as fm
import os, subprocess

# ----- matplotlib 한글 폰트 설정 -----

def setup_matplotlib_korean_font():
    try:
        # check if NanumGothic exists
        fonts = [f.name for f in fm.fontManager.ttflist]
        if not any("NanumGothic" in f for f in fonts):
            print("한글 폰트(NanumGothic) 설치 중...")
            subprocess.run(["sudo", "apt-get", "install", "-y", "fonts-nanum"], check=False)
            matplotlib.font_manager._rebuild()
        matplotlib.rc('font', family='NanumGothic')
        matplotlib.rcParams['axes.unicode_minus'] = False
        print("한글 폰트 설정 완료")
    except Exception as e:
        print(f"폰트 설정 실패: {e} (영문 폰트 사용)")
        matplotlib.rc('font', family='DejaVu Sans')

setup_matplotlib_korean_font()

# ---------------------------
# Configuration / Globals
# ---------------------------
# 포트폴리오 - 필요시 수정
portfolio = {
    "NVDA": {"shares": 50, "avg_price": 123.97},
    "PLTR": {"shares": 10, "avg_price": 151.60},
    "SCHD": {"shares": 2140, "avg_price": 24.37},
    "TSLA": {"shares": 10, "avg_price": 320.745},
}

# 계좌 현금 (USD)
CASH_BALANCE = 16671.21

# OpenAI / External API config (환경변수에서 읽음)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # NewsAPI 사용 시
FRED_API_KEY = os.getenv("FRED_API_KEY")
TRADING_API_KEY = os.getenv("TRADINGECONOMICS_API_KEY", "guest:guest")

# Email config (환경변수 필요)
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# OpenAI 설정
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4o-mini"
matplotlib.rc('font', family='NanumGothic')  # 또는 AppleGothic, Malgun Gothic
# ---------------------------
# Helpers
# ---------------------------
def gpt_chat(prompt: str, retries: int = 3, backoff: int = 5) -> str:
    """OpenAI ChatCompletion 호출 + 간단한 retry"""
    for attempt in range(retries):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            return resp.choices[0].message.content.strip()
        except openai.error.RateLimitError as e:
            wait = backoff * (attempt + 1)
            print(f"[GPT] Rate limit: 재시도 {attempt+1}/{retries} - {wait}s 대기")
            time.sleep(wait)
        except Exception as e:
            print(f"[GPT] 오류: {e}")
            time.sleep(backoff)
    return "GPT 요청 실패(재시도 초과)"

def get_usd_to_cad_rate() -> float:
    try:
        res = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=CAD", timeout=10)
        data = res.json()
        rate = data.get("rates", {}).get("CAD")
        if rate:
            return float(rate)
    except Exception as e:
        print(f"환율 조회 실패: {e}")
    # fallback
    return 1.38

def get_stock_prices(ticker: str):
    """
    return (price_today, price_yesterday) or (None, None) if missing
    safe: checks length
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")["Close"]
        if getattr(hist, "empty", True) or len(hist) == 0:
            return None, None
        elif len(hist) == 1:
            return float(hist.iloc[-1]), None
        else:
            return float(hist.iloc[-1]), float(hist.iloc[-2])
    except Exception as e:
        print(f"[get_stock_prices] {ticker} 에러: {e}")
        return None, None

def get_rsi_macd_values(ticker: str, period: str = "365d"):
    """
    Returns (rsi_float_or_None, macd_hist_float_or_None)
    Uses yfinance history and safe operations.
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return None, None
        close = df["Close"].dropna()
        if len(close) < 15:
            return None, None

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        latest_rsi = rsi_series.dropna().iloc[-1] if not rsi_series.dropna().empty else None

        # MACD histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = (macd_line - signal).dropna()
        latest_macd = macd_hist.iloc[-1] if not macd_hist.empty else None

        return (float(latest_rsi) if latest_rsi is not None else None,
                float(latest_macd) if latest_macd is not None else None)
    except Exception as e:
        print(f"[get_rsi_macd_values] {ticker} 에러: {e}")
        return None, None

# ---------------------------
# Sections (HTML generators)
# ---------------------------
def get_portfolio_overview_html():
    """종목별 현황 + 합계 + 현금비중 한 테이블로 표시"""
    usd_to_cad = get_usd_to_cad_rate()
    total_value = 0
    total_cost = 0
    total_profit = 0
    total_daily = 0

    html = "<h3>💼 전체 포트폴리오 현황</h3>"
    html += "<table border='1' cellpadding='5' style='border-collapse:collapse;'>"
    html += "<tr><th>종목</th><th>보유수량</th><th>현재가(USD)</th><th>평단가</th><th>일일손익</th><th>누적손익</th><th>수익률</th></tr>"

    for ticker, info in portfolio.items():
        price_today, price_yesterday = get_stock_prices(ticker)
        if price_today is None:
            price_today = info["avg_price"]
            price_yesterday = info["avg_price"]

        daily_profit = (price_today - price_yesterday) * info["shares"]
        cost = info["avg_price"] * info["shares"]
        value = price_today * info["shares"]
        profit = value - cost
        rate = (profit / cost) * 100 if cost > 0 else 0

        total_value += value
        total_cost += cost
        total_profit += profit
        total_daily += daily_profit

        html += (f"<tr><td>{ticker}</td><td>{info['shares']}</td>"
                 f"<td>{price_today:.2f}</td><td>{info['avg_price']:.2f}</td>"
                 f"<td style='color:{'green' if daily_profit>=0 else 'red'}'>{daily_profit:+.2f}</td>"
                 f"<td style='color:{'green' if profit>=0 else 'red'}'>{profit:+.2f}</td>"
                 f"<td style='color:{'green' if rate>=0 else 'red'}'>{rate:+.2f}%</td></tr>")

    total_with_cash = total_value + CASH_BALANCE
    cash_ratio = (CASH_BALANCE / total_with_cash) * 100 if total_with_cash > 0 else 0
    total_rate = (total_profit / total_cost) * 100 if total_cost > 0 else 0

    html += (f"<tr><td><strong>합계</strong></td><td>-</td><td>-</td><td>-</td>"
             f"<td><strong>{total_daily:+.2f}</strong></td>"
             f"<td><strong>{total_profit:+.2f}</strong></td>"
             f"<td><strong>{total_rate:+.2f}%</strong></td></tr>")
    html += "</table>"
    html += f"<p>💰 현금 보유액: <strong>{CASH_BALANCE:,.2f}$</strong> (비중 {cash_ratio:.2f}%)</p>"
    html += f"<p>총 평가금액: <strong>{total_with_cash:,.2f}$</strong> ({total_with_cash * usd_to_cad:,.2f} CAD)</p>"
    return html

def get_monthly_economic_indicators_html():
    """최근 12개월 주요 경제지표 표 + GPT 해석"""
    indicators = {
        "CPIAUCSL": "소비자물가(CPI)",
        "UNRATE": "실업률",
        "FEDFUNDS": "기준금리"
    }
    frames = {}
    for series, name in indicators.items():
        try:
            if not FRED_API_KEY:
                continue
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_API_KEY}&file_type=json"
            r = requests.get(url, timeout=10)
            obs = pd.DataFrame(r.json().get("observations", []))
            obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
            obs["date"] = pd.to_datetime(obs["date"])
            obs = obs.dropna().tail(12)
            frames[name] = obs
        except Exception as e:
            print(f"[FRED] {name} 로드 실패: {e}")

    if not frames:
        return "<p>📊 주요 경제지표 데이터를 불러올 수 없습니다 (FRED 키 필요)</p>"

    html = "<h4>📊 주요 경제지표 월별 변화</h4>"
    for name, df in frames.items():
        html += f"<h5>{name}</h5><table border='1' cellpadding='5' style='border-collapse:collapse;'><tr><th>월</th><th>값</th><th>전월 대비</th></tr>"
        prev_val = None
        for _, row in df.iterrows():
            val = row["value"]
            diff = ""
            color = "black"
            if prev_val is not None:
                delta = val - prev_val
                diff = f"{delta:+.2f}"
                color = "red" if delta > 0 else "blue"
            html += f"<tr><td>{row['date'].strftime('%Y-%m')}</td><td>{val:.2f}</td><td style='color:{color}'>{diff}</td></tr>"
            prev_val = val
        html += "</table>"

    # GPT 해석
    gpt_prompt = f"""
최근 12개월 미국 경제지표입니다:
{ {name: df[['date','value']].to_dict(orient='records') for name, df in frames.items()} }

각 지표별로 변화 추세를 분석하고, 투자자 입장에서 인플레이션 압력, 경기 둔화/회복, 금리 전망을 bullet point로 제시하세요.
"""
    gpt_out = gpt_chat(gpt_prompt)
    html += "<div style='margin-top:10px; padding:8px; background:#f6f6f6; border-radius:8px;'>"
    html += gpt_out.replace("\n", "<br>")
    html += "</div>"
    return html

def get_market_outlook_html():
    indices_html = get_indices_status_html()
    gpt_prompt = f"""
오늘 주요 지수 현황:
{indices_html}

작업:
- 전일 대비 상승/하락을 간단히 분석
- 단기 시장 심리 (위험선호 / 위험회피) 평가
- 기술주, 배당주, 채권시장 투자 전략 bullet point 제안
"""
    gpt_out = gpt_chat(gpt_prompt)
    return f"<h4>📈 주요 지수 및 시장 전망</h4>{indices_html}<div style='margin-top:8px; background:#f0f0f0; padding:6px; border-radius:8px;'>{gpt_out.replace('\n','<br>')}</div>"
    
def generate_profit_chart():
    """Bar chart of per-stock profit -> returns base64 img tag"""
    tickers = list(portfolio.keys())
    profits = []
    for t, info in portfolio.items():
        price_today, _ = get_stock_prices(t)
        if price_today is None:
            price_today = info["avg_price"]
        profit = (price_today - info["avg_price"]) * info["shares"]
        profits.append(profit)

    plt.figure(figsize=(8, 3))
    bars = plt.bar(tickers, profits)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("종목별 손익")
    plt.ylabel("손익 (USD)")
    # color coding
    for bar, val in zip(bars, profits):
        bar.set_color('green' if val >= 0 else 'red')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"<img src='data:image/png;base64,{img_b64}' alt='profit chart'/>"

def get_alerts_html():
    html = "<h3>🚨 수익률 경고</h3>"
    items = []
    for t, info in portfolio.items():
        price_today, _ = get_stock_prices(t)
        if price_today is None:
            continue
        rate = ((price_today - info["avg_price"]) / info["avg_price"]) * 100
        if rate > 20:
            items.append(f"<li><strong>{t}</strong>: 수익률 {rate:.2f}% → 수익 실현 고려</li>")
    if not items:
        return "<h3>🚨 수익률 경고</h3><p>현재 경고 조건에 해당하는 종목 없음</p>"
    return "<h3>🚨 수익률 경고</h3><ul>" + "".join(items) + "</ul>"

def get_market_icon_legend_html():
    html = "<h4>아이콘 설명</h4><table border='1' cellpadding='5' style='border-collapse:collapse;'>"
    html += "<tr><td>🚀</td><td>강한 상승 기대</td></tr>"
    html += "<tr><td>📈</td><td>상승 기대</td></tr>"
    html += "<tr><td>⚖️</td><td>중립</td></tr>"
    html += "<tr><td>⚠️</td><td>하락 우려</td></tr>"
    html += "</table>"
    return html

# ---------------------------
# News summary
# ---------------------------
def get_news_summary_html():
    """Fetch up to 3 news articles per ticker (NewsAPI) and summarize with GPT.
       Titles are links; GPT produces numbered summaries + short term / long term points.
    """
    html = "<div>"
    for ticker in portfolio.keys():
        html += f"<div style='border:1px solid #ddd; padding:10px; margin:8px 0; border-radius:8px;'>"
        html += f"<h4>{ticker} 관련 뉴스</h4>"

        # fetch articles
        articles = []
        if NEWS_API_KEY:
            try:
                params = {
                    "q": ticker,
                    "pageSize": 3,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "apiKey": NEWS_API_KEY
                }
                r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
                j = r.json()
                articles = j.get("articles", [])[:3]
            except Exception as e:
                print(f"[NewsAPI] {ticker} 에러: {e}")
        else:
            # fallback: no NewsAPI key -> skip external fetch
            articles = []

        if not articles:
            html += "<p style='color:gray;'>관련 뉴스 없음 또는 NEWS_API_KEY 미설정</p></div>"
            continue

        # list titles as links
        html += "<ul>"
        articles_text = ""
        for idx, a in enumerate(articles, start=1):
            title = a.get("title") or "제목 없음"
            link = a.get("url") or "#"
            desc = a.get("description") or ""
            html += f"<li><a href='{link}' target='_blank'>{title}</a></li>"
            articles_text += f"[{idx}] 제목: {title}\n설명: {desc}\n링크: {link}\n"
        html += "</ul>"

        # GPT prompt: numbered summary + short/long implication
        prompt = f"""
아래는 {ticker} 관련 최근 뉴스 3개(제목+설명)입니다:

{articles_text}

작업:
1) 각 기사별로 [1], [2], [3] 번호를 붙여 한국어로 핵심을 bullet point로 정리하세요.
2) 마지막에 '📌 단기 시사점' 과 '📌 장기 시사점'을 줄바꿈하여 제시하세요.
3) 출력은 읽기 쉽게 줄바꿈/번호/불릿을 사용하세요.
"""
        summary = gpt_chat(prompt)
        summary = summary.replace("```", "")
        summary_html = summary.replace("\n", "<br>")
        html += f"<div style='margin-left:12px; color:#333;'>{summary_html}</div>"
        html += "</div>"

    html += "</div>"
    return html

# ---------------------------
# Indicators + GPT analysis incl. sell levels
# ---------------------------
def get_portfolio_indicators_html():
    """Table of RSI/MACD and fundamentals + GPT-driven interpretation with 1st/2nd sell and stoploss"""
    html = "<h4>📊 종목별 판단 지표</h4>"
    html += "<table border='1' cellpadding='5' style='border-collapse:collapse;'>"
    html += "<tr><th>종목</th><th>RSI</th><th>MACD(hist)</th><th>PER</th><th>Fwd PER</th><th>PBR</th><th>ROE</th><th>EPS</th><th>부채비율</th></tr>"

    indicators_for_gpt = {}
    current_prices = {}

    for t, info in portfolio.items():
        # fetch info safely
        try:
            stock = yf.Ticker(t)
            yinfo = stock.info or {}
        except Exception:
            yinfo = {}

        per = yinfo.get("trailingPE", "N/A")
        fwd_per = yinfo.get("forwardPE", "N/A")
        pbr = yinfo.get("priceToBook", "N/A")
        roe = yinfo.get("returnOnEquity", "N/A")
        eps = yinfo.get("trailingEps", "N/A")
        debt_to_equity = yinfo.get("debtToEquity", "N/A")

        rsi, macd = get_rsi_macd_values(t, period="365d")
        rsi_disp = f"{rsi:.2f}" if isinstance(rsi, (int, float)) else "데이터 부족"
        macd_disp = f"{macd:.4f}" if isinstance(macd, (int, float)) else "데이터 부족"

        price_today, _ = get_stock_prices(t)
        if price_today is None:
            price_today = info["avg_price"]

        current_prices[t] = price_today
        indicators_for_gpt[t] = {
            "현재가": price_today,
            "RSI": rsi_disp,
            "MACD": macd_disp,
            "PER": per,
            "Forward PER": fwd_per,
            "PBR": pbr,
            "ROE": roe,
            "EPS": eps,
            "부채비율": debt_to_equity
        }

        html += (f"<tr><td>{t}</td><td>{rsi_disp}</td><td>{macd_disp}</td>"
                 f"<td>{per}</td><td>{fwd_per}</td><td>{pbr}</td><td>{roe}</td><td>{eps}</td><td>{debt_to_equity}</td></tr>")

    html += "</table>"

    # GPT: interpret and give 1st/2nd sell and stoploss
    prompt = f"""
아래는 종목별 주요 지표와 현재가입니다:

{indicators_for_gpt}

작업:
1) 각 종목별로 종목명을 굵게 표기한 뒤 bullet point로 다음을 정리하세요:
   - 기술적 지표(RSI, MACD) 해석
   - 재무 지표(PER, PBR, ROE, EPS, 부채비율) 해석
   - 1차 매도 목표가: 현재가 대비 +5~10% 수준 (달러로 제시)
   - 2차 매도 목표가: 현재가 대비 +15~20% 수준 (달러로 제시)
   - 손절가(Stop-loss): 현재가 대비 -5~10% 수준 (달러로 제시)
   - 📌 투자자 시사점 (단기/장기)
2) 한국어로 간결하게 정리하세요. 가격은 달러 기호($) 포함.
"""
    comments = gpt_chat(prompt)
    comments = comments.replace("```", "")
    comments_html = comments.replace("\n", "<br>")
    html += "<h4>🔎 종목별 지표 해석 + 매도/손절 전략</h4>"
    html += f"<div style='margin-left:12px; color:#333;'>{comments_html}</div>"
    return html

# ---------------------------
# Indices & Economic table
# ---------------------------
def get_indices_status_html():
    """Basic major indices snapshot"""
    index_map = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^VIX": "VIX",
        "^TNX": "US 10Y"
    }
    html = "<table border='1' cellpadding='5' style='border-collapse:collapse;'><tr><th>지수</th><th>현재값</th><th>전일대비</th><th>해석</th></tr>"
    for symbol, name in index_map.items():
        try:
            hist = yf.Ticker(symbol).history(period="2d")["Close"]
            if getattr(hist, "empty", True) or len(hist) < 2:
                html += f"<tr><td>{name}</td><td colspan='3'>데이터 부족</td></tr>"
                continue
            today = float(hist.iloc[-1])
            yesterday = float(hist.iloc[-2])
            change = today - yesterday
            pct = (change / yesterday) * 100
            color = "green" if change >= 0 else "red"
            html += f"<tr><td>{name}</td><td>{today:.2f}</td><td style='color:{color}'>{change:+.2f} ({pct:+.2f}%)</td><td>-</td></tr>"
        except Exception as e:
            html += f"<tr><td>{name}</td><td colspan='3'>에러: {e}</td></tr>"
    html += "</table>"
    return html

def get_economic_table_html():
    """Simple placeholder economic indicators using FRED if available"""
    html = "<h4>📊 주요 경제지표 (요약)</h4>"
    try:
        # If FRED key available, attempt a simple fetch for FEDFUNDS (as example)
        if FRED_API_KEY:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={FRED_API_KEY}&file_type=json"
            r = requests.get(url, timeout=10).json()
            obs = r.get("observations", [])
            latest = obs[-1]["value"] if obs else "N/A"
            html += f"<p>미국 기준금리 (FEDFUNDS): {latest}</p>"
        else:
            html += "<p>FRED API 키 미설정(또는 비공개) — 간단 요약만 표시합니다.</p>"
    except Exception as e:
        html += f"<p>경제지표 로드 실패: {e}</p>"
    return html

# ---------------------------
# US Economic Calendar (TradingEconomics)
# ---------------------------
def get_us_economic_calendar_html():
    try:
        today = datetime.today()
        start_date = today.replace(day=1).strftime("%Y-%m-%d")
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        end_date = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://api.tradingeconomics.com/calendar?country=united states&start={start_date}&end={end_date}&c={TRADING_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json() if r.status_code == 200 else []

        if not data:
            gpt_fallback = gpt_chat("""
이번 달 미국 주요 경제 이벤트 예상 (CPI, PPI, FOMC, 실업률 발표 등)에 대해
투자자가 주의할 포인트를 bullet point로 작성해 주세요.
""")
            return f"<p>TradingEconomics 데이터 없음</p><div style='background:#f6f6f6;padding:8px;border-radius:8px;'>{gpt_fallback.replace('\n','<br>')}</div>"

        html = "<h4>🗓️ 이번 달 미국 경제 발표 일정</h4><table border='1' cellpadding='5' style='border-collapse:collapse;'><tr><th>날짜</th><th>이벤트</th><th>실제/예상</th></tr>"
        for ev in data:
            html += f"<tr><td>{ev.get('Date','')}</td><td>{ev.get('Event','')}</td><td>{ev.get('Actual','')} / {ev.get('Forecast','')}</td></tr>"
        html += "</table>"
        return html
    except Exception as e:
        return f"<p>경제 캘린더 로드 실패: {e}</p>"

# ---------------------------
# Email send
# ---------------------------
def send_email_html(subject: str, html_body: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("메일 관련 환경변수(EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER)가 설정되어 있지 않습니다. 메일 전송 생략.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    part = MIMEText(html_body, "html")
    msg.attach(part)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("이메일 발송 완료")
    except Exception as e:
        print(f"메일 전송 실패: {e}")

# ---------------------------
# Investment assessment (portfolio-level)
# ---------------------------
def get_investment_assessment_html():
    # gather current prices
    current_prices = {t: (get_stock_prices(t)[0] or portfolio[t]["avg_price"]) for t in portfolio.keys()}
    cash_ratio = (CASH_BALANCE / (sum((current_prices[t]*portfolio[t]['shares'] for t in portfolio)) + CASH_BALANCE)) * 100
    context = ("지금은 MDT 오전, 시장 개장 전" if 6 <= datetime.now().hour < 12 else "지금은 MDT 오후, 장 마감 후")

    prompt = f"""
{context}
포트폴리오: {portfolio}
현재가: {current_prices}
계좌현금: ${CASH_BALANCE:.2f} (현금비중: {cash_ratio:.2f}%)

작업:
1) 포트폴리오 전체 전략(방어/공격/리밸런싱 권장 등)을 bullet point로 작성하세요.
2) 각 종목별로 다음을 포함한 전략을 작성하세요:
   - 간단한 지표 해석 요약
   - 1차 매도 목표가(현재가 +5~10%), 2차 매도(현재가 +15~20%), 손절가(현재가 -5~10%)를 달러 단위로 제시
   - 간단한 이유(기술적/재무/뉴스)
3) 한국어로 간결하게 정리하세요.
"""
    out = gpt_chat(prompt)
    out = out.replace("```", "")
    return "<div style='margin-left:12px;'>" + out.replace("\n", "<br>") + "</div>"

# ---------------------------
# Main report assembly
# ---------------------------
def daily_report_html():
    today_str = datetime.today().strftime("%Y-%m-%d")
    alerts_html = get_alerts_html()
    chart_html = generate_profit_chart()
    portfolio_overview = get_portfolio_overview_html()
    portfolio_indicators = get_portfolio_indicators_html()
    news_html = get_news_summary_html()
    assessment_html = get_investment_assessment_html()
    monthly_economic_html = get_monthly_economic_indicators_html()
    market_outlook_html = get_market_outlook_html()
    calendar_html = get_us_economic_calendar_html()
    icons_html = get_market_icon_legend_html()

    body = f"""
    <html><body style="font-family: Arial, sans-serif; line-height:1.5;">
    <h2>📊 오늘의 투자 리포트 ({today_str})</h2>
    {alerts_html}
    <h3>💹 포트폴리오 손익 차트</h3>
    {chart_html}
    <h3>💼 전체 포트폴리오 현황</h3>
    {portfolio_overview}
    <h3>📊 종목별 판단 지표</h3>
    {portfolio_indicators}
    <h3>📰 종목별 뉴스 요약</h3>
    {news_html}
    <h3>🧐 투자 전략 종합 평가</h3>
    {assessment_html}
    <h3>📊 주요 경제지표 월별 변화</h3>
    {monthly_economic_html}
    <h3>📈 주요 지수 및 시장 전망</h3>
    {market_outlook_html}
    <h3>🗓️ 이번 달 미국 경제 발표 일정</h3>
    {calendar_html}
    <hr>
    {icons_html}
    </body></html>
    """
    send_email_html(f"오늘의 투자 리포트 - {today_str}", body)
    print("리포트 생성 및 발송 시도 완료.")

# ---------------------------
# Run guard (skip weekends)
# ---------------------------
if __name__ == "__main__":
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        print("주말이므로 리포트를 실행하지 않습니다.")
        exit(0)
    daily_report_html()
