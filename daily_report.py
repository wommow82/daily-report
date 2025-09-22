#!/usr/bin/env python3
# coding: utf-8

import os, time, subprocess, io, base64, smtplib, requests
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import openai

# ============================
# 한글 폰트 설정
# ============================
def setup_matplotlib_korean_font():
    try:
        fonts = [f.name for f in fm.fontManager.ttflist]
        if not any("NanumGothic" in f for f in fonts):
            print("한글 폰트(NanumGothic) 설치 중...")
            subprocess.run(["sudo", "apt-get", "install", "-y", "fonts-nanum"], check=False)
            matplotlib.font_manager._rebuild()  # 🔧 폰트 캐시 갱신
        # 🔧 NanumGothic 적용
        matplotlib.rcParams["font.family"] = "NanumGothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
        print("✅ NanumGothic 폰트 적용 완료")
    except Exception as e:
        print(f"⚠️ 폰트 설정 실패: {e} → DejaVu Sans 사용")
        matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ============================
# 환경 변수 & 포트폴리오 설정
# ============================
portfolio = {
    "NVDA": {"shares": 50, "avg_price": 123.971},
    "PLTR": {"shares": 10, "avg_price": 151.60},
    "SCHD": {"shares": 2140, "avg_price": 24.3777},
    "TSLA": {"shares": 10, "avg_price": 320.745},
}
CASH_BALANCE = 16684.93

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
TRADING_API_KEY = os.getenv("TRADINGECONOMICS_API_KEY", "guest:guest")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4o-mini"

# ============================
# GPT 호출 함수 (재시도 포함)
# ============================
def gpt_chat(prompt, retries=3):
    for i in range(retries):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            return resp.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            wait = 5 * (i + 1)
            print(f"[GPT] Rate limit, {wait}s 대기 후 재시도")
            time.sleep(wait)
        except Exception as e:
            print(f"[GPT] 오류: {e}")
            time.sleep(5)
    return "GPT 요청 실패"

# ============================
# 유틸리티 함수
# ============================
def get_usd_to_cad_rate():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=CAD", timeout=10)
        return float(r.json().get("rates", {}).get("CAD", 1.38))
    except:
        return 1.38

def get_stock_prices(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")["Close"]
        if len(hist) == 0: return None, None
        if len(hist) == 1: return float(hist.iloc[-1]), None
        return float(hist.iloc[-1]), float(hist.iloc[-2])
    except:
        return None, None

def get_rsi_macd_values(ticker, period="365d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        close = df["Close"].dropna()
        if len(close) < 15: return None, None
        delta = close.diff()
        gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
        avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return rsi.dropna().iloc[-1], hist.dropna().iloc[-1]
    except:
        return None, None

# ============================
# 리포트 섹션
# ============================

def get_total_profit():
    """포트폴리오 전체 손익과 수익률 계산"""
    total_cost = 0
    total_value = 0
    for t, info in portfolio.items():
        ticker = yf.Ticker(t)
        price = ticker.history(period="1d")["Close"].iloc[-1]
        qty = info.get("quantity", 0)
        avg = info.get("avg_price", 0)
        total_cost += avg * qty
        total_value += price * qty
    profit = total_value - total_cost
    profit_rate = (profit / total_cost * 100) if total_cost > 0 else 0
    return profit, profit_rate
    
def get_portfolio_overview_html():
    usd_to_cad = get_usd_to_cad_rate()
    total_value = total_cost = total_profit = total_daily = 0
    html = "<h3>💼 전체 포트폴리오</h3><table border='1'><tr><th>종목</th><th>수량</th><th>현재가</th><th>평단</th><th>일일손익</th><th>누적손익</th><th>수익률</th></tr>"
    for t, info in portfolio.items():
        price_today, price_yesterday = get_stock_prices(t)
        if price_today is None:
            price_today = price_yesterday = info["avg_price"]
        daily_profit = (price_today - price_yesterday) * info["shares"]
        cost = info["avg_price"] * info["shares"]
        value = price_today * info["shares"]
        profit = value - cost
        rate = (profit / cost) * 100
        total_value += value; total_cost += cost
        total_profit += profit; total_daily += daily_profit

        # 수익률 강조 색상
        rate_color = "green" if rate >= 0 else "red"
        rate_icon = "🟢" if rate >= 10 else ("🟠" if abs(rate) >= 5 else "")
        html += f"<tr><td>{t}</td><td>{info['shares']}</td><td>{price_today:.2f}</td><td>{info['avg_price']:.2f}</td><td style='color:{'green' if daily_profit>=0 else 'red'}'>{daily_profit:+.2f}</td><td style='color:{'green' if profit>=0 else 'red'}'>{profit:+.2f}</td><td style='color:{rate_color}'>{rate_icon} {rate:+.2f}%</td></tr>"

    html += f"<tr><td><b>합계</b></td><td>-</td><td>-</td><td>-</td><td>{total_daily:+.2f}</td><td>{total_profit:+.2f}</td><td>{(total_profit/total_cost)*100:.2f}%</td></tr></table>"
    html += f"<p>💰 현금 보유액: {CASH_BALANCE:.2f}$ (비중 {(CASH_BALANCE/(total_value+CASH_BALANCE))*100:.2f}%)</p>"
    html += f"<p>총 평가금액: {total_value + CASH_BALANCE:.2f}$ / {(total_value + CASH_BALANCE)*usd_to_cad:.2f} CAD</p>"
    return html

def generate_profit_chart():
    tickers = []; profits = []
    for t, info in portfolio.items():
        price, _ = get_stock_prices(t)
        if price is None: price = info["avg_price"]
        profit = (price - info["avg_price"]) * info["shares"]
        tickers.append(t); profits.append(profit)
    plt.figure(figsize=(6,3))
    bars = plt.bar(tickers, profits)
    for b, val in zip(bars, profits): b.set_color('green' if val>=0 else 'red')
    plt.axhline(0, color='gray', linestyle='--'); plt.title("종목별 손익")
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png')
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return f"<img src='data:image/png;base64,{img}'/>"

import pandas as pd

def get_portfolio_indicators_html():
    rows = []
    for t, info in portfolio.items():
        t_upper = t.upper()
        yinfo = yf.Ticker(t).info or {}
        rsi, macd = get_rsi_macd_values(t)
        rsi_text = "과매수" if rsi and rsi > 70 else ("과매도" if rsi and rsi < 30 else "중립")
        macd_text = "상승 추세" if macd and macd > 0 else ("하락 추세" if macd and macd < 0 else "중립")

        rows.append({
            "종목": f"<b>{t_upper}</b>",
            "RSI": f"{rsi:.2f} ({rsi_text})" if rsi else "N/A",
            "MACD": f"{macd:.2f} ({macd_text})" if macd else "N/A",
            "1차 매도": "+5%",
            "2차 매도": "+15%",
            "손절": "-7%"
        })

    df = pd.DataFrame(rows)
    table_html = df.to_html(escape=False, index=False, justify="center", border=1)

    # GPT 해석 (추가 코멘트)
    gpt_out = gpt_chat(
        "다음 종목별 RSI와 MACD를 기반으로 간단한 해석과 매매 전략 코멘트를 작성해줘. "
        "각 종목별로 핵심 포인트는 굵게 표시하고, 추가 설명은 줄바꿈 + 들여쓰기."
    )
    strategy_html = "".join(
        f"<p style='margin-left:15px;'>{line}</p>" for line in gpt_out.splitlines() if line.strip()
    )

    return f"""
    <div style='background:#f9f9f9;padding:10px;border-radius:8px;'>
        <h4>📊 종목별 판단 지표</h4>
        {table_html}
        <h4>📌 해석 및 전략</h4>
        {strategy_html}
    </div>
    """

def get_news_summary_html():
    html = "<h3>📰 종목별 뉴스</h3>"
    for t in portfolio:
        t_upper = t.upper()
        html += f"<h4>📌 <b>{t_upper}</b></h4>"
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": t, "apiKey": NEWS_API_KEY, "pageSize": 6, "sortBy": "publishedAt"},
                timeout=10,
            )
            articles = r.json().get("articles", [])
            filtered = [a for a in articles if t_upper in (a.get("title","")+a.get("description","")).upper()][:3]
            if not filtered:
                html += "<p style='color:gray;'>관련 뉴스 없음</p>"
                continue

            news_text = ""
            for i, a in enumerate(filtered, 1):
                title = a.get("title", "제목 없음")
                desc = a.get("description", "")
                url = a.get("url", "#")
                html += f"<p><b>{i}. <a href='{url}'>{title}</a></b></p>"
                if desc:
                    html += f"<p style='margin-left:20px;color:#555;'>{desc}</p>"
                news_text += f"[{i}] {title} - {desc}\n"

            # GPT 번역 요약
            summary = gpt_chat(
                f"{t_upper} 관련 뉴스:\n{news_text}\n"
                "위 기사 내용을 한국어로 요약하고, 기사 주제는 굵게 표시, 세부내용은 들여쓰기 + 불릿으로 정리."
            )
            formatted = ""
            for line in summary.splitlines():
                if line.strip().startswith("**"):  # 대주제
                    formatted += f"<p><b>{line.strip('**')}</b></p>"
                elif line.strip():
                    formatted += f"<p style='margin-left:20px;'>• {line.strip()}</p>"

            html += f"<div style='background:#eef;padding:8px;border-radius:8px;'>{formatted}</div>"

        except Exception as e:
            html += f"<p style='color:red;'>뉴스 로드 실패: {e}</p>"

    return html

import yfinance as yf

def get_market_outlook_html():
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
        "US 10Y": "^TNX",
        "Gold": "GC=F",
    }
    data = []

    for name, symbol in tickers.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                price_today = hist["Close"].iloc[-1]
                price_yesterday = hist["Close"].iloc[-2]
                change = ((price_today - price_yesterday) / price_yesterday) * 100
                data.append({
                    "지수": name,
                    "현재": f"{price_today:,.2f}",
                    "변동률": f"{change:+.2f}%",
                    "전략": "기술적 분석에 기반한 전략 작성 예정"
                })
        except Exception as e:
            data.append({"지수": name, "현재": "N/A", "변동률": "N/A", "전략": f"데이터 로드 실패: {e}"})

    df = pd.DataFrame(data)
    table_html = df.to_html(index=False, justify="center", escape=False, border=1)

    gpt_out = gpt_chat("위 지수 데이터를 기반으로 전반적인 투자 전략을 한 줄로 작성해줘.")
    return f"""
    <div style='background:#f9f9f9;padding:10px;border-radius:8px;'>
        {table_html}
        <p><b>📌 전반적 전략:</b> {gpt_out}</p>
    </div>
    """

def fetch_economic_indicators():
    """
    FRED API 기반으로 CPI, 실업률, GDP 성장률, 소매판매 지표를 최근 6개월치 불러와 표로 반환
    """
    indicators = {
        "소비자물가지수(CPI)": "CPIAUCSL",
        "실업률": "UNRATE",
        "GDP 성장률": "A191RL1Q225SBEA",
        "소매판매": "RSAFS",
    }

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")

    data = {"지표": []}
    months = []

    for name, code in indicators.items():
        url = (
            f"{FRED_API_BASE}?series_id={code}&api_key={FRED_API_KEY}"
            f"&file_type=json&observation_start={start_date}&observation_end={end_date}"
        )
        try:
            r = requests.get(url, timeout=10)
            observations = r.json().get("observations", [])
            monthly_values = {}
            for obs in observations:
                date = obs["date"][:7]  # YYYY-MM
                monthly_values[date] = float(obs["value"]) if obs["value"] != "." else None

            if not months:
                months = sorted(list(monthly_values.keys())[-6:])
                for m in months:
                    data[m] = []

            data["지표"].append(name)
            for m in months:
                data[m].append(monthly_values.get(m, None))

        except Exception as e:
            print(f"❌ {name} 로드 실패: {e}")

    return pd.DataFrame(data)

def get_monthly_economic_indicators_html():
    try:
        df = fetch_economic_indicators()
        if df.empty:
            return "<p style='color:gray;'>📊 최근 경제지표 없음</p>"

        return f"""
        <div style='background:#f9f9f9;padding:10px;border-radius:8px;overflow-x:auto;'>
            <h4>📊 주요 경제지표 (최근 6개월)</h4>
            {df.to_html(index=False, justify="center", border=1)}
        </div>
        """
    except Exception as e:
        return f"<p style='color:red;'>경제지표 로드 실패: {e}</p>"

def get_us_economic_calendar_html():
    try:
        events = [
            {"날짜": "2025-10-13", "주제": "소비자물가지수(CPI) 발표", "설명": "예상 인플레이션 변화 및 소비자 지출 패턴 통찰"},
            {"날짜": "2025-10-25", "주제": "FOMC 회의록 공개", "설명": "연준의 금리 정책 방향성 파악"},
        ]
        df = pd.DataFrame(events)
        return f"""
        <div style='background:#f9f9f9;padding:10px;border-radius:8px;'>
            {df.to_html(index=False, justify="center", border=1)}
        </div>
        """
    except Exception as e:
        return f"<p style='color:red;'>경제 일정 로드 실패: {e}</p>"

# ============================
# 메일 전송
# ============================
def send_email_html(subject, html_body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("메일 설정 없음 → 메일 전송 생략")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_SENDER, EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            # 🔧 여기서 msg.as_string() 추가
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("메일 발송 완료")
    except Exception as e:
        print(f"메일 전송 실패: {e}")

# ============================
# 리포트 조립
# ============================
def daily_report_html():
    today = datetime.today().strftime("%Y-%m-%d")
    html = f"""
    <html><body style="font-family:Arial, sans-serif; line-height:1.6;">
    <h2 style="text-align:center;">📊 오늘의 투자 리포트 ({today})</h2>
    <hr>

    <h3>💼 포트폴리오 요약</h3>
    {get_portfolio_overview_html()}

    <h3>📊 종목별 판단 지표</h3>
    {get_portfolio_indicators_html()}

    <h3>📰 종목별 뉴스</h3>
    {get_news_summary_html()}

    <h3>📉 주요 지수 및 시장 전망</h3>
    {get_market_outlook_html()}

    <h3>📆 미국 경제 발표 일정</h3>
    {get_us_economic_calendar_html()}

    <h3>📊 주요 경제지표 월별 변화</h3>
    {get_monthly_economic_indicators_html()}

    </body></html>
    """
    send_email_html(f"오늘의 투자 리포트 - {today}", html)
    print("✅ 리포트 생성 및 메일 발송 완료")

# ============================
# 메인 실행
# ============================
# if __name__ == "__main__":
#     if datetime.now().weekday() >= 5:
#         print("주말이므로 리포트 실행 안 함")
#     else:
#         daily_report_html()

if __name__ == "__main__":
    # 주말에도 실행 → 조건 제거
    daily_report_html()
