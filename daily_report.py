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

def get_portfolio_indicators_html():
    html = "<h4>📊 종목별 판단 지표</h4><table border='1'><tr><th>종목</th><th>RSI</th><th>MACD</th><th>PER</th><th>PBR</th><th>ROE</th></tr>"
    indicators = {}
    for t, info in portfolio.items():
        yinfo = yf.Ticker(t).info or {}
        rsi, macd = get_rsi_macd_values(t)
        indicators[t] = {
            "RSI": rsi,
            "MACD": macd,
            "PER": yinfo.get("trailingPE"),
            "PBR": yinfo.get("priceToBook"),
            "ROE": yinfo.get("returnOnEquity"),
        }
        html += "<tr>"
        html += f"<td>{t}</td>"
        html += f"<td>{rsi:.2f}</td>" if rsi else "<td>N/A</td>"
        html += f"<td>{macd:.2f}</td>" if macd else "<td>N/A</td>"
        html += f"<td>{yinfo.get('trailingPE'):.2f}</td>" if yinfo.get("trailingPE") else "<td>N/A</td>"
        html += f"<td>{yinfo.get('priceToBook'):.2f}</td>" if yinfo.get("priceToBook") else "<td>N/A</td>"
        html += f"<td>{yinfo.get('returnOnEquity'):.2f}</td>" if yinfo.get("returnOnEquity") else "<td>N/A</td>"
        html += "</tr>"
    html += "</table>"

    # GPT 해석 + 투자전략 코멘트
    gpt_out = gpt_chat(
        f"종목별 지표: {indicators}\n"
        f"각 종목의 RSI/MACD 해석 + 1차/2차 매도 목표가(+5%, +15%)와 손절가(-7%) 추천 bullet point 작성"
    )
    gpt_html = "<ul>" + "".join(
        [f"<li>{line.strip('-• ').capitalize()}</li>" for line in gpt_out.splitlines() if line.strip()]
    ) + "</ul>"
    html += f"<div style='background:#f6f6f6;padding:8px;border-radius:8px;'>{gpt_html}</div>"
    return html

def get_news_summary_html():
    html = "<h3>📰 종목별 뉴스</h3>"
    for t in portfolio:
        html += f"<h4>📌 {t}</h4>"
        if not NEWS_API_KEY:
            html += "<p style='color:gray;'>NEWS_API_KEY 없음 → 뉴스 생략</p>"
            continue
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": t, "apiKey": NEWS_API_KEY, "pageSize": 3, "sortBy": "publishedAt"},
                timeout=10,
            )
            articles = r.json().get("articles", [])[:3]
            if not articles:
                html += "<p style='color:gray;'>뉴스 없음</p>"
                continue

            # 뉴스 제목 + 요약 텍스트
            news_text = ""
            html += "<ul>"
            for i, a in enumerate(articles, 1):
                title = a.get("title", "제목 없음")
                url = a.get("url", "#")
                desc = a.get("description", "")
                html += f"<li><a href='{url}'>{i}. {title}</a></li>"
                news_text += f"[{i}] {title} - {desc}\n"
            html += "</ul>"

            # GPT 요약
            summary = gpt_chat(f"{t} 관련 뉴스:\n{news_text}\n각 기사 핵심 bullet + 단기/장기 시사점 작성")
            gpt_html = "<ul>" + "".join(
                [f"<li>{line.strip('-• ').capitalize()}</li>" for line in summary.splitlines() if line.strip()]
            ) + "</ul>"
            html += f"<div style='background:#eef;padding:8px;border-radius:8px;'>{gpt_html}</div>"
        except Exception as e:
            html += f"<p style='color:red;'>뉴스 로드 실패: {e}</p>"
    return html

def get_market_outlook_html():
    indices = {
        "^GSPC": "S&P500",
        "^IXIC": "NASDAQ",
        "^DJI": "DowJones",
        "^VIX": "VIX (공포지수)",
        "^TNX": "미국 10년물 국채",
        "GC=F": "Gold"
    }
    html = "<h4>📈 주요 지수 및 시장 전망</h4><table border='1'><tr><th>지수</th><th>현재</th><th>전일대비</th></tr>"
    idx_data = {}
    for sym, name in indices.items():
        try:
            hist = yf.Ticker(sym).history(period="2d")["Close"]
            today, yesterday = hist.iloc[-1], hist.iloc[-2]
            change = today - yesterday
            idx_data[name] = {"today": float(today), "change": float(change)}
            html += f"<tr><td>{name}</td><td>{today:.2f}</td><td style='color:{'green' if change>=0 else 'red'}'>{change:+.2f}</td></tr>"
        except:
            html += f"<tr><td>{name}</td><td colspan='2'>데이터 없음</td></tr>"
    html += "</table>"

    # GPT 해석
    gpt_out = gpt_chat(f"오늘 주요 지수: {idx_data} 투자 전략 bullet point 작성")
    gpt_html = "<ul>" + "".join(
        [f"<li>{line.strip('-• ').capitalize()}</li>" for line in gpt_out.splitlines() if line.strip()]
    ) + "</ul>"
    html += f"<div style='background:#f0f0f0;padding:8px;border-radius:8px;'>{gpt_html}</div>"
    return html

def get_monthly_economic_indicators_html():
    indicators = {"CPIAUCSL": "CPI", "UNRATE": "실업률"}
    frames = {}
    for s, n in indicators.items():
        try:
            if not FRED_API_KEY:
                continue
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s}&api_key={FRED_API_KEY}&file_type=json"
            obs = pd.DataFrame(requests.get(url, timeout=10).json().get("observations", []))
            obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
            obs["date"] = pd.to_datetime(obs["date"])
            frames[n] = obs.dropna().tail(6)
        except:
            pass

    if not frames:
        return "<p style='color:gray;'>📊 경제지표 로드 실패</p>"

    # 가로 테이블 구성
    html = "<h4>📊 경제지표 월별 변화</h4><table border='1'><tr><th>지표</th>"
    months = frames[list(frames.keys())[0]]["date"].dt.strftime("%Y-%m").tolist()
    for m in months:
        html += f"<th>{m}</th>"
    html += "</tr>"

    for name, df in frames.items():
        html += f"<tr><td>{name}</td>"
        for val in df["value"]:
            html += f"<td>{val:.2f}</td>"
        html += "</tr>"
    html += "</table>"

    # GPT 해석
    gpt_out = gpt_chat(f"최근 경제지표: {frames} 투자자 관점에서 bullet point로 해석")
    gpt_html = "<ul>" + "".join(
        [f"<li>{line.strip('-• ').capitalize()}</li>" for line in gpt_out.splitlines() if line.strip()]
    ) + "</ul>"
    html += f"<div style='background:#f6f6f6;padding:8px;border-radius:8px;'>{gpt_html}</div>"
    return html

def get_us_economic_calendar_html():
    try:
        today = datetime.today()
        start = today.replace(day=1).strftime("%Y-%m-%d")
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://api.tradingeconomics.com/calendar?country=united states&start={start}&end={end}&c={TRADING_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json() if r.status_code == 200 else []
        if not data:
            fallback = gpt_chat("이번 달 미국 주요 경제 이벤트 예상 bullet point 작성")
            fallback_html = "<ul>" + "".join(
                [f"<li>{line.strip('-• ').capitalize()}</li>" for line in fallback.splitlines() if line.strip()]
            ) + "</ul>"
            return f"<h4>🗓️ 경제 일정</h4><p style='color:gray;'>TradingEconomics 데이터 없음</p><div style='background:#f8f8f8;padding:8px;border-radius:8px;'>{fallback_html}</div>"

        html = "<h4>🗓️ 이번 달 미국 경제 발표 일정</h4><table border='1'><tr><th>날짜</th><th>이벤트</th><th>예상</th></tr>"
        for ev in data:
            html += f"<tr><td>{ev.get('Date')}</td><td>{ev.get('Event')}</td><td>{ev.get('Forecast')}</td></tr>"
        html += "</table>"
        return html
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
    today = datetime.today()
    today_str = today.strftime("%Y-%m-%d")

    html = f"""
    <html><body style="font-family:Arial, sans-serif;">
    <h2>📊 오늘의 투자 리포트 ({today_str})</h2>

    {get_portfolio_overview_html()}
    {generate_profit_chart()}

    {get_portfolio_indicators_html()}

    {get_news_summary_html()}

    {get_market_outlook_html()}

    {get_monthly_economic_indicators_html()}

    {get_us_economic_calendar_html()}

    </body></html>
    """
    send_email_html(f"오늘의 투자 리포트 - {today_str}", html)
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
