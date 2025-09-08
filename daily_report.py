#!/usr/bin/env python3
# coding: utf-8

import os
import yfinance as yf
import requests
import smtplib
import matplotlib.pyplot as plt
import io
import base64
import openai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from googletrans import Translator

# ====== 환경 변수에서 불러오기 (GitHub Secrets) ======
# 환경변수 불러오기
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# 번역기 초기화
translator = Translator()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing. Make sure it's set in GitHub Secrets and passed to the workflow.")

openai.api_key = OPENAI_API_KEY

# ChatGPT 호출 예시
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
except openai.error.RateLimitError as e:
    print("Rate limit exceeded:", e)

# ====== 포트폴리오 구성 ======
portfolio = {
    "NVDA": {"shares": 128, "avg_price": 123.97},
    "PLTR": {"shares": 10, "avg_price": 151.60},
    "RGTI": {"shares": 50, "avg_price": 19.02},
    "SCHD": {"shares": 2140, "avg_price": 24.37},
    "TSLA": {"shares": 10, "avg_price": 320.745},
}

# ====== 환율 가져오기 ======
def get_usd_to_cad_rate():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=CAD"
        res = requests.get(url).json()
        return res["rates"]["CAD"]
    except Exception as e:
        print(f"환율 가져오기 실패: {e}")
        return 1.3829

# ====== 기술적 지표 (RSI & MACD) ======
def get_rsi_macd(ticker):
    data = yf.Ticker(ticker).history(period="60d")
    if data.empty:
        return "데이터 없음"

    close = data["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 0

    if latest_rsi >= 70:
        rsi_status = "📈 과매수"
    elif latest_rsi <= 30:
        rsi_status = "📉 과매도"
    else:
        rsi_status = "⚖️ 중립"

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    if macd_line.empty or signal_line.empty:
        macd_trend = "데이터 없음"
    else:
        macd_trend = "📈 상승" if macd_line.iloc[-1] > signal_line.iloc[-1] else "📉 하락"

    return f"RSI: {latest_rsi:.1f} ({rsi_status}), MACD: {macd_trend}"

# ====== 포트폴리오 HTML ======
def get_portfolio_status_html():
    usd_to_cad = get_usd_to_cad_rate()
    total_usd, total_cad, total_cost, total_profit = 0, 0, 0, 0

    html = "<table border='1' cellpadding='5'>"
    html += "<tr><th>종목</th><th>보유수량</th><th>현재가 / 평단가 (USD)</th><th>총 투자금액 (USD)</th><th>전일 대비</th><th>평가금액 (USD)</th><th>평가금액 (CAD)</th><th>손익 (USD)</th><th>수익률</th><th>RSI / MACD</th></tr>"

    for ticker, info in portfolio.items():
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")["Close"]  # ← 기간 늘림
        if len(hist) < 2:
            html += f"<tr><td>{ticker}</td><td colspan='9' style='color:gray;'>데이터 부족</td></tr>"
            continue

        price_today, price_yesterday = hist.iloc[-1], hist.iloc[-2]
        change = price_today - price_yesterday
        change_rate = (change / price_yesterday) * 100 if price_yesterday != 0 else 0
        change_color = "green" if change > 0 else "red"

        cost = info["avg_price"] * info["shares"]
        value_usd = price_today * info["shares"]
        value_cad = value_usd * usd_to_cad
        profit = value_usd - cost
        rate = (profit / cost) * 100 if cost > 0 else 0
        profit_color = "green" if profit > 0 else "red"
        rate_color = "green" if rate > 0 else "red"
        indicators = get_rsi_macd(ticker)

        total_usd += value_usd
        total_cad += value_cad
        total_cost += cost
        total_profit += profit

        html += f"<tr><td>{ticker}</td><td>{info['shares']}</td>"
        html += f"<td>{price_today:.2f}$ / {info['avg_price']:.2f}$</td><td>{cost:,.2f}$</td>"
        html += f"<td><span style='color:{change_color}'>{change:+.2f}$ ({change_rate:+.2f}%)</span></td>"
        html += f"<td>{value_usd:,.2f}$</td><td>{value_cad:,.2f} CAD</td>"
        html += f"<td><span style='color:{profit_color}'>{profit:+,.2f}$</span></td>"
        html += f"<td><span style='color:{rate_color}'>{rate:+.2f}%</span></td><td>{indicators}</td></tr>"

    total_rate = (total_profit / total_cost) * 100 if total_cost > 0 else 0
    total_profit_color = "green" if total_profit > 0 else "red"
    total_rate_color = "green" if total_rate > 0 else "red"

    html += f"<tr><td colspan='3'><strong>총 투자금액</strong></td><td><strong>{total_cost:,.2f}$</strong></td>"
    html += f"<td></td><td><strong>{total_usd:,.2f}$</strong></td><td><strong>{total_cad:,.2f} CAD</strong></td>"
    html += f"<td><strong><span style='color:{total_profit_color}'>{total_profit:+,.2f}$</span></strong></td>"
    html += f"<td><strong><span style='color:{total_rate_color}'>{total_rate:+.2f}%</span></strong></td><td></td></tr>"
    html += "</table>"
    return html

# ====== 그래프 생성 ======
def generate_profit_chart():
    tickers, profits = [], []
    for ticker, info in portfolio.items():
        hist = yf.Ticker(ticker).history(period="5d")["Close"]
        if hist.empty:
            continue
        price = hist.iloc[-1]
        profit = (price - info["avg_price"]) * info["shares"]
        tickers.append(ticker)
        profits.append(profit)

    if not tickers:
        return "<p>📉 데이터 부족으로 그래프 생성 불가</p>"

    plt.figure(figsize=(8, 4))
    plt.bar(tickers, profits, color=["green" if p > 0 else "red" for p in profits])
    plt.title("종목별 손익 추이")
    plt.ylabel("손익 ($)")
    plt.axhline(0, color='gray', linestyle='--')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"<img src='data:image/png;base64,{img_base64}'/>"

# ====== 이메일 발송 ======
def send_email_html(subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

# ====== 메인 실행 ======
def daily_report_html():
    today = datetime.today().strftime("%Y-%m-%d")
    portfolio_html = get_portfolio_status_html()
    chart_html = generate_profit_chart()

    body = f"""
    <html><body>
    <h2>📊 오늘의 투자 리포트 ({today})</h2>
    {chart_html}
    <h3>💼 포트폴리오 현황</h3>
    {portfolio_html}
    </body></html>
    """
    send_email_html("오늘의 투자 리포트", body)
    print("✅ 이메일 발송 완료")

if __name__ == "__main__":
    daily_report_html()
