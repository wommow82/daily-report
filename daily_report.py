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
        model="gpt-4o-mini",
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

indices = ["^GSPC", "^IXIC", "^DJI", "^VIX", "^TNX"]

# ====== 실시간 환율 가져오기 ======
def get_usd_to_cad_rate():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=CAD"
        res = requests.get(url).json()
        rate = res["rates"]["CAD"]
        return rate
    except Exception as e:
        print(f"환율 가져오기 실패: {e}")
        return 1.3829  # 실패 시 기본값

# ====== 아이콘 설명 ======
def get_market_icon_legend_html():
    html = "<h3 style='margin-left:20px;'>📊 시장 전망 아이콘 설명</h3>"
    html += "<table border='1' cellpadding='5'>"
    html += "<tr><th>아이콘</th><th>의미</th></tr>"
    html += "<tr><td>🚀</td><td>강한 상승 기대 (급등 가능성)</td></tr>"
    html += "<tr><td>📈</td><td>상승 기대 (안정적 상승 흐름)</td></tr>"
    html += "<tr><td>⚖️</td><td>중립 / 혼조세 (방향성 불확실)</td></tr>"
    html += "<tr><td>⚠️</td><td>하락 우려 (주의 필요)</td></tr>"
    html += "<tr><td>📉</td><td>급락 가능성 (강한 하락 압력)</td></tr>"
    html += "<tr><td>🌪️</td><td>불안정 / 변동성 확대 (시장 혼란)</td></tr>"
    html += "<tr><td>🧘</td><td>안정적 흐름 (변동성 낮음)</td></tr>"
    html += "</table><br>"
    return html

    
# ====== 기술적 지표 계산 ======
def get_rsi_macd(ticker):
    data = yf.Ticker(ticker).history(period="60d")
    close = data["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1]

    # RSI 해석 추가
    if latest_rsi >= 70:
        rsi_status = "📈 과매수"
    elif latest_rsi <= 30:
        rsi_status = "📉 과매도"
    else:
        rsi_status = "⚖️ 중립"

    # MACD 계산
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_trend = "📈 상승" if macd_line.iloc[-1] > signal_line.iloc[-1] else "📉 하락"

    return f"RSI: {latest_rsi:.1f} ({rsi_status}), MACD: {macd_trend}"

# ====== 포트폴리오 HTML ======
def get_portfolio_status_html():
    usd_to_cad = get_usd_to_cad_rate()
    total_usd = 0
    total_cad = 0
    total_cost = 0
    total_profit = 0
    total_daily_profit = 0

    # ✅ 종목별 현황 표
    html = "<h4>📌 종목별 현황</h4>"
    html += "<table border='1' cellpadding='5'>"
    html += (
        "<tr>"
        "<th>종목</th>"
        "<th>보유수량</th>"
        "<th>현재가 / 평단가 (USD)</th>"
        "<th>일일 손익 (USD)</th>"
        "<th>누적 손익 (USD)</th>"
        "<th>수익률</th>"
        "<th>RSI / MACD</th>"
        "</tr>"
    )

    for ticker, info in portfolio.items():
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")["Close"]
        price_today = hist.iloc[-1]
        price_yesterday = hist.iloc[-2]

        # 📊 일일 손익
        daily_profit = (price_today - price_yesterday) * info["shares"]
        daily_profit_color = "green" if daily_profit > 0 else "red"

        # 📊 누적 손익
        cost = info["avg_price"] * info["shares"]
        value_usd = price_today * info["shares"]
        profit = value_usd - cost
        profit_color = "green" if profit > 0 else "red"

        # 📊 수익률
        rate = (profit / cost) * 100
        rate_color = "green" if rate > 0 else "red"

        # 기술적 지표
        indicators = get_rsi_macd(ticker)

        # 총합 계산
        total_usd += value_usd
        total_cad += value_usd * usd_to_cad
        total_cost += cost
        total_profit += profit
        total_daily_profit += daily_profit

        # 행 추가
        html += (
            f"<tr><td>{ticker}</td><td>{info['shares']}</td>"
            f"<td>{price_today:.2f}$ / {info['avg_price']:.2f}$</td>"
            f"<td><span style='color:{daily_profit_color}'>{daily_profit:+,.2f}$</span></td>"
            f"<td><span style='color:{profit_color}'>{profit:+,.2f}$</span></td>"
            f"<td><span style='color:{rate_color}'>{rate:+.2f}%</span></td>"
            f"<td>{indicators}</td></tr>"
        )

    html += "</table><br>"

    # ✅ 포트폴리오 요약 표
    total_rate = (total_profit / total_cost) * 100 if total_cost > 0 else 0
    total_profit_color = "green" if total_profit > 0 else "red"
    total_daily_profit_color = "green" if total_daily_profit > 0 else "red"
    total_rate_color = "green" if total_rate > 0 else "red"

    html += "<h4>📌 전체 포트폴리오 요약</h4>"
    html += "<table border='1' cellpadding='5'>"
    html += (
        "<tr>"
        "<th>총 투자금액 (USD)</th>"
        "<th>총 평가금액 (USD)</th>"
        "<th>총 평가금액 (CAD)</th>"
        "<th>오늘 일일 손익 (USD)</th>"
        "<th>총 누적 손익 (USD)</th>"
        "<th>총 수익률</th>"
        "</tr>"
    )

    html += (
        f"<tr><td>{total_cost:,.2f}$</td>"
        f"<td>{total_usd:,.2f}$</td><td>{total_cad:,.2f} CAD</td>"
        f"<td><span style='color:{total_daily_profit_color}'>{total_daily_profit:+,.2f}$</span></td>"
        f"<td><span style='color:{total_profit_color}'>{total_profit:+,.2f}$</span></td>"
        f"<td><span style='color:{total_rate_color}'>{total_rate:+.2f}%</span></td></tr>"
    )

    html += "</table>"
    return html


# ====== 포트폴리오 전체 정리 ======
def get_portfolio_summary_html():
    usd_to_cad = get_usd_to_cad_rate()
    total_usd = 0
    total_cost = 0
    total_profit = 0
    total_daily_profit = 0

    html = "<h4>📌 전체 포트폴리오 요약</h4>"
    html += "<table border='1' cellpadding='5'>"
    html += (
        "<tr>"
        "<th>종목</th><th>보유수량</th>"
        "<th>현재가 (USD)</th><th>일일 손익 (USD)</th>"
        "<th>누적 손익 (USD)</th><th>수익률</th>"
        "</tr>"
    )

    for ticker, info in portfolio.items():
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")["Close"]
        price_today = hist.iloc[-1]
        price_yesterday = hist.iloc[-2]

        daily_profit = (price_today - price_yesterday) * info["shares"]
        cost = info["avg_price"] * info["shares"]
        value_usd = price_today * info["shares"]
        profit = value_usd - cost
        rate = (profit / cost) * 100

        total_usd += value_usd
        total_cost += cost
        total_profit += profit
        total_daily_profit += daily_profit

        daily_color = "green" if daily_profit > 0 else "red"
        profit_color = "green" if profit > 0 else "red"
        rate_color = "green" if rate > 0 else "red"

        html += (
            f"<tr><td>{ticker}</td><td>{info['shares']}</td>"
            f"<td>{price_today:.2f}$</td>"
            f"<td><span style='color:{daily_color}'>{daily_profit:+,.2f}$</span></td>"
            f"<td><span style='color:{profit_color}'>{profit:+,.2f}$</span></td>"
            f"<td><span style='color:{rate_color}'>{rate:+.2f}%</span></td></tr>"
        )

    total_rate = (total_profit / total_cost) * 100 if total_cost > 0 else 0
    total_daily_color = "green" if total_daily_profit > 0 else "red"
    total_profit_color = "green" if total_profit > 0 else "red"
    total_rate_color = "green" if total_rate > 0 else "red"

    # 합계 행
    html += (
        f"<tr><td><strong>합계</strong></td><td>-</td>"
        f"<td>-</td>"
        f"<td><span style='color:{total_daily_color}'><strong>{total_daily_profit:+,.2f}$</strong></td>"
        f"<td><span style='color:{total_profit_color}'><strong>{total_profit:+,.2f}$</strong></td>"
        f"<td><span style='color:{total_rate_color}'><strong>{total_rate:+.2f}%</strong></td></tr>"
    )

    html += "</table>"
    html += f"<p>총 평가금액: {total_usd:,.2f}$ / {total_usd*usd_to_cad:,.2f} CAD</p>"
    return html
    
# ====== 수익 추이 그래프 ======
def generate_profit_chart():
    tickers = list(portfolio.keys())
    profits = []
    for ticker, info in portfolio.items():
        price = yf.Ticker(ticker).history(period="2d")["Close"].iloc[-1]
        profit = (price - info["avg_price"]) * info["shares"]
        profits.append(profit)

    plt.figure(figsize=(8, 4))
    bars = plt.bar(tickers, profits, color=["green" if p > 0 else "red" for p in profits])
    plt.title("종목별 손익 추이")
    plt.ylabel("손익 ($)")
    plt.axhline(0, color='gray', linestyle='--')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"<img src='data:image/png;base64,{img_base64}'/>"
    

# ====== 수익률 경고 알림 ======
def get_alerts_html():
    html = "<h3>🚨 수익률 경고</h3><ul>"
    for ticker, info in portfolio.items():
        price = yf.Ticker(ticker).history(period="2d")["Close"].iloc[-1]
        rate = ((price - info["avg_price"]) / info["avg_price"]) * 100
        if rate > 20:
            html += f"<li><strong>{ticker}</strong>: 수익률 {rate:.2f}% → 수익 실현 고려!</li>"
    html += "</ul>" if html != "<h3>🚨 수익률 경고</h3><ul>" else "<p>⚠️ 현재 수익률 경고 조건에 해당하는 종목 없음</p>"
    return html

# ====== 뉴스 요약 및 번역 함수 (최적화 버전) ======
def get_news_summary_html():
    html = "<h3>📰 종목별 뉴스 요약</h3>"

    for ticker in portfolio.keys():
        html += f"<div style='border:1px solid #ccc; padding:10px; margin:10px; border-radius:10px;'>"
        html += f"<h4>{ticker} 관련 뉴스</h4>"

        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=3&sortBy=publishedAt&language=en"
            response = requests.get(url).json()
            articles = response.get("articles", [])
            if not articles:
                html += "<p style='color:gray;'>관련 뉴스 없음</p>"
                html += "</div>"
                continue

            articles_text = ""
            for idx, article in enumerate(articles, 1):
                title = article.get("title", "제목 없음")
                description = article.get("description", "설명 없음")
                link = article.get("url", "#")
                articles_text += f"\n[{idx}] 제목: {title}\n설명: {description}\n링크: {link}\n"
                html += f"<p>• <a href='{link}' target='_blank'>{title}</a></p>"

            prompt = f"""
아래는 {ticker} 관련 최근 뉴스 3개입니다:

{articles_text}

👉 각 기사별 핵심 요약과 투자자 관점 코멘트를 한국어로 작성하고,
마지막에 {ticker}에 대한 단기/장기 투자 시사점을 정리해 주세요.
"""

            gpt_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            summary = gpt_response.choices[0].message.content.strip()
            # 코드블록 제거
            if summary.startswith("```"):
                summary = summary.replace("```html", "").replace("```", "").strip()

            html += f"<div style='margin-left:20px; color:#444;'>{summary}</div>"

        except Exception as e:
            html += f"<p style='color:gray;'>요약 실패: {e}</p>"

        html += "</div>"
    return html

# ====== 투자 전략 평가 ======
def get_investment_assessment_html():
    try:
        hour = datetime.now().hour  # 서버 기준 (UTC라면 MDT 변환 필요)
        if 6 <= hour < 12:  # 예: MDT 오전
            context = "지금은 MDT 오전, 시장 개장 전입니다. 오늘 장에서 주목해야 할 뉴스와 지표를 기반으로 투자 방향을 제안하세요."
        else:  # 예: MDT 오후
            context = "지금은 MDT 오후, 시장이 마감되었습니다. 오늘 하루 시장 변화를 요약하고, 내일 장에서 주의해야 할 점을 알려주세요."

        prompt = f"""
{context}

📌 포트폴리오 종목:
{portfolio}

📌 요청 사항:
1. 종목별 단기/장기 전략 (매수/보유/매도 권고 포함)
2. 포트폴리오 차원 전략 (현금 비중, 리밸런싱 방향)
3. 한국어로 5~7줄 작성
"""

        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        assessment = gpt_response.choices[0].message.content.strip()
        if assessment.startswith("```"):
            assessment = assessment.replace("```html", "").replace("```", "").strip()

        html = "<h3>🧐 투자 전략 종합 평가</h3>"
        html += f"<div style='margin-left:20px; color:#333;'>{assessment}</div>"
        return html
    except Exception as e:
        return f"<h3>🧐 투자 전략 종합 평가</h3><p style='color:gray;'>평가 생성 실패: {e}</p>"

# ====== 주요 지수 HTML ======
def get_indices_status_html():
    index_info = {
        "^GSPC": ("S&P 500", "미국 대형주 500개로 구성된 대표적인 주가지수.", "미국 증시의 전반적인 흐름을 반영합니다."),
        "^IXIC": ("NASDAQ", "기술주 중심의 지수.", "기술주가 강세일수록 상승 가능성이 높습니다."),
        "^DJI": ("다우존스", "미국의 대표적인 30개 대기업 지수.", "전통 산업과 대형주의 흐름을 보여줍니다."),
        "^VIX": ("VIX", "시장의 불안감 또는 공포 수준.", "높을수록 시장 불안, 낮을수록 안정입니다."),
        "^TNX": ("미국 10년물 국채 수익률", "장기 금리의 기준.", "금리 상승은 부담, 하락은 완화 신호입니다."),
        "GC=F": ("Gold", "안전자산으로서의 금 가격 지수.", "금값 상승은 시장 불안 또는 인플레이션 우려를 반영합니다.")
    }

    html = "<h3 style='margin-left:20px;'>📈 주요 지수 및 시장 전망</h3>"
    html += "<table border='1' cellpadding='5'><tr><th>지수</th><th>현재값</th><th>전일 대비</th><th>시장 전망</th><th>설명</th><th>현재 해석</th></tr>"

    for symbol, (label, desc, insight) in index_info.items():
        try:
            hist = yf.Ticker(symbol).history(period="2d")["Close"]
            if len(hist) < 2:
                raise ValueError("데이터 부족")

            today = hist.iloc[-1]
            yesterday = hist.iloc[-2]
            change = today - yesterday
            change_rate = (change / yesterday) * 100
            change_color = "green" if change > 0 else "red"

            # 시장 전망 아이콘
            if abs(change_rate) < 1:
                outlook_icon = "⚖️"
            elif change_rate >= 3:
                outlook_icon = "🚀"
            elif change_rate > 0:
                outlook_icon = "📈"
            elif change_rate <= -3:
                outlook_icon = "📉"
            else:
                outlook_icon = "⚠️"

            if symbol == "^VIX" and change > 0:
                outlook_icon = "🌪️"
            elif symbol == "^VIX" and change <= 0:
                outlook_icon = "🧘"
            elif symbol == "^TNX" and change > 0:
                outlook_icon = "⚠️"
            elif symbol == "^TNX" and change <= 0:
                outlook_icon = "📈"

            html += f"<tr><td>{label}</td><td>{today:.2f}</td>"
            html += f"<td><span style='color:{change_color}'>{change:+.2f} ({change_rate:+.2f}%)</span></td>"
            html += f"<td>{outlook_icon}</td><td>{desc}</td><td>{insight}</td></tr>"

        except Exception as e:
            html += f"<tr><td>{label}</td><td colspan='5' style='color:gray;'>데이터 오류: {e}</td></tr>"

    html += "</table>"
    return html

# ====== 경제지표 HTML 생성 ======
def get_economic_table_html():
    indicators = {
        "FEDFUNDS": {
            "label": "미국 기준금리 (%)",
            "desc": "대출·소비·투자에 직접적인 영향을 미침.",
            "direction": "down",
            "insight": "금리가 낮아지면 주식시장에 긍정적입니다."
        },
        "CPIAUCSL": {
            "label": "소비자물가지수 (CPI)",
            "desc": "인플레이션 지표.",
            "direction": "down",
            "insight": "물가가 안정되면 금리 인상 부담이 줄어들어 주식에 긍정적입니다."
        },
        "UNRATE": {
            "label": "실업률 (%)",
            "desc": "경기 침체 또는 회복의 신호.",
            "direction": "down",
            "insight": "실업률이 낮아지면 경기 회복 신호로 주식시장에 긍정적입니다."
        }
    }

    months = [str(m).zfill(2) for m in range(1, 13)]
    month_labels = [f"{m}월" for m in months]
    data = {}
    icon_map = {}

    # 데이터 수집 및 아이콘 판단
    for series_id in indicators.keys():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        res = requests.get(url).json()
        obs = res.get("observations", [])
        monthly_values = {o["date"][5:7]: o["value"] for o in obs if o["date"].startswith("2025")}
        data[series_id] = monthly_values

        # 시장 전망 아이콘 계산
        recent_months = sorted(monthly_values.keys())[-3:]
        recent_values = [float(monthly_values[m]) for m in recent_months if monthly_values.get(m, "-") != "-"]
        icon = "⚖️"  # 기본값

        if len(recent_values) >= 2:
            delta = recent_values[-1] - recent_values[-2]
            if indicators[series_id]["direction"] == "down":
                if delta < -0.2:
                    icon = "🚀"
                elif delta < -0.01:
                    icon = "📈"
                elif abs(delta) <= 0.01:
                    icon = "⚖️"
                elif delta > 0.2:
                    icon = "📉"
                elif delta > 0.01:
                    icon = "⚠️"
            elif indicators[series_id]["direction"] == "up":
                if delta > 0.2:
                    icon = "🚀"
                elif delta > 0.01:
                    icon = "📈"
                elif abs(delta) <= 0.01:
                    icon = "⚖️"
                elif delta < -0.2:
                    icon = "📉"
                elif delta < -0.01:
                    icon = "⚠️"

        icon_map[series_id] = icon

    # 📌 첫 번째 표: 요약 정보 (시장 전망 제거)
    html = "<h3 style='margin-left:20px;'>📌 주요 경제지표 요약</h3>"
    html += "<table border='1' cellpadding='5'><tr><th>지표</th><th>설명</th><th>현재 해석</th></tr>"

    for series_id, info in indicators.items():
        html += f"<tr><td>{info['label']}</td><td>{info['desc']}</td><td>{info['insight']}</td></tr>"
    html += "</table><br>"

    # 📊 두 번째 표: 월별 변화 + 아이콘
    html += "<h3 style='margin-left:20px;'>📊 주요 경제지표 월별 변화 (2025년)</h3>"
    html += "<table border='1' cellpadding='5'><tr><th>지표</th>"
    for label in month_labels:
        html += f"<th>{label}</th>"
    html += "</tr>"

    for series_id, info in indicators.items():
        icon = icon_map.get(series_id, "⚖️")
        html += f"<tr><td>{info['label']} {icon}</td>"
        for m in months:
            value = data.get(series_id, {}).get(m, "-")
            html += f"<td>{value}</td>"
        html += "</tr>"

    html += "</table>"
    return html

# ====== 이메일 발송 함수 ======
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

# ====== 메인 리포트 생성 및 실행 ======
def daily_report_html():
    today = datetime.today().strftime("%Y-%m-%d")
    portfolio_html = get_portfolio_status_html()
    portfolio_summary_html = get_portfolio_summary_html()   # ✅ 추가
    indices_html = get_indices_status_html()
    news_summary_html = get_news_summary_html()
    economic_html = get_economic_table_html()
    chart_html = generate_profit_chart()
    alerts_html = get_alerts_html()
    icon_legend_html = get_market_icon_legend_html()
    assessment_html = get_investment_assessment_html()

    body = f"""
    <html><body>
    <h2>📊 오늘의 투자 리포트 ({today})</h2>
    {alerts_html}
    {chart_html}
    <h3>💼 포트폴리오 현황</h3>
    {portfolio_html}
    {portfolio_summary_html}   <!-- ✅ 요약 표 추가 -->
    <h3>📰 종목별 뉴스 요약</h3>
    {news_summary_html}
    {assessment_html}
    {icon_legend_html}
    <h3>📈 주요 지수</h3>
    {indices_html}
    <h3>📊 주요 경제지표</h3>
    {economic_html}
    </body></html>
    """
    send_email_html("오늘의 투자 리포트", body)
    print("✅ 이메일 발송 완료")

# ====== 실행 트리거 ======
if __name__ == "__main__":
    daily_report_html()          
