#!/usr/bin/env python3
# coding: utf-8

import os, time, subprocess, io, base64, smtplib, requests
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import openai
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    "CRCL": {"shares": 20, "avg_price": 137.32},
}
CASH_BALANCE = 13925.60

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
TRADING_API_KEY = os.getenv("TRADINGECONOMICS_API_KEY", "guest:guest")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

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

def get_portfolio_indicators_html():
    rows_indicators = []
    rows_strategies = []

    for t, info in portfolio.items():
        t_upper = t.upper()
        ticker_obj = yf.Ticker(t)
        yinfo = ticker_obj.info or {}

        # --- 기술적 지표 ---
        rsi, macd = get_rsi_macd_values(t)

        # RSI 색깔 강조
        if rsi is not None:
            if rsi > 70:
                rsi_val = f"🔴 {rsi:.2f} (과매수)"
            elif rsi < 30:
                rsi_val = f"🟢 {rsi:.2f} (과매도)"
            else:
                rsi_val = f"⚫️ {rsi:.2f} (중립)"
        else:
            rsi_val = "N/A"

        # MACD 색깔 강조
        if macd is not None:
            if macd > 0:
                macd_val = f"🔴 {macd:.2f} (상승)"
            elif macd < 0:
                macd_val = f"🟢 {macd:.2f} (하락)"
            else:
                macd_val = f"⚫️ {macd:.2f} (중립)"
        else:
            macd_val = "N/A"

        # --- 재무 지표 ---
        per = yinfo.get("trailingPE")
        pbr = yinfo.get("priceToBook")
        roe = yinfo.get("returnOnEquity")
        eps = yinfo.get("trailingEps")
        fwd_per = yinfo.get("forwardPE")

        per_val = f"{per:.2f}" if per else "N/A"
        pbr_val = f"{pbr:.2f}" if pbr else "N/A"
        roe_val = f"{roe*100:.2f}%" if roe else "N/A"
        eps_val = f"{eps:.2f}" if eps else "N/A"
        fwd_per_val = f"{fwd_per:.2f}" if fwd_per else "N/A"

        # --- balance_sheet에서 부채비율 계산 ---
        debt_ratio = None
        try:
            bs = ticker_obj.balance_sheet
            if "Total Debt" in bs.index and "Total Assets" in bs.index:
                total_debt = bs.loc["Total Debt"].iloc[0]
                total_assets = bs.loc["Total Assets"].iloc[0]
                if total_assets and total_debt is not None:
                    debt_ratio = (total_debt / total_assets) * 100
        except Exception as e:
            print(f"❌ {t_upper} 부채비율 계산 실패: {e}")
        debt_val = f"{debt_ratio:.2f}%" if debt_ratio is not None else "N/A"

        # --- 변동성(β)에 따른 손절 조정 ---
        beta = yinfo.get("beta")
        if beta:
            if beta < 0.8:
                stop_loss_val = 0.95  # -5%
            elif beta <= 1.2:
                stop_loss_val = 0.93  # -7%
            else:
                stop_loss_val = 0.90  # -10%
        else:
            stop_loss_val = 0.93  # 기본값 -7%

        # --- 매도/손절 가격 계산 ---
        # 현재가 불러오기
        price_today = ticker_obj.history(period="1d")["Close"].iloc[-1]

        # 현재가 기준으로 매도/손절 계산
        sell_1 = f"${price_today*1.03:.2f}"  # 현재가 +3%
        sell_2 = f"${price_today*1.10:.2f}"  # 현재가 +10%
        stop_loss = f"${price_today*0.97:.2f}"  # 현재가 -3%

        # --- GPT 매매 전략 ---
        strategy_prompt = (
            f"{t_upper} 기술적 지표: RSI {rsi_val}, MACD {macd_val}\n"
            f"재무 지표: PER {per_val}, PBR {pbr_val}, ROE {roe_val}, EPS {eps_val}, 부채비율 {debt_val}, Forward PER {fwd_per_val}\n"
            "기본전략과 추가 고려사항을 분리해 한국어로 작성. "
            "출력은 ● 기본전략 / + 세부내용, ● 추가 고려사항 / + 세부내용 형식으로."
        )
        strategy_raw = gpt_chat(strategy_prompt)

        formatted_strategy = ""
        for line in strategy_raw.splitlines():
            if line.strip().startswith("●"):
                formatted_strategy += f"<b>{line.strip()}</b><br>"
            elif line.strip().startswith("+"):
                formatted_strategy += f"<span style='margin-left:20px;'>{line.strip()}</span><br>"
            else:
                formatted_strategy += f"{line.strip()}<br>"

        # 표 1 (지표용)
        rows_indicators.append({
            "종목": f"<b>{t_upper}</b>",
            "RSI": rsi_val,
            "MACD": macd_val,
            "PER": per_val,
            "PBR": pbr_val,
            "ROE": roe_val,
            "EPS": eps_val,
            "부채비율": debt_val,
            "Fwd PER": fwd_per_val
        })

        # 표 2 (전략용)
        rows_strategies.append({
            "종목": f"<b>{t_upper}</b>",
            "1차 매도": sell_1,
            "2차 매도": sell_2,
            "손절": stop_loss,
            "매매 전략": formatted_strategy if formatted_strategy else "N/A"
        })

    # 두 개의 표 생성
    df_indicators = pd.DataFrame(rows_indicators)
    df_strategies = pd.DataFrame(rows_strategies)

    table1_html = df_indicators.to_html(escape=False, index=False, justify="center", border=1, na_rep="-")
    table2_html = df_strategies.to_html(escape=False, index=False, justify="center", border=1, na_rep="-")

    return f"""
    <div style='background:#f9f9f9;padding:10px;border-radius:8px;'>
        <h4>📊 종목별 판단 지표</h4>
        {table1_html}
        <br><br>
        <h4>📈 종목별 매매 전략</h4>
        {table2_html}
    </div>
    """
    
def get_news_summary_html():
    """
    포트폴리오 종목별 뉴스를 가져와 요약 후 HTML로 출력
    """
    html = "<h3>📰 종목별 뉴스</h3>"
    for t in portfolio:
        t_upper = t.upper()
        html += f"<h4>📌 <b>{t_upper}</b></h4>"
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": t,
                    "apiKey": NEWS_API_KEY,
                    "pageSize": 6,
                    "sortBy": "publishedAt",
                    "language": "en",  # 언어 고정 (필요시 조정)
                },
                timeout=10,
            )
            articles = r.json().get("articles", [])

            # 안전하게 title/description 기본값 처리
            filtered = [
                a for a in articles
                if t_upper in ((a.get("title") or "") + (a.get("description") or "")).upper()
            ][:3]

            if not filtered:
                html += "<p style='color:gray;'>⚠️ 관련 뉴스 없음</p>"
                continue

            news_text = ""
            for i, a in enumerate(filtered, 1):
                title = a.get("title") or "제목 없음"
                desc = a.get("description") or ""
                url = a.get("url") or "#"

                html += f"<p><b>{i}. <a href='{url}' target='_blank'>{title}</a></b></p>"
                if desc:
                    html += f"<p style='margin-left:20px;color:#555;'>{desc}</p>"

                news_text += f"[{i}] {title} - {desc}\n"

            # GPT 요약
            summary = gpt_chat(
                f"{t_upper} 관련 뉴스:\n{news_text}\n"
                "뉴스 요약을 한국어로 작성하고, 각 주제는 ● 로 시작, 세부내용은 + 기호로 시작해 들여쓰기 해줘."
            )

            formatted = ""
            for line in summary.splitlines():
                if line.strip().startswith("●"):
                    formatted += f"<p><b>{line.strip()}</b></p>"
                elif line.strip().startswith("+"):
                    formatted += f"<p style='margin-left:20px;'>{line.strip()}</p>"

            html += f"<div style='background:#eef;padding:8px;border-radius:8px;'>{formatted}</div>"

        except Exception as e:
            html += f"<p style='color:red;'>❌ 뉴스 로드 실패: {e}</p>"

    return html

def fetch_economic_indicators():
    """
    미국 주요 경제지표 최근 6개월 데이터를 가져옴.
    - 1차: FRED API
    - 2차: TradingEconomics API (fallback)
    - 분기 데이터(GDP 등)는 최근 2분기만 뽑아서 병합
    """
    indicators = {
        "소비자물가지수(CPI)": {"fred": "CPIAUCSL", "te": "united states/cpi", "freq": "M"},
        "실업률": {"fred": "UNRATE", "te": "united states/unemployment rate", "freq": "M"},
        "GDP 성장률": {"fred": "A191RL1Q225SBEA", "te": "united states/gdp growth rate", "freq": "Q"},
        "개인소비지출(PCE)": {"fred": "PCE", "te": "united states/personal spending", "freq": "M"},
        "연방기금금리": {"fred": "FEDFUNDS", "te": "united states/interest rate", "freq": "M"},
        "신규실업수당청구": {"fred": "ICSA", "te": "united states/jobless claims", "freq": "W"},  # 주간 → 월별 평균 변환
    }

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")  # 최근 6개월
    data = {"지표": []}
    months = []

    for name, cfg in indicators.items():
        monthly_values = {}

        # ------------------------
        # 1차: FRED API 시도
        # ------------------------
        try:
            url = (
                f"{FRED_API_BASE}?series_id={cfg['fred']}&api_key={FRED_API_KEY}"
                f"&file_type=json&observation_start={start_date}&observation_end={end_date}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            observations = r.json().get("observations", [])

            for obs in observations:
                date = obs["date"][:7]  # YYYY-MM
                val = None if obs["value"] in [".", ""] else float(obs["value"])
                monthly_values[date] = val

            if monthly_values:
                print(f"✅ {name} → FRED API 성공")
            else:
                raise ValueError("빈 데이터")

        except Exception as e:
            print(f"⚠️ {name} FRED 실패: {e} → TradingEconomics로 대체 시도")

            # ------------------------
            # 2차: TradingEconomics API 시도
            # ------------------------
            try:
                url = (
                    f"https://api.tradingeconomics.com/historical/{cfg['te']}?"
                    f"c={TRADING_API_KEY}&d1={start_date}&d2={end_date}&f=json"
                )
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                te_data = r.json()

                for obs in te_data:
                    date = obs.get("Date", "")[:7]
                    val = obs.get("Value", None)
                    monthly_values[date] = val

                if monthly_values:
                    print(f"✅ {name} → TradingEconomics API 성공")
                else:
                    raise ValueError("빈 데이터")

            except Exception as e2:
                print(f"❌ {name} TE도 실패: {e2}")

        # ------------------------
        # 데이터 정규화
        # ------------------------
        if cfg["freq"] == "Q":
            # 분기 데이터 → 최근 2분기만 추출
            quarterly_keys = sorted(monthly_values.keys())[-2:]
            monthly_values = {k: monthly_values[k] for k in quarterly_keys}
        elif cfg["freq"] == "W":
            # 주간 데이터 → 월별 평균 변환
            tmp = {}
            for k, v in monthly_values.items():
                if v is None:
                    continue
                ym = k
                tmp.setdefault(ym, []).append(v)
            monthly_values = {ym: sum(vals) / len(vals) for ym, vals in tmp.items()}

        # ------------------------
        # 결과 저장
        # ------------------------
        if not months and monthly_values:
            months = sorted(list(monthly_values.keys())[-6:])
            for m in months:
                data[m] = []

        data["지표"].append(name)
        for m in months:
            data[m].append(monthly_values.get(m, None))

    return pd.DataFrame(data)

def get_monthly_economic_indicators_html():
    """
    📊 주요 경제지표 월별 변화를 HTML 표 + 설명 + GPT 요약으로 반환
    """
    try:
        df = fetch_economic_indicators()
        if df.empty or len(df) == 0:
            return "<p style='color:red;'>⚠️ 경제지표 데이터를 불러오지 못했습니다. (API 키/범위 확인 필요)</p>"

        explanations = {
            "소비자물가지수(CPI)": "소비자가 지불하는 물가 수준, 인플레이션의 핵심 지표.",
            "실업률": "경제 내 고용 상황을 나타내는 지표, 낮을수록 고용 시장이 양호.",
            "GDP 성장률": "국내총생산의 성장률, 경제 전반의 성장 속도를 보여줌.",
            "개인소비지출(PCE)": "소비자의 지출을 측정, 연준이 선호하는 인플레이션 지표.",
            "연방기금금리": "미국 기준금리, 금융시장과 경기 전반에 직접 영향.",
            "신규실업수당청구": "실업급여 신규 신청 건수, 경기 둔화 여부를 조기 반영."
        }

        # 표 HTML
        table_html = df.to_html(index=False, justify="center", border=1, na_rep='-')

        # 기본 설명
        explanation_html = "<ul>"
        for indicator, desc in explanations.items():
            explanation_html += f"<li><b>{indicator}</b>: {desc}</li>"
        explanation_html += "</ul>"

        # GPT 자동 요약 (투자 관점 해석)
        gpt_summary = gpt_chat(
            "다음은 최근 6개월간 주요 미국 경제지표(CPI, 실업률, GDP, PCE, 금리, 신규실업수당청구) 데이터입니다.\n"
            "각 지표가 투자자에게 어떤 의미를 가지는지, 그리고 향후 시장에 미칠 영향을 한국어로 5줄 이내로 요약해줘."
        )
        gpt_html = "".join([f"<p>{line}</p>" for line in gpt_summary.splitlines() if line.strip()])

        return f"""
        <div style='background:#f9f9f9;padding:10px;border-radius:8px;overflow-x:auto;'>
            <h4>📊 주요 경제지표 (최근 6개월)</h4>
            {table_html}
            <h5>ℹ️ 지표 설명</h5>
            {explanation_html}
            <h5>💡 투자자 관점 요약</h5>
            {gpt_html}
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

    <h3></h3>
    {get_portfolio_overview_html()}

    <h3>📊 종목별 판단 지표 및 전략</h3>
    {get_portfolio_indicators_html()}

    <h3></h3>
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
