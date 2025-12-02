import os
import time
import smtplib
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# =========================
# 공통 유틸
# =========================

def fmt_money(x, currency_symbol="$", digits=2):
    try:
        return f"{currency_symbol}{float(x):,.{digits}f}"
    except Exception:
        return "N/A"


def fmt_pct(x, digits=2):
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "N/A"


def safe_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def colorize_value_html(text, raw_value):
    """양수 → 초록, 음수 → 빨강."""
    try:
        val = float(raw_value)
    except Exception:
        return text

    if val > 0:
        color = "#008000"  # green
    elif val < 0:
        color = "#cc0000"  # red
    else:
        return text

    return f'<span style="color:{color}">{text}</span>'


# =========================
# Google Sheets 클라이언트
# =========================

def get_gspread_client():
    json_keyfile = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not json_keyfile:
        raise EnvironmentError(
            "환경변수 GOOGLE_APPLICATION_CREDENTIALS 가 설정되어 있지 않습니다."
        )

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile, scope)
    return gspread.authorize(creds)


def open_gsheet(gs_id, retries=3, delay=5):
    if not gs_id:
        raise EnvironmentError("환경변수 GSHEET_ID 가 설정되어 있지 않습니다.")

    client = get_gspread_client()
    for i in range(retries):
        try:
            return client.open_by_key(gs_id)
        except gspread.exceptions.APIError as e:
            if "503" in str(e) and i < retries - 1:
                print(
                    f"⚠️ Google API 503 오류 발생, {delay}초 후 재시도... "
                    f"({i + 1}/{retries})"
                )
                time.sleep(delay)
                continue
            raise

# =========================
# 야후 파이낸스 디버그
# =========================


def build_yfinance_debug_section_html(tickers=None) -> str:
    """
    yfinance 연결 상태를 HTML 텍스트로 요약.
    리포트 하단에 붙여서 확인할 수 있다.
    """
    if tickers is None:
        tickers = ["NVDA", "TSLA", "SCHD"]

    rows = []
    for t in tickers:
        status = {"price": "N/A", "info": "N/A", "news": "N/A"}

        try:
            tk = yf.Ticker(t)
        except Exception as e:
            rows.append(f"<tr><td>{t}</td><td colspan='3'>Ticker 객체 생성 실패: {e}</td></tr>")
            continue

        # 가격
        try:
            hist = tk.history(period="5d")["Close"].dropna()
            if hist.empty:
                status["price"] = "가격 데이터 없음"
            else:
                status["price"] = f"{float(hist.iloc[-1]):.2f}"
        except Exception as e:
            status["price"] = f"에러: {e}"

        # info
        try:
            info = tk.info or {}
            name = info.get("shortName") or info.get("longName") or "N/A"
            fpe = info.get("forwardPE", "N/A")
            status["info"] = f"{name} / Fwd PER: {fpe}"
        except Exception as e:
            status["info"] = f"info 에러: {e}"

        # news
        try:
            news_list = tk.news or []
            status["news"] = f"{len(news_list)}개"
        except Exception as e:
            status["news"] = f"news 에러: {e}"

        rows.append(
            f"<tr>"
            f"<td>{t}</td>"
            f"<td>{status['price']}</td>"
            f"<td>{status['info']}</td>"
            f"<td>{status['news']}</td>"
            f"</tr>"
        )

    table_html = (
        "<h3>🔍 yfinance 연결 상태 (디버그)</h3>"
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-size:12px;'>"
        "<tr><th>Ticker</th><th>최근 종가</th><th>info 요약</th><th>news 개수</th></tr>"
        + "".join(rows) +
        "</table>"
    )
    return table_html


def debug_yfinance_connectivity(tickers=None):
    """
    yfinance가 Yahoo Finance에서 데이터를 제대로 가져오는지 간단 점검하는 함수.
    - 가격(history)
    - info 일부 필드
    - news 개수 / 제목
    을 콘솔 로그로 출력한다.
    """
    if tickers is None:
        tickers = ["NVDA", "TSLA", "SCHD"]

    print("========== [DEBUG] yfinance connectivity test ==========")
    for t in tickers:
        print(f"\n[DEBUG] Ticker: {t}")
        try:
            tk = yf.Ticker(t)
        except Exception as e:
            print(f"  - Ticker 생성 실패: {e}")
            continue

        # 1) 가격 확인
        try:
            hist = tk.history(period="5d")["Close"].dropna()
            if hist.empty:
                print("  - 최근 5일 가격 데이터: 없음 (empty)")
            else:
                last_price = float(hist.iloc[-1])
                print(f"  - 최근 종가: {last_price:.2f}")
        except Exception as e:
            print(f"  - 가격 데이터 조회 실패: {e}")

        # 2) info 일부 확인
        try:
            info = tk.info or {}
            name = info.get("shortName") or info.get("longName") or "N/A"
            fpe = info.get("forwardPE", "N/A")
            print(f"  - 종목명(shortName): {name}")
            print(f"  - Fwd PER(info['forwardPE']): {fpe}")
        except Exception as e:
            print(f"  - info 조회 실패: {e}")

        # 3) 뉴스 확인
        try:
            news_list = tk.news or []
            print(f"  - news 개수: {len(news_list)}")
            for i, n in enumerate(news_list[:3]):
                title = n.get("title")
                provider = n.get("provider")
                print(f"    [{i}] provider={provider}, title={title}")
        except Exception as e:
            print(f"  - news 조회 실패: {e}")
    print("========================================================")
    

# =========================
# 시세 / 환율 유틸
# =========================

def get_last_and_prev_close(ticker, period="2y"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            return None, None, None
        closes = hist["Close"].dropna()
        if len(closes) == 0:
            return None, None, None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
        return last, prev, closes
    except Exception:
        return None, None, None


def get_usd_cad_rate():
    """1 USD = ? CAD"""
    try:
        hist = yf.Ticker("CAD=X").history(period="5d")
        if hist is None or hist.empty:
            return 1.35
        rate = float(hist["Close"].dropna().iloc[-1])
        return rate if rate > 0 else 1.35
    except Exception:
        return 1.35


def get_fx_multipliers(base_currency):
    base = (base_currency or "USD").upper()
    usd_cad = get_usd_cad_rate()  # 1 USD = usd_cad CAD

    if base == "USD":
        fx_usd_to_base = 1.0
        fx_cad_to_base = 1.0 / usd_cad
    elif base == "CAD":
        fx_usd_to_base = usd_cad
        fx_cad_to_base = 1.0
    else:
        fx_usd_to_base = 1.0
        fx_cad_to_base = 1.0

    return fx_usd_to_base, fx_cad_to_base


# =========================
# Google Sheet 로드 / 전처리
# =========================

def load_portfolio_from_gsheet():
    """
    Sheets 구조:
      Holdings:
        - Ticker, Shares, AvgPrice, Type(TFSA/RESP)
      Settings:
        - TFSA_CashUSD, RESP_CashCAD
        - TFSA_NetDepositCAD, RESP_NetDepositCAD
        - BaseCurrency
    """
    gs_id = os.environ.get("GSHEET_ID")
    sh = open_gsheet(gs_id)

    ws_hold = sh.worksheet("Holdings")
    df_hold = pd.DataFrame(ws_hold.get_all_records())

    ws_settings = sh.worksheet("Settings")
    df_settings = pd.DataFrame(ws_settings.get_all_records())

    if "Key" not in df_settings.columns or "Value" not in df_settings.columns:
        raise ValueError("Settings 시트에는 'Key', 'Value' 열이 필요합니다.")

    settings = dict(zip(df_settings["Key"].astype(str), df_settings["Value"]))

    tfsa_cash_usd = safe_float(
        settings.get("TFSA_CashUSD", settings.get("CashUSD", 0.0)), 0.0
    )
    resp_cash_cad = safe_float(settings.get("RESP_CashCAD", 0.0), 0.0)

    tfsa_netdep_cad = safe_float(settings.get("TFSA_NetDepositCAD", 0.0), 0.0)
    resp_netdep_cad = safe_float(settings.get("RESP_NetDepositCAD", 0.0), 0.0)

    base_currency = str(settings.get("BaseCurrency", "USD")).upper()

    for col in ["Ticker", "Shares", "AvgPrice"]:
        if col not in df_hold.columns:
            raise ValueError(f"'Holdings' 시트에 '{col}' 열이 없습니다.")

    df_hold["Ticker"] = df_hold["Ticker"].astype(str).str.strip().str.upper()
    df_hold["Shares"] = pd.to_numeric(df_hold["Shares"], errors="coerce").fillna(0.0)
    df_hold["AvgPrice"] = pd.to_numeric(df_hold["AvgPrice"], errors="coerce").fillna(
        0.0
    )

    if "Type" not in df_hold.columns:
        df_hold["Type"] = "TFSA"
    else:
        df_hold["Type"] = (
            df_hold["Type"].fillna("TFSA").astype(str).str.strip().str.upper()
        )

    return (
        df_hold,
        tfsa_cash_usd,
        resp_cash_cad,
        base_currency,
        tfsa_netdep_cad,
        resp_netdep_cad,
    )


# -----------------------------
# 기사 번역
# -----------------------------

def translate_to_korean(text: str) -> str:
    """
    OpenAI API를 사용해 영어 문장을 한국어 자연스러운 문장으로 번역.
    핵심 요점 요약도 함께 자동 처리됨.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "영어 뉴스 제목을 한국어 자연스러운 한 줄 문장으로 요약·번역하세요. 과도한 의역 금지. 핵심만 담기."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=80,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # 실패하면 원문을 그대로 반환
        return text


# =========================
# 계좌별 평가/손익 계산
# =========================

def enrich_holdings_with_prices(
    df_hold,
    base_currency,
    tfsa_cash_usd,
    resp_cash_cad,
    tfsa_netdep_cad,
    resp_netdep_cad,
):
    """
    TFSA: USD 계좌
    RESP: CAD 계좌
    - summary[acc]["*_native"]는 계좌 통화 기준 값
    - summary[acc]["*"] (base)은 BaseCurrency 기준 값
    """
    df = df_hold.copy()

    fx_usd_to_base, fx_cad_to_base = get_fx_multipliers(base_currency)
    usd_cad = get_usd_cad_rate()
    cad_to_usd = 1.0 / usd_cad if usd_cad != 0 else 1.0

    accounts = ["TFSA", "RESP"]
    summary = {
        acc: {
            "holdings_value_today": 0.0,
            "holdings_value_yesterday": 0.0,
            "cash_native": 0.0,
            "cash_base": 0.0,
            "holdings_value_today_native": 0.0,
            "holdings_value_yesterday_native": 0.0,
            "net_deposit_cad": 0.0,
            "net_deposit_native": 0.0,
        }
        for acc in accounts
    }

    # 현금 (native)
    summary["TFSA"]["cash_native"] = tfsa_cash_usd   # USD
    summary["RESP"]["cash_native"] = resp_cash_cad   # CAD
    # 현금 (base)
    summary["TFSA"]["cash_base"] = tfsa_cash_usd * fx_usd_to_base
    summary["RESP"]["cash_base"] = resp_cash_cad * fx_cad_to_base

    # 순투입자본 CAD
    summary["TFSA"]["net_deposit_cad"] = tfsa_netdep_cad
    summary["RESP"]["net_deposit_cad"] = resp_netdep_cad
    # 순투입자본 native
    summary["TFSA"]["net_deposit_native"] = tfsa_netdep_cad * cad_to_usd  # USD
    summary["RESP"]["net_deposit_native"] = resp_netdep_cad              # CAD

    # 결과 컬럼 초기화
    df["LastPrice"] = np.nan                 # native
    df["PrevClose"] = np.nan                 # native
    df["LastPriceBase"] = np.nan
    df["PrevCloseBase"] = np.nan
    df["PositionValueNative"] = np.nan       # native
    df["PositionValueBase"] = np.nan
    df["PositionPrevValueBase"] = np.nan
    df["ProfitLossBase"] = np.nan
    df["ProfitLossNative"] = np.nan          # native
    df["ProfitLossPct"] = np.nan

    for idx, row in df.iterrows():
        ticker = row["Ticker"]
        shares = safe_float(row["Shares"], 0.0)
        avg_price = safe_float(row["AvgPrice"], 0.0)
        acc_type = str(row["Type"]).upper()
        if acc_type not in accounts:
            acc_type = "TFSA"
            df.at[idx, "Type"] = "TFSA"

        if acc_type == "TFSA":
            fx_to_base = fx_usd_to_base
        else:
            fx_to_base = fx_cad_to_base

        last, prev, closes = get_last_and_prev_close(ticker)
        if last is None:
            last = avg_price
        if prev is None:
            prev = last

        position_value_native = shares * last
        position_prev_native = shares * prev

        position_value_base = position_value_native * fx_to_base
        position_prev_value_base = position_prev_native * fx_to_base

        cost_native = shares * avg_price
        cost_base = cost_native * fx_to_base
        profit_base = position_value_base - cost_base
        profit_native = profit_base / fx_to_base if fx_to_base != 0 else profit_base
        profit_pct = (profit_base / cost_base * 100.0) if cost_base != 0 else 0.0

        df.at[idx, "LastPrice"] = last
        df.at[idx, "PrevClose"] = prev
        df.at[idx, "LastPriceBase"] = last * fx_to_base
        df.at[idx, "PrevCloseBase"] = prev * fx_to_base
        df.at[idx, "PositionValueNative"] = position_value_native
        df.at[idx, "PositionValueBase"] = position_value_base
        df.at[idx, "PositionPrevValueBase"] = position_prev_value_base
        df.at[idx, "ProfitLossBase"] = profit_base
        df.at[idx, "ProfitLossNative"] = profit_native
        df.at[idx, "ProfitLossPct"] = profit_pct

        summary[acc_type]["holdings_value_today"] += position_value_base
        summary[acc_type]["holdings_value_yesterday"] += position_prev_value_base
        summary[acc_type]["holdings_value_today_native"] += position_value_native
        summary[acc_type]["holdings_value_yesterday_native"] += position_prev_native

    # 계좌별 today/yesterday/Δ (native) + deposit 대비 손익 (native)
    for acc in accounts:
        hv_today_native = summary[acc]["holdings_value_today_native"]
        hv_yesterday_native = summary[acc]["holdings_value_yesterday_native"]
        cash_native = summary[acc]["cash_native"]
        net_dep_native = summary[acc]["net_deposit_native"]

        today_native = hv_today_native + cash_native
        yesterday_native = hv_yesterday_native + cash_native
        diff_native = today_native - yesterday_native
        pct_native = (
            diff_native / yesterday_native * 100.0 if yesterday_native != 0 else 0.0
        )

        pl_vs_dep_native = today_native - net_dep_native
        pl_vs_dep_pct_native = (
            pl_vs_dep_native / net_dep_native * 100.0
            if net_dep_native != 0
            else 0.0
        )

        summary[acc]["total_today_native"] = today_native
        summary[acc]["total_yesterday_native"] = yesterday_native
        summary[acc]["total_diff_native"] = diff_native
        summary[acc]["total_diff_pct_native"] = pct_native
        summary[acc]["pl_vs_deposit_native"] = pl_vs_dep_native
        summary[acc]["pl_vs_deposit_pct_native"] = pl_vs_dep_pct_native

        # 기준통화 기준 (detail/table 용)
        hv_today_base = summary[acc]["holdings_value_today"]
        hv_yesterday_base = summary[acc]["holdings_value_yesterday"]
        cash_base = summary[acc]["cash_base"]
        today_base = hv_today_base + cash_base
        yesterday_base = hv_yesterday_base + cash_base
        diff_base = today_base - yesterday_base
        pct_base = (
            diff_base / yesterday_base * 100.0 if yesterday_base != 0 else 0.0
        )

        summary[acc]["total_today"] = today_base
        summary[acc]["total_yesterday"] = yesterday_base
        summary[acc]["total_diff"] = diff_base
        summary[acc]["total_diff_pct"] = pct_base

    # TOTAL (기준통화 기준, 참고용)
    total_today_base = summary["TFSA"]["total_today"] + summary["RESP"]["total_today"]
    total_yesterday_base = (
        summary["TFSA"]["total_yesterday"] + summary["RESP"]["total_yesterday"]
    )
    total_diff_base = total_today_base - total_yesterday_base
    total_pct_base = (
        total_diff_base / total_yesterday_base * 100.0
        if total_yesterday_base != 0
        else 0.0
    )

    summary["TOTAL"] = {
        "total_today": total_today_base,
        "total_yesterday": total_yesterday_base,
        "total_diff": total_diff_base,
        "total_diff_pct": total_pct_base,
    }

    summary["meta"] = {
        "base_currency": base_currency,
        "fx_usd_to_base": fx_usd_to_base,
        "fx_cad_to_base": fx_cad_to_base,
    }

    return df, summary


# =========================
# 투자 분석 보조 함수 (중단기 + SCHD 배당)
# =========================

def analyze_midterm_ticker(ticker):
    """
    중단기 투자 분석용 함수 (조심스러운 뉴스 처리 버전).

    1) 수치:
       - 중기 상승 확률 %
       - 매수 타이밍 %
       - 매도 타이밍 %
       - 1년 목표수익 범위

    2) '핵심 투자 코멘트' (Comment, HTML):
       - yfinance의 tk.news에서 신뢰도 있는 헤드라인이 있을 때만
         최대 2개를 골라 한국어로 자연스럽게 번역해서 보여준다.
       - 헤드라인이 없거나 구조가 이상하면
         깔끔하게 "뉴스 없음" 문구를 출력한다.
    """
    tk = yf.Ticker(ticker)

    # -----------------------------
    # 1. 가격 데이터 확보
    # -----------------------------
    try:
        hist = tk.history(period="2y")
        closes = hist["Close"].dropna()
        if len(closes) < 60:
            raise ValueError("가격 데이터 부족")
    except Exception:
        comment_html = (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약:</strong><br>"
            "- 가격·뉴스 데이터 부족으로 헤드라인을 가져올 수 없습니다."
            "</p>"
        )
        return {
            "Ticker": ticker,
            "UpProb": None,
            "BuyTiming": None,
            "SellTiming": None,
            "TargetRange": "데이터 부족",
            "Comment": comment_html,
        }

    last = float(closes.iloc[-1])

    # -----------------------------
    # 2. 수익률·변동성 계산 (수치용)
    # -----------------------------
    # 1년 수익률
    if len(closes) > 252:
        start_1y = float(closes.iloc[-252])
    else:
        start_1y = float(closes.iloc[0])
    ret_1y = (last / start_1y - 1.0) * 100.0 if start_1y > 0 else 0.0

    # 3개월 수익률
    if len(closes) > 63:
        start_3m = float(closes.iloc[-63])
        ret_3m = (last / start_3m - 1.0) * 100.0 if start_3m > 0 else 0.0
    else:
        ret_3m = ret_1y / 4.0

    # 연간 변동성
    rets = np.log(closes / closes.shift(1)).dropna()
    vol_annual = float(rets.std() * np.sqrt(252)) if len(rets) > 0 else 0.0

    # -----------------------------
    # 3. 투자 신호 (상승확률, 매수/매도 타이밍, 목표수익 범위)
    # -----------------------------
    score = 50.0
    # 모멘텀(1년 수익률) 반영
    score += float(np.tanh(ret_1y / 40.0)) * 25.0
    # 변동성 패널티
    score -= float(np.tanh(vol_annual * 2.0)) * 15.0
    up_prob = max(5.0, min(95.0, score))

    last_252 = closes[-252:] if len(closes) >= 252 else closes
    low_52w = float(last_252.min())
    high_52w = float(last_252.max())
    if high_52w > low_52w:
        pos = (last - low_52w) / (high_52w - low_52w)  # 0~1
    else:
        pos = 0.5

    buy_score = max(5.0, min(95.0, (1.0 - pos) * 60.0 + max(0.0, -ret_3m) * 0.5))
    sell_score = max(5.0, min(95.0, pos * 60.0 + max(0.0, ret_1y) * 0.5))

    band = vol_annual * 100.0
    low = max(-50.0, ret_1y - band)
    high = min(100.0, ret_1y + band)
    target_range = f"{low:,.1f}% ~ {high:,.1f}%"

    # -----------------------------
    # 4. yfinance 뉴스에서 "신뢰도 있는" 헤드라인 최대 2개 추출
    # -----------------------------
    try:
        raw_news = tk.news
    except Exception:
        raw_news = None

    news_list = raw_news or []
    reliable_headlines = []

    for n in news_list:
        try:
            title = (n.get("title") or "").strip()
        except Exception:
            title = ""

        # 너무 짧거나 비어 있으면 버림
        if len(title) < 8:
            continue

        ts = n.get("providerPublishTime")
        if ts:
            try:
                ts_int = int(ts)
                date_str = datetime.fromtimestamp(ts_int).strftime("%Y-%m-%d")
            except Exception:
                date_str = ""
        else:
            date_str = ""

        reliable_headlines.append((date_str, title))
        if len(reliable_headlines) >= 2:
            break

    # -----------------------------
    # 5. 헤드라인이 없으면 "뉴스 없음"을 명시적으로 표시
    # -----------------------------
    if not reliable_headlines:
        comment_html = (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약:</strong><br>"
            "- 현재 이 종목에 대해 yfinance를 통해 가져올 수 있는 신뢰할 만한 헤드라인이 없습니다. "
            "실적 발표, 섹터 뉴스, 정책·금리 이벤트를 별도로 확인하는 것이 좋습니다."
            "</p>"
        )
    else:
        # -----------------------------
        # 6. 헤드라인을 한국어로 번역하여 최대 2개만 표기
        # -----------------------------
        lines = []
        for date_str, title in reliable_headlines:
            # OpenAI로 한국어 번역 (실패 시 원문 사용)
            kr = translate_to_korean(title)
            if date_str:
                lines.append(f"- {date_str}: {kr}")
            else:
                lines.append(f"- {kr}")

        news_html = "<br>".join(lines)

        comment_html = (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약:</strong><br>"
            f"{news_html}"
            "</p>"
        )

    # -----------------------------
    # 7. 결과 반환
    # -----------------------------
    return {
        "Ticker": ticker,
        "UpProb": up_prob,
        "BuyTiming": buy_score,
        "SellTiming": sell_score,
        "TargetRange": target_range,
        "Comment": comment_html,
    }


def build_midterm_context(ticker: str) -> str:
    """
    '주요 맥락' 열: 수치 + 15자 이내 짧은 분석만 제공.
    - 1년 수익률
    - 연 변동성
    - Fwd PER
    """
    tk = yf.Ticker(ticker)

    # ===== 가격 기반 수치 =====
    try:
        hist = tk.history(period="1y")["Close"].dropna()
        if len(hist) < 2:
            raise ValueError("데이터 부족")

        last = float(hist.iloc[-1])
        start = float(hist.iloc[0])
        ret_1y = (last / start - 1.0) * 100.0

        rets = np.log(hist / hist.shift(1)).dropna()
        vol_annual = float(rets.std() * np.sqrt(252))
        vol_pct = vol_annual * 100.0
    except Exception:
        ret_1y, vol_pct = None, None

    # ===== Fwd PER =====
    try:
        info = tk.info or {}
        fpe = safe_float(info.get("forwardPE"), None)
    except Exception:
        fpe = None

    # ===== 라벨링 규칙 =====
    def label_return(x):
        if x is None:
            return "N/A"
        if x > 10: return "강한 상승"
        if x < -10: return "약세 흐름"
        return "보합권"

    def label_vol(x):
        if x is None:
            return "N/A"
        if x > 60: return "고변동성"
        if x > 30: return "중간 변동성"
        return "저변동성"

    def label_fpe(x):
        if x is None:
            return "N/A"
        if x > 40: return "밸류 부담"
        if x >= 15: return "중립 밸류"
        return "저평가 구간"

    # ===== 출력 구성 =====
    lines = []

    # 1년 수익률
    if ret_1y is not None:
        lines.append(f"· 1년 수익률: {ret_1y:+.1f}% ({label_return(ret_1y)})")
    else:
        lines.append("· 1년 수익률: N/A")

    # 연 변동성
    if vol_pct is not None:
        lines.append(f"· 연 변동성: {vol_pct:.1f}% ({label_vol(vol_pct)})")
    else:
        lines.append("· 연 변동성: N/A")

    # Fwd PER
    if fpe is not None:
        lines.append(f"· Fwd PER: {fpe:.1f}배 ({label_fpe(fpe)})")
    else:
        lines.append("· Fwd PER: N/A")

    return "<br>".join(lines)
    

def build_midterm_analysis_html(df_enriched):
    """
    1) 요약표 : Ticker + 확률/타이밍/목표수익
    2) 상세표 : '핵심 투자 코멘트' + '주요맥락'
    """
    tfsa_tickers = (
        df_enriched[df_enriched["Type"].str.upper() == "TFSA"]["Ticker"]
        .dropna()
        .unique()
        .tolist()
    )
    tickers = [t for t in tfsa_tickers if t.upper() != "SCHD"]

    if not tickers:
        return "<p>TFSA 중단기 대상 종목이 없습니다.</p>"

    rows_summary = []
    rows_detail = []

    for t in sorted(tickers):
        stat = analyze_midterm_ticker(t)
        ctx = build_midterm_context(t)

        # ① 요약 테이블 행
        if stat["UpProb"] is not None:
            up_str = colorize_value_html(fmt_pct(stat["UpProb"]), stat["UpProb"])
            buy_str = colorize_value_html(fmt_pct(stat["BuyTiming"]), stat["BuyTiming"])
            sell_str = colorize_value_html(fmt_pct(stat["SellTiming"]), stat["SellTiming"])
        else:
            up_str = buy_str = sell_str = "N/A"

        rows_summary.append(
            {
                "Ticker": stat["Ticker"],
                "중기 상승 확률 %": up_str,
                "매수 타이밍 %": buy_str,
                "매도 타이밍 %": sell_str,
                "1년 목표수익 범위": stat["TargetRange"],
            }
        )

        # ② 상세 테이블 행
        rows_detail.append(
            {
                "Ticker": stat["Ticker"],
                "핵심 투자 코멘트": stat["Comment"],
                "주요맥락": ctx,
            }
        )

    df_sum = pd.DataFrame(rows_summary)
    df_det = pd.DataFrame(rows_detail)

    html_summary = df_sum.to_html(index=False, escape=False)
    html_detail = df_det.to_html(index=False, escape=False)

    return (
        "<h3>① 요약 테이블</h3>"
        + html_summary
        + "<br/><br/>"
        + "<h3>② 상세 테이블 (핵심 투자 코멘트 + 주요맥락)</h3>"
        + html_detail
    )

def simulate_schd_to_target(
    current_shares,
    start_price,
    start_yearly_div_ps,
    div_cagr,
    monthly_buy=200.0,
    target_monthly_income=1000.0,
    max_years=60,
):
    """
    DRIP + 매월 200 USD 추가 매수로
    월 배당 1,000 USD 도달까지 걸리는 기간(년/월)을 '연 단위'로 시뮬레이션.
    - price는 연간 동안 일정하다고 가정(보수적)
    - div_cagr: 연간 배당 성장률 (하한 설정 필요)
    """
    target_annual = target_monthly_income * 12.0

    shares = float(current_shares)
    yearly_div_ps = float(start_yearly_div_ps)
    price = float(start_price)

    years = 0
    prev_income = shares * yearly_div_ps

    while years < max_years:
        annual_income = shares * yearly_div_ps
        if annual_income >= target_annual:
            # 직전 연도 대비 선형 보간으로 개략적인 개월 수 추정
            if annual_income <= prev_income:
                frac = 0.0
            else:
                frac = (target_annual - prev_income) / (annual_income - prev_income)
                frac = max(0.0, min(1.0, frac))
            months = int(round(frac * 12))
            return years, months

        # DRIP + 연간 추가 매수(12 * monthly_buy)
        extra_yearly = monthly_buy * 12.0
        if price > 0:
            shares += (annual_income + extra_yearly) / price

        # 다음 해 배당 성장 반영
        yearly_div_ps *= (1.0 + div_cagr)

        prev_income = annual_income
        years += 1

    # max_years 안에 도달 못하면 보수적으로 반환
    return years, 0


def build_schd_dividend_html():
    """
    SCHD 최근 10년(완료 연도) 배당 및 가격 기반:
      - Historical: 연말 종가, 연간 배당, YoY 성장, 배당 수익률
      - Forecast: 최근 5년 배당 CAGR, 최근 3년 가격 CAGR 기반 향후 2년 예상
    """
    tk = yf.Ticker("SCHD")
    try:
        hist = tk.history(period="12y")
        divs = tk.dividends.dropna()
    except Exception:
        return "<p>SCHD 배당 데이터를 불러오지 못했습니다.</p>"

    if hist is None or hist.empty or divs.empty:
        return "<p>SCHD 배당 데이터가 충분하지 않습니다.</p>"

    today = datetime.today()
    current_year = today.year

    # 연도별 배당 합계
    div_by_year = divs.groupby(divs.index.year).sum()

    # 연도별 연말 종가 (마지막 거래일 기준)
    close = hist["Close"].dropna()
    close_by_year_end = close.groupby(close.index.year).last()

    # 공통 연도 중 완료된 연도만 사용 (현재 연도 제외)
    years = sorted(y for y in div_by_year.index if y in close_by_year_end.index and y < current_year)
    if not years:
        return "<p>SCHD 연도별 배당 데이터가 부족합니다.</p>"

    # 최근 10개 연도만
    years = years[-10:]

    records = []
    prev_div = None
    for y in years:
        div_ps = float(div_by_year.get(y, 0.0))
        price_end = float(close_by_year_end.get(y, np.nan))
        yield_pct = div_ps / price_end * 100.0 if price_end > 0 else np.nan
        if prev_div is not None and prev_div > 0:
            yoy = (div_ps / prev_div - 1.0) * 100.0
        else:
            yoy = np.nan
        prev_div = div_ps

        records.append(
            {
                "Year": y,
                "Type": "Historical",
                "Year-end Price": price_end,
                "Dividend / Share": div_ps,
                "YoY Dividend Growth %": yoy,
                "Dividend Yield %": yield_pct,
            }
        )

    df_hist = pd.DataFrame(records).sort_values("Year")

    # 배당 CAGR (최근 최대 5년)
    recent_div = df_hist.tail(min(5, len(df_hist)))
    if len(recent_div) >= 2:
        d0 = recent_div["Dividend / Share"].iloc[0]
        dN = recent_div["Dividend / Share"].iloc[-1]
        n = recent_div["Year"].iloc[-1] - recent_div["Year"].iloc[0]
        if d0 > 0 and n > 0:
            div_cagr = (dN / d0) ** (1.0 / n) - 1.0
        else:
            div_cagr = 0.0
    else:
        div_cagr = 0.0

    # 가격 CAGR (최근 최대 3년)
    recent_price = df_hist.tail(min(3, len(df_hist)))
    if len(recent_price) >= 2:
        p0 = recent_price["Year-end Price"].iloc[0]
        pN = recent_price["Year-end Price"].iloc[-1]
        n2 = recent_price["Year"].iloc[-1] - recent_price["Year"].iloc[0]
        if p0 > 0 and n2 > 0:
            price_cagr = (pN / p0) ** (1.0 / n2) - 1.0
        else:
            price_cagr = 0.0
    else:
        price_cagr = 0.0

    # 과도한 성장률 클리핑
    div_cagr = max(-0.10, min(0.15, div_cagr))     # -10% ~ +15%
    price_cagr = max(-0.10, min(0.15, price_cagr)) # -10% ~ +15%

    last_year = int(df_hist["Year"].max())
    last_div = float(df_hist[df_hist["Year"] == last_year]["Dividend / Share"].iloc[0])
    last_price = float(df_hist[df_hist["Year"] == last_year]["Year-end Price"].iloc[0])

    forecast_records = []
    prev_div_f = last_div
    prev_price_f = last_price
    for i in range(1, 3):  # 향후 2년
        year_f = last_year + i
        div_f = prev_div_f * (1.0 + div_cagr)
        price_f = prev_price_f * (1.0 + price_cagr)
        yield_f = div_f / price_f * 100.0 if price_f > 0 else np.nan
        yoy_f = (div_f / prev_div_f - 1.0) * 100.0 if prev_div_f > 0 else np.nan

        forecast_records.append(
            {
                "Year": year_f,
                "Type": "Forecast",
                "Year-end Price": price_f,
                "Dividend / Share": div_f,
                "YoY Dividend Growth %": yoy_f,
                "Dividend Yield %": yield_f,
            }
        )

        prev_div_f = div_f
        prev_price_f = price_f

    df_all = pd.concat([df_hist, pd.DataFrame(forecast_records)], ignore_index=True)

    df_all["Year-end Price"] = df_all["Year-end Price"].map(lambda x: fmt_money(x, "$"))
    df_all["Dividend / Share"] = df_all["Dividend / Share"].map(lambda x: fmt_money(x, "$"))
    df_all["Dividend Yield %"] = df_all["Dividend Yield %"].map(
        lambda x: fmt_pct(x) if pd.notnull(x) else "N/A"
    )
    df_all["YoY Dividend Growth %"] = df_all["YoY Dividend Growth %"].map(
        lambda x: fmt_pct(x) if pd.notnull(x) else "N/A"
    )

    return df_all[
        ["Year", "Type", "Year-end Price", "Dividend / Share", "YoY Dividend Growth %", "Dividend Yield %"]
    ].to_html(index=False, escape=False)

def build_schd_dividend_summary_text(current_shares):
    """
    SCHD 장기 배당 분석 (모든 값을 CAD 기준으로 환산).

    가정:
    - DRIP 적용
    - 매월 200 USD → CAD 변환 후 매수
    - 연평균 배당 성장률 = 11% 고정
    - 목표 배당: 월 CAD 1,000 (연 CAD 12,000)
    """
    current_shares = safe_float(current_shares, 0.0)
    if current_shares <= 0:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> N/A (보유 SCHD 없음)</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    tk = yf.Ticker("SCHD")

    # ===== 1) 배당 데이터 =====
    try:
        divs = tk.dividends.dropna()
    except Exception:
        divs = pd.Series(dtype=float)

    if divs.empty:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> 데이터 부족</p>"
            "<p><strong>월 CAD 1,000 달성:</strong> 계산 불가</p>"
        )

    # 연간 총 배당(USD)
    div_by_year = divs.groupby(divs.index.year).sum()
    years = sorted(div_by_year.index)
    last_year = years[-1]
    last_div_ps_usd = float(div_by_year[last_year])

    # ===== 2) 현재 SCHD 가격 =====
    try:
        px = tk.history(period="1mo")["Close"].dropna()
        price_usd = float(px.iloc[-1]) if not px.empty else 75.0
    except Exception:
        price_usd = 75.0

    # ===== 3) USD→CAD 환율 fetch =====
    try:
        fx = yf.Ticker("CAD=X").history(period="1d")["Close"].dropna()
        usd_to_cad = float(fx.iloc[-1]) if not fx.empty else 1.35
    except Exception:
        usd_to_cad = 1.35

    # ===== 4) 현재 연 배당금(CAD) =====
    current_annual_income_cad = current_shares * last_div_ps_usd * usd_to_cad

    # ===== 5) 성장률 고정 =====
    g = 0.11   # 11%

    # ===== 6) 배당 수익률 계산 (USD 기준) =====
    y = last_div_ps_usd / price_usd if price_usd > 0 else 0.035
    if y <= 0:
        y = 0.035  # fallback

    # ===== 7) 매월 200 USD → CAD 변환 후 투자 =====
    monthly_usd = 200.0
    monthly_cad = monthly_usd * usd_to_cad
    annual_contrib_cad = monthly_cad * 12.0

    # USD 기준 배당 수익률이므로 CAD로 변환하기 위해 같은 배수 적용
    # (DRIP는 주식 수 증가 → 배당 USD → CAD 변환)
    # 성장률 공식 그대로 적용하며 마지막에 CAD로 해석하면 consistency 유지됨.

    A = annual_contrib_cad * (y / g)

    target = 12000.0  # CAD 기준
    numerator = target + A
    denominator = current_annual_income_cad + A

    if numerator <= denominator:
        n_years = 0.0
    else:
        n_years = np.log(numerator / denominator) / np.log(1.0 + g)

    n_years = max(0.0, n_years)
    years_int = int(n_years)
    months_int = int(round((n_years - years_int) * 12.0))

    # ===== 8) 결과 출력 =====
    txt = (
        f"<p><strong>현재 예상 연 배당금(CAD):</strong> "
        f"{fmt_money(current_annual_income_cad, '$')} "
        f"(보유 {current_shares:,.0f}주 기준)</p>"
        f"<p><strong>월 CAD 1,000 달성 예상:</strong> "
        f"약 {years_int}년 {months_int}개월 "
        f"(DRIP + 매월 200 USD 투자 / 배당 성장률 11%)</p>"
    )
    return txt
    

# =========================
# HTML 리포트 생성
# =========================

def build_html_report(df_enriched, account_summary):
    """
    포트폴리오 HTML 리포트 생성:
      - 🏦 Account Summary (TFSA / RESP / Total)
      - 📂 TFSA Holdings
      - 🎓 RESP Holdings
      - 🔍 yfinance 연결 상태 (디버그)
    """
    base_ccy = account_summary["meta"]["base_currency"]
    ccy_symbol = "$"  # CAD / USD 모두 $ 로 표시

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ========================
    # 1) 계좌별 요약 테이블
    # ========================
    summary_rows = []
    for acc in ["TFSA", "RESP", "TOTAL"]:
        if acc not in account_summary:
            continue
        s = account_summary[acc]
        summary_rows.append(
            {
                "Account": acc,
                f"Total (Today, {base_ccy})": fmt_money(s["total_today"], ccy_symbol),
                f"Δ vs Yesterday ({base_ccy})": fmt_money(
                    s["total_diff"], ccy_symbol
                ),
                "Δ %": fmt_pct(s["total_diff_pct"]),
                "Cash (base)": fmt_money(s.get("cash_base", 0.0), ccy_symbol),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    # 상단 한 줄 요약 (TOTAL 기준)
    total_line_html = ""
    total_info = account_summary.get("TOTAL")
    if total_info:
        total_assets = fmt_money(total_info["total_today"], ccy_symbol)
        total_diff = fmt_money(total_info["total_diff"], ccy_symbol)
        total_diff_pct = fmt_pct(total_info["total_diff_pct"])
        total_line_html = (
            f"<p><strong>Total Assets (총 자산):</strong> {total_assets}  "
            f"<strong>Δ vs. Yesterday:</strong> {total_diff} ({total_diff_pct})</p>"
        )

    # ========================
    # 2) Holdings 뷰 가공
    # ========================
    df_view = df_enriched.copy()

    # 숫자 포맷 적용 (있을 때만)
    if "Shares" in df_view.columns:
        df_view["Shares"] = df_view["Shares"].map(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else ""
        )
    if "AvgPrice" in df_view.columns:
        df_view["AvgPrice"] = df_view["AvgPrice"].map(
            lambda x: fmt_money(x, ccy_symbol) if pd.notnull(x) else ""
        )
    if "LastPriceBase" in df_view.columns:
        df_view["LastPriceBase"] = df_view["LastPriceBase"].map(
            lambda x: fmt_money(x, ccy_symbol) if pd.notnull(x) else ""
        )
    if "PositionValueBase" in df_view.columns:
        df_view["PositionValueBase"] = df_view["PositionValueBase"].map(
            lambda x: fmt_money(x, ccy_symbol) if pd.notnull(x) else ""
        )
    if "ProfitLossBase" in df_view.columns:
        df_view["ProfitLossBase"] = df_view["ProfitLossBase"].map(
            lambda x: fmt_money(x, ccy_symbol) if pd.notnull(x) else ""
        )
    if "ProfitLossPct" in df_view.columns:
        df_view["ProfitLossPct"] = df_view["ProfitLossPct"].map(
            lambda x: fmt_pct(x) if pd.notnull(x) else ""
        )

    cols_order = [
        "Ticker",
        "Type",
        "Shares",
        "AvgPrice",
        "LastPriceBase",
        "PositionValueBase",
        "ProfitLossBase",
        "ProfitLossPct",
    ]
    # 없는 열은 빼고 테이블 구성
    cols_order = [c for c in cols_order if c in df_view.columns]

    def _table_for_account(acc_type: str) -> str:
        # Type 열이 TFSA / RESP 로 들어있다고 가정
        if "Type" not in df_view.columns:
            return "<p>Type 열이 없어 Holdings를 표시할 수 없습니다.</p>"
        sub = df_view[df_view["Type"].str.upper() == acc_type].copy()
        if sub.empty:
            return f"<p>No holdings for {acc_type}.</p>"
        return sub[cols_order].to_html(index=False, escape=False)

    tfsa_table = _table_for_account("TFSA")
    resp_table = _table_for_account("RESP")

    # ========================
    # 3) 스타일 & 전체 HTML 템플릿
    # ========================
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; background:#fafafa; }
      h1 { text-align:center; }
      h2 { margin-top:30px; color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:5px; }
      table { border-collapse: collapse; width:100%; margin:10px 0; }
      th, td { border:1px solid #ddd; padding:6px 8px; font-size:12px; }
      th { background:#f0f0f0; text-align:center; }
      td { text-align:right; }
      td:first-child, th:first-child { text-align:left; }
      .section { margin-bottom: 30px; }
      .muted { color:#777; font-size:12px; }
    </style>
    """

    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        {style}
      </head>
      <body>
        <h1>📊 Daily Portfolio Report</h1>
        <p class="muted" style="text-align:center">
          Generated at {now_str} (BaseCurrency: {base_ccy})
        </p>

        <div class="section">
          <h2>🏦 Account Summary (TFSA / RESP / Total)</h2>
          {total_line_html}
          {df_summary.to_html(index=False, escape=False)}
        </div>

        <div class="section">
          <h2>📂 TFSA Holdings</h2>
          {tfsa_table}
        </div>

        <div class="section">
          <h2>🎓 RESP Holdings</h2>
          {resp_table}
        </div>
    """

    # ========================
    # 4) yfinance 디버그 섹션 추가 (있으면)
    # ========================
    try:
        debug_html = build_yfinance_debug_section_html(["NVDA", "TSLA", "SCHD"])
        html += "<hr>" + debug_html
    except NameError:
        # 디버그 함수가 정의 안 되어 있으면 그냥 건너뜀
        pass

    html += """
      </body>
    </html>
    """

    return html

    # ---------- 2) 상세 보유 종목 테이블 (TFSA: USD, RESP: CAD) ----------
    def make_holdings_table(acc_type):
        sub = df_enriched[df_enriched["Type"].str.upper() == acc_type].copy()
        if sub.empty:
            return f"<p>No holdings for {acc_type}.</p>"

        # 공통 포맷
        sub["Shares"] = sub["Shares"].map(lambda x: f"{float(x):,.2f}")
        sub["AvgPrice"] = sub["AvgPrice"].map(lambda x: fmt_money(x, ccy_symbol))

        # native 가격/평가/손익
        sub["LastPriceNativeFmt"] = sub["LastPrice"].map(
            lambda x: fmt_money(x, ccy_symbol)
        )
        sub["PositionValueNativeFmt"] = sub["PositionValueNative"].map(
            lambda x: fmt_money(x, ccy_symbol)
        )

        # Profit/Loss native + 색상
        raw_pl_native = sub["ProfitLossNative"].tolist()
        raw_pl_pct = sub["ProfitLossPct"].tolist()

        pl_native_fmt = []
        for v in raw_pl_native:
            v_num = safe_float(v, 0.0)
            text = fmt_money(v_num, ccy_symbol)
            pl_native_fmt.append(colorize_value_html(text, v_num))

        pl_pct_fmt = []
        for v in raw_pl_pct:
            v_num = safe_float(v, 0.0)
            text = fmt_pct(v_num)
            pl_pct_fmt.append(colorize_value_html(text, v_num))

        sub["ProfitLossNativeFmt"] = pl_native_fmt
        sub["ProfitLossPctFmt"] = pl_pct_fmt

        cols = [
            "Ticker",
            "Type",
            "Shares",
            "AvgPrice",
            "LastPriceNativeFmt",
            "PositionValueNativeFmt",
            "ProfitLossNativeFmt",
            "ProfitLossPctFmt",
        ]
        rename_map = {
            "LastPriceNativeFmt": "LastPrice",
            "PositionValueNativeFmt": "PositionValue",
            "ProfitLossNativeFmt": "Profit/Loss",
            "ProfitLossPctFmt": "Profit/Loss %",
        }

        sub = sub[cols].rename(columns=rename_map)
        return sub.to_html(index=False, escape=False)

    tfsa_table = make_holdings_table("TFSA")
    resp_table = make_holdings_table("RESP")

    # ---------- 3) 중단기 투자 분석 (TFSA, SCHD 제외) ----------
    midterm_html = build_midterm_analysis_html(df_enriched)

    # ---------- 4) SCHD 배당 분석 + DRIP/월 200 매수 시뮬레이션 ----------
    schd_div_html = build_schd_dividend_html()

    # 현재 보유 SCHD 수량 합계
    try:
        schd_shares = float(
            df_enriched[df_enriched["Ticker"].str.upper() == "SCHD"]["Shares"].sum()
        )
    except Exception:
        schd_shares = 0.0

    schd_summary_text = build_schd_dividend_summary_text(schd_shares)

    # ---------- 5) HTML 템플릿 ----------
    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background:#fafafa; }
    h1 { text-align:center; }
    h2 { margin-top:30px; color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:5px; }
    h3 { margin-top:20px; color:#34495e; }
    table { border-collapse: collapse; width:100%; margin:10px 0; }
    th, td { border:1px solid #ddd; padding:6px; text-align:center; font-size:13px; }
    th { background:#f4f6f6; }
    .muted { color:#666; font-size:12px; }
    .section { background:white; border:1px solid #ddd; border-radius:8px; padding:10px; margin:15px 0; }
    </style>
    """

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        {style}
      </head>
      <body>
        <h1>📊 Daily Portfolio Report</h1>
        <p class="muted" style="text-align:center">
          Generated at {now_str} (BaseCurrency: {base_ccy})
        </p>

        <div class="section">
          <h2>🏦 Account Summary (TFSA / RESP / Total)</h2>
          {total_assets_line}
          {df_summary.to_html(index=False, escape=False)}
        </div>

        <div class="section">
          <h2>📂 TFSA Holdings (in USD)</h2>
          {tfsa_table}
        </div>

        <div class="section">
          <h2>🎓 RESP Holdings (in CAD)</h2>
          {resp_table}
        </div>

        <div class="section">
          <h2>📈 중단기 투자의 통합 분석 (SCHD 제외)</h2>
          <p class="muted">
            ※ 가격 모멘텀·변동성·간단 밸류에이션·최근 뉴스(제목) 기반의 휴리스틱 지표입니다.
            실제 투자 판단은 별도 리스크 검토가 필요합니다.
          </p>
          {midterm_html}
        </div>

        <div class="section">
          <h2>💰 장기 투자의 배당금 분석 (SCHD)</h2>
          {schd_summary_text}
          <p class="muted">
            ※ 지난 10년(완료 연도) 배당·가격 데이터와 최근 5년/3년 성장률을 기반으로 한 단순 추정치입니다.
            DRIP과 매월 200 USD 추가 매수를 가정한 시뮬레이션입니다.
          </p>
          {schd_div_html}
        </div>
      </body>
    </html>
    """
    return html


# =========================
# 이메일 전송
# =========================

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


# =========================
# main
# =========================

def main():
    (
        df_hold,
        tfsa_cash_usd,
        resp_cash_cad,
        base_currency,
        tfsa_netdep_cad,
        resp_netdep_cad,
    ) = load_portfolio_from_gsheet()

    df_enriched, acc_summary = enrich_holdings_with_prices(
        df_hold,
        base_currency=base_currency,
        tfsa_cash_usd=tfsa_cash_usd,
        resp_cash_cad=resp_cash_cad,
        tfsa_netdep_cad=tfsa_netdep_cad,
        resp_netdep_cad=resp_netdep_cad,
    )

    html_doc = build_html_report(df_enriched, acc_summary)

    outname = f"portfolio_daily_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")

    subject = f"📊 Portfolio Daily Report - {datetime.now().strftime('%Y-%m-%d')}"
    send_email_html(subject, html_doc)


if __name__ == "__main__":
    # 1) 야후 파이낸스 디버그
    debug_yfinance_connectivity(["NVDA", "TSLA", "SCHD"])

    # 2) 기존 리포트 생성 로직
    main()
