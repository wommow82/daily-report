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
# 시세 / 환율 유틸
# =========================

def get_last_and_prev_close(ticker, period="5d"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            return None, None
        closes = hist["Close"].dropna()
        if len(closes) == 0:
            return None, None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
        return last, prev
    except Exception:
        return None, None


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

        last, prev = get_last_and_prev_close(ticker)
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
# 추가 분석용 헬퍼
# =========================

def _last_ma(series, window):
    """rollling MA; 데이터 부족시 마지막 값 사용."""
    series = series.dropna()
    if len(series) == 0:
        return None
    if len(series) < window:
        return float(series.iloc[-1])
    return float(series.rolling(window).mean().iloc[-1])


def build_midterm_analysis_table(df_enriched):
    """
    TFSA 종목(단, SCHD 제외)에 대해:
      - 중기 상승 확률 %
      - 매수 타이밍 %
      - 매도 타이밍 %
      - 1년 목표수익 범위 (USD, 보유수량 기준)
      - 리스크 요인 (텍스트)
    """
    df_tfsa = df_enriched[df_enriched["Type"].str.upper() == "TFSA"].copy()
    if df_tfsa.empty:
        return "<p>No TFSA holdings.</p>"

    tickers = sorted(df_tfsa["Ticker"].unique())
    tickers_mid = [t for t in tickers if t != "SCHD"]
    rows = []

    for ticker in tickers_mid:
        sub = df_tfsa[df_tfsa["Ticker"] == ticker]
        shares = safe_float(sub["Shares"].sum(), 0.0)

        try:
            hist = yf.Ticker(ticker).history(period="1y")
        except Exception:
            hist = None

        if hist is None or hist.empty or shares == 0:
            rows.append(
                {
                    "Ticker": ticker,
                    "중기 상승 확률 %": "N/A",
                    "매수 타이밍 %": "N/A",
                    "매도 타이밍 %": "N/A",
                    "1년 목표수익 범위 (USD)": "N/A",
                    "리스크 요인": "데이터 부족",
                }
            )
            continue

        close = hist["Close"].dropna()
        last = float(close.iloc[-1])
        ma20 = _last_ma(close, 20)
        ma50 = _last_ma(close, 50)
        ma200 = _last_ma(close, 200)
        high_52w = float(close.max())
        low_52w = float(close.min())

        # 1) 중기 상승 확률 (단순 규칙 기반 스코어)
        score = 50
        if ma200 is not None and last > ma200:
            score += 15
        elif ma200 is not None:
            score -= 15

        if ma50 is not None and last > ma50:
            score += 10
        elif ma50 is not None:
            score -= 10

        if ma20 is not None and last > ma20:
            score += 5
        elif ma20 is not None:
            score -= 5

        mid_prob = max(5, min(95, score))

        # 2) 매도/매수 타이밍: 52주 밴드 내 위치
        if high_52w == low_52w:
            pos = 0.5
        else:
            pos = (last - low_52w) / (high_52w - low_52w)  # 0=저점,1=고점

        sell_timing = int(round(20 + 80 * pos))      # 고점 근처일수록 높음
        buy_timing = int(round(90 - 70 * pos))       # 저점 근처일수록 높음
        sell_timing = max(5, min(95, sell_timing))
        buy_timing = max(5, min(95, buy_timing))

        # 3) 1년 목표수익 범위 (USD, 보유수량 기준, 과거 수익/변동성 기준)
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) > 20:
            ann_ret = float(daily_ret.mean() * 252)
            ann_vol = float(daily_ret.std() * np.sqrt(252))
        else:
            ann_ret = 0.0
            ann_vol = 0.3  # default

        low_ret = ann_ret - 0.5 * ann_vol
        high_ret = ann_ret + 0.5 * ann_vol
        # 범위 sanity
        low_ret = max(low_ret, -0.9)   # -90% 이하 잘라냄
        high_ret = min(high_ret, 1.5)  # +150% 이상 잘라냄

        cur_val = shares * last
        profit_low = cur_val * low_ret
        profit_high = cur_val * high_ret
        if profit_low > profit_high:
            profit_low, profit_high = profit_high, profit_low
        profit_range_str = f"{fmt_money(profit_low, '$')} ~ {fmt_money(profit_high, '$')}"

        # 4) 리스크 요인 (변동성 기준 텍스트)
        vol = abs(ann_vol)
        if vol >= 0.5:
            risk_text = "매우 높은 변동성, 급락 리스크 큼"
        elif vol >= 0.35:
            risk_text = "높은 변동성, 실적·뉴스에 민감"
        elif vol >= 0.25:
            risk_text = "중간 변동성, 경기·금리 변화 영향"
        else:
            risk_text = "비교적 안정적, 시장 전반 리스크 중심"

        rows.append(
            {
                "Ticker": ticker,
                "중기 상승 확률 %": f"{mid_prob}%",
                "매수 타이밍 %": f"{buy_timing}%",
                "매도 타이밍 %": f"{sell_timing}%",
                "1년 목표수익 범위 (USD)": profit_range_str,
                "리스크 요인": risk_text,
            }
        )

    if not rows:
        return "<p>No TFSA tickers for mid-term analysis.</p>"

    df_mid = pd.DataFrame(rows)
    return df_mid.to_html(index=False, escape=False)


def build_schd_dividend_table():
    """
    SCHD 지난 10년 연간 배당 + 연말가 + 배당수익률,
    최근 5년 배당 성장률 기반으로 향후 2년 예상 배당.
    """
    ticker = "SCHD"
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="10y")
        div = tkr.dividends
    except Exception:
        hist = None
        div = None

    if hist is None or hist.empty or div is None or div.empty:
        return "<p>SCHD 배당 데이터를 불러오지 못했습니다.</p>"

    price_year = hist["Close"].resample("Y").last()
    div_year = div.resample("Y").sum()

    df = pd.concat([price_year, div_year], axis=1)
    df.columns = ["Price", "Dividend"]
    df = df.dropna()
    df["Year"] = df.index.year
    df["YieldPct"] = (df["Dividend"] / df["Price"]) * 100.0

    df_hist = df.tail(10).copy()

    # 최근 5년 배당 CAGR 기반 예측
    recent = df_hist.tail(5)
    if len(recent) >= 2 and recent["Dividend"].iloc[0] > 0:
        cagr = (
            recent["Dividend"].iloc[-1] / recent["Dividend"].iloc[0]
        ) ** (1.0 / (len(recent) - 1)) - 1.0
    else:
        cagr = 0.0

    last_year = int(df_hist["Year"].iloc[-1])
    last_div = float(df_hist["Dividend"].iloc[-1])

    f1_div = last_div * (1 + cagr)
    f2_div = f1_div * (1 + cagr)

    rows = []
    for _, r in df_hist.iterrows():
        rows.append(
            {
                "Year": int(r["Year"]),
                "Annual Dividend (USD)": fmt_money(r["Dividend"], "$"),
                "Year-end Price (USD)": fmt_money(r["Price"], "$"),
                "Dividend Yield %": fmt_pct(r["YieldPct"]),
                "Type": "Actual",
            }
        )

    # 예상 2년 (가격은 N/A)
    rows.append(
        {
            "Year": last_year + 1,
            "Annual Dividend (USD)": fmt_money(f1_div, "$"),
            "Year-end Price (USD)": "N/A",
            "Dividend Yield %": "N/A",
            "Type": "Forecast",
        }
    )
    rows.append(
        {
            "Year": last_year + 2,
            "Annual Dividend (USD)": fmt_money(f2_div, "$"),
            "Year-end Price (USD)": "N/A",
            "Dividend Yield %": "N/A",
            "Type": "Forecast",
        }
    )

    df_out = pd.DataFrame(rows)
    return df_out.to_html(index=False, escape=False)


def build_news_table(df_enriched):
    """
    TFSA 전체 티커에 대해 yfinance 뉴스 사용.
    - 한 티커당 최대 3건
    - 제목을 한 줄 요약 + 링크로 사용
    """
    df_tfsa = df_enriched[df_enriched["Type"].str.upper() == "TFSA"].copy()
    if df_tfsa.empty:
        return "<p>No TFSA holdings for news.</p>"

    tickers = sorted(df_tfsa["Ticker"].unique())
    rows = []

    for ticker in tickers:
        try:
            news_list = yf.Ticker(ticker).news
        except Exception:
            news_list = []

        if not news_list:
            continue

        for item in news_list[:3]:
            title = item.get("title", "").strip()
            link = item.get("link", "")
            publisher = item.get("publisher", "")
            ts = item.get("providerPublishTime") or item.get("pubDate") or 0
            try:
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
            except Exception:
                date_str = ""

            headline_link = (
                f'<a href="{link}" target="_blank">{title}</a>' if link else title
            )
            summary = item.get("summary") or title

            rows.append(
                {
                    "Ticker": ticker,
                    "Date": date_str,
                    "Source": publisher,
                    "Headline / Summary": summary,
                    "Link": headline_link,
                }
            )

    if not rows:
        return "<p>No recent news found via Yahoo Finance.</p>"

    df_news = pd.DataFrame(rows)
    df_news = df_news.sort_values(["Ticker", "Date"], ascending=[True, False])
    return df_news.to_html(index=False, escape=False)


# =========================
# HTML 리포트 생성
# =========================

def build_html_report(df_enriched, account_summary):
    base_ccy = account_summary["meta"]["base_currency"]
    ccy_symbol = "$"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 전체 자산 CAD 기준 한 줄 요약 ----------
    usd_cad = get_usd_cad_rate()

    tfsa_today_usd = account_summary.get("TFSA", {}).get(
        "total_today_native", 0.0
    )  # USD
    tfsa_yest_usd = account_summary.get("TFSA", {}).get(
        "total_yesterday_native", 0.0
    )
    resp_today_cad = account_summary.get("RESP", {}).get(
        "total_today_native", 0.0
    )
