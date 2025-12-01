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
# 투자 분석 보조 함수 (중단기 + SCHD 배당 + 뉴스)
# =========================

def analyze_midterm_ticker(ticker):
    """
    yfinance 가격 데이터를 사용해
    - 6개월 모멘텀, 1년 변동성 등을 기반으로
      '중기 상승 확률, 매수/매도 타이밍, 1년 목표수익 범위, 리스크 요인'을
      단순 휴리스틱으로 계산한다.
    """
    try:
        hist = yf.Ticker(ticker).history(period="2y")
        closes = hist["Close"].dropna()
        if len(closes) < 60:
            raise ValueError("가격 데이터 부족")
    except Exception:
        return {
            "Ticker": ticker,
            "UpProb": "N/A",
            "BuyTiming": "N/A",
            "SellTiming": "N/A",
            "TargetRange": "데이터 부족",
            "Risk": "시세 데이터 부족",
        }

    last = float(closes.iloc[-1])

    # 6개월, 1년 수익률 (가능한 경우만)
    def pct_ret(days):
        if len(closes) <= days:
            return None
        start = float(closes.iloc[-days])
        return (last / start - 1.0) * 100.0 if start != 0 else None

    ret_6m = pct_ret(126)
    ret_1y = pct_ret(252)

    # 일간 로그수익률 → 연간 변동성
    rets = np.log(closes / closes.shift(1)).dropna()
    vol_annual = float(rets.std() * np.sqrt(252)) if len(rets) > 0 else 0.0

    # 상승 확률 휴리스틱: 50% + 모멘텀/2 - 변동성*10 (클리핑)
    base_prob = 50.0
    if ret_6m is not None:
        base_prob += ret_6m / 2.0
    prob = base_prob - vol_annual * 10.0
    prob = max(5.0, min(95.0, prob))

    # 매수 타이밍: 52주 고점 대비 괴리
    max_52w = float(closes[-252:].max()) if len(closes) >= 20 else float(closes.max())
    drawdown = (last / max_52w - 1.0) if max_52w > 0 else 0.0
    buy_timing = max(0.0, min(100.0, -drawdown * 200.0))  # 싸질수록 ↑

    # 매도 타이밍: 6개월 수익률과 변동성 결합
    sell_base = 0.0
    if ret_6m is not None:
        sell_base = max(0.0, min(100.0, (ret_6m - vol_annual * 100.0)))
    sell_timing = sell_base

    # 1년 목표 수익 범위: (모멘텀 기반 기대수익 ± 변동성)
    if ret_6m is not None:
        exp_ret = ret_6m * 2.0  # 단순 annualize
    elif ret_1y is not None:
        exp_ret = ret_1y
    else:
        exp_ret = 0.0

    vol_pct = vol_annual * 100.0
    low = exp_ret - vol_pct
    high = exp_ret + vol_pct
    low = max(-80.0, low)
    high = min(150.0, high)
    target_range = f"{low:,.1f}% ~ {high:,.1f}%"

    # 리스크 요인 텍스트
    if vol_annual > 0.6:
        risk = "매우 높은 변동성, 단기 급락 위험 큼"
    elif vol_annual > 0.4:
        risk = "높은 변동성, 거시/실적에 매우 민감"
    elif vol_annual > 0.25:
        risk = "중간 이상의 변동성, 조정 구간 주의"
    else:
        risk = "비교적 안정적이나 시장/섹터 리스크 존재"

    return {
        "Ticker": ticker,
        "UpProb": prob,
        "BuyTiming": buy_timing,
        "SellTiming": sell_timing,
        "TargetRange": target_range,
        "Risk": risk,
    }


def build_midterm_analysis_html(df_enriched):
    """
    TFSA 종목 중 SCHD를 제외한 티커에 대해
    중단기 투자 통합 분석 표를 HTML로 생성.
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

    rows_raw = []
    for t in sorted(tickers):
        stat = analyze_midterm_ticker(t)
        rows_raw.append(stat)

    rows = []
    for stat in rows_raw:
        rows.append(
            {
                "Ticker": stat["Ticker"],
                "중기 상승 확률 %": stat["UpProb"],
                "매수 타이밍 %": stat["BuyTiming"],
                "매도 타이밍 %": stat["SellTiming"],
                "1년 목표수익 범위": stat["TargetRange"],
                "리스크 요인": stat["Risk"],
            }
        )

    df_mid = pd.DataFrame(rows)

    # 색깔 적용 (상승확률, 매수/매도 타이밍)
    def colorize_pct_series(series):
        out = []
        for v in series:
            if isinstance(v, (int, float, float)):
                text = fmt_pct(v)
                out.append(colorize_value_html(text, v))
            else:
                out.append(v)
        return out

    df_mid["중기 상승 확률 %"] = colorize_pct_series(df_mid["중기 상승 확률 %"])
    df_mid["매수 타이밍 %"] = colorize_pct_series(df_mid["매수 타이밍 %"])
    df_mid["매도 타이밍 %"] = colorize_pct_series(df_mid["매도 타이밍 %"])

    return df_mid[
        ["Ticker", "중기 상승 확률 %", "매수 타이밍 %", "매도 타이밍 %", "1년 목표수익 범위", "리스크 요인"]
    ].to_html(index=False, escape=False)


def build_schd_dividend_html():
    """
    SCHD 지난 10년 배당/가격 기반으로 향후 2년 배당 예상 표 생성.
    """
    ticker = yf.Ticker("SCHD")
    try:
        hist = ticker.history(period="10y")
        divs = ticker.dividends.dropna()
    except Exception:
        return "<p>SCHD 배당 데이터를 불러오지 못했습니다.</p>"

    if hist is None or hist.empty or divs.empty:
        return "<p>SCHD 배당 데이터가 충분하지 않습니다.</p>"

    # 연도별 배당 합계
    div_by_year = divs.groupby(divs.index.year).sum()

    # 연도별 평균 가격
    price_by_year = hist["Close"].groupby(hist.index.year).mean()

    years = sorted(set(div_by_year.index) & set(price_by_year.index))
    if len(years) < 3:
        return "<p>SCHD 연도별 배당 데이터가 부족합니다.</p>"

    records = []
    for y in years:
        div_ps = float(div_by_year.get(y, 0.0))
        price_avg = float(price_by_year.get(y, np.nan))
        yield_pct = div_ps / price_avg * 100.0 if price_avg > 0 else np.nan
        records.append(
            {
                "Year": y,
                "Type": "Historical",
                "Avg Price": price_avg,
                "Dividend / Share": div_ps,
                "Dividend Yield %": yield_pct,
            }
        )

    df_hist = pd.DataFrame(records).sort_values("Year")

    # 최근 5년 기준 CAGR 계산 (배당, 가격)
    recent = df_hist[df_hist["Type"] == "Historical"].tail(5)
    if len(recent) >= 2:
        y0 = recent["Year"].iloc[0]
        yN = recent["Year"].iloc[-1]
        n_years = yN - y0
        if n_years <= 0:
            div_cagr = 0.0
            price_cagr = 0.0
        else:
            div_cagr = (
                (recent["Dividend / Share"].iloc[-1] / recent["Dividend / Share"].iloc[0]) ** (1 / n_years) - 1
            )
            price_cagr = (
                (recent["Avg Price"].iloc[-1] / recent["Avg Price"].iloc[0]) ** (1 / n_years) - 1
            )
    else:
        div_cagr = 0.0
        price_cagr = 0.0

    last_year = int(df_hist["Year"].max())
    last_div = float(df_hist[df_hist["Year"] == last_year]["Dividend / Share"].iloc[0])
    last_price = float(df_hist[df_hist["Year"] == last_year]["Avg Price"].iloc[0])

    forecast_records = []
    for i in range(1, 3):  # 향후 2년
        year_f = last_year + i
        div_f = last_div * ((1 + div_cagr) ** i)
        price_f = last_price * ((1 + price_cagr) ** i)
        yield_f = div_f / price_f * 100.0 if price_f > 0 else np.nan
        forecast_records.append(
            {
                "Year": year_f,
                "Type": "Forecast",
                "Avg Price": price_f,
                "Dividend / Share": div_f,
                "Dividend Yield %": yield_f,
            }
        )

    df_all = pd.concat([df_hist, pd.DataFrame(forecast_records)], ignore_index=True)
    df_all["Avg Price"] = df_all["Avg Price"].map(lambda x: fmt_money(x, "$"))
    df_all["Dividend / Share"] = df_all["Dividend / Share"].map(lambda x: fmt_money(x, "$"))
    df_all["Dividend Yield %"] = df_all["Dividend Yield %"].map(lambda x: fmt_pct(x) if pd.notnull(x) else "N/A")

    return df_all[["Year", "Type", "Avg Price", "Dividend / Share", "Dividend Yield %"]].to_html(
        index=False, escape=False
    )


def build_news_section_html(df_enriched):
    """
    TFSA 티커별로 yfinance 뉴스(.news)를 가져와
    날짜/출처/제목(링크)을 표로 만든다.
    """
    tfsa_tickers = (
        df_enriched[df_enriched["Type"].str.upper() == "TFSA"]["Ticker"]
        .dropna()
        .unique()
        .tolist()
    )
    if not tfsa_tickers:
        return "<p>TFSA 종목이 없습니다.</p>"

    sections = []
    for t in sorted(tfsa_tickers):
        try:
            news_items = yf.Ticker(t).news or []
        except Exception:
            news_items = []

        if not news_items:
            sections.append(f"<h3>{t}</h3><p>관련 뉴스 데이터를 불러올 수 없습니다.</p>")
            continue

        rows = []
        for item in news_items[:5]:
            title = item.get("title", "No title")
            link = item.get("link", "#")
            provider = item.get("provider", "") or item.get("publisher", "")
            ts = item.get("providerPublishTime") or item.get("published_time", None)
            if ts is not None:
                try:
                    dt = datetime.fromtimestamp(int(ts))
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = ""
            else:
                date_str = ""

            title_link = f'<a href="{link}">{title}</a>'
            rows.append(
                {
                    "Date": date_str,
                    "Source": provider,
                    "Title / Summary": title_link,
                }
            )

        df_news = pd.DataFrame(rows)
        sections.append(f"<h3>{t}</h3>" + df_news.to_html(index=False, escape=False))

    return "<br/>".join(sections)


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
    )  # CAD
    resp_yest_cad = account_summary.get("RESP", {}).get(
        "total_yesterday_native", 0.0
    )

    total_today_cad = tfsa_today_usd * usd_cad + resp_today_cad
    total_yest_cad = tfsa_yest_usd * usd_cad + resp_yest_cad
    total_diff_cad = total_today_cad - total_yest_cad
    total_diff_pct = (
        total_diff_cad / total_yest_cad * 100.0 if total_yest_cad != 0 else 0.0
    )

    total_today_str = fmt_money(total_today_cad, "$")
    total_diff_str = fmt_money(total_diff_cad, "$")
    total_diff_pct_str = fmt_pct(total_diff_pct)

    total_diff_str_colored = colorize_value_html(total_diff_str, total_diff_cad)
    total_diff_pct_str_colored = colorize_value_html(
        total_diff_pct_str, total_diff_pct
    )

    total_assets_line = (
        f"<p><strong>Total Assets (총 자산, CAD):</strong> "
        f"{total_today_str}&nbsp;&nbsp;&nbsp;"
        f"<strong>Δ vs. Yesterday (전일 대비 변화):</strong> "
        f"{total_diff_str_colored} ({total_diff_pct_str_colored})</p>"
    )

    # ---------- 1) 계좌 요약 테이블 (TFSA/RESP) ----------
    summary_rows = []
    for acc in ["TFSA", "RESP"]:
        if acc not in account_summary:
            continue
        s = account_summary[acc]

        acc_label = "TFSA (USD)" if acc == "TFSA" else "RESP (CAD)"

        total_today = s["total_today_native"]
        total_diff = s["total_diff_native"]
        total_diff_pct = s["total_diff_pct_native"]
        net_dep_native = s.get("net_deposit_native", 0.0)
        pl_vs_dep_native = s.get("pl_vs_deposit_native", 0.0)
        pl_vs_dep_pct_native = s.get("pl_vs_deposit_pct_native", 0.0)
        cash_native = s.get("cash_native", 0.0)

        total_today_str_acc = fmt_money(total_today, ccy_symbol)
        diff_str = fmt_money(total_diff, ccy_symbol)
        diff_pct_str = fmt_pct(total_diff_pct)
        net_dep_str = fmt_money(net_dep_native, ccy_symbol)
        pl_vs_dep_str = fmt_money(pl_vs_dep_native, ccy_symbol)
        pl_vs_dep_pct_str = fmt_pct(pl_vs_dep_pct_native)
        cash_str = fmt_money(cash_native, ccy_symbol)

        diff_str_colored = colorize_value_html(diff_str, total_diff)
        diff_pct_str_colored = colorize_value_html(diff_pct_str, total_diff_pct)
        pl_vs_dep_str_colored = colorize_value_html(pl_vs_dep_str, pl_vs_dep_native)
        pl_vs_dep_pct_str_colored = colorize_value_html(
            pl_vs_dep_pct_str, pl_vs_dep_pct_native
        )

        summary_rows.append(
            {
                "Account": acc_label,
                "Net Deposit (Base)": net_dep_str,
                "Total (Today, Base)": total_today_str_acc,
                "Δ vs Yesterday (Base)": diff_str_colored,
                "Δ %": diff_pct_str_colored,
                "P/L vs Deposit (Base)": pl_vs_dep_str_colored,
                "P/L vs Deposit %": pl_vs_dep_pct_str_colored,
                "Cash (Base)": cash_str,
            }
        )

    df_summary = pd.DataFrame(summary_rows)

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

    # ---------- 4) SCHD 배당 분석 ----------
    schd_div_html = build_schd_dividend_html()

    # ---------- 5) 뉴스/분석 섹션 ----------
    news_html = build_news_section_html(df_enriched)

    # ---------- 6) HTML 템플릿 ----------
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
          <h2>📈 중단기 투자의 통합 분석 (TFSA, SCHD 제외)</h2>
          <p class="muted">※ 단순 가격 모멘텀·변동성 기반 휴리스틱으로 계산된 참고용 지표입니다. 실제 투자 판단은 별도 리스크 검토가 필요합니다.</p>
          {midterm_html}
        </div>

        <div class="section">
          <h2>💰 장기 투자의 배당금 분석 (SCHD)</h2>
          {schd_div_html}
        </div>

        <div class="section">
          <h2>🔎 참고한 주요 뉴스/분석 (TFSA 보유 종목)</h2>
          {news_html}
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
    main()
