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
        return f"{float(x):,.{digits}f}{'' if currency_symbol == '' else ''}".join([currency_symbol, ""])[0:len(currency_symbol)+len(f"{float(x):,.{digits}f}")]
    except Exception:
        try:
            float(x)
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
    """
    값이 양수면 초록, 음수면 빨강 폰트로 감싸는 helper.
    text: 이미 포맷된 문자열 (예: "$123.45", "3.21%")
    raw_value: 부호 판단용 숫자값
    """
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
    """
    ServiceAccountCredentials + gspread 로 Google Sheets 클라이언트를 생성.
    필요 환경변수:
        - GOOGLE_APPLICATION_CREDENTIALS: 서비스 계정 JSON 파일 경로
    """
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
    """
    Google Sheet 열기 (503 에러 대비 재시도 포함)

    필요 환경변수:
        - GSHEET_ID: 포트폴리오 스프레드시트 ID
    """
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
    """
    단일 종목의 마지막 종가 / 그 전 종가를 반환.
    조회 실패 시 (None, None) 반환.
    """
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
    """
    Yahoo Finance 의 'CAD=X' (USD/CAD) 환율 사용.
    - 반환값: 1 USD = rate CAD
    - 실패하면 1.35 (기본값) 사용
    """
    try:
        hist = yf.Ticker("CAD=X").history(period="5d")
        if hist is None or hist.empty:
            return 1.35
        rate = float(hist["Close"].dropna().iloc[-1])
        return rate if rate > 0 else 1.35
    except Exception:
        return 1.35


def get_fx_multipliers(base_currency):
    """
    BaseCurrency 에 따라 USD/CAD → 기준통화 변환 계수 리턴.
    - base_currency: 'USD' 또는 'CAD'

    반환:
        fx_usd_to_base, fx_cad_to_base
    """
    base = (base_currency or "USD").upper()
    usd_cad = get_usd_cad_rate()  # 1 USD = usd_cad CAD

    if base == "USD":
        fx_usd_to_base = 1.0
        fx_cad_to_base = 1.0 / usd_cad
    elif base == "CAD":
        fx_usd_to_base = usd_cad
        fx_cad_to_base = 1.0
    else:
        # 기타 통화는 임시로 1:1
        fx_usd_to_base = 1.0
        fx_cad_to_base = 1.0

    return fx_usd_to_base, fx_cad_to_base


# =========================
# Google Sheet 로드 / 전처리
# =========================

def load_portfolio_from_gsheet():
    """
    Google Sheet (GSHEET_ID)에서
    - Holdings 시트
    - Settings 시트
    를 읽어 포트폴리오 정보를 반환.

    구조:
        Holdings:
            - Ticker
            - Shares
            - AvgPrice
            - Type (TFSA / RESP)

        Settings:
            - TFSA_CashUSD
            - RESP_CashCAD
            - TFSA_NetDepositCAD
            - RESP_NetDepositCAD
            - BaseCurrency ('USD' 또는 'CAD')
        (기존 CashUSD 만 있던 경우 TFSA_CashUSD가 없으면 fallback)
    """
    gs_id = os.environ.get("GSHEET_ID")
    sh = open_gsheet(gs_id)

    # Holdings
    ws_hold = sh.worksheet("Holdings")
    df_hold = pd.DataFrame(ws_hold.get_all_records())

    # Settings
    ws_settings = sh.worksheet("Settings")
    df_settings = pd.DataFrame(ws_settings.get_all_records())

    if "Key" not in df_settings.columns or "Value" not in df_settings.columns:
        raise ValueError("Settings 시트에는 'Key', 'Value' 열이 필요합니다.")

    settings = dict(zip(df_settings["Key"].astype(str), df_settings["Value"]))

    # 현금
    tfsa_cash_usd = safe_float(
        settings.get("TFSA_CashUSD", settings.get("CashUSD", 0.0)), 0.0
    )
    resp_cash_cad = safe_float(settings.get("RESP_CashCAD", 0.0), 0.0)

    # 순투입자본 (CAD 기준)
    tfsa_netdep_cad = safe_float(settings.get("TFSA_NetDepositCAD", 0.0), 0.0)
    resp_netdep_cad = safe_float(settings.get("RESP_NetDepositCAD", 0.0), 0.0)

    base_currency = str(settings.get("BaseCurrency", "USD")).upper()

    # Holdings 전처리
    for col in ["Ticker", "Shares", "AvgPrice"]:
        if col not in df_hold.columns:
            raise ValueError(f"'Holdings' 시트에 '{col}' 열이 없습니다.")

    df_hold["Ticker"] = df_hold["Ticker"].astype(str).str.strip().str.upper()
    df_hold["Shares"] = pd.to_numeric(df_hold["Shares"], errors="coerce").fillna(0.0)
    df_hold["AvgPrice"] = pd.to_numeric(df_hold["AvgPrice"], errors="coerce").fillna(
        0.0
    )

    # Type 열: 없으면 기본값 TFSA
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
    Holdings DataFrame 에 시세와 평가액/손익을 붙이고,
    TFSA / RESP / TOTAL 요약을 함께 반환.

    반환:
        df_enriched, account_summary
    """
    df = df_hold.copy()

    fx_usd_to_base, fx_cad_to_base = get_fx_multipliers(base_currency)

    accounts = ["TFSA", "RESP"]
    summary = {
        acc: {
            "holdings_value_today": 0.0,
            "holdings_value_yesterday": 0.0,
            "cash_native": 0.0,
            "cash_base": 0.0,
            "net_deposit_cad": 0.0,
            "net_deposit_base": 0.0,
            "pl_vs_deposit_base": 0.0,
            "pl_vs_deposit_pct": 0.0,
        }
        for acc in accounts
    }

    # 계좌별 현금
    summary["TFSA"]["cash_native"] = tfsa_cash_usd
    summary["TFSA"]["cash_base"] = tfsa_cash_usd * fx_usd_to_base

    summary["RESP"]["cash_native"] = resp_cash_cad
    summary["RESP"]["cash_base"] = resp_cash_cad * fx_cad_to_base

    # 순투입자본(CAD) → 기준통화로 환산
    summary["TFSA"]["net_deposit_cad"] = tfsa_netdep_cad
    summary["RESP"]["net_deposit_cad"] = resp_netdep_cad

    summary["TFSA"]["net_deposit_base"] = tfsa_netdep_cad * fx_cad_to_base
    summary["RESP"]["net_deposit_base"] = resp_netdep_cad * fx_cad_to_base

    # 결과 컬럼 초기화
    df["LastPrice"] = np.nan
    df["PrevClose"] = np.nan
    df["LastPriceBase"] = np.nan
    df["PrevCloseBase"] = np.nan
    df["PositionValueNative"] = np.nan
    df["PositionValueBase"] = np.nan
    df["PositionPrevValueBase"] = np.nan
    df["ProfitLossBase"] = np.nan
    df["ProfitLossPct"] = np.nan

    for idx, row in df.iterrows():
        ticker = row["Ticker"]
        shares = safe_float(row["Shares"], 0.0)
        avg_price = safe_float(row["AvgPrice"], 0.0)
        acc_type = str(row["Type"]).upper()
        if acc_type not in accounts:
            acc_type = "TFSA"
            df.at[idx, "Type"] = "TFSA"

        # 계좌별 통화 가정
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
        profit_pct = (profit_base / cost_base * 100.0) if cost_base != 0 else 0.0

        df.at[idx, "LastPrice"] = last
        df.at[idx, "PrevClose"] = prev
        df.at[idx, "LastPriceBase"] = last * fx_to_base
        df.at[idx, "PrevCloseBase"] = prev * fx_to_base
        df.at[idx, "PositionValueNative"] = position_value_native
        df.at[idx, "PositionValueBase"] = position_value_base
        df.at[idx, "PositionPrevValueBase"] = position_prev_value_base
        df.at[idx, "ProfitLossBase"] = profit_base
        df.at[idx, "ProfitLossPct"] = profit_pct

        summary[acc_type]["holdings_value_today"] += position_value_base
        summary[acc_type]["holdings_value_yesterday"] += position_prev_value_base

    # 계좌별 today / yesterday / Δ 및 deposit 대비 손익
    for acc in accounts:
        today = summary[acc]["holdings_value_today"] + summary[acc]["cash_base"]
        yesterday = summary[acc]["holdings_value_yesterday"] + summary[acc]["cash_base"]
        diff = today - yesterday
        pct = (diff / yesterday * 100.0) if yesterday != 0 else 0.0

        net_dep_base = summary[acc]["net_deposit_base"]
        pl_vs_dep = today - net_dep_base
        pl_vs_dep_pct = (pl_vs_dep / net_dep_base * 100.0) if net_dep_base != 0 else 0.0

        summary[acc]["total_today"] = today
        summary[acc]["total_yesterday"] = yesterday
        summary[acc]["total_diff"] = diff
        summary[acc]["total_diff_pct"] = pct
        summary[acc]["pl_vs_deposit_base"] = pl_vs_dep
        summary[acc]["pl_vs_deposit_pct"] = pl_vs_dep_pct

    # TOTAL (TFSA + RESP)
    total_today = summary["TFSA"]["total_today"] + summary["RESP"]["total_today"]
    total_yesterday = (
        summary["TFSA"]["total_yesterday"] + summary["RESP"]["total_yesterday"]
    )
    total_diff = total_today - total_yesterday
    total_pct = (total_diff / total_yesterday * 100.0) if total_yesterday != 0 else 0.0

    total_net_dep_cad = (
        summary["TFSA"]["net_deposit_cad"] + summary["RESP"]["net_deposit_cad"]
    )
    # TOTAL 순투입자본(CAD)도 CAD→base 로 변환
    fx_usd_to_base, fx_cad_to_base = get_fx_multipliers(base_currency)
    total_net_dep_base = total_net_dep_cad * fx_cad_to_base
    total_pl_vs_dep = total_today - total_net_dep_base
    total_pl_vs_dep_pct = (
        total_pl_vs_dep / total_net_dep_base * 100.0
        if total_net_dep_base != 0
        else 0.0
    )

    summary["TOTAL"] = {
        "total_today": total_today,
        "total_yesterday": total_yesterday,
        "total_diff": total_diff,
        "total_diff_pct": total_pct,
        "cash_native": summary["TFSA"]["cash_native"]
        + summary["RESP"]["cash_native"],
        "cash_base": summary["TFSA"]["cash_base"] + summary["RESP"]["cash_base"],
        "net_deposit_cad": total_net_dep_cad,
        "net_deposit_base": total_net_dep_base,
        "pl_vs_deposit_base": total_pl_vs_dep,
        "pl_vs_deposit_pct": total_pl_vs_dep_pct,
    }

    summary["meta"] = {
        "base_currency": base_currency,
        "fx_usd_to_base": fx_usd_to_base,
        "fx_cad_to_base": fx_cad_to_base,
    }

    return df, summary


# =========================
# HTML 리포트 생성
# =========================

def build_html_report(df_enriched, account_summary):
    base_ccy = account_summary["meta"]["base_currency"]
    ccy_symbol = "$"  # CAD / USD 모두 $ 표시

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) 계좌 요약 테이블 구성
    summary_rows = []
    for acc in ["TFSA", "RESP", "TOTAL"]:
        if acc not in account_summary:
            continue
        s = account_summary[acc]

        # raw values
        total_today = s["total_today"]
        total_diff = s["total_diff"]
        total_diff_pct = s["total_diff_pct"]
        net_dep_cad = s.get("net_deposit_cad", 0.0)
        net_dep_base = s.get("net_deposit_base", 0.0)
        pl_vs_dep = s.get("pl_vs_deposit_base", 0.0)
        pl_vs_dep_pct = s.get("pl_vs_deposit_pct", 0.0)

        # 포맷
        total_today_str = fmt_money(total_today, ccy_symbol)
        diff_str = fmt_money(total_diff, ccy_symbol)
        diff_pct_str = fmt_pct(total_diff_pct)
        net_dep_cad_str = fmt_money(net_dep_cad, "C$")
        net_dep_base_str = fmt_money(net_dep_base, ccy_symbol)
        pl_vs_dep_str = fmt_money(pl_vs_dep, ccy_symbol)
        pl_vs_dep_pct_str = fmt_pct(pl_vs_dep_pct)

        # 색상 적용 (오름/내림)
        diff_str_colored = colorize_value_html(diff_str, total_diff)
        diff_pct_str_colored = colorize_value_html(diff_pct_str, total_diff_pct)
        pl_vs_dep_str_colored = colorize_value_html(pl_vs_dep_str, pl_vs_dep)
        pl_vs_dep_pct_str_colored = colorize_value_html(
            pl_vs_dep_pct_str, pl_vs_dep_pct
        )

        summary_rows.append(
            {
                "Account": acc,
                f"Total (Today, {base_ccy})": total_today_str,
                f"Δ vs Yesterday ({base_ccy})": diff_str_colored,
                "Δ %": diff_pct_str_colored,
                "Net Deposit (CAD)": net_dep_cad_str,
                f"Net Deposit ({base_ccy})": net_dep_base_str,
                "P/L vs Deposit": pl_vs_dep_str_colored,
                "P/L vs Deposit %": pl_vs_dep_pct_str_colored,
                "Cash (base)": fmt_money(s.get("cash_base", 0.0), ccy_symbol),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    # 2) 상세 보유 종목 테이블 (TFSA/RESP)
    df_view = df_enriched.copy()

    # 기본 포맷
    df_view["Shares"] = df_view["Shares"].map(lambda x: f"{float(x):,.2f}")
    df_view["AvgPrice"] = df_view["AvgPrice"].map(
        lambda x: fmt_money(x, ccy_symbol)
    )
    df_view["LastPriceBase"] = df_view["LastPriceBase"].map(
        lambda x: fmt_money(x, ccy_symbol)
    )
    df_view["PositionValueBase"] = df_view["PositionValueBase"].map(
        lambda x: fmt_money(x, ccy_symbol)
    )

    # Profit/Loss 컬럼은 색깔 입혀서 포맷
    raw_pl_base = df_enriched["ProfitLossBase"].tolist()
    raw_pl_pct = df_enriched["ProfitLossPct"].tolist()

    pl_base_str_list = []
    for v in raw_pl_base:
        v_num = safe_float(v, 0.0)
        text = fmt_money(v_num, ccy_symbol)
        pl_base_str_list.append(colorize_value_html(text, v_num))

    pl_pct_str_list = []
    for v in raw_pl_pct:
        v_num = safe_float(v, 0.0)
        text = fmt_pct(v_num)
        pl_pct_str_list.append(colorize_value_html(text, v_num))

    df_view["ProfitLossBase"] = pl_base_str_list
    df_view["ProfitLossPct"] = pl_pct_str_list

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
    for col in cols_order:
        if col not in df_view.columns:
            raise ValueError(f"Missing column in df_view: {col}")

    def _table_for_account(acc_type):
        sub = df_view[df_view["Type"].str.upper() == acc_type].copy()
        if sub.empty:
            return f"<p>No holdings for {acc_type}.</p>"
        return sub[cols_order].to_html(index=False, escape=False)

    tfsa_table = _table_for_account("TFSA")
    resp_table = _table_for_account("RESP")

    # 3) HTML 템플릿
    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background:#fafafa; }
    h1 { text-align:center; }
    h2 { margin-top:30px; color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:5px; }
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
    # 1) Google Sheet 에서 포트폴리오 로드
    (
        df_hold,
        tfsa_cash_usd,
        resp_cash_cad,
        base_currency,
        tfsa_netdep_cad,
        resp_netdep_cad,
    ) = load_portfolio_from_gsheet()

    # 2) 시세/평가액 계산
    df_enriched, acc_summary = enrich_holdings_with_prices(
        df_hold,
        base_currency=base_currency,
        tfsa_cash_usd=tfsa_cash_usd,
        resp_cash_cad=resp_cash_cad,
        tfsa_netdep_cad=tfsa_netdep_cad,
        resp_netdep_cad=resp_netdep_cad,
    )

    # 3) HTML 리포트 생성
    html_doc = build_html_report(df_enriched, acc_summary)

    # 4) 로컬 파일 저장 (선택)
    outname = f"portfolio_daily_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")

    # 5) 이메일 발송
    subject = f"📊 Portfolio Daily Report - {datetime.now().strftime('%Y-%m-%d')}"
    send_email_html(subject, html_doc)


if __name__ == "__main__":
    main()
