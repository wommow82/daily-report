
import os
import gspread
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from openai import OpenAI
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def _get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), scope)
    client = gspread.authorize(creds)
    return client

def _open_gsheet(gs_id):
    client = _get_gspread_client()
    return client.open_by_key(gs_id)

def load_holdings_watchlist_settings():
    gs_id = os.environ.get("GSHEET_ID")
    sh = _open_gsheet(gs_id)
    df_hold = pd.DataFrame(sh.worksheet("Holdings").get_all_records())
    df_watch = pd.DataFrame(sh.worksheet("Watchlist").get_all_records())
    df_set = pd.DataFrame(sh.worksheet("Settings").get_all_records())
    settings = dict(zip(df_set["Key"], df_set["Value"]))
    return df_hold, df_watch, settings

def safe_last_and_prev_close(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="10d", interval="1d")["Close"].dropna()
        if len(hist) >= 2:
            prev = float(hist.iloc[-2])
            last = float(hist.iloc[-1])
        elif len(hist) == 1:
            prev = last = float(hist.iloc[-1])
        else:
            return None, None
        return last, prev
    except Exception:
        return None, None

def colorize_change(last, prev, money=True):
    if last is None or prev is None:
        return "-", 0.0
    diff = last - prev
    pct = 0.0 if prev == 0 else (diff / prev) * 100.0
    color = "#1f6feb" if diff > 0 else ("#d73a49" if diff < 0 else "#6a737d")
    fmt_val = f"${last:,.2f}" if money else f"{last:,.2f}"
    html = f"<span style='color:{color}; font-weight:600'>{fmt_val} ({pct:+.2f}%)</span>"
    return html, pct

def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def holdings_section(df_hold, settings):
    rows = []
    total_value = 0.0
    cash_usd = float(settings.get("CashUSD", 0) or 0)

    for _, r in df_hold.iterrows():
        ticker = str(r.get("Ticker","")).strip()
        shares = float(r.get("Shares",0) or 0)
        avgp = float(r.get("AvgPrice",0) or 0)
        last, prev = safe_last_and_prev_close(ticker)
        if last is None:
            last = avgp
        if prev is None:
            prev = avgp
        value = shares * last
        prev_value = shares * prev
        total_value += value
        last_html, pct = colorize_change(last, prev, money=True)
        rows.append({
            "Ticker": ticker,
            "Shares": f"{shares:.2f}",
            "Avg Price (매입가)": money(avgp),
            "Last Price (현재가)": last_html,
            "Prev Close (전일종가)": money(prev),
            "Value (자산가치)": money(value),
            "Prev Value (전일자산)": money(prev_value),
        })

    total_value += cash_usd
    rows.append({
        "Ticker": "CASH",
        "Shares": "-",
        "Avg Price (매입가)": "-",
        "Last Price (현재가)": money(1.0),
        "Prev Close (전일종가)": money(1.0),
        "Value (자산가치)": money(cash_usd),
        "Prev Value (전일자산)": money(cash_usd),
    })

    headers = ["Ticker","Shares","Avg Price (매입가)","Last Price (현재가)","Prev Close (전일종가)","Value (자산가치)","Prev Value (전일자산)"]
    thead = "".join([f"<th>{h}</th>" for h in headers])
    trs = []
    for row in rows:
        tds = "".join([f"<td>{row[h]}</td>" for h in headers])
        trs.append(f"<tr>{tds}</tr>")
    table_html = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    return table_html, rows, total_value

def watchlist_section(df_watch):
    items = []
    for _, r in df_watch.iterrows():
        t = str(r.get("Ticker","")).strip()
        if not t:
            continue
        last, prev = safe_last_and_prev_close(t)
        if last is None or prev is None:
            cell = f"<b>{t}</b>: 데이터 없음"
        else:
            last_html, pct = colorize_change(last, prev, money=True)
            cell = f"<b>{t}</b>: {last_html}"
        items.append(f"<li>{cell}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>관심종목 데이터가 없습니다.</p>"

def index_section():
    idx_map = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
    rows = []
    for name, sym in idx_map.items():
        last, prev = safe_last_and_prev_close(sym)
        if last is None or prev is None:
            colored = "-"
        else:
            diff = last - prev
            pct = 0.0 if prev == 0 else (diff/prev)*100.0
            color = "#1f6feb" if diff > 0 else ("#d73a49" if diff < 0 else "#6a737d")
            colored = f"<span style='color:{color}; font-weight:600'>{money(last)} ({diff:+.2f}, {pct:+.2f}%)</span>"
        rows.append({"Index": name, "Latest": colored})
    thead = "<th>Index</th><th>Latest</th>"
    trs = "".join([f"<tr><td>{r['Index']}</td><td>{r['Latest']}</td></tr>" for r in rows])
    return "<h2>📈 Major Index (주요 지수)</h2><table><thead><tr>"+thead+"</tr></thead><tbody>"+trs+"</tbody></table>"

def monthly_economic_indicators_section():
    proxys = {
        "WTI Crude (유가)": "CL=F",
        "Gold (금)": "GC=F",
        "10Y Yield (미국10년)": "^TNX",
        "Dollar Index (달러지수)": "DX-Y.NYB",
        "VIX (변동성)": "^VIX"
    }
    start = date(date.today().year, 1, 1)
    end = date.today()
    months = pd.period_range(start=start, end=end, freq="M").to_timestamp()
    data = {}
    for label, sym in proxys.items():
        try:
            df = yf.download(sym, start=start, end=end, interval="1d")["Close"].dropna()
            m = df.resample("M").last()
            row = []
            for mth in months:
                v = m.loc[mth] if mth in m.index else np.nan
                row.append(v)
            data[label] = row
        except Exception:
            data[label] = [np.nan]*len(months)
    df = pd.DataFrame(data, index=[d.strftime("%b") for d in months]).T
    df = df.applymap(lambda x: "-" if pd.isna(x) else (f"{x:,.2f}"))
    return "<h2>📊 Economic Indicators (경제 지표)</h2>" + df.to_html(escape=False)

def recent_news_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>뉴스 API 키가 없습니다.</p>"
    url = f"https://newsapi.org/v2/top-headlines?language=en&category=business&pageSize=5&apiKey={api_key}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>뉴스 로드 실패</p>"
    articles = r.json().get("articles", [])
    items = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        date_str = (a.get("publishedAt") or "")[:10]
        trans = ""
        api_key_openai = os.environ.get("OPENAI_API_KEY")
        if api_key_openai and (title or desc):
            try:
                client = OpenAI(api_key=api_key_openai)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"Translate into Korean:\nTitle: {title}\nDescription: {desc}"}],
                    max_tokens=150
                )
                trans = resp.choices[0].message.content
            except Exception:
                pass
        items.append(f"<div class='card'><b>{title}</b> <small>({date_str})</small><br><small>{desc}</small><br><i>{trans}</i></div>")
    return "<h2>📰 Market News (시장 뉴스)</h2>" + "".join(items)

def policy_focus_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Policy Focus (정책 포커스)</h2><p>뉴스 API 키가 없습니다.</p>"
    url = f'https://newsapi.org/v2/everything?q=Trump+policy+economy&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}'
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return "<h2>📰 Policy Focus (정책 포커스)</h2><p>로드 실패</p>"
    articles = r.json().get("articles", [])
    items = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        date_str = (a.get("publishedAt") or "")[:10]
        trans = ""
        api_key_openai = os.environ.get("OPENAI_API_KEY")
        if api_key_openai and (title or desc):
            try:
                client = OpenAI(api_key=api_key_openai)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"Translate into Korean:\nTitle: {title}\nDescription: {desc}"}],
                    max_tokens=150
                )
                trans = resp.choices[0].message.content
            except Exception:
                pass
        items.append(f"<div class='card'><b>{title}</b> <small>({date_str})</small><br><small>{desc}</small><br><i>{trans}</i></div>")
    return "<h2>📰 Policy Focus (정책 포커스)</h2>" + "".join(items)

def strategy_summary_section(hold_rows):
    bullets = []
    for r in hold_rows:
        t = r["Ticker"]
        if t == "CASH":
            continue
        last_html = r["Last Price (현재가)"]
        pct = 0.0
        if "(" in last_html and "%" in last_html:
            try:
                pct_str = last_html.split("(")[1].split("%)")[0].replace("%","")
                pct = float(pct_str)
            except Exception:
                pass
        if pct >= 2:
            suggestion = "상승 모멘텀 — 분할매도 고려"
        elif pct <= -2:
            suggestion = "하락 — 리스크 점검/분할매수 검토"
        else:
            suggestion = "중립 — 보유 유지"
        bullets.append(f"🔵 <b>{t}</b>: {suggestion}")
    return "<h2>📝 Strategy Summary (전략 요약)</h2><div class='card'>" + "<br>".join(bullets) + "</div>"

def gpt_opinion_sections(df_hold, df_watch):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "<h2>🤖 GPT Opinion (투자의견)</h2><p>OpenAI API 키가 없습니다.</p>"
    client = OpenAI(api_key=api_key)
    try:
        ctx_hold = df_hold.to_dict(orient="records")
        resp1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"아래 '보유 종목' 데이터에 대해 핵심 포인트를 요약하고 오늘의 투자의견을 5줄 이내 한국어로 제시해줘:\n{ctx_hold}"}],
            max_tokens=250
        )
        text_hold = resp1.choices[0].message.content
    except Exception as e:
        text_hold = f"에러: {e}"
    try:
        ctx_watch = df_watch.to_dict(orient="records")
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"아래 '관심 종목' 데이터에 대해 관찰 포인트와 진입 전략을 5줄 이내 한국어로 제시해줘:\n{ctx_watch}"}],
            max_tokens=250
        )
        text_watch = resp2.choices[0].message.content
    except Exception as e:
        text_watch = f"에러: {e}"
    html = (
        "<h2>🤖 GPT Opinion (투자의견)</h2>"
        "<div class='gpt-box'><b>보유 종목 의견</b><br>" + text_hold + "</div>"
        "<div class='gpt-box'><b>관심 종목 의견</b><br>" + text_watch + "</div>"
    )
    return html

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

def build_report_html():
    df_hold, df_watch, settings = load_holdings_watchlist_settings()
    hold_table_html, hold_rows, total_value = holdings_section(df_hold, settings)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background:#fafafa; }
    h1 { text-align:center; }
    h2 { margin-top:30px; color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:5px; }
    table { border-collapse: collapse; width:100%; margin:10px 0; }
    th, td { border:1px solid #ddd; padding:8px; text-align:center; }
    th { background:#f4f6f6; }
    .card { background:white; border:1px solid #ddd; border-radius:8px; padding:10px; margin:10px 0; }
    .gpt-box { background:#f0f4ff; padding:15px; border-radius:10px; margin:10px 0; }
    </style>
    """
    html = f"""
    <html><head><meta charset='utf-8'>{style}</head><body>
    <h1>📊 Portfolio Report (포트폴리오 리포트)</h1>
    <p style='text-align:center;color:gray;'>Generated at {now}</p>

    <h2>📂 Holdings (보유 종목)</h2>
    {hold_table_html}

    <h2>👀 Watchlist (관심 종목)</h2>
    {watchlist_section(df_watch)}

    {index_section()}
    {monthly_economic_indicators_section()}
    {recent_news_section()}
    {policy_focus_section()}
    {strategy_summary_section(hold_rows)}
    {gpt_opinion_sections(df_hold, df_watch)}

    </body></html>
    """
    return html

def main():
    html_doc = build_report_html()
    outname = f"portfolio_gsheet_policy_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")
    send_email_html(f"📊 Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}", html_doc)

if __name__ == "__main__":
    main()
