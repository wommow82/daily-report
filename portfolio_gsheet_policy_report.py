import os
import io
import base64
import gspread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import openai
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta

# ------------------------------------------------------------
# Google Sheets 연결
# ------------------------------------------------------------
def _get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), scope)
    client = gspread.authorize(creds)
    return client

def _open_gsheet(gs_id):
    client = _get_gspread_client()
    return client.open_by_key(gs_id)

# ------------------------------------------------------------
# 데이터 로딩
# ------------------------------------------------------------
def load_holdings_watchlist_settings():
    gs_id = os.environ.get("GSHEET_ID")
    sh = _open_gsheet(gs_id)
    df_hold = pd.DataFrame(sh.worksheet("Holdings").get_all_records())
    df_watch = pd.DataFrame(sh.worksheet("Watchlist").get_all_records())
    df_set = pd.DataFrame(sh.worksheet("Settings").get_all_records())
    settings = dict(zip(df_set["Key"], df_set["Value"]))
    return df_hold, df_watch, settings

# ------------------------------------------------------------
# 차트 생성 유틸
# ------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ------------------------------------------------------------
# 포트폴리오 분석
# ------------------------------------------------------------
def portfolio_analysis(df_hold, settings):
    df_hold["Value"] = df_hold["Shares"] * df_hold["AvgPrice"]
    total_value = df_hold["Value"].sum() + float(settings.get("CashUSD", 0))
    df_hold["Weight"] = df_hold["Value"] / total_value * 100

    fig, ax = plt.subplots()
    ax.pie(df_hold["Weight"], labels=df_hold["Ticker"], autopct="%1.1f%%")
    ax.set_title("Portfolio Weights (포트폴리오 비중)")
    img = fig_to_base64(fig)
    plt.close(fig)

    return df_hold, f"<img src='data:image/png;base64,{img}'/>"

# ------------------------------------------------------------
# 주요 지수 섹션
# ------------------------------------------------------------
def index_section():
    tickers = ["^GSPC", "^IXIC", "^DJI"]
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period="5d")
            data[t] = round(df["Close"].iloc[-1], 2)
        except Exception:
            data[t] = "N/A"
    html = "<h2>📈 Major Index (주요 지수)</h2><ul>"
    for k,v in data.items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"
    return html

# ------------------------------------------------------------
# 뉴스 요약 섹션
# ------------------------------------------------------------
def recent_news_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>No NEWS_API_KEY provided.</p>"
    url = f"https://newsapi.org/v2/top-headlines?language=en&category=business&pageSize=5&apiKey={api_key}"
    r = requests.get(url)
    if r.status_code != 200:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>Failed to fetch news.</p>"
    articles = r.json().get("articles", [])
    items = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        items.append(f"<li>{title}<br><small>{desc}</small></li>")
    return "<h2>📰 Market News (시장 뉴스)</h2><ul>" + "".join(items) + "</ul>"

# ------------------------------------------------------------
# 정책 포커스 섹션 (수정 완료)
# ------------------------------------------------------------
def policy_focus_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Policy Focus (정책 포커스)</h2><p>No NEWS_API_KEY provided.</p>"
    url = f"https://newsapi.org/v2/everything?q=Trump+policy+economy&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    r = requests.get(url)
    if r.status_code != 200:
        return "<h2>📰 Policy Focus (정책 포커스)</h2><p>Failed to fetch news.</p>"
    articles = r.json().get("articles", [])
    items = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        items.append(f"<li>{title}<br><small>{desc}</small></li>")
    return "<h2>📰 Policy Focus (정책 포커스)</h2><ul>" + "".join(items) + "</ul>"

# ------------------------------------------------------------
# GPT 투자 의견 (선택)
# ------------------------------------------------------------
def gpt_investment_opinion(context):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "<h2>🤖 GPT Opinion (투자의견)</h2><p>No OPENAI_API_KEY provided.</p>"
    openai.api_key = api_key
    try:
        prompt = f"Summarize and give investment opinion for context: {context}"
        resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=200)
        text = resp["choices"][0]["text"]
        return f"<h2>🤖 GPT Opinion (투자의견)</h2><p>{text}</p>"
    except Exception as e:
        return f"<h2>🤖 GPT Opinion (투자의견)</h2><p>Error: {e}</p>"

# ------------------------------------------------------------
# 전체 리포트 조립
# ------------------------------------------------------------
def build_report_html():
    df_hold, df_watch, settings = load_holdings_watchlist_settings()
    df_hold, pie_html = portfolio_analysis(df_hold, settings)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    context = f"Holdings: {df_hold.to_dict()} Settings: {settings}"

    html = f"""
    <html><head><meta charset='utf-8'></head><body>
    <h1>📊 Portfolio Report (포트폴리오 리포트)</h1>
    <p>Generated at {now}</p>
    <h2>📂 Holdings (보유 종목)</h2>
    {df_hold.to_html(index=False)}
    <h2>💰 Portfolio Allocation (포트폴리오 비중)</h2>
    {pie_html}
    {index_section()}
    {recent_news_section()}
    {policy_focus_section()}
    {gpt_investment_opinion(context)}
    </body></html>
    """
    return html

# ------------------------------------------------------------
# 메일 전송
# ------------------------------------------------------------

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main():
    html_doc = build_report_html()
    outname = f"portfolio_gsheet_policy_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")

    # 메일 발송
    send_email_html(f"📊 Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}", html_doc)

if __name__ == "__main__":
    main()
