import os
import io
import base64
import gspread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from openai import OpenAI
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    cash_usd = float(settings.get("CashUSD", 0))
    total_value = 0

    # 종목별 평가금액
    for idx, row in df_hold.iterrows():
        ticker = row["Ticker"]
        sh = float(row["Shares"])
        try:
            price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        except Exception:
            price = row["AvgPrice"]
        if pd.isna(price):  # fallback
            price = row["AvgPrice"]
        df_hold.loc[idx, "LastPrice"] = round(price, 2)
        df_hold.loc[idx, "Value"] = round(sh * price, 2)

    df_hold["Shares"] = df_hold["Shares"].round(2)
    df_hold["AvgPrice"] = df_hold["AvgPrice"].round(2)
    df_hold["Value"] = df_hold["Value"].round(2)
    total_value = df_hold["Value"].sum() + cash_usd

    # Weight 계산
    df_hold["Weight"] = round(df_hold["Value"] / total_value * 100, 2)

    # 현금 추가
    cash_row = {
        "Ticker": "CASH",
        "Shares": "-",
        "AvgPrice": "-",
        "LastPrice": 1.00,
        "Value": round(cash_usd, 2),
        "Weight": round(cash_usd / total_value * 100, 2)
    }
    df_hold = pd.concat([df_hold, pd.DataFrame([cash_row])], ignore_index=True)

    # 파이차트
    if total_value > 0:
        fig, ax = plt.subplots()
        ax.pie(df_hold["Weight"], labels=df_hold["Ticker"], autopct="%1.1f%%")
        ax.set_title("Portfolio Weights (포트폴리오 비중)")
        img = fig_to_base64(fig)
        plt.close(fig)
        pie_html = f"<img src='data:image/png;base64,{img}'/>"
    else:
        pie_html = "<p>No data for allocation chart</p>"

    return df_hold, pie_html

# ------------------------------------------------------------
# 주요 지수 섹션
# ------------------------------------------------------------
def index_section():
    tickers = {"S&P500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
    rows = []
    for name, t in tickers.items():
        try:
            df = yf.download(t, period="5d")
            last = round(float(df["Close"].iloc[-1]), 2)
            rows.append({"Index": name, "Value": last})
        except Exception:
            rows.append({"Index": name, "Value": "N/A"})
    df_idx = pd.DataFrame(rows)
    return "<h2>📈 Major Index (주요 지수)</h2>" + df_idx.to_html(index=False)

# ------------------------------------------------------------
# 뉴스 요약 섹션 (한글 번역 추가)
# ------------------------------------------------------------
def recent_news_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>뉴스 API 키가 없습니다.</p>"
    url = f"https://newsapi.org/v2/top-headlines?language=en&category=business&pageSize=5&apiKey={api_key}"
    r = requests.get(url)
    if r.status_code != 200:
        return "<h2>📰 Market News (시장 뉴스)</h2><p>뉴스 로드 실패</p>"
    articles = r.json().get("articles", [])
    items = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        date = (a.get("publishedAt") or "")[:10]
        trans = ""
        api_key_openai = os.environ.get("OPENAI_API_KEY")
        if api_key_openai:
            try:
                client = OpenAI(api_key=api_key_openai)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"Translate into Korean:\\nTitle: {title}\\nDescription: {desc}"}],
                    max_tokens=150
                )
                trans = resp.choices[0].message.content
            except:
                pass
        items.append(f"<div class='card'><b>{title}</b> <small>({date})</small><br><small>{desc}</small><br><i>{trans}</i></div>")
    return "<h2>📰 Market News (시장 뉴스)</h2>" + "".join(items)

# ------------------------------------------------------------
# 정책 포커스 섹션 (한글 번역 추가)
# ------------------------------------------------------------
def policy_focus_section():
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "<h2>📰 Policy Focus (정책 포커스)</h2><p>뉴스 API 키가 없습니다.</p>"
    url = f"https://newsapi.org/v2/everything?q=Trump+policy+economy&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    r = requests.get(url)
    if r.status_code != 200:
        return "<h2>📰 Policy Focus (정책 포커스)</h2><p>로드 실패</p>"
    articles = r.json().get("articles", [])
    items = []
    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        trans = ""
        api_key_openai = os.environ.get("OPENAI_API_KEY")
        if api_key_openai:
            try:
                client = OpenAI(api_key=api_key_openai)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"Translate into Korean:\nTitle: {title}\nDescription: {desc}"}],
                    max_tokens=150
                )
                trans = resp.choices[0].message.content
            except:
                pass
        items.append(f"<div class='card'><b>{title}</b><br><small>{desc}</small><br><i>{trans}</i></div>")
    return "<h2>📰 Policy Focus (정책 포커스)</h2>" + "".join(items)

# ------------------------------------------------------------
# GPT 투자 의견
# ------------------------------------------------------------
def gpt_investment_opinion(context):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "<p>OpenAI API 키가 없습니다.</p>"
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"아래 포트폴리오 현황을 요약하고 투자 의견을 5줄 이내 한국어로 제시:\n{context}"}],
            max_tokens=300
        )
        text = resp.choices[0].message.content
        return f"<div class='gpt-box'>{text}</div>"
    except Exception as e:
        return f"<p>Error: {e}</p>"

# ------------------------------------------------------------
# 메일 발송
# ------------------------------------------------------------
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
# 전체 리포트 조립 (레이아웃 개선)
# ------------------------------------------------------------
def build_report_html():
    df_hold, df_watch, settings = load_holdings_watchlist_settings()
    df_hold, pie_html = portfolio_analysis(df_hold, settings)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    context = f"Holdings: {df_hold.to_dict()} Settings: {settings}"

    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background:#fafafa; }
    h1 { text-align:center; }
    h2 { margin-top:30px; color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:5px; }
    table { border-collapse: collapse; width:100%; margin:10px 0; }
    th, td { border:1px solid #ddd; padding:8px; text-align:center; }
    th { background:#f4f6f6; }
    .card { background:white; border:1px solid #ddd; border-radius:8px; padding:10px; margin:10px 0; }
    .gpt-box { background:#f0f4ff; padding:15px; border-radius:10px; }
    </style>
    """

    html = f"""
    <html><head><meta charset='utf-8'>{style}</head><body>
    <h1>📊 Portfolio Report (포트폴리오 리포트)</h1>
    <p style='text-align:center;color:gray;'>Generated at {now}</p>

    <h2>📂 Holdings (보유 종목)</h2>
    {df_hold.to_html(index=False)}

    <h2>💰 Portfolio Allocation (포트폴리오 비중)</h2>
    <div class='card'>{pie_html}</div>

    {index_section()}
    {recent_news_section()}
    {policy_focus_section()}

    <h2>🤖 GPT Opinion (투자의견)</h2>
    {gpt_investment_opinion(context)}
    </body></html>
    """
    return html

# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main():
    html_doc = build_report_html()
    outname = f"portfolio_gsheet_policy_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")

    send_email_html(f"📊 Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}", html_doc)

if __name__ == "__main__":
    main()
