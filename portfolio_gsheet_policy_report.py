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
# 정책 포커스 섹션 (수정된 부분)
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
        content = title + " " + desc
        items.append(f"<li>{title}<br><small>{desc}</small></li>")

    html = "<h2>📰 Policy Focus (정책 포커스)</h2><ul>" + "".join(items) + "</ul>"
    return html

# ------------------------------------------------------------
# 전체 리포트 조립
# ------------------------------------------------------------
def build_report_html():
    df_hold, df_watch, settings = load_holdings_watchlist_settings()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""
    <html><head><meta charset='utf-8'></head><body>
    <h1>📊 Portfolio Report (포트폴리오 리포트)</h1>
    <p>Generated at {now}</p>
    {policy_focus_section()}
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

if __name__ == "__main__":
    main()
