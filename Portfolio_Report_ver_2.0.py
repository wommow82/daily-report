import os
import time
import smtplib
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from email.mime_text import MIMEText
from email.mime.multipart import MIMEMultipart


# =========================
# 공통 유틸
# =========================

def safe_float(val, default=0.0):
    try:
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def colorize_number(val, text=None):
    """
    숫자 값에 따라 색상 span 태그 반환
    - 양수: 초록색
    - 음수: 빨간색
    - 0 또는 None: 원본 텍스트 그대로
    """
    if text is None:
        text = str(val)

    if val is None:
        return text

    try:
        val = float(val)
    except Exception:
        return text

    if val > 0:
        color = "#008000"  # green
    elif val < 0:
        color = "#cc0000"  # red
    else:
        return text

    return f'<span style="color:{color}">{text}</span>'


import os

# =========================
# 헬퍼 함수 (요약)
# =========================

import json
from datetime import datetime, timedelta

def _short_ko_summary_15(text):
    """
    주어진 영어 뉴스 텍스트를 한국어 15자 내외로 아주 짧게 요약.

    - OPENAI_API_KEY 필요
    - 에러나 키 없으면 기본 문구 반환
    """
    text = (text or "").strip()
    if not text:
        return "요약불가"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "요약불가"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = (
            "다음 뉴스 내용을 바탕으로, 주가에 중요한 핵심만 "
            "한국어 15자 이내로 아주 짧게 요약해줘.\n"
            "문장 1개, 불필요한 수식어 최소화:\n\n"
            f"{text}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        summary = (resp.choices[0].message.content or "").strip()
        summary = summary.replace("\n", " ").strip()
        return summary[:15] if summary else "요약실패"
    except Exception as e:
        print(f"[WARN] _short_ko_summary_15 오류: {e}")
        return "요약실패"


def _classify_news_sentiment_and_pick_reps(ticker, articles):
    """
    기사 리스트에 대해 긍정/부정 감성을 분류하고,
    각 그룹에서 '대표 기사 최대 2개'를 골라 15자 요약을 만든다.

    입력:
        ticker   : 종목 티커 (예: "TSLA")
        articles : _fetch_news_for_ticker_midterm 결과 리스트
                   각 원소는 {title, description, source, published} 형태 가정

    반환:
        {
          "pos_count": int,             # 긍정 뉴스 수
          "neg_count": int,             # 부정 뉴스 수
          "pos_repr": str or None,      # 긍정 대표: A요약 | B요약
          "neg_repr": str or None,      # 부정 대표: C요약 | D요약
        }
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not articles:
        return {
            "pos_count": 0,
            "neg_count": 0,
            "pos_repr": None,
            "neg_repr": None,
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[WARN] _classify_news_sentiment_and_pick_reps import 오류: {e}")
        return {
            "pos_count": 0,
            "neg_count": 0,
            "pos_repr": None,
            "neg_repr": None,
        }

    # 1) 번호를 붙여 프롬프트 구성
    items = []
    for i, a in enumerate(articles, start=1):
        title = (a.get("title") or "").strip()
        desc = (a.get("description") or "").strip()
        src = (a.get("source") or "").strip()
        date = a.get("published") or ""
        item = f"[{i}] {date} {src} - {title}"
        if desc:
            item += f"\n{desc}"
        items.append(item)

    bundle_text = "\n\n".join(items)

    prompt = f"""
너는 미국 주식 애널리스트이다.
아래는 {ticker} 관련 뉴스 목록이다.

각 뉴스가 주가에 미치는 방향성을
'긍정', '부정', '중립' 중 하나로만 분류해라.

JSON 형식으로만 답하라. 예시는 다음과 같다.
{{
  "items": [
    {{"index": 1, "sentiment": "긍정"}},
    {{"index": 2, "sentiment": "부정"}},
    {{"index": 3, "sentiment": "중립"}}
  ]
}}

뉴스 목록:
{bundle_text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        items_sent = data.get("items", [])
    except Exception as e:
        print(f"[WARN] 감성 분류 실패, 모두 중립 처리: {e}")
        items_sent = []

    # 2) 분류 결과 집계
    pos_idx = set()
    neg_idx = set()

    for x in items_sent:
        try:
            idx = int(x.get("index"))
            sent = (x.get("sentiment") or "").strip()
        except Exception:
            continue
        if sent == "긍정":
            pos_idx.add(idx)
        elif sent == "부정":
            neg_idx.add(idx)

    pos_count = len(pos_idx)
    neg_count = len(neg_idx)

    # 3) 대표 인덱스 (최신순이라고 가정하고 작은 index 우선)
    pos_sorted = sorted(pos_idx)
    neg_sorted = sorted(neg_idx)

    def _build_repr(indices):
        if not indices:
            return None
        chosen = list(indices)[:2]  # 최대 2개
        summaries = []
        for i in chosen:
            if 1 <= i <= len(articles):
                art = articles[i - 1]
                text = (art.get("title") or "") + "\n" + (art.get("description") or "")
                summaries.append(_short_ko_summary_15(text))
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]
        return f"{summaries[0]} | {summaries[1]}"

    pos_repr = _build_repr(pos_sorted)
    neg_repr = _build_repr(neg_sorted)

    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_repr": pos_repr,
        "neg_repr": neg_repr,
    }


def _fetch_news_for_ticker_midterm(ticker, api_key, page_size=3, days=7):
    """
    종목 뉴스 가져오기 (중기 분석용):
    - 1순위: NewsAPI
    - 2순위: Google News RSS fallback

    Returns:
        list of dict: [{title, url, source, published}, ...]
    """
    from datetime import datetime, timedelta
    import requests
    import feedparser

    articles = []

    # 1️⃣ NewsAPI 시도
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "from": (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "source": a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", "")[:10],
                    "description": a.get("description", ""),
                })
    except Exception as e:
        print(f"⚠️ NewsAPI 오류(midterm): {e}")

    # 2️⃣ fallback → Google News RSS
    if not articles:
        try:
            rss_url = (
                f"https://news.google.com/rss/search?"
                f"q={ticker}+stock&hl=en&gl=US&ceid=US:en"
            )
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:page_size]:
                src = "Google News"
                if hasattr(entry, "source") and getattr(entry, "source"):
                    try:
                        src = getattr(entry, "source").get("title", "Google News")
                    except Exception:
                        src = "Google News"

                published = ""
                if hasattr(entry, "published"):
                    published = entry.published[:16]

                articles.append({
                    "title": entry.title,
                    "url": entry.link,
                    "source": src,
                    "published": published,
                    "description": getattr(entry, "summary", ""),
                })
        except Exception as e:
            print(f"⚠️ Google News RSS 오류(midterm): {e}")

    return articles


# =========================
# NewsAPI/RSS 기반 종목 뉴스 → 주가 영향 중심 요약 → HTML
# =========================

def build_midterm_news_comment_from_apis_combined(ticker, max_items=10, days=30):
    """
    중기 분석 섹션에서 사용할 '최근 1개월 뉴스 요약' HTML 생성.

    - 조회 기간: 기본 최근 30일 (1개월)
    - 소스: NewsAPI → 실패 시 Google News RSS
    - 최대 max_items개 기사 사용
    - 티커/회사명 포함 여부로 1차 필터링
    - 기사들을 최신순으로 정렬
    - OpenAI로 긍정/부정 감성 분류
    - 긍정/부정 뉴스 갯수 표시
    - 긍정/부정 각각 대표 뉴스 최대 2개를 뽑아
      15자 내외 한글 요약을 "A요약 | B요약" 형식으로 출력
      · 긍정: 초록색
      · 부정: 빨간색

    반환:
        HTML 문자열 (<p> ... </p>)
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 1개월):</strong><br>"
            "- NEWS_API_KEY가 설정되어 있지 않아 뉴스를 불러올 수 없습니다."
            "</p>"
        )

    # 1) NewsAPI + Google News로 기사 목록 가져오기 (최근 days일 기준)
    articles = _fetch_news_for_ticker_midterm(
        ticker=ticker,
        api_key=api_key,
        page_size=max_items,
        days=days,
    )

    if not articles:
        return (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 1개월):</strong><br>"
            f"- 최근 {days}일 내 {ticker} 관련 주요 뉴스를 찾지 못했습니다."
            "</p>"
        )

    # 1-1) 실제 날짜 기준 최근 30일만 추가 필터링
    cutoff = datetime.utcnow() - timedelta(days=30)
    filtered_recent = []
    for a in articles:
        p = (a.get("published") or "").strip()
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(p[:len(fmt)], fmt)
                break
            except Exception:
                continue
        if dt is None:
            filtered_recent.append(a)
        else:
            if dt >= cutoff:
                filtered_recent.append(a)

    if not filtered_recent:
        return (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 1개월):</strong><br>"
            f"- 최근 30일 내 {ticker} 관련 유효한 날짜의 뉴스를 찾지 못했습니다."
            "</p>"
        )

    articles = filtered_recent

    # 2) 티커/회사명 기준으로 관련 기사 필터링
    ticker_upper = ticker.upper()
    keywords = [ticker_upper]

    company_map = {
        "NVDA": "NVIDIA",
        "TSLA": "TESLA",
        "SCHD": "SCHD",
    }
    if ticker_upper in company_map:
        keywords.append(company_map[ticker_upper].upper())

    filtered = []
    for a in articles:
        text_all = (
            (a.get("title") or "") + " " + (a.get("description") or "")
        ).upper()
        if any(k in text_all for k in keywords):
            filtered.append(a)

    # 필터링 결과가 너무 적으면, 원본 리스트도 일부 사용
    if len(filtered) >= 3:
        use_articles = filtered[:max_items]
    else:
        use_articles = articles[:max_items]

    # 2-1) 최신 뉴스 우선 정렬 (published 기준 내림차순)
    def _parse_dt(a):
        p = (a.get("published") or "").strip()
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(p[:len(fmt)], fmt)
            except Exception:
                continue
        return datetime.min

    use_articles = sorted(use_articles, key=_parse_dt, reverse=True)

    # 3) 긍정/부정 분류 및 대표 뉴스 선별 + 15자 요약(최대 2개씩, "A | B" 형식)
    sent_info = _classify_news_sentiment_and_pick_reps(ticker, use_articles)
    pos_count = sent_info["pos_count"]
    neg_count = sent_info["neg_count"]
    pos_repr = sent_info["pos_repr"]
    neg_repr = sent_info["neg_repr"]

    # 4) HTML 구성 (색상 강조)
    lines = []
    lines.append(
        f"<span style='color:green;'>긍정 뉴스 {pos_count}건</span>, "
        f"<span style='color:red;'>부정 뉴스 {neg_count}건</span>"
    )

    if pos_repr:
        lines.append(
            f"<span style='color:green;'>· 대표 긍정: {pos_repr}</span>"
        )
    if neg_repr:
        lines.append(
            f"<span style='color:red;'>· 대표 부정: {neg_repr}</span>"
        )

    html_body = "<br>".join(lines)

    html = (
        "<p style='text-align:left;'>"
        "<strong>뉴스 요약 (최근 1개월, 주가 영향 이슈):</strong><br>"
        f"{html_body}"
        "</p>"
    )
    return html


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

def get_last_and_prev_close(ticker, period="2y"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if len(hist) < 2:
            return None, None

        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        return float(last), float(prev)
    except Exception:
        return None, None


def get_fx_rate_usdcad():
    try:
        fx = yf.Ticker("CAD=X").history(period="2d")
        if fx.empty:
            return None
        return float(fx["Close"].iloc[-1])
    except Exception:
        return None


# =========================
# 포트폴리오/시트 로딩
# =========================

def load_holdings_from_gsheet(sheet):
    """
    Google Sheet에서 Holdings 탭을 읽어 DataFrame으로 반환.
    """
    try:
        ws = sheet.worksheet("Holdings")
    except gspread.WorksheetNotFound:
        raise RuntimeError("Holdings 워크시트를 찾을 수 없습니다.")

    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df


def load_settings_from_gsheet(sheet):
    """
    Settings 탭에서 계정별 설정값(예: NetDeposit, 목표치 등)을 읽어온다.
    """
    try:
        ws = sheet.worksheet("Settings")
    except gspread.WorksheetNotFound:
        raise RuntimeError("Settings 워크시트를 찾을 수 없습니다.")

    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df


# =========================
# HTML 빌더 (보유 종목/계좌 요약 등)
# =========================

def build_holdings_table_html(df_holdings, account_name="TFSA"):
    """
    보유 종목 테이블을 HTML로 생성.
    df_holdings: 해당 계좌(TFSA/RESP)만 필터링된 DataFrame
    """
    if df_holdings.empty:
        return f"<p>{account_name} 계좌의 보유 종목이 없습니다.</p>"

    cols = ["Ticker", "Name", "Currency", "Shares", "AvgPrice", "MarketPrice", "MarketValue", "GainLoss", "GainLossPct"]
    df = df_holdings.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    df["MarketValue"] = df["MarketValue"].apply(lambda x: float(x) if x not in [None, ""] else 0.0)
    df["GainLoss"] = df["GainLoss"].apply(lambda x: float(x) if x not in [None, ""] else 0.0)
    df["GainLossPct"] = df["GainLossPct"].apply(lambda x: float(x) if x not in [None, ""] else 0.0)

    headers = [
        "Ticker",
        "종목명",
        "통화",
        "보유주식",
        "평단가",
        "현재가",
        "평가금액",
        "손익",
        "손익률",
    ]
    html = [
        f"<h3>{account_name} 보유 종목</h3>",
        "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse;'>",
        "<thead><tr>",
    ]
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        gain = row["GainLoss"]
        gain_pct = row["GainLossPct"]
        gain_html = colorize_number(gain, f"{gain:,.2f}")
        gain_pct_html = colorize_number(gain_pct, f"{gain_pct:.2f}%")

        html.append("<tr>")
        html.append(f"<td>{row['Ticker']}</td>")
        html.append(f"<td>{row['Name']}</td>")
        html.append(f"<td>{row['Currency']}</td>")
        html.append(f"<td>{row['Shares']}</td>")
        html.append(f"<td>{row['AvgPrice']}</td>")
        html.append(f"<td>{row['MarketPrice']}</td>")
        html.append(f"<td>{row['MarketValue']:,.2f}</td>")
        html.append(f"<td>{gain_html}</td>")
        html.append(f"<td>{gain_pct_html}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    return "".join(html)


def build_account_summary_html(df_holdings, fx_usdcad):
    """
    전체 계좌(TFSA + RESP)의 평가금액 및 손익 요약 HTML.
    """
    if df_holdings.empty:
        return "<p>보유 종목 데이터가 없습니다.</p>"

    df = df_holdings.copy()
    for col in ["MarketValue", "GainLoss"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].apply(lambda x: float(x) if x not in [None, ""] else 0.0)

    total_mv_usd = df[df["Currency"] == "USD"]["MarketValue"].sum()
    total_mv_cad = df[df["Currency"] == "CAD"]["MarketValue"].sum()
    total_gain_usd = df[df["Currency"] == "USD"]["GainLoss"].sum()
    total_gain_cad = df[df["Currency"] == "CAD"]["GainLoss"].sum()

    if fx_usdcad:
        total_mv_usd_in_cad = total_mv_usd * fx_usdcad
        total_gain_usd_in_cad = total_gain_usd * fx_usdcad
    else:
        total_mv_usd_in_cad = None
        total_gain_usd_in_cad = None

    total_mv_cad_all = total_mv_cad + (total_mv_usd_in_cad or 0.0)
    total_gain_cad_all = total_gain_cad + (total_gain_usd_in_cad or 0.0)

    gain_html = colorize_number(total_gain_cad_all, f"{total_gain_cad_all:,.2f} CAD")

    html = [
        "<h3>전체 계좌 요약 (TFSA + RESP)</h3>",
        "<ul>",
        f"<li>총 평가금액 (CAD 환산): {total_mv_cad_all:,.2f} CAD</li>",
        f"<li>총 손익 (CAD 환산): {gain_html}</li>",
        "</ul>",
    ]
    return "".join(html)


# =========================
# Mid-term (6~12개월) 분석 섹션
# =========================

def analyze_midterm_ticker(ticker):
    """
    개별 종목에 대한 6~12개월 중기 분석 HTML 생성.

    - yfinance로 가격/변동성 등 지표 수집
    - build_midterm_news_comment_from_apis_combined() 사용
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1y")
        if hist.empty:
            return f"<p>{ticker}: 최근 1년 데이터가 없습니다.</p>"

        last_price = hist["Close"].iloc[-1]
        start_price = hist["Close"].iloc[0]
        ret_1y = (last_price / start_price - 1.0) * 100.0

        # 단순 연 변동성 (일간 수익률의 표준편차 × sqrt(252))
        daily_ret = hist["Close"].pct_change().dropna()
        vol_annual = daily_ret.std() * np.sqrt(252) * 100.0

        try:
            fwd_pe = data.info.get("forwardPE", None)
        except Exception:
            fwd_pe = None

        pe_text = f"{fwd_pe:.1f}배" if fwd_pe not in [None, 0] else "N/A"

        if ret_1y > 30:
            ret_comment = "강한 상승 추세 구간"
        elif ret_1y > 0:
            ret_comment = "완만한 상승 추세"
        elif ret_1y > -20:
            ret_comment = "보합~조정 구간"
        else:
            ret_comment = "뚜렷한 하락 추세"

        if vol_annual > 60:
            vol_comment = "매우 높은 변동성"
        elif vol_annual > 30:
            vol_comment = "중간 수준 변동성"
        else:
            vol_comment = "비교적 안정적 변동성"

        if fwd_pe and fwd_pe > 60:
            val_comment = "밸류에이션 부담 구간 가능성"
        elif fwd_pe and fwd_pe > 25:
            val_comment = "성장주 프리미엄 구간"
        elif fwd_pe:
            val_comment = "상대적으로 합리적인 밸류에이션"
        else:
            val_comment = "밸류에이션 정보 부족"

        rows = []
        rows.append(
            (
                "1년 수익률",
                f"{ret_1y:+.1f}%",
                f"최근 1년간 종가 기준 주가 수익률 – {ret_comment}",
            )
        )
        rows.append(
            (
                "연 변동성",
                f"{vol_annual:.1f}%",
                f"연 환산 가격 등락 폭 – {vol_comment}",
            )
        )
        rows.append(
            (
                "Fwd PER",
                pe_text,
                f"향후 1년 예상 이익 대비 현재 주가 배수 – {val_comment}",
            )
        )

        html_parts = []
        html_parts.append(f"<h3>📊 {ticker} 6~12개월 중기 지표 요약</h3>")
        html_parts.append(
            "<table border='1' cellspacing='0' cellpadding='4' "
            "style='border-collapse:collapse;'>"
        )
        html_parts.append("<thead><tr><th>지표</th><th>값</th><th>해석</th></tr></thead><tbody>")

        for name, val, desc in rows:
            html_parts.append("<tr>")
            html_parts.append(f"<td>{name}</td>")
            html_parts.append(f"<td>{val}</td>")
            html_parts.append(f"<td>{desc}</td>")
            html_parts.append("</tr>")

        html_parts.append("</tbody></table>")

        comment_html = build_midterm_news_comment_from_apis_combined(ticker)
        html_parts.append(comment_html)

        return "".join(html_parts)

    except Exception as e:
        return f"<p>{ticker} 중기 분석 중 오류 발생: {e}</p>"


# =========================
# HTML 리포트 전체 빌더
# =========================

def build_html_report(df_holdings, settings_df, fx_usdcad):
    """
    이메일로 보낼 전체 HTML 리포트 생성.
    - Holdings를 TFSA / RESP로 나누어 테이블 생성
    - 계좌 요약
    - Mid-term Investment Analysis (NVDA, TSLA 등)
    """
    html = []
    html.append("<html><body>")
    html.append(f"<h2>📈 Daily Portfolio Report ({datetime.now().strftime('%Y-%m-%d')})</h2>")

    html.append("<hr>")
    html.append("<h2>📌 전체 계좌 요약</h2>")
    html.append(build_account_summary_html(df_holdings, fx_usdcad))

    html.append("<hr>")
    html.append("<h2>📂 계좌별 보유 종목</h2>")

    if "Account" in df_holdings.columns:
        tfsa_df = df_holdings[df_holdings["Account"] == "TFSA"]
        resp_df = df_holdings[df_holdings["Account"] == "RESP"]
    else:
        tfsa_df = df_holdings.copy()
        resp_df = df_holdings.iloc[0:0].copy()

    html.append(build_holdings_table_html(tfsa_df, "TFSA"))
    html.append("<br>")
    html.append(build_holdings_table_html(resp_df, "RESP"))

    html.append("<hr>")
    html.append("<h2>📈 Mid-term Investment Analysis (6~12개월)</h2>")
    html.append("<p style='font-size:12px;color:#555;'>"
                "※ 예시: NVDA, TSLA에 대해 6~12개월 관점의 지표/뉴스 요약을 제공합니다."
                "</p>")

    for ticker in ["NVDA", "TSLA"]:
        html.append("<hr>")
        html.append(analyze_midterm_ticker(ticker))

    html.append("</body></html>")
    return "".join(html)


# =========================
# 이메일 전송
# =========================

def send_email_report(html_body):
    """
    SMTP를 이용해 HTML 리포트 메일 발송
    """
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    mail_from = os.environ.get("MAIL_FROM")
    mail_to = os.environ.get("MAIL_TO")

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, mail_from, mail_to]):
        raise EnvironmentError("SMTP 관련 환경변수가 충분히 설정되어 있지 않습니다.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = mail_from
    msg["To"] = mail_to

    part_html = MIMEText(html_body, "html")
    msg.attach(part_html)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(mail_from, [mail_to], msg.as_string())


# =========================
# 메인 실행 로직
# =========================

def main():
    gsheet_id = os.environ.get("GSHEET_ID")
    if not gsheet_id:
        raise EnvironmentError("환경변수 GSHEET_ID 가 설정되어 있지 않습니다.")

    sheet = open_gsheet(gsheet_id)

    df_holdings = load_holdings_from_gsheet(sheet)
    settings_df = load_settings_from_gsheet(sheet)

    fx_usdcad = get_fx_rate_usdcad()

    html = build_html_report(df_holdings, settings_df, fx_usdcad)
    send_email_report(html)


if __name__ == "__main__":
    main()
