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


def colorize_midterm_metric(metric_name, value):
    """
    중단기 분석용 퍼센트 색칠 함수.

    metric_name:
      - "UpProb"     : 중기 상승 확률 %
      - "BuyTiming"  : 매수 타이밍 %
      - "SellTiming" : 매도 타이밍 %

    색 규칙:
      1) UpProb, BuyTiming (높을수록 '좋음'):
         - >= 70 : 초록 (green)
         - 40~69 : 주황 (orange)
         - <  40 : 빨강 (red)

      2) SellTiming (높을수록 '매도 신호'):
         - >= 70 : 빨강 (red)
         - 40~69 : 주황 (orange)
         - <  40 : 초록 (green)
    """
    if value is None:
        return "N/A"

    try:
        v = float(value)
    except Exception:
        return "N/A"

    metric_name = str(metric_name)

    if metric_name in ("UpProb", "BuyTiming"):
        # 높을수록 좋은 경우
        if v >= 70:
            color = "green"
        elif v >= 40:
            color = "orange"
        else:
            color = "red"
    elif metric_name == "SellTiming":
        # 높을수록 매도 신호인 경우 (반대 의미)
        if v >= 70:
            color = "red"
        elif v >= 40:
            color = "orange"
        else:
            color = "green"
    else:
        # 정의되지 않은 metric은 색칠하지 않음
        return f"{v:.0f}%"

    return f'<span style="color:{color}; font-weight:bold;">{v:.0f}%</span>'


import os

# =========================
# 헬퍼 함수 (요약)
# =========================

import json
from datetime import datetime, timedelta





def _html_escape(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def build_earnings_analysis_by_metric_html(ticker: str, fundamentals: dict) -> str:
    """
    항목별(매출/운영소득/순이익)로 POS/NEG/NEU를 각각 판단해 표시하는 HTML.

    - 기존 파일에 이미 있는 함수/헬퍼를 최대한 재사용:
      * _format_big_number(n): B/M 축약
      * _html_escape(s): HTML escape

    출력 예(색상 적용):
      · [POS][매출] 51.24B → 47.52B 대비 증가
      · [POS][운영소득] 20.54B → 20.44B 소폭 증가
      · [NEG][순이익] 2.71B → 18.34B 대비 감소
    """

    t = (ticker or "").strip().upper()
    if not t:
        return "<p style='text-align:left;'><strong>실적 분석(항목별):</strong> 업데이트 없음</p>"

    if not fundamentals:
        return "<p style='text-align:left;'><strong>실적 분석(항목별):</strong> 업데이트 없음</p>"

    # 근거 수치 추출
    rev_last = fundamentals.get("revenue_last")
    rev_prev = fundamentals.get("revenue_prev")
    op_last = fundamentals.get("operating_income_last")
    op_prev = fundamentals.get("operating_income_prev")
    ni_last = fundamentals.get("net_income_last")
    ni_prev = fundamentals.get("net_income_prev")

    # 최소 하나라도 있어야 의미가 있음
    has_any = any(v is not None for v in [rev_last, rev_prev, op_last, op_prev, ni_last, ni_prev])
    if not has_any:
        return "<p style='text-align:left;'><strong>실적 분석(항목별):</strong> 업데이트 없음</p>"

    # 비교 함수
    def _cmp(a, b):
        try:
            if a is None or b is None:
                return None
            a = float(a)
            b = float(b)
            if a > b:
                return 1
            if a < b:
                return -1
            return 0
        except Exception:
            return None

    # 태그 매핑
    def _tag(cmp_result):
        if cmp_result is None:
            return "[NEU]"
        if cmp_result > 0:
            return "[POS]"
        if cmp_result < 0:
            return "[NEG]"
        return "[NEU]"

    # 태그별 스타일
    def _styles(tag):
        # (tag_style, text_style)
        if tag == "[POS]":
            return ("color:green; font-weight:700;", "color:green;")
        if tag == "[NEG]":
            return ("color:red; font-weight:700;", "color:red;")
        return ("color:black; font-weight:700;", "color:black;")

    # 소폭 변화 판단(상대 3% 미만이면 '소폭' 표시)
    def _nuance(last, prev, cmp_result):
        if cmp_result not in (1, -1):
            return ""
        try:
            if prev is None:
                return ""
            prev_f = float(prev)
            last_f = float(last)
            if prev_f == 0:
                return ""
            pct = (last_f - prev_f) / abs(prev_f)
            return "소폭 " if abs(pct) < 0.03 else ""
        except Exception:
            return ""

    # 한 줄 생성
    def _line(metric_ko, last, prev):
        c = _cmp(last, prev)
        tag = _tag(c)
        tag_style, txt_style = _styles(tag)

        last_s = _format_big_number(last) if last is not None else "확인 불가"
        prev_s = _format_big_number(prev) if prev is not None else "확인 불가"

        if c is None:
            change = "비교 불가"
        elif c > 0:
            change = "증가"
        elif c < 0:
            change = "감소"
        else:
            change = "변화 없음"

        nu = _nuance(last, prev, c)

        tag_html = f"<span style='{tag_style}'>{_html_escape(tag)}</span>"
        metric_html = f"<span style='{tag_style}'>[{_html_escape(metric_ko)}]</span>"
        txt = f"{last_s} → {prev_s} 대비 {nu}{change}"
        txt_html = f"<span style='{txt_style}'>{_html_escape(txt)}</span>"

        return f"· {tag_html} {metric_html} {txt_html}"

    lines = [
        _line("매출", rev_last, rev_prev),
        _line("운영소득", op_last, op_prev),
        _line("순이익", ni_last, ni_prev),
    ]

    # 전부 비교 불가면 업데이트 없음
    if _cmp(rev_last, rev_prev) is None and _cmp(op_last, op_prev) is None and _cmp(ni_last, ni_prev) is None:
        return "<p style='text-align:left;'><strong>실적 분석(항목별):</strong> 업데이트 없음</p>"

    return (
        "<p style='text-align:left;'>"
        "<strong>실적 분석(항목별):</strong><br>"
        + "<br>".join(lines)
        + "</p>"
    )


def fetch_recent_quarterly_fundamentals_yf(ticker: str) -> dict:
    """
    yfinance로부터 최근 분기 손익(가능 범위) 근거 수치를 수집한다.
    - 반환 dict는 GPT 실적 분석에 들어가는 근거 데이터
    - 데이터가 없으면 최소 dict({ticker})만 반환하거나 빈 dict 반환
    """
    import pandas as pd

    t = (ticker or "").strip().upper()
    if not t:
        return {}

    try:
        import yfinance as yf
    except Exception:
        return {}

    out = {"ticker": t}

    try:
        tk = yf.Ticker(t)

        qf = None
        try:
            qf = tk.quarterly_financials
        except Exception:
            qf = None

        # 버전에 따라 다른 속성일 수 있어 fallback
        if qf is None or getattr(qf, "empty", True):
            try:
                qf = tk.income_stmt
            except Exception:
                qf = None

        if qf is None or getattr(qf, "empty", True):
            return out

        cols = list(qf.columns)
        if not cols:
            return out

        # 날짜 정렬(방어)
        try:
            cols_sorted = sorted(cols, key=lambda x: pd.to_datetime(x, errors="coerce"))
        except Exception:
            cols_sorted = cols

        last = cols_sorted[-1]
        prev = cols_sorted[-2] if len(cols_sorted) >= 2 else None

        def _get(row_names, col):
            for rn in row_names:
                if rn in qf.index:
                    v = qf.loc[rn, col]
                    try:
                        return float(v)
                    except Exception:
                        try:
                            return float(pd.to_numeric(v, errors="coerce"))
                        except Exception:
                            return None
            return None

        out["quarter_last"] = str(last)
        out["revenue_last"] = _get(["Total Revenue", "Revenue", "TotalRevenue"], last)
        out["operating_income_last"] = _get(["Operating Income", "OperatingIncome"], last)
        out["net_income_last"] = _get(["Net Income", "NetIncome"], last)

        if prev is not None:
            out["quarter_prev"] = str(prev)
            out["revenue_prev"] = _get(["Total Revenue", "Revenue", "TotalRevenue"], prev)
            out["operating_income_prev"] = _get(["Operating Income", "OperatingIncome"], prev)
            out["net_income_prev"] = _get(["Net Income", "NetIncome"], prev)

        return out

    except Exception:
        return out


def _format_big_number(n):
    """
    숫자 축약 표시(B/M). 실패 시 '확인 불가'
    """
    try:
        x = float(n)
    except Exception:
        return "확인 불가"

    absx = abs(x)
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    return f"{x:,.0f}"





# =========================
# 뉴스 여러개 종합요약 함수
# =========================

def _summarize_news_bundle_ko_price_focus(ticker, articles):
    """
    여러 개의 영어 뉴스(articles)를 받아서
    각 기사별로
      - 주가 영향 관점 감성(긍정/부정/중립)
      - 한국어 한 줄 요약
    을 뽑은 뒤,

    최종적으로는 다음 형식의 여러 줄 텍스트를 만든다:

      긍정 뉴스 X건, 부정 뉴스 Y건
      · 대표 긍정: A기사 요약 | B기사 요약
      · 대표 부정: C기사 요약 | D기사 요약

    - 긍정/부정 각각 최대 2개까지 사용
    - 기사 수가 부족하면 있는 만큼만 사용
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "긍정 뉴스 0건, 부정 뉴스 0건"

    if not articles:
        return "긍정 뉴스 0건, 부정 뉴스 0건"

    try:
        from openai import OpenAI
    except ImportError:
        return "긍정 뉴스 0건, 부정 뉴스 0건"

    client = OpenAI(api_key=api_key)

    # 1) 기사 묶음을 번호 붙여 하나의 텍스트로 만든다.
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

    # 2) 각 기사별 감성 + 한국어 요약을 JSON으로 요청
    prompt = f"""
너는 미국 주식 애널리스트이다.
아래는 {ticker} 관련 뉴스 목록이다.

각 뉴스에 대해 다음 정보를 추출하라:

- sentiment: "긍정", "부정", "중립" 중 하나
- summary_ko: 주가에 중요한 내용을 담은 한국어 한 문장 요약 (20~30자 내외)

JSON 형식으로만 답하라. 예시는 다음과 같다.
{{
  "items": [
    {{"index": 1, "sentiment": "긍정", "summary_ko": "테슬라 유럽 판매 회복으로 수요 기대"}},
    {{"index": 2, "sentiment": "부정", "summary_ko": "유럽 보조금 축소로 전기차 성장 둔화 우려"}},
    ...
  ]
}}

뉴스 목록:
{bundle_text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        items_data = data.get("items", [])
    except Exception as e:
        print(f"[WARN] _summarize_news_bundle_ko_price_focus JSON 파싱 오류: {e}")
        return "긍정 뉴스 0건, 부정 뉴스 0건"

    # 3) 긍정/부정 리스트로 분리 (기사 순서 유지)
    pos_summaries = []
    neg_summaries = []

    for x in items_data:
        try:
            idx = int(x.get("index"))
        except Exception:
            continue
        if not (1 <= idx <= len(articles)):
            continue

        sent = (x.get("sentiment") or "").strip()
        summary_ko = (x.get("summary_ko") or "").strip()
        if not summary_ko:
            continue

        # 너무 길면 살짝 자르기
        if len(summary_ko) > 40:
            summary_ko = summary_ko[:40]

        if sent == "긍정":
            pos_summaries.append(summary_ko)
        elif sent == "부정":
            neg_summaries.append(summary_ko)
        else:
            # 중립은 여기서는 사용하지 않음
            pass

    pos_count = len(pos_summaries)
    neg_count = len(neg_summaries)

    lines = []
    lines.append(f"긍정 뉴스 {pos_count}건, 부정 뉴스 {neg_count}건")

    # 4) 긍정 대표 줄 1개 (최대 2개 기사 요약 사용)
    if pos_count > 0:
        left = pos_summaries[0]
        right = pos_summaries[1] if pos_count > 1 else pos_summaries[0]
        lines.append(f"· 대표 긍정: {left} | {right}")

    # 5) 부정 대표 줄 1개 (최대 2개 기사 요약 사용)
    if neg_count > 0:
        left = neg_summaries[0]
        right = neg_summaries[1] if neg_count > 1 else neg_summaries[0]
        lines.append(f"· 대표 부정: {left} | {right}")

    # 총 줄 수 제한 (예외 안전)
    lines = lines[:5]

    return "\n".join(lines)


# =========================
# NEWS API / Google 뉴스 가져오는 함수
# =========================



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
                })
        except Exception as e:
            print(f"⚠️ Google News RSS 오류(midterm): {e}")

    return articles

def _get_next_earnings_date_yfinance(ticker: str):
    """
    yfinance로부터 다음 실적발표일(가능하면)을 가져온다.
    - 성공 시: 'YYYY-MM-DD' 문자열 반환
    - 실패/데이터 없음: None 반환

    참고: yfinance의 calendar 포맷은 종목/버전에 따라 dict/DF/Series 등으로 변동 가능하므로
    최대한 방어적으로 처리한다.
    """
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import datetime

        t = yf.Ticker(ticker)
        cal = getattr(t, "calendar", None)

        if cal is None:
            return None

        # 1) DataFrame 케이스
        if hasattr(cal, "empty") and cal.empty is False:
            # 보통 index에 항목명이 있고 첫 컬럼에 값이 들어있다.
            # 예: index: ['Earnings Date', ...], values: [[Timestamp(...)]]
            try:
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].values
                    # val이 (1, n) 형태인 경우가 많음
                    # 가능한 첫 번째 Timestamp를 뽑는다
                    candidate = None
                    for x in val.flatten():
                        if x is None:
                            continue
                        candidate = x
                        break
                    if candidate is None:
                        return None
                    # pandas Timestamp / datetime 처리
                    if hasattr(candidate, "to_pydatetime"):
                        candidate = candidate.to_pydatetime()
                    if isinstance(candidate, datetime):
                        return candidate.strftime("%Y-%m-%d")
                    # 문자열이면 앞 10자리로 정규화
                    s = str(candidate).strip()
                    return s[:10] if s else None
            except Exception:
                pass

            # 다른 형태: column에 'Earnings Date'가 있을 수도 있음
            try:
                if "Earnings Date" in cal.columns:
                    candidate = cal["Earnings Date"].iloc[0]
                    if hasattr(candidate, "to_pydatetime"):
                        candidate = candidate.to_pydatetime()
                    if isinstance(candidate, datetime):
                        return candidate.strftime("%Y-%m-%d")
                    s = str(candidate).strip()
                    return s[:10] if s else None
            except Exception:
                pass

        # 2) dict-like 케이스
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date") or cal.get("EarningsDate") or cal.get("earningsDate")
            if ed is None:
                return None

            # ed가 리스트/튜플로 (start, end) 같이 오기도 함
            if isinstance(ed, (list, tuple)) and len(ed) > 0:
                ed = ed[0]

            # Timestamp/datetime
            if hasattr(ed, "to_pydatetime"):
                ed = ed.to_pydatetime()
            if hasattr(ed, "strftime"):
                try:
                    return ed.strftime("%Y-%m-%d")
                except Exception:
                    pass

            s = str(ed).strip()
            return s[:10] if s else None

        # 3) 그 외 (Series 등) -> 문자열 처리 시도
        s = str(cal).strip()
        if not s:
            return None
        # 너무 길면 의미 없으므로 포기
        return None

    except Exception:
        return None


def build_earnings_date_html(ticker: str) -> str:
    """
    뉴스 요약 위에 붙일 실적발표일 HTML 블록 생성.
    데이터 없으면 '업데이트 없음' 표기.
    """
    d = _get_next_earnings_date_yfinance(ticker)
    label = d if d else "업데이트 없음"
    html = (
        "<p style='text-align:left;'>"
        "<strong>실적발표일:</strong> "
        f"{label}"
        "</p>"
    )
    return html


def build_midterm_news_comment_from_apis_combined(ticker, max_items=10, days=14):
    """
    중기 분석 섹션에서 사용할 '최근 2주 뉴스 요약' HTML 생성.

    출력 순서:
    1) yfinance 기반 '실적발표일' (없으면 '업데이트 없음')
    2) 항목별 '실적 분석' (매출/운영소득/순이익 각각 POS/NEG/NEU)
    3) 뉴스 요약 (최근 2주, 주가 영향 이슈)

    추가:
    - 뉴스 요약 라인 색상:
      · 대표 긍정: 초록색
      · 대표 부정: 빨간색
    """

    # 1) 실적발표일
    earnings_html = build_earnings_date_html(ticker)

    # 2) 항목별 실적 분석
    fundamentals = fetch_recent_quarterly_fundamentals_yf(ticker)
    earnings_metric_html = build_earnings_analysis_by_metric_html(ticker, fundamentals)

    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return (
            earnings_html
            + earnings_metric_html
            + "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 2주):</strong><br>"
            "- NEWS_API_KEY가 설정되어 있지 않아 뉴스를 불러올 수 없습니다."
            "</p>"
        )

    # 3) NewsAPI + Google News로 기사 목록 가져오기 (days=14)
    articles = _fetch_news_for_ticker_midterm(
        ticker=ticker,
        api_key=api_key,
        page_size=max_items,
        days=days,
    )

    if not articles:
        return (
            earnings_html
            + earnings_metric_html
            + "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 2주):</strong><br>"
            f"- 최근 {days}일 내 {ticker} 관련 주요 뉴스를 찾지 못했습니다."
            "</p>"
        )

    # 3-1) 실제 최근 14일만 필터링
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(days=14)

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
            # 날짜 파싱 실패는 일단 포함(정보 손실 방지)
            filtered_recent.append(a)
        else:
            if dt >= cutoff:
                filtered_recent.append(a)

    if not filtered_recent:
        return (
            earnings_html
            + earnings_metric_html
            + "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 2주):</strong><br>"
            f"- 최근 14일 내 {ticker} 관련 유효한 날짜의 뉴스를 찾지 못했습니다."
            "</p>"
        )

    articles = filtered_recent

    # 3-2) 티커/회사명 기준 관련 기사 필터
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
        text_all = ((a.get("title") or "") + " " + (a.get("description") or "")).upper()
        if any(k in text_all for k in keywords):
            filtered.append(a)

    use_articles = filtered[:max_items] if len(filtered) >= 3 else articles[:max_items]

    # 3-3) 최신 뉴스 우선 정렬
    def _parse_dt(a):
        p = (a.get("published") or "").strip()
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(p[:len(fmt)], fmt)
            except Exception:
                continue
        return datetime.min

    use_articles = sorted(use_articles, key=_parse_dt, reverse=True)

    # 4) 여러 기사 → 텍스트 요약 (긍정/부정 + 대표 2개씩)
    summary_ko = _summarize_news_bundle_ko_price_focus(ticker, use_articles)

    # 5) 줄 단위로 나눠 색 입히기
    raw_lines = [ln.strip() for ln in summary_ko.splitlines() if ln.strip()]
    colored_lines = []

    for ln in raw_lines:
        if ln.startswith("· 대표 긍정:"):
            colored_lines.append(f"<span style='color:green;'>{_html_escape(ln)}</span>")
        elif ln.startswith("· 대표 부정:"):
            colored_lines.append(f"<span style='color:red;'>{_html_escape(ln)}</span>")
        else:
            colored_lines.append(_html_escape(ln))

    html_body = "<br>".join(colored_lines) if colored_lines else "관련 뉴스를 요약할 수 없습니다."

    news_html = (
        "<p style='text-align:left;'>"
        "<strong>뉴스 요약 (최근 2주, 주가 영향 이슈):</strong><br>"
        f"{html_body}"
        "</p>"
    )

    return earnings_html + earnings_metric_html + news_html


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


def open_gsheet(gs_id, retries=6, delay=5):
    """
    Google Sheets를 open_by_key로 여는 함수.
    - Google API에서 간헐적으로 500/503/502/504/429 등의 일시 오류가 발생할 수 있어 재시도 로직을 포함한다.
    - retries: 총 시도 횟수 (기본 6회)
    - delay: 초기 대기 시간(초) (기본 5초) -> 이후 지수적으로 증가

    기존 코드에서는 503만 재시도했지만, 500도 흔한 일시 장애이므로 포함한다.
    """
    if not gs_id:
        raise EnvironmentError("환경변수 GSHEET_ID 가 설정되어 있지 않습니다.")

    # 지터를 쓰기 위해 random 사용
    import random

    # 재시도할 HTTP status code 집합
    # 429: Too Many Requests (rate limit)
    # 500: Internal error
    # 502/503/504: Bad gateway / Service unavailable / Gateway timeout
    retryable_codes = {"429", "500", "502", "503", "504"}

    last_exc = None
    for i in range(retries):
        try:
            # 매번 client를 새로 생성 (토큰/세션 이슈 완화 목적)
            client = get_gspread_client()
            return client.open_by_key(gs_id)

        except gspread.exceptions.APIError as e:
            last_exc = e
            msg = str(e)

            # 에러 문자열에 포함된 status code를 단순 탐지
            is_retryable = any(code in msg for code in retryable_codes)

            # 재시도 불가이거나 마지막 시도면 그대로 raise
            if (not is_retryable) or (i >= retries - 1):
                raise

            # 지수 백오프 + 지터 (동시 실행 시 재충돌 방지)
            # 예: delay=5 -> 5, 10, 20, 40...
            backoff = delay * (2 ** i)
            jitter = random.uniform(0, 1.0)  # 0~1초
            sleep_s = backoff + jitter

            # 로그
            # 예: APIError: [500]: Internal error encountered.
            code_hint = None
            for code in retryable_codes:
                if code in msg:
                    code_hint = code
                    break
            code_hint = code_hint or "UNKNOWN"

            print(
                f"⚠️ Google API {code_hint} 오류 발생, {sleep_s:.1f}초 후 재시도... "
                f"({i + 1}/{retries})"
            )
            time.sleep(sleep_s)
            continue

        except Exception as e:
            # 네트워크 타임아웃 등 gspread APIError 외 예외도 간헐적일 수 있어 제한적으로 재시도
            last_exc = e

            if i >= retries - 1:
                raise

            backoff = delay * (2 ** i)
            sleep_s = backoff + 0.5
            print(
                f"⚠️ Google Sheets 연결 중 예외({type(e).__name__}) 발생, {sleep_s:.1f}초 후 재시도... "
                f"({i + 1}/{retries})"
            )
            time.sleep(sleep_s)
            continue

    # 논리적으로 여기 오면 안 되지만, 안전장치
    if last_exc:
        raise last_exc
    raise RuntimeError("open_gsheet 실패: 알 수 없는 이유로 시트 오픈에 실패했습니다.")



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
    중단기(6~12개월) 투자 분석용 함수.

    구성 요소:
    1) 가격 기반 수치:
       - UpProb       : 중기 상승 확률 %
       - BuyTiming    : 매수 타이밍 %
       - SellTiming   : 매도 타이밍 %
       - TargetRange  : 1년 목표수익 범위 (ex: "12~25%")

    2) 뉴스 기반 분석:
       - NewsAPI + Google News RSS로 최근 기사 최대 10개 가져옴
       - 관련 기사만 필터링
       - 여러 기사를 통합해 한국어 20줄 내외 bullet 종합 분석 생성
       - build_midterm_news_comment_from_apis_combined() 사용

    반환 형식:
    {
        "Ticker": str,
        "UpProb": float,
        "BuyTiming": float,
        "SellTiming": float,
        "TargetRange": str,
        "Comment": HTML 문자열
    }
    """

    # -----------------------------
    # 1. 가격 데이터 (1년)
    # -----------------------------
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")["Close"].dropna()

        if hist.empty or len(hist) < 40:
            raise ValueError("데이터 부족")

        closes = hist.copy()
        last = float(closes.iloc[-1])
    except Exception as e:
        print(f"[WARN] analyze_midterm_ticker({ticker}) 가격 데이터 오류: {e}")
        comment_html = build_midterm_news_comment_from_apis_combined(ticker)
        return {
            "Ticker": ticker,
            "UpProb": None,
            "BuyTiming": None,
            "SellTiming": None,
            "TargetRange": "데이터 부족",
            "Comment": comment_html,
        }

    # -----------------------------
    # 2. 수익률/변동성
    # -----------------------------
    # 1년 수익률
    start_1y = float(closes.iloc[-252]) if len(closes) > 252 else float(closes.iloc[0])
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
    vol_pct = vol_annual * 100.0

    # -----------------------------
    # 3. 투자 신호 계산 (휴리스틱)
    # -----------------------------
    # 상승 확률 점수
    score = 50.0
    score += float(np.tanh(ret_1y / 40.0)) * 25.0
    score += float(np.tanh(ret_3m / 20.0)) * 20.0
    score -= float(np.tanh(vol_annual * 2.0)) * 15.0

    up_prob = max(5.0, min(95.0, score))

    # 1년 고가/저가 기준 포지션
    hi_1y = float(closes.max())
    lo_1y = float(closes.min())
    if hi_1y > lo_1y:
        pos = (last - lo_1y) / (hi_1y - lo_1y)
    else:
        pos = 0.5

    # 매수/매도 타이밍
    buy_timing = max(5.0, min(95.0, (1.0 - pos) * 100.0))
    sell_timing = max(5.0, min(95.0, pos * 100.0))

    # 1년 목표수익 범위
    base = (up_prob - 50.0) / 50.0  # -1 ~ +1 정도
    low_pct = 10.0 + base * 15.0
    high_pct = 10.0 + base * 35.0
    high_pct += min(10.0, vol_pct * 0.1)  # 변동성 반영

    low_pct = max(-20.0, min(25.0, low_pct))
    high_pct = max(low_pct + 5.0, min(60.0, high_pct))

    target_range = f"{low_pct:.0f}~{high_pct:.0f}%"

    # -----------------------------
    # 4. 뉴스 종합 분석 (한국어 20줄 내외)
    # -----------------------------
    comment_html = build_midterm_news_comment_from_apis_combined(ticker)

    # -----------------------------
    # 5. 반환
    # -----------------------------
    return {
        "Ticker": ticker,
        "UpProb": round(up_prob, 1),
        "BuyTiming": round(buy_timing, 1),
        "SellTiming": round(sell_timing, 1),
        "TargetRange": target_range,
        "Comment": comment_html,
    }


    

def build_midterm_analysis_html(df_enriched):
    """
    📈 중단기 투자의 통합 분석 (TFSA 종목만, SCHD 제외)

    1) 요약표 : Ticker + 중기 상승 확률 % / 매수 타이밍 % / 매도 타이밍 % / 1년 목표수익 범위
    2) 상세표 : '핵심 투자 코멘트' (실적발표일 → 실적분석(GPT) → 뉴스요약)
       ※ '주요맥락' 컬럼 제거
    """

    # 0) 방어 코드: 필요한 컬럼 확인
    if "Ticker" not in df_enriched.columns or "Type" not in df_enriched.columns:
        return "<p>Type/Ticker 컬럼이 없어 중단기 분석을 생성할 수 없습니다.</p>"

    # 1) TFSA 계좌의 Ticker만 추출
    tfsa_tickers = (
        df_enriched[df_enriched["Type"].astype(str).str.upper() == "TFSA"]["Ticker"]
        .astype(str)
        .dropna()
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    # 2) SCHD 제외
    tickers = [t for t in tfsa_tickers if t.upper() != "SCHD"]

    if not tickers:
        return "<p>TFSA 중단기 대상 종목이 없습니다.</p>"

    rows_summary = []
    rows_detail = []

    # 3) 각 종목별 중단기 분석
    for t in sorted(tickers):
        try:
            stat = analyze_midterm_ticker(t)
        except Exception as e:
            print(f"[WARN] analyze_midterm_ticker 실패: {t}, {e}")
            continue

        # ① 요약 테이블 행 (퍼센트 색칠 적용)
        if stat["UpProb"] is not None:
            up_str = colorize_midterm_metric("UpProb", stat["UpProb"])
            buy_str = colorize_midterm_metric("BuyTiming", stat["BuyTiming"])
            sell_str = colorize_midterm_metric("SellTiming", stat["SellTiming"])
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

        # ② 상세 테이블 행 (주요맥락 제거)
        rows_detail.append(
            {
                "Ticker": stat["Ticker"],
                "핵심 투자 코멘트": stat["Comment"],
            }
        )

    # 4) DataFrame → HTML
    df_sum = pd.DataFrame(rows_summary)
    df_det = pd.DataFrame(rows_detail)

    html_summary = df_sum.to_html(index=False, escape=False)
    html_detail = df_det.to_html(index=False, escape=False)

    return (
        "<h3>① 요약 테이블</h3>"
        + html_summary
        + "<br/><br/>"
        + "<h3>② 상세 테이블 (핵심 투자 코멘트)</h3>"
        + html_detail
    )




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

def build_schd_dividend_summary_text(df_enriched):
    """
    SCHD 장기 배당 분석 요약 (yfinance 기반, 현재 동작 방식 유지 + TFSA 원천징수 15% 반영)

    핵심:
    - 연배당/주(USD): yfinance 배당 지급 내역을 연도별 합산 후, '현재 연도 제외'한 마지막 완료 연도 사용
    - 배당 성장률(CAGR): 최근 완료 연도 기반 CAGR(기본 5년 창, -10%~+15% 클리핑)
    - 배당수익률: 완료 연도 연배당/주 ÷ 현재가(USD) (세전 근사)
    - 원천징수(기본 15%): 계좌(Type)별로 적용하여 '세후(Net) 연배당(CAD)' 계산 및 목표 기간 계산에 반영
      * RRSP: 0%
      * TFSA/RESP/기타: 15% (보수적 기본)
    - 목표(월 CAD 1,000): 세후(Net) 배당 기준으로 추정
    - 투자원금 배당률(YOC): 세후(Net) 연배당(CAD) ÷ 총 투자원금(CAD)

    출력 템플릿(줄 줄인 버전):
    현재 예상 연 배당금(CAD): C$... (보유 ...주 기준)
    월 CAD 1,000 배당 달성 예상: 약 X년 Y개월
    (원천징수 반영 / DRIP + 매월 200 USD(환전 후 투자) / 배당 성장률 CAGR a% / 배당수익률 b%(세전) / c%(세후) / 투자원금 배당률: d%(세후))
    """

    import numpy as np
    import pandas as pd
    from datetime import datetime

    try:
        import yfinance as yf
    except Exception:
        yf = None

    # -----------------------------
    # 0) 입력 방어
    # -----------------------------
    if df_enriched is None or getattr(df_enriched, "empty", True):
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> 데이터 부족</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    sub = df_enriched.copy()
    if "Ticker" not in sub.columns:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> df_enriched에 Ticker 컬럼이 없습니다</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    # -----------------------------
    # 1) df_enriched에서 SCHD 보유/원금(CAD) 자동 산출
    # -----------------------------
    sub["TickerU"] = sub["Ticker"].astype(str).str.upper().str.strip()
    schd_df = sub[sub["TickerU"] == "SCHD"].copy()

    if schd_df.empty:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> N/A (SCHD 보유 없음)</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    # 필요한 컬럼 방어
    for col in ["Shares", "AvgPrice"]:
        if col not in schd_df.columns:
            return (
                f"<p><strong>현재 예상 연 배당금(CAD):</strong> SCHD 계산에 필요한 컬럼 누락: {col}</p>"
                "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
            )

    schd_df["SharesNum"] = pd.to_numeric(schd_df["Shares"], errors="coerce").fillna(0.0)
    schd_df["AvgPriceNum"] = pd.to_numeric(schd_df["AvgPrice"], errors="coerce").fillna(0.0)

    schd_shares_total = float(schd_df["SharesNum"].sum())
    if schd_shares_total <= 0:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> N/A (SCHD 보유 없음)</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    # 환율(USD→CAD): CAD=X 사용 (기존 스타일)
    if yf is None:
        usd_to_cad = 1.35
    else:
        try:
            fx = yf.Ticker("CAD=X").history(period="5d")["Close"].dropna()
            usd_to_cad = float(fx.iloc[-1]) if not fx.empty else 1.35
        except Exception:
            usd_to_cad = 1.35

    # Type 정규화
    if "Type" in schd_df.columns:
        schd_df["TypeU"] = schd_df["Type"].astype(str).str.upper().str.strip()
    else:
        # Type이 없으면 보수적으로 TFSA로 가정(원천징수 15% 적용)
        schd_df["TypeU"] = "TFSA"

    # 총 투자원금(CAD) 계산
    # - TFSA/RRSP(USD로 가정): Shares * AvgPrice(USD) * usd_to_cad
    # - RESP(CAD로 가정): Shares * AvgPrice(CAD)
    schd_df["CostNative"] = schd_df["SharesNum"] * schd_df["AvgPriceNum"]
    schd_df["CostCAD"] = np.where(
        schd_df["TypeU"] == "RESP",
        schd_df["CostNative"],
        schd_df["CostNative"] * usd_to_cad
    )
    total_cost_cad = float(schd_df["CostCAD"].sum())
    if total_cost_cad < 0:
        total_cost_cad = 0.0

    # -----------------------------
    # 2) 원천징수율(Effective) 계산: 계좌별 주식수 가중 평균
    # -----------------------------
    # 가정(보수적):
    # - RRSP: 0% (미국 배당 원천징수 면제 취급)
    # - TFSA/RESP/기타: 15%
    def _withholding_rate_by_type(t: str) -> float:
        t = (t or "").upper().strip()
        if t == "RRSP":
            return 0.0
        # TFSA, RESP, 기타 모두 15%로 처리 (보수적, 일관성 목적)
        return 0.15

    schd_df["WHTRate"] = schd_df["TypeU"].apply(_withholding_rate_by_type)
    # shares 가중 평균 원천징수율
    eff_wht = float(
        (schd_df["SharesNum"] * schd_df["WHTRate"]).sum() / schd_shares_total
    )
    eff_wht = max(0.0, min(0.30, eff_wht))  # 방어

    # -----------------------------
    # 3) 배당 데이터(yfinance) → 완료 연도 연배당/주(USD)
    # -----------------------------
    if yf is None:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> yfinance 로드 실패</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    today = datetime.today()
    current_year = today.year

    try:
        tk = yf.Ticker("SCHD")
        divs = tk.dividends.dropna()
    except Exception:
        divs = pd.Series(dtype=float)

    if divs is None or divs.empty:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> 배당 데이터 부족</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    div_by_year = divs.groupby(divs.index.year).sum()
    years_done = sorted([int(y) for y in div_by_year.index if int(y) < current_year])

    if not years_done:
        years_all = sorted([int(y) for y in div_by_year.index])
        if not years_all:
            return (
                "<p><strong>현재 예상 연 배당금(CAD):</strong> 연도별 배당 데이터 부족</p>"
                "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
            )
        last_done_year = years_all[-1]
    else:
        last_done_year = years_done[-1]

    last_div_ps_usd = safe_float(div_by_year.get(last_done_year, 0.0), 0.0)
    if last_div_ps_usd <= 0:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> 완료 연도 연배당/주(USD) 계산 실패</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    # -----------------------------
    # 4) 현재가(USD) 및 배당수익률(세전/세후)
    # -----------------------------
    try:
        hist_px = yf.Ticker("SCHD").history(period="5d")["Close"].dropna()
        price_usd = float(hist_px.iloc[-1]) if not hist_px.empty else 0.0
    except Exception:
        price_usd = 0.0

    # 세전 배당수익률(근사)
    y_gross = (last_div_ps_usd / price_usd) if price_usd > 0 else 0.035
    if y_gross <= 0:
        y_gross = 0.035

    # 세후 배당수익률(근사): 원천징수 반영
    y_net = y_gross * (1.0 - eff_wht)

    # -----------------------------
    # 5) 배당 성장률 CAGR (최근 완료 연도 기반, 세전/세후 동일한 성장률로 취급)
    # -----------------------------
    g_fallback = 0.11
    g = g_fallback

    if len(years_done) >= 2:
        N = 5
        use_years = years_done[-N:] if len(years_done) >= N else years_done
        y0 = use_years[0]
        yN = use_years[-1]
        d0 = safe_float(div_by_year.get(y0, 0.0), 0.0)
        dN = safe_float(div_by_year.get(yN, 0.0), 0.0)
        n = int(yN - y0)
        if d0 > 0 and dN > 0 and n > 0:
            g = (dN / d0) ** (1.0 / n) - 1.0

    g = max(-0.10, min(0.15, safe_float(g, g_fallback)))

    # -----------------------------
    # 6) 현재 연 배당금(CAD): 세전/세후
    # -----------------------------
    annual_div_cad_gross = schd_shares_total * last_div_ps_usd * usd_to_cad
    annual_div_cad_net = annual_div_cad_gross * (1.0 - eff_wht)

    # -----------------------------
    # 7) 투자원금 배당률(YOC): 세후 기준
    # -----------------------------
    if total_cost_cad > 0:
        yoc_net = annual_div_cad_net / total_cost_cad
    else:
        yoc_net = 0.0

    # -----------------------------
    # 8) 목표 도달 기간: 세후(Net) 배당 기준으로 추정
    # -----------------------------
    # - 시작 연배당: annual_div_cad_net
    # - 기여금: 매월 200 USD 환전 후 투자(가정)
    # - 모델 내 yield도 세후 수익률(y_net) 사용 (목표는 '실수령 월 1,000 CAD'로 해석)
    monthly_usd = 200.0
    monthly_cad = monthly_usd * usd_to_cad
    annual_contrib_cad = monthly_cad * 12.0
    target_annual_cad = 12_000.0

    if g <= 0:
        annual_cad_str = fmt_money(annual_div_cad_net, "C$")
        g_str = fmt_pct(g * 100.0)
        yg_str = fmt_pct(y_gross * 100.0)
        yn_str = fmt_pct(y_net * 100.0)
        yoc_str = fmt_pct(yoc_net * 100.0)
        wht_str = fmt_pct(eff_wht * 100.0)

        return (
            "<p style='text-align:left;'>"
            f"<strong>현재 예상 연 배당금(CAD):</strong> {annual_cad_str} (보유 {schd_shares_total:,.0f}주 기준)<br>"
            "<strong>월 CAD 1,000 배당 달성 예상:</strong> g<=0 구간은 단순 모델로 추정 불안정<br>"
            f"(원천징수 {wht_str} 반영 / DRIP + 매월 200 USD(환전 후 투자) / "
            f"배당 성장률 CAGR {g_str} / 배당수익률 {yg_str}(세전) / {yn_str}(세후) / 투자원금 배당률: {yoc_str}(세후))"
            "</p>"
        )

    # A = 연간기여금 × (수익률 / 성장률)  (세후 y_net 사용)
    A = annual_contrib_cad * (y_net / g)

    numerator = target_annual_cad + A
    denominator = annual_div_cad_net + A

    if numerator <= denominator:
        n_years = 0.0
    else:
        n_years = np.log(numerator / denominator) / np.log(1.0 + g)

    n_years = max(0.0, float(n_years))
    years_int = int(n_years)
    months_int = int(round((n_years - years_int) * 12.0))
    if months_int == 12:
        years_int += 1
        months_int = 0

    # -----------------------------
    # 9) 출력(요청 템플릿): 세후(Net) 연배당 표시
    # -----------------------------
    annual_cad_str = fmt_money(annual_div_cad_net, "C$")
    g_str = fmt_pct(g * 100.0)
    yg_str = fmt_pct(y_gross * 100.0)
    yn_str = fmt_pct(y_net * 100.0)
    yoc_str = fmt_pct(yoc_net * 100.0)
    wht_str = fmt_pct(eff_wht * 100.0)

    return (
        "<p style='text-align:left;'>"
        f"<strong>현재 예상 연 배당금(CAD):</strong> {annual_cad_str} (보유 {schd_shares_total:,.0f}주 기준)<br>"
        f"<strong>월 CAD 1,000 배당 달성 예상:</strong> 약 {years_int}년 {months_int}개월<br>"
        f"(원천징수 {wht_str} 반영 / DRIP + 매월 200 USD(환전 후 투자) / "
        f"배당 성장률 CAGR {g_str} / 배당수익률 {yg_str}(세전) / {yn_str}(세후) / 투자원금 배당률: {yoc_str}(세후))"
        "</p>"
    )

    

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
    def make_holdings_table(acc_type, tickers_include=None, add_summary_row=False):
        sub = df_enriched[df_enriched["Type"].str.upper() == acc_type].copy()

        # (NEW) 전략/그룹 분리를 위한 티커 필터
        if tickers_include is not None:
            inc = set([str(t).strip().upper() for t in tickers_include])
            sub = sub[sub["Ticker"].astype(str).str.upper().isin(inc)].copy()
            if sub.empty:
                return "<p>해당 그룹에 포함된 종목이 없습니다.</p>"

        if sub.empty:
            return f"<p>No holdings for {acc_type}.</p>"

        # ✅ (추가) TFSA/RESP 공통 "오늘의 Profit/Loss" = (LastPrice - PrevClose) * Shares
        # - 전일 대비 '일간 손익'을 의미
        shares_num = pd.to_numeric(sub["Shares"], errors="coerce").fillna(0.0)
        last_num = pd.to_numeric(sub["LastPrice"], errors="coerce").fillna(0.0)
        prev_num = pd.to_numeric(sub["PrevClose"], errors="coerce").fillna(0.0)

        today_pl_native = (last_num - prev_num) * shares_num  # 계좌 통화(nativ) 기준
        today_pl_fmt = []
        for v in today_pl_native.tolist():
            v_num = safe_float(v, 0.0)
            text = fmt_money(v_num, ccy_symbol)
            today_pl_fmt.append(colorize_value_html(text, v_num))
        sub["TodayPLNativeFmt"] = today_pl_fmt

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
        sub_raw_pl_native = sub["ProfitLossNative"].copy()
        sub_raw_avg_price = sub["AvgPrice"].copy()
        sub_raw_shares = sub["Shares"].copy()

        raw_pl_native = sub_raw_pl_native.tolist()
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

        # 컬럼 구성: TFSA/RESP 모두 Today's P/L 추가
        cols = [
            "Ticker",
            "Shares",
            "AvgPrice",
            "LastPriceNativeFmt",
            "PositionValueNativeFmt",
            "TodayPLNativeFmt",          # ✅ TFSA/RESP 공통 추가
            "ProfitLossNativeFmt",
            "ProfitLossPctFmt",
        ]
        rename_map = {
            "LastPriceNativeFmt": "LastPrice",
            "PositionValueNativeFmt": "PositionValue",
            "TodayPLNativeFmt": "Today's P/L",   # ✅ 표 헤더
            "ProfitLossNativeFmt": "Profit/Loss",
            "ProfitLossPctFmt": "Profit/Loss %",
        }

        sub = sub[cols].rename(columns=rename_map)

        # =========================
        # (옵션) 그룹 합계(SUM) 행 추가
        # - Profit/Loss: 합계
        # - Profit/Loss %: (합계 P/L) / (합계 Cost Basis)  ← 가중 평균 수익률
        # - Today's P/L: 합계
        # =========================
        if add_summary_row and (tickers_include is not None) and (len(sub) > 0):
            try:
                total_today_pl = float(today_pl_native.sum())
            except Exception:
                total_today_pl = 0.0

            try:
                pl_native_num = pd.to_numeric(sub_raw_pl_native, errors="coerce").fillna(0.0)
                total_pl = float(pl_native_num.sum())
            except Exception:
                total_pl = 0.0

            try:
                avg_num = pd.to_numeric(sub_raw_avg_price, errors="coerce").fillna(0.0)
                sh_num = pd.to_numeric(sub_raw_shares, errors="coerce").fillna(0.0)
                total_cost = float((avg_num * sh_num).sum())
            except Exception:
                total_cost = 0.0

            total_pl_pct = (total_pl / total_cost) if total_cost != 0 else 0.0

            sum_row = {
                "Ticker": "<strong>SUM</strong>",
                "Shares": "—",
                "AvgPrice": "—",
                "LastPrice": "—",
                "PositionValue": "—",
                "Today's P/L": colorize_value_html(fmt_money(total_today_pl, ccy_symbol), total_today_pl),
                "Profit/Loss": colorize_value_html(fmt_money(total_pl, ccy_symbol), total_pl),
                "Profit/Loss %": colorize_value_html(fmt_pct(total_pl_pct), total_pl_pct),
            }
            sub = pd.concat([sub, pd.DataFrame([sum_row])], ignore_index=True)

        return sub.to_html(index=False, escape=False)

    # TFSA 전략 분리: SCHD(배당/인컴) vs 나머지(성장/모멘텀)
    tfsa_all = df_enriched[df_enriched["Type"].astype(str).str.upper() == "TFSA"].copy()

    if tfsa_all.empty:
        tfsa_dividend_table = "<p>No holdings for TFSA.</p>"
        tfsa_growth_table = "<p>No holdings for TFSA.</p>"
    else:
        tfsa_tickers = (
            tfsa_all["Ticker"].astype(str).str.upper().str.strip().replace("", np.nan).dropna().unique().tolist()
        )
        dividend_tickers = ["SCHD"]
        dividend_set = set(dividend_tickers)
        growth_tickers = [t for t in tfsa_tickers if t not in dividend_set]

        tfsa_dividend_table = make_holdings_table("TFSA", tickers_include=dividend_tickers)
        tfsa_growth_table = make_holdings_table("TFSA", tickers_include=growth_tickers, add_summary_row=True)

    resp_table = make_holdings_table("RESP")

    # ---------- 3) 중단기 투자 분석 (전체 보유 종목) ----------
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

    schd_summary_text = build_schd_dividend_summary_text(df_enriched)

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

          <h3>💰 배당(인컴) 전략</h3>
          {tfsa_dividend_table}

          <h3>🚀 성장/모멘텀(중단기) 전략</h3>
          {tfsa_growth_table}
        </div>

        <div class="section">
          <h2>🎓 RESP Holdings (in CAD)</h2>
          {resp_table}
        </div>

        <div class="section">
          <h2>📈 중단기 투자의 통합 분석 (전체 보유 종목)</h2>
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

    # 2) 기존 리포트 생성 로직
    main()
