import os
import re
import time
import smtplib
import requests
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
    각 그룹에서 대표 요약 문자열을 생성한다.

    규칙:
    - 긍정/부정 각각:
      · pos_count / neg_count: 해당 감성으로 분류된 기사 개수
      · pos_repr / neg_repr:
          - 기사가 0개이면 None
          - 기사가 1개이면 '요약1' (단일 문장, '|' 없음)
          - 기사가 2개 이상이면 '요약1 | 요약2'

    입력:
        ticker   : 종목 티커 (예: "TSLA")
        articles : _fetch_news_for_ticker_midterm 결과 리스트
                   각 원소는 {title, description, source, published} 형태 가정

    반환:
        {
          "pos_count": int,
          "neg_count": int,
          "pos_repr": str or None,
          "neg_repr": str or None,
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

    # 1) 기사들을 하나의 텍스트로 묶어 번호를 붙여 전달
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

    # 2) 분류 결과를 바탕으로 긍정/부정 인덱스 집계
    pos_idx = set()
    neg_idx = set()

    for x in items_sent:
        try:
            idx = int(x.get("index"))
            sent = (x.get("sentiment") or "").strip()
        except Exception:
            continue
        if not (1 <= idx <= len(articles)):
            continue
        if sent == "긍정":
            pos_idx.add(idx)
        elif sent == "부정":
            neg_idx.add(idx)
        # "중립"은 카운트하지 않음

    pos_count = len(pos_idx)
    neg_count = len(neg_idx)

    # 3) 대표 요약 문자열 생성 로직
    def _build_repr(idx_set):
        """
        idx_set(기사 인덱스 집합)에서 대표 요약 문자열 생성:

        - len == 0 → None
        - len == 1 → 기사 1개를 요약한 단일 문자열
        - len >= 2 → 첫 두 개 기사를 요약하여 '요약1 | 요약2'
        """
        if not idx_set:
            return None

        sorted_idx = sorted(idx_set)

        # 기사 1개인 경우: 단일 요약만
        if len(sorted_idx) == 1:
            rep_i = sorted_idx[0]
            art = articles[rep_i - 1]
            text = ((art.get("title") or "") + "\n" + (art.get("description") or "")).strip()
            summary = _short_ko_summary_15(text)
            return summary

        # 기사 2개 이상: 앞에서 2개만 사용
        rep_indices = sorted_idx[:2]
        summaries = []
        for rep_i in rep_indices:
            art = articles[rep_i - 1]
            text = ((art.get("title") or "") + "\n" + (art.get("description") or "")).strip()
            summaries.append(_short_ko_summary_15(text))

        # 비어있는 요약 제거
        summaries = [s for s in summaries if s]
        if not summaries:
            return None
        if len(summaries) == 1:
            # 이 경우는 거의 없겠지만 방어적으로 단일 요약만 리턴
            return summaries[0]
        return f"{summaries[0]} | {summaries[1]}"

    pos_repr = _build_repr(pos_idx)
    neg_repr = _build_repr(neg_idx)

    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_repr": pos_repr,
        "neg_repr": neg_repr,
    }


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
# NewsAPI/RSS 기반 종목 뉴스 → 주가 영향 중심 요약(30줄) → HTML 생성 함수
# =========================

def build_midterm_news_comment_from_apis_combined(ticker, max_items=10, days=30):
    """
    중기 분석 섹션에서 사용할 '최근 1개월 뉴스 요약' HTML 생성.

    - 최근 30일 뉴스만 사용
    - NewsAPI → 실패 시 Google News RSS
    - 최대 max_items개 기사 사용
    - 티커/회사명 필터
    - _summarize_news_bundle_ko_price_focus()를 호출해
      "긍정/부정 개수 + 대표 2개씩 (맥락 | 키워드)" 텍스트를 생성
    - HTML로 변환할 때:
      · 긍정 라인은 초록색
      · 부정 라인은 빨간색
      · 전체 왼쪽 정렬
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약 (최근 1개월):</strong><br>"
            "- NEWS_API_KEY가 설정되어 있지 않아 뉴스를 불러올 수 없습니다."
            "</p>"
        )

    # 1) NewsAPI + Google News로 기사 목록 가져오기
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

    # 1-1) 실제 최근 30일만 필터링
    from datetime import datetime, timedelta
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

    # 2) 티커/회사명 기준 관련 기사 필터
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

    if len(filtered) >= 3:
        use_articles = filtered[:max_items]
    else:
        use_articles = articles[:max_items]

    # 2-1) 최신 뉴스 우선 정렬
    def _parse_dt(a):
        p = (a.get("published") or "").strip()
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(p[:len(fmt)], fmt)
            except Exception:
                continue
        return datetime.min

    use_articles = sorted(use_articles, key=_parse_dt, reverse=True)

    # 3) 여러 기사 → 텍스트 요약 (긍정/부정 + 대표 2개씩)
    summary_ko = _summarize_news_bundle_ko_price_focus(ticker, use_articles)

    # 4) 줄 단위로 나눠 색 입히기
    raw_lines = [ln.strip() for ln in summary_ko.splitlines() if ln.strip()]
    colored_lines = []

    for ln in raw_lines:
        if ln.startswith("· 대표 긍정:"):
            colored_lines.append(
                f"<span style='color:green;'>{ln}</span>"
            )
        elif ln.startswith("· 대표 부정:"):
            colored_lines.append(
                f"<span style='color:red;'>{ln}</span>"
            )
        else:
            # 첫 줄 "긍정 뉴스 X건, 부정 뉴스 Y건" 등은 기본 색
            colored_lines.append(ln)

    html_body = "<br>".join(colored_lines) if colored_lines else "관련 뉴스를 요약할 수 없습니다."

    html = (
        "<p style='text-align:left;'>"
        "<strong>뉴스 요약 (최근 1개월, 주가 영향 이슈):</strong><br>"
        f"{html_body}"
        "</p>"
    )
    return html


# =========================
# NEWS API / Google 뉴스 가져오는 함수
# =========================

def _summarize_news_ko_15(text):
    """
    뉴스 제목/본문을 받아 한국어 15자 이내로 요약.

    - OPENAI_API_KEY 없으면: '뉴스요약불가' 반환
    - 예외 발생 시: '뉴스요약실패' 반환
    """
    text = (text or "").strip()
    if not text:
        return "뉴스요약불가"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "뉴스요약불가"

    try:
        # 지연 import (상단에 이미 있다면 제거해도 됨)
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "다음 영어 뉴스의 핵심 투자 포인트를 "
            "한국어로 15자 이내로 아주 짧게 요약해줘.\n"
            "문장 1개만, 부호 최소화:\n\n"
            f"{text}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        summary = resp.choices[0].message.content.strip()
        # 혹시 길게 나와도 15자로 강제 자르기
        summary = summary.replace("\n", " ").strip()
        return summary[:15] if summary else "뉴스요약실패"
    except Exception:
        return "뉴스요약실패"


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
    - 긍정/부정 각각 대표 뉴스 1개를 뽑아 15자 내외 한글 요약
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

    # 1-1) 추가적으로 '실제 날짜 기준'으로 최근 30일만 필터링 (RSS 대비 방어용)
    cutoff = datetime.utcnow() - timedelta(days=30)
    filtered_recent = []
    for a in articles:
        # _fetch_news_for_ticker_midterm에서 "published" 필드를 넣었다고 가정
        p = (a.get("published") or "").strip()
        dt = None
        # 간단한 파싱: ISO 또는 YYYY-MM-DD 형태 위주
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(p[:len(fmt)], fmt)
                break
            except Exception:
                continue
        if dt is None:
            # 날짜 파싱 실패 시 일단 포함 (너무 보수적으로 버리지 않기 위해)
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

    # 3) 긍정/부정 분류 및 대표 뉴스 선별 + 15자 요약
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



def _extract_article_date_midterm(article):
    """
    뉴스 dict에서 날짜를 안전하게 추출 (NewsAPI / RSS 공용)
    """
    from datetime import datetime

    date_raw = (
        article.get("publishedAt")
        or article.get("pubDate")
        or article.get("date")
        or article.get("published")
        or ""
    )
    if not date_raw:
        return "N/A"

    # ISO8601 시도 (NewsAPI)
    try:
        dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # RSS 등 기타 포맷: 앞 10자리만
        return date_raw[:10]


def build_midterm_news_comment_from_apis(ticker, max_items=2, days=7):
    """
    NVDA/TSLA 중기 분석 섹션에서 사용할 뉴스 요약 HTML 생성.

    - 소스: NewsAPI → 실패 시 Google News RSS
    - 최대 max_items개
    - 각 뉴스는 한국어 15자 이내 요약
    - 반환: HTML <p> 블록
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약:</strong><br>"
            "- NEWS_API_KEY가 설정되어 있지 않아 뉴스를 불러올 수 없습니다."
            "</p>"
        )

    articles = _fetch_news_for_ticker_midterm(
        ticker=ticker,
        api_key=api_key,
        page_size=max_items,
        days=days,
    )

    if not articles:
        return (
            "<p style='text-align:left;'>"
            "<strong>뉴스 요약:</strong><br>"
            f"- 최근 {days}일 내 주요 뉴스를 찾지 못했습니다."
            "</p>"
        )

    lines = []
    for a in articles[:max_items]:
        title = (a.get("title") or "").strip()
        desc = (a.get("description") or "").strip()
        base_text = (title + "\n" + desc).strip()

        summary_ko = _summarize_news_ko_15(base_text)
        date_str = _extract_article_date_midterm(a)
        src = (a.get("source") or "").strip()

        if src:
            line = f"- {date_str} {src}: {summary_ko}"
        else:
            line = f"- {date_str}: {summary_ko}"

        lines.append(line)

    html = "<p style='text-align:left;'><strong>뉴스 요약:</strong><br>"
    html += "<br>".join(lines)
    html += "</p>"
    return html



# =========================
# =========================
# Congress Trading (Nancy Pelosi)
# =========================

def _safe_requests_get_json(url, params=None, headers=None, timeout=25, retries=3, backoff=2.0, return_meta=False):
    """간단한 GET+JSON 헬퍼. 네트워크/일시 오류 시 재시도.

    - 기본(return_meta=False): 기존과 동일하게 json(or None)만 반환
    - return_meta=True: (json_or_none, meta_dict) 반환
      meta_dict 예: {"url": "...", "status_code": 200, "error": None}
    """
    import requests

    last_exc = None
    last_status = None
    last_err = None

    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            last_status = getattr(r, "status_code", None)

            # 429는 잠깐 쉬었다가 재시도(가능하면 Retry-After 반영)
            if last_status == 429 and i < retries - 1:
                retry_after = r.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else (backoff ** i)
                except Exception:
                    wait = backoff ** i
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()
            meta = {"url": r.url, "status_code": last_status, "error": None}
            return (data, meta) if return_meta else data
        except Exception as e:
            last_exc = e
            last_err = str(e)
            # 마지막이면 None 반환 (리포트 전체 실패 방지)
            if i >= retries - 1:
                meta = {"url": url, "status_code": last_status, "error": last_err}
                return (None, meta) if return_meta else None
            time.sleep(backoff ** i)

    meta = {"url": url, "status_code": last_status, "error": str(last_exc) if last_exc else "unknown"}
    return (None, meta) if return_meta else None

def _redact_apikey_in_text(text):
    """로그/진단 출력에서 apikey 노출을 방지한다."""
    if not text:
        return text
    try:
        return re.sub(r'([?&]apikey=)[^&\s]+', r'\1REDACTED', str(text), flags=re.IGNORECASE)
    except Exception:
        return str(text)



def fetch_house_stockwatcher_trades():
    """HouseStockWatcher 공개 데이터 소스를 시도한다(키 불필요).

    1) https://housestockwatcher.com/api (가끔 502/다운 가능)
    2) S3 미러 JSON (대용량일 수 있음):
       https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json

    반환: (data_or_none, diag_list)
      diag_list 예:
        [{"source":"hsw_primary","url": "...", "status_code": 200, "error": None}, ...]
    """
    diag = []

    primary = "https://housestockwatcher.com/api"
    data, meta = _safe_requests_get_json(primary, return_meta=True)
    meta.update({"source": "hsw_primary"})
    diag.append(meta)
    if data:
        return data, diag

    # 미러 소스 fallback (대용량 가능 → timeout을 조금 늘림)
    mirror = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
    data2, meta2 = _safe_requests_get_json(mirror, timeout=40, retries=2, return_meta=True)
    meta2.update({"source": "hsw_mirror"})
    diag.append(meta2)
    return data2, diag


def fetch_quiver_recent_congress_trades():
    """QuiverQuant 'Congress Trading' 대시보드(HTML)에서 최근 거래 테이블을 파싱한다.

    주의:
      - 이 함수는 공개 웹페이지의 HTML 테이블을 파싱하는 방식(스크래핑)이다.
      - 사이트 구조/정책 변경 시 깨질 수 있다.
      - 과도한 호출은 서비스에 부담이 될 수 있으므로, 1일 1회 수준을 권장한다.

    반환: (rows, meta)
      rows: dict 리스트 (가능한 한 FMP/HSW 스키마와 유사한 키로 맞춤)
      meta: {source, status_code, url, error}
    """
    url = "https://www.quiverquant.com/congresstrading/"
    meta = {"source": "quiver_recent_html", "status_code": None, "url": url, "error": None}
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        meta["status_code"] = resp.status_code
        resp.raise_for_status()

        # pandas.read_html는 내부적으로 lxml/bs4를 사용할 수 있음. 환경에 따라 실패 가능.
        try:
            tables = pd.read_html(resp.text)
        except Exception as e:
            meta["error"] = f"read_html_failed: {e}"
            return [], meta

        df = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if ("politician" in cols) and ("filed" in cols) and ("traded" in cols):
                df = t
                break
        if df is None:
            meta["error"] = "no_matching_table"
            return [], meta

        # 표 컬럼명 정규화
        df.columns = [str(c).strip() for c in df.columns]

        rows = []
        for _, r in df.iterrows():
            stock = str(r.get("Stock", "")).strip()
            politician = str(r.get("Politician", "")).strip()

            # ticker 추정: 'AAPL APPLE INC...' 형태에서 첫 토큰
            ticker = ""
            if stock:
                m = re.match(r"^([A-Z][A-Z0-9\.\-]{0,9})\b", stock)
                if m:
                    ticker = m.group(1)

            rows.append(
                {
                    "ticker": ticker,
                    "asset_description": stock,
                    "representative": politician,
                    "transaction_type": str(r.get("Transaction", "")).strip(),
                    "amount": str(r.get("Amount", "")).strip() if "Amount" in df.columns else "",
                    "disclosure_date": str(r.get("Filed", "")).strip(),
                    "transaction_date": str(r.get("Traded", "")).strip(),
                    "description": str(r.get("Description", "")).strip(),
                    "source": "quiver_html",
                }
            )

        return rows, meta

    except Exception as e:
        meta["error"] = str(e)
        return [], meta



def fetch_fmp_house_trades_by_name(name: str):
    """FMP House Trades By Name.

    반환: (data_or_none, meta)
    - FMP_API_KEY가 없으면 (None, {"source":"fmp","error":"missing_api_key"})
    """
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return None, {"source": "fmp", "url": None, "status_code": None, "error": "missing_api_key"}
    url = "https://financialmodelingprep.com/stable/house-trades-by-name"
    params = {"name": name, "apikey": api_key}
    data, meta = _safe_requests_get_json(url, params=params, return_meta=True)
    meta.update({"source": "fmp"})
    # apikey 노출 방지
    meta["url"] = _redact_apikey_in_text(meta.get("url") or url)
    meta["error"] = _redact_apikey_in_text(meta.get("error"))
    return data, meta


def fetch_fmp_house_latest(pages: int = 3, limit: int = 200):
    """FMP Latest House Financial Disclosures.

    Endpoint:
      https://financialmodelingprep.com/stable/house-latest?page=0&limit=100

    반환: (list_or_none, diag_list)
    """
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return None, [{"source": "fmp_house_latest", "url": None, "status_code": None, "error": "missing_api_key"}]

    base = "https://financialmodelingprep.com/stable/house-latest"
    all_rows = []
    diag = []
    for pageno in range(max(pages, 1)):
        params = {"page": pageno, "limit": limit, "apikey": api_key}
        data, meta = _safe_requests_get_json(base, params=params, return_meta=True)
        meta.update({"source": "fmp_house_latest"})
        meta["url"] = _redact_apikey_in_text(meta.get("url") or base)
        meta["error"] = _redact_apikey_in_text(meta.get("error"))
        diag.append(meta)

        rows = _normalize_trade_rows(data)
        if rows:
            all_rows.extend(rows)

        # 더 이상 결과가 없으면 종료(보수적)
        if data is None or (isinstance(data, list) and len(data) == 0):
            break

    return (all_rows if all_rows else None), diag


def _normalize_trade_rows(raw):
    """서로 다른 소스의 raw JSON을 list[dict]로 정규화."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # 흔한 케이스: {"data":[...]} 또는 {"results":[...]}
        for k in ["data", "results", "items"]:
            if k in raw and isinstance(raw[k], list):
                return raw[k]
        # 단일 object
        return [raw]
    return []


def _pick(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return default


def get_pelosi_trades(days=120, max_rows=25):
    """Nancy Pelosi 관련 최근 트레이드를 가져온다.

    우선순위(기본):
      - FMP_API_KEY가 있으면: FMP → HouseStockWatcher
      - 키가 없으면: HouseStockWatcher → (FMP는 생략)

    반환: (trades_list, diag_list)
      trades_list: 정규화된 dict 리스트 (최신순, 최대 max_rows)
      diag_list: 시도한 소스들의 메타(HTTP 상태/에러) 목록
    """
    name = os.environ.get("PELOSI_NAME", "Nancy Pelosi").strip()
    name_l = name.lower()
    diag = []

    has_fmp = bool(os.environ.get("FMP_API_KEY"))

    rows = []
    # 1) 소스 선택 순서
    if has_fmp:
        # 1) FMP by-name (일부 플랜에서 402 Payment Required 가능)
        raw, meta = fetch_fmp_house_trades_by_name(name)
        diag.append(meta)
        rows = _normalize_trade_rows(raw)

        # 2) by-name가 402면, house-latest로 대체(페이지를 가져와 이름으로 필터)
        if (not rows) and (meta.get("status_code") == 402):
            raw_latest, diag_latest = fetch_fmp_house_latest(pages=3, limit=200)
            diag.extend(diag_latest)
            rows = _normalize_trade_rows(raw_latest)

        # 3) 그래도 없으면 HouseStockWatcher 시도
        if not rows:
            raw2, diag2 = fetch_house_stockwatcher_trades()
            diag.extend(diag2)
            rows = _normalize_trade_rows(raw2)

        # 4) 최후의 수단: QuiverQuant 공개 대시보드(HTML) 파싱
        if not rows:
            raw3, meta3 = fetch_quiver_recent_congress_trades()
            diag.append(meta3)
            rows = _normalize_trade_rows(raw3)
    else:
        raw2, diag2 = fetch_house_stockwatcher_trades()
        diag.extend(diag2)
        rows = _normalize_trade_rows(raw2)

        if not rows:
            raw3, meta3 = fetch_quiver_recent_congress_trades()
            diag.append(meta3)
            rows = _normalize_trade_rows(raw3)

    # Pelosi 이름 필터 (소스별 키가 다름)
    filtered = []
    for r in rows:
        who = str(_pick(r, ["representative", "representative_name", "name", "politician", "member"], "")).lower()
        if "pelosi" in who or name_l in who:
            filtered.append(r)

    if not filtered:
        return [], diag

    # 날짜 파싱 + 최근 N일 필터
    import pandas as pd

    def parse_dt(x):
        if not x:
            return pd.NaT
        try:
            return pd.to_datetime(x, utc=True, errors="coerce")
        except Exception:
            return pd.NaT

    norm = []
    for r in filtered:
        filed = parse_dt(_pick(r, ["disclosure_date", "filingDate", "filed", "filedDate", "reportDate"]))
        trade = parse_dt(_pick(r, ["transaction_date", "transactionDate", "transactionDateStart", "transactionDateBegin", "transactionDate"]))
        if pd.isna(trade):
            trade = parse_dt(_pick(r, ["transaction_date_end", "transactionDateEnd"]))
        norm.append((filed, trade, r))

    # 기준일
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    kept = []
    for filed, trade, r in norm:
        dt_ref = filed if not pd.isna(filed) else trade
        if pd.isna(dt_ref) or dt_ref >= cutoff:
            kept.append((filed, trade, r))

    # 최신순 정렬 (filed 우선)
    kept.sort(
        key=lambda x: (
            x[0] if not pd.isna(x[0]) else pd.Timestamp.min,
            x[1] if not pd.isna(x[1]) else pd.Timestamp.min,
        ),
        reverse=True,
    )

    return [r for _, _, r in kept[:max_rows]], diag


def build_pelosi_trades_section_html(df_enriched, days=120, max_rows=25):
    """리포트용 Pelosi 트레이드 섹션 HTML."""
    portfolio_tickers = set(df_enriched["Ticker"].astype(str).str.upper().tolist())

    trades, diag = get_pelosi_trades(days=days, max_rows=max_rows)

    # 데이터가 없을 때도 리포트는 계속 진행
    if not trades:
        has_key = bool(os.environ.get("FMP_API_KEY"))
        # 진단 정보(가능한 범위 내)
        diag_lines = []
        for d in (diag or [])[:5]:
            src = d.get("source")
            sc = d.get("status_code")
            err = _redact_apikey_in_text(d.get("error"))
            url = _redact_apikey_in_text(d.get("url"))
            diag_lines.append(f"- {src}: status={sc}, error={err}, url={url}")
        diag_html = "<br/>".join(diag_lines) if diag_lines else "- (no diagnostics)"

        key_hint = ""
        if not has_key:
            key_hint = "<br/>- FMP_API_KEY가 없으면 FMP 경로를 사용하지 않습니다(현재: 미설정)."
        else:
            key_hint = "<br/>- FMP_API_KEY는 설정되어 있으나 호출이 실패했을 수 있습니다(401/403/429 여부는 진단을 확인)."

        return (
            "<p class='muted'>Pelosi 트레이드 데이터를 가져오지 못했습니다."
            "<br/>- 원인 후보: 데이터 소스 장애(5xx), 인증 실패(401/403), 호출 제한(429), 스키마 변경"
            f"{key_hint}"
            "<br/><br/><b>진단(최근 시도):</b><br/>"
            f"{diag_html}"
            "</p>"
        )

    # 표 렌더링용 정규화
    rows = []
    for r in trades:
        symbol = str(_pick(r, ["ticker", "symbol", "assetSymbol", "asset_symbol"], "")).upper().strip()
        tx_type = str(_pick(r, ["transaction_type", "transactionType", "transaction", "type"], "")).strip()
        filed = _pick(r, ["disclosure_date", "filingDate", "filed", "filedDate", "reportDate"], "")
        trade = _pick(r, ["transaction_date", "transactionDate", "transactionDateStart", "transactionDateBegin", "transactionDate"], "")
        amount = _pick(r, ["amount", "amountRange", "range", "amount_range"], "")
        owner = _pick(r, ["owner", "owner_type", "ownerType"], "")
        desc = _pick(r, ["asset_description", "assetDescription", "description", "asset", "security"], "")

        # 내 보유 종목과 매칭 표시
        hit = "✅" if symbol and symbol in portfolio_tickers else ""
        if hit:
            symbol_disp = f"<b>{symbol}</b>"
        else:
            symbol_disp = symbol

        rows.append(
            {
                "Hit": hit,
                "Symbol": symbol_disp,
                "Type": tx_type,
                "TradeDate": trade,
                "FiledDate": filed,
                "Amount": amount,
                "Owner": owner,
                "Desc": desc,
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)

    hit_html = ""
    if "Hit" in df.columns and (df["Hit"] == "✅").any():
        hits = df[df["Hit"] == "✅"].copy()
        # 같은 심볼 중복 축소
        hits2 = hits.copy()
        hits2["RawSymbol"] = hits2["Symbol"].astype(str).str.replace("<b>", "", regex=False).str.replace("</b>", "", regex=False)
        hits2 = hits2.drop_duplicates(subset=["RawSymbol"], keep="first").drop(columns=["RawSymbol"])
        hit_html = "<p><b>내 보유 종목과 겹치는 최근 공시:</b></p>" + hits2.to_html(index=False, escape=False)

    table_html = df.to_html(index=False, escape=False)

    # 공시 지연 주의 문구
    caution = "<p class='muted'>※ STOCK Act 공시는 거래 후 일정 기간 지연될 수 있어, 매매 신호가 아니라 참고 지표로만 사용 권장</p>"

    return hit_html + "<p><b>최근 Pelosi 관련 공시(최신순):</b></p>" + caution + table_html

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
    📈 중단기 투자의 통합 분석 (TFSA 종목만, SCHD 제외)

    1) 요약표 : Ticker + 중기 상승 확률 % / 매수 타이밍 % / 매도 타이밍 % / 1년 목표수익 범위
    2) 상세표 : '핵심 투자 코멘트' + '주요맥락'

    대상:
      - df_enriched 중에서 Type == 'TFSA' 인 종목만 포함
      - 그 중 Ticker != 'SCHD'
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

    # 3) 각 종목별 중단기 분석 + 맥락 생성
    for t in sorted(tickers):
        try:
            stat = analyze_midterm_ticker(t)
        except Exception as e:
            print(f"[WARN] analyze_midterm_ticker 실패: {t}, {e}")
            continue

        try:
            ctx = build_midterm_context(t)
        except Exception as e:
            print(f"[WARN] build_midterm_context 실패: {t}, {e}")
            ctx = "맥락 정보를 불러오지 못했습니다."

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

        # ② 상세 테이블 행
        rows_detail.append(
            {
                "Ticker": stat["Ticker"],
                "핵심 투자 코멘트": stat["Comment"],
                "주요맥락": ctx,
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
    SCHD 장기 배당 분석 (모든 값을 CAD 기준으로 계산 및 표시).

    가정:
    - DRIP 적용 (배당금 재투자)
    - 매월 200 USD를 환전(CAD)해서 매수
    - 연평균 배당 성장률 g = 11% 고정
    - 목표 배당: 월 CAD 1,000 (연 CAD 12,000)
    """
    current_shares = safe_float(current_shares, 0.0)
    if current_shares <= 0:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> N/A (보유 SCHD 없음)</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    tk = yf.Ticker("SCHD")

    # 1) 배당 데이터 (USD 기준)
    try:
        divs = tk.dividends.dropna()
    except Exception:
        divs = pd.Series(dtype=float)

    if divs.empty:
        return (
            "<p><strong>현재 예상 연 배당금(CAD):</strong> 데이터 부족</p>"
            "<p><strong>월 CAD 1,000 배당 달성 예상:</strong> 계산 불가</p>"
        )

    # 연간 총 배당(USD/주)
    div_by_year = divs.groupby(divs.index.year).sum()
    years = sorted(div_by_year.index)
    last_year = years[-1]
    last_div_ps_usd = float(div_by_year[last_year])  # 마지막 완료 연도 배당(USD/주)

    # 2) 현재 SCHD 가격(USD)
    try:
        px = tk.history(period="1mo")["Close"].dropna()
        price_usd = float(px.iloc[-1]) if not px.empty else 75.0
    except Exception:
        price_usd = 75.0  # fallback

    # 3) USD→CAD 환율
    try:
        fx = yf.Ticker("CAD=X").history(period="1d")["Close"].dropna()
        usd_to_cad = float(fx.iloc[-1]) if not fx.empty else 1.35
    except Exception:
        usd_to_cad = 1.35

    # 4) 현재 연 배당금(CAD 기준)
    #    (보유주수 × 연간 배당(USD/주) × 환율)
    current_annual_income_cad = current_shares * last_div_ps_usd * usd_to_cad

    # 5) 배당 성장률 (고정 가정)
    g = 0.11   # 11%

    # 6) 현재 배당 수익률 (USD 기준)
    #    y = 연간 배당 / 현재가
    y = last_div_ps_usd / price_usd if price_usd > 0 else 0.035
    if y <= 0:
        y = 0.035  # 최소 3.5%로 보수적 가정

    # 7) 매월 200 USD를 CAD로 환전 후 투자
    monthly_usd = 200.0
    monthly_cad = monthly_usd * usd_to_cad
    annual_contrib_cad = monthly_cad * 12.0

    # 8) 단순화된 해석:
    #    - 연간 배당 수익률 y, 배당 성장률 g
    #    - 연간 기여금(투자액)으로 인한 "추가 배당 성장 효과"를 A로 흡수
    #
    #    A = 연간 기여금 × (수익률 / 성장률)
    #    목표: I(t) ≥ Target,  I(t)는 배당 성장/기여 효과가 합쳐진 값
    #    n_years = ln((Target + A) / (I0 + A)) / ln(1 + g)
    #
    #    여기서 모든 단위는 CAD 기준으로 처리.
    A = annual_contrib_cad * (y / g)

    target_annual_cad = 12_000.0  # 연 CAD 12,000 = 월 CAD 1,000
    numerator = target_annual_cad + A
    denominator = current_annual_income_cad + A

    if numerator <= denominator:
        n_years = 0.0
    else:
        n_years = np.log(numerator / denominator) / np.log(1.0 + g)

    n_years = max(0.0, n_years)
    years_int = int(n_years)
    months_int = int(round((n_years - years_int) * 12.0))

    # 9) 출력 (통화 기호는 CAD임을 명시하기 위해 "C$" 사용)
    txt = (
        f"<p><strong>현재 예상 연 배당금(CAD):</strong> "
        f"{fmt_money(current_annual_income_cad, 'C$')} "
        f"(보유 {current_shares:,.0f}주 기준)</p>"
        f"<p><strong>월 CAD 1,000 배당 달성 예상:</strong> "
        f"약 {years_int}년 {months_int}개월 "
        f"(DRIP + 매월 200 USD(환전 후 투자) / 배당 성장률 11% 가정)</p>"
    )
    return txt
    

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
        return sub.to_html(index=False, escape=False)

    tfsa_table = make_holdings_table("TFSA")
    resp_table = make_holdings_table("RESP")

    # ---------- 3) 중단기 투자 분석 (전체 보유 종목) ----------
    midterm_html = build_midterm_analysis_html(df_enriched)
    # ---------- 3.5) Congress Trading Watch (Nancy Pelosi) ----------
    pelosi_html = build_pelosi_trades_section_html(df_enriched)

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
          <h2>🏛️ Congress Trading Watch (Nancy Pelosi)</h2>
          <p class="muted">
            ※ 공개 공시(STOCK Act) 기반 데이터입니다. 거래 발생일과 보고일 사이에 지연이 있을 수 있습니다(최대 45일).
            데이터 소스/가용성에 따라 누락 또는 오류가 있을 수 있습니다.
          </p>
          {pelosi_html}
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
