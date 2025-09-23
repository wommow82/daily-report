
#!/usr/bin/env python3
# coding: utf-8

"""
Daily Recommendation Report with ETF & Crypto ETF (v3)
-----------------------------------------------------
- STOCK/ETF 분리 출력 (타이밍/재무)
- 섹터/카테고리 라벨 추가
- 모든 수치 소수점 2자리 표기
- 최근 7일 뉴스 기반 추천 섹션(종목 | 짧은 설명)
- 이메일 발송 유지 (SMTP)
"""

import os, smtplib, html
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------
# Config
# ------------------------
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
REPORT_DATE    = datetime.now().strftime("%Y-%m-%d")

# ------------------------
# Universe & Classification
# ------------------------
DEFAULT_STOCKS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","MA",
    "UNH","HD","PG","XOM","CVX","LLY","JNJ","MRK","KO","PEP","DIS","INTC","AMD","QCOM","AVGO",
    "TXN","IBM","ORCL","ADBE","CRM","NOW","PFE","BMY","ABT","TMO","DHR","ISRG","GE","CAT","DE",
    "NKE","WMT","COST","BAC","GS","MS","WFC","C","NFLX","SHOP","PYPL","NEE","DUK","SO",
    "LMT","RTX","NOC","BA","GD"]
DEFAULT_ETFS = ["SPY","VOO","IVV","VTI","QQQ","DIA","IWM","XLK","XLF","XLE","XLV","XLY","XLP","XLU",
    "XLI","XLB","XLRE","XLC","ARKK","ARKW","SMH","SOXX","IBB","TAN","HACK","CIBR"]
CRYPTO_ETFS = ["BITO","IBIT","FBTC","ARKB","BRRR","EETH","ETHE"]

ETF_GROUPS = {
    "Broad Market": {"SPY","VOO","IVV","VTI","QQQ","DIA","IWM"},
    "Sector ETF": {"XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLI","XLB","XLRE","XLC"},
    "Thematic ETF": {"ARKK","ARKW","SMH","SOXX","IBB","TAN","HACK","CIBR"},
    "Crypto ETF": set(CRYPTO_ETFS),
}

UNIVERSE = [t.strip().upper() for t in os.getenv("UNIVERSE_TICKERS","").split(",") if t.strip()] \
            or DEFAULT_STOCKS + DEFAULT_ETFS + CRYPTO_ETFS

def is_etf(ticker: str) -> bool:
    return ticker in (DEFAULT_ETFS + CRYPTO_ETFS)

def classify_stock_category(sector: str, industry: str, ticker: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    t = ticker.upper()
    if "semiconductor" in i or t in {"NVDA","AMD","INTC","QCOM","AVGO","TXN","TSM"}:
        return "반도체"
    if "software" in i or t in {"MSFT","ADBE","CRM","NOW","ORCL","IBM","GOOGL"}:
        return "테크"
    if "internet" in i or t in {"META","GOOGL","AMZN","NFLX","SHOP"}:
        return "테크"
    if "aerospace" in i or "defense" in i or t in {"LMT","RTX","NOC","BA","GD"}:
        return "방위산업"
    if "oil" in i or "gas" in i or s == "energy" or t in {"XOM","CVX","COP"}:
        return "에너지"
    if s in {"healthcare"} or t in {"LLY","PFE","JNJ","MRK","ABT","TMO","DHR","ISRG","BMY"}:
        return "헬스케어"
    if s in {"financial services","financial","banks"} or t in {"JPM","BAC","GS","MS","WFC","C"}:
        return "금융"
    if s in {"consumer cyclical","consumer defensive"} or t in {"NKE","WMT","HD","COST","KO","PEP","DIS"}:
        return "소비재"
    if s in {"industrials"} or t in {"GE","CAT","DE"}:
        return "산업재"
    if s in {"utilities"} or t in {"NEE","DUK","SO"}:
        return "유틸리티"
    if s in {"real estate"}:
        return "리츠/부동산"
    return "기타"

def classify_etf_group(ticker: str) -> str:
    u = ticker.upper()
    for g, tickers in ETF_GROUPS.items():
        if u in tickers:
            return g
    if u in CRYPTO_ETFS or "BTC" in u or "ETH" in u:
        return "Crypto ETF"
    return "ETF"

# ------------------------
# News-based Recommendation (7 days)
# ------------------------
TOPIC_REASON_MAP = {
    "White House policy": "백악관 정책 뉴스 → 정치 불확실성 관련 종목 주목",
    "President remarks": "대통령 발언 관련 → 정책 수혜 기대 종목 주목",
    "executive order": "행정명령 발동 → 빅테크/방위산업 관련 수혜 가능",
    "tariffs": "관세 관련 뉴스 → 제조업/수출주 영향",
    "defense budget": "국방 예산 확대 → 방위산업주(LMT, RTX, NOC 등) 수혜",
    "NATO": "NATO 이슈 → 국방주, 방위산업주 주목",
    "healthcare policy": "헬스케어 정책 변화 → 제약/보험주 영향",
    "drug pricing": "약가 규제 뉴스 → 제약/바이오주 관심",
    "energy policy": "에너지 정책 발표 → 정유/에너지주 수혜",
    "OPEC": "OPEC 관련 뉴스 → 원유·에너지 섹터 영향",
    "semiconductor subsidies": "반도체 보조금 정책 → NVDA/AMD/SMH 수혜",
    "CHIPS Act": "CHIPS Act → 미국 반도체 산업 수혜 기대",
    "AI regulation": "AI 규제 논의 → Big Tech (MSFT, GOOGL, META) 주목",
    "crypto ETF flows": "비트코인 ETF 자금 유입 → Crypto ETF 강세 기대",
    "bitcoin inflows": "BTC 자금 유입 증가 → 비트코인 ETF 관심",
    "ethereum ETF": "이더리움 ETF 이슈 → ETH ETF 강세 가능성",
}

def fetch_news(query, from_days=7, page_size=30):
    if not NEWS_API_KEY:
        return []
    try:
        from_date = (datetime.utcnow() - timedelta(days=from_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "from": from_date, "sortBy": "publishedAt", "language": "en",
                  "pageSize": page_size, "apiKey": NEWS_API_KEY}
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []

def build_news_recos():
    topics = list(TOPIC_REASON_MAP.keys())
    keyword_map = {
        "defense": ["LMT","RTX","NOC","GD","BA"],
        "semiconductor": ["NVDA","AMD","QCOM","INTC","AVGO","TXN","SMH","SOXX"],
        "chip": ["NVDA","AMD","QCOM","INTC","AVGO","TXN","SMH","SOXX"],
        "tariff": ["AAPL","CAT","DE","GM","F"],
        "drug": ["LLY","PFE","MRK","BMY","JNJ","IBB"],
        "medicare": ["UNH","CI","HUM","CVS"],
        "energy": ["XOM","CVX","COP","XLE"],
        "opec": ["XOM","CVX","COP","XLE"],
        "ai": ["NVDA","MSFT","GOOGL","META","ADBE","CRM","NOW","ARKK"],
        "executive order": ["NVDA","MSFT","GOOGL","META","LMT","RTX"],
        "crypto": ["IBIT","FBTC","ARKB","BITO","BRRR","EETH","ETHE"],
        "bitcoin": ["IBIT","FBTC","ARKB","BITO","BRRR"],
        "ethereum": ["EETH","ETHE"]
    }

    articles = []
    for q in topics:
        arts = fetch_news(q, from_days=7, page_size=30)
        for a in arts:
            a["_topic"] = q
        articles.extend(arts)

    scores, reasons = {}, {}
    for a in articles:
        content = (a.get("title") or "" + " " + a.get("description") or "").lower()
        topic = a.get("_topic","")
        for key, tickers in keyword_map.items():
            if key in content:
                for t in tickers:
                    scores[t] = scores.get(t, 0.0) + 1.0
                    if t not in reasons:
                        reasons[t] = TOPIC_REASON_MAP.get(topic, f"{topic} 관련 뉴스")

    rows = [{"종목":k, "설명":reasons.get(k,"")} for k,v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return pd.DataFrame(rows).head(10)

# ------------------------
# (이하 주식/ETF 분석 코드 동일 - 생략)
# ------------------------
