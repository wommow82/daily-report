# -*- coding: utf-8 -*-
# =====================================================================
# Market_Report ver 1  (단일 파일 버전, B 리포트)
#   - 이 파일 하나만 저장소에 올리면 됩니다 (+ 워크플로 yml, requirements_ver4.txt).
#   - 아래 EMBEDDED_* 블록은 별도 모듈 파일(simulation_module.py, ledger.py, combined_sim.py,
#     pnl_overview.py, history_log.py, shares_log.py, dividends_log.py, technicals.py, ai_briefing.py)의 내용을 그대로 담은 것입니다.
# =====================================================================
import sys as _sys
import types as _types


def _register_embedded(name, source):
    mod = _types.ModuleType(name)
    mod.__file__ = "<embedded " + name + ">"
    _sys.modules[name] = mod
    exec(compile(source, "<embedded " + name + ">", "exec"), mod.__dict__)

# ---------- EMBEDDED_TECHNICALS ----------
EMBEDDED_TECHNICALS = r'''# -*- coding: utf-8 -*-
"""
technicals.py - 종목별 차트 기술지표 요약 (계산은 전부 코드, AI 호출 없음)

보여주는 것: 현재가, 50일/200일 이동평균, RSI(14), 52주 고점 대비, 3개월 수익률
기준(출처)
  - RSI: 14일 기준, 70 초과 '과열권', 30 미만 '침체권'
    https://www.webull.com/learn/courseware/9fAaC0/Relative-Strength-Index-RSI
  - 50일선이 200일선 위로 올라선 것을 골든크로스(상승 전환 신호), 아래로 내려선 것을 데드크로스(하락 전환 신호)라 부름
    https://www.forex.com/en-us/news-and-analysis/golden-cross-vs-death-cross-indicators/
  - 현재가와 이동평균의 위/아래 비교, 52주 고점 대비, 수익률은 단순 사실 표시이며 앞날을 예측하지 않음
"""
import html as _html

RSI_PERIOD = 14
MIN_BARS = 60          # 이보다 적으면 계산하지 않음
BARS_1M, BARS_3M, BARS_52W = 21, 63, 252


def _esc(s):
    return _html.escape(str(s), quote=True)


def sma(values, n):
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def rsi_wilder(values, period=RSI_PERIOD):
    """Wilder 평활 RSI. 데이터가 period+1개보다 적으면 None."""
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for a, b in zip(values[:-1], values[1:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def compute(closes):
    """종가 리스트(오래된 것 -> 최근)로 지표 계산. 부족하면 None."""
    c = [float(x) for x in closes if x is not None and float(x) > 0]
    if len(c) < MIN_BARS:
        return None
    last = c[-1]
    win = c[-BARS_52W:]
    hi, lo = max(win), min(win)
    out = {
        "last": last,
        "sma50": sma(c, 50),
        "sma200": sma(c, 200),
        "rsi": rsi_wilder(c),
        "hi52": hi,
        "lo52": lo,
        "from_hi": (last / hi - 1.0) * 100.0,
        "ret3m": (last / c[-1 - BARS_3M] - 1.0) * 100.0 if len(c) > BARS_3M else None,
        "ret1m": (last / c[-1 - BARS_1M] - 1.0) * 100.0 if len(c) > BARS_1M else None,
        "bars": len(c),
    }
    return out


def labels(m):
    """지표 값을 (이름, 색 종류) 라벨로. 색 종류: good/bad/warn/info"""
    out = []
    if m.get("sma200") is not None:
        out.append(("200일선 위", "good") if m["last"] >= m["sma200"] else ("200일선 아래", "bad"))
    if m.get("sma50") is not None and m.get("sma200") is not None:
        out.append(("50일선>200일선(골든크로스 상태)", "good") if m["sma50"] > m["sma200"]
                   else ("50일선<200일선(데드크로스 상태)", "bad"))
    elif m.get("sma50") is not None:
        out.append(("50일선 위", "good") if m["last"] >= m["sma50"] else ("50일선 아래", "bad"))
    r = m.get("rsi")
    if r is not None:
        if r > 70:
            out.append(("RSI 과열권", "warn"))
        elif r < 30:
            out.append(("RSI 침체권", "info"))
    return out


NEAR_HIGH, FAR_FROM_HIGH = -5.0, -20.0   # 52주 고점 대비 구간(이 코드에서 정한 설명용 구간, 통상 기준 아님)


def interpret(m):
    """지표를 한두 문장으로 풀어쓴다. 현재 상태 설명만 하고 예측·매매 의견은 쓰지 않는다."""
    last, s50, s200, rsi, fh = m["last"], m.get("sma50"), m.get("sma200"), m.get("rsi"), m.get("from_hi")
    if s200 is not None and s50 is not None:
        if last >= s200 and s50 > s200:
            trend = "중장기 이동평균이 상승 배열입니다(현재가가 200일선 위, 50일선이 200일선 위)."
        elif last < s200 and s50 < s200:
            trend = "중장기 이동평균이 하락 배열입니다(현재가가 200일선 아래, 50일선이 200일선 아래)."
        elif last >= s200:
            trend = "현재가는 200일선 위지만 50일선이 200일선 아래라 추세 신호가 엇갈립니다."
        else:
            trend = "현재가는 200일선 아래지만 50일선이 200일선 위라 추세 신호가 엇갈립니다."
    elif s50 is not None:
        trend = "현재가가 50일선 위입니다." if last >= s50 else "현재가가 50일선 아래입니다."
    else:
        trend = "이동평균을 계산할 데이터가 부족합니다."
    extra = []
    if rsi is not None:
        if rsi > 70:
            extra.append(f"RSI {rsi:.0f}로 단기 과열권입니다.")
        elif rsi < 30:
            extra.append(f"RSI {rsi:.0f}로 단기 침체권입니다.")
    if fh is not None:
        if fh >= NEAR_HIGH:
            extra.append("52주 고점 부근입니다.")
        elif fh <= FAR_FROM_HIGH:
            extra.append(f"52주 고점보다 {abs(fh):.0f}% 낮은 위치입니다.")
    return " ".join([trend] + extra)


_COLOR = {"good": "#2e7d32", "bad": "#c62828", "warn": "#e65100", "info": "#1976d2", "mixed": "#757575"}


def verdict(m):
    """추세 신호만으로 종합 판정: good(상승 배열) / bad(하락 배열) / mixed(엇갈림·데이터 부족).
    RSI·52주 고점은 판정에 넣지 않고 해석 문장에만 적는다."""
    last, s50, s200 = m["last"], m.get("sma50"), m.get("sma200")
    if s50 is None or s200 is None:
        return "mixed"
    if last >= s200 and s50 > s200:
        return "good"
    if last < s200 and s50 < s200:
        return "bad"
    return "mixed"


_VERDICT_LABEL = {"good": "긍정 신호", "bad": "부정 신호", "mixed": "신호 혼재"}


def _num(v, fmt="{:,.2f}", dash="-"):
    return dash if v is None else fmt.format(v)


def fetch_closes(ticker, period="2y"):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period=period)["Close"].dropna()
    return [float(x) for x in h.tolist()]


def build_technicals_report(tickers, fetch_fn=None):
    """반환: (html, facts). 예외를 던지지 않는다. facts는 다른 곳에서 쓸 수 있는 문장 목록."""
    try:
        fetch_fn = fetch_fn or fetch_closes
        tickers = [t for t in dict.fromkeys(str(x).strip() for x in tickers) if t]
        if not tickers:
            return "<p class='muted'>분석할 성장/모멘텀 종목이 없습니다.</p>", []
        rows, problems, facts = "", [], []
        for t in tickers:
            try:
                m = compute(fetch_fn(t))
            except Exception as e:
                problems.append(f"{t}: {e}")
                continue
            if m is None:
                problems.append(f"{t}: 가격 데이터가 부족합니다(최소 {MIN_BARS}일)")
                continue
            chips = "".join(
                f"<span style='background:{_COLOR[k]};color:#fff;border-radius:10px;padding:1px 7px;"
                f"font-size:10px;margin:1px 2px 1px 0;display:inline-block;'>{_esc(n)}</span>"
                for n, k in labels(m))
            rsi = m["rsi"]
            rsi_col = _COLOR["warn"] if (rsi is not None and (rsi > 70)) else (
                _COLOR["info"] if (rsi is not None and rsi < 30) else "inherit")
            fh = m["from_hi"]
            rows += (
                f"<tr><td style='padding:4px 6px;text-align:left;'><strong>{_esc(t)}</strong></td>"
                f"<td style='padding:4px 6px;text-align:right;'>{_num(m['last'])}</td>"
                f"<td style='padding:4px 6px;text-align:right;'>{_num(m['sma50'])}</td>"
                f"<td style='padding:4px 6px;text-align:right;'>{_num(m['sma200'])}</td>"
                f"<td style='padding:4px 6px;text-align:right;color:{rsi_col};'>{_num(rsi, '{:.0f}')}</td>"
                f"<td style='padding:4px 6px;text-align:right;'>{_num(fh, '{:+.1f}%')}</td>"
                f"<td style='padding:4px 6px;text-align:right;'>{_num(m['ret3m'], '{:+.1f}%')}</td>"
                f"<td style='padding:4px 6px;text-align:left;'>{chips}</td></tr>"
                f"<tr><td colspan='8' style='padding:0 6px 6px;text-align:left;font-size:12px;"
                f"color:{_COLOR[verdict(m)]};font-weight:600;border-bottom:1px solid #eee;'>"
                f"→ [{_VERDICT_LABEL[verdict(m)]}] {_esc(interpret(m))}</td></tr>")
            facts.append(f"{t}: 현재가 {m['last']:.2f}, RSI {_num(rsi, '{:.0f}')}, 52주 고점 대비 {fh:+.1f}%")
        if not rows:
            return ("<p class='muted'>기술지표를 계산하지 못했습니다: " + _esc("; ".join(problems[:6])) + "</p>"), []
        head = ("<tr style='border-bottom:1px solid #ddd;color:#666;'>"
                "<th style='padding:4px 6px;text-align:left;'>종목</th><th style='text-align:right;'>현재가</th>"
                "<th style='text-align:right;'>50일선</th><th style='text-align:right;'>200일선</th>"
                "<th style='text-align:right;'>RSI(14)</th><th style='text-align:right;'>52주 고점 대비</th>"
                "<th style='text-align:right;'>3개월</th><th style='text-align:left;'>상태</th></tr>")
        note = ("<p class='muted' style='text-align:left;'>※ 기준: RSI 70 초과 과열권·30 미만 침체권(14일), "
                "50일선이 200일선 위=골든크로스 상태·아래=데드크로스 상태. 현재 상태를 요약한 것이며 앞날을 예측하지 않습니다. "
                "'→' 해석 줄은 코드가 지표 값으로 만든 현재 상태 설명이며, 이동평균 배열만으로 녹색(긍정)·빨강(부정)·회색(혼재)을 정합니다(RSI·52주 고점은 색에 반영하지 않음)(52주 고점 -5% 이내=부근, -20% 이하=크게 낮음은 이 코드에서 정한 구간). "
                "지표는 종가 기준이고 데이터가 부족한 항목은 '-'입니다. "
                "<a href='https://www.webull.com/learn/courseware/9fAaC0/Relative-Strength-Index-RSI' style='color:#1976d2;'>RSI 출처</a> · "
                "<a href='https://www.forex.com/en-us/news-and-analysis/golden-cross-vs-death-cross-indicators/' style='color:#1976d2;'>이동평균 출처</a></p>")
        prob = (f"<p class='muted' style='text-align:left;'>계산하지 못한 종목: {_esc('; '.join(problems[:6]))}</p>"
                if problems else "")
        return ("<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;font-size:12px;'>"
                + head + rows + "</table></div>" + note + prob), facts
    except Exception as e:
        return f"<p class='muted'>기술지표 요약 실패: {_esc(e)}</p>", []
'''
_register_embedded('technicals', EMBEDDED_TECHNICALS)

# ---------- EMBEDDED_MARKET_NEWS ----------
EMBEDDED_MARKET_NEWS = r'''# -*- coding: utf-8 -*-
"""
market_news.py - B 리포트의 뉴스 두 가지

  1) 큰 변동 종목 뉴스: 전일 대비 |변동| >= MOVER_PCT 인 보유 개별주에 최근 헤드라인(제목+링크)을 붙인다. AI 없음.
  2) 오늘의 시장 뉴스: 가져온 헤드라인 중에서만 AI가 3~5개를 고르고 '시장에 왜 중요한지'를 한 줄로 쓴다.
     - 근거 ID(N1, N2...)에 없는 항목은 버린다(AI가 기억으로 지어내는 것 방지).
     - 매매 지시 문구는 걸러낸다. AI 호출 실패 시 AI 없이 헤드라인만 보여준다.
  - 헤드라인(제목)만 읽는다. 기사 본문은 읽지 않으므로 요약은 제목 수준이다.
  - 뉴스 소스: NewsAPI(everything 검색) -> 없으면 Google News RSS. 실패해도 예외를 던지지 않는다.
"""
import html as _html
import os
import re
from datetime import datetime, timedelta

MOVER_PCT = 3.0            # 이 % 이상 움직인 종목에 뉴스를 붙인다(환경변수 MOVER_PCT로 변경)
MARKET_QUERY = "stock market OR Wall Street OR Nasdaq OR S&P 500 OR Federal Reserve OR Treasury yields"
MAX_FETCH = 30             # AI에 보여줄 헤드라인 최대 개수
MAX_PICK = 5
GOOD, BAD, WARN, MUTED = "#2e7d32", "#c62828", "#e65100", "#757575"

_BANNED = re.compile(
    r"(사세요|파세요|사십시오|파십시오|매수\s*(하세요|하십시오|하라|해야|추천|권고|권장)|"
    r"매도\s*(하세요|하십시오|하라|해야|추천|권고|권장)|비중\s*(늘리세요|줄이세요|확대하세요|축소하세요)|"
    r"손절\s*(하세요|해야)|익절\s*(하세요|해야)|담으세요|정리하세요|팔아야|사야\s*(합니다|해요|한다)|"
    r"\bbuy now\b|\bsell now\b|you should (buy|sell))", re.I)

NEWS_TOOL = {
    "name": "submit_market_news",
    "description": "주어진 헤드라인 중에서만 시장에 중요한 뉴스를 고른다.",
    "input_schema": {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "고른 헤드라인의 ID(예: N3). 목록에 있는 것만."},
                "title_ko": {"type": "string", "description": "헤드라인을 자연스러운 한국어로 번역(제목에 없는 내용 추가 금지, 고유명사는 원어 유지 가능)"},
                "why": {"type": "string", "description": "이 뉴스가 시장·투자자에게 왜 중요한지 한국어 한 문장(제목에 있는 사실만, 추측·매매 지시 금지)"},
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "제목에 직접 나온 관련 티커(없으면 빈 배열)"},
            },
            "required": ["id", "title_ko", "why"]}}},
        "required": ["items"],
    },
}

SYSTEM = (
    "당신은 투자 일지를 정리하는 조수입니다. 아래 '헤드라인 목록'은 오늘 수집한 뉴스 제목입니다.\n"
    "규칙: (1) 목록에 있는 ID만 고르세요. 목록에 없는 사건·숫자·인물은 쓰지 마세요. "
    "(2) 주식시장 전체나 큰 기업·금리·경제지표에 영향이 있을 만한 것 3~5개만 고르고, 연예·사생활성 가십은 주가·경영에 영향을 준 사실만 중립적으로 다루세요. "
    "(3) 'why'는 제목에 있는 사실만으로 한국어 한 문장. 원인을 단정하거나 앞날을 예측하지 마세요. "
    "(4) 사라/파세요 같은 매매 지시는 절대 쓰지 마세요. (5) title_ko는 제목을 충실하게 번역만 하고 내용을 보태지 마세요."
)


def _esc(s):
    return _html.escape(str(s), quote=True)


def _safe_url(u):
    u = str(u or "")
    return u if u.startswith(("http://", "https://")) else ""


# =========================
# 헤드라인 수집
# =========================

def fetch_headlines(query, api_key=None, page_size=10, days=2, http_get=None, rss_fn=None):
    """[{title,url,source,published}, ...]. NewsAPI -> Google News RSS 순. 실패는 빈 목록."""
    out = []
    if api_key:
        try:
            if http_get is None:
                import requests
                http_get = requests.get
            r = http_get("https://newsapi.org/v2/everything", params={
                "q": query, "apiKey": api_key, "language": "en", "sortBy": "publishedAt",
                "pageSize": page_size, "from": (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")}, timeout=20)
            if r.status_code == 200:
                for a in r.json().get("articles", []):
                    if a.get("title"):
                        out.append({"title": a.get("title"), "url": a.get("url"),
                                    "source": (a.get("source") or {}).get("name", ""),
                                    "published": (a.get("publishedAt") or "")[:10]})
        except Exception as e:
            print(f"⚠️ NewsAPI 오류(B): {e}")
    if not out:
        try:
            if rss_fn is None:
                import feedparser
                from urllib.parse import quote_plus
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en")
                entries = feed.entries[:page_size]
            else:
                entries = rss_fn(query)[:page_size]
            for e in entries:
                src = "Google News"
                try:
                    if getattr(e, "source", None):
                        src = e.source.get("title", src)
                except Exception:
                    pass
                out.append({"title": getattr(e, "title", ""), "url": getattr(e, "link", ""), "source": src,
                            "published": (getattr(e, "published", "") or "")[:16]})
        except Exception as e:
            print(f"⚠️ Google News RSS 오류(B): {e}")
    seen, uniq = set(), []
    for a in out:
        k = (a.get("title") or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq


# =========================
# 1) 큰 변동 종목 뉴스
# =========================

def movers(day_changes, threshold=None):
    """day_changes: {티커: 전일 대비 %}. 기준 이상 움직인 종목을 절댓값 큰 순으로."""
    th = MOVER_PCT if threshold is None else threshold
    return [t for t, p in sorted(day_changes.items(), key=lambda kv: -abs(kv[1] or 0))
            if p is not None and abs(p) >= th]


def mover_news_html(day_changes, news_fn, threshold=None, limit_stocks=5, per_stock=3):
    """큰 변동 종목별 헤드라인 표. 해당 종목이 없으면 한 줄 안내."""
    th = MOVER_PCT if threshold is None else threshold
    mv = movers(day_changes, th)[:limit_stocks]
    if not mv:
        return f"<p style='color:{MUTED};font-size:13px;'>어제 ±{th:g}% 이상 움직인 보유 종목이 없습니다.</p>"
    rows = ""
    for t in mv:
        p = day_changes[t]
        try:
            arts = (news_fn(t) or [])[:per_stock]
        except Exception as e:
            print(f"⚠️ {t} 뉴스 실패: {e}")
            arts = []
        col = GOOD if p > 0 else BAD
        if arts:
            links = "<br>".join(
                (f"• <a href='{_esc(_safe_url(a.get('url')))}' style='color:#1a56b0;text-decoration:none;'>{_esc(a.get('title'))}</a> "
                 f"<span style='color:{MUTED};font-size:11px;'>{_esc(a.get('source', ''))} {_esc(a.get('published', ''))}</span>")
                if _safe_url(a.get("url")) else f"• {_esc(a.get('title'))}" for a in arts)
        else:
            links = f"<span style='color:{MUTED};'>관련 뉴스를 찾지 못했습니다</span>"
        rows += (f"<tr style='border-top:1px solid #eee;'><td style='padding:6px;vertical-align:top;white-space:nowrap;'>"
                 f"<strong>{_esc(t)}</strong><br><span style='color:{col};font-weight:600;'>{p:+.1f}%</span></td>"
                 f"<td style='padding:6px;font-size:12px;'>{links}</td></tr>")
    return (f"<table style='width:100%;border-collapse:collapse;font-size:13px;'>{rows}</table>"
            f"<p style='font-size:11px;color:{MUTED};margin:4px 0 0;'>※ 헤드라인 검색 결과이며, 주가 변동의 '원인'을 확인한 것이 아닙니다.</p>")


# =========================
# 2) 오늘의 시장 뉴스
# =========================

def build_user_message(headlines):
    lines = [f"N{i + 1}. {a['title']} ({a.get('source', '')}, {a.get('published', '')})" for i, a in enumerate(headlines)]
    return "헤드라인 목록:\n" + "\n".join(lines) + "\n\n3~5개를 골라 submit_market_news 도구로 제출하세요."


def select_market_news(headlines, call_fn):
    """call_fn(system, user, tool)->JSON 텍스트. 반환: [{'a':기사,'why':str,'tickers':[...]}]. 검증 통과분만."""
    import json
    hs = headlines[:MAX_FETCH]
    if not hs:
        return []
    txt = call_fn(SYSTEM, build_user_message(hs), NEWS_TOOL)
    try:
        data = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt or "", re.S)
        data = json.loads(m.group(0)) if m else {}
    out, used = [], set()
    for it in (data.get("items") or []):
        m = re.fullmatch(r"N(\d+)", str(it.get("id", "")).strip())
        why = str(it.get("why", "")).strip()
        if not m or not why or _BANNED.search(why):
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(hs)) or idx in used:
            continue
        used.add(idx)
        tks = [str(t).upper() for t in (it.get("tickers") or []) if re.fullmatch(r"[A-Za-z.\-]{1,8}", str(t))]
        tks = [t for t in tks if t in hs[idx]["title"].upper()]      # 제목에 실제로 나온 티커만
        tko = str(it.get("title_ko", "")).strip()
        if _BANNED.search(tko):
            tko = ""
        out.append({"a": hs[idx], "why": why, "tickers": tks, "title_ko": tko})
        if len(out) >= MAX_PICK:
            break
    return out


def market_news_html(headlines, call_fn=None):
    """AI 선별(가능하면) + 표. AI 실패·키 없음이면 상위 헤드라인 5개만."""
    if not headlines:
        return f"<p style='color:{MUTED};font-size:13px;'>시장 뉴스를 가져오지 못했습니다.</p>"
    picked, note = [], ""
    if call_fn is not None:
        try:
            picked = select_market_news(headlines, call_fn)
        except Exception as e:
            note = f"AI 선별 실패({_esc(str(e)[:80])}) — 헤드라인만 표시합니다."
    if not picked:
        picked = [{"a": a, "why": "", "tickers": []} for a in headlines[:MAX_PICK]]
        note = note or ("AI 선별 없이 최신 헤드라인만 표시합니다." if call_fn is None else "AI가 고른 항목이 검증을 통과하지 못해 헤드라인만 표시합니다.")
    rows = ""
    for p in picked:
        a = p["a"]
        shown = p.get("title_ko") or a["title"]
        title = (f"<a href='{_esc(_safe_url(a.get('url')))}' style='color:#1a56b0;text-decoration:none;font-weight:600;'>{_esc(shown)}</a>"
                 if _safe_url(a.get("url")) else f"<strong>{_esc(shown)}</strong>")
        orig = f"<div style='color:{MUTED};font-size:11px;'>원문: {_esc(a['title'])}</div>" if p.get("title_ko") else ""
        tk = f" <span style='color:#555;font-size:11px;'>[{', '.join(_esc(t) for t in p['tickers'])}]</span>" if p["tickers"] else ""
        why = f"<div style='color:#333;margin-top:2px;'>→ {_esc(p['why'])}</div>" if p["why"] else ""
        rows += (f"<tr style='border-top:1px solid #eee;'><td style='padding:6px;font-size:13px;'>{title}{tk}{orig}"
                 f"<div style='color:{MUTED};font-size:11px;'>{_esc(a.get('source', ''))} {_esc(a.get('published', ''))}</div>{why}</td></tr>")
    foot = (f"<p style='font-size:11px;color:{MUTED};margin:4px 0 0;'>※ 헤드라인(제목)만 읽고 정리한 것입니다. 제목 번역은 AI가 했으니 중요한 건 원문을 확인하고, 기사 본문은 읽지 않았으니 링크로 원문을 보세요."
            + (f" {note}" if note else "") + "</p>")
    return f"<table style='width:100%;border-collapse:collapse;'>{rows}</table>{foot}"
'''
_register_embedded('market_news', EMBEDDED_MARKET_NEWS)

# ---------- EMBEDDED_MARKET_REPORT ----------
EMBEDDED_MARKET_REPORT = r'''# -*- coding: utf-8 -*-
"""
market_report.py - B 리포트(시장·종목): 어젯밤 시황, 보유 개별주 카드, 앞으로 2주 일정

원칙
  - 표·핵심 요약·기술지표는 AI 없이 계산한다. AI는 (선택) 세계 경제 브리핑과 시장 뉴스 선별에만 쓴다.
  - 가져오지 못한 값은 '-'로 두고 이유를 아래에 적는다. 예외를 던지지 않는다.
  - 가격 데이터: yfinance(비공식, 형식이 바뀔 수 있음). 일정: yfinance의 calendar / get_earnings_dates.
    https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html
  - VIX 기준(20 미만 안정, 30 초과 변동성 큼): Robinhood의 VIX 설명(앞선 대화에서 확인한 통상 기준)
"""
import html as _html
import os
from datetime import date, datetime, timedelta

import technicals as tech
import market_news as mn

HORIZON_DAYS = 14          # 일정 표시 범위
EPS_BAND = 0.05           # 예상 EPS가 전년 동기 대비 ±5% 이상이면 증가/감소로 색 표시(이 코드의 설명용 기준)
STALE_DAYS = 4             # 마지막 가격이 이보다 오래되면 '오래됨' 표시
GOOD, BAD, WARN, MUTED = "#2e7d32", "#c62828", "#e65100", "#757575"

# (표시 이름, 야후 티커, 종류) - 종류 'yield'면 변화량을 %p로 보여준다
MARKET_ITEMS = [
    ("S&P 500", "^GSPC", "idx"),
    ("나스닥 종합", "^IXIC", "idx"),
    ("다우", "^DJI", "idx"),
    ("VIX", "^VIX", "vix"),
    ("미 10년 금리(%)", "^TNX", "yield"),
    ("USD/CAD", "USDCAD=X", "fx"),
]


def _esc(s):
    return _html.escape(str(s), quote=True)


# =========================
# 데이터
# =========================

def fetch_series(ticker, period="2y"):
    """[(날짜 'YYYY-MM-DD', 종가), ...] 오래된 것 -> 최근."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period=period)["Close"].dropna()
    return [(idx.strftime("%Y-%m-%d"), float(v)) for idx, v in h.items()]


def _to_date(v):
    if v is None:
        return None
    try:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        import pandas as pd
        ts = pd.Timestamp(v)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def fetch_events(ticker):
    """다음 실적일·배당락일. 못 찾으면 None. 반환: {'earnings': date|None, 'ex_div': date|None}"""
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {"earnings": None, "ex_div": None}
    today = date.today()
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            ed = ed if isinstance(ed, (list, tuple)) else [ed]
            ds = sorted(d for d in (_to_date(x) for x in ed) if d and d >= today)
            out["earnings"] = ds[0] if ds else None
            out["ex_div"] = _to_date(cal.get("Ex-Dividend Date"))
    except Exception:
        pass
    if out["earnings"] is None:
        try:
            df = t.get_earnings_dates(limit=8)
            if df is not None and len(df):
                ds = sorted(d for d in (_to_date(i) for i in df.index) if d and d >= today)
                out["earnings"] = ds[0] if ds else None
        except Exception:
            pass
    if out["ex_div"] is not None and out["ex_div"] < today:
        out["ex_div"] = None
    return out


def _f(x):
    """숫자로 바꾸되 비어 있거나 NaN이면 None."""
    try:
        v = float(x)
        return None if v != v else v
    except Exception:
        return None


def fetch_earnings(ticker):
    """지난 실적(실제 vs 예상 EPS)과 다음 분기 애널리스트 예상치(야후 집계). 못 찾은 부분은 None.
    반환: {'last': {date, actual, estimate, surprise_pct}|None, 'next': {avg, low, high, n, year_ago, growth}|None}
    yfinance: get_earnings_dates / get_earnings_history / get_earnings_estimate
    https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html"""
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {"last": None, "next": None}
    today = date.today()
    try:                                                   # 지난 실적: 발표일이 지났고 실제 EPS가 있는 가장 최근 행
        df = t.get_earnings_dates(limit=12)
        if df is not None and len(df):
            cols = {str(c).lower(): c for c in df.columns}
            c_act, c_est = cols.get("reported eps"), cols.get("eps estimate")
            c_sur = next((c for k, c in cols.items() if k.startswith("surprise")), None)
            rows = []
            for idx, r in df.iterrows():
                d = _to_date(idx)
                act = _f(r[c_act]) if c_act is not None else None
                if d and d <= today and act is not None:
                    rows.append((d, act, _f(r[c_est]) if c_est is not None else None,
                                 _f(r[c_sur]) if c_sur is not None else None))
            if rows:
                d, act, est, sur = max(rows)
                if sur is None and est:
                    sur = (act - est) / abs(est) * 100.0
                out["last"] = {"date": d, "actual": act, "estimate": est, "surprise_pct": sur}
    except Exception:
        pass
    try:                                                   # 다음 분기 예상치: '0q'(이번 분기) 행
        est = t.get_earnings_estimate()
        if est is not None and len(est):
            row = est.loc["0q"] if "0q" in est.index else est.iloc[0]
            g = lambda k: _f(row[k]) if k in est.columns else None
            if g("avg") is not None:
                out["next"] = {"avg": g("avg"), "low": g("low"), "high": g("high"),
                               "n": g("numberOfAnalysts"), "year_ago": g("yearAgoEps"), "growth": g("growth")}
    except Exception:
        pass
    return out


def earnings_reaction(series, d):
    """발표일 d 전 거래일 종가 -> d 다음 거래일 종가 변화(%). 장 마감 전/후 발표 모두 포함. 데이터가 부족하면 None."""
    if not series or d is None:
        return None
    ds = [_to_date(x[0]) for x in series]
    i = next((k for k, x in enumerate(ds) if x and x >= d), None)
    if i is None or i < 1 or i + 1 >= len(series) or not series[i - 1][1]:
        return None
    return (series[i + 1][1] / series[i - 1][1] - 1.0) * 100.0


# =========================
# 계산
# =========================

def is_etf(ticker):
    """yfinance quoteType이 ETF/MUTUALFUND면 True. 알 수 없으면 False(개별주로 취급)."""
    try:
        import yfinance as yf
        qt = str((yf.Ticker(ticker).info or {}).get("quoteType", "")).upper()
        return qt in ("ETF", "MUTUALFUND")
    except Exception:
        return False


def pick_stocks(df_hold, income_tickers, exclude=None, etf_fn=None):
    """보유 수량이 있는 TFSA 개별주를 평가액 큰 순서로.
    제외: RESP, 0주, IncomeTickers, Settings의 MarketExclude(exclude), ETF(etf_fn이 True인 것)."""
    import pandas as pd
    inc = {str(t).strip().upper() for t in (income_tickers or [])} | {str(t).strip().upper() for t in (exclude or [])}
    d = df_hold.copy()
    d["_sh"] = pd.to_numeric(d["Shares"], errors="coerce").fillna(0)
    d = d[(d["_sh"] > 0) & (d["Type"].astype(str).str.upper() == "TFSA")]
    if "PositionValueBase" in d.columns:
        d = d.sort_values("PositionValueBase", ascending=False)
    out = []
    for t in d["Ticker"].astype(str):
        t = t.strip()
        if not t or t.upper() in inc or t in out:
            continue
        if etf_fn is not None:
            try:
                if etf_fn(t):
                    continue
            except Exception:
                pass
        out.append(t)
    return out


def day_change(series):
    """(마지막 날짜, 마지막 값, 전일 대비 변화율%, 전일 대비 변화량). 부족하면 None."""
    if not series or len(series) < 2:
        return None
    (d1, v1), (d0, v0) = series[-1], series[-2]
    if not v0:
        return None
    return d1, v1, (v1 / v0 - 1.0) * 100.0, v1 - v0


def _stale(last_date, today):
    d = _to_date(last_date)
    return d is not None and (today - d).days > STALE_DAYS


def _pct_html(p, mode="normal"):
    """mode: normal(오르면 녹색) / inverse(오르면 빨강, VIX) / neutral(색 없음, 환율·금리)"""
    if p is None:
        return "-"
    if mode == "neutral":
        col = "#222"
    elif mode == "inverse":
        col = BAD if p > 0 else (GOOD if p < 0 else MUTED)
    else:
        col = GOOD if p > 0 else (BAD if p < 0 else MUTED)
    return f"<span style='color:{col};font-weight:600;'>{p:+.2f}%</span>"


def vix_label(v):
    if v is None:
        return ""
    if v < 20:
        return "<span style='color:%s;'>안정권(20 미만)</span>" % GOOD
    if v > 30:
        return "<span style='color:%s;'>변동성 큼(30 초과)</span>" % BAD
    return "<span style='color:%s;'>보통(20~30)</span>" % WARN


def _vix_text(v):
    return "안정권(20 미만)" if v < 20 else ("변동성 큼(30 초과)" if v > 30 else "보통(20~30)")


def _dday(d, today):
    n = (d - today).days
    return "오늘" if n == 0 else (f"D-{n}" if n > 0 else f"D+{abs(n)}")


# =========================
# 렌더(표 중심)
# =========================

TH = "style='text-align:%s;padding:5px 6px;color:#666;font-weight:600;border-bottom:1px solid #ccc;'"
TD = "padding:6px;border-top:1px solid #eee;"


def _table(headers, rows, aligns=None):
    aligns = aligns or ["left"] * len(headers)
    th = "".join(f"<th {TH % a}>{h}</th>" for h, a in zip(headers, aligns))
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px;'><tr>{th}</tr>{''.join(rows)}</table>"


def _market_data(fetch_fn, problems):
    out = []
    for name, tk, kind in MARKET_ITEMS:
        try:
            dc = day_change(fetch_fn(tk, "1mo"))
        except Exception as e:
            dc = None
            problems.append(f"{name}: {e}")
        out.append((name, kind, dc))
    return out


def _market_table(mdata, today):
    rows = []
    for name, kind, dc in mdata:
        if dc is None:
            rows.append(f"<tr><td style='{TD}'>{_esc(name)}</td><td style='{TD}text-align:right;'>-</td>"
                        f"<td style='{TD}text-align:right;'>-</td><td style='{TD}'></td></tr>")
            continue
        d1, v1, pct, diff = dc
        if kind == "yield":
            chg, val = f"<span style='font-weight:600;'>{diff:+.2f}%p</span>", f"{v1:.2f}"
        else:
            chg = _pct_html(pct, "inverse" if kind == "vix" else ("neutral" if kind == "fx" else "normal"))
            val = f"{v1:,.2f}" if kind != "fx" else f"{v1:.4f}"
        note = vix_label(v1) if kind == "vix" else ""
        if _stale(d1, today):
            note = (note + " " if note else "") + f"<span style='color:{WARN};'>({d1} 이후 갱신 없음)</span>"
        rows.append(f"<tr><td style='{TD}'>{_esc(name)}</td><td style='{TD}text-align:right;'>{val}</td>"
                    f"<td style='{TD}text-align:right;'>{chg}</td><td style='{TD}font-size:12px;'>{note}</td></tr>")
    return _table(["지표", "종가", "전일 대비", ""], rows, ["left", "right", "right", "left"])


def _trend(m):
    if not m:
        return "-", MUTED, ""
    vd = tech.verdict(m)
    label, col = {"good": ("상승 추세", GOOD), "bad": ("하락 추세", BAD)}.get(vd, ("엇갈림", MUTED))
    rsi = m.get("rsi")
    flag = "과열" if (rsi is not None and rsi > 70) else ("침체" if (rsi is not None and rsi < 30) else "")
    return label, col, flag


def _stock_rows(stocks, data, today):
    """data[tk] = {series, events, earn}. 요약 표 행과 변동 dict 반환."""
    rows, changes = [], {}
    for tk in stocks:
        d = data[tk]
        dc = day_change(d["series"])
        if dc is None:
            rows.append(f"<tr><td style='{TD}'><strong>{_esc(tk)}</strong></td>"
                        f"<td colspan='4' style='{TD}color:{MUTED};'>가격을 가져오지 못했습니다</td></tr>")
            continue
        d1, v1, pct, _ = dc
        changes[tk] = pct
        m = tech.compute([v for _, v in d["series"]]) if d["series"] else None
        label, col, flag = _trend(m)
        r3 = f"{m['ret3m']:+.1f}%" if m and m.get("ret3m") is not None else "-"
        stale = f"<br><span style='color:{WARN};font-size:11px;'>({d1} 이후 갱신 없음)</span>" if _stale(d1, today) else ""
        e = d["events"].get("earnings")
        nxt = f"{e:%m/%d} ({_dday(e, today)})" if e and e >= today else "-"
        flag_t = f" <span style='color:{WARN};font-size:11px;'>RSI {flag}</span>" if flag else ""
        rows.append(f"<tr><td style='{TD}'><strong>{_esc(tk)}</strong>{stale}</td>"
                    f"<td style='{TD}text-align:right;'>{v1:,.2f}<br>{_pct_html(pct)}</td>"
                    f"<td style='{TD}text-align:right;'>{r3}</td>"
                    f"<td style='{TD}'><span style='color:{col};font-weight:600;'>{label}</span>{flag_t}</td>"
                    f"<td style='{TD}'>{nxt}</td></tr>")
    return rows, changes


def _earnings_table(stocks, data):
    rows = []
    for tk in stocks:
        d = data[tk]
        earn = d["earn"] or {}
        last, nxt = earn.get("last"), earn.get("next")
        if not last and not nxt:
            rows.append(f"<tr><td style='{TD}'><strong>{_esc(tk)}</strong></td><td colspan='2' style='{TD}color:{MUTED};'>실적 정보를 가져오지 못했습니다</td></tr>")
            continue
        lt = "-"
        if last:
            sur = last.get("surprise_pct")
            col = GOOD if (sur or 0) > 0 else (BAD if (sur or 0) < 0 else MUTED)
            word = "" if sur is None else (" 상회" if sur > 0 else (" 하회" if sur < 0 else " 부합"))
            est = f" / 예상 {last['estimate']:.2f}" if last.get("estimate") is not None else ""
            sur_t = f"<br><span style='color:{col};font-weight:600;'>{sur:+.1f}%{word}</span>" if sur is not None else ""
            rx = earnings_reaction(d["series"], last["date"])
            rx_t = f"<br>발표 후 주가 {_pct_html(rx)}" if rx is not None else ""
            lt = f"{last['date']:%m/%d} · EPS {last['actual']:.2f}{est}{sur_t}{rx_t}"
        nt = "-"
        if nxt:
            rng = f"<br>범위 {nxt['low']:.2f}~{nxt['high']:.2f}" if nxt.get("low") is not None and nxt.get("high") is not None else ""
            n = f", 애널리스트 {int(nxt['n'])}명" if nxt.get("n") else ""
            ya = ""
            if nxt.get("year_ago") is not None:
                gr = f" ({nxt['growth'] * 100:+.0f}%)" if nxt.get("growth") is not None else ""
                ya = f"<br>전년 동기 {nxt['year_ago']:.2f}{gr}"
            g = nxt.get("growth")
            if g is None and nxt.get("year_ago") and nxt["year_ago"] > 0 and nxt.get("avg") is not None:
                g = nxt["avg"] / nxt["year_ago"] - 1.0
            if g is None:
                gcol, glab = MUTED, "비교 불가"
            elif g >= EPS_BAND:
                gcol, glab = GOOD, "전년 대비 증가"
            elif g <= -EPS_BAND:
                gcol, glab = BAD, "전년 대비 감소"
            else:
                gcol, glab = MUTED, "전년 비슷"
            nt = (f"<span style='color:{gcol};font-weight:700;'>평균 {nxt['avg']:.2f} ({glab})</span>{rng}{n}{ya}")
        rows.append(f"<tr><td style='{TD}vertical-align:top;'><strong>{_esc(tk)}</strong></td>"
                    f"<td style='{TD}font-size:12px;'>{lt}</td><td style='{TD}font-size:12px;'>{nt}</td></tr>")
    return (_table(["종목", "지난 실적 (EPS 실제 / 예상)", "이번 분기 예상 EPS"], rows)
            + f"<p style='font-size:11px;color:{MUTED};margin:4px 0 0;'>※ 예상 EPS는 애널리스트 집계 평균이며 제 예측이 아닙니다. 색: 전년 동기 대비 ±5% 이상 증가 녹색 · 감소 빨강 · 그 안쪽/비교 불가 회색(5%는 이 코드의 설명용 기준).</p>")


def _calendar_block(all_events, today):
    items = []
    end = today + timedelta(days=HORIZON_DAYS)
    for tk, ev in all_events.items():
        for key, label in (("earnings", "실적 발표"), ("ex_div", "배당락일")):
            d = ev.get(key)
            if d and today <= d <= end:
                items.append((d, tk, label))
    items.sort()
    if not items:
        return f"<p style='color:{MUTED};font-size:13px;'>앞으로 {HORIZON_DAYS}일 안에 확인된 일정이 없습니다.</p>"
    rows = [f"<tr><td style='{TD}'>{d:%m/%d (%a)}</td><td style='{TD}'>{_dday(d, today)}</td>"
            f"<td style='{TD}'><strong>{_esc(tk)}</strong></td><td style='{TD}'>{label}</td></tr>" for d, tk, label in items]
    return _table(["날짜", "남은 날", "종목", "내용"], rows)


def _key_points(mdata, changes, all_events, today, mover_pct):
    """오늘 핵심 요약(규칙 기반, AI 없음). 줄마다 사실만."""
    pts = []
    idx = [(n, dc[2]) for n, k, dc in mdata if k == "idx" and dc]
    if idx:
        pts.append("미국 증시: " + " · ".join(f"{n} {p:+.2f}%" for n, p in idx))
    for n, k, dc in mdata:
        if k == "vix" and dc:
            pts.append(f"VIX(공포지수) {dc[1]:.1f} — {_vix_text(dc[1])}")
    mv = mn.movers(changes, mover_pct)
    if mv:
        pts.append("큰 변동 보유 종목: " + ", ".join(f"{t} {changes[t]:+.1f}%" for t in mv[:5]))
    soon = sorted((ev["earnings"], tk) for tk, ev in all_events.items() if ev.get("earnings") and 0 <= (ev["earnings"] - today).days <= 7)
    if soon:
        pts.append("이번 주 실적 발표: " + ", ".join(f"{tk} {d:%m/%d}({_dday(d, today)})" for d, tk in soon))
    exd = sorted((ev["ex_div"], tk) for tk, ev in all_events.items() if ev.get("ex_div") and 0 <= (ev["ex_div"] - today).days <= 7)
    if exd:
        pts.append("이번 주 배당락: " + ", ".join(f"{tk} {d:%m/%d}({_dday(d, today)})" for d, tk in exd))
    if not pts:
        return ""
    lis = "".join(f"<li style='margin:3px 0;'>{_esc(p)}</li>" for p in pts)
    return (f"<div style='background:#f3f7ff;border:1px solid #d6e2f7;border-radius:8px;padding:8px 12px;margin:10px 0;'>"
            f"<div style='font-weight:700;font-size:15px;'>⭐ 오늘 핵심</div><ul style='margin:4px 0 0;padding-left:18px;font-size:14px;'>{lis}</ul></div>")


def build_market_report(stocks, fetch_fn=None, events_fn=None, today=None, ai_html="", earnings_fn=None,
                        mover_news_fn=None, market_news_html="", mover_pct=None):
    """반환: (html 본문 조각, problems). 예외를 던지지 않는다.
    mover_news_fn(ticker)->[{title,url,source,published}]  / market_news_html: 미리 만든 '시장 뉴스' 조각."""
    fetch_fn = fetch_fn or fetch_series
    events_fn = events_fn or fetch_events
    earnings_fn = earnings_fn or fetch_earnings
    today = today or date.today()
    problems = []
    if mover_pct is None:
        try:
            mover_pct = float(os.environ.get("MOVER_PCT", mn.MOVER_PCT))
        except ValueError:
            mover_pct = mn.MOVER_PCT
    try:
        mdata = _market_data(fetch_fn, problems)
        market = _market_table(mdata, today)
    except Exception as e:
        mdata, market = [], f"<p style='color:{MUTED};'>시황을 만들지 못했습니다: {_esc(e)}</p>"
    data = {}
    for tk in stocks:
        try:
            series = fetch_fn(tk, "2y")
        except Exception as e:
            series = []
            problems.append(f"{tk}: 가격 {e}")
        try:
            ev = events_fn(tk) or {}
        except Exception as e:
            ev = {}
            problems.append(f"{tk}: 일정 {e}")
        try:
            earn = earnings_fn(tk) or {}
        except Exception as e:
            earn = {}
            problems.append(f"{tk}: 실적 {e}")
        data[tk] = {"series": series, "events": ev, "earn": earn}
    h2 = "style='margin:20px 0 6px;font-size:16px;'"
    if stocks:
        try:
            srows, changes = _stock_rows(stocks, data, today)
            summary = _table(["종목", "종가 / 전일", "3개월", "추세", "다음 실적"], srows, ["left", "right", "right", "left", "left"])
            summary += (f"<p style='font-size:11px;color:{MUTED};margin:4px 0 0;'>※ 추세: 현재가·50일선·200일선 위치로 본 현재 상태이며 앞날 예측이 아닙니다. "
                        f"RSI 과열/침체는 단기 과열(70↑)·침체(30↓) 표시입니다.</p>")
            earn_tbl = _earnings_table(stocks, data)
        except Exception as e:
            changes, summary, earn_tbl = {}, f"<p style='color:{MUTED};'>종목 표를 만들지 못했습니다: {_esc(e)}</p>", ""
            problems.append(f"종목 표 {e}")
    else:
        changes = {}
        summary = f"<p style='color:{MUTED};font-size:13px;'>표시할 보유 개별주가 없습니다.</p>"
        earn_tbl = ""
    all_events = {tk: d["events"] for tk, d in data.items()}
    try:
        key = _key_points(mdata, changes, all_events, today, mover_pct)
    except Exception as e:
        key = ""
        problems.append(f"핵심 요약 {e}")
    mover_html = ""
    if stocks and mover_news_fn is not None:
        try:
            mover_html = f"<h2 {h2}>🔥 큰 변동 종목 뉴스</h2>" + mn.mover_news_html(changes, mover_news_fn, mover_pct)
        except Exception as e:
            problems.append(f"변동 종목 뉴스 {e}")
    html = (
        key
        + f"<h2 {h2}>🌎 어젯밤 시황</h2>{market}"
        + (f"<h2 {h2}>📰 오늘의 시장 뉴스</h2>{market_news_html}" if market_news_html else "")
        + f"<h2 {h2}>🔎 내 개별주</h2>{summary}"
        + mover_html
        + (f"<h2 {h2}>📊 실적 (지난 결과 · 이번 분기 예상)</h2>{earn_tbl}" if earn_tbl else "")
        + f"<h2 {h2}>📅 앞으로 {HORIZON_DAYS}일 일정</h2>{_calendar_block(all_events, today)}"
        + (f"<h2 {h2}>🧠 AI 브리핑 (세계 경제)</h2>{ai_html}" if ai_html else "")
        + f"<p style='font-size:11px;color:{MUTED};margin-top:14px;'>※ 가격·일정·실적은 yfinance(비공식)에서 가져온 값이며 틀리거나 늦을 수 있고, "
          f"실적일은 회사가 바꿀 수 있습니다. 뉴스는 헤드라인 검색 결과이고 원인을 확정한 것이 아닙니다. VIX 20 미만 안정·30 초과 변동성 큼은 통상 기준입니다. "
          f"투자 판단의 단독 근거로 쓰지 마세요.</p>")
    if problems:
        html += (f"<p style='font-size:11px;color:{WARN};'>가져오지 못한 항목: " + _esc("; ".join(problems[:8])) + "</p>")
    return html, problems


def wrap_email(body_html, today=None):
    today = today or date.today()
    return ("<html><head><meta charset='utf-8'></head><body style='font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;padding:12px;color:#222;'>"
            f"<h1 style='font-size:20px;margin:0 0 4px;'>🌅 시장·종목 리포트</h1>"
            f"<div style='font-size:12px;color:{MUTED};'>{today:%Y-%m-%d (%a)} · 포트폴리오 리포트와는 별도 메일입니다</div>"
            f"{body_html}</body></html>")
'''
_register_embedded('market_report', EMBEDDED_MARKET_REPORT)

# ---------- 메인 스크립트 ----------
# -*- coding: utf-8 -*-
"""
Market_Report ver 1 - B 리포트(시장·종목)
  - 포트폴리오 리포트(Portfolio_Report_ver_4_1.py)와 별도 메일/별도 워크플로로 돌아간다.
  - 같은 저장소에 Portfolio_Report_ver_4_1.py가 있어야 한다(시트 읽기·메일 보내기를 그대로 가져다 쓴다).
  - 내용: 오늘 핵심 → 시황 → 시장 뉴스 → 내 개별주 표(ETF 제외) → 큰 변동 종목 뉴스 → 실적 표 → 14일 일정 → AI 브리핑.
  - ENABLE_AI_BRIEFING=0 이면 세계 경제 AI 브리핑을, ENABLE_MARKET_NEWS=0 이면 뉴스 섹션 전체를 끈다.
  - ANTHROPIC_API_KEY가 없으면 시장 뉴스는 AI 선별 없이 최신 헤드라인만 보여준다.
  - Settings 탭 MarketExclude(쉼표 구분)에 적은 티커는 개별주 목록에서 뺀다. ETF는 자동 제외(yfinance quoteType, 못 알아내면 개별주로 취급).
"""
import os
from datetime import datetime

import pandas as pd

import Portfolio_Report_ver_4_1 as A
import market_news as mn
from market_report import build_market_report, pick_stocks, wrap_email, is_etf


def main():
    df_hold, *_ = A.load_portfolio_from_gsheet()

    _cache = {}

    def _open_sheet():
        if "sh" not in _cache:
            _cache["sh"] = A.open_gsheet(os.environ.get("GSHEET_ID"))
        return _cache["sh"]

    inc_cfg = {"income_tickers": ["SCHD"], "asof": None}
    try:
        inc_cfg = A.read_income_settings(_open_sheet())
    except Exception as e:
        print(f"⚠️ Settings(IncomeTickers) 읽기 실패(기본값 사용): {e}")
    try:
        trade_rows, trade_warn = A.load_trades(_open_sheet)
        df_hold, _info = A.apply_trades(df_hold, trade_rows, usd_cad=A.get_usd_cad_rate(), asof=inc_cfg["asof"])
    except Exception as e:
        print(f"⚠️ Trades 반영 실패(기준 Holdings로 계속): {e}")

    exclude = []
    try:
        for row in _open_sheet().worksheet("Settings").get_all_values()[1:]:
            if len(row) >= 2 and str(row[0]).strip().lower() == "marketexclude":
                exclude = [t.strip().upper() for t in str(row[1]).replace(";", ",").split(",") if t.strip()]
    except Exception as e:
        print(f"⚠️ Settings(MarketExclude) 읽기 실패(무시): {e}")
    stocks = pick_stocks(df_hold, inc_cfg.get("income_tickers"), exclude=exclude, etf_fn=is_etf)
    print("개별주:", stocks)
    ai_html = ""
    if os.environ.get("ENABLE_AI_BRIEFING", "1") == "1":
        try:
            ai_html = A.build_ai_briefing(
                held_tickers=stocks, portfolio_facts=[], sim_facts=[],
                open_sheet_fn=_open_sheet,
                news_fn=lambda t: A._fetch_news_for_ticker_midterm(t, os.environ.get("NEWS_API_KEY"), page_size=3, days=7),
                fred_key=os.environ.get("FRED_API_KEY"),
            )
        except Exception as e:
            ai_html = f"<p style='color:#757575;'>AI 브리핑을 만들지 못했습니다: {e}</p>"
    market_news_html, mover_fn = "", None
    if os.environ.get("ENABLE_MARKET_NEWS", "1") == "1":
        news_key = os.environ.get("NEWS_API_KEY")
        mover_fn = lambda t: mn.fetch_headlines(f"{t} stock", news_key, page_size=3, days=3)
        try:
            heads = mn.fetch_headlines(mn.MARKET_QUERY, news_key, page_size=mn.MAX_FETCH, days=2)
            call_fn = None
            if os.environ.get("ANTHROPIC_API_KEY"):
                call_fn = lambda sys_, usr, tool: A.call_claude(sys_, usr, max_tokens=2000, tool=tool)
            market_news_html = mn.market_news_html(heads, call_fn)
        except Exception as e:
            market_news_html = f"<p style='color:#757575;'>시장 뉴스를 만들지 못했습니다: {e}</p>"
    body, problems = build_market_report(stocks, ai_html=ai_html, mover_news_fn=mover_fn, market_news_html=market_news_html)
    html_doc = wrap_email(body)

    outname = f"market_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")
    A.send_email_html(f"🌅 시장·종목 리포트 - {datetime.now().strftime('%Y-%m-%d')}", html_doc)


if __name__ == "__main__":
    main()
