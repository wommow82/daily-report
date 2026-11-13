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

# ---------- EMBEDDED_MARKET_REPORT ----------
EMBEDDED_MARKET_REPORT = r'''# -*- coding: utf-8 -*-
"""
market_report.py - B 리포트(시장·종목): 어젯밤 시황, 보유 개별주 카드, 앞으로 2주 일정

원칙
  - 1단계는 AI를 쓰지 않는다(계산·표시만). 외부 값은 가져온 그대로 보여주고 해석은 규칙(기술지표)만 쓴다.
  - 가져오지 못한 값은 '-'로 두고 이유를 아래에 적는다. 예외를 던지지 않는다.
  - 가격 데이터: yfinance(비공식, 형식이 바뀔 수 있음). 일정: yfinance의 calendar / get_earnings_dates.
    https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html
  - VIX 기준(20 미만 안정, 30 초과 변동성 큼): Robinhood의 VIX 설명(앞선 대화에서 확인한 통상 기준)
"""
import html as _html
from datetime import date, datetime, timedelta

import technicals as tech

HORIZON_DAYS = 14          # 일정 표시 범위
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

def pick_stocks(df_hold, income_tickers):
    """보유 수량이 있는 TFSA 개별주(IncomeTickers 제외)를 평가액 큰 순서로. RESP는 제외."""
    import pandas as pd
    inc = {str(t).strip().upper() for t in (income_tickers or [])}
    d = df_hold.copy()
    d["_sh"] = pd.to_numeric(d["Shares"], errors="coerce").fillna(0)
    d = d[(d["_sh"] > 0) & (d["Type"].astype(str).str.upper() == "TFSA")]
    if "PositionValueBase" in d.columns:
        d = d.sort_values("PositionValueBase", ascending=False)
    out = []
    for t in d["Ticker"].astype(str):
        t = t.strip()
        if t and t.upper() not in inc and t not in out:
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


# =========================
# 렌더
# =========================

def _market_table(fetch_fn, today, problems):
    rows = ""
    for name, tk, kind in MARKET_ITEMS:
        try:
            dc = day_change(fetch_fn(tk, "1mo"))
        except Exception as e:
            dc = None
            problems.append(f"{name}: {e}")
        if dc is None:
            rows += (f"<tr><td style='padding:4px 6px;'>{_esc(name)}</td><td style='text-align:right;'>-</td>"
                     f"<td style='text-align:right;'>-</td><td></td></tr>")
            continue
        d1, v1, pct, diff = dc
        if kind == "yield":
            chg = f"<span style='font-weight:600;'>{diff:+.2f}%p</span>"          # 금리·환율은 오르내림이 좋고 나쁨을 뜻하지 않아 색 없음
            val = f"{v1:.2f}"
        else:
            chg = _pct_html(pct, "inverse" if kind == "vix" else ("neutral" if kind == "fx" else "normal"))
            val = f"{v1:,.2f}" if kind != "fx" else f"{v1:.4f}"
        note = vix_label(v1) if kind == "vix" else ""
        if _stale(d1, today):
            note = (note + " " if note else "") + f"<span style='color:{WARN};'>({d1} 이후 갱신 없음)</span>"
        rows += (f"<tr><td style='padding:4px 6px;text-align:left;'>{_esc(name)}</td>"
                 f"<td style='text-align:right;'>{val}</td><td style='text-align:right;'>{chg}</td>"
                 f"<td style='text-align:left;padding-left:8px;font-size:11px;'>{note}</td></tr>")
    return ("<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<tr style='border-bottom:1px solid #ddd;color:#666;'><th style='text-align:left;padding:4px 6px;'>지표</th>"
            "<th style='text-align:right;'>최근 종가</th><th style='text-align:right;'>전일 대비</th><th></th></tr>"
            + rows + "</table>")


def _dday(d, today):
    n = (d - today).days
    return "오늘" if n == 0 else (f"D-{n}" if n > 0 else f"D+{abs(n)}")


def _stock_card(tk, series, events, today, earn=None):
    dc = day_change(series)
    m = tech.compute([v for _, v in series]) if series else None
    if dc is None:
        head = f"<strong>{_esc(tk)}</strong> <span style='color:{MUTED};'>가격을 가져오지 못했습니다</span>"
        return f"<div style='border-top:1px solid #eee;padding:8px 0;'>{head}</div>"
    d1, v1, pct, _ = dc
    stale = f" <span style='color:{WARN};font-size:11px;'>({d1} 이후 갱신 없음)</span>" if _stale(d1, today) else ""
    head = (f"<strong style='font-size:14px;'>{_esc(tk)}</strong> &nbsp;{v1:,.2f} &nbsp;{_pct_html(pct)}{stale}")
    body = ""
    if m:
        vd = tech.verdict(m)
        col = {"good": GOOD, "bad": BAD}.get(vd, MUTED)
        label = {"good": "긍정 신호", "bad": "부정 신호"}.get(vd, "신호 혼재")
        rsi = m.get("rsi")
        extra = []
        if rsi is not None:
            extra.append(f"RSI {rsi:.0f}")
        if m.get("from_hi") is not None:
            extra.append(f"52주 고점 대비 {m['from_hi']:+.1f}%")
        if m.get("ret3m") is not None:
            extra.append(f"3개월 {m['ret3m']:+.1f}%")
        body += (f"<div style='font-size:12px;color:#555;margin-top:2px;'>{' · '.join(extra)}</div>"
                 f"<div style='font-size:12px;color:{col};font-weight:600;margin-top:2px;'>→ [{label}] {_esc(tech.interpret(m))}</div>")
    else:
        body += f"<div style='font-size:12px;color:{MUTED};'>기술지표를 계산할 가격 데이터가 부족합니다.</div>"
    ev = []
    if events.get("earnings"):
        ev.append(f"실적 발표 {events['earnings']:%m/%d}({_dday(events['earnings'], today)})")
    if events.get("ex_div"):
        ev.append(f"배당락 {events['ex_div']:%m/%d}({_dday(events['ex_div'], today)})")
    body += (f"<div style='font-size:12px;margin-top:2px;'>📅 {' · '.join(ev)}</div>" if ev
             else f"<div style='font-size:12px;color:{MUTED};margin-top:2px;'>📅 확인된 예정 일정 없음</div>")
    body += _earnings_lines(series, earn or {})
    return f"<div style='border-top:1px solid #eee;padding:8px 0;'>{head}{body}</div>"


def _earnings_lines(series, earn):
    out = ""
    last, nxt = earn.get("last"), earn.get("next")
    if last:
        sur = last.get("surprise_pct")
        col = GOOD if (sur or 0) > 0 else (BAD if (sur or 0) < 0 else MUTED)
        word = "" if sur is None else (" 상회" if sur > 0 else (" 하회" if sur < 0 else " 부합"))
        est = f" vs 예상 {last['estimate']:.2f}" if last.get("estimate") is not None else ""
        sur_t = f" <span style='color:{col};font-weight:600;'>({sur:+.1f}%{word})</span>" if sur is not None else ""
        rx = earnings_reaction(series, last["date"])
        rx_t = f" · 발표 후 주가 {_pct_html(rx)}" if rx is not None else ""
        out += (f"<div style='font-size:12px;margin-top:2px;'>📊 지난 실적({last['date']:%m/%d}): "
                f"EPS {last['actual']:.2f}{est}{sur_t}{rx_t}</div>")
    if nxt:
        rng = (f", 범위 {nxt['low']:.2f}~{nxt['high']:.2f}" if nxt.get("low") is not None and nxt.get("high") is not None else "")
        n = f", 애널리스트 {int(nxt['n'])}명" if nxt.get("n") else ""
        ya = ""
        if nxt.get("year_ago") is not None:
            gr = f" ({nxt['growth'] * 100:+.0f}%)" if nxt.get("growth") is not None else ""
            ya = f", 전년 동기 {nxt['year_ago']:.2f}{gr}"
        out += (f"<div style='font-size:12px;margin-top:2px;color:#444;'>🔮 이번 분기 예상 EPS 평균 {nxt['avg']:.2f}{rng}{n}{ya}"
                f" <span style='color:{MUTED};'>(애널리스트 집계, 제 예측 아님)</span></div>")
    if not last and not nxt:
        out += f"<div style='font-size:12px;color:{MUTED};margin-top:2px;'>📊 실적 정보를 가져오지 못했습니다(ETF이거나 데이터 없음)</div>"
    return out


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
    rows = "".join(
        f"<tr><td style='padding:3px 6px;'>{d:%m/%d (%a)}</td><td style='padding:3px 6px;'>{_dday(d, today)}</td>"
        f"<td style='padding:3px 6px;'><strong>{_esc(tk)}</strong></td><td style='padding:3px 6px;'>{label}</td></tr>"
        for d, tk, label in items)
    return f"<table style='border-collapse:collapse;font-size:13px;'>{rows}</table>"


def build_market_report(stocks, fetch_fn=None, events_fn=None, today=None, ai_html="", earnings_fn=None):
    """반환: (html 본문 조각, problems). 예외를 던지지 않는다."""
    fetch_fn = fetch_fn or fetch_series
    events_fn = events_fn or fetch_events
    earnings_fn = earnings_fn or fetch_earnings
    today = today or date.today()
    problems = []
    try:
        market = _market_table(fetch_fn, today, problems)
    except Exception as e:
        market = f"<p style='color:{MUTED};'>시황을 만들지 못했습니다: {_esc(e)}</p>"
    cards, all_events = "", {}
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
        all_events[tk] = ev
        try:
            earn = earnings_fn(tk) or {}
        except Exception as e:
            earn = {}
            problems.append(f"{tk}: 실적 {e}")
        try:
            cards += _stock_card(tk, series, ev, today, earn)
        except Exception as e:
            problems.append(f"{tk}: 카드 {e}")
    if not stocks:
        cards = f"<p style='color:{MUTED};font-size:13px;'>표시할 보유 개별주가 없습니다.</p>"
    h2 = "style='margin:18px 0 6px;font-size:16px;'"
    html = (
        f"<h2 {h2}>🌎 어젯밤 시황</h2>{market}"
        + (f"<h2 {h2}>🧠 AI 브리핑 (세계 경제)</h2>{ai_html}" if ai_html else "") +
        f"<h2 {h2}>📅 앞으로 {HORIZON_DAYS}일 일정 (보유 개별주)</h2>{_calendar_block(all_events, today)}"
        f"<h2 {h2}>🔎 내 개별주</h2>{cards}"
        f"<p style='font-size:11px;color:{MUTED};margin-top:14px;'>※ 가격·일정·실적은 yfinance(비공식)에서 가져온 값이며 틀리거나 늦을 수 있고, "
        f"실적일은 회사가 바꿀 수 있고, '예상 EPS'는 애널리스트 집계값이지 제 예측이 아닙니다. 기술지표는 현재 상태를 요약한 것이지 앞날의 예측이 아닙니다. "
        f"VIX 20 미만 안정·30 초과 변동성 큼은 통상 기준입니다. 투자 판단의 단독 근거로 쓰지 마세요.</p>")
    if problems:
        html += (f"<p style='font-size:11px;color:{WARN};'>가져오지 못한 항목: "
                 + _esc("; ".join(problems[:8])) + "</p>")
    return html, problems


def wrap_email(body_html, today=None):
    today = today or date.today()
    return ("<html><body style='font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:720px;margin:0 auto;padding:12px;color:#222;'>"
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
  - 내용: 시황 + AI 브리핑(하루 1회) + 14일 일정 + 개별주 카드(기술지표, 지난 실적, 다음 분기 예상 EPS).
  - ENABLE_AI_BRIEFING=0 이면 AI 호출 없이(비용 0) 나머지만 보낸다.
"""
import os
from datetime import datetime

import pandas as pd

import Portfolio_Report_ver_4_1 as A
from market_report import build_market_report, pick_stocks, wrap_email


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

    stocks = pick_stocks(df_hold, inc_cfg.get("income_tickers"))
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
    body, problems = build_market_report(stocks, ai_html=ai_html)
    html_doc = wrap_email(body)

    outname = f"market_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(outname, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Report saved: {outname}")
    A.send_email_html(f"🌅 시장·종목 리포트 - {datetime.now().strftime('%Y-%m-%d')}", html_doc)


if __name__ == "__main__":
    main()
