from __future__ import annotations
import os
import json
import smtplib
import datetime as dt
from typing import Dict, Tuple, List, Optional
import pandas as pd
import gspread
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

# ======================
# 1) Environment
# ======================

def load_env() -> Dict[str, object]:
    """Load env vars (supports local .env via python-dotenv).

    Required:
      - IDENT_SHEET_ID
      - GSPREAD_SERVICE_ACCOUNT_JSON
      - SMTP_HOST, SMTP_PORT
      - SMTP_USER, SMTP_PASSWORD
      - IDENT_SENDER_EMAIL
      - IDENT_ALERT_RECIPIENTS  (comma/semicolon separated)
      - ANTHROPIC_API_KEY       ← NEW: Claude Haiku for AI commentary

    Fallbacks (existing Secrets):
      - EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
    """
    load_dotenv()

    spreadsheet_id = os.getenv("IDENT_SHEET_ID") or os.getenv("GSHEET_ID")
    worksheet_name = os.getenv("IDENT_WORKSHEET_NAME", "Sheet1")
    tz_name        = os.getenv("IDENT_TIMEZONE", "America/Edmonton")

    smtp_host     = os.getenv("SMTP_HOST")
    smtp_port     = int(os.getenv("SMTP_PORT", "587"))
    smtp_user     = os.getenv("SMTP_USER")     or os.getenv("EMAIL_SENDER")
    smtp_password = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASSWORD")
    sender_email  = os.getenv("IDENT_SENDER_EMAIL") or os.getenv("EMAIL_SENDER")
    alert_recipients = (
        os.getenv("IDENT_ALERT_RECIPIENTS")
        or os.getenv("ALERT_RECIPIENTS")
        or os.getenv("EMAIL_RECEIVER")
    )
    sa_json        = os.getenv("GSPREAD_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    anthropic_key  = os.getenv("ANTHROPIC_API_KEY")  # Optional but recommended

    cfg: Dict[str, object] = {
        "spreadsheet_id":   spreadsheet_id,
        "worksheet_name":   worksheet_name,
        "timezone":         tz_name,
        "smtp_host":        smtp_host,
        "smtp_port":        smtp_port,
        "smtp_user":        smtp_user,
        "smtp_password":    smtp_password,
        "sender_email":     sender_email,
        "alert_recipients": alert_recipients,
        "sa_json":          sa_json,
        "anthropic_key":    anthropic_key,
    }

    required = [
        "spreadsheet_id", "smtp_host", "smtp_user",
        "smtp_password", "sender_email", "alert_recipients", "sa_json",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    return cfg


def get_today_local(tz_name: str) -> dt.date:
    if ZoneInfo is None:
        return dt.date.today()
    try:
        return dt.datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return dt.date.today()


# ======================
# 2) Google Sheets Auth
# ======================

def get_gsheet_client(sa_json: Optional[str] = None) -> gspread.client.Client:
    if sa_json:
        info = json.loads(sa_json)
        return gspread.service_account_from_dict(info)
    return gspread.service_account()


# ======================
# 3) Sheet → DataFrame
# ======================

EXPECTED_COLUMNS = [
    "PersonName", "IDType", "Country",
    "ExpiryDate", "AlertDaysBefore", "Active", "LastAlertDate",
]


def load_identifications_df(
    spreadsheet_id: str,
    worksheet_name: str,
    client: gspread.client.Client,
) -> Tuple[pd.DataFrame, gspread.Worksheet]:
    sh   = client.open_by_key(spreadsheet_id)
    ws   = sh.worksheet(worksheet_name)
    rows = ws.get_all_values()
    if not rows:
        raise RuntimeError("Sheet is empty")

    header    = rows[0]
    data_rows = rows[1:]
    records   = []
    for idx, row in enumerate(data_rows, start=2):
        rec = {col: (row[i] if i < len(row) else "") for i, col in enumerate(header)}
        rec["RowNumber"] = idx
        records.append(rec)

    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["ExpiryDate"]     = pd.to_datetime(df["ExpiryDate"], errors="coerce").dt.date
    df["AlertDaysBefore"] = pd.to_numeric(df["AlertDaysBefore"], errors="coerce").fillna(0).astype(int)
    df["Active"]          = (
        df["Active"].astype(str).str.strip().str.upper()
        .isin(["TRUE", "1", "Y", "YES"])
    )
    df["LastAlertDate"]  = pd.to_datetime(df["LastAlertDate"], errors="coerce").dt.date
    return df, ws


# ======================
# 4) Alert Logic
# ======================

def find_ids_to_alert(df_ids: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    df = df_ids.copy()
    df = df[df["Active"] & df["ExpiryDate"].notna()].copy()
    if df.empty:
        return df
    df["DaysToExpiry"]    = (df["ExpiryDate"] - today).apply(lambda x: x.days)
    cond_in_window        = (df["DaysToExpiry"] >= 0) & (df["DaysToExpiry"] <= df["AlertDaysBefore"])
    cond_not_alerted_today = df["LastAlertDate"].isna() | (df["LastAlertDate"] < today)
    return df[cond_in_window & cond_not_alerted_today].copy()


# ======================
# 5) Claude Haiku – AI 코멘트 생성 (NEW)
# ======================

def generate_claude_commentary(df_alerts: pd.DataFrame, today: dt.date, api_key: str) -> str:
    """
    Claude Haiku를 사용해 만료 임박 신분증 목록에 대한
    간결한 한국어 갱신 조언을 생성합니다.
    API key가 없으면 빈 문자열을 반환합니다.
    """
    if not api_key:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # 프롬프트용 데이터 정리
        items = []
        for _, r in df_alerts.iterrows():
            items.append(
                f"- {r.get('PersonName','?')} / {r.get('IDType','?')} ({r.get('Country','?')}) "
                f"→ 만료까지 {int(r.get('DaysToExpiry', 0))}일 (만료일: {r.get('ExpiryDate','')})"
            )
        items_str = "\n".join(items)

        prompt = f"""다음은 오늘({today}) 기준으로 만료가 임박한 가족 신분증 목록입니다:

{items_str}

위 목록을 바탕으로:
1. 가장 급한 항목 1~2개를 강조해 주세요.
2. 각 신분증 종류(여권, 운전면허, 영주권 등)에 맞는 실용적인 갱신 조언을 한두 문장으로 주세요.
3. 전체 요약을 3~5문장의 자연스러운 한국어로 작성해 주세요.
4. 친절하지만 간결하게, HTML 태그 없이 순수 텍스트로만 답하세요."""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    except Exception as e:
        print(f"[Claude commentary skipped] {e}")
        return ""


# ======================
# 6) Email HTML Builder (REDESIGNED)
# ======================

def _esc(s: object) -> str:
    t = "" if s is None else str(s)
    return (t.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))


def _classify_urgency(days: int, alert_before: int) -> str:
    if days <= 0:
        return "imminent"
    if not alert_before or alert_before <= 0:
        return "imminent" if days <= 7 else ("caution" if days <= 30 else "ok")
    ratio = days / float(alert_before)
    return "imminent" if ratio <= 0.25 else ("caution" if ratio <= 0.60 else "ok")


def build_all_alerts_html(
    df_alerts: pd.DataFrame,
    today: dt.date,
    ai_commentary: str = "",
) -> str:
    """
    모던 카드 레이아웃 이메일 HTML.
    - 각 신분증마다 시각적 진행바(남은 기간 비율)
    - 상태별 색상 강조
    - Claude AI 코멘트 섹션 (있을 경우)
    """
    today_str = today.strftime("%Y년 %m월 %d일")

    df = df_alerts.copy()
    if "DaysToExpiry" not in df.columns:
        df["DaysToExpiry"] = (df["ExpiryDate"] - today).apply(lambda x: x.days)

    for c in ["PersonName", "IDType", "Country", "ExpiryDate", "DaysToExpiry"]:
        if c not in df.columns:
            df[c] = ""

    try:
        df = df.sort_values(["DaysToExpiry", "PersonName"], ascending=[True, True])
    except Exception:
        pass

    # ── 카운트 집계 ──────────────────────────────────────────
    cnt = {"imminent": 0, "caution": 0, "ok": 0}
    for _, r in df.iterrows():
        d  = int(r.get("DaysToExpiry", 0))
        ab = int(r.get("AlertDaysBefore", 0)) if r.get("AlertDaysBefore") else 0
        cnt[_classify_urgency(d, ab)] += 1

    # ── 카드 HTML 생성 ────────────────────────────────────────
    PALETTE = {
        "imminent": {"bg": "#FFF1F0", "border": "#FF4D4F", "bar": "#FF4D4F",
                     "badge_bg": "#FF4D4F", "badge_text": "#fff",
                     "label": "⛔ 임박", "icon": "🔴"},
        "caution":  {"bg": "#FFFBE6", "border": "#FA8C16", "bar": "#FA8C16",
                     "badge_bg": "#FA8C16", "badge_text": "#fff",
                     "label": "⚠️ 주의", "icon": "🟡"},
        "ok":       {"bg": "#F6FFED", "border": "#52C41A", "bar": "#52C41A",
                     "badge_bg": "#52C41A", "badge_text": "#fff",
                     "label": "✅ 여유", "icon": "🟢"},
    }

    cards_html: List[str] = []
    for _, r in df.iterrows():
        name       = _esc(str(r.get("PersonName", "")).strip() or "-")
        idtype     = _esc(str(r.get("IDType", "")).strip()     or "-")
        country    = _esc(str(r.get("Country", "")).strip()    or "-")
        expiry_str = _esc(str(r.get("ExpiryDate", "")).strip() or "-")
        days_int   = int(r.get("DaysToExpiry", 0))
        ab         = int(r.get("AlertDaysBefore", 0)) if r.get("AlertDaysBefore") else 0

        urg  = _classify_urgency(days_int, ab)
        pal  = PALETTE[urg]

        # 진행바: ab 대비 days 비율 (0~100%)
        pct  = min(100, max(0, round(days_int / ab * 100))) if ab > 0 else (100 if days_int > 30 else (days_int * 100 // 30))
        bar_label = f"{days_int}일 남음"

        cards_html.append(f"""
        <div style="margin-bottom:14px; border:1.5px solid {pal['border']}; border-radius:12px;
                    background:{pal['bg']}; overflow:hidden;">

          <!-- 카드 헤더 -->
          <div style="display:flex; align-items:center; justify-content:space-between;
                      padding:12px 16px; border-bottom:1px solid {pal['border']}30;">
            <div>
              <span style="font-size:17px; font-weight:800; color:#1a1a2e;">{pal['icon']} {name}</span>
              <span style="margin-left:10px; font-size:13px; color:#555;">{idtype}</span>
              <span style="margin-left:6px; font-size:12px; color:#888;">({country})</span>
            </div>
            <span style="padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700;
                         background:{pal['badge_bg']}; color:{pal['badge_text']};">{pal['label']}</span>
          </div>

          <!-- 카드 바디 -->
          <div style="padding:12px 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
              <span style="font-size:13px; color:#555;">만료일</span>
              <span style="font-size:14px; font-weight:700; color:#1a1a2e;">{expiry_str}</span>
            </div>

            <!-- 진행바 -->
            <div style="background:#e8e8e8; border-radius:999px; height:10px; overflow:hidden; margin-bottom:6px;">
              <div style="width:{pct}%; height:100%; background:{pal['bar']}; border-radius:999px;
                          transition:width 0.4s ease;"></div>
            </div>
            <div style="font-size:11px; color:#888; text-align:right;">{bar_label}</div>
          </div>
        </div>
        """)

    # ── AI 코멘트 섹션 ────────────────────────────────────────
    ai_section = ""
    if ai_commentary:
        ai_section = f"""
        <div style="margin-top:20px; padding:16px 18px; background:#EFF6FF;
                    border-left:4px solid #3B82F6; border-radius:8px;">
          <div style="font-size:13px; font-weight:700; color:#1D4ED8; margin-bottom:8px;">
            🤖 Claude AI 갱신 조언
          </div>
          <div style="font-size:13px; color:#1e3a5f; line-height:1.7; white-space:pre-line;">{_esc(ai_commentary)}</div>
        </div>
        """

    # ── 전체 HTML ─────────────────────────────────────────────
    html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:0; background:#F0F2F5; font-family:'Malgun Gothic', '맑은 고딕', Apple SD Gothic Neo, sans-serif;">

<div style="max-width:600px; margin:32px auto; padding:0 16px 32px;">

  <!-- 헤더 배너 -->
  <div style="background:linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
              border-radius:16px 16px 0 0; padding:28px 28px 20px; color:#fff;">
    <div style="font-size:11px; letter-spacing:3px; text-transform:uppercase;
                opacity:0.6; margin-bottom:8px;">FAMILY ID TRACKER</div>
    <div style="font-size:22px; font-weight:800; letter-spacing:-0.5px;">
      🪪 신분증 만료 임박 알림
    </div>
    <div style="margin-top:8px; font-size:13px; opacity:0.75;">기준일 · {today_str}</div>
  </div>

  <!-- 요약 배지 바 -->
  <div style="background:#fff; padding:14px 24px; display:flex; gap:8px; flex-wrap:wrap;
              border-left:1px solid #e5e7eb; border-right:1px solid #e5e7eb;">
    <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
                border-radius:999px; background:#FFF1F0; border:1px solid #FF4D4F;">
      <span style="font-size:12px; font-weight:700; color:#FF4D4F;">⛔ 임박 {cnt['imminent']}건</span>
    </div>
    <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
                border-radius:999px; background:#FFFBE6; border:1px solid #FA8C16;">
      <span style="font-size:12px; font-weight:700; color:#FA8C16;">⚠️ 주의 {cnt['caution']}건</span>
    </div>
    <div style="display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
                border-radius:999px; background:#F6FFED; border:1px solid #52C41A;">
      <span style="font-size:12px; font-weight:700; color:#52C41A;">✅ 여유 {cnt['ok']}건</span>
    </div>
    <div style="margin-left:auto; display:flex; align-items:center;
                font-size:12px; color:#888;">총 {len(df)}건</div>
  </div>

  <!-- 카드 본문 -->
  <div style="background:#fff; padding:20px 20px 4px;
              border:1px solid #e5e7eb; border-top:none; border-radius:0 0 16px 16px;">

    <p style="margin:0 0 16px 0; font-size:14px; color:#374151; line-height:1.6;">
      아래 신분증의 만료일이 임박하였습니다.<br>
      확인 후 <strong style="color:#b42318;">갱신(RENEW)</strong> 절차를 진행해 주세요.
    </p>

    {''.join(cards_html)}

    {ai_section}

    <!-- 푸터 안내 -->
    <div style="margin-top:20px; padding-top:16px; border-top:1px solid #f3f4f6;
                font-size:11px; color:#9ca3af; line-height:1.8;">
      • 본 메일은 자동 발송됩니다. 시트의 <strong>ExpiryDate</strong> · <strong>AlertDaysBefore</strong> · <strong>Active</strong> 값에 따라 결정됩니다.<br>
      • 동일 기준일 중복 발송 방지를 위해 <strong>LastAlertDate</strong>가 자동 갱신됩니다.<br>
      • AI 조언은 Claude Haiku가 생성하였으며, 실제 갱신 요건은 해당 기관에서 확인하세요.
    </div>
  </div>

</div>
</body>
</html>
    """
    return html


# ======================
# 7) Email Sender
# ======================

def parse_emails(raw: str) -> List[str]:
    if not raw:
        return []
    seen, out = set(), []
    for chunk in raw.replace(";", ",").split(","):
        e = chunk.strip()
        if e and e not in seen:
            out.append(e)
            seen.add(e)
    return out


def send_html_email(smtp_conf: Dict[str, object], to_email: str, subject: str, html_body: str) -> None:
    msg             = MIMEMultipart("alternative")
    msg["Subject"]  = subject
    msg["From"]     = str(smtp_conf["sender_email"])
    msg["To"]       = to_email
    msg.attach(MIMEText(html_body, "html"))

    host, port = str(smtp_conf["smtp_host"]), int(smtp_conf["smtp_port"])
    user, pw   = str(smtp_conf["smtp_user"]), str(smtp_conf["smtp_password"])

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo(); server.starttls(); server.ehlo()
        server.login(user, pw)
        server.sendmail(msg["From"], [to_email], msg.as_string())


# ======================
# 8) Sheet Update
# ======================

def update_last_alert_dates(ws: gspread.Worksheet, df_sent: pd.DataFrame, today: dt.date) -> None:
    if df_sent.empty:
        return
    header = ws.row_values(1)
    if "LastAlertDate" not in header:
        raise RuntimeError("LastAlertDate column not found in sheet header")
    col_idx = header.index("LastAlertDate") + 1
    for _, row in df_sent.iterrows():
        ws.update_cell(int(row["RowNumber"]), col_idx, today.strftime("%Y-%m-%d"))


# ======================
# 9) Main
# ======================

def main() -> None:
    cfg   = load_env()
    today = get_today_local(str(cfg["timezone"]))

    smtp_conf = {
        "smtp_host":    cfg["smtp_host"],
        "smtp_port":    cfg["smtp_port"],
        "smtp_user":    cfg["smtp_user"],
        "smtp_password": cfg["smtp_password"],
        "sender_email": cfg["sender_email"],
    }

    # ── Google Sheets 로드 ──────────────────────────────────────
    client   = get_gsheet_client(sa_json=str(cfg["sa_json"]))
    df_ids, ws = load_identifications_df(
        spreadsheet_id=str(cfg["spreadsheet_id"]),
        worksheet_name=str(cfg["worksheet_name"]),
        client=client,
    )
    df_alerts = find_ids_to_alert(df_ids, today)

    print(f"TODAY: {today} | TOTAL: {len(df_ids)} | ALERTS: {len(df_alerts)}")

    if df_alerts.empty:
        print("NO ALERTS TODAY — exiting")
        return

    # ── Claude Haiku AI 코멘트 생성 ────────────────────────────
    anthropic_key  = str(cfg.get("anthropic_key") or "")
    ai_commentary  = generate_claude_commentary(df_alerts, today, anthropic_key)
    if ai_commentary:
        print("[Claude] Commentary generated successfully")

    # ── 이메일 빌드 & 발송 ─────────────────────────────────────
    subject    = f"🪪 신분증 만료 임박 알림 — {today.strftime('%Y-%m-%d')} ({len(df_alerts)}건)"
    html_body  = build_all_alerts_html(df_alerts, today, ai_commentary=ai_commentary)
    recipients = parse_emails(str(cfg["alert_recipients"]))

    if not recipients:
        raise RuntimeError("IDENT_ALERT_RECIPIENTS (or EMAIL_RECEIVER) is empty or invalid.")

    failures: List[str] = []
    sent_any = False

    for to_email in recipients:
        try:
            send_html_email(smtp_conf, to_email, subject, html_body)
            print(f"[SENT] → {to_email}")
            sent_any = True
        except Exception as e:
            msg = f"Failed → {to_email}: {e}"
            print("[ERROR]", msg)
            failures.append(msg)

    if sent_any:
        update_last_alert_dates(ws, df_alerts, today)
        print(f"[UPDATED] LastAlertDate × {len(df_alerts)}행")

    if failures:
        raise RuntimeError("Email send failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()



# from __future__ import annotations
# import os
# import json
# import smtplib
# import datetime as dt
# from typing import Dict, Tuple, List, Optional
# import pandas as pd
# import gspread
# from dotenv import load_dotenv
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# try:
#     from zoneinfo import ZoneInfo  # Python 3.9+
# except Exception:  # pragma: no cover
#     ZoneInfo = None  # type: ignore
# # ======================
# # 1) Environment
# # ======================

# def load_env() -> Dict[str, object]:
#     """Load env vars (supports local .env via python-dotenv).

#     This script is designed to run on GitHub Actions (Secrets) or locally (.env).
#     It supports both the identification-expiry notifier specific env names and
#     (optionally) your existing daily-report env names as fallbacks.

#     Required (preferred names):
#       - IDENT_SHEET_ID
#       - GSPREAD_SERVICE_ACCOUNT_JSON
#       - SMTP_HOST, SMTP_PORT
#       - SMTP_USER, SMTP_PASSWORD
#       - IDENT_SENDER_EMAIL           (From address; should be 1 email)
#       - IDENT_ALERT_RECIPIENTS       (To addresses; comma/semicolon separated)

#     Fallbacks (if you already have these in Secrets):
#       - EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
#     """
#     load_dotenv()

#     spreadsheet_id = os.getenv("IDENT_SHEET_ID") or os.getenv("GSHEET_ID")
#     worksheet_name = os.getenv("IDENT_WORKSHEET_NAME", "Sheet1")
#     tz_name = os.getenv("IDENT_TIMEZONE", "America/Edmonton")  # Calgary

#     smtp_host = os.getenv("SMTP_HOST")
#     smtp_port = int(os.getenv("SMTP_PORT", "587"))

#     # login creds (fallbacks to existing daily-report secrets)
#     smtp_user = os.getenv("SMTP_USER") or os.getenv("EMAIL_SENDER")
#     smtp_password = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASSWORD")

#     sender_email = os.getenv("IDENT_SENDER_EMAIL") or os.getenv("EMAIL_SENDER")

#     # recipients (comma/semicolon separated); fallback to EMAIL_RECEIVER if present
#     alert_recipients = os.getenv("IDENT_ALERT_RECIPIENTS") or os.getenv("ALERT_RECIPIENTS") or os.getenv("EMAIL_RECEIVER")

#     sa_json = os.getenv("GSPREAD_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
#     admin_email = os.getenv("IDENT_ADMIN_EMAIL")  # optional

#     cfg: Dict[str, object] = {
#         "spreadsheet_id": spreadsheet_id,
#         "worksheet_name": worksheet_name,
#         "timezone": tz_name,
#         "smtp_host": smtp_host,
#         "smtp_port": smtp_port,
#         "smtp_user": smtp_user,
#         "smtp_password": smtp_password,
#         "sender_email": sender_email,
#         "alert_recipients": alert_recipients,
#         "admin_email": admin_email,
#         "sa_json": sa_json,
#     }

#     required = ["spreadsheet_id", "smtp_host", "smtp_user", "smtp_password", "sender_email", "alert_recipients", "sa_json"]
#     missing = [k for k in required if not cfg.get(k)]
#     if missing:
#         raise RuntimeError(f"Missing required environment variables: {missing}")

#     return cfg

# def get_today_local(tz_name: str) -> dt.date:
#     """Return local date using IANA timezone (fallback to dt.date.today())."""
#     if ZoneInfo is None:
#         return dt.date.today()
#     try:
#         return dt.datetime.now(ZoneInfo(tz_name)).date()
#     except Exception:
#         # If timezone string is invalid, degrade gracefully
#         return dt.date.today()
# # ======================
# # 2) Google Sheets Auth
# # ======================
# def get_gsheet_client(sa_json: Optional[str] = None) -> gspread.client.Client:
#     """
#     Preferred: use env var GSPREAD_SERVICE_ACCOUNT_JSON (string content of the JSON key).
#     Fallback: gspread.service_account() which reads ~/.config/gspread/service_account.json.
#     """
#     if sa_json:
#         info = json.loads(sa_json)
#         return gspread.service_account_from_dict(info)
#     return gspread.service_account()
# # ======================
# # 3) Sheet -> DataFrame
# # ======================
# EXPECTED_COLUMNS = [
#     "PersonName",
#     "IDType",
#     "Country",
#     "ExpiryDate",
#     "AlertDaysBefore",
#     "Active",
#     "LastAlertDate",
# ]
# def load_identifications_df(
#     spreadsheet_id: str,
#     worksheet_name: str,
#     client: gspread.client.Client,
# ) -> Tuple[pd.DataFrame, gspread.Worksheet]:
#     """
#     Read worksheet values into a DataFrame and return (df, worksheet).
#     Adds RowNumber for later updates.
#     """
#     sh = client.open_by_key(spreadsheet_id)
#     ws = sh.worksheet(worksheet_name)
#     rows = ws.get_all_values()
#     if not rows:
#         raise RuntimeError("Sheet is empty")
#     header = rows[0]
#     data_rows = rows[1:]
#     records = []
#     for idx, row in enumerate(data_rows, start=2):  # header at row 1
#         rec = {col: (row[i] if i < len(row) else "") for i, col in enumerate(header)}
#         rec["RowNumber"] = idx
#         records.append(rec)
#     df = pd.DataFrame(records)
#     # Ensure expected columns exist even if user hasn't added them yet
#     for col in EXPECTED_COLUMNS:
#         if col not in df.columns:
#             df[col] = ""
#     # Parse & normalize types
#     df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"], errors="coerce").dt.date
#     df["AlertDaysBefore"] = pd.to_numeric(df["AlertDaysBefore"], errors="coerce").fillna(0).astype(int)
#     # Accept TRUE/true/1/yes/y
#     df["Active"] = (
#         df["Active"]
#         .astype(str)
#         .str.strip()
#         .str.upper()
#         .isin(["TRUE", "1", "Y", "YES"])
#     )
#     df["LastAlertDate"] = pd.to_datetime(df["LastAlertDate"], errors="coerce").dt.date
#     return df, ws
# # ======================
# # 4) Alert Logic
# # ======================
# def find_ids_to_alert(df_ids: pd.DataFrame, today: dt.date) -> pd.DataFrame:
#     """
#     Alert criteria per row:
#       - Active is True
#       - ExpiryDate exists
#       - 0 <= DaysToExpiry <= AlertDaysBefore
#       - LastAlertDate is empty OR LastAlertDate < today
#     """
#     df = df_ids.copy()
#     # Active + has expiry date
#     df = df[df["Active"] & df["ExpiryDate"].notna()].copy()
#     if df.empty:
#         return df
#     df["DaysToExpiry"] = (df["ExpiryDate"] - today).apply(lambda x: x.days)
#     cond_in_window = (df["DaysToExpiry"] >= 0) & (df["DaysToExpiry"] <= df["AlertDaysBefore"])
#     cond_not_alerted_today = df["LastAlertDate"].isna() | (df["LastAlertDate"] < today)
#     df = df[cond_in_window & cond_not_alerted_today].copy()
#     return df
# # ======================
# # 5) Email
# # ======================


# def parse_emails(raw: str) -> List[str]:
#     """Split comma/semicolon separated emails into a clean list (deduped, order preserved)."""
#     if not raw:
#         return []
#     items = []
#     for chunk in raw.replace(";", ",").split(","):
#         e = chunk.strip()
#         if e:
#             items.append(e)
#     seen = set()
#     out = []
#     for e in items:
#         if e not in seen:
#             out.append(e)
#             seen.add(e)
#     return out


# def build_all_alerts_html(df_alerts: pd.DataFrame, today: dt.date) -> str:
#     """Build ONE combined, readable HTML email body for all alerts (Korean template)."""
#     today_str = today.strftime("%Y-%m-%d")

#     df = df_alerts.copy()
#     if "DaysToExpiry" not in df.columns:
#         df["DaysToExpiry"] = (df["ExpiryDate"] - today).apply(lambda x: x.days)

#     # Ensure columns exist
#     for c in ["PersonName", "IDType", "Country", "ExpiryDate", "DaysToExpiry"]:
#         if c not in df.columns:
#             df[c] = ""

#     # Sort by urgency, then name/type
#     try:
#         df = df.sort_values(by=["DaysToExpiry", "PersonName", "IDType"], ascending=[True, True, True])
#     except Exception:
#         pass

#     def esc(s: object) -> str:
#         # Minimal HTML escaping
#         t = "" if s is None else str(s)
#         return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
#                  .replace('"', "&quot;").replace("'", "&#39;"))


#     def classify_urgency(days: int, alert_before: int) -> str:
#         """Return one of: 'imminent', 'caution', 'ok'.

#         Uses the sheet's AlertDaysBefore (lead time) as the denominator.
#         - If alert_before <= 0: falls back to absolute thresholds (≤7 / ≤30).
#         - Otherwise: ratio = days / alert_before.
#           * imminent: ratio <= 0.25
#           * caution : ratio <= 0.60
#           * ok      : ratio > 0.60
#         """
#         if days <= 0:
#             return "imminent"
#         if alert_before is None or alert_before <= 0:
#             if days <= 7:
#                 return "imminent"
#             if days <= 30:
#                 return "caution"
#             return "ok"

#         ratio = days / float(alert_before)
#         if ratio <= 0.25:
#             return "imminent"
#         if ratio <= 0.60:
#             return "caution"
#         return "ok"

#     def urgency_badge(days: int, alert_before: int) -> str:
#         cat = classify_urgency(days, alert_before)
#         if cat == "imminent":
#             return "<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#fee2e2;color:#991b1b;font-weight:800;'>⛔ 임박</span>"
#         if cat == "caution":
#             return "<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#ffedd5;color:#9a3412;font-weight:800;'>⚠️ 주의</span>"
#         return "<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#dcfce7;color:#166534;font-weight:800;'>✅ 여유</span>"

#     # category counts for the summary line
#     cnt_imminent = 0
#     cnt_caution = 0
#     cnt_ok = 0

#     rows_html: List[str] = []
#     for _, r in df.iterrows():
#         name = str(r.get("PersonName", "")).strip() or "-"
#         idtype = str(r.get("IDType", "")).strip() or "-"
#         country = str(r.get("Country", "")).strip() or "-"
#         expiry_str = str(r.get("ExpiryDate", "")).strip() or "-"
#         days_int = int(r.get("DaysToExpiry", 0))

#         alert_before_raw = r.get("AlertDaysBefore", 0)
#         try:
#             alert_before = int(alert_before_raw)
#         except Exception:
#             alert_before = 0

#         cat = classify_urgency(days_int, alert_before)
#         if cat == "imminent":
#             cnt_imminent += 1
#             row_bg = "background:#fff1f2;"
#         elif cat == "caution":
#             cnt_caution += 1
#             row_bg = "background:#fffbeb;"
#         else:
#             cnt_ok += 1
#             row_bg = ""

#         rows_html.append(
#             f"""
#             <tr style="{row_bg}">
#               <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#111827;font-weight:700;">{esc(name)}</td>
#               <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#111827;">{esc(idtype)}</td>
#               <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#111827;">{esc(country)}</td>
#               <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#111827;">{esc(expiry_str)}</td>
#               <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;color:#111827;font-weight:700;">{esc(days_int)}</td>
#               <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;">{urgency_badge(days_int, alert_before)}</td>
#             </tr>
#             """
#         )

#     total = len(df)

#     html = f"""
#     <div style="font-family: Arial, Helvetica, sans-serif; background:#f6f7fb; padding:24px;">
#       <div style="max-width:740px; margin:0 auto; background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden;">
#         <div style="padding:18px 20px; background:#111827; color:#ffffff;">
#           <div style="font-size:18px; font-weight:800;">🪪 [신분증 만료 임박 알림]</div>
#           <div style="margin-top:6px; font-size:13px; opacity:0.9;">기준일: <b>{today_str}</b></div>
#         </div>

#         <div style="padding:18px 20px; color:#111827;">
#           <p style="margin:0 0 10px 0; font-size:15px; font-weight:700;">알려드립니다.</p>
#           <p style="margin:0 0 12px 0; font-size:14px;">
#             다음 신분증이 <b style="color:#1d4ed8;">{today_str}</b> 기준으로 만료일이 임박했습니다.
#           </p>
#           <p style="margin:0 0 16px 0; font-size:14px;">
#             확인 후 <b style="color:#b42318;">RENEW</b> 하시기 바랍니다.
#           </p>

#           <div style="margin:14px 0 10px 0; font-size:13px; color:#374151;">
#             총 <b>{total}</b>건
#             <span style="margin-left:10px;">⛔ 임박 <b>{cnt_imminent}</b></span>
#             <span style="margin-left:10px;">⚠️ 주의 <b>{cnt_caution}</b></span>
#             <span style="margin-left:10px;">✅ 여유 <b>{cnt_ok}</b></span>
#             <div style="margin-top:6px; font-size:12px; color:#6b7280;">
#               상태 기준은 각 행의 <b>AlertDaysBefore</b> 대비 남은 기간 비율로 분류합니다: ⛔ ≤25% · ⚠️ ≤60% · ✅ &gt;60%
#             </div>
#           </div>

#           <div style="border:1px solid #e5e7eb; border-radius:12px; overflow:hidden;">
#             <table role="presentation" style="width:100%; border-collapse:collapse;">
#               <thead>
#                 <tr style="background:#f3f4f6;">
#                   <th style="text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #e5e7eb;">이름</th>
#                   <th style="text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #e5e7eb;">신분증 종류</th>
#                   <th style="text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #e5e7eb;">국가</th>
#                   <th style="text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #e5e7eb;">만료일</th>
#                   <th style="text-align:right;padding:10px 12px;font-size:13px;border-bottom:1px solid #e5e7eb;">만료까지(일)</th>
#                   <th style="text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #e5e7eb;">상태</th>
#                 </tr>
#               </thead>
#               <tbody>
#                 {''.join(rows_html)}
#               </tbody>
#             </table>
#           </div>

#           <div style="margin-top:16px; font-size:12px; color:#6b7280;">
#             • 본 메일은 자동 발송입니다. 시트의 <b>ExpiryDate</b>, <b>AlertDaysBefore</b>, <b>Active</b> 값에 따라 결정됩니다.<br/>
#             • 동일 기준일에 중복 발송을 방지하기 위해 <b>LastAlertDate</b>가 자동 갱신됩니다.
#           </div>
#         </div>
#       </div>
#     </div>
#     """
#     return html

# def send_html_email(smtp_conf: Dict[str, object], to_email: str, subject: str, html_body: str) -> None:
#     msg = MIMEMultipart("alternative")
#     msg["Subject"] = subject
#     msg["From"] = str(smtp_conf["sender_email"])
#     msg["To"] = to_email
#     msg.attach(MIMEText(html_body, "html"))
#     host = str(smtp_conf["smtp_host"])
#     port = int(smtp_conf["smtp_port"])
#     user = str(smtp_conf["smtp_user"])
#     password = str(smtp_conf["smtp_password"])
#     with smtplib.SMTP(host, port, timeout=30) as server:
#         server.ehlo()
#         server.starttls()
#         server.ehlo()
#         server.login(user, password)
#         server.sendmail(msg["From"], [to_email], msg.as_string())
# # ======================
# # 6) Sheet Update
# # ======================
# def update_last_alert_dates(ws: gspread.Worksheet, df_sent: pd.DataFrame, today: dt.date) -> None:
#     """
#     Update LastAlertDate for sent rows.
#     Uses per-cell update for simplicity (small volumes). Can be batched later if needed.
#     """
#     if df_sent.empty:
#         return
#     header = ws.row_values(1)
#     if "LastAlertDate" not in header:
#         raise RuntimeError("LastAlertDate column not found in sheet header")
#     col_idx = header.index("LastAlertDate") + 1  # 1-based
#     for _, row in df_sent.iterrows():
#         ws.update_cell(int(row["RowNumber"]), col_idx, today.strftime("%Y-%m-%d"))
# # ======================
# # 7) Main
# # ======================

# def main() -> None:
#     cfg = load_env()

#     # Calgary time (America/Edmonton)
#     today = get_today_local(str(cfg["timezone"]))

#     smtp_conf = {
#         "smtp_host": cfg["smtp_host"],
#         "smtp_port": cfg["smtp_port"],
#         "smtp_user": cfg["smtp_user"],
#         "smtp_password": cfg["smtp_password"],
#         "sender_email": cfg["sender_email"],  # will be used as From
#     }

#     # Load sheet
#     client = get_gsheet_client(sa_json=str(cfg["sa_json"]))
#     df_ids, ws = load_identifications_df(
#         spreadsheet_id=str(cfg["spreadsheet_id"]),
#         worksheet_name=str(cfg["worksheet_name"]),
#         client=client,
#     )

#     df_alerts = find_ids_to_alert(df_ids, today)

#     # Minimal run logs
#     print("TODAY:", today)
#     print("TOTAL ROWS:", len(df_ids))
#     print("ALERT ROWS:", len(df_alerts))
#     if not df_alerts.empty:
#         # Debug view (no sensitive IDs)
#         debug_cols = [c for c in ["PersonName", "IDType", "Country", "ExpiryDate", "AlertDaysBefore", "DaysToExpiry", "LastAlertDate"] if c in df_alerts.columns]
#         try:
#             print(df_alerts[debug_cols].to_string(index=False))
#         except Exception:
#             pass

#     if df_alerts.empty:
#         print("NO ALERTS TODAY — exiting")
#         return

#     # One combined email to fixed recipients (not per PersonEmail)
#     subject = "🪪 [신분증 만료 임박 알림]"
#     html_body = build_all_alerts_html(df_alerts, today)

#     recipients = parse_emails(str(cfg["alert_recipients"]))
#     print("RECIPIENTS:", recipients)
#     if not recipients:
#         raise RuntimeError("IDENT_ALERT_RECIPIENTS (or EMAIL_RECEIVER) is empty or invalid.")

#     # Send to all recipients; From uses IDENT_SENDER_EMAIL (single address recommended)
#     failures: List[str] = []
#     sent_any = False

#     for to_email in recipients:
#         try:
#             send_html_email(smtp_conf, to_email, subject, html_body)
#             print(f"[SENT] {to_email} ({len(df_alerts)} item(s) total)")
#             sent_any = True
#         except Exception as e:
#             msg = f"Failed to send to {to_email}: {e}"
#             print("[ERROR]", msg)
#             failures.append(msg)

#     # Update LastAlertDate for alerted rows only if at least one email was sent
#     if sent_any:
#         update_last_alert_dates(ws, df_alerts, today)
#         print(f"[UPDATED] LastAlertDate updated for {len(df_alerts)} row(s)")

#     # Fail job if any recipients failed
#     if failures:
#         raise RuntimeError("Email send failures:\n" + "\n".join(failures))

# if __name__ == "__main__":
#     main()
