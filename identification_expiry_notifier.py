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
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
# ======================
# 1) Environment
# ======================

def load_env() -> Dict[str, object]:
    """Load env vars (supports local .env via python-dotenv).

    This script is designed to run on GitHub Actions (Secrets) or locally (.env).
    It supports both the identification-expiry notifier specific env names and
    (optionally) your existing daily-report env names as fallbacks.

    Required (preferred names):
      - IDENT_SHEET_ID
      - GSPREAD_SERVICE_ACCOUNT_JSON
      - SMTP_HOST, SMTP_PORT
      - SMTP_USER, SMTP_PASSWORD
      - IDENT_SENDER_EMAIL           (From address; should be 1 email)
      - IDENT_ALERT_RECIPIENTS       (To addresses; comma/semicolon separated)

    Fallbacks (if you already have these in Secrets):
      - EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
    """
    load_dotenv()

    spreadsheet_id = os.getenv("IDENT_SHEET_ID") or os.getenv("GSHEET_ID")
    worksheet_name = os.getenv("IDENT_WORKSHEET_NAME", "Sheet1")
    tz_name = os.getenv("IDENT_TIMEZONE", "America/Edmonton")  # Calgary

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    # login creds (fallbacks to existing daily-report secrets)
    smtp_user = os.getenv("SMTP_USER") or os.getenv("EMAIL_SENDER")
    smtp_password = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASSWORD")

    sender_email = os.getenv("IDENT_SENDER_EMAIL") or os.getenv("EMAIL_SENDER")

    # recipients (comma/semicolon separated); fallback to EMAIL_RECEIVER if present
    alert_recipients = os.getenv("IDENT_ALERT_RECIPIENTS") or os.getenv("ALERT_RECIPIENTS") or os.getenv("EMAIL_RECEIVER")

    sa_json = os.getenv("GSPREAD_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    admin_email = os.getenv("IDENT_ADMIN_EMAIL")  # optional

    cfg: Dict[str, object] = {
        "spreadsheet_id": spreadsheet_id,
        "worksheet_name": worksheet_name,
        "timezone": tz_name,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_user": smtp_user,
        "smtp_password": smtp_password,
        "sender_email": sender_email,
        "alert_recipients": alert_recipients,
        "admin_email": admin_email,
        "sa_json": sa_json,
    }

    required = ["spreadsheet_id", "smtp_host", "smtp_user", "smtp_password", "sender_email", "alert_recipients", "sa_json"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    return cfg

def get_today_local(tz_name: str) -> dt.date:
    """Return local date using IANA timezone (fallback to dt.date.today())."""
    if ZoneInfo is None:
        return dt.date.today()
    try:
        return dt.datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        # If timezone string is invalid, degrade gracefully
        return dt.date.today()
# ======================
# 2) Google Sheets Auth
# ======================
def get_gsheet_client(sa_json: Optional[str] = None) -> gspread.client.Client:
    """
    Preferred: use env var GSPREAD_SERVICE_ACCOUNT_JSON (string content of the JSON key).
    Fallback: gspread.service_account() which reads ~/.config/gspread/service_account.json.
    """
    if sa_json:
        info = json.loads(sa_json)
        return gspread.service_account_from_dict(info)
    return gspread.service_account()
# ======================
# 3) Sheet -> DataFrame
# ======================
EXPECTED_COLUMNS = [
    "PersonName",
    "PersonEmail",
    "IDType",
    "Country",
    "ExpiryDate",
    "AlertDaysBefore",
    "Active",
    "LastAlertDate",
]
def load_identifications_df(
    spreadsheet_id: str,
    worksheet_name: str,
    client: gspread.client.Client,
) -> Tuple[pd.DataFrame, gspread.Worksheet]:
    """
    Read worksheet values into a DataFrame and return (df, worksheet).
    Adds RowNumber for later updates.
    """
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    rows = ws.get_all_values()
    if not rows:
        raise RuntimeError("Sheet is empty")
    header = rows[0]
    data_rows = rows[1:]
    records = []
    for idx, row in enumerate(data_rows, start=2):  # header at row 1
        rec = {col: (row[i] if i < len(row) else "") for i, col in enumerate(header)}
        rec["RowNumber"] = idx
        records.append(rec)
    df = pd.DataFrame(records)
    # Ensure expected columns exist even if user hasn't added them yet
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Parse & normalize types
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"], errors="coerce").dt.date
    df["AlertDaysBefore"] = pd.to_numeric(df["AlertDaysBefore"], errors="coerce").fillna(0).astype(int)
    # Accept TRUE/true/1/yes/y
    df["Active"] = (
        df["Active"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "Y", "YES"])
    )
    df["LastAlertDate"] = pd.to_datetime(df["LastAlertDate"], errors="coerce").dt.date
    return df, ws
# ======================
# 4) Alert Logic
# ======================
def find_ids_to_alert(df_ids: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """
    Alert criteria per row:
      - Active is True
      - ExpiryDate exists
      - 0 <= DaysToExpiry <= AlertDaysBefore
      - LastAlertDate is empty OR LastAlertDate < today
      - PersonEmail contains '@'
    """
    df = df_ids.copy()
    # Active + has expiry date
    df = df[df["Active"] & df["ExpiryDate"].notna()].copy()
    if df.empty:
        return df
    df["DaysToExpiry"] = (df["ExpiryDate"] - today).apply(lambda x: x.days)
    cond_in_window = (df["DaysToExpiry"] >= 0) & (df["DaysToExpiry"] <= df["AlertDaysBefore"])
    cond_not_alerted_today = df["LastAlertDate"].isna() | (df["LastAlertDate"] < today)
    df = df[cond_in_window & cond_not_alerted_today].copy()
    return df
# ======================
# 5) Email
# ======================


def parse_emails(raw: str) -> List[str]:
    """Split comma/semicolon separated emails into a clean list (deduped, order preserved)."""
    if not raw:
        return []
    items = []
    for chunk in raw.replace(";", ",").split(","):
        e = chunk.strip()
        if e:
            items.append(e)
    seen = set()
    out = []
    for e in items:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def build_all_alerts_html(df_alerts: pd.DataFrame, today: dt.date) -> str:
    """Build ONE combined HTML email body for all alerts (Korean template)."""
    today_str = today.strftime("%Y-%m-%d")

    df = df_alerts.copy()
    if "DaysToExpiry" not in df.columns:
        df["DaysToExpiry"] = (df["ExpiryDate"] - today).apply(lambda x: x.days)

    # Stable column order for the table (include name)
    cols = ["PersonName", "IDType", "Country", "ExpiryDate", "DaysToExpiry"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    # Build rows
    rows_html = []
    for _, r in df.iterrows():
        expiry = r.get("ExpiryDate")
        expiry_str = expiry.strftime("%Y-%m-%d") if hasattr(expiry, "strftime") else str(expiry or "")
        rows_html.append(
            "<tr>"
            f"<td>{r.get('PersonName','')}</td>"
            f"<td>{r.get('IDType','')}</td>"
            f"<td>{r.get('Country','')}</td>"
            f"<td>{expiry_str}</td>"
            f"<td>{r.get('DaysToExpiry','')}</td>"
            "</tr>"
        )

    html = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; font-size: 14px; line-height: 1.6;">
      <p>다음 신분증이 <b>{today_str}</b> 기준으로 만료일이 임박했습니다:</p>
      <p style="margin-top: 14px;">확인후 RENEW 하시기 바랍니다.</p>

      <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; margin-top: 10px;">
        <thead>
          <tr>
            <th>이름</th>
            <th>신분증 종류</th>
            <th>국가</th>
            <th>만료일</th>
            <th>만료까지(일)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>

    </div>
    """
    return html

def build_personal_alert_html(df_person: pd.DataFrame, today: dt.date) -> str:
    """
    Korean template per request.
    Output columns: IDType, Country, ExpiryDate, DaysToExpiry

    
          
            
    

          
          Expand Down
    
    
  
    """
    today_str = today.strftime("%Y-%m-%d")
    rows_html = []
    for _, r in df_person.iterrows():
        expiry = r.get("ExpiryDate")
        expiry_str = expiry.strftime("%Y-%m-%d") if isinstance(expiry, dt.date) else ""
        rows_html.append(
            f"""
            <tr>
              <td>{r.get("IDType","")}</td>
              <td>{r.get("Country","")}</td>
              <td>{expiry_str}</td>
              <td>{r.get("DaysToExpiry","")}</td>
            </tr>
            """
        )
    html = f"""
    <p>다음 신분증이 <b>{today_str}</b> 기준으로 만료일이 임박했습니다</p>
    <p>확인후 RENEW 하시기 바랍니다.</p>
    <table border="1" cellspacing="0" cellpadding="6">
      <thead>
        <tr>
          <th>신분증 종류</th>
          <th>국가</th>
          <th>만료일</th>
          <th>만료까지(일)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
    return html
def send_html_email(smtp_conf: Dict[str, object], to_email: str, subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = str(smtp_conf["sender_email"])
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))
    host = str(smtp_conf["smtp_host"])
    port = int(smtp_conf["smtp_port"])
    user = str(smtp_conf["smtp_user"])
    password = str(smtp_conf["smtp_password"])
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(user, password)
        server.sendmail(msg["From"], [to_email], msg.as_string())
# ======================
# 6) Sheet Update
# ======================
def update_last_alert_dates(ws: gspread.Worksheet, df_sent: pd.DataFrame, today: dt.date) -> None:
    """
    Update LastAlertDate for sent rows.
    Uses per-cell update for simplicity (small volumes). Can be batched later if needed.
    """
    if df_sent.empty:
        return
    header = ws.row_values(1)
    if "LastAlertDate" not in header:
        raise RuntimeError("LastAlertDate column not found in sheet header")
    col_idx = header.index("LastAlertDate") + 1  # 1-based
    for _, row in df_sent.iterrows():
        ws.update_cell(int(row["RowNumber"]), col_idx, today.strftime("%Y-%m-%d"))
# ======================
# 7) Main
# ======================

def main() -> None:
    cfg = load_env()

    # Calgary time (America/Edmonton)
    today = get_today_local(str(cfg["timezone"]))

    smtp_conf = {
        "smtp_host": cfg["smtp_host"],
        "smtp_port": cfg["smtp_port"],
        "smtp_user": cfg["smtp_user"],
        "smtp_password": cfg["smtp_password"],
        "sender_email": cfg["sender_email"],  # will be used as From
    }

    # Load sheet
    client = get_gsheet_client(sa_json=str(cfg["sa_json"]))
    df_ids, ws = load_identifications_df(
        spreadsheet_id=str(cfg["spreadsheet_id"]),
        worksheet_name=str(cfg["worksheet_name"]),
        client=client,
    )

    df_alerts = find_ids_to_alert(df_ids, today)

    # Minimal run logs
    print("TODAY:", today)
    print("TOTAL ROWS:", len(df_ids))
    print("ALERT ROWS:", len(df_alerts))
    if not df_alerts.empty:
        # Debug view (no sensitive IDs)
        debug_cols = [c for c in ["PersonName", "PersonEmail", "IDType", "ExpiryDate", "AlertDaysBefore", "DaysToExpiry", "LastAlertDate"] if c in df_alerts.columns]
        try:
            print(df_alerts[debug_cols].to_string(index=False))
        except Exception:
            pass

    if df_alerts.empty:
        print("NO ALERTS TODAY — exiting")
        return

    # One combined email to fixed recipients (not per PersonEmail)
    subject = "[신분증 만료 임박 알림]"
    html_body = build_all_alerts_html(df_alerts, today)

    recipients = parse_emails(str(cfg["alert_recipients"]))
    if not recipients:
        raise RuntimeError("IDENT_ALERT_RECIPIENTS (or EMAIL_RECEIVER) is empty or invalid.")

    # Send to all recipients; From uses IDENT_SENDER_EMAIL (single address recommended)
    failures: List[str] = []
    sent_any = False

    for to_email in recipients:
        try:
            send_html_email(smtp_conf, to_email, subject, html_body)
            print(f"[SENT] {to_email} ({len(df_alerts)} item(s) total)")
            sent_any = True
        except Exception as e:
            msg = f"Failed to send to {to_email}: {e}"
            print("[ERROR]", msg)
            failures.append(msg)

    # Update LastAlertDate for alerted rows only if at least one email was sent
    if sent_any:
        update_last_alert_dates(ws, df_alerts, today)
        print(f"[UPDATED] LastAlertDate updated for {len(df_alerts)} row(s)")

    # Fail job if any recipients failed
    if failures:
        raise RuntimeError("Email send failures:\n" + "\n".join(failures))

if __name__ == "__main__":
    main()
