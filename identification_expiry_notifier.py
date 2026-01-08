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
    """Load env vars (supports local .env via python-dotenv)."""
    load_dotenv()

    cfg: Dict[str, object] = {
        "spreadsheet_id": os.getenv("IDENT_SHEET_ID"),
        "worksheet_name": os.getenv("IDENT_WORKSHEET_NAME", "IDs"),
        "timezone": os.getenv("IDENT_TIMEZONE", "America/Edmonton"),
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER"),
        "smtp_password": os.getenv("SMTP_PASSWORD"),
        "sender_email": os.getenv("IDENT_SENDER_EMAIL"),
        "admin_email": os.getenv("IDENT_ADMIN_EMAIL"),  # optional
        "sa_json": os.getenv("GSPREAD_SERVICE_ACCOUNT_JSON"),  # recommended
    }

    required = ["spreadsheet_id", "smtp_host", "smtp_user", "smtp_password", "sender_email"]
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
    "IDNumber",
    "Country",
    "Issuer",
    "ExpiryDate",
    "AlertDaysBefore",
    "Active",
    "LastAlertDate",
    "Notes",
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

    # Require a plausible email
    df = df[df["PersonEmail"].astype(str).str.contains("@", na=False)]

    return df


# ======================
# 5) Email
# ======================

def build_personal_alert_html(person_name: str, df_person: pd.DataFrame, today: dt.date) -> str:
    rows_html = []
    for _, r in df_person.iterrows():
        expiry = r.get("ExpiryDate")
        expiry_str = expiry.strftime("%Y-%m-%d") if isinstance(expiry, dt.date) else ""
        rows_html.append(
            f"""
            <tr>
              <td>{r.get("IDType","")}</td>
              <td>{r.get("IDNumber","")}</td>
              <td>{r.get("Country","")}</td>
              <td>{r.get("Issuer","")}</td>
              <td>{expiry_str}</td>
              <td>{r.get("DaysToExpiry","")}</td>
              <td>{r.get("Notes","")}</td>
            </tr>
            """
        )

    html = f"""
    <p>Hi {person_name},</p>
    <p>The following identification(s) are approaching their expiry date as of <b>{today.strftime('%Y-%m-%d')}</b>:</p>

    <table border="1" cellspacing="0" cellpadding="6">
      <thead>
        <tr>
          <th>ID Type</th>
          <th>ID Number</th>
          <th>Country</th>
          <th>Issuer</th>
          <th>Expiry Date</th>
          <th>Days to Expiry</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>

    <p>Please review and renew them if necessary.</p>
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
    today = get_today_local(str(cfg["timezone"]))

    smtp_conf = {
        "smtp_host": cfg["smtp_host"],
        "smtp_port": cfg["smtp_port"],
        "smtp_user": cfg["smtp_user"],
        "smtp_password": cfg["smtp_password"],
        "sender_email": cfg["sender_email"],
    }

    # Load sheet
    sa_json = cfg.get("sa_json")
    client = get_gsheet_client(sa_json=str(sa_json) if sa_json else None)
    df_ids, ws = load_identifications_df(
        spreadsheet_id=str(cfg["spreadsheet_id"]),
        worksheet_name=str(cfg["worksheet_name"]),
        client=client,
    )

    df_alerts = find_ids_to_alert(df_ids, today)

    # Minimal run logs (safe)
    print("TODAY:", today)
    print("TOTAL ROWS:", len(df_ids))
    print("ALERT ROWS:", len(df_alerts))

    if not df_alerts.empty:
        print("RECIPIENTS:", df_alerts["PersonEmail"].unique().tolist())
        print(
            df_alerts[
                [
                    "PersonName",
                    "PersonEmail",
                    "IDType",
                    "ExpiryDate",
                    "AlertDaysBefore",
                    "DaysToExpiry",
                    "LastAlertDate",
                ]
            ].to_string(index=False)
        )

    if df_alerts.empty:
        print("NO ALERTS TODAY — exiting")
        return

    # Group by person and send one email per person
    sent_rows: List[pd.DataFrame] = []
    failures: List[str] = []

    grouped = df_alerts.groupby(["PersonName", "PersonEmail"], dropna=False)

    for (person_name, person_email), df_person in grouped:
        person_name = str(person_name) if person_name not in (None, "", "nan") else "there"
        person_email = str(person_email)

        subject = "[ID Expiry Notice] Identification expiry approaching"
        html_body = build_personal_alert_html(person_name, df_person, today)

        try:
            send_html_email(smtp_conf, person_email, subject, html_body)
            sent_rows.append(df_person)
            print(f"[SENT] {person_email} ({len(df_person)} item(s))")
        except Exception as e:
            msg = f"Failed to send to {person_email}: {e}"
            print("[ERROR]", msg)
            failures.append(msg)

            # Optionally notify admin (best-effort)
            admin = cfg.get("admin_email")
            if admin:
                try:
                    send_html_email(
                        smtp_conf,
                        str(admin),
                        "[ID Expiry Notifier] Send Error",
                        f"<p>{msg}</p>",
                    )
                except Exception:
                    pass

    # Update LastAlertDate only for successes
    if sent_rows:
        df_sent = pd.concat(sent_rows, ignore_index=True)
        update_last_alert_dates(ws, df_sent, today)
        print(f"[UPDATED] LastAlertDate updated for {len(df_sent)} row(s)")

    # If any failures occurred, raise so GitHub Actions shows a failure
    if failures:
        raise RuntimeError("Email send failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
