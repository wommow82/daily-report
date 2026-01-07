import os
import datetime as dt
from typing import Dict

import pandas as pd
import gspread
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib


# ======================
# 1. 환경 변수 로드
# ======================

def load_env():
    """
    IDENT_SHEET_ID:      Google Sheet ID (identification_alert 파일)
    IDENT_WORKSHEET_NAME: 워크시트 이름 (기본값: IDs)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD: 이메일 SMTP 설정
    IDENT_SENDER_EMAIL:  발신자 이메일
    IDENT_ADMIN_EMAIL:   에러/요약 수신용 (선택)
    """
    load_dotenv()

    config = {
        "spreadsheet_id": os.getenv("IDENT_SHEET_ID"),
        "worksheet_name": os.getenv("IDENT_WORKSHEET_NAME", "IDs"),
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER"),
        "smtp_password": os.getenv("SMTP_PASSWORD"),
        "sender_email": os.getenv("IDENT_SENDER_EMAIL"),
        "admin_email": os.getenv("IDENT_ADMIN_EMAIL"),  # optional
    }

    missing = [k for k, v in config.items() if v is None and k not in ("admin_email",)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    return config


def get_gsheet_client():
    """
    gspread 인증:
    - 기존 포트폴리오 코드처럼 service_account.json을 사용한다는 가정.
    - 다른 방식(예: OAuth) 쓰고 있으면 그 방식 그대로 맞춰도 됨.
    """
    client = gspread.service_account()  # repo 루트에 service_account.json 필요
    return client


# ======================
# 2. Sheet → DataFrame
# ======================

def load_identifications_df(spreadsheet_id: str, worksheet_name: str, client: gspread.client.Client):
    """
    Google Sheet에서 전체 데이터를 읽어 DataFrame으로 변환.
    Row 번호를 저장해서 나중에 LastAlertDate 업데이트에 사용.
    """
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)

    rows = ws.get_all_values()
    if not rows:
        raise RuntimeError("Sheet is empty")

    header = rows[0]
    data_rows = rows[1:]

    records = []
    # 1행이 header이므로 실제 데이터는 2행부터 시작
    for idx, row in enumerate(data_rows, start=2):
        rec = {col: (row[i] if i < len(row) else "") for i, col in enumerate(header)}
        rec["RowNumber"] = idx
        records.append(rec)

    df = pd.DataFrame(records)

    # 필요한 컬럼이 Sheet에 아직 없을 가능성 대비
    for col in [
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
    ]:
        if col not in df.columns:
            df[col] = ""

    # 타입 변환
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"], errors="coerce").dt.date
    df["AlertDaysBefore"] = pd.to_numeric(df["AlertDaysBefore"], errors="coerce").fillna(0).astype(int)
    df["Active"] = df["Active"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "Y", "YES"])
    df["LastAlertDate"] = pd.to_datetime(df["LastAlertDate"], errors="coerce").dt.date

    return df, ws


# ==========================
# 3. 알림 대상 필터링
# ==========================

def find_ids_to_alert(df_ids: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    df = df_ids.copy()

    # ExpiryDate가 있고 Active인 항목만
    df = df[df["ExpiryDate"].notna() & df["Active"]]

    df["DaysToExpiry"] = (df["ExpiryDate"] - today).apply(lambda x: x.days)

    # 조건:
    #   0 <= DaysToExpiry <= AlertDaysBefore
    #   LastAlertDate가 없거나, today보다 이전
    cond_in_window = (df["DaysToExpiry"] >= 0) & (df["DaysToExpiry"] <= df["AlertDaysBefore"])
    cond_not_alerted_today = df["LastAlertDate"].isna() | (df["LastAlertDate"] < today)

    df_alerts = df[cond_in_window & cond_not_alerted_today].copy()

    # PersonEmail 없는 행은 제외
    df_alerts = df_alerts[df_alerts["PersonEmail"].astype(str).str.contains("@")]

    return df_alerts


# ==========================
# 4. 사람별 이메일 HTML 생성
# ==========================

def build_personal_alert_html(person_name: str, df_person: pd.DataFrame, today: dt.date) -> str:
    """
    한 사람(PersonName/PersonEmail)에 대한 알림 HTML 본문.
    """
    rows_html = []
    for _, r in df_person.iterrows():
        id_type = r.get("IDType", "")
        id_number = r.get("IDNumber", "")
        country = r.get("Country", "")
        issuer = r.get("Issuer", "")
        expiry = r.get("ExpiryDate")
        days_to_expiry = r.get("DaysToExpiry")
        notes = r.get("Notes", "")

        expiry_str = expiry.strftime("%Y-%m-%d") if isinstance(expiry, dt.date) else ""

        row_html = f"""
        <tr>
            <td>{id_type}</td>
            <td>{id_number}</td>
            <td>{country}</td>
            <td>{issuer}</td>
            <td>{expiry_str}</td>
            <td>{days_to_expiry}</td>
            <td>{notes}</td>
        </tr>
        """
        rows_html.append(row_html)

    table_html = f"""
    <p>Hi {person_name},</p>
    <p>The following identification(s) are approaching their expiry date as of {today.strftime("%Y-%m-%d")}:</p>
    <table border="1" cellspacing="0" cellpadding="4">
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

    return table_html


# ==========================
# 5. 이메일 발송
# ==========================

def send_html_email(smtp_conf: Dict, to_email: str, subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_conf["sender_email"]
    msg["To"] = to_email

    part_html = MIMEText(html_body, "html")
    msg.attach(part_html)

    with smtplib.SMTP(smtp_conf["smtp_host"], smtp_conf["smtp_port"]) as server:
        server.starttls()
        server.login(smtp_conf["smtp_user"], smtp_conf["smtp_password"])
        server.sendmail(smtp_conf["sender_email"], [to_email], msg.as_string())


# ==========================
# 6. Sheet에서 LastAlertDate 업데이트
# ==========================

def update_last_alert_dates(ws: gspread.Worksheet, df_alerts: pd.DataFrame, today: dt.date):
    """
    실제로 메일 발송이 완료된 row들에 대해 LastAlertDate를 today로 업데이트.
    """
    if df_alerts.empty:
        return

    header = ws.row_values(1)
    try:
        col_idx = header.index("LastAlertDate") + 1  # 1-based index
    except ValueError:
        raise RuntimeError("LastAlertDate column not found in sheet header")

    for _, row in df_alerts.iterrows():
        row_number = int(row["RowNumber"])
        ws.update_cell(row_number, col_idx, today.strftime("%Y-%m-%d"))


# ==========================
# 7. main 오케스트레이션
# ==========================

def main():
    today = dt.date.today()
    config = load_env()

    smtp_conf = {
        "smtp_host": config["smtp_host"],
        "smtp_port": config["smtp_port"],
        "smtp_user": config["smtp_user"],
        "smtp_password": config["smtp_password"],
        "sender_email": config["sender_email"],
    }

    try:
        client = get_gsheet_client()
        df_ids, ws = load_identifications_df(
            config["spreadsheet_id"],
            config["worksheet_name"],
            client,
        )
    except Exception as e:
        # 초기 로딩 단계에서 실패하면 admin_email로 에러 통지(선택)
        if config.get("admin_email"):
            err_html = f"<p>ID expiry notifier failed:</p><pre>{e}</pre>"
            try:
                send_html_email(
                    smtp_conf,
                    config["admin_email"],
                    "[ID Expiry Notifier] ERROR",
                    err_html,
                )
            except Exception:
                pass
        raise

    df_alerts = find_ids_to_alert(df_ids, today)
    if df_alerts.empty:
        # 오늘은 보낼 알림 없음
        return

    sent_rows = []

    # 사람별로 묶어서 한 사람당 1통
    grouped = df_alerts.groupby(["PersonName", "PersonEmail"])
    for (person_name, person_email), df_person in grouped:
        html_body = build_personal_alert_html(person_name, df_person, today)
        subject = "[ID Expiry Notice] Identification expiry approaching"

        try:
            send_html_email(smtp_conf, person_email, subject, html_body)
            sent_rows.append(df_person)
        except Exception as e:
            # 특정 수신자에게 실패한 경우 admin_email로 통지 가능
            if config.get("admin_email"):
                err_html = (
                    f"<p>Failed to send ID expiry email to {person_email}:</p>"
                    f"<pre>{e}</pre>"
                )
                try:
                    send_html_email(
                        smtp_conf,
                        config["admin_email"],
                        "[ID Expiry Notifier] Send Error",
                        err_html,
                    )
                except Exception:
                    pass

    # 실제로 발송 성공한 row들에 대해서만 LastAlertDate 업데이트
    if sent_rows:
        df_sent = pd.concat(sent_rows, ignore_index=True)
        update_last_alert_dates(ws, df_sent, today)


if __name__ == "__main__":
    main()
