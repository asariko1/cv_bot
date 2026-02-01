import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


def append_missing_question(sheet_id: str, question: str):
    """
    Appends a missing/unanswered question to a Google Sheet.
    """

    service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON in environment")

    creds_info = json.loads(service_account_json)

    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

    service = build("sheets", "v4", credentials=creds)

    body = {
        "values": [[question]]
    }

    service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="A:A",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()