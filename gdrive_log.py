import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


def append_missing_question(sheet_id: str, question: str):
    """
    Appends a missing/unanswered question to a Google Sheet.
    """

    service_account_value = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not service_account_value:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON in environment")

    service_account_value = service_account_value.strip()

    # Case A: JSON pasted directly (Streamlit Secrets)
    if service_account_value.startswith("{"):
        creds_info = json.loads(service_account_value)
    else:
        # Case B: path to JSON file (local dev)
        with open(service_account_value, "r", encoding="utf-8") as f:
            creds_info = json.load(f)

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