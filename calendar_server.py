"""연습용 캘린더 MCP 서버 (stdio). 앱이 자식 프로세스로 실행하고 도구 목록을 조회한다.

도구 3개: list_events(조회) · get_event(다시 조회) · create_event(생성).
모든 도구는 예외를 던지지 않고 {"ok": bool, "error_code": ..., "message": ...} 로 돌려준다.
error_code: no_calendar_id · no_credentials · auth_expired · permission_denied · connection_failed · not_found · api_error
주의: stdout 은 MCP 통신 채널이므로 print 를 쓰지 않는다. 토큰·인증 파일 내용은 어디에도 출력하지 않는다.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).parent / ".env")  # 이미 설정된 환경변수가 우선한다

API_BASE = os.getenv("GOOGLE_CALENDAR_API_BASE", "https://www.googleapis.com/calendar/v3")
SCOPE_READ = "https://www.googleapis.com/auth/calendar.readonly"  # 준비 확인·조회는 읽기 전용 권한만 쓴다
SCOPE_WRITE = "https://www.googleapis.com/auth/calendar.events"  # 생성할 때만
TIMEZONE = "Asia/Seoul"
KST = timezone(timedelta(hours=9))

mcp = FastMCP("travel-calendar")


class CalendarError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


# --- 준비 확인 순서: 캘린더 ID → 인증 정보 → (토큰) → 읽기 전용 조회 ---

def calendar_id() -> str:
    value = os.getenv("GOOGLE_CALENDAR_ID", "").strip()
    if not value:
        raise CalendarError("no_calendar_id", "GOOGLE_CALENDAR_ID 가 설정되지 않았어요.")
    return value


def load_credentials(scope: str):
    path = os.getenv("GOOGLE_CREDENTIALS_FILE", "").strip()
    if not path:
        raise CalendarError("no_credentials", "GOOGLE_CREDENTIALS_FILE 이 설정되지 않았어요.")
    file = Path(path)
    if not file.is_file():
        raise CalendarError("no_credentials", "인증 정보 파일을 찾을 수 없어요.")
    try:
        info = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CalendarError("no_credentials", "인증 정보 파일을 읽을 수 없어요(JSON 형식이 아니에요).") from None
    try:
        if info.get("type") == "service_account":
            from google.oauth2 import service_account

            return service_account.Credentials.from_service_account_info(info, scopes=[scope])
        if info.get("refresh_token"):
            from google.oauth2.credentials import Credentials

            return Credentials.from_authorized_user_info(info, scopes=[scope])
    except (ValueError, KeyError):
        pass  # 키 형식이 깨진 경우: 아래에서 '인증 정보 없음'으로 처리
    raise CalendarError("no_credentials", "인증 정보 형식을 알 수 없어요(서비스 계정 또는 사용자 인증 JSON 이 필요해요).")


def access_token(creds) -> str:
    from google.auth.exceptions import RefreshError, TransportError
    from google.auth.transport.requests import Request

    try:
        creds.refresh(Request())
    except RefreshError:
        raise CalendarError("auth_expired", "Google 인증이 만료되었거나 더 이상 유효하지 않아요.") from None
    except (TransportError, OSError):
        raise CalendarError("connection_failed", "Google 인증 서버에 연결하지 못했어요.") from None
    if not creds.token:
        raise CalendarError("auth_expired", "Google 인증 토큰을 받지 못했어요.")
    return creds.token


def google_request(method: str, path: str, scope: str, **kwargs) -> dict:
    cal = calendar_id()  # 1) ID
    creds = load_credentials(scope)  # 2) 인증 정보
    token = access_token(creds)  # 3) 토큰
    url = f"{API_BASE}/calendars/{quote(cal, safe='')}{path}"
    try:
        resp = httpx.request(method, url, headers={"Authorization": f"Bearer {token}"}, timeout=10, **kwargs)
    except httpx.HTTPError:
        raise CalendarError("connection_failed", "Google Calendar 서버에 연결하지 못했어요.") from None
    if resp.status_code in (200, 201):
        return resp.json()
    detail = ""
    try:
        detail = resp.json().get("error", {}).get("message", "")
    except ValueError:
        pass
    if resp.status_code == 401:
        raise CalendarError("auth_expired", "Google 인증이 거부되었어요(만료되었거나 무효).")
    if resp.status_code == 403:
        raise CalendarError("permission_denied", f"이 캘린더를 읽을 권한이 없어요. {detail}".strip())
    if resp.status_code == 404:
        if path.startswith("/events/"):
            raise CalendarError("not_found", "해당 일정을 찾을 수 없어요.")
        raise CalendarError("permission_denied", "캘린더를 찾을 수 없거나 이 계정에 공유되지 않았어요(권한 없음).")
    raise CalendarError("api_error", f"Google Calendar 오류 (HTTP {resp.status_code}). {detail}".strip())


def failure(err: CalendarError) -> dict:
    return {"ok": False, "error_code": err.code, "message": err.message}


def rfc3339(value: str, end_of_day: bool = False) -> str:
    """'2026-10-24' 같은 날짜는 한국 시간 하루의 시작(또는 끝)으로, 시각이 있으면 그대로 쓴다."""
    value = value.strip()
    if "T" in value:
        return value
    return f"{value}T{'23:59:59' if end_of_day else '00:00:00'}+09:00"


def simplify(event: dict) -> dict:
    start, end = event.get("start", {}), event.get("end", {})
    return {
        "id": event.get("id"),
        "title": event.get("summary", ""),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "link": event.get("htmlLink", ""),
    }


@mcp.tool()
def list_events(time_min: str = "", time_max: str = "", max_results: int = 10) -> dict:
    """[조회] 연습용 캘린더의 일정을 읽기 전용으로 조회한다. time_min·time_max 는 YYYY-MM-DD 또는 RFC3339.
    준비 확인에도 쓰인다(일정을 만들거나 바꾸지 않는다)."""
    try:
        params = {
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": max(1, min(int(max_results), 50)),
            "timeMin": rfc3339(time_min) if time_min else datetime.now(KST).isoformat(timespec="seconds"),
        }
        if time_max:
            params["timeMax"] = rfc3339(time_max, end_of_day=True)
        data = google_request("GET", "/events", SCOPE_READ, params=params)
    except CalendarError as err:
        return failure(err)
    events = [simplify(e) for e in data.get("items", [])]
    return {"ok": True, "error_code": None, "message": f"일정 {len(events)}건", "events": events}


@mcp.tool()
def get_event(event_id: str) -> dict:
    """[다시 조회] 일정 하나를 id 로 다시 읽는다(생성한 일정이 실제로 있는지 확인할 때). 읽기 전용."""
    try:
        data = google_request("GET", f"/events/{quote(event_id, safe='')}", SCOPE_READ)
    except CalendarError as err:
        return failure(err)
    return {"ok": True, "error_code": None, "message": "일정을 찾았어요", "event": simplify(data)}


@mcp.tool()
def create_event(
    title: str, start: str, end: str, description: str = "", location: str = "", time_zone: str = "Asia/Seoul"
) -> dict:
    """[생성] 연습용 캘린더에 일정을 만든다. start·end 는 YYYY-MM-DD(종일) 또는 RFC3339 시각.
    사용자가 확인한 일정만 만들어야 한다."""

    def when(value: str) -> dict:
        value = value.strip()
        return {"date": value} if len(value) == 10 else {"dateTime": value, "timeZone": time_zone or TIMEZONE}

    body = {"summary": title, "start": when(start), "end": when(end)}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    try:
        data = google_request("POST", "/events", SCOPE_WRITE, json=body)
    except CalendarError as err:
        return failure(err)
    return {"ok": True, "error_code": None, "message": "일정을 만들었어요", "event": simplify(data)}


if __name__ == "__main__":
    mcp.run()  # stdio
