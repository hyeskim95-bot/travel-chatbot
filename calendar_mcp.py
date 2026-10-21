"""캘린더 MCP 클라이언트: 서버를 stdio 자식 프로세스로 실행해 도구를 발견하고, 준비 상태를 따로 확인한다.

- 도구 발견(MCP 연결)만으로는 캘린더 ID·Google 인증이 준비됐는지 알 수 없다.
- 그래서 같은 서버의 '조회' 도구(읽기 전용)로 ID → 인증 정보 → 연습용 캘린더 읽기 순서로 검사한다.
  이 검사는 일정을 만들거나 바꾸지 않는다.
- 서버를 실행하지 못하면 도구 연결도 준비 확인도 모두 실패로 처리하고, 성공 표시를 하지 않는다.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# CALENDAR_SERVER_SCRIPT 로 다른 MCP 서버(예: 연습용 모의 서버)를 지정할 수 있다. 시험·시연용 설정이다.
SERVER_SCRIPT = Path(os.getenv("CALENDAR_SERVER_SCRIPT") or Path(__file__).with_name("calendar_server.py"))
EXPECTED_TOOLS = {"list_events": "조회", "get_event": "다시 조회", "create_event": "생성"}
CHECK_TOOL = "list_events"  # 준비 확인에 쓰는 읽기 전용 도구
TIMEOUT_SECONDS = 30

# 준비 상태 코드 → (화면 라벨, 안내)
READY_STATES = {
    "ready": ("준비됨", "캘린더 ID와 Google 인증이 확인됐고, 연습용 캘린더를 읽을 수 있어요."),
    "no_calendar_id": ("ID 없음", ".env 의 GOOGLE_CALENDAR_ID 를 채워 주세요."),
    "no_credentials": ("인증 정보 없음", ".env 의 GOOGLE_CREDENTIALS_FILE 에 인증 JSON 파일 경로를 적어 주세요."),
    "auth_expired": ("인증 만료", "Google 인증이 만료되었거나 무효예요. 인증 정보를 다시 발급해 주세요."),
    "permission_denied": ("권한 없음", "이 캘린더를 인증 계정과 공유했는지 확인해 주세요(읽기 권한 필요)."),
    "connection_failed": ("연결 실패 (Google)", "Google 서버에 연결하지 못했어요. 인터넷 연결을 확인해 주세요."),
    "mcp_failed": ("연결 실패 (MCP)", "캘린더 MCP 서버를 실행하지 못해 준비 상태도 확인하지 못했어요."),
}


@dataclass
class CalendarStatus:
    mcp_ok: bool  # MCP 서버를 실행하고 도구 목록을 받았는지 (Google 준비와는 별개)
    tools: list[dict] = field(default_factory=list)  # [{name, label, description}]
    mcp_error: str = ""
    ready_code: str = "mcp_failed"
    ready_detail: str = ""  # 서버가 알려준 사유(키·토큰 값은 들어 있지 않다)

    @property
    def ready(self) -> bool:
        return self.mcp_ok and self.ready_code == "ready"

    @property
    def ready_label(self) -> str:
        return READY_STATES.get(self.ready_code, (self.ready_code, ""))[0]

    @property
    def ready_hint(self) -> str:
        return READY_STATES.get(self.ready_code, ("", ""))[1]

    @property
    def missing_tools(self) -> list[str]:
        have = {t["name"] for t in self.tools}
        return [n for n in EXPECTED_TOOLS if n not in have]


def _parse_tool_result(result) -> dict:
    """call_tool 결과에서 서버가 돌려준 dict 를 꺼낸다."""
    if getattr(result, "isError", False):
        raise RuntimeError("도구 실행 오류")
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data.get("result", data) if set(data) == {"result"} else data
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise RuntimeError("도구 결과가 비어 있어요")


async def _inspect(params: StdioServerParameters) -> CalendarStatus:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [
                {"name": t.name, "label": EXPECTED_TOOLS.get(t.name, ""), "description": (t.description or "").split("\n")[0]}
                for t in listed.tools
            ]
            status = CalendarStatus(mcp_ok=True, tools=tools)
            if CHECK_TOOL not in {t["name"] for t in tools}:
                status.ready_code, status.ready_detail = "mcp_failed", f"준비 확인에 쓸 '{CHECK_TOOL}' 도구가 없어요."
                return status
            # 준비 확인: 읽기 전용 조회 1건. 서버 안에서 ID → 인증 정보 → 토큰 → 읽기 순서로 검사한다.
            data = _parse_tool_result(await session.call_tool(CHECK_TOOL, {"max_results": 1}))
            if data.get("ok"):
                status.ready_code, status.ready_detail = "ready", ""
            else:
                status.ready_code = data.get("error_code") or "api_error"
                status.ready_detail = data.get("message", "")
            return status


def inspect_calendar(
    extra_env: dict[str, str] | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> CalendarStatus:
    """서버를 실행해 도구를 발견하고 준비 상태를 확인한다. 어떤 실패도 예외로 내보내지 않는다.
    extra_env·command·args 는 시험용으로 서버 환경이나 실행 명령을 바꿀 때 쓴다."""
    params = StdioServerParameters(
        command=command or sys.executable,
        args=args if args is not None else [str(SERVER_SCRIPT)],
        env=extra_env,
        cwd=str(SERVER_SCRIPT.parent),
    )
    try:
        return asyncio.run(asyncio.wait_for(_inspect(params), timeout))
    except BaseException as err:  # 서버 실행 실패, 시간 초과, 통신 오류 모두 'MCP 연결 실패'
        if isinstance(err, KeyboardInterrupt):
            raise
        while isinstance(err, BaseExceptionGroup) and err.exceptions:  # anyio 가 감싼 예외는 안쪽 원인을 보여준다
            err = err.exceptions[0]
        reason = "시간 초과" if isinstance(err, (asyncio.TimeoutError, TimeoutError)) else type(err).__name__
        return CalendarStatus(mcp_ok=False, mcp_error=f"서버를 실행하지 못했어요 ({reason})", ready_code="mcp_failed")


# --- 등록: 생성 도구로 만들고, '다시 조회' 도구로 읽어 와 대조한다 ---

def _same_instant(a: str, b: str) -> bool:
    from datetime import datetime

    try:
        return datetime.fromisoformat(a) == datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return a == b


async def _register(params: StdioServerParameters, rows: list[dict]):
    import calendar_preview
    from registration import RowOutcome

    outcomes = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for row in rows:
                ev = calendar_preview.to_event(row)
                made = _parse_tool_result(await session.call_tool("create_event", ev))
                if not made.get("ok"):
                    outcomes.append(RowOutcome(row["제목"], "failed", message=made.get("message", ""), mismatch=made.get("error_code") or ""))
                    continue
                event_id = (made.get("event") or {}).get("id", "")
                # 다시 조회: 방금 만든 일정이 실제로 있고 내용이 같은지 읽어 와 대조한다.
                back = _parse_tool_result(await session.call_tool("get_event", {"event_id": event_id}))
                if not back.get("ok"):
                    outcomes.append(RowOutcome(row["제목"], "created", event_id, "만들었지만 다시 조회하지 못했어요", verified=False, mismatch="다시 조회 실패"))
                    continue
                got = back["event"]
                problems = []
                if (got.get("title") or "").strip() != ev["title"]:
                    problems.append(f"제목 {got.get('title')!r}")
                if not _same_instant(got.get("start"), ev["start"]):
                    problems.append(f"시작 {got.get('start')}")
                if not _same_instant(got.get("end"), ev["end"]):
                    problems.append(f"끝 {got.get('end')}")
                outcomes.append(
                    RowOutcome(row["제목"], "created", event_id, "다시 조회해 일치", verified=not problems, mismatch=", ".join(problems))
                )
    return outcomes


def register_events(
    rows: list[dict],
    extra_env: dict[str, str] | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    timeout: float = 90,
):
    """검사를 통과한 행들을 만들고 다시 조회해 대조한다. 서버를 실행하지 못하면 모두 실패로 돌려준다(예외를 내보내지 않는다)."""
    from registration import RowOutcome

    params = StdioServerParameters(
        command=command or sys.executable,
        args=args if args is not None else [str(SERVER_SCRIPT)],
        env=extra_env,
        cwd=str(SERVER_SCRIPT.parent),
    )
    try:
        return asyncio.run(asyncio.wait_for(_register(params, rows), timeout))
    except BaseException as err:
        if isinstance(err, KeyboardInterrupt):
            raise
        while isinstance(err, BaseExceptionGroup) and err.exceptions:
            err = err.exceptions[0]
        return [RowOutcome(r["제목"], "failed", message=f"MCP 서버를 쓰지 못했어요 ({type(err).__name__})") for r in rows]
