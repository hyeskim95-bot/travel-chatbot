"""확인한 일정만 캘린더에 등록한다. 확인은 모델의 문장이 아니라 이 코드가 지킨다.

- 확인하면 '한 번 쓰는 표'(ticket)가 발급된다. 표에는 확인한 내용의 서명(HMAC), 소유 세션, 만료 시각이 들어 있다.
- 등록은 표를 쓰기 직전에 서버 쪽에서 다시 검사한다: 보기 전용 · 표 없음 · 소유자 아님 · 취소됨 · 이미 씀 · 만료 ·
  내용이 확인 때와 다름 · 캘린더 준비 안 됨. 하나라도 걸리면 MCP 생성 도구를 부르지 않는다.
- 표는 생성 도구를 부르기 **전에** 먼저 쓴 것으로 처리한다. 더블 클릭·재전송이 와도 두 번째는 '이미 씀'으로 거절된다.
- 같은 일정(캘린더·대화·내용이 같음)은 한 번만 만든다(registrations.json). 다시 확인해도 추가 생성이 없다.
- 등록한 일정은 MCP 의 '다시 조회' 도구로 읽어 와 제목·시작·끝을 대조한다.
Streamlit 에 의존하지 않는다.
"""

import copy
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import calendar_preview

TICKET_TTL_SECONDS = 600
REGISTERED_FILE = Path(__file__).parent / "data" / "registrations.json"
_SECRET = os.urandom(32)  # 서버 프로세스가 사는 동안만 유효한 서명 비밀값(서버를 다시 켜면 모든 표가 무효)
_lock = threading.RLock()
_tickets: dict[str, "Ticket"] = {}

SIGNED_FIELDS = ["제목", "날짜", "시작", "끝", "시간대", "장소", "링크"]  # 등록에 쓰이는 값 전부

# 거절·실패 사유 → 사용자에게 보여줄 안내
MESSAGES = {
    "registered": "등록했어요.",
    "rejected_view_only": "보기 전용 세션이라 확인·등록할 수 없어요.",
    "rejected_no_ticket": "확인하지 않은 미리보기라 등록하지 않았습니다. 먼저 '이 미리보기로 확인'을 눌러 주세요.",
    "rejected_not_owner": "이 확인을 한 세션이 아니라서 등록하지 않았습니다.",
    "rejected_cancelled": "확인을 취소했어요. 등록하지 않았습니다.",
    "rejected_used": "이미 한 번 쓴 확인이에요. 추가로 만들지 않았습니다. 다시 등록하려면 '다시 확인'이 필요해요.",
    "rejected_expired": "확인 시간이 지나 무효가 됐어요. 등록하지 않았습니다. 다시 확인해 주세요.",
    "rejected_changed": "확인한 뒤 내용이 바뀌었어요. 기존 확인을 무효로 했고 등록하지 않았습니다. 다시 확인해 주세요.",
    "rejected_invalid": "미리보기에 문제가 있는 행이 있어 확인·등록할 수 없어요.",
    "calendar_not_ready": "캘린더가 준비되지 않아 등록하지 않았습니다. (확인은 그대로 남아 있어요)",
    "failed": "등록 중 일부 또는 전부 실패했어요.",
}


@dataclass
class Ticket:
    id: str
    owner: str
    signature: str
    rows: list[dict]  # 확인한 내용의 사본: 등록은 항상 이 사본으로만 한다
    expires: float
    used: bool = False
    cancelled: bool = False
    invalidated: bool = False  # 확인 뒤 내용이 바뀌어 무효가 됨


@dataclass
class RowOutcome:
    title: str
    status: str  # created / already_registered / failed
    event_id: str = ""
    message: str = ""
    verified: bool | None = None  # 다시 조회해 대조한 결과(None: 대조하지 않음)
    mismatch: str = ""


@dataclass
class Result:
    status: str  # MESSAGES 의 키
    message: str = ""
    outcomes: list[RowOutcome] = field(default_factory=list)

    @property
    def created(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "created")

    @property
    def ok(self) -> bool:
        return self.status == "registered"


def _row_key(row: dict) -> tuple:
    return tuple(str(row.get(f, "") or "").strip() for f in SIGNED_FIELDS)


def included(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("포함")]


def signature(rows: list[dict]) -> str:
    """포함한 행의 등록용 값을 순서대로 이어 붙인 내용의 HMAC. 한 글자만 달라도 서명이 달라진다."""
    payload = json.dumps([_row_key(r) for r in included(rows)], ensure_ascii=False, separators=(",", ":"))
    return hmac.new(_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def issue_ticket(owner: str, rows: list[dict], view_only: bool = False, now: float | None = None) -> tuple[Ticket | None, str]:
    """미리보기를 확인한다. 보기 전용이거나 행에 문제가 있으면 표를 주지 않는다(서버 쪽 검사)."""
    if view_only:
        return None, "rejected_view_only"
    if not calendar_preview.can_confirm(rows):
        return None, "rejected_invalid"
    with _lock:
        for old in _tickets.values():  # 같은 소유자의 이전 표는 새 확인이 나오면 무효로 한다
            if old.owner == owner and not old.used:
                old.invalidated = True
        t = Ticket(
            id=secrets.token_urlsafe(16),
            owner=owner,
            signature=signature(rows),
            rows=copy.deepcopy(included(rows)),
            expires=(now if now is not None else time.time()) + TICKET_TTL_SECONDS,
        )
        _tickets[t.id] = t
        return t, "ok"


def cancel(ticket_id: str | None, owner: str) -> bool:
    with _lock:
        t = _tickets.get(ticket_id or "")
        if t and t.owner == owner and not t.used:
            t.cancelled = True
            return True
    return False


def _gate(ticket_id: str | None, owner: str, rows: list[dict], view_only: bool, now: float) -> tuple[str, Ticket | None]:
    """등록 전 관문. 통과하면 ('ok', ticket). 호출하는 쪽이 _lock 을 쥐고 있어야 한다."""
    if view_only:
        return "rejected_view_only", None
    t = _tickets.get(ticket_id or "")
    if t is None:
        return "rejected_no_ticket", None
    if t.owner != owner:
        return "rejected_not_owner", None
    if t.cancelled:
        return "rejected_cancelled", t
    if t.used:
        return "rejected_used", t
    if t.invalidated:
        return "rejected_changed", t
    if now > t.expires:
        return "rejected_expired", t
    if not hmac.compare_digest(signature(rows), t.signature):
        t.invalidated = True  # 확인한 내용과 현재 내용이 다르면 기존 확인을 무효화한다
        return "rejected_changed", t
    return "ok", t


def peek(ticket_id: str | None, owner: str, rows: list[dict], view_only: bool, now: float | None = None) -> str:
    """표를 쓰지 않고(무효화도 하지 않고) 지금 등록할 수 있는 상태인지만 본다. 버튼 활성 여부용: 'ok' 또는 거절 사유."""
    with _lock:
        return _gate_readonly(ticket_id, owner, rows, view_only, time.time() if now is None else now)

def _gate_readonly(ticket_id, owner, rows, view_only, now) -> str:
    if view_only:
        return "rejected_view_only"
    t = _tickets.get(ticket_id or "")
    if t is None:
        return "rejected_no_ticket"
    if t.owner != owner:
        return "rejected_not_owner"
    if t.cancelled:
        return "rejected_cancelled"
    if t.used:
        return "rejected_used"
    if t.invalidated:
        return "rejected_changed"
    if now > t.expires:
        return "rejected_expired"
    return "ok" if hmac.compare_digest(signature(rows), t.signature) else "rejected_changed"


# --- 이미 만든 일정 기록 (같은 일정을 두 번 만들지 않기 위한 것) ---

def _load_registered() -> dict:
    try:
        return json.loads(REGISTERED_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save_registered(data: dict) -> None:
    REGISTERED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTERED_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REGISTERED_FILE)


def registered_key(calendar_key: str, chat_id: str, row: dict) -> str:
    raw = json.dumps([calendar_key, chat_id, _row_key(row)], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def count_registered(calendar_key: str, chat_id: str, rows: list[dict]) -> int:
    done = _load_registered()
    return sum(1 for r in included(rows) if registered_key(calendar_key, chat_id, r) in done)


def register(
    ticket_id: str | None,
    owner: str,
    rows: list[dict],
    *,
    view_only: bool,
    calendar_ready: bool,
    chat_id: str,
    calendar_key: str,
    creator,
    now: float | None = None,
) -> Result:
    """확인한 일정만 등록한다. creator(rows) 는 MCP 로 만들고 다시 조회해 대조한 결과(RowOutcome 목록)를 돌려준다."""
    now = time.time() if now is None else now
    with _lock:
        verdict, ticket = _gate(ticket_id, owner, rows, view_only, now)
        if verdict != "ok":
            return Result(verdict, MESSAGES[verdict])
        if not calendar_ready:
            return Result("calendar_not_ready", MESSAGES["calendar_not_ready"])  # 표는 쓰지 않는다
        ticket.used = True  # 생성 도구를 부르기 전에 먼저 쓴 것으로 한다: 재전송·더블 클릭이 와도 두 번 만들지 않는다
        snapshot = copy.deepcopy(ticket.rows)  # 현재 화면이 아니라 확인한 사본으로만 만든다
        done = _load_registered()
        slots: list[RowOutcome | None] = [None] * len(snapshot)  # 결과는 확인한 행 순서대로 둔다
        todo: list[tuple[int, str, dict]] = []
        for i, row in enumerate(snapshot):
            key = registered_key(calendar_key, chat_id, row)
            if key in done:
                slots[i] = RowOutcome(row["제목"], "already_registered", done[key].get("event_id", ""), "이미 등록된 일정이라 다시 만들지 않았어요")
            else:
                todo.append((i, key, row))
        created = list(creator([r for _, _, r in todo])) if todo else []
        created += [RowOutcome(r["제목"], "failed", message="결과를 받지 못했어요") for _, _, r in todo[len(created) :]]
        for (i, key, row), out in zip(todo, created):
            slots[i] = out
            if out.status == "created":
                done[key] = {"event_id": out.event_id, "title": row["제목"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save_registered(done)
        outcomes = [o for o in slots if o is not None]
    bad = [o for o in outcomes if o.status == "failed" or o.verified is False]
    status = "failed" if bad else "registered"
    return Result(status, MESSAGES[status], outcomes)


def calendar_key_from_env() -> str:
    """대상 캘린더를 구분하는 값(ID 의 해시). 등록 기록에 ID 원문을 남기지 않는다."""
    value = os.getenv("GOOGLE_CALENDAR_ID", "").strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] if value else "no-calendar"
