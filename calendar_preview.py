"""여행 계획 → 일정 미리보기. 등록 전에 사람이 보고 고칠 수 있게 행으로 바꾸고, 행마다 검사한다.

이 모듈은 캘린더에 아무것도 만들지 않는다(등록은 다음 단계). Streamlit 에 의존하지 않아 단독으로 시험할 수 있다.
행: {"포함", "제목", "날짜", "시작", "끝", "시간대", "장소", "링크"}
"""

import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = "Asia/Seoul"
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")  # 24시간 HH:MM
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
STOP_AT_LINK = re.compile(r"[\[🔎]")


def _plain(text: str) -> str:
    """마크다운 강조·이모지 표시를 걷어 낸 글."""
    return re.sub(r"[*_`💡🔎]", "", text).strip(" ·|\t")


def rows_from_plan(plan_text: str, date: str, tz: str = DEFAULT_TIMEZONE) -> list[dict]:
    """계획의 '방문 순서' 번호 항목(🔎 장소 링크가 있는 줄)을 일정 행으로 바꾼다.
    시간은 💡 와 🔎 사이의 글에서 가져오고, 읽지 못하면 원문 그대로(또는 빈칸) 두어 검사가 잡도록 한다."""
    rows = []
    for line in plan_text.splitlines():
        m = re.match(r"^\s*\d+[.)]\s*(.*)$", line)
        if not m:
            continue
        body = m.group(1)
        link = LINK_RE.search(body)
        if not link:
            continue
        name, url = link.group(1).strip(), link.group(2)
        before = body[: link.start()]
        when = _plain(before.split("🔎")[0])  # 장소 앞의 시간대 글 (예: '09:30~11:00')
        start, end = "", ""
        clock = re.search(r"(\d{1,2}:\d{2})\s*[~\-–]\s*(\d{1,2}:\d{2})", when)
        if clock:  # '13:00~14:30 식사'처럼 뒤에 활동 이름이 붙어도 시각만 가져온다
            start, end = clock.group(1), clock.group(2)
        elif re.search(r"[~\-–]", when):
            start, end = (_plain(x) for x in re.split(r"\s*[~\-–]\s*", when, maxsplit=1))
        elif when:
            start = when  # 끝이 없는 시간: 끝은 빈칸으로 남겨 '시간 누락'으로 잡는다
        address = _plain(body[link.end() :])
        rows.append(
            {"포함": True, "제목": name, "날짜": date, "시작": start, "끝": end, "시간대": tz, "장소": address, "링크": url}
        )
    return rows


def to_event(row: dict) -> dict:
    """검사를 통과한 행을 MCP 생성 도구에 넘길 값으로 바꾼다. 시각은 행의 시간대 기준 RFC3339(오프셋 포함)."""
    tz = ZoneInfo(row["시간대"])
    day = datetime.strptime(row["날짜"], "%Y-%m-%d").date()

    def at(hhmm: str) -> str:
        h, m = hhmm.split(":")
        return datetime(day.year, day.month, day.day, int(h), int(m), tzinfo=tz).isoformat(timespec="seconds")

    link = str(row.get("링크") or "").strip()
    return {
        "title": row["제목"].strip(),
        "start": at(row["시작"]),
        "end": at(row["끝"]),
        "time_zone": row["시간대"],
        "location": str(row.get("장소") or "").strip(),
        "description": "여행 챗봇이 만든 일정" + (f"\n카카오맵: {link}" if link else ""),
    }


def rows_from_visits(visits: list[dict], tz: str = DEFAULT_TIMEZONE) -> list[dict]:
    """현재 여행 계획의 방문 일정(trip_plan.visits)을 미리보기 행으로 바꾼다. 계획 글이 아니라 계획 상태가 원본이다."""
    rows = []
    for v in visits:
        day, _, start = v["start"].partition(" ")
        rows.append(
            {
                "포함": v.get("included", True),
                "제목": v["title"],
                "날짜": day,
                "시작": start,
                "끝": v["end"].partition(" ")[2],
                "시간대": v.get("tz") or tz,
                "장소": v["address"],
                "링크": v.get("url") or "",
            }
        )
    return rows


def rows_to_table(rows: list[dict]) -> list[dict]:
    """미리보기 행을 trip_plan.apply_table 이 받는 모양으로 바꾼다(시작·끝은 'YYYY-MM-DD HH:MM')."""
    return [
        {
            "title": r["제목"],
            "start": f"{r['날짜']} {r['시작']}".strip(),
            "end": f"{r['날짜']} {r['끝']}".strip(),
            "location": r["장소"],
            "memo": f"장소 정보: {r.get('링크') or ''}",
            "included": bool(r.get("포함", True)),
            "tz": r.get("시간대") or DEFAULT_TIMEZONE,
        }
        for r in rows
    ]


def validate_row(row: dict) -> list[str]:
    """행 하나의 문제를 문장 목록으로 돌려준다(비어 있으면 통과). 값은 고치지 않는다."""
    problems = []
    title = str(row.get("제목") or "").strip()
    date = str(row.get("날짜") or "").strip()
    start = str(row.get("시작") or "").strip()
    end = str(row.get("끝") or "").strip()
    tz = str(row.get("시간대") or "").strip()

    if not title:
        problems.append("제목이 비어 있어요")
    if not date:
        problems.append("날짜가 비어 있어요")
    else:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            problems.append(f"날짜 형식 오류: '{date}' (YYYY-MM-DD)")
    if not start or not end:
        problems.append("시간 누락: " + ("시작·끝" if not start and not end else "시작" if not start else "끝"))
    bad_format = False
    for label, value in (("시작", start), ("끝", end)):
        if value and not TIME_RE.match(value):
            problems.append(f"{label} 시간 형식 오류: '{value}' (24시간 HH:MM, 예 09:30)")
            bad_format = True
    if start and end and not bad_format and end <= start:  # HH:MM 은 문자열 비교가 곧 시각 비교
        problems.append(f"끝({end})이 시작({start})보다 늦어야 해요")
    if not tz:
        problems.append("시간대가 비어 있어요")
    else:
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            problems.append(f"없는 시간대: '{tz}' (예: {DEFAULT_TIMEZONE})")
    return problems


def validate_rows(rows: list[dict]) -> list[list[str]]:
    return [validate_row(r) for r in rows]


def can_confirm(rows: list[dict]) -> bool:
    """'확인'을 누를 수 있는지: 포함한 행이 하나 이상이고, 포함한 모든 행에 문제가 없을 때만."""
    included = [r for r in rows if r.get("포함")]
    return bool(included) and all(not validate_row(r) for r in included)


def mask_calendar_id(calendar_id: str | None = None) -> str:
    """대상 캘린더 ID 를 일부만 보여준다(예: 'pra•••••@group.calendar.google.com')."""
    value = (os.getenv("GOOGLE_CALENDAR_ID", "") if calendar_id is None else calendar_id).strip()
    if not value:
        return "설정 안 됨 (ID 없음)"
    local, at, domain = value.partition("@")
    if at:
        shown = local[:3] if len(local) > 3 else local[:1]
        return f"{shown}•••••@{domain}"  # 가린 글자 수를 늘 같게 해서 ID 길이도 드러나지 않게 한다
    return f"{value[:3]}{'•' * 5}{value[-3:]}" if len(value) > 8 else f"{value[:1]}{'•' * 5}"
