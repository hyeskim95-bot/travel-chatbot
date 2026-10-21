"""대화를 data/chats.json 에 저장·불러오기 (새로고침·서버 재시작에도 유지)."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "chats.json"

# 저장 카테고리 (분류 실패·알 수 없는 값은 '기타')
CATEGORIES = ["여행 일정", "맛집·카페", "숙박", "교통", "준비물·팁", "기타"]
DEFAULT_CATEGORY = "기타"


def _load_all() -> list[dict]:
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8")).get("chats", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_all(chats: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"chats": chats}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, DATA_FILE)  # 쓰는 도중 꺼져도 기존 파일이 깨지지 않게 교체


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def list_chats() -> list[dict]:
    """최근 수정 순."""
    return sorted(_load_all(), key=lambda c: c.get("updated", ""), reverse=True)


def get_chat(chat_id: str) -> dict | None:
    return next((c for c in _load_all() if c["id"] == chat_id), None)


def save_chat(chat_id: str, messages: list[dict], title: str = "", category: str = "") -> None:
    """대화를 저장한다. title·category 가 비어 있으면 기존 값을 유지한다."""
    chats = _load_all()
    now = datetime.now().isoformat(timespec="seconds")
    chat = next((c for c in chats if c["id"] == chat_id), None)
    if chat is None:
        chat = {"id": chat_id, "title": "", "category": DEFAULT_CATEGORY, "created": now}
        chats.append(chat)
    if title:
        chat["title"] = title
    if category:
        chat["category"] = category if category in CATEGORIES else DEFAULT_CATEGORY
    chat["messages"] = messages
    chat["updated"] = now
    _save_all(chats)


def delete_chat(chat_id: str) -> None:
    _save_all([c for c in _load_all() if c["id"] != chat_id])
