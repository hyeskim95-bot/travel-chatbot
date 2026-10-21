"""여행 계획에 쓰는 작은 도우미: 날짜 표기, 제안 시간대, 공통 조건.

5일차 시작 코드(planner.py)의 규칙을 이 앱의 조사 흐름(trip.py)에 맞게 옮긴 것이다.
장소·날씨 조회 자체는 places.py·weather.py·trip.py 가 하고, 여기에는 계획 만들기(trip_plan.py)가 같이 쓰는 값만 둔다.
"""

from datetime import date as _date

WEEKDAYS = "월화수목금토일"
SLOTS = [("10:00", "11:30"), ("13:30", "15:00"), ("15:30", "17:00")]  # 시간 범위를 정하지 않았을 때의 제안 시간대(조회 사실 아님)
TIMEZONE = "Asia/Seoul"


def date_label(day: str) -> str:
    """'2026-10-24(토)' — 요일은 코드가 계산한다(모델이 요일을 틀린 실행 기록이 있음)."""
    try:
        return f"{day}({WEEKDAYS[_date.fromisoformat(day).weekday()]})"
    except ValueError:
        return day


def make_conditions(region: str, day: str, preferences: str = "") -> dict:
    """공통 조건 — 한 곳에서 만들어 A·B 에 같은 값을 넘긴다."""
    return {"region": region.strip(), "date": day.strip(), "timezone": TIMEZONE, "preferences": (preferences or "").strip()}
