"""여행 결과를 화면용 항목으로 정리한다(Streamlit 에 의존하지 않는 순수 함수).

입력은 앱이 이미 만들어 저장한 조회 결과(trip 기록: 현재 여행 계획 · 장소 후보 · 날씨 원결과)뿐이다.
- 값을 새로 만들거나 고쳐 쓰지 않는다. 없는 값은 None 으로 두고 'missing' 목록에 어떤 항목이 비었는지 적는다.
- 모델 문장(답변 글)은 읽지 않는다. 표시를 바꾸거나 펼쳐도 모델을 다시 부르지 않고 같은 저장 값을 다시 그린다.
"""

import planner

WEATHER_REASONS = {
    "out_of_range": "예보 범위(오늘부터 16일) 밖이라 예보가 없어요",
    "date_mismatch": "조회된 날짜가 요청과 달라 쓰지 않았어요",
    "error": "날씨 조회에 실패했어요",
}


def _tail(path: str | None) -> str | None:
    """'음식점 > 한식 > 칼국수' → '칼국수'. 비어 있으면 None."""
    if not path:
        return None
    return path.split(" > ")[-1].strip() or None


def _clean(value) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def build_view(trip: dict | None) -> dict | None:
    """trip 기록 → 화면 항목. 계획이 없는 기록(예전 대화·되묻기·조사만)이면 None."""
    if not trip or not trip.get("complete") or not trip.get("synthesized", True):
        return None
    plan = trip.get("plan")
    if not plan:
        return None
    cond, c = plan["basis"]["cond"], plan["constraints"]
    info = {x["name"]: x for x in plan.get("candidates", [])}  # 장소 이름 → 분류·주소·링크 (조회된 후보)

    visits = []
    for i, v in enumerate(plan["visits"], 1):
        cand = info.get(v["place"]) or info.get(v["title"]) or {}
        row = {
            "order": i,
            "time": f"{v['start'][-5:]}~{v['end'][-5:]}",
            "start": v["start"][-5:],
            "end": v["end"][-5:],
            "title": v["title"],
            "address": _clean(v.get("address")) or _clean(cand.get("address")),
            "url": _clean(v.get("url")) or _clean(cand.get("url")),
            "category": _tail(cand.get("category")),
            "edited": bool(v.get("edited")),
            "included": bool(v.get("included", True)),
        }
        row["missing"] = [label for key, label in (("address", "주소"), ("url", "링크"), ("category", "분류")) if not row[key]]  # 비어 있는 항목
        visits.append(row)

    res_b = (trip.get("results") or {}).get("B") or {}
    data = res_b.get("data") or {}
    status = res_b.get("status") or "error"
    weather = {"status": status, "source": res_b.get("source"), "reason": None}
    if status == "ok":
        weather.update(
            date=data.get("date"),
            weekday=data.get("weekday"),
            date_label=planner.date_label(data["date"]) if data.get("date") else None,
            tmax_c=data.get("tmax_c"),
            tmin_c=data.get("tmin_c"),
            pop_percent=data.get("pop_percent"),
            place=_clean(data.get("place")),
            fetched_at=data.get("fetched_at"),
            coord_source=_clean(data.get("coord_source")),
            packing=list(data.get("packing") or []),
        )
        weather["missing"] = [
            label for key, label in (("tmax_c", "최고 기온"), ("tmin_c", "최저 기온"), ("pop_percent", "강수확률")) if weather.get(key) is None
        ]
    else:
        weather.update(
            date_label=planner.date_label(data["date"]) if data.get("date") else None,
            reason=WEATHER_REASONS.get(status, "날씨를 쓰지 않았어요"),
            range=(f"{data['range_start']} ~ {data['range_end']}" if data.get("range_start") else None),
            packing=[],
            missing=["날씨 값 전체"],
        )

    parts = [("지역", cond["region"]), ("날짜", planner.date_label(cond["date"])), ("취향", cond["preferences"] or "없음")]
    parts.append(("방문", f"{len(visits)}곳"))
    if c.get("start_time") or c.get("end_time"):
        parts.append(("시간 범위", f"{c.get('start_time') or '제한 없음'}~{c.get('end_time') or '제한 없음'}"))
    if c.get("exclude"):
        parts.append(("제외", ", ".join(c["exclude"])))

    notes = list(plan.get("notes") or []) + list(plan["basis"]["check"].get("notes") or [])
    verification = trip.get("verification") or {}
    notes += [f"불일치: {w}" for w in verification.get("warnings", [])]
    seen, unique_notes = set(), []
    for n in notes:
        if n not in seen:
            seen.add(n)
            unique_notes.append(n)

    a = plan["basis"]["a"]
    return {
        "version": plan["version"],
        "summary": parts,
        "visits": visits,
        "weather": weather,
        "notes": unique_notes,
        "has_visits": bool(visits),
        "sources": {
            "places": {"source": a.get("source"), "retrieved_at": a.get("retrieved_at"), "status": a.get("status")},
            "weather": {"source": res_b.get("source"), "retrieved_at": data.get("fetched_at"), "status": status},
        },
    }
