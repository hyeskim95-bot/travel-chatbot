"""현재 여행 계획 — 대화 답변 · 일정 미리보기 · MCP 등록 데이터가 함께 쓰는 하나의 상태.

5일차 시작 코드(trip_plan.py)의 규칙을 그대로 옮겼다. 계획(plan)은 대화(세션)에 하나만 있고 아래 세 경로가 모두 이 계획을 고친다.
- new_plan(새 조사): 지역·날짜·취향이 처음 정해지거나 바뀔 때. 조회 근거(basis)를 새로 만든다.
- update_plan(대화 수정): 시간 범위·방문 수·제외/다시 포함. 다시 검색하지 않고 검증된 후보 안에서 고친다.
- apply_table(표 수정): 사용자가 미리보기 표에서 직접 고친 행을 그대로 계획에 넣는다.
바뀔 때마다 version 이 오른다(이전 확인은 서명이 달라져 자동으로 무효가 된다).
적용할 수 없는 변경은 계획을 그대로 두고 이유를 돌려준다(성공으로 바꾸지 않는다).

basis 의 모양: {"cond": {region, date, timezone, preferences}, "a": {"places": [{name, address, url, category}]},
                "b": {...날씨 요약}, "check": {places_ok, weather_ok, notes}}
"""

import copy
import re

import planner

DEFAULT_COUNT = 3
DEFAULT_WINDOW = ("10:00", "17:00")  # 시간 범위가 없을 때만 쓰는 초기 제안 범위
GAP = 30  # 방문 사이 이동·휴식(분)
MIN_VISIT = 30
MAX_VISIT = 120  # 한 곳에 머무는 시간의 상한(분)


class PlanError(ValueError):
    """요청한 변경을 적용할 수 없음 — 계획은 바뀌지 않는다."""


# ───────── 시간
def norm_time(value) -> str | None:
    """'14:00'·'14시'·'2:30'·'14' → 'HH:MM'. 빈 값은 None."""
    if value is None or str(value).strip() == "":
        return None
    m = re.fullmatch(r"\s*(\d{1,2})(?:[:시]\s*(\d{1,2})?\s*분?)?\s*", str(value))
    if not m:
        raise PlanError(f"시간 형식을 알 수 없습니다: {value} (예: 14:00)")
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise PlanError(f"시간 범위를 벗어났습니다: {value}")
    return f"{h:02d}:{mi:02d}"


def _min(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _fmt(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def make_slots(n: int, start: str | None, end: str | None) -> list[tuple[str, str]]:
    """방문 n곳의 시간대. 범위를 정하지 않았고 3곳 이하면 기본 제안 시간대를 쓴다."""
    if n < 1:
        raise PlanError("방문 수는 1곳 이상이어야 합니다.")
    if start is None and end is None and n <= len(planner.SLOTS):
        return planner.SLOTS[:n]
    s, e = _min(start or DEFAULT_WINDOW[0]), _min(end or DEFAULT_WINDOW[1])
    if e <= s:
        raise PlanError(f"끝 시간({_fmt(e)})이 시작 시간({_fmt(s)})보다 늦어야 합니다.")
    total = e - s
    gap = GAP if n > 1 and (total - GAP * (n - 1)) / n >= 60 else 0
    d = (total - gap * (n - 1)) // n // 5 * 5
    if d < MIN_VISIT:
        raise PlanError(f"{_fmt(s)}–{_fmt(e)} 안에 {n}곳을 넣으면 한 곳에 {MIN_VISIT}분도 안 됩니다. 시간을 늘리거나 방문 수를 줄여 주세요.")
    capped = d > MAX_VISIT  # 시간 범위가 넓으면 한 곳을 길게 늘리지 않고, 남는 시간은 방문 사이 자유 시간으로 둔다
    if capped:
        d = MAX_VISIT
        gap = (total - n * d) // (n - 1) // 5 * 5 if n > 1 else 0
    out, t = [], s
    for i in range(n):
        last = i == n - 1
        out.append((_fmt(t), _fmt(e if last and not capped else t + d)))
        t += d + gap
    if capped and n > 1:  # 마지막 방문이 끝 시간에 맞게 끝나도록 시작을 당긴다
        out[-1] = (_fmt(e - d), _fmt(e))
    return out


# ───────── 후보·제외
def candidates(basis: dict) -> list[dict]:
    """검증된 장소 후보. 취향에 맞는 장소(group 0)가 식사·카페 같은 덧붙인 장소(group 1)보다 먼저 온다.
    각 묶음 안에서는 같은 주소(한 시설의 하위 장소)를 뒤로 미뤄 서로 다른 곳을 먼저 고른다."""
    if not basis["check"]["places_ok"]:
        return []
    out, seen = [], set()
    groups = sorted({p.get("group", 0) for p in basis["a"]["places"]})
    for g in groups:
        first, rest = [], []
        for p in basis["a"]["places"]:
            if p.get("group", 0) != g:
                continue
            (rest if p["address"] in seen else first).append({k: p.get(k) for k in ("name", "address", "url", "category")})
            seen.add(p["address"])
        out += first + rest
    return out

def _key(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def match_places(cands: list[dict], terms: list[str]) -> list[str]:
    """제외·포함할 장소 이름을 후보에서 찾는다. 없거나 여러 곳이 겹치면 적용하지 않는다."""
    names = [c["name"] for c in cands]
    found = []
    for term in terms or []:
        k = _key(term)
        if not k:
            continue
        exact = [n for n in names if _key(n) == k]
        hits = exact or [n for n in names if k in _key(n)]
        if not hits:
            raise PlanError(f"‘{term}’에 해당하는 장소가 현재 후보에 없습니다. 후보: {', '.join(names)}")
        if len(hits) > 1:
            raise PlanError(f"‘{term}’이(가) 여러 장소와 겹칩니다: {', '.join(hits)}. 어느 곳인지 정확한 이름으로 알려 주세요.")
        found += hits
    return found


def _visit(c: dict, day: str, s: str, e: str) -> dict:
    return {"place": c["name"], "title": c["name"], "address": c["address"], "url": c["url"],
            "start": f"{day} {s}", "end": f"{day} {e}", "memo": f"장소 정보: {c['url']}", "edited": False, "included": True}


def _fill(cands, keep_names, exclude, n):
    pool = [c for c in cands if c["name"] not in exclude]
    chosen = [c for name in keep_names for c in pool if c["name"] == name][:n]
    for c in pool:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    if len(chosen) < n:
        raise PlanError(f"제외 조건을 지키면 후보가 {len(chosen)}곳뿐이라 {n}곳 일정을 만들 수 없습니다. 방문 수를 줄이거나 제외를 줄여 주세요.")
    return chosen


def _ensure_tz(plan: dict) -> dict:
    """모든 방문에 시간대(tz)를 채운다. 표에서 시간대를 고치면 그 값이 행마다 남는다."""
    for v in plan["visits"]:
        v.setdefault("tz", plan["timezone"])
    return plan


# ───────── 계획 만들기·고치기
def new_plan(basis: dict, prev: dict | None = None, visit_count=None, start_time=None, end_time=None, exclude_places=None,
             override_table_edits=False) -> dict:
    """새 조사 결과로 계획을 만든다. 생략한 인자는 이전 계획의 조건을 이어받는다(사용자가 바꾸지 않은 내용 유지).
    이전 값과 같은 인자는 바뀐 것으로 보지 않는다. 표에서 고친 일정을 다시 배치해야 하면 confirm_needed.
    반환: {status: ok|not_applied|confirm_needed, message, plan}"""
    cond = basis["cond"]
    pc = (prev or {}).get("constraints", {})
    same_area = bool(prev) and prev["basis"]["cond"]["region"] == cond["region"] and prev["basis"]["cond"]["preferences"] == cond["preferences"]
    notes = []
    if not same_area:
        pc = {}  # 지역·취향이 바뀐 새 여행은 이전 여행의 시간 범위·방문 수를 물려받지 않는다(부산 반나절 → 제주 하루)
    try:
        c = {"visit_count": int(visit_count) if visit_count not in (None, "", 0) else pc.get("visit_count", DEFAULT_COUNT),
             "start_time": norm_time(start_time) if start_time not in (None, "") else pc.get("start_time"),
             "end_time": norm_time(end_time) if end_time not in (None, "") else pc.get("end_time"),
             "exclude": list(pc.get("exclude", [])) if same_area else []}
        if prev and not same_area and any(v.get("edited") for v in prev.get("visits", [])):
            notes.append("지역·취향이 바뀌어 표에서 고친 일정은 새 장소로 다시 만들었습니다.")
        prev_exclude = (prev or {}).get("constraints", {}).get("exclude")
        if prev and not same_area and prev_exclude:
            notes.append(f"지역·취향이 바뀌어 이전 제외 장소({', '.join(prev_exclude)})는 적용하지 않습니다.")
        cands = candidates(basis)
        visits = []
        if cands:
            before_excl = set(c["exclude"])
            c["exclude"] = sorted(before_excl | set(match_places(cands, exclude_places)))
            # 이전 계획과 실제로 다른 값만 '변경'으로 본다 (모델이 기존 값을 다시 넣어도 표 수정이 원복되지 않게)
            changed = bool(prev) and ((c["visit_count"], c["start_time"], c["end_time"]) != (pc.get("visit_count"), pc.get("start_time"), pc.get("end_time"))
                                      or set(c["exclude"]) != before_excl)
            edited = [v for v in (prev or {}).get("visits", []) if v.get("edited")]
            if same_area and edited and changed and not override_table_edits:
                return {"status": "confirm_needed", "plan": prev,
                        "message": "표에서 직접 고친 일정(" + ", ".join(f"{v['title']} {v['start'][-5:]}–{v['end'][-5:]}" for v in edited)
                        + ")이 있습니다. 새 조건으로 다시 배치하면 이 수정이 바뀝니다. 그래도 바꿀지 확인해 주세요."}
            if same_area and prev.get("visits") and not changed:
                # 날짜만 바뀐 경우: 장소·시간(표에서 고친 시간 포함)을 유지하고 날짜만 바꾼다
                names = {x["name"] for x in cands if x["name"] not in c["exclude"]}
                for v in prev["visits"]:
                    if v["place"] in names or v.get("edited"):
                        nv = dict(v)
                        nv["start"] = f"{cond['date']} {v['start'][-5:]}"
                        nv["end"] = f"{cond['date']} {v['end'][-5:]}"
                        visits.append(nv)
                    else:
                        notes.append(f"{v['place']}은(는) 새 조회 결과에 없어 뺐습니다.")
            if not visits:
                chosen = _fill(cands, [], c["exclude"], c["visit_count"])
                slots = make_slots(len(chosen), c["start_time"], c["end_time"])
                visits = [_visit(x, cond["date"], s, e) for x, (s, e) in zip(chosen, slots)]
    except PlanError as e:
        return {"status": "not_applied", "message": str(e), "plan": prev}
    plan = _ensure_tz({"version": (prev or {}).get("version", 0) + 1, "basis": basis, "constraints": c, "candidates": cands,
                       "visits": visits, "timezone": cond["timezone"], "notes": notes})
    return {"status": "ok", "message": " ".join(notes), "plan": plan}


def update_plan(plan: dict | None, start_time=None, end_time=None, visit_count=None, exclude_places=None,
                restore_places=None, override_table_edits=False) -> dict:
    """대화로 받은 시간 범위·방문 수·제외/다시 포함을 현재 계획에 반영한다. 다시 검색하지 않는다.
    반환: {status: updated|no_change|confirm_needed|not_applied, message, plan}"""
    if not plan or not plan.get("candidates"):
        return {"status": "not_applied", "message": "고칠 여행 계획이 없습니다. 먼저 지역과 날짜로 계획을 만들어 주세요.", "plan": plan}
    p = copy.deepcopy(plan)
    c, cands, day = p["constraints"], p["candidates"], p["basis"]["cond"]["date"]
    notes = []
    try:
        new_start = norm_time(start_time) if start_time not in (None, "") else c["start_time"]
        new_end = norm_time(end_time) if end_time not in (None, "") else c["end_time"]
        new_count = int(visit_count) if visit_count not in (None, "", 0) else c["visit_count"]
        excl_add = match_places(cands, exclude_places)
        restore = match_places(cands, restore_places)
        exclude = sorted((set(c["exclude"]) | set(excl_add)) - set(restore))
        restored = [n for n in restore if n in set(c["exclude"])]  # 실제로 제외돼 있던 장소를 다시 넣는 요청
        reschedule = (new_start, new_end, new_count) != (c["start_time"], c["end_time"], c["visit_count"]) or bool(restored)
        edited = [v for v in p["visits"] if v.get("edited")]
        if reschedule and edited and not override_table_edits:
            return {"status": "confirm_needed", "plan": plan,
                    "message": "표에서 직접 고친 일정(" + ", ".join(f"{v['title']} {v['start'][-5:]}–{v['end'][-5:]}" for v in edited)
                    + ")이 있습니다. 새 시간·방문 수로 다시 배치하면 이 수정이 바뀝니다. 그래도 바꿀지 확인해 주세요."}
        if reschedule:
            old_names = [v["place"] for v in p["visits"]]
            keep = restored + [n for n in old_names if n not in exclude and n not in restored]  # 다시 넣는 장소를 앞에 둔다
            chosen = _fill(cands, keep, exclude, new_count)
            slots = make_slots(new_count, new_start, new_end)
            p["visits"] = [_visit(x, day, s, e) for x, (s, e) in zip(chosen, slots)]
            if restored:
                dropped = [n for n in old_names if n not in {x["name"] for x in chosen}]
                notes.append(f"{', '.join(restored)}을(를) 다시 넣었습니다." + (f" 방문 수를 지키려고 {', '.join(dropped)}은(는) 뺐습니다." if dropped else ""))
        elif set(exclude) != set(c["exclude"]):
            # 제외만 바뀜: 남은 일정의 시간(표 수정 포함)은 그대로, 빠진 자리는 다음 후보로 같은 시간에 채운다
            used = {v["place"] for v in p["visits"]}
            pool = [x for x in cands if x["name"] not in exclude and x["name"] not in used]
            visits = []
            for v in p["visits"]:
                if v["place"] not in exclude:
                    visits.append(v)
                elif pool:
                    x = pool.pop(0)
                    visits.append(_visit(x, day, v["start"][-5:], v["end"][-5:]))
                    notes.append(f"{v['place']} 대신 {x['name']}을(를) 같은 시간에 넣었습니다.")
                else:
                    notes.append(f"{v['place']}을(를) 뺐고 대신 넣을 후보가 없어 일정이 줄었습니다.")
            p["visits"] = visits
            new_count = len(visits)
        else:
            return {"status": "no_change", "message": "바뀐 조건이 없어 일정을 그대로 두었습니다.", "plan": plan}
    except PlanError as e:
        return {"status": "not_applied", "message": str(e), "plan": plan}
    p["constraints"] = {"visit_count": new_count, "start_time": new_start, "end_time": new_end, "exclude": exclude}
    p["version"] += 1
    p["notes"] = notes
    return {"status": "updated", "message": " ".join(notes), "plan": _ensure_tz(p)}


# ───────── 표와 계획
def rows(plan: dict) -> list[dict]:
    return [{"title": v["title"], "start": v["start"], "end": v["end"], "location": v["address"], "memo": v["memo"],
             "included": v.get("included", True), "tz": v.get("tz", plan["timezone"])} for v in plan["visits"]]


def _clean(r: dict) -> dict:
    out = {k: ("" if r.get(k) is None else str(r.get(k)).strip()) for k in ("title", "start", "end", "location", "memo")}
    out["included"] = bool(r.get("included", True))
    out["tz"] = str(r.get("tz") or "").strip()
    return out


def apply_table(plan: dict, edited_rows: list[dict]) -> dict | None:
    """표에서 고친 내용을 계획에 넣는다. 바뀐 것이 없으면 None. 빈 행(내용을 모두 지운 행)은 뺀다."""
    filled = [_clean(r) for r in edited_rows if any(str(r.get(k) or "").strip() for k in ("title", "start", "end", "location"))]
    if filled == [_clean(r) for r in rows(plan)]:
        return None
    p = copy.deepcopy(plan)
    old = plan["visits"]
    visits = []
    for i, r in enumerate(filled):
        prev = old[i] if i < len(old) else None
        same_text = prev is not None and {k: _clean(rows(plan)[i])[k] for k in ("title", "start", "end", "location", "memo", "tz")} == {k: r[k] for k in ("title", "start", "end", "location", "memo", "tz")}
        visits.append({"place": prev["place"] if prev and r["title"] == prev["title"] else r["title"],
                       "title": r["title"], "address": r["location"], "url": prev["url"] if prev else "",
                       "start": r["start"], "end": r["end"], "memo": r["memo"],
                       "edited": (prev or {}).get("edited", False) if same_text else True,
                       "included": r["included"], "tz": r["tz"] or plan["timezone"]})
    p["visits"] = visits
    p["constraints"]["visit_count"] = len(visits)
    p["version"] += 1
    p["notes"] = ["표에서 직접 수정함"]
    return p


# ───────── 모델·화면에 보여 줄 요약
def summary_line(plan: dict | None) -> str:
    if not plan:
        return ""
    cond = plan["basis"]["cond"]
    vs = "; ".join(f"{v['start'][-5:]}–{v['end'][-5:]} {v['title']}" + (" (표에서 수정)" if v.get("edited") else "") for v in plan["visits"]) or "일정 없음"
    c = plan["constraints"]
    extra = []
    if c.get("start_time") or c.get("end_time"):
        extra.append(f"시간 범위 {c.get('start_time') or DEFAULT_WINDOW[0]}–{c.get('end_time') or DEFAULT_WINDOW[1]}")
    if c.get("exclude"):
        extra.append("제외 " + ", ".join(c["exclude"]))
    return (f"현재 계획 v{plan['version']}: {cond['region']} · {planner.date_label(cond['date'])} · {cond['preferences'] or '취향 없음'} · "
            f"{len(plan['visits'])}곳 [{vs}]" + (" · " + " · ".join(extra) if extra else ""))


def schedule_markdown(plan: dict) -> str:
    lines = [f"**일정 미리보기에 반영된 계획 (v{plan['version']})**"]
    lines += [f"- {v['start'][-5:]}–{v['end'][-5:]} {v['title']} · {v['address']}" for v in plan["visits"]] or ["- 일정 없음 (장소 조회 결과를 쓸 수 없음)"]
    return "\n".join(lines)


FAKE_HEADER = re.compile(r"\n*(?:-{3,}\n+)?\**일정 미리보기에 반영된 계획[^\n]*\n(?:[-*] [^\n]*\n?)*")


def strip_fake_block(text: str) -> str:
    """앱이 붙이는 '일정 미리보기에 반영된 계획' 블록은 코드만 만든다. 모델 답변에 같은 블록이 있으면 지운다."""
    return FAKE_HEADER.sub("\n", text or "").strip()
