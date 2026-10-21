"""여행 계획 오케스트레이터.

플래너가 공통 조건(지역·여행 날짜·취향) 하나를 만들고, 코드가 정한 순서(A → B → 작성)로
- A 장소 조사: 지역·취향을 받아 장소를 검색하고,
- B 날씨·준비물 조사: 같은 지역·날짜를 받아 날씨를 조회한다.
A·B 는 독립 에이전트가 아니라 코드가 순서대로 부르는 함수이며, 같은 조건 객체에서 값을 꺼내 쓴다.
지역을 바꿔 다시 조사하면 조건을 다시 만들어 A·B 에 똑같이 넘긴다.
"""

import copy
import re
from dataclasses import asdict, dataclass
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import agent
import config
import places
import plan_writer
import planner
import trip_plan
import verify
import weather

# 플래너를 거칠 만한 요청인지 걸러 내는 표현 (그 외 질문은 기존 에이전트가 그대로 처리한다)
TRIP_HINT = re.compile(r"계획|일정|코스|플랜|\d\s*박\s*\d\s*일|당일치기|반나절|하루\s*여행")  # '여행을 계획해줘'처럼 조사가 끼어도 잡는다
CHANGE_HINT = re.compile(
    r"바꿔|바꾸|변경|수정|다시\s*(조사|짜|찾|알아)|말고|대신|(으)?로\s*(가자|갈래|할래)|[한두세네\d]\s*곳|오전|오후|\d\s*시|빼|제외|줄여|늘려"
)
# 사용자가 일부러 다시 검색하라고 한 경우(실패를 고친 뒤 다시 조사 등). 이때는 이전 조사 결과를 재사용하지 않는다.
RESEARCH_AGAIN = re.compile(r"다시\s*(조사|찾|검색|알아)|재조사|재검색")


@dataclass(frozen=True)
class TripCondition:
    """플래너가 한 번 만드는 공통 조건. A·B 는 이 객체에서 각자 필요한 값만 꺼내 쓴다."""

    region: str
    date: str  # YYYY-MM-DD (여행 첫날)
    preference: str = ""

    @property
    def weekday(self) -> str:
        return weather.weekday_name(datetime.strptime(self.date, "%Y-%m-%d").date())


class PlanFields(BaseModel):
    is_trip_plan: bool = Field(description="여행 계획·일정을 세우거나, 이전 여행 계획의 조건(지역·날짜·취향·방문 수·시간 범위·제외 장소)을 바꿔 달라는 요청이면 true")
    region: str = Field(default="", description="이번 대화에서 사용자가 새로 말했거나 바꾼 지역. 없으면 빈 문자열")
    date: str = Field(default="", description="이번 대화에서 말한 여행 날짜 표현 그대로(예: '이번 주 토요일'). 없으면 빈 문자열")
    preference: str = Field(default="", description="이번 대화에서 말한 취향·테마(예: '역사 탐방', '조용한 카페'). 없으면 빈 문자열")
    visit_count: int = Field(default=0, description="방문할 장소 수를 바꾸라고 했으면 그 수(한 곳=1, 두 곳=2). 아니면 0")
    start_time: str = Field(default="", description="사용자가 방문 시간대를 말했으면 그 범위의 시작을 24시간 HH:MM 으로(오후 1시=13:00, 막연히 '오후'=13:00, '오전'=09:00, '저녁'=18:00). 말하지 않았으면 빈 문자열")
    end_time: str = Field(default="", description="사용자가 방문 시간대를 말했으면 그 범위의 끝을 24시간 HH:MM 으로(오후 5시=17:00, 막연히 '오후'=18:00, '오전'=12:00, '저녁'=21:00). 말하지 않았으면 빈 문자열")
    exclude_places: list[str] = Field(default_factory=list, description="빼 달라고 한 장소 이름들. 없으면 빈 목록")
    restore_places: list[str] = Field(default_factory=list, description="앞서 뺐던 장소를 다시 넣어 달라고 한 이름들. 없으면 빈 목록")
    answers_pending: bool = Field(
        default=False,
        description="직전 챗봇 메시지가 여행 지역·날짜를 되묻는 질문이고, 마지막 사용자 메시지가 그 질문에 대한 답이면 true. 다른 질문·잡담·인사면 false",
    )


def prior_trip(history: list[dict]) -> dict | None:
    """가장 최근 여행 계획 기록: {"condition": {...}, "complete": bool, "plan": {...}}. 확인 질문 기록은 건너뛴다."""
    for m in reversed(history[:-1]):
        if m["role"] == "assistant" and m.get("trip") and "pending_confirm" not in m["trip"]:
            return m["trip"]
    return None


def pending_confirm(history: list[dict]) -> dict | None:
    """바로 앞 챗봇 답이 '표에서 고친 일정을 바꿔도 되나요?' 같은 확인 질문이었으면 그때 요청한 값."""
    for m in reversed(history[:-1]):
        if m["role"] == "assistant":
            return (m.get("trip") or {}).get("pending_confirm")
    return None


def wants_planner(history: list[dict]) -> bool:
    last = history[-1]["content"]
    prior = prior_trip(history)
    if prior and not prior.get("complete"):
        return True  # 지역·날짜를 되물은 직후의 짧은 답("경주요")도 플래너가 받는다
    # 현재 계획이 있을 때는 계획을 바꾸라는 말로 보이는 문장(다시 넣어 줘·한 곳만·오후 2시로 등)도 플래너가 먼저 읽어 본다.
    return bool(TRIP_HINT.search(last) or (prior and (CHANGE_HINT.search(last) or agent.looks_like_change(last))))


def extract(history: list[dict], prior: dict | None, hint: str = "") -> PlanFields:
    today = datetime.now().date()
    recent = "\n".join(f"{'사용자' if m['role'] == 'user' else '챗봇'}: {m['content'][:200]}" for m in history[-6:])
    llm = ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, temperature=config.TEMPERATURE)
    plan_line = trip_plan.summary_line((prior or {}).get("plan"))
    prompt = (
        f"오늘은 {today.isoformat()}({weather.weekday_name(today)}요일)입니다.\n"
        f"날짜 달력: {agent.calendar_text(today)}\n"
        f"이전 여행 조건: {prior['condition'] if prior else '없음'}\n"
        + (f"{plan_line}\n" if plan_line else "")
        + f"\n최근 대화:\n{recent}\n\n"
        + agent.with_date_hints(history[-1]["content"], today).replace(history[-1]["content"], "", 1).strip()
        + "\n마지막 사용자 메시지에서 여행 조건을 뽑으세요. region·date·preference 는 사용자가 이번에 새로 말했거나 바꾼 값만 채우고, "
        "말하지 않았으면 빈 문자열로 두세요(이전 조건을 베껴 쓰지 마세요). 사용자가 말하지 않은 지역·날짜를 추측해서 채우지 마세요. "
        "'경주요'처럼 앞선 질문에 대한 짧은 답이면 그 값을 해당 항목에 채우고 answers_pending 을 true 로 하세요. "
        "인사나 다른 질문(예: '경주 박물관 어디 있어?')은 answers_pending 을 false 로 하세요. "
        "방문 수·시간 범위·제외/다시 포함 장소를 바꾸라는 말이면 visit_count·start_time·end_time·exclude_places·restore_places 를 채우세요"
        "(바꾸라고 하지 않은 값은 비워 두세요). " + hint
    )
    return llm.with_structured_output(PlanFields).invoke([HumanMessage(content=prompt)])


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def build_condition(fields: PlanFields, prior: dict | None, history: list[dict] | None = None) -> tuple[TripCondition | None, dict, str]:
    """새로 말한 값과 이전 조건을 합쳐 공통 조건을 만든다. (조건 또는 None, 부분 조건, 되묻는 말)
    사용자가 말하지 않은 지역·날짜를 모델이 채워 넣었으면(추측) 버리고 되묻는다."""
    old = (prior or {}).get("condition", {})
    today = datetime.now().date()
    said = _norm(" ".join(m["content"] for m in (history or []) if m["role"] == "user"))
    said_dates = {d[1][:10] for d in agent.date_hints(" ".join(m["content"] for m in (history or []) if m["role"] == "user"), today)}
    region = fields.region.strip()
    if history is not None and region and _norm(region) not in said and region != old.get("region"):
        region = ""  # 사용자가 한 번도 말하지 않은 지역: 추측이므로 쓰지 않는다
    region = region or old.get("region", "")
    preference = fields.preference.strip() or old.get("preference", "")
    date_iso, bad_date = old.get("date", ""), False
    if fields.date.strip():
        resolved = weather.resolve_date(fields.date, today)
        if resolved is None:
            bad_date = True
        elif history is not None and resolved.isoformat() not in said_dates and _norm(fields.date) not in said and resolved.isoformat() != old.get("date"):
            pass  # 말하지 않은 날짜(추측)는 쓰지 않고 이전 값을 유지한다
        else:
            date_iso = resolved.isoformat()
    partial = {"region": region, "date": date_iso, "preference": preference}
    if region and date_iso and not bad_date:
        return TripCondition(region, date_iso, preference), partial, ""
    if bad_date:
        ask = f"'{fields.date}' 날짜를 이해하지 못했어요. 여행 날짜를 알려주세요. (예: 이번 주 토요일, 10월 24일)"
    elif not region and not date_iso:
        ask = "여행 계획을 세우려면 지역과 여행 날짜가 필요해요. 어느 지역으로, 언제 가시나요? (취향이 있으면 함께 알려주세요)"
    elif not region:
        ask = "어느 지역으로 여행을 가시나요?"
    else:
        ask = f"{region} 여행은 언제 가시나요? 날씨와 준비물을 맞추려면 날짜가 필요해요. (예: 이번 주 토요일)"
    return None, partial, ask


def options_text(opts: dict) -> str:
    """계획 조건(방문 수·시간 범위·제외 장소)을 한 줄로."""
    parts = []
    if opts.get("visit_count"):
        parts.append(f"방문 {opts['visit_count']}곳")
    if opts.get("start_time") or opts.get("end_time"):
        parts.append(f"시간 범위 {opts.get('start_time') or '시작 제한 없음'}~{opts.get('end_time') or '끝 제한 없음'}")
    if opts.get("exclude"):
        parts.append("제외 " + ", ".join(opts["exclude"]))
    return " · ".join(parts)


# --- 역할 함수: 코드가 정한 순서(A → B)로 부른다. 독립 에이전트가 아니다 ---

# 취향 표현 → 카카오맵에서 실제로 검색되는 장소 종류 (최대 2개). 목록에 없는 취향은 그 말을 그대로 검색어로 쓴다.
PREF_KEYWORDS = {
    "역사": ["유적지", "박물관"],
    "문화": ["박물관", "전시관"],
    "예술": ["미술관", "전시관"],
    "자연": ["공원", "해변"],
    "힐링": ["공원", "산책로"],
    "쇼핑": ["시장", "쇼핑몰"],
    "체험": ["체험"],
    "사진": ["포토스팟", "전망대"],
}


# 취향 문장에 이미 장소 종류가 들어 있으면(예: '박물관 중심') 그 종류를 그대로 검색한다.
KNOWN_KINDS = ("박물관", "미술관", "전시관", "기념관", "과학관", "수족관", "공원", "카페", "맛집", "식당", "시장", "사찰", "해수욕장", "유적지", "명소")
FILLER = ("중심으로", "위주로", "중심", "위주", "관련", "탐방", "투어", "코스", "여행")  # 검색어에 넣으면 0건이 되는 말
EXCLUDE_CATEGORY = ("주차장", "기념품")  # 일정 후보가 아닌 분류


def pref_keywords(preference: str) -> list[str]:
    if not preference:
        return []
    kinds = [k for k in KNOWN_KINDS if k in preference][:2]
    if kinds:
        return kinds
    for word, kind in {"실내": "박물관", "실외": "공원", "야외": "공원"}.items():  # 장소 조건은 장소 종류로 바꾼다
        if word in preference:
            return [kind]
    for word, kws in PREF_KEYWORDS.items():
        if word in preference:
            return kws
    cleaned = preference
    for f in FILLER:
        cleaned = cleaned.replace(f, " ")
    cleaned = " ".join(cleaned.split())
    return [cleaned] if cleaned else []  # 조건어는 검색 직전에 places.clean_query 가 걸러 낸다


def research_places(cond: TripCondition) -> dict:
    """A 장소 조사. 지역·취향을 받아 장소를 검색한다."""
    # A 도 B 와 같은 지역·날짜를 공통 조건에서 받는다(날짜는 장소 검색에 쓰지 않지만 입력으로 기록한다).
    inp = {"region": cond.region, "date": cond.date, "preference": cond.preference}
    # 검색 횟수는 최대 4회(= MAX_PLAN_TOOL_CALLS - 날씨 1회). 취향이 있으면 취향 검색이 관광지 검색을 대신한다.
    pref_queries = [f"{cond.region} {kw}" for kw in pref_keywords(cond.preference)]
    # (검색어, 분류, 묶음) — 묶음 0 은 취향·관광지(일정 후보를 넉넉히 10곳), 묶음 1 은 식사·카페(일정이 모자랄 때만 쓰는 덧붙임)
    plan = [(q, "", 0) for q in pref_queries] or [(f"{cond.region} 관광지", "관광명소", 0)]
    plan += [(f"{cond.region} 맛집", "맛집", 1), (f"{cond.region} 카페", "카페", 1)]
    searches, found = [], {}
    auth_error = False
    pref_found = 0
    for query, category, group in plan:
        cleaned = places.clean_query(query)
        try:
            items = places.search_kakao(cleaned, category, 10 if group == 0 else 5)
            status = "success"
        except places.PlaceAuthError:
            # 키·설정 문제는 다시 시도해도 소용없다: 같은 호출을 반복하지 않고 A 만 실패로 기록한다(B 는 계속 조사).
            searches.append({"query": cleaned, "category": category, "status": "auth_error", "results": []})
            auth_error = True
            break
        except places.PlaceSearchError:
            items, status = [], "error"
        items = [{**p, "group": group} for p in items if not any(x in (p.get("category") or "") for x in EXCLUDE_CATEGORY)]
        for p in items:
            found.setdefault(p["url"] or p["name"], p)
        if query in pref_queries:
            pref_found += len(items)
        searches.append({"query": cleaned, "category": category, "status": status, "results": items})
    if cond.preference and pref_queries and pref_found == 0 and not auth_error:
        # 취향 검색이 모두 0건이면 지역은 그대로 두고 일반 관광지로 한 번만 넓힌다(같은 검색 반복 없음).
        query, category = f"{cond.region} 관광지", "관광명소"
        try:
            items = [{**p, "group": 0} for p in places.search_kakao(places.clean_query(query), category, 10) if not any(x in (p.get("category") or "") for x in EXCLUDE_CATEGORY)]
            status = "success"
        except places.PlaceAuthError:
            items, status, auth_error = [], "auth_error", True
        except places.PlaceSearchError:
            items, status = [], "error"
        for p in items:
            found.setdefault(p["url"] or p["name"], p)
        searches.append({"query": places.clean_query(query), "category": category, "status": status, "results": items, "relaxed": True})
    return {
        "role": "A 장소 조사",
        "input": inp,
        "searches": searches,
        "places": list(found.values()),
        "auth_error": auth_error,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
    }


def packing_advice(record: dict) -> list[str]:
    """날씨 값(기온·강수확률)만으로 정하는 준비물 규칙. 예보가 ok 일 때만 쓴다."""
    tips = []
    tmin, tmax, pop = record["tmin"], record["tmax"], record["pop"] or 0
    if tmin <= 5:
        tips.append("두꺼운 외투·목도리 (최저 5°C 이하)")
    elif tmin <= 12 or tmax <= 15:
        tips.append("코트·가디건 같은 겉옷")
    if tmax >= 28:
        tips.append("얇은 옷·모자·선크림 (최고 28°C 이상)")
    if tmax - tmin >= 10:
        tips.append("겹쳐 입기 (일교차 10°C 이상)")
    if pop >= 60:
        tips.append("우산·방수 신발 (강수확률 60% 이상)")
    elif pop >= 30:
        tips.append("접이식 우산 (강수확률 30% 이상)")
    return tips or ["가벼운 평상복"]


def research_weather(cond: TripCondition) -> dict:
    """B 날씨·준비물 조사. 같은 지역과 여행 날짜를 받아 날씨를 조회하고 준비물을 정한다."""
    inp = {"region": cond.region, "date": cond.date}
    try:
        # 장소 검색이 실패해도 날씨는 대체 좌표 조회로 계속 조사한다(한쪽 실패가 다른 쪽을 막지 않게).
        text, record = weather.lookup(cond.region, cond.date, fallback_geocode=True)
    except ToolException as err:
        text = f"날씨 조회 실패 (error): {err} 날씨 값을 지어내지 말고 '날씨 조회 실패 — 날씨·준비물 확인 필요'라고 쓰세요."
        return {"role": "B 날씨·준비물 조사", "input": inp, "text": text, "record": None, "packing": []}
    packing = packing_advice(record) if record["status"] == "ok" else []
    return {"role": "B 날씨·준비물 조사", "input": inp, "text": text, "record": record, "packing": packing}


SOURCE_A = "카카오맵 로컬 API"
SOURCE_B = weather.SOURCE


def result_a(a: dict) -> dict:
    """A 장소 조사 원결과를 {role, name, input, status, source, data} 모양으로 만든다."""
    searches = a["searches"]
    failed = sum(1 for s in searches if s["status"] != "success")
    if a.get("auth_error"):
        status = "auth_error"
    else:
        status = "error" if failed == len(searches) else ("ok" if a["places"] else "empty")
    return {
        "role": "A",
        "name": "장소 조사",
        "input": a["input"],
        "status": status,
        "source": SOURCE_A,
        "data": {
            "searches": [
                {"query": s["query"], "category": s["category"], "status": s["status"], "count": len(s["results"])}
                for s in searches
            ],
            "places": [
                {
                    "name": p["name"],
                    "category": p["category"],
                    "address": p["address"],
                    "url": p["url"],
                    "lat": p.get("lat", ""),
                    "lng": p.get("lng", ""),
                }
                for p in a["places"]
            ],
        },
    }


def result_b(b: dict) -> dict:
    """B 날씨·준비물 조사 원결과를 A 와 같은 모양으로 만든다. 기온은 °C, 강수확률은 % 단위."""
    rec = b["record"]
    if rec is None:
        status, data = "error", {"message": b["text"]}
    elif rec["status"] == "ok":
        status = "ok"
        data = {
            "date": rec["date"],
            "weekday": rec["weekday"],
            "tmax_c": rec["tmax"],
            "tmin_c": rec["tmin"],
            "pop_percent": rec["pop"],
            "place": rec["place"],
            "address": rec["address"],
            "lat": rec["lat"],
            "lng": rec["lng"],
            "fetched_at": rec["fetched_at"],
            "coord_source": rec.get("coord_source", ""),
            "packing": b["packing"],
        }
    elif rec["status"] == "out_of_range":
        status = "out_of_range"
        data = {k: rec[k] for k in ("date", "weekday", "reason", "range_start", "range_end")}
        data["message"] = "예보 없음"
    else:
        status = rec["status"]
        data = {k: rec.get(k) for k in ("date", "weekday", "returned_date")}
    return {"role": "B", "name": "날씨·준비물 조사", "input": b["input"], "status": status, "source": SOURCE_B, "data": data}


def describe_checks(plan: "plan_writer.Plan") -> str:
    """검사 결과를 기록에 남길 한 줄로."""
    if not plan.violations:
        return "검사 통과: 조회하지 않은 사실 없음"
    text = f"초안에서 {len(plan.violations)}건 발견 → 고쳐 다시 씀: " + " / ".join(plan.violations)
    if plan.removed:
        text += f" · 남은 {plan.removed}문장 제거"
    if plan.leftover:
        text += " · 아직 남은 문제: " + " / ".join(plan.leftover)
    return text


def sources_footer(a: dict, b: dict) -> str:
    """답변 맨 아래 출처·조회 시각 줄(코드가 만든다)."""
    rec = b.get("record") or {}
    parts = [f"장소 — 카카오 로컬 검색(조회 {a.get('retrieved_at', '?')})"]
    if rec.get("status") == "ok":
        parts.append(f"날씨 — {rec['source']}(조회 {rec['fetched_at']})")
    return "\n\n출처: " + " · ".join(parts)


def check_notes(res_a: dict, res_b: dict, verdict) -> list[str]:
    """계획 맨 위에 알릴 조회·검사 결과(근거로 쓰지 않은 것과 이유)."""
    notes = []
    for w in verdict.warnings:
        notes.append(f"조건 불일치로 근거에서 뺌: {w}")
    if res_b["status"] == "out_of_range":
        notes.append("날씨: 예보 범위 밖 — 확인 필요")
    elif res_b["status"] not in ("ok", "date_mismatch"):
        notes.append("날씨: 조회 실패 — 확인 필요")
    if res_a["status"] == "empty":
        notes.append("장소: 찾지 못함 — 조건 완화 필요")
    elif res_a["status"] != "ok":
        notes.append("장소: 조회 실패 — 장소 일정을 만들지 않음")
    return notes


def build_basis(cond: TripCondition, a: dict, b: dict, res_a: dict, res_b: dict, verdict) -> dict:
    """계획(trip_plan)이 근거로 쓰는 조회 결과 묶음. 검사를 통과한 것만 places_ok·weather_ok 로 인정한다."""
    return {
        "cond": planner.make_conditions(cond.region, cond.date, cond.preference),
        "a": {
            "places": [{k: p.get(k) for k in ("name", "address", "url", "category", "group")} for p in a["places"]],
            "status": res_a["status"],
            "source": SOURCE_A,
            "retrieved_at": a.get("retrieved_at"),
        },
        "b": {"status": res_b["status"], "data": res_b["data"], "source": SOURCE_B},
        "check": {
            "places_ok": res_a["status"] == "ok" and not verdict.exclude_places,
            "weather_ok": res_b["status"] == "ok" and not verdict.exclude_weather,
            "notes": check_notes(res_a, res_b, verdict),
        },
    }


def plan_message(status: str, message: str, plan: dict | None) -> str:
    """계획을 바꾸지 못했을 때의 답(도구 결과로 만든다: 모델이 '바꿨다'고 쓰지 못하게)."""
    if status == "no_change":
        return "바뀐 조건이 없어 일정은 그대로입니다."
    if status == "confirm_needed":
        return message
    return f"요청한 변경은 아직 적용하지 않았습니다. {message}"


def maybe_run(history: list[dict], synthesize: bool = True) -> agent.AgentResult | None:
    """여행 계획 요청이면 플래너 흐름으로 처리해 결과를 돌려주고, 아니면 None(기존 에이전트가 처리).
    synthesize=False 면 A·B 조사까지만 하고 종합(계획 작성)은 하지 않는다."""
    last = history[-1]["content"]
    pending = pending_confirm(history)
    if not (pending or wants_planner(history)):
        return None
    prior = prior_trip(history)
    override = False
    if pending:  # '표에서 고친 일정을 바꿔도 되나요?'에 대한 답: 동의했을 때만 그 요청을 덮어쓰기로 적용한다
        if not agent.is_affirmative(last):
            run = agent.run_meta([], 0, "답변")
            run.update({"mode": "planner", "grounding": "확인 거절: 일정을 바꾸지 않음"})
            return agent.AgentResult(answer="알겠어요. 표에서 고친 일정은 그대로 두고 바꾸지 않았습니다.", run=run, trip={})
        fields, override = PlanFields(**pending), True
    else:
        try:
            fields = extract(history, prior)
            plan_known = bool((prior or {}).get("plan"))
            if plan_known and not fields.is_trip_plan and agent.looks_like_change(last):
                # 계획을 바꾸라는 말로 보이는데 플래너가 놓쳤으면 한 번 더 확인한다(바뀐 줄 알고 답하는 것을 막는다)
                fields = extract(history, prior, "이 메시지는 현재 계획을 바꾸라는 요청일 가능성이 큽니다. 그렇다면 is_trip_plan=true 로 하고 바꿀 값을 채우세요.")
        except Exception:  # 추출에 실패하면 기존 에이전트로 넘긴다
            return None
        is_pending_ask = bool(prior) and not prior.get("complete")
        # 되묻는 중이라도 그 질문에 대한 답이 아니면(인사·다른 질문) 계획 흐름이 가로채지 않고 기존 에이전트가 답한다.
        if not (fields.is_trip_plan or (is_pending_ask and fields.answers_pending)):
            return None

    cond, partial, ask = build_condition(fields, prior, history)
    if cond is None:  # 필수 조건이 빠졌으면 도구를 부르지 않고 묻고 기다린다. 받은 값은 보관한다.
        run = agent.run_meta([], 1, "답변")
        run["mode"] = "planner"
        return agent.AgentResult(answer=ask, run=run, trip={"condition": partial, "complete": False})

    prev_plan = (prior or {}).get("plan") if prior and prior.get("complete") else None
    old_cond = (prior or {}).get("condition", {})
    option_request = bool(fields.visit_count or fields.start_time or fields.end_time or fields.exclude_places or fields.restore_places)
    prev_raw = (prior or {}).get("raw") if prev_plan else None
    # 지역·날짜·취향이 그대로이고 방문 수·시간·제외만 바뀌었으면 이전 조사 결과를 다시 쓴다(다시 검색하지 않는다).
    reuse = bool(
        prev_plan
        and prev_plan.get("candidates")
        and prev_raw
        and option_request
        and asdict(cond) == old_cond
        and not RESEARCH_AGAIN.search(last)
        and result_a(prev_raw["a"])["status"] == "ok"
        and result_b(prev_raw["b"])["status"] in ("ok", "out_of_range")
    )
    if reuse:
        a, b = copy.deepcopy(prev_raw["a"]), copy.deepcopy(prev_raw["b"])
    else:
        # 한쪽이 실패해도 다른 쪽 조사는 계속한다. 실패한 쪽은 추측으로 채우지 않는다.
        a = research_places(cond)  # 1) A: 지역·취향
        b = research_weather(cond)  # 2) B: 같은 지역·날짜

    # 두 역할의 입력이 같은 조건에서 나왔는지 기록한다.
    shared_region = a["input"]["region"] == b["input"]["region"] == cond.region
    shared_date = a["input"]["date"] == b["input"]["date"] == cond.date
    log: list[dict] = []
    found: dict[str, dict] = {}
    for s in a["searches"]:
        for p in s["results"]:
            found.setdefault(p["url"] or p["name"], p)
        log.append(
            {
                "role": "A",
                "tool": "search_places",
                "status": s["status"],
                "args": {"region": cond.region, "query": s["query"], "category": s["category"]},
                "results": len(s["results"]),
                "searched": s["query"],
                "names": [p["name"] for p in s["results"]],
                "keys": [p["url"] or p["name"] for p in s["results"]],
            }
        )
    record = b["record"]
    ok = bool(record) and record["status"] == "ok"
    log.append(
        {
            "role": "B",
            "tool": "get_weather",
            "status": record["status"] if record else "error",
            "args": {"region": cond.region, "date": cond.date},
            "results": 1 if ok else 0,
            "searched": record["label"] if record else f"{cond.region} {cond.date} 날씨",
            "names": [f"{record['place']} {record['date']}({record['weekday']})"] if ok else [],
            "keys": [],
        }
    )
    for i, e in enumerate(log):  # 코드가 정한 순서라 '재검색'이 아니라 '다음 호출'
        e["step"] = i + 1
        e["outcome"] = agent.outcome_of(e)
        e["next"] = "다음 호출" if i + 1 < len(log) else "답변"
    if reuse:
        log = []  # 재사용이면 이번 실행에서 부른 도구가 없다

    # 종합 직전 검증: A·B 의 지역·날짜·status 를 공통 조건과 대조한다. 통과하면 그대로, 불일치면 잘못된 쪽을 근거에서 뺀다.
    res_a, res_b = result_a(a), result_b(b)
    verdict = verify.verify(cond, res_a, res_b)

    # 현재 계획: 새 조사면 new_plan(이전 조건을 이어받음), 옵션만 바뀌면 update_plan(검증된 후보 안에서 고침)
    if reuse:
        outcome = trip_plan.update_plan(
            prev_plan, fields.start_time, fields.end_time, fields.visit_count, fields.exclude_places, fields.restore_places, override
        )
        ok_status = "updated"
    else:
        basis = build_basis(cond, a, b, res_a, res_b, verdict)
        outcome = trip_plan.new_plan(basis, prev_plan, fields.visit_count, fields.start_time, fields.end_time, fields.exclude_places, override)
        ok_status = "ok"
    if outcome["status"] != ok_status:  # 적용하지 못한 변경: 계획은 그대로 두고 이유를 알린다(성공으로 표시하지 않는다)
        run = agent.run_meta(log, 1, "답변")
        run.update({"mode": "planner", "tool_limit": config.MAX_PLAN_TOOL_CALLS, "grounding": f"계획 변경 {outcome['status']}: 일정을 바꾸지 않음"})
        new_trip = {"pending_confirm": fields.model_dump()} if outcome["status"] == "confirm_needed" else {}
        return agent.AgentResult(answer=plan_message(outcome["status"], outcome["message"], prev_plan), log=agent.strip_keys(log), run=run, trip=new_trip)
    plan = outcome["plan"]

    a_use, b_use = a, b
    if verdict.exclude_weather:
        b_use = {  # 잘못된 날씨는 근거에서 뺀다(경고 문구만 붙이지 않는다)
            **b,
            "record": None,
            "packing": [],
            "kind": "mismatch",
            "text": "날씨 사용 안 함 (불일치): " + " / ".join(verdict.warnings) + ". 날씨 값과 날씨 기반 준비물은 쓰지 말고 날씨 확인이 필요하다고만 쓰세요.",
        }
    if verdict.exclude_places:
        a_use = {**a, "places": [], "searches": []}

    if synthesize and res_a["status"] != "ok":
        # 장소 조회가 실패했으면 장소 일정을 만들지 않고, 유효한 날씨는 따로 보여준다.
        answer = verify.warning_block(verdict) + plan_writer.failure_answer(cond, res_a, res_b, b)
        used = []
        run = agent.run_meta(log, 1, "답변")
        run.update({"mode": "planner", "tool_limit": config.MAX_PLAN_TOOL_CALLS, "grounding": f"장소 조회 {res_a['status']}: 장소 일정 만들지 않음"})
    elif synthesize and verdict.exclude_places:
        answer = verify.warning_block(verdict) + "장소 조사 결과를 쓸 수 없어 계획을 만들지 않았어요. 조건을 확인하고 다시 조사해 주세요."
        used = []
        run = agent.run_meta(log, 1, "답변")
        run.update({"mode": "planner", "tool_limit": config.MAX_PLAN_TOOL_CALLS, "grounding": "검증 불일치: 장소 근거 제외"})
    elif synthesize:
        written = plan_writer.write_plan(last, cond, a_use, b_use, plan)  # 사실(🔎)·제안(💡) 구분 + 일정·미조회 사실 검사
        answer = trip_plan.strip_fake_block(written.answer) + sources_footer(a, b)
        answer = verify.warning_block(verdict) + answer
        used = [p for p in found.values() if p["name"] in answer]
        run = agent.run_meta(log, 1 + written.model_calls, "답변")  # 모델 호출: 플래너 추출 1 + 작성 1(+고쳐 쓰기 1)
        grounding = describe_checks(written)
        if written.forced:
            grounding += " · 방문 순서는 현재 계획 값으로 대체"
        if not verdict.ok:
            grounding = "검증 불일치 " + str(len(verdict.warnings)) + "건 → " + ("날씨 근거 제외" if verdict.exclude_weather else "장소 근거 제외") + " · " + grounding
        if reuse:
            grounding = "이전 조사 결과 재사용(다시 검색하지 않음) · " + grounding
        run.update({"mode": "planner", "tool_limit": config.MAX_PLAN_TOOL_CALLS, "grounding": grounding})
    else:  # 조사만: 종합하지 않고 A·B 원결과를 따로 보여준다
        answer = "🔎 **조사 단계만 실행했어요.** 아래 A·B 원결과를 먼저 확인해 보세요. (아직 종합하지 않았어요)"
        used = []
        run = agent.run_meta(log, 1, "답변")  # 모델 호출: 플래너 추출 1
        run.update({"mode": "planner(조사만)", "tool_limit": config.MAX_PLAN_TOOL_CALLS, "grounding": "종합 전"})
    trip = {
        "condition": asdict(cond),
        "complete": True,
        "A": {"role": a["role"], "input": a["input"], "searches": len(a["searches"]), "places": len(a["places"])},
        "B": {"role": b["role"], "input": b["input"], "status": record["status"] if record else "error"},
        "shared_region": shared_region,
        "shared_date": shared_date,
        "results": {"A": res_a, "B": res_b},  # 같은 모양의 원결과 두 칸
        "synthesized": synthesize,
        "verification": verdict.to_dict(),
        "plan": plan,  # 현재 여행 계획: 답변·일정 미리보기·등록이 같이 쓰는 하나의 상태
        "options": plan["constraints"],
        "reused": reuse,
        "raw": {"a": a, "b": b},  # 옵션만 바뀔 때 다시 쓰는 조사 원결과
    }
    return agent.AgentResult(
        answer=agent.linkify(answer, used),
        places=used,
        weather=[] if verdict.exclude_weather or not record else [record],  # 근거에서 뺀 날씨는 카드로도 보여주지 않는다
        log=agent.strip_keys(log),
        run=run,
        trip=trip,
    )


# --- LangSmith: 한 턴 전체를 하나의 trace 로 묶는다(LANGSMITH_TRACING=true 일 때만 전송, 꺼져 있으면 아무 일도 하지 않는다) ---
try:
    from langsmith import traceable
except ImportError:  # langsmith 가 없으면 추적 없이 동작한다

    def traceable(*_args, **_kwargs):
        return lambda fn: fn


def _trace_inputs(inputs: dict) -> dict:
    history = inputs.get("history") or []
    return {
        "question": history[-1]["content"] if history else "",
        "history_messages": len(history),
        "synthesize": inputs.get("synthesize", True),
    }


def _trace_outputs(result) -> dict:
    return {
        "answer": result.answer,
        "mode": result.run.get("mode", "agent"),
        "tool_calls": result.run.get("tool_calls"),
        "tool_log": result.log,  # 도구 이름·인자·status (키는 들어 있지 않다)
        "grounding": result.run.get("grounding"),
        "condition": (result.trip or {}).get("condition"),
        "options": (result.trip or {}).get("options"),
        "reused_research": (result.trip or {}).get("reused"),
    }


@traceable(name="travel_turn", run_type="chain", tags=["travel-chatbot"], process_inputs=_trace_inputs, process_outputs=_trace_outputs)
def run_turn(history: list[dict], synthesize: bool = True) -> agent.AgentResult:
    """한 턴 처리: 여행 계획 요청은 플래너 흐름으로, 그 밖의 질문은 기존 에이전트로."""
    result = maybe_run(history, synthesize)
    if result is not None:
        return result
    plan = (prior_trip(history) or {}).get("plan")
    if plan:  # 계획 흐름이 아닌 질문에도 앱이 현재 계획(표에서 고친 내용 포함)을 붙여 모델이 계획과 다르게 말하지 않게 한다
        last = history[-1]
        history = history[:-1] + [{**last, "content": last["content"] + "\n\n[현재 계획] " + trip_plan.summary_line(plan)}]
    return agent.run(history)
