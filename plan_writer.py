"""A(장소)·B(날씨) 조사 결과로 읽는 글 여행 계획을 쓰고, 조회하지 않은 사실이 섞이지 않았는지 검사한다.

- 🔎 조회한 사실: 장소명·주소·카카오맵 링크, 날씨 값(날짜·요일·°C·%)·출처·조회 시각. 조사 결과 그대로.
- 💡 제안: 방문 순서·예상 시간대·준비물·이유. 조회 사실을 근거로 새로 짠 것.
- 조회하지 않은 사실(영업시간·이동 시간·거리·입장료 등)은 계획에 넣지 않는다.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import calendar_preview
import config
import places
import weather

PLAN_SYSTEM = """\
당신은 조사 결과로 여행 계획을 쓰는 한국어 여행 플래너입니다. 친근하고 담백한 존댓말, 읽는 글로 쓰세요.

# 사실과 제안을 구분합니다
- 🔎 조회한 사실: 장소 이름·주소·카카오맵 링크, 날씨(날짜·요일·최고/최저 °C·강수확률 %·출처·조회 시각). 조사 결과에 있는 값을 그대로 쓰고 바꾸거나 반올림하지 마세요.
- 💡 제안: 방문 순서, 예상 시간대, 준비물, 그렇게 짠 이유. 조회 사실을 근거로 당신이 짠 것이며 반드시 💡로 표시하세요.
- 조회하지 않은 사실은 계획에 절대 넣지 마세요: 영업시간·휴무일·예약 여부·입장료·메뉴·평점, 장소 사이의 이동 시간·거리·소요 시간. 이런 것은 "방문 전 확인 필요"라고만 쓰세요.
- 장소는 [A 장소 조사 결과] 목록에 있는 것만 쓰세요. 이름·주소·링크는 목록 그대로입니다.

# 형식 (마크다운)
1. `# 🗺️ {지역} 여행 계획` 제목, 이어서 조건(지역·여행 날짜·취향)과 가정(기간·인원)을 알리는 한 문단, 그리고 범례 한 줄: `🔎 조회한 사실 · 💡 제안`
2. `## 🌤️ 날씨 (🔎 조회)` 날짜(요일), 최고·최저 기온(°C), 강수확률(%), 출처(Open-Meteo)와 조회 시각을 그대로. 날씨를 얻지 못했으면 값은 절대 쓰지 말고 이유별로 이렇게만 쓰세요: 조회 실패 → `날씨 조회 실패 — 날씨·준비물 확인 필요`, 예보 범위 밖/지난 날짜 → `예보 없음 — 확인 필요`(status 포함), 날짜·지역 불일치로 뺀 경우 → `날씨 확인 필요`. 현재 날씨나 다른 날짜의 값으로 대신하지 마세요.
3. `## 🗓️ 방문 순서 (💡 제안)` 번호 목록. 항목마다 `1. 💡 **09:30~11:00** · 🔎 [장소명](링크) · 주소` 형식으로 쓰세요(굵은 글씨 안에는 24시간 HH:MM~HH:MM 시간 범위만 쓰고, 식사·카페 같은 활동 이름은 장소 뒤 설명에 쓰세요). 이어서 그 장소를 그 순서·시간대에 넣은 이유를 한두 문장으로 쓰세요. 주소의 구·동이 같은 곳끼리 묶어 순서를 정하고, 취향에 맞는 장소를 먼저(취향이 박물관이면 박물관 위주로) 넣으세요. 식사·카페 시간은 필요할 때만 사이에 넣으세요. 강수확률이 높으면 실내 장소(박물관·전시관·카페)를 중심으로 짜세요. 장소 수는 사용자가 정했으면 그 수대로, 정하지 않았으면 4~6곳.
4. `## 🎒 준비물 (💡 제안)` 날씨 값에 근거한 항목마다 근거를 붙이세요(예: 강수확률 66% → 우산). 날씨를 얻지 못했으면 날씨 기반 준비물은 권하지 말고 `날씨·준비물 확인 필요`라고 쓰세요(장소와 무관한 일반 준비물인 물·편한 신발 정도만 덧붙일 수 있어요).
5. `## 💰 예상 경비 (💡 어림값, 조회한 값 아님)` 항목별 표와 총 예상 경비 한 줄.
6. `## ⚠️ 방문 전 확인` 영업시간·휴무일·장소 간 이동 시간은 조회하지 않았으니 확인이 필요하다고 쓰세요.
"""

# 조회하지 않은 사실의 흔적: 이동 시간, 영업시간
TRAVEL_TIME = re.compile(
    r"(도보|걸어서|차로|자차|버스|지하철|택시|대중교통|이동|운전)[^\n.]{0,15}?\d+\s*(분|시간)|\d+\s*분\s*(거리|소요|정도\s*걸|이내|걸)|약\s*\d+\s*분"
)
HOURS = re.compile(
    r"(영업|운영)\s*시간\s*[:은는]?\s*\d|\d{1,2}\s*시\s*(부터|~|-)\s*\d{1,2}\s*시\s*(까지)?\s*(영업|운영)"
    r"|(휴무|휴관)\s*일?\s*[:은는]?\s*(매주|월|화|수|목|금|토|일)요일|라스트\s*오더|\d{1,2}:\d{2}\s*[~-]\s*\d{1,2}:\d{2}\s*(영업|운영)"
)
KAKAO_LINK = re.compile(r"\]\((https?://place\.map\.kakao\.com/\d+)\)")
TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°C|℃)")
POP = re.compile(r"강수\s*확률[^\d\n]{0,8}(\d+)\s*%")


@dataclass
class Plan:
    answer: str
    violations: list[str] = field(default_factory=list)  # 처음 초안에서 찾은 문제
    revised: bool = False  # 문제를 고쳐 다시 썼는지
    removed: int = 0  # 그래도 남아 코드가 잘라 낸 문장 수
    model_calls: int = 1
    forced: bool = False  # 방문 순서를 코드가 만든 일정으로 바꿨는지
    leftover: list[str] = field(default_factory=list)  # 최종본에 남은 문제


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, temperature=config.TEMPERATURE)


def system_prompt() -> str:
    today = datetime.now().date()
    return PLAN_SYSTEM + f"\n# 오늘 날짜\n오늘은 {today.isoformat()}({weather.weekday_name(today)}요일)입니다.\n"


def format_a(a: dict) -> str:
    """A 결과를 검색별로 묶어 보여준다(같은 장소는 처음 한 번만)."""
    seen, blocks = set(), []
    for s in a["searches"]:
        lines = []
        for p in s["results"]:
            key = p["url"] or p["name"]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {p['name']} | {p['category']} | {p['address']} | {p['url']}")
        if lines:
            blocks.append(f"[검색: {s['query']}]\n" + "\n".join(lines))
    return "\n\n".join(blocks) or "(조회된 장소 없음)"


def format_b(b: dict) -> str:
    rec, ok = b["record"], b["record"] is not None and b["record"]["status"] == "ok"
    if ok:
        return (
            f"status ok | 날짜 {rec['date']}({rec['weekday']}) | 최고 {rec['tmax']} °C | 최저 {rec['tmin']} °C | "
            f"강수확률 {rec['pop']} % | 위치 {rec['place']} | 출처 {rec['source']} | 조회 시각 {rec['fetched_at']}\n"
            f"준비물 근거(규칙으로 정한 값): {', '.join(b['packing'])}"
        )
    return b["text"]  # 예보 없음(out_of_range)·날짜 불일치·오류: 사유 문구 그대로


def has_temp(answer: str, value: float) -> bool:
    """기온 값이 그대로(19.0) 또는 정수 표기(19°C)로 들어 있는지."""
    if str(value) in answer:
        return True
    return float(value).is_integer() and bool(re.search(rf"(?<![\d.]){int(value)}\s*(°C|℃)", answer))


def _key(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def visit_lines(visits: list[dict]) -> str:
    return "\n".join(
        f"{i}. {v['start'][-5:]}~{v['end'][-5:]} · {v['title']} · {v['address']} · {v.get('url') or ''}" for i, v in enumerate(visits, 1)
    )


def render_visits(visits: list[dict]) -> str:
    """코드가 만든 '방문 순서' 섹션. 모델 문장이 일정과 어긋날 때 이것으로 바꾼다."""
    out = ["## 🗓️ 방문 순서 (💡 제안)", ""]
    for i, v in enumerate(visits, 1):
        link = f"[{v['title']}]({v['url']})" if v.get("url") else v["title"]
        out.append(f"{i}. 💡 **{v['start'][-5:]}~{v['end'][-5:]}** · 🔎 {link} · {v['address']}")
    out.append("")
    return "\n".join(out)


def check_visits(answer: str, visits: list[dict]) -> list[str]:
    """계획 글의 방문 순서가 현재 계획(코드가 정한 visits)과 같은지: 장소·순서·시작·끝."""
    rows = calendar_preview.rows_from_plan(answer, "2000-01-01")
    problems = []
    if len(rows) != len(visits):
        problems.append(f"방문 일정은 {len(visits)}곳이어야 하는데 방문 순서에 {len(rows)}곳이 있어요")
    for i, (r, v) in enumerate(zip(rows, visits), 1):
        if _key(r["제목"]) != _key(v["title"]):
            problems.append(f"{i}번째 장소는 '{v['title']}'이어야 해요(현재 '{r['제목']}')")
        elif (r["시작"], r["끝"]) != (v["start"][-5:], v["end"][-5:]):
            problems.append(f"{i}번째 '{v['title']}'의 시간은 {v['start'][-5:]}~{v['end'][-5:]}이어야 해요(현재 {r['시작']}~{r['끝']})")
    return problems


def force_visits(answer: str, visits: list[dict]) -> str:
    """검사를 통과하지 못하면 방문 순서 섹션을 코드가 만든 일정으로 바꾼다(답변과 현재 계획을 맞춘다)."""
    section = render_visits(visits)
    m = re.search(r"^##[^\n]*방문 순서[^\n]*\n.*?(?=^## |\Z)", answer, flags=re.S | re.M)
    if m:
        return answer[: m.start()] + section + "\n" + answer[m.end() :]
    return answer.rstrip() + "\n\n" + section


def check(answer: str, a: dict, b: dict, visits: list[dict] | None = None) -> list[str]:
    """계획에 조회하지 않은 사실이나 틀린 값이 있는지 찾는다. 문제 설명 목록을 돌려준다."""
    problems = check_visits(answer, visits) if visits else []
    if TRAVEL_TIME.search(answer):
        problems.append("장소 사이 이동 시간·소요 시간을 썼어요(조회하지 않은 사실)")
    if HOURS.search(answer):
        problems.append("영업시간·휴무일을 썼어요(조회하지 않은 사실)")
    urls = {p["url"] for p in a["places"] if p["url"]}
    bad_links = [u for u in KAKAO_LINK.findall(answer) if u not in urls]
    if bad_links:
        problems.append(f"A 결과에 없는 장소 링크가 있어요: {', '.join(bad_links[:3])}")
    if "🔎" not in answer or "💡" not in answer:
        problems.append("🔎 조회한 사실 / 💡 제안 구분 표시가 빠졌어요")
    rec = b["record"]
    if rec is not None and rec["status"] == "ok":
        allowed = {round(float(rec["tmax"]), 1), round(float(rec["tmin"]), 1)}
        wrong = [t for t in TEMP.findall(answer) if round(float(t), 1) not in allowed]
        if wrong:
            problems.append(f"조회한 기온(최고 {rec['tmax']}°C, 최저 {rec['tmin']}°C)과 다른 값이 있어요: {', '.join(wrong)}°C")
        pops = [p for p in POP.findall(answer) if int(p) != int(rec["pop"] or 0)]
        if pops:
            problems.append(f"조회한 강수확률({rec['pop']}%)과 다른 값이 있어요: {', '.join(pops)}%")
        for value, label in ((rec["tmax"], "최고 기온"), (rec["tmin"], "최저 기온")):
            if not has_temp(answer, value):
                problems.append(f"날씨 {label} 값이 계획에 없어요")
        if f"{rec['pop']}" not in answer:
            problems.append("날씨 강수확률 값이 계획에 없어요")
        if "Open-Meteo" not in answer or "조회 시각" not in answer:
            problems.append("날씨 출처(Open-Meteo)와 조회 시각이 빠졌어요")
    else:
        leaked = [ln for ln in answer.splitlines() if TEMP.search(ln) or POP.search(ln)]
        if leaked:
            problems.append("예보가 없는데 날씨 수치(기온·강수확률)를 썼어요")
        kind = b.get("kind") or (rec["status"] if rec else "error")
        need = {"error": "조회 실패", "out_of_range": "예보 없음"}.get(kind)
        if need and need not in answer:
            problems.append(f"날씨를 얻지 못한 이유를 `{need}`로 밝히지 않았어요")
        if "확인 필요" not in answer:
            problems.append("날씨·준비물 `확인 필요` 문구가 없어요")
    return problems


def scrub(answer: str, urls: set[str] | None = None) -> tuple[str, int]:
    """고쳐 쓰고도 남은 이동 시간·영업시간 문장과 A 결과에 없는 장소 줄을 잘라 낸다. (최종본, 잘라 낸 수)"""
    removed = 0
    if urls is not None:
        kept = []
        for line in answer.splitlines():
            if any(u not in urls for u in KAKAO_LINK.findall(line)):
                removed += 1  # 조회하지 않은 장소가 든 줄
                continue
            kept.append(line)
        answer = "\n".join(kept)

    def drop(m: re.Match) -> str:
        nonlocal removed
        removed += 1
        return ""

    for pattern in (TRAVEL_TIME, HOURS):
        answer = re.sub(r"[^.\n!?]*(?:" + pattern.pattern + r")[^.\n!?]*[.!?]?", drop, answer)
    return answer, removed


def failure_answer(cond, res_a: dict, res_b: dict, b_raw: dict) -> str:
    """장소 조회가 실패했을 때의 답변: 장소 일정은 만들지 않고, 유효한 날씨는 따로 보여준다. 모델 없이 값 그대로 쓴다."""
    reason = {
        "auth_error": "카카오맵 인증 오류예요. `.env`의 `KAKAO_REST_API_KEY`(REST API 키)와 카카오 Developers 앱의 **카카오맵 사용 설정**을 확인해 주세요",
        "error": "카카오맵 서버에 연결하지 못했거나 오류가 났어요",
        "empty": "조건에 맞는 장소가 조회되지 않았어요",
    }.get(res_a["status"], "장소를 조회하지 못했어요")
    out = [
        f"# 🗺️ {cond.region} 여행 계획 ({cond.date}({cond.weekday}))\n",
        f"> ❌ **장소 조회 실패** — {reason}. 그래서 **장소 일정은 만들지 않았어요.** 장소를 추측해서 채우지 않아요.\n",
    ]
    rec = b_raw["record"]
    if res_b["status"] == "ok":
        d = res_b["data"]
        out += [
            "## 🌤️ 날씨 (🔎 조회) — 따로 보여드려요",
            f"{d['date']}({d['weekday']}) · 최고 **{d['tmax_c']} °C** / 최저 **{d['tmin_c']} °C** · 강수확률 **{d['pop_percent']} %**  ",
            f"출처: Open-Meteo · 조회 시각: {d['fetched_at']} · 위치 좌표: {rec['coord_source']}\n",
            "## 🎒 준비물 (💡 제안)",
            *[f"- 💡 {tip}" for tip in b_raw["packing"]],
            "\n※ 장소 일정이 없어 준비물은 날씨 값만 근거로 했어요.",
        ]
    elif res_b["status"] == "out_of_range":
        d = res_b["data"]
        out += [
            "## 🌤️ 날씨",
            f"**예보 없음 — 확인 필요** (out_of_range): {d['date']}({d['weekday']})는 예보 범위({d['range_start']} ~ {d['range_end']}) 밖이에요. 날씨·준비물 확인 필요.",
        ]
    else:
        out += ["## 🌤️ 날씨", "**날씨 조회 실패 — 날씨·준비물 확인 필요**. 날씨도 조회하지 못해 값은 말씀드리지 않을게요."]
    out.append("\n문제를 해결한 뒤 **`다시 조사해줘`**라고 말씀해 주세요. 같은 조건(지역·날짜·취향)으로 다시 조회하고 종합해 드릴게요.")
    return "\n".join(out)


def write_plan(question: str, cond, a: dict, b: dict, plan: dict | None = None) -> Plan:
    """초안 작성 → 검사 → (문제가 있으면) 한 번 고쳐 쓰기 → 그래도 일정이 어긋나면 방문 순서를 코드가 바꾼다."""
    rec = b["record"]
    weather_ok = rec is not None and rec["status"] == "ok"
    visits = (plan or {}).get("visits") or []
    fixed = ""
    if visits:
        c = plan["constraints"]
        extra = [
            x
            for x in (
                f"시간 범위 {c.get('start_time') or '제한 없음'}~{c.get('end_time') or '제한 없음'}" if c.get("start_time") or c.get("end_time") else "",
                "제외한 장소: " + ", ".join(c["exclude"]) if c.get("exclude") else "",
            )
            if x
        ]
        notes = list(plan.get("notes") or []) + list(plan["basis"]["check"].get("notes") or [])
        fixed = (
            "\n[방문 일정 — 앱이 정한 값입니다. 방문 순서 섹션은 이 장소·순서·시작·끝을 그대로 쓰세요(빼거나 더하거나 시간을 바꾸지 마세요)]\n"
            + visit_lines(visits)
            + ("\n" + " · ".join(extra) if extra else "")
            + ("\n[알림 — 계획 맨 위에 그대로 알리세요] " + " / ".join(notes) if notes else "")
            + "\n"
        )
    prompt = (
        f"사용자 요청: {question}\n\n"
        f"공통 조건: 지역={cond.region}, 여행 날짜={cond.date}({cond.weekday}요일, 첫날 기준), 취향={cond.preference or '없음(일반 관광)'}\n{fixed}\n"
        f"[A 장소 조사 결과: 이 목록의 장소만 사용]\n{format_a(a)}\n\n"
        f"[B 날씨·준비물 조사 결과]\n{format_b(b)}\n\n"
        "위 조사 결과만 근거로 여행 계획을 쓰세요. 기간·인원은 요청에 있으면 따르고, 없으면 당일치기·1~2명으로 가정했다고 "
        "첫 문단에 밝힌 뒤 되묻지 말고 바로 쓰세요. 날씨는 여행 첫날 기준입니다."
        + ("" if weather_ok else " 예보를 얻지 못했으므로 날씨 값과 날씨 기반 준비물은 쓰지 마세요.")
    )
    draft = str(_llm().invoke([SystemMessage(content=system_prompt()), HumanMessage(content=prompt)]).text)
    result = Plan(answer=draft, violations=check(draft, a, b, visits))
    if result.violations:
        fix = (
            prompt
            + f"\n\n[초안]\n{draft}\n\n[고칠 점]\n"
            + "\n".join(f"- {v}" for v in result.violations)
            + "\n\n위 문제를 모두 고쳐 같은 형식으로 계획 전체를 다시 쓰세요. 조회하지 않은 사실은 넣지 마세요."
        )
        result.answer = str(_llm().invoke([SystemMessage(content=system_prompt()), HumanMessage(content=fix)]).text)
        result.revised, result.model_calls = True, 2
    if visits and check_visits(result.answer, visits):
        result.answer = force_visits(result.answer, visits)  # 모델이 일정을 못 지켰으면 방문 순서는 코드가 만든 것으로 쓴다
        result.forced = True
    result.answer, result.removed = scrub(result.answer, {p["url"] for p in a["places"] if p["url"]})
    result.leftover = check(result.answer, a, b, visits)
    return result
