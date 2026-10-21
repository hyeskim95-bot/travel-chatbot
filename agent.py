"""여행 챗봇 에이전트: 모델 + 장소 검색 도구 + 행동 지침(시스템 프롬프트)을 연결한다."""

import re
from dataclasses import dataclass, field
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import config
import places
import weather

# "후보가 없으면 다른 종류도 찾아줘" 같은 종류 확장 요청 표현
WIDEN_REQUEST = re.compile(r"다른\s*(종류|것|곳|장소|유형)|종류(도|까지|를 넓)|넓혀|범위를 넓|비슷한")

# 날씨 수치(기온·강수확률)가 답변에 들어 있는지
WEATHER_VALUE = re.compile(r"\d\s*(°C|℃|도\b)|강수\s*확률\s*\**\s*\d|\d+\s*%")


def weather_unavailable_answer(record: dict) -> str:
    """예보를 쓸 수 없을 때의 고정 안내: 현재 날씨나 다른 날짜 값으로 대신 채우지 않는다."""
    day = f"{record['date']}({record['weekday']})"
    if record["status"] == "out_of_range":
        why = (
            "이미 지난 날짜라" if record.get("reason") == "past" else f"예보 범위(오늘부터 {weather.MAX_FORECAST_DAYS}일) 밖이라"
        )
        return (
            f"{day}은 {why} **예보가 없어요** (out_of_range).\n\n"
            f"Open-Meteo 예보는 {record['range_start']} ~ {record['range_end']} 사이만 조회돼요. "
            "현재 날씨나 다른 날짜의 값으로 대신 알려드리면 준비물이 틀릴 수 있어서 날씨는 말씀드리지 않을게요. "
            "범위 안의 날짜로 다시 물어봐 주세요."
        )
    return (
        f"날씨 서버가 요청한 {day}과 다른 날짜({record.get('returned_date', '?')})의 데이터를 돌려줘서 **그 결과는 쓰지 않았어요** "
        "(date_mismatch). 날씨 값은 확인이 필요해요. 잠시 후 다시 시도해 주세요."
    )


AUTH_ERROR_ANSWER = (
    "장소 조회 서비스(카카오맵) 인증에 실패했어요. 검색 결과가 없는 것이 아니라 **조회 자체가 되지 않은 상태**예요.\n\n"
    "- `.env`의 `KAKAO_REST_API_KEY`가 올바른 **REST API 키**인지 확인해 주세요.\n"
    "- 카카오 Developers 앱에서 **카카오맵 사용 설정**이 켜져 있는지 확인해 주세요.\n\n"
    "설정을 고친 뒤 앱을 다시 시작하면 장소를 조회할 수 있어요. "
    "그 전에는 확인되지 않은 장소를 추천하지 않을게요."
)


@dataclass
class AgentResult:
    answer: str
    places: list[dict] = field(default_factory=list)  # 답변에 실제로 쓰인 조회 장소
    log: list[dict] = field(default_factory=list)  # 실행 기록: 도구·인자·status
    trip: dict = field(default_factory=dict)  # 여행 계획 흐름: 공통 조건, A·B 입력, 완료 여부(되묻는 중이면 complete=False)
    weather: list[dict] = field(default_factory=list)  # 조회한 날씨(날짜·기온·강수확률·출처·조회 시각)
    run: dict = field(default_factory=dict)  # 실행 요약: 도구·모델 호출 수와 한도, 종료 사유, 답변-기록 대조
    stopped: str = ""  # 한도·인증 오류로 멈췄으면 사유 ("limit" / "auth")


def build_agent():
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=config.TEMPERATURE,
    )
    # 카카오 키가 없으면 도구 없이 대화만 한다(날씨도 장소 검색의 좌표를 쓰므로 함께 꺼진다).
    tools = [places.search_places, weather.get_weather] if config.KAKAO_REST_API_KEY else []
    # 한 번의 실행 기준 한도: 도구(장소 검색+날씨) N회·모델 M회에 닿으면 그 자리에서 멈춘다.
    limits = [
        ToolCallLimitMiddleware(run_limit=config.MAX_TOOL_CALLS, exit_behavior="end"),
        ModelCallLimitMiddleware(run_limit=config.MAX_MODEL_CALLS, exit_behavior="end"),
    ]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt(), middleware=limits if tools else [])


def system_prompt() -> str:
    """행동 지침 + 오늘 날짜. '이번 주 토요일' 같은 표현은 이 날짜를 기준으로 도구가 계산한다."""
    today = datetime.now().date()
    return (
        config.SYSTEM_PROMPT
        + f"\n# 오늘 날짜\n오늘은 {today.isoformat()}({weather.weekday_name(today)}요일, Asia/Seoul)입니다.\n"
        + f"날짜 달력(상대 날짜를 이 표로 바꾼다): {calendar_text(today)}\n"
        + "사용자 문장 끝의 [앱이 계산한 날짜] 값을 그대로 쓰고, 직접 요일을 계산하지 않는다. "
        + "문장 끝의 [현재 계획] 줄은 앱이 붙인 최신 계획(표에서 직접 고친 내용 포함)이다. 장소·시간을 말할 때 이 계획과 다르게 말하지 않는다.\n"
    )


def linkify(answer: str, used: list[dict]) -> str:
    """답변에 나온 장소명이 링크가 아니면 첫 등장 위치를 카카오맵 링크로 바꾼다.
    이미 마크다운 링크 안에 있는 이름(예: '[경주 황리단길](..)' 속 '황리단길')은 건드리지 않는다."""
    link_re = re.compile(r"\[[^\]]*\]\([^)]*\)")
    for p in sorted(used, key=lambda x: len(x["name"]), reverse=True):
        if not p["url"] or f"[{p['name']}]" in answer:
            continue
        spans = [m.span() for m in link_re.finditer(answer)]
        for m in re.finditer(re.escape(p["name"]), answer):
            if not any(s <= m.start() and m.end() <= e for s, e in spans):
                answer = answer[: m.start()] + f"[{p['name']}]({p['url']})" + answer[m.end() :]
                break
    return answer

def limit_answer(found: list[dict], log: list[dict]) -> str:
    """한도에 닿았을 때의 답변: 못 찾았음을 알리고 조건 완화를 묻는다."""
    head = f"조회 한도(도구 {config.MAX_TOOL_CALLS}회·모델 {config.MAX_MODEL_CALLS}회)에 닿아 여기서 멈췄어요."
    tried = ", ".join(f"'{effective_query(e)}'" for e in log if e["status"] == "success")
    if not found:
        return (
            f"{head} 조건에 맞는 장소를 **찾지 못했어요.**"
            + (f" (시도한 검색: {tried})" if tried else "")
            + "\n\n조건을 조금 완화해 볼까요? 예를 들어 장소 종류를 넓히거나, 지역 범위를 넓히거나, 까다로운 조건을 빼고 다시 찾아볼 수 있어요."
        )
    lines = "\n".join(f"- [{p['name']}]({p['url']}) — {p['address']}" for p in found[:5])
    return (
        f"{head} 지금까지 조회된 후보만 안내드려요.\n\n{lines}\n\n"
        "원하는 조건에 더 맞는 곳을 찾으려면 조건을 완화해 다시 찾아볼까요?"
    )


def auth_result(call_args: dict | None) -> AgentResult:
    """인증 오류는 같은 호출을 반복하지 않고 바로 멈춘다. 장소를 지어내거나 성공으로 표시하지 않는다."""
    log = annotate(
        [{"tool": "search_places", "args": call_args or {}, "status": "error", "results": 0, "names": [], "keys": []}],
        ended="인증 오류",
    )
    return AgentResult(answer=AUTH_ERROR_ANSWER, log=strip_keys(log), run=run_meta(log, 1, "인증 오류"), stopped="auth")


def run(history: list[dict]) -> AgentResult:
    """대화 기록(마지막은 새 질문)을 에이전트에 넘기고 답변·조회 장소·실행 기록을 돌려준다."""
    messages = []
    recent = history[-config.MAX_HISTORY_MESSAGES :]
    for i, m in enumerate(recent):
        cls = HumanMessage if m["role"] == "user" else AIMessage
        content = m["content"]
        if i == len(recent) - 1 and m["role"] == "user":
            content = with_date_hints(content)  # 앱이 계산한 날짜를 마지막 사용자 문장 뒤에 붙인다
        messages.append(cls(content=content))

    # 같은 인자 반복 금지: 이번 질문 + 이 대화에서 이미 0건이었던 검색. 종류 확장은 사용자가 요청했을 때만 허용.
    failed_before = {
        (
            places.clean_query(places.full_query(m_log["args"].get("region", ""), m_log["args"].get("query", ""))),
            m_log["args"].get("category", ""),
        )
        for m in history
        for m_log in m.get("log", [])
        if m_log.get("results") == 0 and m_log.get("status") == "success"
    }
    places.reset_seen(failed_before, allow_widen=bool(WIDEN_REQUEST.search(history[-1]["content"])))

    try:
        result = build_agent().invoke({"messages": messages}, config={"recursion_limit": 40})
    except places.PlaceAuthError as err:
        return auth_result(err.call_args)
    new_messages = result["messages"][len(messages) :]

    calls: dict[str, dict] = {}
    found: dict[str, dict] = {}
    log: list[dict] = []
    weather_records: list[dict] = []
    model_calls = 0
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.response_metadata:  # 모델이 만든 메시지만 센다(한도 종료 메시지는 제외)
                model_calls += 1
            for call in msg.tool_calls:
                calls[call["id"]] = call
        elif isinstance(msg, ToolMessage):
            call = calls.get(msg.tool_call_id, {})
            args = call.get("args", {})
            tool_name = msg.name or call.get("name", "")
            if tool_name == "get_weather":
                # 날씨 결과: 날짜(요일)·기온·강수확률·출처·조회 시각이 든 기록 1건
                record = msg.artifact if isinstance(msg.artifact, dict) else None
                if record:
                    weather_records.append(record)
                ok = bool(record) and record["status"] == "ok"
                log.append(
                    {
                        "tool": tool_name,
                        # ok / out_of_range(예보 범위 밖) / date_mismatch(반환 날짜 불일치) / error
                        "status": record["status"] if record else msg.status,
                        "args": args,
                        "results": 1 if ok else 0,
                        "searched": record["label"] if record else f"{args.get('region', '')} {args.get('date', '')} 날씨",
                        "names": [f"{record['place']} {record['date']}({record['weekday']})"] if ok else [],
                        "keys": [],
                    }
                )
                continue
            items = msg.artifact or []
            for p in items:
                found.setdefault(p["url"] or p["name"], p)
            log.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "status": msg.status,
                    "results": len(items),
                    "searched": places.clean_query(places.full_query(args.get("region", ""), args.get("query", ""))),
                    "names": [p["name"] for p in items],
                    "keys": [p["url"] or p["name"] for p in items],
                }
            )

    # 한도 도달: 미들웨어가 끼워 넣은 종료 메시지(모델이 만든 것이 아님)로 끝난 경우
    last = new_messages[-1]
    limited = isinstance(last, AIMessage) and not last.response_metadata and not last.tool_calls
    if limited:
        log = annotate(log, ended="한도")
        return AgentResult(
            answer=limit_answer(list(found.values()), log),
            log=strip_keys(log),
            run=run_meta(log, model_calls, "한도"),
            stopped="limit",
        )

    answer = str(last.text)
    # 예보를 얻지 못했는데(범위 밖·날짜 불일치) 답변에 날씨 수치가 있으면 쓰지 않고 안내 문구로 바꾼다.
    unusable = [w for w in weather_records if w["status"] != "ok"]
    if unusable and not any(w["status"] == "ok" for w in weather_records) and WEATHER_VALUE.search(answer):
        answer = weather_unavailable_answer(unusable[-1])
    grounding = "해당 없음"
    if log:
        # 답변의 장소 = 마지막 기록(재검색이면 마지막 검색)의 결과. 어긋나면 한 번 고쳐 쓴다.
        basis = basis_keys(log)
        stale = mismatched(answer, found, basis)
        grounding = "일치"
        if stale:
            allowed = [p for k, p in found.items() if k in basis]
            answer = correct(history[-1]["content"], answer, allowed, weather_records)
            model_calls += 1
            grounding = f"수정됨(마지막 검색 결과 기준, 뺀 장소: {', '.join(stale)})"
            if mismatched(answer, found, basis):
                grounding += " · 일부 불일치 남음"
    log = annotate(log, ended="답변")
    used = [p for k, p in found.items() if p["name"] in answer and (not log or k in basis_keys(log))]
    run = run_meta(log, model_calls, "답변")
    run["grounding"] = grounding
    return AgentResult(
        answer=linkify(answer, used), places=used, log=strip_keys(log), run=run, weather=weather_records
    )


# --- 실행 기록 (내부 생각 전문은 기록하지 않고, 도구 이름·인자·status·결과·다음 행동만 남긴다) ---

def outcome_of(entry: dict) -> str:
    if entry["status"] == "out_of_range":
        return "예보 없음"
    if entry["status"] == "date_mismatch":
        return "날짜 불일치"
    if entry["status"] not in ("success", "ok"):
        return "오류"
    if entry["results"] == 0:
        return "0건"
    if entry["tool"] == "get_weather":
        return "조회됨"
    return "충분" if entry["results"] >= places.ENOUGH else "부족"


def annotate(log: list[dict], ended: str) -> list[dict]:
    """각 호출에 판정(outcome)과 다음 행동(next)을 기록 값으로 붙인다."""
    for i, e in enumerate(log):
        e["step"] = i + 1
        e["outcome"] = outcome_of(e)
        if i + 1 < len(log):
            retry = "재조회" if e["tool"] == "get_weather" else "재검색"
            e["next"] = retry if e["outcome"] in ("0건", "부족", "오류") else "다음 호출"
        else:
            e["next"] = {"답변": "답변", "한도": "중단(한도)", "인증 오류": "중단(인증 오류)"}[ended]
    return log


def strip_keys(log: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "keys"} for e in log]


def run_meta(log: list[dict], model_calls: int, ended: str) -> dict:
    return {
        "tool_calls": len(log),
        "tool_limit": config.MAX_TOOL_CALLS,
        "model_calls": model_calls,
        "model_limit": config.MAX_MODEL_CALLS,
        "ended": ended,
    }


def basis_keys(log: list[dict]) -> set[str]:
    """답변의 근거가 되는 장소: 각 '찾기 흐름'의 마지막 검색 결과.
    앞 검색이 충분하지 않아 다시 검색한 경우(재검색)는 한 흐름으로 보고 마지막 검색만 근거로 쓴다."""
    basis: set[str] = set()
    for i, e in enumerate(log):
        next_is_weather = i + 1 < len(log) and log[i + 1]["tool"] == "get_weather"
        is_last_of_chain = i + 1 == len(log) or outcome_of(e) == "충분" or next_is_weather
        if is_last_of_chain:
            basis.update(e.get("keys", []))
    return basis


def mismatched(answer: str, found: dict[str, dict], basis: set[str]) -> list[str]:
    """답변에 나오지만 근거(마지막 검색 결과)에 없는 장소 이름."""
    stale = [p["name"] for k, p in found.items() if k not in basis and p["name"] in answer]
    # 근거에 들어 있는 이름의 일부라서 걸린 경우(예: '경주 박물관' ⊂ '국립경주 박물관')는 제외한다.
    basis_names = [p["name"] for k, p in found.items() if k in basis]
    return [n for n in stale if not any(n in b for b in basis_names)]


def correct(question: str, draft: str, allowed: list[dict], weather_records: list[dict] | None = None) -> str:
    """초안 답변을 근거 장소만 쓰도록 한 번 고쳐 쓴다."""
    llm = ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, temperature=config.TEMPERATURE)
    prompt = (
        f"사용자 질문: {question}\n\n"
        f"답변에 쓸 수 있는 장소(마지막 검색 결과, 이것만 사용):\n{places.format_places(allowed) if allowed else '(없음)'}\n\n"
        + (f"날씨 조회 결과(수치는 그대로 유지):\n{weather_records}\n\n" if weather_records else "")
        + f"초안 답변:\n{draft}\n\n"
        "초안 답변에서 위 목록에 없는 장소는 모두 빼고, 같은 형식과 말투로 다시 써 주세요. "
        "목록에 없는 정보는 추가하지 마세요. 장소가 모자라면 그렇다고 솔직히 쓰세요."
    )
    return str(llm.invoke([SystemMessage(content=system_prompt()), HumanMessage(content=prompt)]).text)


def explain(log: list[dict], run: dict | None) -> list[str]:
    """기록 값(검색어·결과 수·판정·다음 행동·종료 사유)만으로 '무엇을 → 왜 다시 → 언제 끝'을 한 줄씩 설명한다."""
    if not log:
        return ["도구를 부르지 않고 대화로 답했어요."]
    lines = []
    for e in log:
        a = e["args"]
        q = effective_query(e)
        cat = f" [{a['category']}]" if a.get("category") else ""
        outcome = e.get("outcome") or outcome_of(e)
        nxt = e.get("next", "")
        why = {
            "재검색": f"{outcome}이라 다시 검색",
            "재조회": f"{outcome}이라 다시 조회",
            "다음 호출": "결과를 받았으니 다음 호출",
            "답변": (
                "충분하니 답변으로 종료"
                if outcome in ("충분", "조회됨")
                else "예보가 없어 날씨 값을 말하지 않고 답변으로 종료"
                if outcome in ("예보 없음", "날짜 불일치")
                else "더 찾지 않고 답변으로 종료"
            ),
        }.get(nxt, nxt)
        amount = "" if e["tool"] == "get_weather" else f"({e['results']}곳)"
        lines.append(f"{e.get('step', '')}. '{q}'{cat} → {outcome}{amount} → {why}")
    if run:
        lines.append(f"종료: {run.get('ended', '')} (도구 {run.get('tool_calls', len(log))}회, 모델 {run.get('model_calls', '?')}회)")
    return lines


def effective_query(entry: dict) -> str:
    """실제로 카카오에 보낸 검색어(조건어를 뺀 값). 기록에 없으면 인자로 다시 계산한다."""
    if entry.get("searched"):
        return entry["searched"]
    a = entry["args"]
    return places.clean_query(places.full_query(a.get("region", ""), a.get("query", ""))) or "(검색어 없음)"


def summary_sentence(log: list[dict], run: dict | None) -> str:
    """실행 기록 값만으로 반복 과정을 한 문장으로 요약한다(모델 답변·내부 생각은 보지 않는다)."""
    if not log:
        return "도구를 부르지 않고 대화로 바로 답했습니다."
    parts = []
    for e in log:
        cat = f"({e['args']['category']})" if e["args"].get("category") else ""
        outcome = e.get("outcome") or outcome_of(e)
        result = outcome if e["tool"] == "get_weather" else ("0건" if e["results"] == 0 else f"{outcome} {e['results']}곳")
        if outcome == "오류":
            result = "오류"
        parts.append(f"'{effective_query(e)}'{cat} → {result}")
    end = {"답변": "답변으로 종료", "한도": "한도에 닿아 중단", "인증 오류": "인증 오류로 중단"}.get(
        (run or {}).get("ended", "답변"), "종료"
    )
    return f"도구 {len(log)}회: " + ", ".join(parts) + f" 순으로 진행했고, 마지막 결과를 근거로 {end}했습니다."

# --- 5일차 시작 코드의 날짜·변경 요청 도우미 (agent.py 의 규칙을 옮김) ---
import planner  # noqa: E402

_WD = r"(월|화|수|목|금|토|일)요일"
_WEEKDAYS = planner.WEEKDAYS


def calendar_text(today) -> str:
    """오늘부터 21일의 날짜·요일 표. 상대 날짜를 이 표로 바꾸게 한다."""
    from datetime import timedelta

    return ", ".join(
        f"{(today + timedelta(days=i)).isoformat()}({_WEEKDAYS[(today + timedelta(days=i)).weekday()]})" for i in range(21)
    )


def date_hints(text: str, today) -> list[tuple[str, str]]:
    """사용자 문장의 상대 날짜를 코드로 계산한다(모델이 '이번 주 토요일'을 목요일로 바꾼 기록이 있음).
    반환: [(원래 표현, 'YYYY-MM-DD(요일)')]"""
    from datetime import date, timedelta

    found, used = [], []

    def add(m, d):
        if any(a < m.end() and m.start() < b for a, b in used):
            return
        used.append((m.start(), m.end()))
        found.append((m.group(0), planner.date_label(d.isoformat())))

    for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
        try:
            add(m, date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in re.finditer(r"(이번\s*주|이번|다음\s*주|담주)\s*" + _WD, text):
        monday = today - timedelta(days=today.weekday())
        if m.group(1).replace(" ", "") in ("다음주", "담주"):
            monday += timedelta(days=7)
        add(m, monday + timedelta(days=_WEEKDAYS.index(m.group(2))))
    for m in re.finditer(r"다음\s*달\s*(\d{1,2})일", text):
        y, mo = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        try:
            add(m, date(y, mo, int(m.group(1))))
        except ValueError:
            pass
    for m in re.finditer(r"(\d{1,2})월\s*(\d{1,2})일", text):
        try:
            d = date(today.year, int(m.group(1)), int(m.group(2)))
            add(m, d if d >= today else date(today.year + 1, d.month, d.day))
        except ValueError:
            pass
    for m in re.finditer(r"(\d{1,3})\s*일\s*(뒤|후)", text):
        add(m, today + timedelta(days=int(m.group(1))))
    for word, n in (("오늘", 0), ("내일", 1), ("모레", 2), ("글피", 3)):
        for m in re.finditer(word, text):
            add(m, today + timedelta(days=n))
    for m in re.finditer(_WD, text):  # '토요일에'처럼 앞말이 없으면 다가오는 그 요일
        add(m, today + timedelta(days=(_WEEKDAYS.index(m.group(1)) - today.weekday()) % 7))
    return found


def with_date_hints(text: str, today=None) -> str:
    """모델에 보낼 문장 뒤에 앱이 계산한 날짜를 붙인다(화면에는 원래 문장만 보임)."""
    from datetime import date

    hints = date_hints(text, today or date.today())
    if not hints:
        return text
    return text + "\n\n[앱이 계산한 날짜 — 그대로 사용: " + ", ".join(f"'{a}' = {b}" for a, b in hints) + "]"


CHANGE_CUES = re.compile(r"바꿔|바꾸|변경|수정|줄여|늘려|빼|제외|추가|넣어|대신|옮겨|미뤄|당겨|오전|오후|\d+\s*시|\d+\s*곳|한\s*곳|두\s*곳|세\s*곳|날짜|지역")


def looks_like_change(text: str, today=None) -> bool:
    """현재 계획을 바꾸라는 요청으로 보이는지(날짜 표현 또는 변경 표현). 플래너가 놓쳤을 때 한 번 더 확인하는 데만 쓴다."""
    from datetime import date

    return bool(date_hints(text, today or date.today())) or bool(CHANGE_CUES.search(text))


AFFIRM = re.compile(r"^\s*(네|예|응|어|그래|좋아|좋습니다|괜찮|진행|그렇게|덮어|바꿔도|상관없)")


def is_affirmative(text: str) -> bool:
    """확인 질문에 대한 '네·그래요·진행해 주세요' 같은 동의."""
    return bool(AFFIRM.search(text or ""))
