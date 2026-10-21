"""대화 기억 + 카카오맵 장소 조회로 계획을 세우는 여행 챗봇."""

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)  # LangSmith 등 환경변수는 langchain 을 쓰기 전에 읽어 둔다

import httpx
import openai
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from trip import options_text

import agent
import calendar_mcp
import calendar_preview
import config
import registration
import trip
import trip_plan
import view_model
import storage
import ui_style

# 첫 화면에 보이는 입력 예시(누르면 그대로 질문으로 보낸다)
EXAMPLES = [
    "모레 경주에서 박물관 중심으로 하루 여행 계획을 만들어 줘.",
    "이번 주 토요일 부산 날씨 알려줘.",
    "제주 애월 분위기 좋은 카페 추천해 줘.",
]
INTRO = "지역·날짜·취향만 알려 주세요. 장소와 날씨를 조회해서 근거 있는 하루 여행 계획을 만들어 드려요."


def tracing_state() -> tuple[str, str]:
    """LangSmith 추적 상태: (라벨, 이유). API 키 값은 어디에도 내보내지 않는다."""
    enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("1", "true", "yes")
    project = os.getenv("LANGSMITH_PROJECT", "").strip() or "default"
    if not enabled:
        return "꺼짐", "LANGSMITH_TRACING=true 로 바꾸면 켜져요."
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        return "키 없음", "LANGSMITH_TRACING 은 켜졌지만 LANGSMITH_API_KEY 가 비어 있어 전송되지 않아요."
    return "켜짐", f"프로젝트 {project} 로 전송해요."


def show_tracing_status() -> None:
    label, why = tracing_state()
    icon = {"켜짐": "🧭", "키 없음": "⚠️", "꺼짐": "⚪"}[label]
    st.caption(f"{icon} LangSmith 추적: {label} — {why}")


def refresh_calendar() -> None:
    if st.session_state.get("view_only"):  # 보기 전용 세션은 캘린더에 연결하지 않는다(서버도 띄우지 않는다)
        st.session_state.calendar_status = None
        return
    """캘린더 MCP 서버를 실행해 도구를 발견하고, 준비 상태를 읽기 전용 조회로 따로 확인한다(일정은 만들지 않는다)."""
    with st.spinner("캘린더 MCP 서버를 확인하는 중..."):
        st.session_state.calendar_status = calendar_mcp.inspect_calendar()


def show_calendar_status() -> None:
    """사이드바 카드: MCP 연결(도구 발견)과 Google 준비 상태를 서로 구분해서 보여준다."""
    with st.expander("📅 캘린더 연결"):
        if st.session_state.get("view_only"):
            st.caption("👁️ 보기 전용 세션: 캘린더 연결·확인·등록은 할 수 없어요.")
            return
        if "calendar_status" not in st.session_state:  # 앱을 처음 열 때 한 번만 확인한다
            refresh_calendar()
        s = st.session_state.calendar_status
        if s.mcp_ok:
            st.markdown("**MCP 서버** ✅ 연결됨 (stdio 자식 프로세스)")
            names = " · ".join(f"{t['label']} `{t['name']}`" for t in s.tools if t["label"])
            st.caption(f"🛠️ 도구 {len(s.tools)}개: {names}")
            if s.missing_tools:
                st.warning("없는 도구: " + ", ".join(s.missing_tools))
        else:
            st.markdown(f"**MCP 서버** ❌ 연결 실패 (MCP) — {s.mcp_error}")
        if s.ready:
            st.markdown(f"**Google 준비** ✅ {s.ready_label}")
        elif s.ready_code == "mcp_failed":
            st.markdown("**Google 준비** ❌ 연결 실패 (MCP) — 준비 확인도 하지 못했어요 (성공 표시 없음)")
        else:
            st.markdown(f"**Google 준비** ⚠️ {s.ready_label}")
        if not s.ready:
            st.caption(s.ready_hint + (f" ({s.ready_detail})" if s.ready_detail and s.ready_code != "mcp_failed" else ""))
        st.caption("도구 발견만으로는 Google 인증·캘린더 준비를 알 수 없어서, 읽기 전용 조회로 따로 확인해요. 일정은 만들지 않아요.")
        st.button("🔄 다시 확인", on_click=refresh_calendar, key="calendar_recheck")


PREVIEW_COLUMNS = ["포함", "제목", "날짜", "시작", "끝", "시간대", "장소"]


def _clean(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()


def show_registration_result(result) -> None:
    """등록 결과: 만든 일정과, 다시 조회해 대조한 결과를 행마다 보여준다."""
    if result.ok and result.created == 0:
        st.info("새로 만든 일정이 없어요. 모두 이미 등록돼 있어서 추가로 만들지 않았어요.")
    elif result.ok:
        st.success(f"등록했어요: 새로 만든 일정 {result.created}건 (다시 조회해 대조 완료).")
    elif result.status == "failed":
        st.warning(f"{result.message} 새로 만든 일정 {result.created}건.")
    else:
        st.error(result.message)
    for o in result.outcomes:
        if o.status == "created":
            mark = "✅" if o.verified else "⚠️"
            st.markdown(f"- {mark} **{o.title}** — 만들고 다시 조회해 " + ("일치했어요" if o.verified else f"대조가 맞지 않아요 ({o.mismatch})"))
        elif o.status == "already_registered":
            st.markdown(f"- ⏭️ **{o.title}** — 이미 등록된 일정이라 다시 만들지 않았어요")
        else:
            st.markdown(f"- ❌ **{o.title}** — 만들지 못했어요 ({o.message or o.mismatch})")


def show_preview(idx: int, m: dict) -> None:
    """여행 계획을 일정 미리보기로 바꿔 보여준다. 고칠 수 있고, 확인한 일정만 등록한다(확인은 앱 코드가 지킨다)."""
    trip_info = m.get("trip") or {}
    if not trip_info.get("complete") or not trip_info.get("synthesized", True):
        return
    plan = trip_info.get("plan")
    # 미리보기 표의 원본은 계획 글이 아니라 현재 여행 계획(visits)이다. 예전에 저장된 대화(계획 없음)만 글에서 읽는다.
    rows = (
        calendar_preview.rows_from_visits(plan["visits"])
        if plan
        else calendar_preview.rows_from_plan(m["content"], trip_info["condition"]["date"])
    )
    if not rows:
        return
    chat_id = st.session_state.chat_id
    key = f"preview_{chat_id}_{idx}_v{plan['version'] if plan else 0}"  # 계획 버전이 바뀌면 표를 새 계획으로 다시 그린다
    owner = st.session_state.owner_id
    view_only = st.session_state.view_only
    status = st.session_state.get("calendar_status")
    ready = bool(status and status.ready) and not view_only
    cal_key = registration.calendar_key_from_env()

    with st.expander("📅 일정 미리보기", expanded=True):
        st.markdown(
            f"**대상 캘린더** `{calendar_preview.mask_calendar_id()}` · 준비 상태: "
            + ("보기 전용" if view_only else (f"{status.ready_label}" if status else "확인 전"))
        )
        if view_only:
            st.info("👁️ 보기 전용 세션이에요. 캘린더 연결·확인·등록은 할 수 없어요.")
        count_slot = st.empty()
        edited = st.data_editor(
            pd.DataFrame(rows),
            key=key,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            column_order=PREVIEW_COLUMNS,
            disabled=view_only,
            column_config={
                "포함": st.column_config.CheckboxColumn("포함", default=True, width="small"),
                "제목": st.column_config.TextColumn("제목", required=False),
                "날짜": st.column_config.TextColumn("날짜", help="YYYY-MM-DD"),
                "시작": st.column_config.TextColumn("시작", help="24시간 HH:MM (예 09:30)", width="small"),
                "끝": st.column_config.TextColumn("끝", help="24시간 HH:MM, 시작보다 늦어야 해요", width="small"),
                "시간대": st.column_config.TextColumn("시간대", default=calendar_preview.DEFAULT_TIMEZONE),
                "장소": st.column_config.TextColumn("장소"),
            },
        )
        current = [{k: (bool(v) if k == "포함" and not pd.isna(v) else _clean(v)) for k, v in r.items()} for r in edited.to_dict("records")]
        for r in current:
            r["포함"] = bool(r.get("포함"))
        any_problem = False
        for i, r in enumerate(current, 1):
            problems = calendar_preview.validate_row(r)
            if problems:
                any_problem = True
                st.warning(f"⚠️ {i}행 '{r['제목'] or '(제목 없음)'}': " + " / ".join(problems) + ("" if r["포함"] else " (제외됨)"))
        if plan and current and not view_only and not any_problem:
            # 표에서 직접 고친 내용을 현재 여행 계획에 넣는다(잘못된 행이 없을 때만). 이후 대화 수정은 이 계획을 기준으로 한다.
            new_plan = trip_plan.apply_table(plan, calendar_preview.rows_to_table(current))
            if new_plan is not None:
                trip_info["plan"] = new_plan
                trip_info["options"] = new_plan["constraints"]
                storage.save_chat(chat_id, st.session_state.history)
                st.rerun()
        ok = calendar_preview.can_confirm(current)
        if ok:
            st.caption("✅ 포함한 모든 행이 정상이에요.")
        elif any_problem:
            st.caption("문제가 있는 행을 고치거나 '포함'을 해제하면 확인할 수 있어요.")
        else:
            st.caption("포함한 행이 하나 이상 있어야 확인할 수 있어요.")

        registered_n = registration.count_registered(cal_key, chat_id, current)
        count_slot.markdown(f"**생성된 일정 {registered_n}건**")

        # 확인 → 한 번 쓰는 표. 등록 버튼은 '표가 지금 내용과 같고 아직 안 썼을 때'만 켜진다(서버 쪽에서도 다시 검사한다).
        ticket_id = st.session_state.get(key + "_ticket")
        state = registration.peek(ticket_id, owner, current, view_only)
        c1, c2, c3 = st.columns(3)
        if c1.button("🔁 다시 확인" if ticket_id else "✅ 이 미리보기로 확인", key=key + "_ok", disabled=(not ok) or view_only):
            ticket, why = registration.issue_ticket(owner, current, view_only)
            st.session_state[key + "_ticket"] = ticket.id if ticket else None
            st.session_state.pop(key + "_result", None)
            st.session_state[key + "_msg"] = "" if ticket else registration.MESSAGES[why]
            st.rerun()
        if c2.button("📅 캘린더에 등록", key=key + "_reg", type="primary", disabled=(state != "ok") or not ready):
            result = registration.register(
                ticket_id, owner, current,
                view_only=view_only, calendar_ready=ready, chat_id=chat_id, calendar_key=cal_key,
                creator=calendar_mcp.register_events,
            )
            st.session_state[key + "_result"] = result
            m["registration"] = {
                "status": result.status,
                "created": result.created,
                "at": datetime.now().isoformat(timespec="seconds"),
                "outcomes": [{"title": o.title, "status": o.status, "verified": o.verified} for o in result.outcomes],
            }
            storage.save_chat(chat_id, st.session_state.history)
            st.rerun()
        if c3.button("✖️ 확인 취소", key=key + "_cancel", disabled=state != "ok"):
            registration.cancel(ticket_id, owner)
            st.session_state[key + "_msg"] = registration.MESSAGES["rejected_cancelled"]
            st.rerun()

        result = st.session_state.get(key + "_result")
        if result is not None:
            show_registration_result(result)
        elif state == "ok":
            n = len(registration.included(current))
            st.success(f"확인됨: {n}개 행. 이제 '캘린더에 등록'을 누를 수 있어요. (아직 등록하지 않았어요)")
        elif state == "rejected_no_ticket":
            st.caption("확인하기 전이라 '캘린더에 등록'은 비활성이에요. 먼저 미리보기를 확인해 주세요.")
        elif state in registration.MESSAGES:
            st.info(registration.MESSAGES[state])
        if st.session_state.get(key + "_msg"):
            st.caption(st.session_state[key + "_msg"])
        if result is None and m.get("registration"):
            reg = m["registration"]
            st.caption(f"이전 등록 기록: {reg['at']} · 새로 만든 일정 {reg['created']}건")


def show_places(items: list[dict]) -> None:
    if not items:
        return
    with st.expander(f"📍 카카오맵에서 조회한 장소 ({len(items)}곳)"):
        for p in items:
            line = f"**[{p['name']}]({p['url']})**" if p["url"] else f"**{p['name']}**"
            st.markdown(f"{line}  \n{p['category'].split(' > ')[-1]} · {p['address']}")


@contextmanager
def panel(label: str, expanded: bool = False, flat: bool = False):
    """펼침 칸. 탭 안처럼 expander 를 겹칠 수 없는 곳(flat)에서는 제목만 달고 내용을 그대로 보인다."""
    if flat:
        st.markdown(f"**{label}**")
        yield
    else:
        with st.expander(label, expanded=expanded):
            yield


STATUS_ICON = {"ok": "✅", "empty": "⚪", "out_of_range": "⚠️", "date_mismatch": "⚠️", "error": "❌", "auth_error": "❌"}


def render_result(res: dict) -> None:
    """A·B 가 같은 모양({role, name, input, status, source, data})으로 돌려준 원결과 한 칸."""
    st.markdown(f"**{res['role']} · {res['name']} 원결과**")
    st.markdown(f"{STATUS_ICON.get(res['status'], '❔')} status: `{res['status']}` · source: {res['source']}")
    st.caption("input: " + ", ".join(f"{k}={v or '없음'}" for k, v in res["input"].items()))
    d = res["data"]
    if res["role"] == "A":
        for s in d["searches"]:
            mark = "✅" if s["status"] == "success" else "❌"
            kind = f" [{s['category']}]" if s["category"] else ""
            st.markdown(f"- {mark} 검색 '{s['query']}'{kind} → {s['count']}곳")
        if not d["places"]:
            st.caption("조회된 장소가 없어요. 장소는 추측해서 채우지 않아요.")
        for p in d["places"]:
            link = f"[{p['name']}]({p['url']})" if p["url"] else p["name"]
            st.markdown(f"**{link}**  \n{p['category'].split(' > ')[-1]} · {p['address']}")
        return
    if res["status"] == "ok":
        c1, c2, c3 = st.columns(3)
        c1.metric("최고", f"{d['tmax_c']} °C")
        c2.metric("최저", f"{d['tmin_c']} °C")
        c3.metric("강수확률", f"{d['pop_percent']} %")
        st.markdown(f"📅 {d['date']}({d['weekday']}) · 📍 {d['place']} · 조회 시각 {d['fetched_at']}")
        st.markdown("🎒 준비물 근거: " + " / ".join(d["packing"]))
        if d.get("coord_source"):
            st.caption(f"좌표 출처: {d['coord_source']}")
    elif res["status"] == "out_of_range":
        why = "이미 지난 날짜" if d.get("reason") == "past" else "예보 범위 밖"
        st.warning(f"📅 {d['date']}({d['weekday']}) **예보 없음** (out_of_range): {why}. 조회 가능 {d['range_start']} ~ {d['range_end']}")
    elif res["status"] == "date_mismatch":
        st.warning(f"반환 날짜({d.get('returned_date')})가 요청({d.get('date')})과 달라 결과를 쓰지 않았어요.")
    else:
        st.error(d.get("message", "조회하지 못했어요."))


def show_results(trip: dict, flat: bool = False) -> None:
    """종합하기 전에 A·B 원결과를 두 칸으로 나란히 펼쳐 본다."""
    results = trip.get("results")
    if not results:
        return
    label = "🔎 조사 원결과 · A / B (종합 전)" if not trip.get("synthesized", True) else "🔎 조사 원결과 · A / B"
    with panel(label, expanded=not trip.get("synthesized", True), flat=flat):
        col_a, col_b = st.columns(2)
        with col_a:
            render_result(results["A"])
        with col_b:
            render_result(results["B"])


def show_verification(v: dict | None, flat: bool = False) -> None:
    """종합 직전 검증: A·B 의 지역·날짜·status 를 공통 조건과 대조한 결과."""
    if not v:
        return
    if v["ok"]:
        st.caption("✅ 종합 전 검증 통과: 지역·날짜가 공통 조건과 같음" + (" · " + " / ".join(v["notes"]) if v["notes"] else ""))
    else:
        out = [x for x, flag in (("날씨", v["exclude_weather"]), ("장소", v["exclude_places"])) if flag]
        st.warning("⚠️ **불일치 경고**: " + " / ".join(v["warnings"]) + f" → {', '.join(out)}을(를) 계획의 근거에서 뺐어요.")
    with panel("🧪 종합 전 검증 상세", flat=flat):
        for c in v["checks"]:
            st.markdown(f"- {'✅' if c['pass'] else '❌'} {c['name']} · 기대 `{c['expected']}` / 실제 `{c['actual']}`")


def show_trip(trip: dict | None, flat: bool = False) -> None:
    """여행 계획 흐름: 플래너가 만든 공통 조건과 A·B 가 받은 입력(같은 조건에서 나왔는지)을 보여준다."""
    if not trip or not trip.get("complete"):
        return
    c, a, b = trip["condition"], trip["A"], trip["B"]
    with st.container(border=True):
        st.markdown("**📋 공통 조건** · 플래너가 한 번 만들어 A·B 에 똑같이 넘겼어요")
        st.markdown(f"- 지역 **{c['region']}** · 여행 날짜 **{c['date']}** · 취향 {c['preference'] or '없음'}")
        st.markdown(f"- 🅰️ {a['role']} 입력: 지역={a['input']['region']}, 날짜={a['input'].get('date', '')}, 취향={a['input']['preference'] or '없음'} (검색 {a['searches']}회 → 장소 {a['places']}곳)")
        st.markdown(f"- 🅱️ {b['role']} 입력: 지역={b['input']['region']}, 날짜={b['input']['date']} (status: {b['status']})")
        same = trip.get("shared_region") and trip.get("shared_date", True)
        st.caption("✅ A·B 입력의 지역·날짜가 공통 조건과 같음" if same else "⚠️ A·B 입력의 지역이나 날짜가 서로 달라요")
        show_verification(trip.get("verification"), flat)
        if trip.get("options"):
            summary = options_text(trip["options"])
            if summary:
                st.markdown(f"- ⚙️ 계획 조건: {summary}")
        if trip.get("reused"):
            st.caption("♻️ 이전 조사 결과를 재사용했어요 — 지역·날짜·취향이 그대로라서 장소·날씨를 다시 검색하지 않았어요.")
    show_results(trip, flat)


def show_weather(records: list[dict]) -> None:
    """날씨 조회 결과 카드: 날짜(요일)·최고/최저·강수확률 + 출처·조회 시각·사용한 좌표."""
    for w in records:
        status = w.get("status", "ok")  # 예전에 저장된 기록에는 status 가 없다
        if status == "out_of_range":
            why = "이미 지난 날짜예요" if w.get("reason") == "past" else "예보 범위 밖이에요"
            st.warning(
                f"📅 {w['date']}({w['weekday']}) · **예보 없음** (out_of_range): {why}. "
                f"조회 가능 기간은 {w['range_start']} ~ {w['range_end']}이고, 다른 날씨로 대신 채우지 않았어요."
            )
            continue
        if status == "date_mismatch":
            st.warning(
                f"📅 {w['date']}({w['weekday']}) · **날짜 불일치** (date_mismatch): 반환된 날짜({w.get('returned_date')})가 "
                "요청과 달라 결과를 쓰지 않았어요. 날씨는 확인이 필요해요."
            )
            continue
        with st.container(border=True):
            st.markdown(f"**🌤️ {w['date']}({w['weekday']}) · {w['place']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("최고", f"{w['tmax']}°C")
            c2.metric("최저", f"{w['tmin']}°C")
            c3.metric("강수확률", f"{w['pop']}%")
            st.caption(
                f"출처: {w['source']} · 조회 시각: {w['fetched_at']} · "
                f"좌표: 경도(x) {float(w['lng']):.4f}, 위도(y) {float(w['lat']):.4f} ({w['address']})"
            )


def render_run(log: list[dict], run: dict | None) -> None:
    """한 번의 실행 기록: 한눈에 보기(기록 값으로 설명) + 호출 순서(도구 이름·인자·status)."""
    st.markdown(f"**요약:** {agent.summary_sentence(log, run)}")
    for line in agent.explain(log, run):
        st.markdown(f"- {line}")
    if run and run.get("grounding") not in (None, "해당 없음"):
        st.caption(f"답변-기록 대조: {run['grounding']}")
    for entry in log:
        icon = {"success": "✅", "ok": "✅", "out_of_range": "⚠️", "date_mismatch": "⚠️"}.get(entry["status"], "❌")
        widened = entry["args"].get("widened_kind")
        region = entry["args"].get("region")
        tag = (
            (f" · 역할 {entry['role']}" if entry.get("role") else "")
            + (f" · 📍 region: {region}" if region else "")
            + (f" · 종류: {entry['args']['category']}" if entry["args"].get("category") else "")
            + (f" · 실제 검색어: '{entry['searched']}'" if entry.get("searched") else "")
            + (f" · 🔀 넓힌 종류: {widened}" if widened else "")
        )
        st.markdown(
            f"{entry.get('step', '')}. {icon} `{entry['tool']}` · status: `{entry['status']}` · 결과 {entry['results']}곳{tag}"
        )
        st.json(entry["args"], expanded=False)
        if entry.get("names"):
            st.caption("반환: " + ", ".join(entry["names"]))


def show_log(log: list[dict], run: dict | None = None, flat: bool = False) -> None:
    """답변 아래 실행 기록. 제목에 호출 횟수와 한도를 보여준다(도구를 안 썼으면 0회)."""
    if run:
        label = (
            f"🔧 실행 기록 (도구 호출 {run['tool_calls']}/{run['tool_limit']}회 · "
            f"모델 {run['model_calls']}/{run['model_limit']}회 · 종료: {run['ended']})"
        )
    else:
        label = f"🔧 실행 기록 (도구 호출 {len(log)}회)"
    with panel(label, flat=flat):
        render_run(log, run)


def show_compare(message: dict) -> None:
    """같은 요청의 두 실행(A 처음, B 다시)을 나란히 보여준다."""
    other = message["compare"]
    with st.expander("🔬 실행 비교 (A: 처음 실행 · B: 다시 실행)", expanded=True):
        col_a, col_b = st.columns(2)
        for col, name, log, run, places_ in (
            (col_a, "A · 처음 실행", message["log"], message.get("run"), message.get("places", [])),
            (col_b, "B · 다시 실행", other["log"], other.get("run"), other.get("places", [])),
        ):
            with col:
                st.markdown(f"**{name}**")
                render_run(log, run)
                st.caption("답변의 장소: " + (", ".join(p["name"] for p in places_) or "없음"))
        queries = lambda log: [  # noqa: E731
            agent.places.full_query(e["args"].get("region", ""), e["args"].get("query", "")) for e in log
        ]
        qa, qb = queries(message["log"]), queries(other["log"])
        st.info(
            f"A는 도구 {len(qa)}회({' → '.join(qa) or '없음'}), B는 도구 {len(qb)}회({' → '.join(qb) or '없음'})로 "
            + ("같은 검색 흐름이었어요." if qa == qb else "검색 흐름이 달랐어요. 위 한눈에 보기에서 왜 다시 검색했는지 비교해 보세요.")
        )

def render_structured(m: dict, view: dict) -> None:
    """저장된 조회 값만으로 그리는 여행 결과 화면(모델을 다시 부르지 않는다).
    순서: 조건 요약 한 줄 → 일정 표 → 장소 카드 → 날씨 → 자세한 설명 → 조회 근거. 없는 값은 '없음'으로 표시한다."""
    st.markdown(" ".join(f":blue-badge[{k}] {v}" for k, v in view["summary"]))

    st.markdown("#### 🗓️ 일정")
    if view["has_visits"]:
        df = pd.DataFrame(
            [
                {"순서": v["order"], "시간": v["time"], "장소": v["title"], "분류": v["category"] or "분류 없음",
                 "주소": v["address"] or "주소 없음", "링크": v["url"]}
                for v in view["visits"]
            ]
        )
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={
                "순서": st.column_config.NumberColumn(width="small"),
                "시간": st.column_config.TextColumn(width="small"),
                "링크": st.column_config.LinkColumn("링크", display_text="카카오맵", help="조회된 카카오맵 장소 링크"),
            },
        )
    else:
        st.info("일정에 넣을 장소가 없어요. 조회된 장소가 없으면 만들어 채우지 않아요.")

    if view["has_visits"]:
        st.markdown("#### 📍 장소 정보")
        for i, v in enumerate(view["visits"]):
            if i % 2 == 0:  # 줄 단위로 만들어 화면 순서와 읽는 순서(1,2,3)가 같게 한다
                cols = st.columns(2)
            with cols[i % 2].container(border=True):
                st.markdown(f"**{v['order']}. {v['title']}**")
                st.caption(f"{v['time']} · {v['category'] or '분류 없음'}")
                st.markdown(f"📍 {v['address'] or '주소 없음'}")
                if v["url"]:
                    st.markdown(f"[카카오맵에서 보기]({v['url']})")
                else:
                    st.caption("링크 없음")

    w = view["weather"]
    st.markdown("#### 🌤️ 날씨")
    if w["status"] == "ok":
        st.caption(f"{w['date_label']} · {w['place'] or '위치 없음'}")
        c1, c2, c3 = st.columns(3)
        for col, label, val, unit in ((c1, "최고 기온", w["tmax_c"], "°C"), (c2, "최저 기온", w["tmin_c"], "°C"), (c3, "강수확률", w["pop_percent"], "%")):
            col.metric(label, f"{val} {unit}" if val is not None else "값 없음")
        if w["packing"]:
            st.markdown("🎒 준비물 · " + " / ".join(w["packing"]))
        else:
            st.caption("🎒 준비물 근거 없음")
    else:
        st.warning(f"{w['date_label'] or '날짜 미정'} · 날씨 값 없음 — {w['reason']}" + (f" (조회 가능 {w['range']})" if w.get("range") else ""))

    for note in view["notes"]:
        st.caption("ℹ️ " + note)

    with st.expander("자세한 설명 보기"):
        st.markdown(m["content"])  # 저장된 답변 글을 그대로 보인다

    with st.expander("조회 근거 보기"):
        tab_cond, tab_a, tab_b, tab_log = st.tabs(["조건·검증", "장소 원결과", "날씨 원결과", "실행 기록"])
        t = m.get("trip") or {}
        with tab_cond:
            show_trip({**t, "results": None}, flat=True)
            s = view["sources"]
            st.caption(f"장소 출처: {s['places']['source'] or '없음'} · 조회 시각 {s['places']['retrieved_at'] or '없음'}")
            st.caption(f"날씨 출처: {s['weather']['source'] or '없음'} · 조회 시각 {s['weather']['retrieved_at'] or '없음'}")
        with tab_a:
            if t.get("results"):
                render_result(t["results"]["A"])
        with tab_b:
            if t.get("results"):
                render_result(t["results"]["B"])
        with tab_log:
            if "log" in m:
                show_log(m["log"], m.get("run"), flat=True)
            else:
                st.caption("실행 기록이 없어요.")


def render_message(m: dict, idx: int, latest_plan: bool) -> None:
    """저장된 대화 한 칸을 그린다. 계획 결과는 구조화 화면, 그 외는 예전 방식."""
    view = view_model.build_view(m.get("trip")) if m["role"] == "assistant" else None
    if view:
        render_structured(m, view)
    else:
        st.markdown(m["content"])
        show_places(m.get("places", []))
        show_trip(m.get("trip"))
        if not (m.get("trip") or {}).get("results"):  # 여행 계획 흐름은 B 원결과 칸이 날씨를 보여준다
            show_weather(m.get("weather", []))
        if m["role"] == "assistant" and "log" in m:  # 실행 기록이 없던 예전 저장 대화는 건너뛴다
            show_log(m["log"], m.get("run"))
            if "compare" in m:
                show_compare(m)
    if latest_plan:
        show_preview(idx, m)


def friendly_error(err: Exception) -> str:
    # 예외 메시지에는 키 일부가 포함될 수 있으므로 원문 str(err) 는 보여주지 않는다.
    if isinstance(err, openai.BadRequestError):
        reason = getattr(err, "message", "") or "요청 형식 오류"
        if config.OPENAI_API_KEY:
            reason = reason.replace(config.OPENAI_API_KEY, "***")
        return f"요청이 거부되었습니다: {reason}"
    if isinstance(err, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. .env 의 OPENAI_API_KEY 를 확인해 주세요."
    if isinstance(err, openai.NotFoundError):
        return f"모델 '{config.OPENAI_MODEL}' 을(를) 사용할 수 없습니다. .env 의 OPENAI_MODEL 을 확인해 주세요."
    if isinstance(err, openai.RateLimitError):
        return "요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도해 주세요."
    if isinstance(err, GraphRecursionError):
        return "장소 조회가 너무 많이 반복되어 중단했어요. 질문을 더 구체적으로 나눠서 다시 물어봐 주세요."
    if isinstance(err, httpx.HTTPError):
        return "네트워크 오류가 발생했습니다. 인터넷 연결을 확인해 주세요."
    if isinstance(err, openai.APIConnectionError):
        return "OpenAI 서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요."
    return f"답변을 가져오는 중 오류가 발생했습니다. ({type(err).__name__})"


def classify(question: str, answer: str) -> tuple[str, str]:
    """대화의 (제목, 카테고리)를 정한다. 실패하면 질문 앞부분과 '기타'."""
    fallback = (question[:20], storage.DEFAULT_CATEGORY)
    try:
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
        )
        prompt = (
            "아래 여행 대화를 분류하세요. JSON 한 줄만 출력하세요: "
            '{"title": "목적지와 핵심을 담은 15자 이내 제목", "category": "<카테고리>"}\n'
            f"카테고리는 다음 중 하나: {', '.join(storage.CATEGORIES)}\n"
            "여행 전체 계획이면 '여행 일정', 식당·카페 중심이면 '맛집·카페'로 하세요.\n\n"
            f"질문: {question}\n답변(앞부분): {answer[:600]}"
        )
        text = str(llm.invoke(prompt).text)
        data = json.loads(text[text.index("{") : text.rindex("}") + 1])
        title = str(data.get("title", "")).strip()[:30] or fallback[0]
        category = data.get("category", "")
        return title, category if category in storage.CATEGORIES else storage.DEFAULT_CATEGORY
    except Exception:
        return fallback


def open_chat(chat_id: str) -> None:
    chat = storage.get_chat(chat_id)
    if chat:
        st.session_state.chat_id = chat_id
        st.session_state.history = chat["messages"]
        st.query_params["chat"] = chat_id


def start_new_chat() -> None:
    st.session_state.chat_id = storage.new_id()
    st.session_state.history = []
    st.query_params.clear()
    if st.session_state.get("view_only"):
        st.query_params["view"] = "1"  # 보기 전용은 새 대화로 넘어가도 유지한다


def remove_chat(chat_id: str) -> None:
    storage.delete_chat(chat_id)
    if st.session_state.chat_id == chat_id:
        start_new_chat()


st.set_page_config(page_title="여행 챗봇", page_icon="✈️", layout="centered", initial_sidebar_state="auto")  # 넓은 화면에서는 열리고, 좁은 화면(휴대폰)에서는 본문이 먼저 보이게 접힌다
st.markdown(ui_style.CSS, unsafe_allow_html=True)  # 색의 큰 틀은 .streamlit/config.toml, 세부는 ui_style.py

# 세션 소유자 id 와 보기 전용 여부: 한 번 정해지면 이 세션 안에서 바뀌지 않는다(보기 전용을 풀 수 없다).
if "owner_id" not in st.session_state:
    st.session_state.owner_id = uuid.uuid4().hex
if "view_only" not in st.session_state:
    st.session_state.view_only = config.APP_VIEW_ONLY or st.query_params.get("view") == "1"

if "chat_id" not in st.session_state:
    # 새로고침해도 주소의 ?chat=... 으로 대화를 복원한다.
    saved_id = st.query_params.get("chat", "")
    start_new_chat()
    open_chat(saved_id)

with st.sidebar:
    st.markdown('<div class="ramos-brand">✈️ 여행 챗봇</div>', unsafe_allow_html=True)
    st.button("✏️ 새 대화 시작", use_container_width=True, on_click=start_new_chat, type="primary")

    st.markdown('<div class="ramos-section">저장된 대화</div>', unsafe_allow_html=True)
    saved = storage.list_chats()
    if not saved:
        st.caption("아직 저장된 대화가 없어요. 질문하면 자동으로 저장돼요.")
    for category in storage.CATEGORIES:
        group = [c for c in saved if c["category"] == category]
        if not group:
            continue
        with st.expander(f"{category} ({len(group)})", expanded=any(c["id"] == st.session_state.chat_id for c in group)):
            for c in group:
                open_col, del_col = st.columns([5, 1])
                label = ("▶ " if c["id"] == st.session_state.chat_id else "") + (c["title"] or "제목 없음")
                open_col.button(label, key=f"open_{c['id']}", on_click=open_chat, args=(c["id"],), use_container_width=True)
                del_col.button("🗑️", key=f"del_{c['id']}", on_click=remove_chat, args=(c["id"],), help="삭제")

    # 설정은 모두 여기에 모은다: 답변 방식, 연결 상태, 캘린더
    st.markdown('<div class="ramos-section">설정</div>', unsafe_allow_html=True)
    st.toggle(
        "🔎 조사만 보기 (종합 안 함)",
        key="investigate_only",
        help="여행 계획 요청에서 A 장소 조사·B 날씨 조사 원결과만 보여주고 계획으로 종합하지 않아요.",
    )
    st.caption(f"모델: {config.OPENAI_MODEL}")
    if config.KAKAO_REST_API_KEY:
        st.caption("🗺️ 카카오맵 장소 조회: 켜짐 (국내 장소)")
    else:
        st.caption("🗺️ 카카오맵 장소 조회: 꺼짐 (KAKAO_REST_API_KEY 없음)")
    show_tracing_status()
    show_calendar_status()

# 첫 화면 위쪽: 제목 · 한 문장 소개 · 입력 예시. 설정은 사이드바에, 본문은 대화만 둔다.
st.title("여행 챗봇")
st.markdown(f'<p class="ramos-intro">{INTRO}</p>', unsafe_allow_html=True)

if not config.OPENAI_API_KEY:
    st.warning(".env 파일에 OPENAI_API_KEY 가 없습니다. .env.example 을 참고해 설정해 주세요.")

if not st.session_state.history:
    st.markdown('<div class="ramos-label">이렇게 물어보세요</div>', unsafe_allow_html=True)
    with st.container(key="examples"):
        for i, example in enumerate(EXAMPLES):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = example  # 예시를 누르면 그대로 질문으로 보낸다
                st.rerun()
# 일정 미리보기는 가장 최근 여행 계획에만 붙인다(위젯이 너무 많아지지 않게).
_plan_idx = max(
    (i for i, h in enumerate(st.session_state.history) if h["role"] == "assistant" and (h.get("trip") or {}).get("complete") and (h["trip"].get("synthesized", True))),
    default=-1,
)
for _i, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"]):
        render_message(m, _i, _i == _plan_idx)
# 같은 요청을 다시 실행해 두 기록을 나란히 비교한다 (마지막 답변에만 제공).
_last = st.session_state.history[-1] if st.session_state.history else None
if _last and _last["role"] == "assistant" and "log" in _last and "compare" not in _last:
    if st.button("🔬 같은 질문으로 다시 실행해 비교", key="compare_btn"):
        with st.spinner("같은 질문으로 다시 실행하는 중..."):
            try:
                again = trip.run_turn(
                    st.session_state.history[:-1],
                    synthesize=not st.session_state.get("investigate_only", False),
                    langsmith_extra={"metadata": {"chat_id": st.session_state.chat_id, "rerun_for_compare": True}},
                )
            except Exception as err:
                st.error(friendly_error(err))
            else:
                _last["compare"] = {
                    "answer": again.answer,
                    "places": again.places,
                    "log": again.log,
                    "run": again.run,
                }
                storage.save_chat(st.session_state.chat_id, st.session_state.history)
                st.rerun()

question = st.chat_input("지역·날짜·취향을 알려 주세요. 예) 모레 경주 박물관 중심 하루 여행")
if question is None:
    question = st.session_state.pop("pending_question", None)  # 예시 버튼으로 들어온 질문

if question is not None:
    if not question.strip():
        st.info("질문을 입력해 주세요.")
    elif not config.OPENAI_API_KEY:
        st.error("API 키가 설정되지 않아 답변할 수 없습니다.")
    else:
        question = question.strip()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("답변을 만드는 중..."):
                try:
                    result = trip.run_turn(
                        st.session_state.history + [{"role": "user", "content": question}],
                        synthesize=not st.session_state.get("investigate_only", False),
                        langsmith_extra={"metadata": {"chat_id": st.session_state.chat_id}},
                    )
                    answer = result.answer
                except Exception as err:
                    st.error(friendly_error(err))  # 실패한 질문은 기록에 남기지 않는다
                else:
                    st.markdown(answer)
                    show_places(result.places)
                    show_trip(result.trip)
                    if not (result.trip or {}).get("results"):
                        show_weather(result.weather)
                    show_log(result.log, result.run)
                    st.session_state.history.append({"role": "user", "content": question})
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "places": result.places,
                            "weather": result.weather,
                            "trip": result.trip,
                            "log": result.log,
                            "run": result.run,
                        }
                    )
                    saved_chat = storage.get_chat(st.session_state.chat_id)
                    title, category = ("", "") if saved_chat and saved_chat["title"] else classify(question, answer)
                    storage.save_chat(st.session_state.chat_id, st.session_state.history, title, category)
                    st.query_params["chat"] = st.session_state.chat_id
                    st.rerun()  # 사이드바의 저장 목록을 갱신

