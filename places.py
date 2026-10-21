"""카카오 로컬 API 장소 검색 (LangChain 도구)."""

from contextvars import ContextVar

import httpx
from langchain_core.tools import ToolException, tool

import config

KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

# 카카오 카테고리 그룹 코드 (모델이 쓰기 쉬운 이름 → 코드)
CATEGORIES = {
    "맛집": "FD6",
    "카페": "CE7",
    "관광명소": "AT4",
    "숙박": "AD5",
    "문화시설": "CT1",
    "지하철역": "SW8",
    "주차장": "PK6",
}


# 카카오 키워드 검색은 장소명·종류에 대한 검색이라, 아래 같은 '조건어'가 들어가면 결과가 0건이 된다.
# (예: '경주 실내 박물관' 0건, '경주 박물관' 5건). 조건은 검색어가 아니라 결과를 보고 판단한다.
CONDITION_WORDS = {
    "실내", "야외", "부모님", "부모님과", "아이와", "아이랑", "아이", "가족", "가족과", "커플", "혼자", "친구와",
    "조용한", "한적한", "분위기", "분위기좋은", "분위기 좋은", "예쁜", "인기", "유명한", "맛있는", "가성비",
    "추천", "근처", "주변", "갈", "가기", "좋은",
}


def clean_query(query: str) -> str:
    """검색어에서 조건어를 뺀다. 전부 빠져 비게 되면 원래 검색어를 그대로 쓴다."""
    kept = [w for w in query.split() if w not in CONDITION_WORDS]
    return " ".join(kept) if len(kept) >= 2 else query.strip()


def full_query(region: str, query: str) -> str:
    """지역이 검색어에 없으면 앞에 붙인다."""
    region, query = region.strip(), query.strip()
    return query if region in query else f"{region} {query}".strip()


class PlaceSearchError(Exception):
    """사용자에게 보여줘도 안전한 장소 검색 오류 (키 값은 포함하지 않는다)."""


class PlaceAuthError(Exception):
    """키·앱 설정 문제로 다시 시도해도 소용없는 인증 오류.
    ToolException 이 아니므로 에이전트 실행을 곧바로 중단시키고, agent.run 이 안내 답변을 만든다.
    call_args: 실행 기록에 남길 도구 인자."""

    call_args: dict | None = None


def search_kakao(query: str, category: str = "", size: int = 5) -> list[dict]:
    if not config.KAKAO_REST_API_KEY:
        raise PlaceSearchError("KAKAO_REST_API_KEY 가 설정되지 않았습니다.")
    params = {"query": query, "size": max(1, min(size, 10))}
    code = CATEGORIES.get(category)
    if code:
        params["category_group_code"] = code
    try:
        resp = httpx.get(
            KEYWORD_URL,
            params=params,
            headers={"Authorization": f"KakaoAK {config.KAKAO_REST_API_KEY}"},
            timeout=10,
        )
    except httpx.HTTPError:
        raise PlaceSearchError("카카오 서버에 연결하지 못했습니다.") from None
    if resp.status_code in (401, 403):
        raise PlaceAuthError(f"카카오 인증 실패 (HTTP {resp.status_code})")
    if resp.status_code == 429:
        raise PlaceSearchError("장소 검색 호출 한도를 넘었어요. 잠시 후 다시 시도해 주세요.")
    if resp.status_code != 200:
        raise PlaceSearchError(f"카카오 장소 검색 오류 (HTTP {resp.status_code})")
    return [
        {
            "name": d["place_name"],
            "category": d.get("category_name", ""),
            "address": d.get("road_address_name") or d.get("address_name", ""),
            "phone": d.get("phone", ""),
            "url": d.get("place_url", ""),
            "lat": d.get("y", ""),
            "lng": d.get("x", ""),
        }
        for d in resp.json().get("documents", [])
    ]


# 이번 질문에서 이미 보낸 (검색어, 종류). agent.run 이 질문마다 비운다.
_seen: ContextVar[set | None] = ContextVar("seen_searches", default=None)

ENOUGH = 3  # 후보가 이 수 이상이면 추가 검색 없이 답해도 충분하다고 안내


_allow_widen: ContextVar[bool] = ContextVar("allow_widen", default=True)


def reset_seen(failed_before: set | None = None, allow_widen: bool = False) -> None:
    """질문마다 호출한다. failed_before: 이 대화에서 이미 0건이었던 (검색어, 종류).
    allow_widen: 사용자가 '다른 종류도'를 요청했는지."""
    _seen.set(set(failed_before or ()))
    _allow_widen.set(allow_widen)


def format_places(places: list[dict]) -> str:
    verdict = (
        f"후보 {len(places)}곳: 충분합니다. 추가 검색 없이 이 결과로 답하세요."
        if len(places) >= ENOUGH
        else f"후보 {len(places)}곳: 적습니다. 핵심 조건은 유지하고 검색어나 종류를 바꿔 한 번 더 찾을 수 있습니다."
    )
    return verdict + "\n" + "\n".join(
        f"- {p['name']} | {p['category']} | {p['address']} | {p.get('phone') or '전화 정보 없음'} | {p['url']}"
        for p in places
    )


@tool(response_format="content_and_artifact")
def search_places(region: str, query: str, category: str = "", widened_kind: str = "") -> tuple[str, list[dict]]:
    """카카오맵에서 실제 장소를 검색한다. 장소 추천이나 일정에 넣을 장소는 반드시 이 도구로 먼저 조회한다.
    날짜는 필요 없다. 인사·일반 상담처럼 장소가 필요 없는 질문에는 호출하지 않는다.

    region: 검색할 지역(시·군·동·역 이름 등). 필수. 사용자의 질문이나 앞선 대화에 지역이 없으면
        이 도구를 호출하지 말고 먼저 사용자에게 지역을 묻는다. 예: '경주', '제주 애월'.
    query: 장소 종류/이름 중심의 짧은 검색어(지역 제외). 예: '돼지국밥', '박물관'.
        실내·조용한·부모님과 같은 조건어는 넣지 않는다(검색이 0건이 된다). 조건은 결과를 보고 고른다.
    category: 선택. 맛집, 카페, 관광명소, 숙박, 문화시설, 지하철역, 주차장 중 하나. 모르면 비워 둔다.
        이 값은 결과를 해당 분류로 제한하므로, 다른 종류로 넓힐 때는 새 종류에 맞게 바꾸거나 비운다.
    widened_kind: 사용자가 '후보가 없으면 다른 종류도'라고 요청해서 장소 종류를 넓혀 검색할 때만,
        넓힌 종류를 적는다(예: '카페'). 그 외에는 반드시 비워 둔다.
    """
    if widened_kind.strip() and not _allow_widen.get():
        raise ToolException(
            "사용자가 다른 종류로 넓혀 달라고 요청하지 않았습니다. widened_kind 를 비우고, "
            "같은 종류 안에서 검색어만 바꾸거나 지금 결과로 답하세요."
        )
    region = region.strip()
    if not region:
        raise ToolException(
            "지역이 없어 검색할 수 없습니다. 도구를 다시 부르지 말고 사용자에게 어느 지역인지만 물어보세요."
        )
    query = full_query(region, query)
    cleaned = clean_query(query)
    seen = _seen.get()
    key = (cleaned, category)
    if seen is not None:
        if key in seen:
            raise ToolException(
                "같은 검색어·종류로 이미 검색했습니다(조건어를 뺀 검색어 기준, 이전 대화에서 0건이었던 검색 포함). "
                "반복하지 말고 검색어나 종류를 바꾸거나 지금 결과로 답하세요."
            )
        seen.add(key)
    try:
        places = search_kakao(cleaned, category)
    except PlaceAuthError as err:
        # 재시도해도 소용없는 실패: ToolException 으로 바꾸지 않고 그대로 올려 실행을 멈춘다.
        err.call_args = {"region": region, "query": query, "category": category, "widened_kind": widened_kind}
        raise
    except PlaceSearchError as err:
        # ToolException 은 실행 기록의 status 를 'error' 로 남기고, 메시지만 모델에게 전달한다.
        raise ToolException(f"검색 실패: {err}") from None
    note = (
        f"(검색어의 조건어를 빼고 '{cleaned}'로 검색했습니다. 실내/야외·분위기·동행 같은 조건은 검색되지 않으니 "
        "결과 목록을 보고 직접 골라 답에 반영하세요.)\n"
        if cleaned != query.strip()
        else ""
    )
    if not places:
        return (
            note
            + "검색 결과 0건입니다. 지역 같은 핵심 조건은 유지하고, 검색어를 '지역 + 장소 종류'로 단순하게 바꾸거나 "
            "category 필터를 바꿔 한 번만 다시 검색하세요. 같은 인자는 반복할 수 없습니다. "
            "사용자가 요청하지 않았다면 다른 종류로 넓히지 마세요.",
            [],
        )
    return note + format_places(places), places


# 도구 오류를 예외로 터뜨리지 않고 status='error' 메시지로 모델에게 돌려준다.
search_places.handle_tool_error = True
