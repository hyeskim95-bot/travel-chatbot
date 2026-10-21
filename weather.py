"""날씨 조회: 상대 날짜 → 구체 날짜(코드가 계산), 장소 검색 좌표 → Open-Meteo 예보 (LangChain 도구)."""

import calendar
import re
from datetime import date, datetime, timedelta

import httpx
from langchain_core.tools import ToolException, tool

import places

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo.com (CC BY 4.0)"  # 무료 API 는 비상업 용도·출처 표기 조건이 있다(open-meteo.com/en/terms)
MAX_FORECAST_DAYS = 16  # Open-Meteo 예보 범위: 오늘 포함 16일
WEEKDAYS = "월화수목금토일"
_WD = {c: i for i, c in enumerate(WEEKDAYS)}


def weekday_name(d: date) -> str:
    return WEEKDAYS[d.weekday()]


def add_months(d: date, n: int) -> date:
    """n개월 뒤(앞) 같은 날짜. 그 달에 없는 날(예: 31일)이면 그 달의 마지막 날로 맞춘다."""
    year, month0 = divmod(d.year * 12 + (d.month - 1) + n, 12)
    month = month0 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def resolve_date(expr: str, today: date) -> date | None:
    """'이번 주 토요일', '내일', '10월 24일', '2026-10-24' 같은 표현을 구체 날짜로 바꾼다.
    해석할 수 없으면 None. 모델이 날짜를 계산하지 않도록 이 함수가 맡는다."""
    s = expr.strip().replace(" ", "")
    s = re.sub(r"(에는|에|쯤|경|즈음|정도|이요|요)$", "", s)  # '한 달 뒤에', '토요일이요', '24일쯤' 같은 꼬리말
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    fixed = {"오늘": 0, "내일": 1, "모레": 2, "글피": 3, "어제": -1, "그제": -2, "그저께": -2}
    if s in fixed:
        return today + timedelta(days=fixed[s])
    nums = {"한": 1, "두": 2, "세": 3, "네": 4, "일": 1}
    m = re.fullmatch(r"(\d+)일(후|뒤|전)", s)
    if m:
        return today + timedelta(days=int(m[1]) * (-1 if m[2] == "전" else 1))
    m = re.fullmatch(r"(\d+|한|두|세|네)(달|개월)(후|뒤|전)", s)  # 한 달 뒤, 2개월 후
    if m:
        n = int(m[1]) if m[1].isdigit() else nums[m[1]]
        return add_months(today, n * (-1 if m[3] == "전" else 1))
    m = re.fullmatch(r"(\d+|한|두|세|네|일)주(?:일)?(후|뒤|전)", s)  # 2주 뒤, 일주일 뒤
    if m:
        n = int(m[1]) if m[1].isdigit() else nums[m[1]]
        return today + timedelta(weeks=n * (-1 if m[2] == "전" else 1))
    monday = today - timedelta(days=today.weekday())
    m = re.fullmatch(r"(지난|저번|이번|다음|다다음)주?([월화수목금토일])요일", s)
    if m:
        weeks = {"지난": -1, "저번": -1, "이번": 0, "다음": 1, "다다음": 2}[m[1]]
        return monday + timedelta(weeks=weeks, days=_WD[m[2]])
    m = re.fullmatch(r"(이번|다음)주말", s)
    if m:
        return monday + timedelta(weeks={"이번": 0, "다음": 1}[m[1]], days=5)
    m = re.fullmatch(r"([월화수목금토일])요일", s)  # 요일만 말하면 오늘 이후 가장 가까운 그 요일
    if m:
        return today + timedelta(days=(_WD[m[1]] - today.weekday()) % 7)
    m = re.fullmatch(r"(지난|저번|이번|다음|다다음)(?:달|월)(\d{1,2})일", s)  # 다음 달 20일
    if m:
        base = add_months(today.replace(day=1), {"지난": -1, "저번": -1, "이번": 0, "다음": 1, "다다음": 2}[m[1]])
        try:
            return base.replace(day=int(m[2]))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})월(\d{1,2})일", s)
    if m:
        try:
            d = date(today.year, int(m[1]), int(m[2]))
        except ValueError:
            return None
        return d if d >= today else d.replace(year=today.year + 1)
    return None


def fetch_forecast(lat: str, lng: str, target: date) -> dict:
    """Open-Meteo 일별 예보 1일치. 실패하면 ToolException(사용자에게 보여도 안전한 메시지)."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "Asia/Seoul",
        "start_date": target.isoformat(),
        "end_date": target.isoformat(),
    }
    try:
        resp = httpx.get(FORECAST_URL, params=params, timeout=10)
    except httpx.HTTPError:
        raise ToolException("날씨 서버(Open-Meteo)에 연결하지 못했습니다.") from None
    if resp.status_code != 200:
        raise ToolException(f"날씨 조회 오류 (HTTP {resp.status_code}). 예보 범위를 벗어난 날짜일 수 있습니다.")
    daily = resp.json().get("daily", {})
    try:
        out = {
            "time": daily["time"][0],  # API 가 돌려준 날짜: 요청한 날짜와 같은지 호출하는 쪽에서 확인한다
            "tmax": daily["temperature_2m_max"][0],
            "tmin": daily["temperature_2m_min"][0],
            "pop": daily["precipitation_probability_max"][0],
        }
    except (KeyError, IndexError):
        raise ToolException("날씨 데이터를 받지 못했습니다.") from None
    if out["tmax"] is None or out["tmin"] is None:
        raise ToolException("날씨 데이터를 받지 못했습니다.")
    return out


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
COORD_KAKAO = "카카오 장소 검색"
COORD_FALLBACK = "OpenStreetMap Nominatim (장소 검색 실패로 대체)"


def geocode_fallback(region: str) -> dict:
    """장소 검색(카카오)이 안 될 때만 쓰는 대체 좌표 조회. 찾지 못하면 ToolException."""
    try:
        resp = httpx.get(
            NOMINATIM_URL,
            params={"q": region, "format": "jsonv2", "countrycodes": "kr", "limit": 1, "accept-language": "ko"},
            headers={"User-Agent": "travel-chatbot-class-project/1.0"},
            timeout=10,
        )
        items = resp.json() if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        items = []
    if not items:
        raise ToolException(f"'{region}' 위치를 찾지 못했습니다(장소 검색과 대체 좌표 조회 모두 실패).")
    it = items[0]
    return {"name": region, "address": it["display_name"][:60], "lat": it["lat"], "lng": it["lon"]}


def lookup(region: str, date: str, fallback_geocode: bool = False) -> tuple[str, dict]:
    """날씨 조회 본체. 에이전트 도구(get_weather)와 여행 계획의 B 역할 함수가 같이 쓴다.
    ToolException 은 사용자에게 보여도 안전한 메시지를 담는다.
    fallback_geocode=True 면 장소 검색이 실패해도 대체 좌표 조회로 날씨를 계속 조회한다(여행 계획의 B 역할)."""
    region = region.strip()
    if not region:
        raise ToolException("지역이 없어 날씨를 조회할 수 없습니다. 도구를 다시 부르지 말고 사용자에게 지역을 물어보세요.")
    today = datetime.now().date()
    target = resolve_date(date, today)
    if target is None:
        raise ToolException(
            f"날짜 '{date}'를 해석하지 못했습니다. 도구를 다시 부르지 말고 사용자에게 날짜를 확인하세요."
        )
    # 예보 범위(오늘 ~ 오늘+15일) 밖이면 API 를 부르지 않는다. 현재 날씨나 다른 날짜 값으로 대신 채우지 않는다.
    last_day = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    if target < today or target > last_day:
        reason = "past" if target < today else "future"
        record = {
            "status": "out_of_range",
            "reason": reason,
            "date": target.isoformat(),
            "weekday": weekday_name(target),
            "range_start": today.isoformat(),
            "range_end": last_day.isoformat(),
            "source": SOURCE,
            "label": f"{region} {target.isoformat()}({weekday_name(target)}) 날씨",
        }
        why = "이미 지난 날짜" if reason == "past" else f"예보 범위(오늘부터 {MAX_FORECAST_DAYS}일) 밖"
        text = (
            f"예보 없음 (out_of_range): {record['date']}({record['weekday']})는 {why}입니다. "
            f"예보는 {record['range_start']} ~ {record['range_end']} 만 있습니다. "
            "현재 날씨나 다른 날짜의 값으로 대신 답하지 말고, 날씨 값은 말하지 마세요. 범위 안의 날짜를 안내하세요."
        )
        return text, record

    # 좌표는 모델이 아니라 장소 검색 결과(x 경도, y 위도)에서 가져온다.
    coord_source = COORD_KAKAO
    try:
        found = places.search_kakao(region, size=1)  # 인증 오류(PlaceAuthError)는 대체 조회가 꺼져 있으면 그대로 올려 실행을 멈춘다
    except (places.PlaceAuthError, places.PlaceSearchError):
        if not fallback_geocode:
            raise
        found, coord_source = [geocode_fallback(region)], COORD_FALLBACK
    if not found:
        raise ToolException(f"'{region}' 위치를 장소 검색에서 찾지 못했습니다. 지역 이름을 다시 확인해 주세요.")
    place = found[0]
    forecast = fetch_forecast(place["lat"], place["lng"], target)

    fetched_at = datetime.now().isoformat(timespec="seconds")
    if forecast["time"] != target.isoformat():
        # 반환 날짜가 요청과 다르면 그 결과를 쓰지 않는다.
        record = {
            "status": "date_mismatch",
            "date": target.isoformat(),
            "weekday": weekday_name(target),
            "returned_date": forecast["time"],
            "source": SOURCE,
            "fetched_at": fetched_at,
            "label": f"{region} {target.isoformat()}({weekday_name(target)}) 날씨",
        }
        text = (
            f"날짜 불일치 (date_mismatch): 요청 {record['date']} / 반환 {forecast['time']}. "
            "이 결과는 사용할 수 없습니다. 날씨 값(기온·강수확률)을 답하지 말고 확인이 필요하다고 알리세요."
        )
        return text, record

    record = {
        "status": "ok",
        "date": target.isoformat(),
        "returned_date": forecast["time"],
        "weekday": weekday_name(target),
        "tmax": forecast["tmax"],
        "tmin": forecast["tmin"],
        "pop": forecast["pop"],
        "place": place["name"],
        "address": place["address"],
        "lat": place["lat"],
        "lng": place["lng"],
        "coord_source": coord_source,
        "source": SOURCE,
        "fetched_at": fetched_at,
        "label": f"{region} {target.isoformat()}({weekday_name(target)}) 날씨",
    }
    text = (
        f"status ok | 날짜: {record['date']}({record['weekday']}) | 최고 {record['tmax']}°C | 최저 {record['tmin']}°C | "
        f"강수확률 {record['pop']}% | 위치: {record['place']} ({record['address']}, 경도 {record['lng']}, 위도 {record['lat']}) | "
        f"출처: Open-Meteo | 조회 시각: {fetched_at}"
    )
    return text, record


@tool(response_format="content_and_artifact")
def get_weather(region: str, date: str) -> tuple[str, dict]:
    """여행지의 하루치 날씨 예보(최고·최저 기온, 강수확률)를 조회한다. 옷차림·우산·준비물 조언에 쓴다.
    지역과 날짜가 모두 있을 때만 호출한다. 날짜가 없으면 호출하지 말고 사용자에게 날짜를 먼저 묻는다.

    region: 날씨를 볼 지역이나 장소 이름. 필수. 예: '경주', '제주 애월'. 좌표는 직접 만들지 않는다
        (도구가 장소 검색 결과의 좌표를 쓴다).
    date: 사용자가 말한 날짜 표현 그대로(예: '이번 주 토요일', '내일', '10월 24일') 또는 YYYY-MM-DD.
        날짜 계산은 도구가 하므로 직접 계산하지 않는다.
    """
    return lookup(region, date)


get_weather.handle_tool_error = True
