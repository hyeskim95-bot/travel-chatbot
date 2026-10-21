"""종합 직전 검증: A·B 결과의 지역·날짜·status 를 공통 조건과 대조한다.

지역을 바꿨는데 이전 지역 날씨가 섞이는 것을 막는다. 입력이 모두 공통 조건과 같고 status 가 ok 면 그냥 통과시키고,
불일치가 있으면 잘못된 쪽(주로 날씨)을 계획의 근거에서 빼고 경고를 남긴다. 경고 문구만 붙이고 그대로 쓰지 않는다.
"""

import math
from dataclasses import dataclass, field
from statistics import median

# 날씨 위치와 장소 후보 위치가 이보다 멀면 서로 다른 지역으로 본다 (예: 경주 ↔ 부산은 약 85km).
MAX_DISTANCE_KM = 60


@dataclass
class Verdict:
    ok: bool  # 경고 없이 통과했는지
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # 불일치 경고(사용자에게 보여줄 문장)
    notes: list[str] = field(default_factory=list)  # 불일치는 아니지만 알아둘 상태(예: 예보 없음)
    exclude_weather: bool = False  # 날씨를 계획의 근거에서 뺀다
    exclude_places: bool = False  # 장소를 계획의 근거에서 뺀다

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "warnings": self.warnings,
            "notes": self.notes,
            "exclude_weather": self.exclude_weather,
            "exclude_places": self.exclude_places,
        }


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _check(name: str, expected, actual, passed: bool, culprit: str = "") -> dict:
    return {"name": name, "expected": expected, "actual": actual, "pass": bool(passed), "culprit": culprit}


def verify(cond, a: dict, b: dict) -> Verdict:
    """cond: TripCondition, a·b: trip.result_a / result_b 가 만든 같은 모양의 결과."""
    checks = [
        _check("A 입력 지역 = 공통 조건", cond.region, a["input"].get("region"), a["input"].get("region") == cond.region, "A"),
        _check("B 입력 지역 = 공통 조건", cond.region, b["input"].get("region"), b["input"].get("region") == cond.region, "B"),
        _check("A 입력 날짜 = 공통 조건", cond.date, a["input"].get("date"), a["input"].get("date") == cond.date, "A"),
        _check("B 입력 날짜 = 공통 조건", cond.date, b["input"].get("date"), b["input"].get("date") == cond.date, "B"),
    ]
    weather_ok = b["status"] == "ok"
    if weather_ok:
        # 날씨 값 자체가 어느 날짜·어느 위치의 것인지도 대조한다.
        d = b["data"]
        checks.append(_check("날씨 값의 날짜 = 공통 날짜", cond.date, d.get("date"), d.get("date") == cond.date, "B"))
        coords = [(float(p["lat"]), float(p["lng"])) for p in a["data"]["places"] if p.get("lat") and p.get("lng")]
        if coords:
            dist = median(haversine_km(float(d["lat"]), float(d["lng"]), la, ln) for la, ln in coords)
            checks.append(
                _check(f"날씨 위치 ↔ 장소 위치 거리 ≤ {MAX_DISTANCE_KM}km", f"≤ {MAX_DISTANCE_KM}km", f"{dist:.0f}km", dist <= MAX_DISTANCE_KM, "B")
            )

    verdict = Verdict(ok=True, checks=checks)
    for c in checks:
        if c["pass"]:
            continue
        verdict.ok = False
        if c["culprit"] == "A":
            verdict.exclude_places = True
        else:
            verdict.exclude_weather = True
        verdict.warnings.append(f"{c['name']} 불일치: 기대 {c['expected']} / 실제 {c['actual']}")

    # status 는 불일치와 별개로 기록한다(예보 없음 등은 경고가 아니라 알림).
    for res in (a, b):
        if res["status"] != "ok":
            verdict.notes.append(f"{res['role']} {res['name']} status: {res['status']}")
        checks.append(_check(f"{res['role']} status", "ok", res["status"], res["status"] == "ok", "status"))
    return verdict


def warning_block(verdict: Verdict) -> str:
    """계획 맨 앞에 붙이는 불일치 경고(날씨를 뺐다는 사실과 확인이 필요하다는 안내)."""
    if verdict.ok or not verdict.warnings:
        return ""
    what = []
    if verdict.exclude_weather:
        what.append("날씨는 계획의 근거에서 뺐고, 잘못된 날씨로 계획을 확정하지 않았어요. **날씨 확인이 필요해요.**")
    if verdict.exclude_places:
        what.append("장소 조사 결과가 공통 조건과 달라 장소를 근거에서 뺐어요. 다시 조사가 필요해요.")
    lines = "\n".join(f"> - {w}" for w in verdict.warnings)
    return "> ⚠️ **불일치 경고** — 조사 결과가 공통 조건과 맞지 않아요.\n" + lines + "\n> " + " ".join(what) + "\n\n"
