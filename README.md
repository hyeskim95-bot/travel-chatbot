# 여행 챗봇

지역·날짜·취향을 말하면 카카오맵 장소와 Open-Meteo 날씨를 조회해 근거 있는 하루 여행 계획을 만들어 주는 Streamlit 챗봇입니다. (LangChain · OpenAI · Google Calendar MCP · LangSmith 추적)

## 실행
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # .env 에 키를 채운다 (절대 커밋하지 않는다)
streamlit run app.py
```

## 환경변수 (`.env.example` 참고)
`OPENAI_API_KEY`, `OPENAI_MODEL`, `KAKAO_REST_API_KEY`, `GOOGLE_CALENDAR_ID`, `GOOGLE_CREDENTIALS_FILE`, `LANGSMITH_*`

## 구성
- `app.py` 화면 · `trip.py` 계획 흐름 · `trip_plan.py` 계획 상태 · `plan_writer.py` 답변 작성/검사
- `places.py` 카카오 장소 · `weather.py` 날씨 · `view_model.py` 결과 화면 데이터
- `calendar_*.py`, `registration.py` 캘린더 미리보기·등록(확인 후에만 등록)
- `docs/` 2단계 결과 화면 스크린샷과 데이터 대조 기록