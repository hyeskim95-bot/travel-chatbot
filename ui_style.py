"""화면 모양: 라모스 색(파랑 #0F3192 · 회색 #7F7F7F · 흰색)과 글자 위계·여백.

색의 큰 틀(배경·강조색·사이드바)은 .streamlit/config.toml 의 theme 가 정하고,
테마 옵션으로는 안 되는 세부만 여기서 CSS 로 맞춘다: 제목 색, 대화 말풍선, 예시 버튼, 사이드바 간격.
"""

BLUE = "#0F3192"
GRAY = "#7F7F7F"
TINT = "#F2F4FA"

CSS = f"""
<style>
:root {{ --ramos-blue: {BLUE}; --ramos-gray: {GRAY}; --ramos-tint: {TINT}; }}

/* 본문 폭과 위아래 여백 */
.block-container {{ max-width: 52rem; padding-top: 2.4rem; padding-bottom: 6rem; }}

/* 글자 위계: 제목·소제목은 파랑, 설명은 회색 */
div[data-testid="stHeading"] h1, [data-testid="stMarkdownContainer"] h1 {{ color: var(--ramos-blue) !important; letter-spacing: -0.02em; margin-bottom: 0.2rem; }}
div[data-testid="stHeading"] h2, div[data-testid="stHeading"] h3,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{
  color: var(--ramos-blue) !important; letter-spacing: -0.01em;
}}
[data-testid="stCaptionContainer"], .ramos-intro {{ color: var(--ramos-gray); }}
.ramos-intro {{ font-size: 1.05rem; line-height: 1.55; margin: 0 0 1.2rem 0; }}
.ramos-label {{ color: var(--ramos-gray); font-size: 0.85rem; font-weight: 600; margin: 0 0 0.4rem 0; }}

/* 입력 예시 버튼: 흰 바탕 + 파랑 테두리, 올리면 옅은 파랑 */
.st-key-examples button {{
  background: #fff; color: var(--ramos-blue); border: 1px solid var(--ramos-blue);
  text-align: left; justify-content: flex-start; font-weight: 500;
}}
.st-key-examples button > div, .st-key-examples button [data-testid="stMarkdownContainer"], .st-key-examples button p {{
  width: 100%; text-align: left !important; justify-content: flex-start !important;
}}
.st-key-examples button:hover {{ background: var(--ramos-tint); border-color: var(--ramos-blue); color: var(--ramos-blue); }}

/* 대화: 사용자 말풍선은 옅은 파랑, 답변은 흰 바탕 */
[data-testid="stChatMessage"] {{ border-radius: 0.9rem; padding: 0.9rem 1.1rem; margin-bottom: 0.4rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background: var(--ramos-tint); }}

/* 사이드바: 구역 제목은 작은 회색 대문자 느낌, 구역 사이 간격 */
[data-testid="stSidebar"] .ramos-section {{
  color: var(--ramos-gray); font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; margin: 1.1rem 0 0.4rem 0;
}}
[data-testid="stSidebar"] .ramos-brand {{ color: var(--ramos-blue); font-size: 1.25rem; font-weight: 700; margin: 0 0 0.6rem 0; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ background: #fff; border-radius: 0.6rem; }}
</style>
"""

