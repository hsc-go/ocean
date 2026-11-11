"""해양 식물플랑크톤 분류 실험실 - 홈 화면 (Streamlit)

실행: streamlit run main.py
데이터: main.py 와 같은 폴더에 있는 새만금 해양수질자동측정망 CSV 파일
"""

import io
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "해양 식물플랑크톤 분류 실험실"
APP_ICON = "🌊"

DATA_FILENAME = "해양환경공단_해양수질자동측정망_새만금_raw_file.csv"
MEASURE_COLS = ["전기전도도", "수온", "수소이온농도", "용존산소량", "탁도", "염분", "클로로필"]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# 디자인: 흰색 배경 + 청록(#0E8C8C) + 남색(#12274D)
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, input, textarea, button {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #FFFFFF;
}
.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}
[data-testid="stHeader"] { height: 0; }

/* 상단 제목 영역 */
.hero {
    display: flex; align-items: center; gap: 22px;
    padding: 6px 0 22px 0;
    border-bottom: 3px solid #0E8C8C;
    margin-bottom: 30px;
}
.hero-icon {
    flex: 0 0 auto;
    width: 76px; height: 76px; border-radius: 20px;
    background: #12274D;
    display: flex; align-items: center; justify-content: center;
    font-size: 42px;
    box-shadow: 6px 6px 0 #0E8C8C;
}
.hero-title {
    font-size: 2.35rem; font-weight: 900; line-height: 1.2;
    color: #12274D; letter-spacing: -0.02em; margin: 0;
}
.hero-sub { color: #4A6072; font-size: 1.02rem; margin-top: 6px; }

/* 섹션 제목 */
.sec-title {
    font-size: 1.4rem; font-weight: 700; color: #12274D;
    margin: 44px 0 14px 0; padding-left: 12px;
    border-left: 6px solid #0E8C8C;
    letter-spacing: -0.01em;
}
.sec-desc { color: #4A6072; font-size: 0.95rem; margin: -6px 0 14px 18px; }

/* 소개 패널 */
.panel {
    background: #F4FAFA; border: 1px solid #CFE6E6; border-radius: 10px;
    padding: 22px 26px; color: #1D2B3A; line-height: 1.75; font-size: 1rem;
}
.panel b { color: #12274D; }
.panel ul { margin: 8px 0 0 0; padding-left: 20px; }
.panel li { margin-bottom: 4px; }

/* 숫자 카드 */
.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.stat {
    background: #FFFFFF; border: 1px solid #D5E3E8; border-top: 5px solid #0E8C8C;
    border-radius: 6px 6px 12px 12px; padding: 18px 20px;
}
.stat.navy { border-top-color: #12274D; }
.stat-label { color: #4A6072; font-size: 0.9rem; font-weight: 500; }
.stat-value { color: #12274D; font-size: 1.85rem; font-weight: 900; margin: 4px 0 2px 0; letter-spacing: -0.02em; }
.stat-value.small { font-size: 1.3rem; padding: 6px 0 4px 0; }
.stat-sub { color: #6B7F8C; font-size: 0.82rem; }

/* 표 */
.table-wrap { overflow-x: auto; border: 1px solid #D5E3E8; border-radius: 10px; }
table.lab-table { border-collapse: collapse; width: 100%; font-size: 0.93rem; }
table.lab-table thead th {
    background: #12274D; color: #FFFFFF; font-weight: 700; text-align: left;
    padding: 11px 14px; white-space: nowrap;
}
table.lab-table tbody td {
    padding: 10px 14px; border-top: 1px solid #E3EDF0; color: #1D2B3A; vertical-align: top;
}
table.lab-table tbody tr:nth-child(even) td { background: #F4FAFA; }
table.lab-table td.key { font-weight: 700; color: #0B6E6E; white-space: nowrap; }
.caption { color: #6B7F8C; font-size: 0.85rem; margin-top: 8px; }

/* 입력 위젯 */
label, [data-testid="stWidgetLabel"] p { color: #12274D !important; font-weight: 700 !important; }
.stTextInput input, .stTextArea textarea {
    background: #FFFFFF !important; color: #1D2B3A !important;
    border: 1px solid #B9D3D8 !important; border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0E8C8C !important; box-shadow: 0 0 0 2px rgba(14,140,140,0.2) !important;
}
.stDownloadButton button {
    background: #0E8C8C; color: #FFFFFF; border: none; border-radius: 8px; font-weight: 700;
}
.stDownloadButton button:hover { background: #12274D; color: #FFFFFF; }

.source-box {
    background: #FFFFFF; border: 1px solid #D5E3E8; border-left: 6px solid #12274D;
    border-radius: 8px; padding: 16px 20px; color: #1D2B3A; line-height: 1.8; font-size: 0.97rem;
}
.source-box b { color: #12274D; }

@media (max-width: 800px) {
    .card-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 1.6rem; }
    .hero-icon { width: 58px; height: 58px; font-size: 32px; }
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data(raw_bytes: bytes):
    """CSV 바이트를 읽어 (원본, 정제본) 데이터프레임을 반환한다."""
    raw = None
    for enc in ("utf-8-sig", "cp949"):  # 공공데이터 한글 CSV는 cp949인 경우가 많음
        try:
            raw = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError("CSV 인코딩을 인식하지 못했습니다. UTF-8 또는 CP949 파일을 사용해 주세요.")

    clean = raw.copy()
    clean["관측일자"] = pd.to_datetime(clean["관측일자"], errors="coerce")
    for col in MEASURE_COLS:
        # '미수신' 같은 문자열은 결측(NaN)으로 처리
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return raw, clean


def find_data_file():
    base = Path(__file__).resolve().parent
    exact = base / DATA_FILENAME
    if exact.exists():
        return exact
    candidates = sorted(base.glob("*새만금*.csv"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def section(title, desc=""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)


def html_table(df, key_first_col=False):
    """데이터프레임을 연구실 스타일 HTML 표로 만든다."""
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for i, v in enumerate(r):
            cls = ' class="key"' if (key_first_col and i == 0) else ""
            cells.append(f"<td{cls}>{escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table class="lab-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def stat_card(label, value, sub="", navy=False, small=False):
    cls = "stat navy" if navy else "stat"
    vcls = "stat-value small" if small else "stat-value"
    return (
        f'<div class="{cls}"><div class="stat-label">{label}</div>'
        f'<div class="{vcls}">{value}</div><div class="stat-sub">{sub}</div></div>'
    )


# ---------------------------------------------------------------------------
# 홈 화면
# ---------------------------------------------------------------------------
def render_home(raw, clean):
    # 맨 위: 아이콘 + 앱 제목
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">{APP_ICON}</div>
            <div>
                <div class="hero-title">{APP_TITLE}</div>
                <div class="hero-sub">해양수질자동측정망 관측값으로 식물플랑크톤 번성 상태를 탐구하는 데이터 실험실</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. 프로젝트 제목과 연구 목적
    section("프로젝트 소개")
    st.markdown(
        """
        <div class="panel">
            <b>프로젝트 제목</b> &nbsp;새만금 해역 수질 자료로 살펴보는 식물플랑크톤 번성 분류<br><br>
            <b>연구 목적</b><br>
            바닷물 속 식물플랑크톤의 양은 눈으로 직접 세기 어렵습니다. 이 프로젝트는 5분마다 자동으로
            기록되는 수질 관측값 중 <b>클로로필</b>을 식물플랑크톤 양의 지표로 삼아, 다음 질문에 답하는 것을 목표로 합니다.
            <ul>
                <li>수온, 염분, 용존산소량 같은 수질 항목은 클로로필 농도와 어떤 관계가 있을까?</li>
                <li>수질 항목만으로 식물플랑크톤이 많은 시기와 적은 시기를 구분(분류)할 수 있을까?</li>
                <li>계절과 하루 중 시간대에 따라 번성 양상은 어떻게 달라질까?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. 식물플랑크톤과 클로로필의 관계
    section("식물플랑크톤과 클로로필의 생물학적 관계")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(
            """
            <div class="panel">
                <b>식물플랑크톤</b>은 물속에 떠다니며 광합성을 하는 아주 작은 생물입니다.
                규조류, 와편모류 등이 있고, 해양 먹이사슬의 가장 아래에서 산소를 만들어 냅니다.<br><br>
                <b>클로로필 a</b>는 이들이 광합성에 쓰는 초록색 색소입니다. 식물플랑크톤이 많을수록
                물속 클로로필 농도도 높아지므로, 클로로필은 <b>식물플랑크톤 양(생물량)을 간접적으로 알려 주는 지표</b>로 널리 쓰입니다.<br><br>
                다만 클로로필은 <b>양</b>만 알려 줄 뿐 <b>어떤 종</b>인지는 구분하지 못합니다.
                그래서 이 실험실에서는 다른 수질 항목을 함께 이용해 번성 상태를 분류해 봅니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        relation = pd.DataFrame(
            [
                ("수온", "적정 범위까지는 증식 속도가 빨라져 클로로필이 오를 수 있음 (종마다 적정 수온이 다름)"),
                ("염분", "담수가 유입되면 낮아지고, 염분 변화에 따라 우세한 종류가 바뀔 수 있음"),
                ("용존산소량", "낮 동안 광합성으로 증가, 과도한 번성 뒤 분해 과정에서 감소할 수 있음"),
                ("수소이온농도", "광합성이 이산화탄소를 소비해 pH가 올라갈 수 있음"),
                ("탁도", "식물플랑크톤 자체가 입자라 높아질 수 있으나, 너무 탁하면 빛을 막아 증식을 억제할 수 있음"),
            ],
            columns=["수질 항목", "클로로필과의 예상 관계"],
        )
        st.markdown(html_table(relation, key_first_col=True), unsafe_allow_html=True)
        st.markdown(
            '<div class="caption">위 관계는 일반적인 생물학적 가설입니다. 실제 새만금 데이터에서 나타나는지는 이후 분석에서 확인합니다.</div>',
            unsafe_allow_html=True,
        )

    # 3. 숫자 카드
    section("데이터 한눈에 보기")
    n_rows = len(clean)
    start, end = clean["관측일자"].min(), clean["관측일자"].max()
    span_days = (end - start).days + 1
    interval = clean["관측일자"].diff().dropna().dt.total_seconds().median() / 60
    n_missing = int(clean[MEASURE_COLS].isna().sum().sum())
    pct_missing = n_missing / (n_rows * len(MEASURE_COLS)) * 100
    station = ", ".join(sorted(clean["정점명"].astype(str).unique()))
    codes = ", ".join(str(c) for c in sorted(clean["정점코드"].unique()))

    cards = [
        stat_card("전체 데이터 수", f"{n_rows:,}행", "측정 1회가 1행"),
        stat_card("관측 기간", f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}", f"총 {span_days}일", navy=True, small=True),
        stat_card("관측 정점", station, f"정점코드 {codes}"),
        stat_card("전체 열 수", f"{raw.shape[1]}개", f"수질 측정 항목 {len(MEASURE_COLS)}개", navy=True),
        stat_card("관측 간격", f"약 {interval:.0f}분", "중앙값 기준"),
        stat_card("미수신·결측 값", f"{n_missing:,}개", f"전체 측정값의 {pct_missing:.2f}%", navy=True),
    ]
    st.markdown(f'<div class="card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    # 4. 열 이름 설명 표
    section("열 이름과 뜻", "단위는 이 항목들에 일반적으로 쓰이는 단위이며, 원본 파일에는 단위가 적혀 있지 않습니다.")
    columns_info = pd.DataFrame(
        [
            ("정점명", "Station name", "관측 장소의 이름", "-", "식별"),
            ("정점코드", "Station code", "관측 장소를 구분하는 번호", "-", "식별"),
            ("관측일자", "Observation time", "측정한 날짜와 시각 (5분 간격)", "연-월-일 시:분", "시간"),
            ("전기전도도", "Conductivity", "물이 전기를 통하는 정도. 녹아 있는 이온이 많을수록(염분이 높을수록) 커짐", "mS/cm", "환경 변수"),
            ("수온", "Water temperature", "바닷물의 온도", "°C", "환경 변수"),
            ("수소이온농도", "pH", "물이 산성인지 염기성인지 나타내는 값", "pH", "환경 변수"),
            ("용존산소량", "Dissolved oxygen (DO)", "물에 녹아 있는 산소의 양", "mg/L", "환경 변수"),
            ("탁도", "Turbidity", "물이 흐린 정도", "NTU", "환경 변수"),
            ("염분", "Salinity", "물 1kg에 녹아 있는 염류의 양", "psu", "환경 변수"),
            ("클로로필", "Chlorophyll-a", "식물플랑크톤의 광합성 색소 농도. 식물플랑크톤 양의 지표", "µg/L", "핵심 지표"),
        ],
        columns=["열 이름", "영문 표기", "우리말 뜻", "일반적 단위", "역할"],
    )
    st.markdown(html_table(columns_info, key_first_col=True), unsafe_allow_html=True)

    # 5. 원본 데이터 처음 5행
    section("원본 데이터 미리보기", "파일에 저장된 그대로의 처음 5행입니다.")
    st.markdown(html_table(raw.head(5)), unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">‘미수신’ 같은 문자열이 들어 있는 열은 분석 단계에서 숫자로 바꾸며 결측 처리합니다.</div>',
        unsafe_allow_html=True,
    )

    # 6. 데이터 출처 + 탐구 메모
    section("데이터 출처와 탐구 메모")
    st.markdown(
        f"""
        <div class="source-box">
            <b>제공 기관</b> &nbsp;해양환경공단<br>
            <b>자료 이름</b> &nbsp;해양수질자동측정망 새만금<br>
            <b>자료 기준일</b> &nbsp;2024-12-31<br>
            <b>사용 파일</b> &nbsp;{escape(DATA_FILENAME)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        researcher = st.text_input("연구자 이름", key="researcher", placeholder="예: 홍길동")
    with c2:
        source_link = st.text_input("자료를 받은 주소(URL)", key="source_link", placeholder="https://")
    memo = st.text_area(
        "탐구 메모",
        key="memo",
        height=200,
        placeholder="관찰한 점, 세워 본 가설, 더 알아보고 싶은 질문을 자유롭게 적어 보세요.",
    )

    memo_text = (
        f"[{APP_TITLE}] 탐구 메모\n"
        f"작성 시각: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"연구자: {researcher}\n"
        f"자료 주소: {source_link}\n\n"
        f"{memo}\n"
    )
    st.download_button(
        "메모를 텍스트 파일로 저장",
        data=memo_text.encode("utf-8"),
        file_name="탐구메모.txt",
        mime="text/plain",
    )
    st.caption("메모는 이 브라우저 세션에만 남습니다. 창을 닫기 전에 저장해 주세요.")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    path = find_data_file()
    if path is not None:
        raw_bytes = path.read_bytes()
    else:
        st.markdown(
            f'<div class="hero"><div class="hero-icon">{APP_ICON}</div>'
            f'<div><div class="hero-title">{APP_TITLE}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.warning(
            f"main.py와 같은 폴더에서 '{DATA_FILENAME}' 파일을 찾지 못했습니다. "
            "파일을 그 폴더에 넣거나 아래에서 직접 올려 주세요."
        )
        uploaded = st.file_uploader("새만금 해양수질 CSV 파일", type=["csv"])
        if uploaded is None:
            st.stop()
        raw_bytes = uploaded.getvalue()

    try:
        raw, clean = load_data(raw_bytes)
    except Exception as e:
        st.error(f"데이터를 읽는 중 문제가 생겼습니다: {e}")
        st.stop()

    render_home(raw, clean)


main()
