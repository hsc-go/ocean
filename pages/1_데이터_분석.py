"""데이터 분석 페이지 - 수온, 염분, pH, 탁도와 클로로필 농도의 관계

main.py 와 같은 폴더의 pages/ 안에 두면 Streamlit이 자동으로 메뉴에 추가합니다.
"""

from pathlib import Path
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

APP_TITLE = "해양 식물플랑크톤 분류 실험실"
PAGE_TITLE = "데이터 분석"

DATA_URL = "https://raw.githubusercontent.com/hsc-go/data/main/ocean_water_quality.csv"
LOCAL_FILENAME = "ocean_water_quality.csv"  # 인터넷 연결이 안 될 때 쓰는 예비 파일(선택)

TEAL = "#0E8C8C"
NAVY = "#12274D"

# 화면에 보이는 이름 -> CSV의 실제 열 이름
COLS = {
    "수온": "수온",
    "염분": "염분",
    "pH": "수소이온농도",
    "탁도": "탁도",
    "클로로필 농도": "클로로필",
}
FACTORS = ["수온", "염분", "pH", "탁도"]  # 사용자가 고를 수 있는 환경 요인
TARGET = "클로로필 농도"
ALL5 = FACTORS + [TARGET]

# 측정기 오류로 의심되는 값을 거르는 기준 (이 연구에서 정한 기준)
RULES = [
    ("수온", "0°C 초과 ~ 40°C 이하", lambda s: (s > 0) & (s <= 40)),
    ("염분", "0 초과 ~ 40 이하", lambda s: (s > 0) & (s <= 40)),
    ("pH", "6.0 ~ 10.0", lambda s: s.between(6, 10)),
    ("탁도", "0 이상", lambda s: s >= 0),
    ("클로로필 농도", "0 이상", lambda s: s >= 0),
]

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 디자인: 흰색 배경 + 청록(#0E8C8C) + 남색(#12274D)
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, input, textarea, button {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #FFFFFF; }
.block-container { max-width: 1180px; padding-top: 4rem; padding-bottom: 4rem; }

.hero {
    display: flex; align-items: center; gap: 20px;
    padding: 4px 0 20px 0; border-bottom: 3px solid #0E8C8C; margin-bottom: 26px;
}
.hero-icon {
    flex: 0 0 auto; width: 64px; height: 64px; border-radius: 18px; background: #12274D;
    display: flex; align-items: center; justify-content: center; font-size: 34px;
    box-shadow: 5px 5px 0 #0E8C8C;
}
.hero-title { font-size: 2rem; font-weight: 900; line-height: 1.2; color: #12274D; letter-spacing: -0.02em; }
.hero-sub { color: #4A6072; font-size: 1rem; margin-top: 4px; }

.sec-title {
    font-size: 1.35rem; font-weight: 700; color: #12274D; margin: 44px 0 12px 0;
    padding-left: 12px; border-left: 6px solid #0E8C8C; letter-spacing: -0.01em;
}
.sec-desc { color: #4A6072; font-size: 0.95rem; margin: 0 0 14px 18px; line-height: 1.6; }

.panel {
    background: #F4FAFA; border: 1px solid #CFE6E6; border-radius: 10px;
    padding: 18px 24px; color: #1D2B3A; line-height: 1.75; font-size: 0.98rem;
}
.panel b { color: #12274D; }
.notice {
    background: #FFFFFF; border: 2px solid #12274D; border-left: 8px solid #12274D;
    border-radius: 8px; padding: 14px 20px; color: #1D2B3A; line-height: 1.75; font-size: 0.97rem;
    margin: 6px 0 16px 0;
}
.notice b { color: #12274D; }

.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 10px 0 14px 0; }
.card-grid.four { grid-template-columns: repeat(4, 1fr); }
.stat {
    background: #FFFFFF; border: 1px solid #D5E3E8; border-top: 5px solid #0E8C8C;
    border-radius: 6px 6px 12px 12px; padding: 14px 18px;
}
.stat.navy { border-top-color: #12274D; }
.stat-label { color: #4A6072; font-size: 0.88rem; font-weight: 500; }
.stat-value { color: #12274D; font-size: 1.6rem; font-weight: 900; margin: 2px 0; letter-spacing: -0.02em; }
.stat-sub { color: #6B7F8C; font-size: 0.8rem; }

.table-wrap { overflow-x: auto; border: 1px solid #D5E3E8; border-radius: 10px; }
table.lab-table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
table.lab-table thead th {
    background: #12274D; color: #FFFFFF; font-weight: 700; text-align: left;
    padding: 10px 14px; white-space: nowrap;
}
table.lab-table tbody td { padding: 9px 14px; border-top: 1px solid #E3EDF0; color: #1D2B3A; vertical-align: top; }
table.lab-table tbody tr:nth-child(even) td { background: #F4FAFA; }
table.lab-table td.key { font-weight: 700; color: #0B6E6E; white-space: nowrap; }
.caption { color: #6B7F8C; font-size: 0.85rem; margin-top: 8px; line-height: 1.6; }

label, [data-testid="stWidgetLabel"] p { color: #12274D !important; font-weight: 700 !important; }

@media (max-width: 800px) {
    .card-grid, .card-grid.four { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 1.5rem; }
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 데이터 불러오기와 정리
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터를 불러오는 중입니다...")
def load_raw():
    """원격 CSV를 읽어 원본 데이터프레임을 반환한다."""
    try:
        return pd.read_csv(DATA_URL, encoding="utf-8-sig")
    except Exception as remote_error:
        local = Path(__file__).resolve().parent.parent / LOCAL_FILENAME
        if not local.exists():
            raise RuntimeError(
                f"데이터 주소에서 파일을 읽지 못했습니다. 주소와 인터넷 연결을 확인해 주세요. ({remote_error})"
            )
        for enc in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(local, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError("예비 파일의 인코딩을 인식하지 못했습니다. UTF-8 또는 CP949로 저장해 주세요.")


@st.cache_data(show_spinner=False)
def prepare(raw):
    """분석에 쓸 열을 숫자로 바꾸고, 오류로 의심되는 값을 거른다."""
    work = pd.DataFrame({"관측일자": pd.to_datetime(raw["관측일자"], errors="coerce")})
    non_numeric = {}
    for label, col in COLS.items():
        converted = pd.to_numeric(raw[col], errors="coerce")
        # 원래 값이 있었는데 숫자로 못 바꾼 경우(예: '미수신')
        non_numeric[label] = int((raw[col].notna() & converted.isna()).sum())
        work[label] = converted

    n_raw = len(work)
    valid = work["관측일자"].notna()
    for label in ALL5:
        valid &= work[label].notna()
    n_missing = int(n_raw - valid.sum())

    rule_rows = []
    for label, desc, ok in RULES:
        series = work[label]
        bad = int((series.notna() & ~ok(series)).sum())  # 값이 있는데 정상 범위 밖인 행
        rule_rows.append((label, desc, bad))
        valid &= ok(series)

    df = work[valid].copy().reset_index(drop=True)
    df["관측일시"] = df["관측일자"].dt.strftime("%Y-%m-%d %H:%M")
    df["연월"] = df["관측일자"].dt.to_period("M").astype(str)
    report = {
        "n_raw": n_raw,
        "n_used": len(df),
        "n_missing": n_missing,
        "non_numeric": non_numeric,
        "rules": rule_rows,
    }
    return df, report


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def section(title, desc=""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)


def html_table(df, key_first_col=False):
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


def stat_card(label, value, sub="", navy=False):
    cls = "stat navy" if navy else "stat"
    return (
        f'<div class="{cls}"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div><div class="stat-sub">{sub}</div></div>'
    )


def card_grid(cards, four=False):
    cls = "card-grid four" if four else "card-grid"
    st.markdown(f'<div class="{cls}">{"".join(cards)}</div>', unsafe_allow_html=True)


def style_fig(fig, height=460):
    """모든 plotly 그래프에 같은 연구실 스타일을 입힌다."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Noto Sans KR, Malgun Gothic, sans-serif", color=NAVY, size=13),
        margin=dict(l=60, r=30, t=40, b=55),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=TEAL, font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#E3EDF0", linecolor="#B9D3D8", zeroline=False)
    fig.update_yaxes(gridcolor="#E3EDF0", linecolor="#B9D3D8", zeroline=False)
    return fig


def trim_range(series, q=0.01):
    """상·하위 1% 극단값을 뺀 축 범위(약간의 여백 포함)."""
    lo, hi = series.quantile(q), series.quantile(1 - q)
    pad = (hi - lo) * 0.05
    return [lo - pad, hi + pad]


def strength(r):
    a = abs(r)
    if a < 0.2:
        return "거의 없음"
    if a < 0.4:
        return "약함"
    if a < 0.6:
        return "보통"
    return "강함"


def direction(r):
    return "함께 증가하는 경향 (+)" if r > 0 else "반대로 변하는 경향 (−)"


# ---------------------------------------------------------------------------
# 페이지 본문
# ---------------------------------------------------------------------------
def render_page(raw, df, report):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">📊</div>
            <div>
                <div class="hero-title">{PAGE_TITLE}</div>
                <div class="hero-sub">수온·염분·pH·탁도와 클로로필 농도 사이의 관계를 그래프로 살펴봅니다</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="panel">
            <b>이 페이지에서 하는 일</b><br>
            클로로필 농도는 식물플랑크톤 양을 간접적으로 알려 주는 지표입니다.
            환경 요인 하나를 골라 클로로필과 어떤 관계가 있는지 보고, 월별 변화, 고농도와 저농도 비교, 상관계수까지 차례로 확인합니다.<br>
            분석에는 원본 <b>{report['n_raw']:,}행</b> 중 모든 값이 정상 범위에 있는 <b>{report['n_used']:,}행</b>을 사용했습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 열 이름과 데이터 유형 확인 + 정리 방법
    with st.expander("사용한 열과 데이터 정리 방법 보기"):
        dtype_rows = []
        for label, col in COLS.items():
            dtype = str(raw[col].dtype)
            n_bad = report["non_numeric"][label]
            if n_bad:
                kind = f"문자열 (숫자가 아닌 값 {n_bad:,}개 포함, 예: ‘미수신’)"
                how = "숫자로 바꾸고, 바꿀 수 없는 값은 결측으로 처리"
            else:
                kind = "숫자"
                how = "그대로 사용"
            dtype_rows.append((label, col, dtype, kind, how))
        dtype_rows.append(("관측 날짜", "관측일자", str(raw["관측일자"].dtype), "문자열 (날짜·시각)", "날짜 형식으로 바꿔 월별 집계에 사용"))
        st.markdown("**분석에 사용한 열**")
        st.markdown(
            html_table(
                pd.DataFrame(dtype_rows, columns=["화면 이름", "CSV 열 이름", "읽어 온 데이터 유형", "내용", "처리 방법"]),
                key_first_col=True,
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**오류로 의심되는 값 걸러내기**")
        rule_df = pd.DataFrame(
            [(label, desc, f"{bad:,}행") for label, desc, bad in report["rules"]],
            columns=["항목", "정상 범위로 본 기준", "기준 밖 행 수"],
        )
        st.markdown(html_table(rule_df, key_first_col=True), unsafe_allow_html=True)
        st.markdown(
            f'<div class="caption">비어 있거나 숫자가 아니어서 쓸 수 없는 행: {report["n_missing"]:,}행. '
            "위 정상 범위는 센서 오류로 의심되는 값(예: 0 또는 음수, 지나치게 큰 값)을 거르기 위해 이 연구에서 정한 기준이며, "
            "여러 항목이 겹치는 행은 한 번만 제외됩니다. 클로로필의 매우 높은 값은 실제 번성일 수 있어 지우지 않았습니다.</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ 1. 환경 요인 선택
    section("분석할 환경 요인 고르기", "고른 요인은 아래 산점도의 x축이 되고, y축은 항상 클로로필 농도입니다.")
    factor = st.radio("환경 요인", FACTORS, horizontal=True, key="factor")

    # ------------------------------------------------------------------ 2. 산점도
    section(f"{factor}와(과) 클로로필 농도의 산점도", "점 하나가 실제 관측 한 번입니다. 점에 마우스를 올리면 그 시각의 측정값을 모두 볼 수 있습니다.")
    zoom = st.checkbox("극단값을 빼고 확대해서 보기 (상·하위 1% 범위만 화면에 표시)", key="zoom")

    fig = px.scatter(
        df,
        x=factor,
        y=TARGET,
        custom_data=["관측일시", "수온", "염분", "pH", "탁도", "클로로필 농도"],
        opacity=0.45,
        color_discrete_sequence=[TEAL],
    )
    fig.update_traces(
        marker=dict(size=6),
        name="관측값",
        showlegend=True,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "수온: %{customdata[1]:.2f}<br>"
            "염분: %{customdata[2]:.2f}<br>"
            "pH: %{customdata[3]:.2f}<br>"
            "탁도: %{customdata[4]:.0f}<br>"
            "클로로필 농도: %{customdata[5]:.2f}<extra></extra>"
        ),
    )

    model = LinearRegression().fit(df[[factor]], df[TARGET])
    xs = pd.DataFrame({factor: [df[factor].min(), df[factor].max()]})
    fig.add_trace(
        go.Scatter(
            x=xs[factor],
            y=model.predict(xs),
            mode="lines",
            name="직선 추세선",
            line=dict(color=NAVY, width=3),
            hoverinfo="skip",
        )
    )
    style_fig(fig, 500)
    fig.update_xaxes(title=factor)
    fig.update_yaxes(title=TARGET)
    if zoom:
        fig.update_xaxes(range=trim_range(df[factor]))
        fig.update_yaxes(range=trim_range(df[TARGET]))
    st.plotly_chart(fig)

    r = df[factor].corr(df[TARGET])
    card_grid(
        [
            stat_card("관측 점 수", f"{len(df):,}개", "그래프에 그려진 관측"),
            stat_card("상관계수 r", f"{r:+.2f}", f"관계의 세기: {strength(r)}", navy=True),
            stat_card("추세선 기울기", f"{model.coef_[0]:+.2f}", f"{factor}이(가) 1 늘 때 클로로필 평균 변화"),
        ]
    )
    st.markdown(
        f'<div class="caption">{factor}와(과) 클로로필은 {direction(r)}이며 관계의 세기는 “{strength(r)}”입니다. '
        "상관은 두 값이 함께 움직이는 정도일 뿐, 하나가 다른 하나의 원인이라는 뜻은 아닙니다.</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 3. 월별 클로로필 평균
    section("월별 클로로필 농도 변화", "관측 날짜를 월별로 묶어 클로로필 농도의 평균을 계산했습니다.")
    monthly = df.groupby("연월")[TARGET].agg(평균="mean", 관측수="count").reset_index()
    fig = go.Figure(
        go.Scatter(
            x=monthly["연월"],
            y=monthly["평균"],
            mode="lines+markers",
            line=dict(color=NAVY, width=3),
            marker=dict(color=TEAL, size=10, line=dict(color=NAVY, width=1.5)),
            customdata=monthly["관측수"],
            hovertemplate="%{x}<br>평균 클로로필 농도: %{y:.2f}<br>관측 수: %{customdata:,}개<extra></extra>",
            name="월 평균",
        )
    )
    style_fig(fig, 420)
    fig.update_xaxes(title="관측 월", type="category")
    fig.update_yaxes(title="월 평균 클로로필 농도")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    few = monthly[monthly["관측수"] < monthly["관측수"].median() * 0.5]
    note = "평균은 매우 높은 값 몇 개에도 크게 달라질 수 있습니다."
    if len(few):
        names = ", ".join(f"{m}({n:,}개)" for m, n in zip(few["연월"], few["관측수"]))
        note = f"관측 수가 유난히 적은 달은 평균이 불안정할 수 있습니다: {names}. " + note
    st.markdown(f'<div class="caption">{note}</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------ 4. 고농도와 저농도 비교
    section("고농도와 저농도 비교", "클로로필 농도가 높은 시기와 낮은 시기에 환경 요인이 어떻게 다른지 비교합니다.")

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    threshold = train[TARGET].quantile(0.75)
    df = df.assign(집단=df[TARGET].ge(threshold).map({True: "고농도", False: "저농도"}))
    n_high = int((df["집단"] == "고농도").sum())
    n_low = int((df["집단"] == "저농도").sum())

    st.markdown(
        f"""
        <div class="notice">
            <b>고농도 기준은 연구용 상대 기준입니다.</b><br>
            훈련 데이터({len(train):,}행)의 클로로필 농도 <b>75번째 백분위수 = {threshold:.2f}</b>를 기준으로,
            이 값 이상이면 <b>고농도</b>, 미만이면 <b>저농도</b>로 나눴습니다.
            이 데이터 안에서 상대적으로 높은 값을 구분하려는 연구용 기준이며, 적조 판정 같은 공식·절대 기준이 아닙니다.
            기준값은 훈련 데이터만으로 정했고, 아래 그림에는 전체 분석 데이터({len(df):,}행)를 이 기준으로 나눠 그렸습니다.
            (훈련 {len(train):,}행 / 테스트 {len(test):,}행, 무작위 분할·시드 42)
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_grid(
        [
            stat_card("고농도 기준값", f"{threshold:.2f}", "훈련 데이터 75번째 백분위수"),
            stat_card("고농도 관측 수", f"{n_high:,}개", f"전체의 {n_high / len(df) * 100:.1f}%", navy=True),
            stat_card("저농도 관측 수", f"{n_low:,}개", f"전체의 {n_low / len(df) * 100:.1f}%"),
        ]
    )

    show_points = st.checkbox("상자 밖의 값(점)도 함께 표시", key="show_points")
    fig = make_subplots(rows=1, cols=4, subplot_titles=FACTORS, horizontal_spacing=0.07)
    for i, label in enumerate(FACTORS, start=1):
        for group, color in (("저농도", NAVY), ("고농도", TEAL)):
            sub = df[df["집단"] == group]
            fig.add_trace(
                go.Box(
                    y=sub[label],
                    name=group,
                    marker_color=color,
                    line=dict(color=color),
                    fillcolor="rgba(14,140,140,0.25)" if group == "고농도" else "rgba(18,39,77,0.20)",
                    boxmean=True,
                    boxpoints="outliers" if show_points else False,
                    legendgroup=group,
                    showlegend=(i == 1),
                ),
                row=1,
                col=i,
            )
    style_fig(fig, 460)
    fig.update_layout(margin=dict(l=50, r=20, t=70, b=40))
    for ann in fig.layout.annotations:  # 소제목 스타일
        ann.font = dict(size=15, color=NAVY)
    st.plotly_chart(fig)
    st.markdown(
        '<div class="caption">상자는 중간 50% 범위, 상자 속 가로선은 중앙값, 점선은 평균입니다. '
        "두 집단의 상자가 많이 겹치면 그 요인만으로는 고농도와 저농도를 구분하기 어렵다는 뜻입니다.</div>",
        unsafe_allow_html=True,
    )

    summary = (
        df.groupby("집단")
        .agg(
            관측수=(TARGET, "size"),
            클로로필=(TARGET, "median"),
            수온=("수온", "median"),
            염분=("염분", "median"),
            pH=("pH", "median"),
            탁도=("탁도", "median"),
        )
        .reindex(["저농도", "고농도"])
        .reset_index()
    )
    table = pd.DataFrame(
        {
            "집단": summary["집단"],
            "관측 수": summary["관측수"].map("{:,}".format),
            "클로로필 중앙값": summary["클로로필"].map("{:.2f}".format),
            "수온 중앙값": summary["수온"].map("{:.2f}".format),
            "염분 중앙값": summary["염분"].map("{:.2f}".format),
            "pH 중앙값": summary["pH"].map("{:.2f}".format),
            "탁도 중앙값": summary["탁도"].map("{:.1f}".format),
        }
    )
    st.markdown(html_table(table, key_first_col=True), unsafe_allow_html=True)

    # ------------------------------------------------------------------ 5. 상관관계 히트맵
    section("상관관계 분석", "수온, 염분, pH, 탁도, 클로로필 농도 사이의 상관계수입니다. +1에 가까울수록 함께 증가하고, −1에 가까울수록 반대로 변합니다.")
    method_label = st.radio(
        "상관계수 계산 방식",
        ["피어슨 (직선 관계)", "스피어먼 (순위 기준, 극단값에 덜 민감)"],
        horizontal=True,
        key="corr_method",
    )
    method = "pearson" if method_label.startswith("피어슨") else "spearman"
    corr = df[ALL5].corr(method=method)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        aspect="auto",
        color_continuous_scale=[[0, NAVY], [0.5, "#FFFFFF"], [1, TEAL]],
    )
    style_fig(fig, 480)
    fig.update_layout(coloraxis_colorbar=dict(title="상관계수"), margin=dict(l=80, r=30, t=30, b=60))
    fig.update_traces(hovertemplate="%{y} - %{x}<br>상관계수: %{z:.3f}<extra></extra>")
    fig.update_xaxes(side="bottom", showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)

    with_chl = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    top = with_chl.index[0]
    rank_df = pd.DataFrame(
        {
            "환경 요인": with_chl.index,
            "클로로필과의 상관계수": [f"{v:+.2f}" for v in with_chl.values],
            "관계의 세기": [strength(v) for v in with_chl.values],
            "방향": [direction(v) for v in with_chl.values],
        }
    )
    st.markdown(html_table(rank_df, key_first_col=True), unsafe_allow_html=True)
    st.markdown(
        f'<div class="caption">이 데이터에서 클로로필과 가장 뚜렷하게 함께 움직인 요인은 <b>{top}</b>'
        f"(상관계수 {with_chl[top]:+.2f})입니다. 상관계수는 관계의 세기만 알려 줄 뿐 원인을 증명하지는 않으며, "
        "계절처럼 눈에 보이지 않는 다른 요인이 함께 작용했을 수 있습니다.</div>",
        unsafe_allow_html=True,
    )


def main():
    try:
        raw = load_raw()
        missing = [c for c in ["관측일자", *COLS.values()] if c not in raw.columns]
        if missing:
            raise RuntimeError(f"CSV에서 필요한 열을 찾지 못했습니다: {', '.join(missing)} (현재 열: {', '.join(raw.columns)})")
        df, report = prepare(raw)
        if df.empty:
            raise RuntimeError("정리 후 남은 데이터가 없습니다. 원본 파일의 값을 확인해 주세요.")
    except Exception as e:
        st.error(f"데이터를 준비하지 못했습니다: {e}")
        if st.button("다시 시도"):
            load_raw.clear()
            st.rerun()
        st.stop()

    render_page(raw, df, report)


main()
