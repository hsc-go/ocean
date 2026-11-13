"""분류 실험실 페이지 - 해양 환경 조건으로 클로로필 고농도 상태 분류

main.py 와 같은 폴더의 pages/ 안에 두면 Streamlit이 자동으로 메뉴에 추가합니다.
모델: scikit-learn LogisticRegression (입력 4개: 수온, 염분, pH, 탁도)
"""

import math
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

APP_TITLE = "해양 식물플랑크톤 분류 실험실"
PAGE_TITLE = "분류 실험실"

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
FEATURES = ["수온", "염분", "pH", "탁도"]  # 입력변수 (클로로필 농도는 넣지 않음)
TARGET = "클로로필 농도"
TRAIN_RATIO = 0.8
HIGH_PERCENTILE = 0.75

# 입력칸 설정: 이름 -> (간격, 표시 형식)
INPUT_STEP = {"수온": (0.1, "%.1f"), "염분": (0.1, "%.1f"), "pH": (0.01, "%.2f"), "탁도": (1.0, "%.0f")}

# 측정기 오류로 의심되는 값을 거르는 기준 (데이터 분석 페이지와 같은 기준)
RULES = [
    ("수온", lambda s: (s > 0) & (s <= 40)),
    ("염분", lambda s: (s > 0) & (s <= 40)),
    ("pH", lambda s: s.between(6, 10)),
    ("탁도", lambda s: s >= 0),
    ("클로로필 농도", lambda s: s >= 0),
]

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}",
    page_icon="🧪",
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
.panel ul { margin: 6px 0 0 0; padding-left: 20px; }
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
.stat-sub { color: #6B7F8C; font-size: 0.8rem; line-height: 1.5; }

.verdict {
    border-radius: 12px; padding: 22px 26px; color: #FFFFFF; margin: 8px 0 14px 0;
}
.verdict.high { background: #0E8C8C; }
.verdict.low { background: #12274D; }
.verdict-label { font-size: 0.95rem; opacity: 0.9; }
.verdict-value { font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em; margin-top: 2px; }

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
.stNumberInput input {
    background: #FFFFFF !important; color: #1D2B3A !important;
    border: 1px solid #B9D3D8 !important; border-radius: 8px !important;
}
.stButton button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: #0E8C8C; border: none; color: #FFFFFF; font-weight: 700;
    border-radius: 8px; padding: 0.6rem 2.2rem;
}
.stButton button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: #12274D; color: #FFFFFF;
}

@media (max-width: 800px) {
    .card-grid, .card-grid.four { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 1.5rem; }
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. 데이터 전처리
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
    """수온·염분·pH·탁도·클로로필 농도를 숫자로 바꾸고, 오류로 의심되는 값을 거른다."""
    work = pd.DataFrame({"관측일자": pd.to_datetime(raw["관측일자"], errors="coerce")})
    for label, col in COLS.items():
        work[label] = pd.to_numeric(raw[col], errors="coerce")  # '미수신' 같은 문자열은 결측

    n_raw = len(work)
    valid = work["관측일자"].notna()
    for label in FEATURES + [TARGET]:
        valid &= work[label].notna()
    for label, ok in RULES:
        valid &= ok(work[label])

    df = work[valid].copy().reset_index(drop=True)
    return df, {"n_raw": n_raw, "n_used": len(df)}


# ---------------------------------------------------------------------------
# 2. 학습/평가 분리와 모델 학습
# ---------------------------------------------------------------------------
def split_by_time(df, ratio=TRAIN_RATIO):
    """관측 날짜순으로 앞 80%를 훈련, 뒤 20%를 평가 데이터로 나눈다.

    같은 관측 시각이 두 데이터에 나뉘어 들어가지 않도록, 행이 아니라
    '서로 다른 관측 시각' 기준으로 80% 지점을 자른다.
    """
    times = df["관측일자"].drop_duplicates().sort_values().reset_index(drop=True)
    cut = int(len(times) * ratio)
    cutoff = times.iloc[cut - 1]
    train = df[df["관측일자"] <= cutoff].copy()
    test = df[df["관측일자"] > cutoff].copy()
    # 같은 관측 시각이 양쪽에 동시에 있지 않은지 확인
    if set(train["관측일자"]) & set(test["관측일자"]):
        raise RuntimeError("훈련 데이터와 평가 데이터에 같은 관측 시각이 겹칩니다.")
    return train, test


@st.cache_data(show_spinner="모델을 학습하는 중입니다...")
def train_model(df):
    train, test = split_by_time(df)

    # 고농도 기준값은 훈련 데이터에서만 계산하고, 평가 데이터에도 같은 값을 쓴다.
    threshold = float(train[TARGET].quantile(HIGH_PERCENTILE))
    y_train = (train[TARGET] >= threshold).astype(int)
    y_test = (test[TARGET] >= threshold).astype(int)
    if y_train.nunique() < 2:
        raise RuntimeError("훈련 데이터에 고농도와 저농도가 모두 있어야 학습할 수 있습니다.")

    X_train, X_test = train[FEATURES], test[FEATURES]  # 클로로필 농도는 입력변수에 넣지 않는다.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "model": model,
        "threshold": threshold,
        "train": train,
        "test": test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": {
            "정확도": accuracy_score(y_test, pred),
            "정밀도": precision_score(y_test, pred, zero_division=0),
            "재현율": recall_score(y_test, pred, zero_division=0),
            "F1 점수": f1_score(y_test, pred, zero_division=0),
        },
        "baseline_acc": float((y_test == 0).mean()),  # 평가 데이터를 전부 '저농도'로 답했을 때의 정확도
        "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
        "coef": model.named_steps["logisticregression"].coef_[0],
    }


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


def style_fig(fig, height=400):
    """모든 plotly 그래프에 같은 연구실 스타일을 입힌다."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Noto Sans KR, Malgun Gothic, sans-serif", color=NAVY, size=13),
        margin=dict(l=60, r=30, t=30, b=55),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=TEAL, font_size=13),
    )
    fig.update_xaxes(gridcolor="#E3EDF0", linecolor="#B9D3D8", zeroline=False)
    fig.update_yaxes(gridcolor="#E3EDF0", linecolor="#B9D3D8", zeroline=False)
    return fig


def fmt_period(frame):
    return f"{frame['관측일자'].min():%Y-%m-%d} ~ {frame['관측일자'].max():%Y-%m-%d}"


def floor_to(x, step):
    return math.floor(x / step) * step


def ceil_to(x, step):
    return math.ceil(x / step) * step


# ---------------------------------------------------------------------------
# 페이지 본문
# ---------------------------------------------------------------------------
def render_page(df, report, result):
    model = result["model"]
    train, test = result["train"], result["test"]
    threshold = result["threshold"]
    y_train, y_test = result["y_train"], result["y_test"]

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">🧪</div>
            <div>
                <div class="hero-title">{PAGE_TITLE}</div>
                <div class="hero-sub">해양 환경 조건을 입력하면 클로로필 고농도 상태를 분류합니다</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 학습 과정 요약
    section("모델은 이렇게 학습했습니다")
    st.markdown(
        f"""
        <div class="panel">
            <b>입력변수</b> 수온, 염분, pH, 탁도 (4개) &nbsp;·&nbsp; <b>정답 레이블</b> 클로로필 농도가 기준값 이상이면 1(고농도), 미만이면 0(저농도)<br>
            <b>클로로필 농도는 입력변수에 넣지 않았습니다.</b> 정답을 만드는 데만 쓰고, 모델은 환경 조건 네 가지만 보고 판단합니다.
            <ul>
                <li><b>데이터 전처리</b>: CSV에서 수온·염분·pH·탁도·클로로필 농도를 뽑아 숫자로 바꾸고, 비어 있거나 센서 오류가 의심되는 값은 제외했습니다.
                    원본 {report['n_raw']:,}행 중 {report['n_used']:,}행을 사용합니다. (오류 의심 기준은 ‘데이터 분석’ 페이지와 같습니다.)</li>
                <li><b>학습·평가 분리</b>: 관측 날짜순으로 앞쪽 80%를 훈련 데이터, 뒤쪽 20%를 평가 데이터로 나눴습니다.
                    같은 관측 시각의 기록은 한쪽에만 들어가도록 시각 단위로 잘랐습니다.</li>
                <li><b>고농도 기준값</b>: 훈련 데이터 클로로필 농도의 75번째 백분위수를 훈련 데이터에서만 계산하고, 평가 데이터에도 같은 값을 적용했습니다.</li>
                <li><b>모델</b>: 로지스틱 회귀(LogisticRegression). 단위가 서로 다른 입력을 같은 척도로 맞추는 표준화를 앞에 붙였습니다.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="notice">
            <b>고농도 기준값 {threshold:.2f}은 연구용 상대 기준입니다.</b>
            훈련 데이터 안에서 클로로필 농도가 상대적으로 높은 쪽 25%를 ‘고농도’라고 부르기로 한 것이며,
            적조 판정 같은 공식·절대 기준이 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    split_table = pd.DataFrame(
        [
            ("훈련 데이터", fmt_period(train), f"{len(train):,}", f"{y_train.mean() * 100:.1f}%"),
            ("평가 데이터", fmt_period(test), f"{len(test):,}", f"{y_test.mean() * 100:.1f}%"),
        ],
        columns=["구분", "관측 기간", "행 수", "고농도(1) 비율"],
    )
    st.markdown(html_table(split_table, key_first_col=True), unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">평가 데이터는 훈련 데이터보다 뒤의 기간이라 계절이 달라 고농도 비율이 훈련 때와 다를 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 3. 사용자 입력
    section("환경 조건 입력", "아래 입력 범위는 CSV 데이터에서 실제로 관측된 최솟값과 최댓값입니다.")
    values = {}
    cols = st.columns(4)
    for col_ui, name in zip(cols, FEATURES):
        step, fmt = INPUT_STEP[name]
        lo = floor_to(float(df[name].min()), step)
        hi = ceil_to(float(df[name].max()), step)
        default = min(max(round(float(train[name].median()) / step) * step, lo), hi)
        with col_ui:
            values[name] = st.number_input(
                name,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                format=fmt,
                key=f"input_{name}",
            )
            st.caption(f"관측 범위 {fmt % df[name].min()} ~ {fmt % df[name].max()}")

    run = st.button("분류 실행", type="primary")
    if run:
        x_new = pd.DataFrame([[values[n] for n in FEATURES]], columns=FEATURES)
        proba = model.predict_proba(x_new)[0]
        classes = list(model.classes_)
        st.session_state["clf_result"] = {
            "inputs": dict(values),
            "label": int(model.predict(x_new)[0]),
            "p_high": float(proba[classes.index(1)]),
            "p_low": float(proba[classes.index(0)]),
        }

    # ------------------------------------------------------------------ 4. 분류 결과
    section("분류 결과")
    res = st.session_state.get("clf_result")
    if res is None:
        st.markdown(
            '<div class="caption">환경 조건을 입력하고 ‘분류 실행’ 버튼을 누르면 결과가 여기에 표시됩니다.</div>',
            unsafe_allow_html=True,
        )
    else:
        is_high = res["label"] == 1
        st.markdown(
            f"""
            <div class="verdict {'high' if is_high else 'low'}">
                <div class="verdict-label">모델의 분류 결과</div>
                <div class="verdict-value">{'고농도' if is_high else '저농도'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_grid(
            [
                stat_card("고농도일 예측 확률", f"{res['p_high'] * 100:.1f}%", "모델이 계산한 값", navy=False),
                stat_card("저농도일 예측 확률", f"{res['p_low'] * 100:.1f}%", "모델이 계산한 값", navy=True),
                stat_card("분류 기준", "50%", "고농도 확률이 50% 이상이면 고농도"),
            ]
        )

        fig = go.Figure(
            go.Bar(
                x=[res["p_low"] * 100, res["p_high"] * 100],
                y=["저농도", "고농도"],
                orientation="h",
                marker_color=[NAVY, TEAL],
                text=[f"{res['p_low'] * 100:.1f}%", f"{res['p_high'] * 100:.1f}%"],
                textposition="outside",
                hovertemplate="%{y}일 확률: %{x:.1f}%<extra></extra>",
            )
        )
        style_fig(fig, 220)
        fig.update_xaxes(title="예측 확률 (%)", range=[0, 112])
        fig.update_layout(margin=dict(l=70, r=30, t=10, b=50))
        st.plotly_chart(fig)

        st.markdown("**입력한 환경 조건 요약**")
        summary_rows = []
        for name in FEATURES:
            _, fmt = INPUT_STEP[name]
            summary_rows.append(
                (
                    name,
                    fmt % res["inputs"][name],
                    fmt % train[name].median(),
                    f"{fmt % train[name].min()} ~ {fmt % train[name].max()}",
                )
            )
        st.markdown(
            html_table(
                pd.DataFrame(summary_rows, columns=["환경 요인", "입력한 값", "훈련 데이터 중앙값", "훈련 데이터 관측 범위"]),
                key_first_col=True,
            ),
            unsafe_allow_html=True,
        )

        outside = [
            n for n in FEATURES if not (train[n].min() <= res["inputs"][n] <= train[n].max())
        ]
        if outside:
            st.warning(
                f"{', '.join(outside)} 값이 훈련 데이터의 관측 범위 밖입니다. 모델이 본 적 없는 조건이라 예측을 믿기 어렵습니다."
            )

        # -------------------------------------------------------------- 5. 결과 해석
        section("결과 해석")
        if is_high:
            msg = (
                "입력한 환경 조건은 이 데이터에서 <b>클로로필 농도가 상대적으로 높았던</b> 환경 조건과 유사합니다."
            )
        else:
            msg = (
                "입력한 환경 조건은 이 데이터에서 <b>클로로필 농도가 상대적으로 낮았던</b> 환경 조건과 유사합니다."
            )
        st.markdown(f'<div class="panel">{msg}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="notice">
                <b>꼭 알아두세요.</b> 이 결과는 실제 적조 발생 여부나 특정 미생물이 증식하고 있는지를 확정하지 않습니다.
                수온·염분·pH·탁도라는 네 가지 측정값이 과거 관측과 얼마나 비슷한지를 통계적으로 비교한 결과일 뿐입니다.
                ‘고농도’는 훈련 데이터의 75번째 백분위수({threshold:.2f})를 기준으로 한 연구용 상대 구분이고,
                한 관측 정점({fmt_period(df)})의 자료로 학습한 모델이라 다른 해역이나 다른 시기에는 맞지 않을 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ 모델 성능
    section("모델은 얼마나 맞힐까? (평가 데이터)", "학습에 쓰지 않은 뒤쪽 20% 기간의 데이터로 모델을 시험한 결과입니다.")
    m = result["metrics"]
    card_grid(
        [
            stat_card("정확도", f"{m['정확도'] * 100:.1f}%", "전체 중 맞힌 비율"),
            stat_card("정밀도", f"{m['정밀도'] * 100:.1f}%", "고농도라고 한 것 중 실제 고농도 비율", navy=True),
            stat_card("재현율", f"{m['재현율'] * 100:.1f}%", "실제 고농도 중 찾아낸 비율"),
            stat_card("F1 점수", f"{m['F1 점수'] * 100:.1f}%", "정밀도와 재현율의 균형", navy=True),
        ],
        four=True,
    )
    st.markdown(
        f'<div class="caption">참고: 평가 데이터 전체를 ‘저농도’라고만 답해도 정확도는 {result["baseline_acc"] * 100:.1f}%입니다. '
        "정확도를 이 값과 함께 비교해서 봐야 하고, 고농도를 잘 찾는지는 정밀도와 재현율로 확인하세요.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**혼동행렬 (평가 데이터)**")
        cm_df = pd.DataFrame(
            result["cm"],
            index=["실제 저농도", "실제 고농도"],
            columns=["예측 저농도", "예측 고농도"],
        )
        fig = px.imshow(
            cm_df,
            text_auto="d",
            aspect="auto",
            color_continuous_scale=[[0, "#FFFFFF"], [1, TEAL]],
        )
        style_fig(fig, 360)
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=90, r=20, t=10, b=50))
        fig.update_traces(hovertemplate="%{y} / %{x}<br>%{z:,}개<extra></extra>")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
    with right:
        st.markdown("**모델이 본 요인별 영향 (표준화 계수)**")
        coef = pd.Series(result["coef"], index=FEATURES)
        fig = go.Figure(
            go.Bar(
                x=coef.index,
                y=coef.values,
                marker_color=[TEAL if v >= 0 else NAVY for v in coef.values],
                text=[f"{v:+.2f}" for v in coef.values],
                textposition="outside",
                hovertemplate="%{x}<br>계수: %{y:+.3f}<extra></extra>",
            )
        )
        style_fig(fig, 360)
        fig.update_yaxes(title="계수 (+는 고농도 쪽)")
        fig.update_layout(margin=dict(l=70, r=20, t=10, b=50))
        st.plotly_chart(fig)
    st.markdown(
        '<div class="caption">계수가 +이면 그 값이 클수록 고농도로 분류될 가능성이 높아지는 방향, −이면 낮아지는 방향입니다. '
        "입력을 표준화한 뒤의 계수라서 요인끼리 크기를 비교할 수 있습니다. 원인을 뜻하는 것은 아닙니다.</div>",
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
        result = train_model(df)
    except Exception as e:
        st.error(f"모델을 준비하지 못했습니다: {e}")
        if st.button("다시 시도"):
            load_raw.clear()
            st.rerun()
        st.stop()

    render_page(df, report, result)


main()
