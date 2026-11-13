"""해양 환경 비교 실험실 페이지 - 두 환경 조건의 클로로필 고농도 가능성 비교

main.py 와 같은 폴더의 pages/ 안에 두면 Streamlit이 자동으로 메뉴에 추가합니다.
모델은 '분류 실험실' 페이지와 같은 방식으로 학습합니다.
(관측 날짜순 앞 80% 훈련, 고농도 기준 = 훈련 데이터 클로로필의 75번째 백분위수, LogisticRegression)
"""

import math
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

APP_TITLE = "해양 식물플랑크톤 분류 실험실"
PAGE_TITLE = "해양 환경 비교 실험실"

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

# 슬라이더 간격, 표시 형식, 소수 자릿수
STEP = {"수온": 0.1, "염분": 0.1, "pH": 0.01, "탁도": 1.0}
FMT = {"수온": "%.1f", "염분": "%.1f", "pH": "%.2f", "탁도": "%.0f"}
DEC = {"수온": 1, "염분": 1, "pH": 2, "탁도": 0}

# 측정기 오류로 의심되는 값을 거르는 기준 (다른 페이지와 같은 기준)
RULES = [
    ("수온", lambda s: (s > 0) & (s <= 40)),
    ("염분", lambda s: (s > 0) & (s <= 40)),
    ("pH", lambda s: s.between(6, 10)),
    ("탁도", lambda s: s >= 0),
    ("클로로필 농도", lambda s: s >= 0),
]

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}",
    page_icon="⚖️",
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

.env-tag {
    display: inline-block; padding: 6px 18px; border-radius: 999px; font-weight: 900;
    font-size: 1.05rem; color: #FFFFFF; margin-bottom: 8px;
}
.env-tag.a { background: #0E8C8C; }
.env-tag.b { background: #12274D; }

.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 10px 0 14px 0; }
.card-grid.two { grid-template-columns: repeat(2, 1fr); }
.card-grid.four { grid-template-columns: repeat(4, 1fr); }
.stat {
    background: #FFFFFF; border: 1px solid #D5E3E8; border-top: 5px solid #0E8C8C;
    border-radius: 6px 6px 12px 12px; padding: 14px 18px;
}
.stat.navy { border-top-color: #12274D; }
.stat-label { color: #4A6072; font-size: 0.88rem; font-weight: 500; }
.stat-value { color: #12274D; font-size: 1.6rem; font-weight: 900; margin: 2px 0; letter-spacing: -0.02em; }
.stat-sub { color: #6B7F8C; font-size: 0.8rem; line-height: 1.5; }

.verdict { border-radius: 12px; padding: 18px 24px; color: #FFFFFF; margin: 6px 0 12px 0; }
.verdict.high { background: #0E8C8C; }
.verdict.low { background: #12274D; }
.verdict-label { font-size: 0.95rem; opacity: 0.9; }
.verdict-value { font-size: 2rem; font-weight: 900; letter-spacing: -0.02em; margin-top: 2px; }

.table-wrap { overflow-x: auto; border: 1px solid #D5E3E8; border-radius: 10px; }
table.lab-table { border-collapse: collapse; width: 100%; font-size: 0.93rem; }
table.lab-table thead th {
    background: #12274D; color: #FFFFFF; font-weight: 700; text-align: left;
    padding: 10px 14px; white-space: nowrap;
}
table.lab-table tbody td { padding: 9px 14px; border-top: 1px solid #E3EDF0; color: #1D2B3A; vertical-align: top; }
table.lab-table tbody tr:nth-child(even) td { background: #F4FAFA; }
table.lab-table td.key { font-weight: 700; color: #0B6E6E; white-space: nowrap; }
.caption { color: #6B7F8C; font-size: 0.85rem; margin-top: 8px; line-height: 1.6; }

label, [data-testid="stWidgetLabel"] p { color: #12274D !important; font-weight: 700 !important; }
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
# 데이터 불러오기와 전처리
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

    valid = work["관측일자"].notna()
    for label in FEATURES + [TARGET]:
        valid &= work[label].notna()
    for label, ok in RULES:
        valid &= ok(work[label])
    return work[valid].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# 학습/평가 분리와 모델 학습 ('분류 실험실' 페이지와 같은 방식)
# ---------------------------------------------------------------------------
def split_by_time(df, ratio=TRAIN_RATIO):
    """관측 날짜순 앞 80% 훈련 / 뒤 20% 평가. 같은 관측 시각이 양쪽에 겹치지 않게 시각 단위로 자른다."""
    times = df["관측일자"].drop_duplicates().sort_values().reset_index(drop=True)
    cut = int(len(times) * ratio)
    cutoff = times.iloc[cut - 1]
    train = df[df["관측일자"] <= cutoff].copy()
    test = df[df["관측일자"] > cutoff].copy()
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
        "metrics": {
            "정확도": accuracy_score(y_test, pred),
            "정밀도": precision_score(y_test, pred, zero_division=0),
            "재현율": recall_score(y_test, pred, zero_division=0),
            "F1 점수": f1_score(y_test, pred, zero_division=0),
        },
        "baseline_acc": float((y_test == 0).mean()),
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


def env_result_html(env, res):
    """환경 하나의 분류 결과 카드(판정 + 두 확률)."""
    is_high = res["label"] == 1
    return (
        f'<div class="verdict {"high" if is_high else "low"}">'
        f'<div class="verdict-label">환경 {env}의 분류 결과</div>'
        f'<div class="verdict-value">{"고농도" if is_high else "저농도"}</div></div>'
        '<div class="card-grid two">'
        + stat_card("고농도일 예측 확률", f"{res['p_high']:.1f}%", "모델이 계산한 값")
        + stat_card("저농도일 예측 확률", f"{res['p_low']:.1f}%", "모델이 계산한 값", navy=True)
        + "</div>"
    )


def classify(model, values):
    """입력 한 건을 분류해 판정과 확률(%)을 돌려준다."""
    x_new = pd.DataFrame([[values[n] for n in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(x_new)[0]
    classes = list(model.classes_)
    return {
        "label": int(model.predict(x_new)[0]),
        "p_high": float(proba[classes.index(1)]) * 100,
        "p_low": float(proba[classes.index(0)]) * 100,
    }


# ---------------------------------------------------------------------------
# 페이지 본문
# ---------------------------------------------------------------------------
def render_page(df, result):
    model = result["model"]
    train = result["train"]
    threshold = result["threshold"]
    m = result["metrics"]

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">⚖️</div>
            <div>
                <div class="hero-title">{PAGE_TITLE}</div>
                <div class="hero-sub">두 가지 해양 환경 조건을 나란히 놓고 클로로필 고농도 가능성을 비교합니다</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="panel">
            환경 A와 B의 수온, 염분, pH, 탁도를 각각 입력하고 비교 버튼을 누르면, ‘분류 실험실’과 <b>같은 로지스틱 회귀 모델</b>이
            두 환경을 고농도 또는 저농도로 분류하고 예측 확률을 보여 줍니다.
            슬라이더의 처음 값은 CSV 데이터의 분포를 따라, 환경 A는 각 요인의 <b>하위 25% 지점</b>, 환경 B는 <b>상위 25% 지점</b>(75번째 백분위수)으로 맞춰 두었습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="notice">
            <b>고농도 기준값 {threshold:.2f}은 연구용 상대 기준입니다.</b>
            훈련 데이터(관측 날짜순 앞 80%, {fmt_period(train)})의 클로로필 농도 75번째 백분위수이며, 적조 판정 같은 공식·절대 기준이 아닙니다.<br>
            이 모델의 평가 데이터(뒤 20%) 성능은 정확도 {m['정확도'] * 100:.1f}%, 정밀도 {m['정밀도'] * 100:.1f}%, 재현율 {m['재현율'] * 100:.1f}%입니다.
            (평가 데이터를 전부 ‘저농도’라고 답했을 때의 정확도는 {result['baseline_acc'] * 100:.1f}%) 자세한 평가는 ‘분류 실험실’ 페이지에서 볼 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 1. 환경 A와 B 입력
    section("환경 A와 B 입력", "슬라이더 범위는 CSV 데이터에서 실제로 관측된 최솟값~최댓값입니다.")
    inputs = {"A": {}, "B": {}}
    left, right = st.columns(2, gap="large")
    for env, col_ui, q in (("A", left, 0.25), ("B", right, 0.75)):
        with col_ui:
            st.markdown(f'<span class="env-tag {env.lower()}">환경 {env}</span>', unsafe_allow_html=True)
            for name in FEATURES:
                step = STEP[name]
                lo = floor_to(float(df[name].min()), step)
                hi = ceil_to(float(df[name].max()), step)
                default = round(float(df[name].quantile(q)) / step) * step
                default = min(max(default, lo), hi)
                inputs[env][name] = st.slider(
                    name,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    format=FMT[name],
                    key=f"cmp_{env}_{name}",
                )

    run = st.button("두 환경 비교하기", type="primary")
    if run:
        a_in, b_in = dict(inputs["A"]), dict(inputs["B"])
        st.session_state["cmp_result"] = {
            "inputs": {"A": a_in, "B": b_in},
            "A": classify(model, a_in),
            "B": classify(model, b_in),
        }

    res = st.session_state.get("cmp_result")
    if res is None:
        st.markdown(
            '<div class="caption">두 환경의 값을 정한 뒤 ‘두 환경 비교하기’ 버튼을 누르면 결과가 아래에 표시됩니다.</div>',
            unsafe_allow_html=True,
        )
        return

    a_in, b_in = res["inputs"]["A"], res["inputs"]["B"]
    ra, rb = res["A"], res["B"]

    # ------------------------------------------------------------------ 2. 분류
    section("각 환경의 분류 결과")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(env_result_html("A", ra), unsafe_allow_html=True)
    with col_b:
        st.markdown(env_result_html("B", rb), unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">고농도 예측 확률이 50% 이상이면 고농도, 미만이면 저농도로 분류합니다.</div>',
        unsafe_allow_html=True,
    )

    outside = []
    for env in ("A", "B"):
        for n in FEATURES:
            if not (train[n].min() <= res["inputs"][env][n] <= train[n].max()):
                outside.append(f"환경 {env}의 {n}")
    if outside:
        st.warning(
            f"{', '.join(outside)} 값이 훈련 데이터의 관측 범위 밖입니다. 모델이 본 적 없는 조건이라 예측을 믿기 어렵습니다."
        )

    # ------------------------------------------------------------------ 3. 결과 비교 (막대그래프)
    section("고농도 예측 확률 비교")
    fig = go.Figure(
        go.Bar(
            x=["환경 A", "환경 B"],
            y=[ra["p_high"], rb["p_high"]],
            marker_color=[TEAL if ra["label"] == 1 else NAVY, TEAL if rb["label"] == 1 else NAVY],
            text=[f"{ra['p_high']:.1f}%", f"{rb['p_high']:.1f}%"],
            textposition="outside",
            width=0.45,
            hovertemplate="%{x}<br>고농도 예측 확률: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#6B7F8C",
        annotation_text="분류 기준 50%",
        annotation_position="top right",
    )
    style_fig(fig, 400)
    fig.update_yaxes(title="고농도 예측 확률 (%)", range=[0, 112])
    fig.update_layout(margin=dict(l=70, r=30, t=20, b=50))
    st.plotly_chart(fig)
    st.markdown(
        '<div class="caption">막대 색은 분류 결과를 뜻합니다. 청록색은 고농도, 남색은 저농도로 분류된 환경입니다.</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 4. 환경 조건 비교 (표)
    section("환경 조건 비교", "차이는 환경 B에서 환경 A를 뺀 값입니다.")
    rows = []
    for n in FEATURES:
        d = b_in[n] - a_in[n]
        rows.append(
            (
                n,
                f"{a_in[n]:.{DEC[n]}f}",
                f"{b_in[n]:.{DEC[n]}f}",
                f"{d:+.{DEC[n]}f}",
            )
        )
    st.markdown(
        html_table(pd.DataFrame(rows, columns=["환경 요인", "환경 A", "환경 B", "차이 (B − A)"]), key_first_col=True),
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 5. 결과 해석
    section("결과 해석")
    diff = rb["p_high"] - ra["p_high"]
    name = {"A": "환경 A", "B": "환경 B"}
    if abs(diff) < 1:
        line1 = (
            f"두 환경의 고농도 예측 확률은 거의 같습니다. (환경 A {ra['p_high']:.1f}%, 환경 B {rb['p_high']:.1f}%, "
            f"차이 {abs(diff):.1f}%p)"
        )
    else:
        hi_env, lo_env = ("B", "A") if diff > 0 else ("A", "B")
        hi_p = res[hi_env]["p_high"]
        lo_p = res[lo_env]["p_high"]
        line1 = (
            f"<b>{name[hi_env]}</b>의 고농도 예측 확률이 {hi_p:.1f}%로, {name[lo_env]}({lo_p:.1f}%)보다 "
            f"<b>{abs(diff):.1f}%p 더 높습니다.</b> "
            f"즉 {name[hi_env]}의 환경 조건이 이 데이터에서 클로로필 농도가 상대적으로 높았던 시기의 조건과 더 비슷합니다."
        )

    if ra["label"] == rb["label"]:
        kind = "고농도" if ra["label"] == 1 else "저농도"
        line2 = f"두 환경 모두 <b>{kind}</b>로 분류되었습니다."
    else:
        hi_side = "A" if ra["label"] == 1 else "B"
        lo_side = "B" if hi_side == "A" else "A"
        line2 = f"{name[hi_side]}는 <b>고농도</b>, {name[lo_side]}는 <b>저농도</b>로 분류되었습니다."

    # 모델 안에서 두 환경의 차이를 만든 요인 (표준화한 입력 × 계수)
    z = model.named_steps["standardscaler"].transform(
        pd.DataFrame([[a_in[n] for n in FEATURES], [b_in[n] for n in FEATURES]], columns=FEATURES)
    )
    coef = model.named_steps["logisticregression"].coef_[0]
    contrib = pd.Series(coef * (z[1] - z[0]), index=FEATURES)
    if contrib.abs().max() == 0:
        line3 = "두 환경의 입력값이 같아서 비교할 차이가 없습니다."
    else:
        top = contrib.abs().idxmax()
        push = "높이는" if contrib[top] > 0 else "낮추는"
        line3 = (
            f"모델 계산상 두 환경의 차이를 가장 크게 만든 요인은 <b>{top}</b>입니다. "
            f"{top}이(가) 환경 A의 {a_in[top]:.{DEC[top]}f}에서 환경 B의 {b_in[top]:.{DEC[top]}f}로 달라진 것이 "
            f"환경 B의 고농도 확률을 {push} 방향으로 작용했습니다."
        )

    st.markdown(
        f'<div class="panel">{line1}<br>{line2}<br>{line3}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="notice">
            <b>꼭 알아두세요.</b> 이 비교는 실제 적조 발생 여부나 특정 미생물의 증식 여부를 확정하지 않습니다.
            수온·염분·pH·탁도라는 네 가지 측정값이 과거 관측과 얼마나 비슷한지를 통계적으로 견준 결과이고,
            ‘고농도’는 훈련 데이터의 75번째 백분위수({threshold:.2f})를 기준으로 한 연구용 상대 구분입니다.
            한 관측 정점({fmt_period(df)})의 자료로 학습한 모델이라 다른 해역이나 시기에는 맞지 않을 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    try:
        raw = load_raw()
        missing = [c for c in ["관측일자", *COLS.values()] if c not in raw.columns]
        if missing:
            raise RuntimeError(f"CSV에서 필요한 열을 찾지 못했습니다: {', '.join(missing)} (현재 열: {', '.join(raw.columns)})")
        df = prepare(raw)
        if df.empty:
            raise RuntimeError("정리 후 남은 데이터가 없습니다. 원본 파일의 값을 확인해 주세요.")
        result = train_model(df)
    except Exception as e:
        st.error(f"모델을 준비하지 못했습니다: {e}")
        if st.button("다시 시도"):
            load_raw.clear()
            st.rerun()
        st.stop()

    render_page(df, result)


main()
