"""배경 지식 페이지 - 식물플랑크톤·클로로필과 수온·염분·pH·탁도의 관계

main.py 와 같은 폴더의 pages/ 안에 두면 Streamlit이 자동으로 메뉴에 추가합니다.
이 페이지는 데이터를 계산하지 않는 설명 페이지입니다.
"""

from html import escape

import streamlit as st

APP_TITLE = "해양 식물플랑크톤 분류 실험실"
PAGE_TITLE = "배경 지식"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}",
    page_icon="📖",
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
    font-size: 1.35rem; font-weight: 700; color: #12274D; margin: 46px 0 12px 0;
    padding-left: 12px; border-left: 6px solid #0E8C8C; letter-spacing: -0.01em;
}
.sec-desc { color: #4A6072; font-size: 0.95rem; margin: 0 0 14px 18px; line-height: 1.6; }

.summary {
    background: #F4FAFA; border: 1px solid #CFE6E6; border-left: 8px solid #0E8C8C;
    border-radius: 10px; padding: 22px 28px; color: #1D2B3A; line-height: 1.8; font-size: 1.08rem;
}
.summary b { color: #12274D; }
.summary .sub { display: block; margin-top: 10px; font-size: 1rem; color: #33475B; }

.panel {
    background: #F4FAFA; border: 1px solid #CFE6E6; border-radius: 10px;
    padding: 18px 24px; color: #1D2B3A; line-height: 1.75; font-size: 0.98rem;
}
.panel b { color: #12274D; }
.notice {
    background: #FFFFFF; border: 2px solid #12274D; border-left: 8px solid #12274D;
    border-radius: 8px; padding: 14px 20px; color: #1D2B3A; line-height: 1.75; font-size: 0.97rem;
    margin: 14px 0 6px 0;
}
.notice b { color: #12274D; }

.table-wrap { overflow-x: auto; border: 1px solid #D5E3E8; border-radius: 10px; }
table.lab-table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }
table.lab-table thead th {
    background: #12274D; color: #FFFFFF; font-weight: 700; text-align: left;
    padding: 11px 16px; white-space: nowrap;
}
table.lab-table thead th.hi { background: #0E8C8C; }
table.lab-table tbody td {
    padding: 11px 16px; border-top: 1px solid #E3EDF0; color: #1D2B3A; vertical-align: top; line-height: 1.65;
}
table.lab-table tbody tr:nth-child(even) td { background: #F4FAFA; }
table.lab-table td.key { font-weight: 700; color: #0B6E6E; white-space: nowrap; }
table.lab-table td.risk { border-top: 2px solid #12274D; }
.caption { color: #6B7F8C; font-size: 0.85rem; margin-top: 8px; line-height: 1.6; }

.pair-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
.pair {
    background: #FFFFFF; border: 1px solid #D5E3E8; border-top: 5px solid #0E8C8C;
    border-radius: 6px 6px 12px 12px; padding: 16px 20px; color: #1D2B3A; line-height: 1.7; font-size: 0.96rem;
}
.pair.navy { border-top-color: #12274D; }
.pair-title { color: #12274D; font-weight: 900; font-size: 1.08rem; margin-bottom: 4px; }

.key-message {
    background: #12274D; border-left: 10px solid #0E8C8C; border-radius: 12px;
    padding: 26px 32px; margin: 22px 0 8px 0; color: #FFFFFF; line-height: 1.8;
}
.key-message .km-label {
    display: inline-block; background: #0E8C8C; color: #FFFFFF; font-weight: 700; font-size: 0.85rem;
    padding: 3px 12px; border-radius: 999px; margin-bottom: 10px;
}
.key-message .km-title { font-size: 1.45rem; font-weight: 900; letter-spacing: -0.01em; margin-bottom: 8px; }
.key-message p { margin: 6px 0; font-size: 1rem; color: #E8F1F5; }
.key-message b { color: #FFFFFF; }

@media (max-width: 800px) {
    .pair-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.5rem; }
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def section(title, desc=""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)


def html_table(headers, rows, highlight_col=None, risk_last_row=False):
    """머리글과 행 목록으로 연구실 스타일 HTML 표를 만든다."""
    head = "".join(
        f'<th class="hi">{escape(h)}</th>' if i == highlight_col else f"<th>{escape(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r_idx, row in enumerate(rows):
        is_risk = risk_last_row and r_idx == len(rows) - 1
        cells = []
        for c_idx, value in enumerate(row):
            classes = []
            if c_idx == 0:
                classes.append("key")
            if is_risk:
                classes.append("risk")
            cls = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{cls}>{escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table class="lab-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# 페이지 본문
# ---------------------------------------------------------------------------
def render_page():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">📖</div>
            <div>
                <div class="hero-title">{PAGE_TITLE}</div>
                <div class="hero-sub">식물플랑크톤이 많을 때와 적을 때 바다는 어떻게 다를까, 그리고 수질 항목들은 서로 어떻게 이어져 있을까</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 1. 한 줄 요약
    section("한 줄 요약")
    st.markdown(
        """
        <div class="summary">
            <b>클로로필 a는 식물플랑크톤의 광합성 색소라서, 농도가 높을수록 식물플랑크톤이 많다는 뜻입니다.</b>
            <span class="sub">
                다만 클로로필은 식물플랑크톤의 <b>양</b>만 알려 주고, <b>어떤 종</b>인지는 알려 주지 않습니다.
                그래서 클로로필이 높다고 해서 해로운 종이 늘었다고 단정할 수는 없습니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 2. 많을 때와 적을 때
    section("식물플랑크톤이 많을 때와 적을 때", "클로로필 농도가 높은 바다와 낮은 바다는 이렇게 달라질 수 있습니다.")
    compare_rows = [
        (
            "물의 모습",
            "입자가 늘어 물이 탁해지고, 초록빛이나 갈색빛을 띱니다.",
            "물이 맑고 푸르게 보입니다.",
        ),
        (
            "산소",
            "낮에는 광합성으로 산소가 늘고 밤에는 호흡으로 줄어, 하루 안에서 변동이 큽니다.",
            "이런 하루 변동이 상대적으로 작습니다.",
        ),
        (
            "pH",
            "광합성이 이산화탄소를 써서 pH가 올라갈 수 있습니다.",
            "이런 상승 효과가 작아 pH가 상대적으로 안정적입니다.",
        ),
        (
            "영양염",
            "질소·인 같은 영양염을 많이 소모합니다.",
            "소모가 적어 영양염이 남아 있을 수도 있고, 영양염이나 빛·수온이 부족해 식물플랑크톤이 늘지 못한 것일 수도 있습니다.",
        ),
        (
            "먹이사슬",
            "동물플랑크톤을 비롯해 이를 먹는 생물의 먹이가 풍부해집니다.",
            "먹이가 적어 해양의 생산성이 낮습니다.",
        ),
        (
            "위험 신호",
            "지나치게 번성했다가 죽어 분해되면 산소가 크게 줄어 저산소가 생길 수 있습니다. 일부 종은 독소를 냅니다.",
            "해로운 번성의 위험은 낮지만, 너무 낮으면 생산성이 낮은 상태입니다.",
        ),
    ]
    st.markdown(
        html_table(
            ["구분", "많을 때 (클로로필 높음)", "적을 때 (클로로필 낮음)"],
            compare_rows,
            highlight_col=1,
            risk_last_row=True,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="notice">
            <b>많다고 나쁜 것이 아닙니다.</b>
            적당한 양의 식물플랑크톤은 먹이사슬의 바탕입니다.
            문제가 되는 것은 영양염이 넘쳐서 식물플랑크톤이 <b>과도하게 번성하는 경우(부영양화)</b>입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 3. 네 가지 환경 요인과 클로로필
    section("네 가지 환경 요인과 클로로필의 관계", "이 앱이 분류에 쓰는 입력변수 네 가지가 클로로필과 이어질 것으로 본 이유입니다.")
    factor_rows = [
        (
            "수온",
            "식물플랑크톤의 증식 속도와 광합성 속도에 영향을 줍니다. 수온에 따라 물이 층을 이루는 정도도 달라져 영양염이 섞이는 양에 영향을 줄 수 있습니다.",
            "종마다 알맞은 수온이 다릅니다. 수온은 계절과 함께 움직이므로 빛의 양 같은 다른 요인과 구분하기 어렵습니다.",
        ),
        (
            "염분",
            "염분은 강물 같은 담수가 얼마나 들어왔는지를 보여 줍니다. 담수와 함께 영양염이 들어올 수 있고, 염분이 달라지면 우세한 식물플랑크톤의 종류도 바뀔 수 있습니다.",
            "염분이 낮다는 것은 담수가 들어왔다는 신호일 뿐, 영양염의 양을 직접 재는 값은 아닙니다.",
        ),
        (
            "pH",
            "광합성이 이산화탄소를 쓰면 pH가 올라갑니다. 식물플랑크톤이 많은 시기의 흔적이 pH에 나타날 수 있습니다.",
            "pH는 번성의 원인이면서 결과일 수 있습니다. 식물플랑크톤이 pH를 바꾸기도 하고, pH가 달라진 환경이 식물플랑크톤에게 영향을 주기도 해서 어느 쪽이 먼저인지 구분하기 어렵습니다.",
        ),
        (
            "탁도",
            "식물플랑크톤 자체가 물속의 입자라서 많아지면 탁도가 올라갈 수 있습니다. 한편 탁도가 높으면 빛이 깊이 들어가지 못해 광합성이 줄 수도 있습니다.",
            "탁도는 원인이면서 결과일 수 있습니다. 흙 같은 입자 때문에 탁한 것인지, 식물플랑크톤 때문에 탁한 것인지 탁도만으로는 알 수 없습니다.",
        ),
    ]
    st.markdown(
        html_table(["환경 요인", "이어질 것으로 본 이유", "주의할 점"], factor_rows),
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ 4. 변수들 사이의 관계
    section("변수들 사이의 관계", "수질 항목끼리도 서로 영향을 주고받습니다. 클로로필과 따로 떼어서 이해하기 어려운 이유입니다.")
    st.markdown(
        """
        <div class="pair-grid">
            <div class="pair">
                <div class="pair-title">수온과 염분</div>
                염분은 주로 강물과 비 같은 담수가 얼마나 들어오느냐에 좌우되고, 수온은 계절과 햇빛에 좌우됩니다.
                서로가 서로의 원인은 아니지만 계절에 따라 함께 변해서 같이 움직이는 것처럼 보일 수 있습니다.
            </div>
            <div class="pair navy">
                <div class="pair-title">염분과 탁도</div>
                강물이 들어오면 흙 같은 입자도 함께 들어옵니다.
                그래서 염분은 낮아지고 탁도는 높아지는 경향이 나타날 수 있습니다.
            </div>
            <div class="pair navy">
                <div class="pair-title">pH와 용존산소량</div>
                광합성은 산소를 만들고 이산화탄소를 쓰기 때문에 용존산소량과 pH가 함께 오르는 경향이 있습니다.
                호흡과 분해는 반대로 산소를 쓰고 이산화탄소를 내놓아 둘을 함께 낮춥니다.
            </div>
            <div class="pair">
                <div class="pair-title">수온과 용존산소량</div>
                물이 따뜻할수록 물에 녹을 수 있는 산소의 양이 줄어듭니다.
                다만 수온이 올라도 광합성이 활발하면 산소가 늘 수 있어서, 두 효과가 섞여 나타납니다.
            </div>
            <div class="pair">
                <div class="pair-title">탁도와 클로로필</div>
                식물플랑크톤이 많아지면 입자가 늘어 탁도가 올라갑니다.
                반대로 너무 탁하면 빛이 막혀 식물플랑크톤이 늘기 어렵습니다.
                서로가 서로의 원인이면서 결과인 관계입니다.
            </div>
        </div>
        <div class="caption">용존산소량은 CSV에 들어 있지만 이 앱의 분류 입력변수(수온·염분·pH·탁도)에는 넣지 않았습니다. 여기서는 pH와의 관계를 이해하기 위한 배경으로만 설명합니다.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="key-message">
            <span class="km-label">가장 중요한 메시지</span>
            <div class="km-title">변수들은 ‘계절’이라는 같은 요인과 함께 움직입니다.<br>그래서 한 요인만의 영향을 따로 떼어 말하기 어렵습니다.</div>
            <p>수온은 계절에 따라 오르내리고, 비가 오는 시기도 계절에 따라 달라져 염분과 탁도에 영향을 줍니다. 햇빛의 양도 계절마다 달라 광합성에 영향을 줍니다.</p>
            <p>그래서 어떤 요인이 클로로필과 함께 움직인다고 해서 그 요인이 클로로필을 <b>바꾼다</b>고 말할 수는 없습니다. 눈에 보이지 않는 계절 요인이 두 값을 함께 움직이게 했을 수 있기 때문입니다.</p>
            <p>이 앱의 상관계수나 분류 모델의 계수를 읽을 때도 같은 점을 기억해 주세요. 값의 부호가 위 설명과 다르게 나오면, 변수들이 서로 얽혀 있어서일 수 있습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="notice">
            <b>이 페이지는 일반적인 해양학 설명입니다.</b>
            새만금 관측 데이터에서 이런 관계가 실제로 나타나는지는 이 페이지가 확인해 주지 않습니다.
            ‘데이터 분석’ 페이지에서 실제 데이터로 계산한 값을 보고 비교해 보세요.
        </div>
        """,
        unsafe_allow_html=True,
    )


render_page()
