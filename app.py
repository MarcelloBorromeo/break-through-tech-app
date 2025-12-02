import streamlit as st
import pandas as pd
import time
from engine import InvestorDataPipeline, InvestorMatchingGraph

# -----------------------------------------------------------
# IEC PREMIUM UI THEME — UPDATED FONTS
# -----------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800;900&family=Open+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

html, body, div, span, textarea, input, button {
    font-family: 'Open Sans', sans-serif !important;
}

.main-title, .section-header, .investor-name, .stButton>button {
    font-family: 'Montserrat', sans-serif !important;
}

body, .stApp {
    background: linear-gradient(180deg, #E8F1FF 0%, #FFFFFF 60%) !important;
}

.block-container {
    max-width: 1000px !important;
}

.main-title {
    text-align: center;
    font-size: 46px !important;
    font-weight: 800 !important;
    color: #000000 !important;
}

.subheader {
    text-align: center;
    font-size: 20px;
    color: #167DFF;
    font-family: 'Open Sans', sans-serif !important;
}

.section-header {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #0E3A75 !important;
}

.card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 22px;
    margin-top: 10px;
    border: 1.5px solid #AFCBFF;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}

.investor-name {
    font-size: 22px;
    font-weight: 700;
    color: #167DFF;
}

.explanation {
    text-align: justify;
    color: #0E3A75;
    font-size: 16px;
}

.web-summary {
    font-size: 14px;
    color: #167DFF;
    font-weight: 600;
}

.stButton>button {
    background: #167DFF;
    color: white;
    border-radius: 8px;
    height: 50px;
    width: 100%;
    border: none;
    font-size: 18px;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Page Title
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>Venture Investor Matching Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>IEC-powered precision investor recommendations</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# Load Engine
# -----------------------------------------------------------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    key = st.secrets["OPENROUTER_API_KEY"]
    return InvestorMatchingGraph(dp, key)

engine = load_engine()


# -----------------------------------------------------------
# Sidebar Inputs
# -----------------------------------------------------------
with st.sidebar:
    st.header("Startup Profile")

    industry = st.text_input("Industry", "Software")

    deal = st.number_input(
        "Deal Size ($M)",
        value=50.0,
        min_value=0.0,
        max_value=1_000_000_000.0,
    )

    growth = st.number_input(
        "Growth YoY",
        value=0.35,
        min_value=0.0,
        max_value=1.0,
    )

    desc = st.text_area("Description", "AI workflow automation platform.")

    # -------------------------------
    # AI Toggle Slider
    # -------------------------------
    st.subheader("AI Explanation")

    ai_explanation = st.checkbox("Enable AI-generated reasoning", value=True)

    toggle_color = "#167DFF" if ai_explanation else "#666666"
    toggle_text = "AI Explanation: ON" if ai_explanation else "AI Explanation: OFF"

    st.markdown(
        f"""
        <div style="
            background:{toggle_color};
            padding:10px;
            text-align:center;
            color:white;
            border-radius:6px;
            font-weight:700;">
            {toggle_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------
# Run Matching
# -----------------------------------------------------------
if st.button("Run Matching"):

    step_display = st.empty()

    def update_step(text):
        step_display.markdown(
            f"<div class='progress-text'>{text}</div>", unsafe_allow_html=True
        )

    with st.spinner("Running investor matching pipeline…"):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc,
            },
            progress_callback=update_step,
            use_llm=ai_explanation,   # ← NEW TOGGLE
        )

    # Done message
    done_msg = st.empty()
    done_msg.markdown("<div class='progress-text' style='color:green;'>✔ Done</div>", unsafe_allow_html=True)
    time.sleep(2)
    done_msg.empty()


    # -----------------------------------------------------------
    # Top 3 MATCHES TABLE
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Top 3 Matches</div>", unsafe_allow_html=True)

    df = pd.DataFrame(results)

    table_html = """
<style>
.match-table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(255,255,255,0.80);
    border-radius: 12px;
    border: 1.5px solid #AFCBFF;
}
.match-table th {
    background-color: #167DFF;
    color: white;
    padding: 12px;
    text-align: center;
    font-weight: 700;
}
.match-table td {
    padding: 12px;
    text-align: center;
    font-weight: 600;
    color: #0E3A75;
}
</style>
<table class="match-table">
<tr><th>Investor</th><th>Match Score</th></tr>
"""

    for _, row in df.iterrows():
        table_html += f"""
<tr>
<td>{row['investor']}</td>
<td>{row['final']}</td>
</tr>
"""

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)


    # -----------------------------------------------------------
    # Reasoning Cards (ONLY if AI is ON)
    # -----------------------------------------------------------
    if ai_explanation:
        st.markdown("<div class='section-header'>Reasoning</div>", unsafe_allow_html=True)

        for r in results:
            st.markdown(f"""
            <div class='card'>
                <div class='investor-name'>{r['investor']}</div>
                <div class='explanation'>{r['explanation']}</div>
                <div class='web-summary'><strong>Web Summary:</strong> {r['web']}</div>
            </div>
            """, unsafe_allow_html=True)
