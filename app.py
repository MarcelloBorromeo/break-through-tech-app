import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph


# -----------------------------------------------------------
# IEC ULTRA-PREMIUM THEME — GLOBAL CSS
# -----------------------------------------------------------
st.markdown("""
<style>

    /* Global Background */
    body, .stApp {
        background: linear-gradient(180deg, #E8F1FF 0%, #FFFFFF 60%) !important;
    }

    /* Container Width + Centering */
    .block-container {
        max-width: 1000px !important;
        margin-top: -20px;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #0E3A75;
        letter-spacing: .5px;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subheader {
        text-align: center;
        font-size: 20px;
        color: #167DFF;
        margin-bottom: 30px;
    }

    /* Glass Card */
    .card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(14px);
        border-radius: 14px;
        padding: 25px;
        margin-top: 18px;
        border: 1px solid #D9E8FF;
        box-shadow: 0 6px 22px rgba(0,0,0,0.06);
    }

    /* Investor Name */
    .investor-name {
        font-size: 22px;
        font-weight: 700;
        color: #167DFF;
        margin-bottom: 10px;
    }

    /* Explanation Text */
    .explanation {
        color: #0E3A75;
        font-size: 16px;
        line-height: 1.55;
        margin-bottom: 12px;
    }

    /* Web Summary */
    .web-summary {
        font-size: 14px;
        color: #167DFF;
        font-weight: 600;
        margin-top: -6px;
    }

    /* Section Header */
    .section-header {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0E3A75 !important;
        margin-top: 40px;
        margin-bottom: 12px;
    }

    /* Table Formatting */
    table {
        border-collapse: collapse;
        width: 100%;
        background: white;
    }
    th {
        background-color: #167DFF !important;
        color: white !important;
        text-align: center !important;
        font-size: 16px !important;
        padding: 8px !important;
    }
    td {
        background-color: #F4F8FF !important;
        padding: 10px !important;
        text-align: center !important;
        font-size: 15px !important;
    }

    /* Progress text */
    .progress-text {
        text-align: center;
        color: #0E3A75;
        font-size: 16px;
        margin-top: 5px;
    }

    /* Buttons */
    .stButton>button {
        background: #167DFF;
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        border: none;
        font-size: 17px;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: #0E3A75;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Hero Title
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
    deal = st.number_input("Deal Size ($M)", 50.0)
    growth = st.number_input("Growth YoY", 0.35)
    desc = st.text_area("Description", "AI workflow automation platform.")


# -----------------------------------------------------------
# Matching Logic
# -----------------------------------------------------------
if st.button("Run Matching"):
    step_display = st.empty()

    def update_step(text):
        step_display.markdown(f"<div class='progress-text'>{text}</div>", unsafe_allow_html=True)

    with st.spinner("Running investor matching pipeline…"):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc
            },
            progress_callback=update_step
        )

    step_display.markdown("<div class='progress-text' style='color:green;'>✔ Done</div>", unsafe_allow_html=True)


    # -----------------------------------------------------------
    # Top 3 MATCHES TABLE
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Top 3 Matches</div>", unsafe_allow_html=True)

    df = pd.DataFrame(results)

    st.dataframe(
        df[["investor", "final"]].rename(columns={
            "investor": "Investor",
            "final": "Match Score"
        }),
        use_container_width=True
    )


    # -----------------------------------------------------------
    # Reasoning Cards
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Reasoning</div>", unsafe_allow_html=True)

    for r in results:
        st.markdown(f"""
        <div class='card'>
            <div class='investor-name'>{r['investor']}</div>
            <div class='explanation'>{r['explanation']}</div>
            <div class='web-summary'><strong>Web Summary:</strong> {r['web']}</div>
        </div>
        """, unsafe_allow_html=True)
