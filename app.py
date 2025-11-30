import streamlit as st
import pandas as pd
import time
from engine import InvestorDataPipeline, SuperInvestorMatcher


st.set_page_config(layout="wide")

# ------------------------------------------------------
# IEC STYLE
# ------------------------------------------------------
st.markdown("""
<style>
.block-container { max-width: 1000px; }
h1 { text-align:center; color:#000; font-weight:800; margin-bottom:4px; }
.sub { text-align:center; color:#167DFF; margin-bottom:18px; }
.section { font-size:24px; font-weight:700; color:#0E3A75; margin-top:18px; margin-bottom:6px; }

.card {
    background:#ffffffD0;
    border-radius:12px;
    border:1.5px solid #AFCBFF;
    padding:18px;
    margin-top:10px;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
}

.investor { font-size:20px; font-weight:700; color:#167DFF; }
.expl { text-align:justify; color:#0E3A75; font-size:15px; line-height:1.5; }
.web { font-size:13px; color:#167DFF; font-weight:600; margin-top:6px; }

.progress-text { text-align:left; color:#0E3A75; margin-top:6px; }
table {
    border-collapse: collapse;
    width:100%;
    background:white;
    border:1.4px solid #AFCBFF;
}
th {
    background:#167DFF;
    color:white;
    padding:9px;
    font-size:17px;
}
td {
    background:#FFFFFF;
    color:#0E3A75;
    padding:10px;
    font-size:16px;
    text-align:center;
}
td:first-child { text-align:left; padding-left:12px; }
.stButton>button {
    width:100%;
    height:50px;
    background:#167DFF;
    color:white;
    font-size:18px;
    border:none;
    border-radius:8px;
}
.stButton>button:hover { background:#0E3A75; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Title
# ------------------------------------------------------
st.markdown("<h1>Venture Investor Matching Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub'>IEC-powered precision investor recommendations</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# Load Engine
# ------------------------------------------------------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    key = st.secrets["OPENROUTER_API_KEY"]
    return SuperInvestorMatcher(dp, key)

matcher = load_engine()


# ------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------
with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal = st.number_input("Deal Size ($M)", 50.0)
    growth = st.number_input("Revenue Growth YoY", 0.35)
    desc = st.text_area("Description", "AI workflow automation platform.")


# ------------------------------------------------------
# Spinner Progress
# ------------------------------------------------------
step_display = st.empty()
def progress(text):
    step_display.markdown(f"<div class='progress-text'>{text}</div>", unsafe_allow_html=True)


# ------------------------------------------------------
# Run Button
# ------------------------------------------------------
if st.button("Run Matching"):

    with st.spinner("Running investor matching pipeline…"):
        startup = {
            "industry": industry,
            "deal_size_m": deal,
            "revenue_growth_yoy": growth,
            "description": desc
        }
        results = matcher.run_pipeline(startup, progress_cb=progress)

    # Done animation
    done = st.empty()
    done.markdown("<div class='progress-text' style='color:green;'>✔ Done</div>", unsafe_allow_html=True)
    time.sleep(2)
    done.empty()

    # Display Table
    top3 = results[:3]
    df = pd.DataFrame(top3)[["investor","score"]].rename(columns={"investor":"Investor","score":"Match Score"})
    st.markdown("<div class='section'>Top 3 Matches</div>", unsafe_allow_html=True)
    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    # Display Reasoning
    st.markdown("<div class='section'>Reasoning</div>", unsafe_allow_html=True)
    for r in top3:
        st.markdown(f"""
        <div class="card">
            <div class="investor">{r['investor']}</div>
            <div class="expl">{r['explanation']}</div>
            <div class="web"><strong>Web Summary:</strong> {r['web']}</div>
        </div>
        """, unsafe_allow_html=True)
