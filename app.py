# app.py
import streamlit as st
import pandas as pd
import time
from engine import HybridDataPipeline, SuperInvestorMatcher

st.set_page_config(layout="wide")

# minimal CSS tuned to your last requests (title black, justified text, outlined cards, tightened spacing)
st.markdown("""
<style>
.block-container { max-width: 1000px; }
h1 { text-align:center; color:#000000; font-weight:800; margin-bottom:6px; }
.sub { text-align:center; color:#167DFF; margin-bottom:14px; }
.card { background: rgba(255,255,255,0.85); border-radius:12px; padding:18px; margin-top:10px; border:1.4px solid #AFCBFF; box-shadow:0 4px 12px rgba(0,0,0,0.04); }
.investor { font-size:20px; font-weight:700; color:#167DFF; margin-bottom:8px; }
.expl { text-align:justify; color:#0E3A75; font-size:15px; line-height:1.5; margin-bottom:8px; }
.web { font-size:13px; color:#167DFF; font-weight:600; margin-top: -6px; }
.section { font-size:24px; font-weight:700; color:#0E3A75; margin-top:18px; margin-bottom:8px; }
.progress-text { text-align:left; color:#0E3A75; margin-top:6px; }
table { border-collapse: collapse; width:100%; border:1.4px solid #AFCBFF; }
th { background:#167DFF; color:white; text-align:center; padding:9px; font-size:17px; }
td { background:#FFFFFF; color:#0E3A75; padding:10px; text-align:center; font-size:16px; }
td:first-child { text-align:left; padding-left:18px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Venture Investor Matching Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub'>IEC-powered precision investor recommendations</div>", unsafe_allow_html=True)

# Load engine (cached)
@st.cache_resource
def load_engine():
    dp = HybridDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_profiles()
    # default alpha (text weight). You can expose it in UI if desired.
    matcher = SuperInvestorMatcher(dp, openrouter_key=st.secrets["OPENROUTER_API_KEY"], alpha=0.7)
    return matcher

matcher = load_engine()

# Sidebar inputs (minimal essential fields + optional numeric fields)
with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal_size = st.number_input("Deal Size ($M)", value=50.0)
    growth = st.number_input("Revenue Growth YoY", value=0.35)
    description = st.text_area("Description (1-2 sentences)", value="AI-powered workflow automation platform.")
    st.markdown("---")
    st.write("Optional numeric signals (can improve matches):")
    revenue = st.number_input("Revenue LTM ($M)", value=50.0)
    # you can add more optional numeric inputs if desired

# spinner-only progress text
step_display = st.empty()

def progress_cb(text):
    # called by engine to show current step (left-justified)
    step_display.markdown(f"<div class='progress-text'>{text}</div>", unsafe_allow_html=True)

if st.button("Run Matching"):
    with st.spinner("Running investor matching pipeline…"):
        startup = {
            "industry": industry,
            "Deal_Size_M": deal_size,
            "Revenue_Growth_YoY": growth,
            "description": description,
            "Revenue_LTM_M": revenue,
            # ensure keys align to pipeline expectations
        }
        results = matcher.run_pipeline(startup, top_k=5, progress_cb=progress_cb)

    # left-justified done message that fades
    done = st.empty()
    done.markdown("<div class='progress-text' style='color:green;'>✔ Done</div>", unsafe_allow_html=True)
    time.sleep(2)
    done.empty()

    # show top 3 only
    top3 = results[:3]

    # table: Investor + Match Score (combined_score)
    df = pd.DataFrame(top3)[["investor", "combined_score"]].rename(columns={"investor":"Investor", "combined_score":"Match Score"})
    st.markdown("<div class='section'>Top 3 Matches</div>", unsafe_allow_html=True)
    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    st.markdown("<div class='section'>Reasoning</div>", unsafe_allow_html=True)
    for r in top3:
        st.markdown(f"""
            <div class="card">
                <div class="investor">{r['investor']}</div>
                <div class="expl">{r['explanation']}</div>
                <div class="web"><strong>Web Summary:</strong> {r['web']}</div>
            </div>
        """, unsafe_allow_html=True)
