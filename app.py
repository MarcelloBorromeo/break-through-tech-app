import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph

st.set_page_config(layout="wide")

# Centered title (HTML)
st.markdown(
    "<h1 style='text-align:center;'>Venture Investor Matching Engine</h1>",
    unsafe_allow_html=True
)

# --------------------------------------------
# Load engine
# --------------------------------------------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    key = st.secrets["OPENROUTER_API_KEY"]
    return InvestorMatchingGraph(dp, key)

engine = load_engine()

# --------------------------------------------
# Sidebar Inputs
# --------------------------------------------
with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal = st.number_input("Deal Size ($M)", 50.0)
    growth = st.number_input("Growth YoY", 0.35)
    desc = st.text_area("Description", "AI workflow automation platform.")

# --------------------------------------------
# Spinner + inline step text
# --------------------------------------------
if st.button("Run Matching"):
    step_display = st.empty()

    def update_step(text):
        step_display.markdown(f"**{text}**")

    with st.spinner("Running investor matching pipeline..."):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc
            },
            progress_callback=update_step
        )

    # --------------------------------------------
    # Display Results
    # --------------------------------------------
    df = pd.DataFrame(results)

    st.subheader("Top 3 Investor Matches")
    st.dataframe(
        df[["investor", "final"]].rename(columns={
            "investor": "Investor",
            "final": "Match Score"
        }),
        use_container_width=True
    )

    st.subheader("Reasoning")
    for r in results:
        st.markdown(f"### {r['investor']}")
        st.write(r["explanation"])
        st.write(f"**Web Summary:** {r['web']}")
        st.markdown("---")
