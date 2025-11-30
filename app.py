import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph

st.set_page_config(
    page_title="Investor Matching Engine",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀 Venture Investor Matching Engine")

# --------------------------
# Load pipeline once
# --------------------------
@st.cache_resource
def load_engine():
    data = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    data.generate_embeddings()
    graph = InvestorMatchingGraph(data)
    return graph

engine = load_engine()

# --------------------------
# Sidebar Inputs
# --------------------------
with st.sidebar:
    st.header("Startup Profile")

    industry = st.text_input("Industry", "Software")

    deal_size = st.number_input(
        "Deal Size ($M)",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=5.0
    )

    growth = st.number_input(
        "Revenue Growth YoY",
        min_value=-1.0,
        max_value=10.0,
        value=0.35,
        step=0.05
    )

    description = st.text_area(
        "Startup Description",
        "AI-powered B2B SaaS platform for workflow automation.",
        height=150
    )


# --------------------------
# Run Button
# --------------------------
if st.button("🔍 Find Matching Investors"):
    startup = {
        "industry": industry,
        "deal_size_m": deal_size,
        "revenue_growth_yoy": growth,
        "description": description
    }

    with st.spinner("Running investor matching pipeline..."):
        results = engine.run_pipeline(startup)

    df = pd.DataFrame(results)

    # --------------------------
    # Display Results
    # --------------------------
    st.subheader("🏆 Top Investor Matches")
    st.dataframe(
        df[["investor_name", "final_score", "embedding_likelihood", "llm_adjustment"]],
        use_container_width=True
    )

    st.subheader("🧠 LLM Explanations + Web Context")

    for r in results:
        st.markdown(f"### **{r['investor_name']} — Score {r['final_score']}**")
        st.write("**Reasoning:**", r["explanation"])
        st.write("**Web Context:**", r["web_context"])
        st.write("---")
