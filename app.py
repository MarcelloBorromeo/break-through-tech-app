import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph

st.set_page_config(page_title="Investor Matching Engine", layout="wide")
st.title("🚀 Venture Investor Matching Engine")

@st.cache_resource
def load_engine():
    # CSV filename with [Complete]
    data = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    data.generate_embeddings()
    graph = InvestorMatchingGraph(data)
    return graph

engine = load_engine()

with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal_size = st.number_input("Deal Size ($M)", 80.0)
    growth = st.number_input("Revenue Growth YoY", 0.35)
    description = st.text_area(
        "Startup Description",
        "B2B SaaS platform for enterprise automation using AI.",
        height=150
    )

if st.button("🔍 Find Investors"):
    startup = {
        "industry": industry,
        "deal_size_m": deal_size,
        "revenue_growth_yoy": growth,
        "description": description
    }

    with st.spinner("Running matching engine..."):
        results = engine.run_pipeline(startup)
        df = pd.DataFrame(results)

    st.subheader("Top Matches")
    st.dataframe(df[['investor_name', 'final_score', 'embedding_likelihood', 'llm_adjustment']])

    st.subheader("LLM Explanations")
    for r in results:
        st.markdown(f"### {r['investor_name']} — Score {r['final_score']}")
        st.write("**Reasoning:**", r["explanation"])
        st.write("**Web Context:**", r["web_context"])
        st.write("---")
