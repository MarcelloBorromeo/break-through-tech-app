import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph
import time

st.set_page_config(page_title="Investor Matcher", layout="wide", page_icon="🚀")
st.title("Venture Investor Matching Engine")

# ---------- Load engine once ----------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    # pick key from OpenRouter secrets
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "")
    graph = InvestorMatchingGraph(dp, openrouter_key=openrouter_key)
    return graph

engine = load_engine()

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal_size = st.number_input("Deal Size ($M)", min_value=0.0, value=50.0, step=5.0)
    growth = st.number_input("Revenue Growth YoY", min_value=-1.0, value=0.35, step=0.05)
    description = st.text_area("Startup Description", "AI-powered B2B SaaS platform for workflow automation.", height=150)
    st.markdown("---")
    st.write("Tip: keep description concise (1–2 short sentences).")

# ---------- Progress UI elements ----------
progress_bar = st.progress(0)
status_text = st.empty()
step_log = st.empty()  # will show recent step messages

# simple step logging helper that prints top-N recent messages
_log_history = []
def log_step(step: str, detail: str, percent: int):
    global _log_history
    _log_history.append(f"{step}: {detail} ({percent}%)")
    # keep only last 6 messages
    _log_history = _log_history[-6:]
    progress_bar.progress(min(max(percent, 0), 100))
    status_text.markdown(f"**Status:** {step} — {detail}")
    step_log.write("\n".join(_log_history))

# ---------- Run button ----------
if st.button("🔍 Find Matching Investors"):
    startup = {
        "industry": industry,
        "deal_size_m": deal_size,
        "revenue_growth_yoy": growth,
        "description": description
    }

    # run pipeline and update UI via callback
    with st.spinner("Running investor matching pipeline..."):
        try:
            results = engine.run_pipeline(startup_profile=startup, cb=log_step)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            # make sure progress bar shows failure
            progress_bar.progress(0)
            st.stop()

    # ---------- Display results ----------
    if not results:
        st.warning("No results returned.")
    else:
        df = pd.DataFrame(results)
        st.subheader("🏆 Top Investor Matches")
        st.dataframe(
            df[["investor_name", "final_score", "embedding_likelihood", "llm_adjustment"]],
            use_container_width=True
        )

        st.subheader("🧠 LLM Explanations + Web Context")
        for r in results:
            st.markdown(f"### **{r['investor_name']} — Score {r['final_score']}**")
            # explanation kept concise (one sentence)
            st.write("**Reasoning:**", r.get("explanation", "—"))
            st.write("**Web Context:**")
            st.write(r.get("web_context", "(none)"))
            st.write("---")

    # final UI state
    progress_bar.progress(100)
    status_text.markdown("**Status:** Complete — results ready")
