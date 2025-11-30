import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
from typing import Dict, List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer


# ================================================================
# OPENROUTER CHAT COMPLETIONS FUNCTION
# ================================================================
def or_chat(model, messages, api_key, max_tokens=400):
    """
    Calls OpenRouter's API using OpenAI-style chat/completions.
    Supports Claude 3.5 Sonnet via anthropic/claude-3.5-sonnet.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Investor Matcher App",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        raise Exception(f"OpenRouter Error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ================================================================
# DATA PIPELINE
# ================================================================
class InvestorDataPipeline:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.investor_embeddings = {}
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.investor_fingerprint_vectors = {}
        self._clean_data()

    def _clean_data(self):
        self.df["Investor_Name"] = (
            self.df["Investor_Name"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )
        self.df["Business_Overview"].fillna("", inplace=True)
        self.df["Competitive Analysis "].fillna("", inplace=True)

    def create_investor_context(self, investor_name: str) -> str:
        deals = self.df[self.df["Investor_Name"] == investor_name]
        companies = deals["Portfolio_Company"].unique()
        industries = deals["Industry"].unique()

        ctx = f"Investor: {investor_name}\n"
        ctx += f"Portfolio Companies: {', '.join(companies[:10])}\n"
        ctx += f"Industries: {', '.join(industries)}\n"
        ctx += f"Deals: {len(deals)}\n\nSample Profiles:\n"

        for _, row in deals.head(3).iterrows():
            ctx += f"- {row['Portfolio_Company']}: {row['Business_Overview'][:200]}...\n"

        return ctx

    def create_investor_fingerprint(self, investor_name: str) -> str:
        deals = self.df[self.df["Investor_Name"] == investor_name]
        parts = []

        parts.append(f"Portfolio: {', '.join(deals['Portfolio_Company'].unique())}")
        parts.append(f"Industries: {', '.join(deals['Industry'].unique())}")

        for t in deals["Business_Overview"].dropna().unique():
            parts.append(t)
        for t in deals["Competitive Analysis "].dropna().unique():
            parts.append(t)

        deal_sizes = deals["Deal_Size_M"].dropna()
        if len(deal_sizes) > 0:
            parts.append(f"Typical deal size: ${deal_sizes.mean():.1f}M")

        return " ".join(" ".join(parts).split())

    def generate_embeddings(self):
        investors = self.df["Investor_Name"].unique()
        fingerprints = {}

        for inv in investors:
            fp = self.create_investor_fingerprint(inv)
            ctx = self.create_investor_context(inv)
            fingerprints[inv] = fp
            self.investor_embeddings[inv] = {"fingerprint": fp, "context": ctx}

        self.vectorizer.fit(fingerprints.values())

        for inv in investors:
            vec = self.vectorizer.transform([fingerprints[inv]]).toarray()[0]
            self.investor_fingerprint_vectors[inv] = vec

    def calculate_embedding_similarity(self, startup_vector, investor_name):
        inv_vec = self.investor_fingerprint_vectors[investor_name]
        dot = np.dot(startup_vector, inv_vec)

        norm_s = np.linalg.norm(startup_vector)
        norm_i = np.linalg.norm(inv_vec)

        if norm_s == 0 or norm_i == 0:
            return 0.0

        return round((dot / (norm_s * norm_i)) * 100, 2)


# ================================================================
# MATCHING PIPELINE
# ================================================================
class GraphState(TypedDict):
    startup_profile: Dict
    all_investors: List[str]
    candidate_investors: List[Dict]
    ranked_results: List[Dict]


class InvestorMatchingGraph:
    def __init__(self, data_pipeline: InvestorDataPipeline):
        self.data = data_pipeline

    # ------------------------------------------
    def fetch_investor_web_context(self, investor_name: str) -> str:
        prompt = f"""
Find public info about the investor '{investor_name}'.
Summarize their focus, thesis, stage, check size, and notable deals.
"""

        try:
            result = or_chat(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": prompt}],
                api_key=st.secrets["OPENROUTER_API_KEY"],
                max_tokens=200
            )
            return result
        except Exception as e:
            return f"(Web context unavailable: {e})"

    # ------------------------------------------
    def retrieve_candidates(self, state):
        s = state["startup_profile"]
        txt = f"{s['industry']} {s['description']}"
        vec = self.data.vectorizer.transform([txt]).toarray()[0]

        scores = []
        for inv in state["all_investors"]:
            sim = self.data.calculate_embedding_similarity(vec, inv)
            scores.append({"investor_name": inv, "embedding_likelihood": sim})

        scores.sort(key=lambda x: x["embedding_likelihood"], reverse=True)
        state["candidate_investors"] = scores[:3]
        return state

    # ------------------------------------------
    def reason_about_fit(self, state):
        startup = state["startup_profile"]

        for cand in state["candidate_investors"]:
            inv = cand["investor_name"]
            context = self.data.investor_embeddings[inv]["context"]
            web = self.fetch_investor_web_context(inv)

            prompt = f"""
Evaluate investor–startup fit.

STARTUP:
Industry: {startup['industry']}
Deal Size: {startup['deal_size_m']}
Growth: {startup['revenue_growth_yoy']}
Description: {startup['description']}

INVESTOR (Dataset):
{context}

INVESTOR (Web):
{web}

Embedding Score: {cand['embedding_likelihood']}

Return ONLY JSON:
{{
  "adjustment": -20 to 20,
  "explanation": "2–3 sentences"
}}
"""

            try:
                response = or_chat(
                    model="anthropic/claude-3.5-sonnet",
                    messages=[{"role": "user", "content": prompt}],
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                    max_tokens=400
                )

                js = json.loads(response)
                cand["llm_adjustment"] = js.get("adjustment", 0)
                cand["explanation"] = js.get("explanation", "")
                cand["web_context"] = web

            except Exception as e:
                cand["llm_adjustment"] = 0
                cand["explanation"] = f"LLM error: {e}"
                cand["web_context"] = web

        return state

    # ------------------------------------------
    def rank_investors(self, state):
        for cand in state["candidate_investors"]:
            score = cand["embedding_likelihood"] + cand["llm_adjustment"]
            cand["final_score"] = max(0, min(100, round(score, 2)))

        state["ranked_results"] = sorted(
            state["candidate_investors"],
            key=lambda x: x["final_score"],
            reverse=True
        )
        return state

    # ------------------------------------------
    def run_pipeline(self, startup_profile):
        state = GraphState(
            startup_profile=startup_profile,
            all_investors=list(self.data.investor_embeddings.keys()),
            candidate_investors=[],
            ranked_results=[]
        )

        state = self.retrieve_candidates(state)
        state = self.reason_about_fit(state)
        state = self.rank_investors(state)

        return state["ranked_results"]
