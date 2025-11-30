# engine.py
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Callable, Optional, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity


ProgressCallback = Optional[Callable[[str], None]]


def or_chat(model: str, messages: List[Dict], api_key: str, max_tokens: int = 300):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"OpenRouter Error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


class HybridDataPipeline:
    """
    Builds:
    - TF-IDF text fingerprints
    - Scaled numeric fingerprints
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.text_cols = ["Portfolio_Company", "Industry", "Business_Overview", "Competitive Analysis "]
        self.text_cols = [c for c in self.text_cols if c in self.df.columns]

        self.numeric_cols = ["Deal_Size_M", "Revenue_LTM_M", "Revenue_Growth_YoY", "Gross_Margin"]
        self.numeric_cols = [c for c in self.numeric_cols if c in self.df.columns]

        self.cat_cols = ["Industry"]

        self.vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2))
        self.investor_text_vectors = {}
        self.investor_text_fingerprints = {}

        self.preprocessor = None
        self.investor_numeric_profiles = None

        self._clean_names()

    def _clean_names(self):
        self.df["Investor_Name"] = (
            self.df["Investor_Name"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

    # ---------------- TEXT FINGERPRINTS ---------------------
    def _build_text_profiles(self):
        investors = self.df["Investor_Name"].unique()
        fps = {}

        for inv in investors:
            rows = self.df[self.df["Investor_Name"] == inv]
            parts = []
            for col in self.text_cols:
                parts.append(" ".join(rows[col].dropna().astype(str).unique()))
            fp = " ".join(parts).strip()
            fps[inv] = fp
            self.investor_text_fingerprints[inv] = fp

        self.vectorizer.fit(fps.values())

        for inv in investors:
            self.investor_text_vectors[inv] = self.vectorizer.transform([fps[inv]]).toarray()[0]

    # ---------------- NUMERIC PROFILES ----------------------
    def _build_numeric_profiles(self):
        numeric = self.numeric_cols.copy()
        for col in numeric:
            self.df[f"{col}_missing"] = self.df[col].isna().astype(int)
        numeric_with_flags = numeric + [f"{c}_missing" for c in numeric]

        # fill numerics
        medians = self.df[numeric].median()
        self.df[numeric] = self.df[numeric].fillna(medians)

        num_transform = ("num", StandardScaler(), numeric_with_flags)
        cat_transform = ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), self.cat_cols)

        self.preprocessor = ColumnTransformer(
            transformers=[num_transform, cat_transform],
            remainder="drop",
            sparse_threshold=0
        )

        fit_df = self.df[numeric_with_flags + self.cat_cols]
        self.preprocessor.fit(fit_df)

        transformed = self.preprocessor.transform(fit_df)
        investors = self.df["Investor_Name"].values

        # Build investor numeric profiles by averaging
        df_numeric = pd.DataFrame(transformed)
        df_numeric["Investor_Name"] = investors
        self.investor_numeric_profiles = df_numeric.groupby("Investor_Name").mean()

    # ---------------- PUBLIC BUILD ----------------------
    def generate_profiles(self):
        self._build_text_profiles()
        self._build_numeric_profiles()

    # ---------------- GETTERS ----------------------
    def get_investor_list(self):
        return list(self.investor_numeric_profiles.index)

    def get_text_matrix(self):
        return np.vstack([self.investor_text_vectors[i] for i in self.get_investor_list()])

    def get_numeric_matrix(self):
        return self.investor_numeric_profiles.values

    def get_investor_context(self, inv: str) -> str:
        return self.investor_text_fingerprints.get(inv, "")


class SuperInvestorMatcher:
    """
    Fusion model:
    - TF-IDF text similarity
    - Numeric scaled similarity
    - LLM reasoning
    """

    def __init__(self, pipeline: HybridDataPipeline, openrouter_key: str, alpha: float = 0.7):
        self.data = pipeline
        self.key = openrouter_key
        self.alpha = alpha
        self.model = "anthropic/claude-3.5-sonnet"

        self.investors = self.data.get_investor_list()
        self.text_matrix = self.data.get_text_matrix()
        self.numeric_matrix = self.data.get_numeric_matrix()

    def _cos_sim(self, vec, mat):
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        sims = cosine_similarity(vec, mat)[0]
        return np.nan_to_num(sims)

    def retrieve(self, startup, top_k=5):
        # TEXT
        startup_text = f"{startup.get('industry','')} {startup.get('description','')}"
        text_vec = self.data.vectorizer.transform([startup_text]).toarray()[0]
        text_sim = self._cos_sim(text_vec, self.text_matrix) * 100

        # NUMERIC
        r = {}
        medians = self.data.df[self.data.numeric_cols].median()
        for col in self.data.numeric_cols:
            val = startup.get(col)
            if val is None:
                r[col] = medians[col]
                r[f"{col}_missing"] = 1
            else:
                r[col] = val
                r[f"{col}_missing"] = 0
        for c in self.data.cat_cols:
            r[c] = startup.get(c, startup.get("industry", ""))

        row_df = pd.DataFrame([r])
        num_vec = self.data.preprocessor.transform(row_df)[0]
        num_sim = self._cos_sim(num_vec, self.numeric_matrix) * 100

        fused = self.alpha * text_sim + (1 - self.alpha) * num_sim

        results = []
        for i, inv in enumerate(self.investors):
            results.append({
                "investor": inv,
                "text_score": round(text_sim[i], 2),
                "numeric_score": round(num_sim[i], 2),
                "combined_score": round(fused[i], 2),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]

    def fetch_web(self, investor):
        sys = "Return 2 short bullet points about this VC's stage & focus. <=10 words each."
        usr = f"Investor: {investor}"
        try:
            out = or_chat(self.model, [{"role":"system","content":sys},{"role":"user","content":usr}], self.key, max_tokens=120)
            return out.strip()
        except:
            return "(web unavailable)"

    def reason(self, startup, cand, dataset_ctx, web):
        sys = (
            "You are a concise VC analyst. Use embedding scores, numeric signals, dataset context, and web summary "
            "to write one paragraph (max 4 sentences) explaining fit."
        )
        usr = (
            f"STARTUP:\nIndustry:{startup.get('industry')}\nDeal:${startup.get('Deal_Size_M')}M\n"
            f"Growth:{startup.get('Revenue_Growth_YoY')}\nDesc:{startup.get('description')}\n\n"
            f"SCORES:\ntext={cand['text_score']}\nnumeric={cand['numeric_score']}\ncombined={cand['combined_score']}\n\n"
            f"DATASET CONTEXT:\n{dataset_ctx}\n\nWEB SUMMARY:\n{web}"
        )
        try:
            out = or_chat(self.model, [{"role":"system","content":sys},{"role":"user","content":usr}], self.key, max_tokens=220)
        except Exception as e:
            out = f"(LLM error: {e})"
        cand["explanation"] = out.strip()
        cand["web"] = web
        return cand

    def run_pipeline(self, startup, top_k=5, progress_cb:ProgressCallback=None):
        if progress_cb: progress_cb("Computing similarities…")
        cands = self.retrieve(startup, top_k=top_k)

        results = []
        for c in cands:
            inv = c["investor"]
            if progress_cb: progress_cb(f"Fetching web: {inv}…")
            web = self.fetch_web(inv)
            ds = self.data.get_investor_context(inv)
            if progress_cb: progress_cb(f"Reasoning: {inv}…")
            out = self.reason(startup, c, ds, web)
            results.append(out)

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        if progress_cb: progress_cb("Done.")
        return results
