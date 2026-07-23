import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import streamlit as st

JOB_DATASET = "handeyilmaz/job-descriptions-ready-to-use"
CANDIDATE_DATASET = "handeyilmaz/candidate-profiles-ready-to-use"


def tfidf_match_label(score: float) -> str:
    if score >= 0.053:
        return "Strong match"
    elif score >= 0.044:
        return "Good match"
    elif score >= 0.026:
        return "Possible match"
    else:
        return "No match"


def sbert_match_label(score: float) -> str:
    if score >= 0.429:
        return "Strong match"
    elif score >= 0.417:
        return "Good match"
    elif score >= 0.349:
        return "Possible match"
    else:
        return "No match"


@st.cache_resource(show_spinner="Loading datasets and fitting TF-IDF vectorizers...")
def load_and_fit_vectorizers():
    
    from utils import clean_text

    # Load full job descriptions dataset 
    job_ds = load_dataset(JOB_DATASET, split="train")
    job_df = job_ds.to_pandas()

    # Load full candidate profiles dataset 
    cand_ds = load_dataset(CANDIDATE_DATASET, split="train")
    cand_df = cand_ds.to_pandas()

    vectorizers = {}
    groups = job_df["position_group"].dropna().unique().tolist()

    for group in groups:
        job_texts = job_df[job_df["position_group"] == group]["Long Description"].dropna().tolist()
        job_texts = [clean_text(t) for t in job_texts]

        cand_texts = cand_df[cand_df["position_group"] == group]["Moreinfo"].dropna().tolist()
        cand_texts = [clean_text(t) for t in cand_texts]

        all_texts = [t for t in job_texts + cand_texts if len(t) > 20]

        if len(all_texts) < 5:
            continue

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95,
            max_features=10000,
            ngram_range=(1, 1),
        )
        vectorizer.fit(all_texts)
        vectorizers[group] = vectorizer

    return vectorizers


@st.cache_resource(show_spinner="Loading SBERT model...")
def load_sbert_model():
    """Load the SBERT model used in the thesis (all-MiniLM-L6-v2)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def match_candidates_to_jobs(
    candidates: list,
    jobs: list,
    vectorizers: dict,
    threshold_tfidf: float = 0.10,
    threshold_sbert: float = 0.40,
    use_sbert: bool = True,
) -> dict:
   
    from utils import clean_text

    # Load SBERT if needed
    sbert_model = load_sbert_model() if use_sbert else None

    results = {}

    for candidate in candidates:
        cname = candidate["name"]
        cgroup = candidate["position_group"]
        ctext = clean_text(candidate["text"])

        results[cname] = []

        # Only compare against jobs in the same position group
        group_jobs = [j for j in jobs if j["position_group"] == cgroup]

        if not group_jobs:
            continue

        job_texts = [clean_text(j["text"]) for j in group_jobs]

        # ── TF-IDF scoring ────────────────────────────────────────────────────
        tfidf_scores = [0.0] * len(group_jobs)
        vectorizer = vectorizers.get(cgroup)
        if vectorizer:
            try:
                job_vectors = vectorizer.transform(job_texts)
                cand_vector = vectorizer.transform([ctext])
                tfidf_scores = cosine_similarity(cand_vector, job_vectors)[0].tolist()
            except Exception:
                pass

        # ── SBERT scoring ─────────────────────────────────────────────────────
        sbert_scores = [0.0] * len(group_jobs)
        if use_sbert and sbert_model:
            try:
                all_texts = [ctext] + job_texts
                embeddings = sbert_model.encode(
                    all_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                cand_emb = embeddings[0:1]
                job_embs = embeddings[1:]
                sbert_scores = cosine_similarity(cand_emb, job_embs)[0].tolist()
            except Exception:
                pass

        # ── Collect matches ───────────────────────────────────────────────────
        matches = []
        for idx, job in enumerate(group_jobs):
            ts = float(tfidf_scores[idx])
            ss = float(sbert_scores[idx])

            # Show match if it passes either threshold
            if ts >= threshold_tfidf or ss >= threshold_sbert:
                matches.append({
                    "job": job["title"],
                    "company": job.get("company", ""),
                    "group": cgroup,
                    "tfidf_score": ts,
                    "sbert_score": ss,
                    "tfidf_label": tfidf_match_label(ts),
                    "sbert_label": sbert_match_label(ss),
                })

        # Sort by TF-IDF score 
        matches.sort(key=lambda x: x["tfidf_score"], reverse=True)
        results[cname] = matches

    return results