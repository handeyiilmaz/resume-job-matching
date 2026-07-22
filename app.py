import streamlit as st
from utils import parse_uploaded_file, extract_candidate_fields, extract_job_fields
from matcher import load_and_fit_vectorizers, load_sbert_model, match_candidates_to_jobs

st.set_page_config(
    page_title="Resume–Job Matcher",
    
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Resume–Job Matching")
st.markdown(
    "Upload candidate profiles and job descriptions from the dataset. "
    "Candidates are matched to jobs within the same **position group** using "
    "**TF-IDF** and **SBERT** cosine similarity."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    st.markdown("**TF-IDF threshold**")
    threshold_tfidf = st.slider(
        "TF-IDF minimum score",
        min_value=0.05,
        max_value=0.80,
        value=0.10,
        step=0.05,
        help="TF-IDF cosine similarity scores are naturally lower due to vector sparsity.",
    )

    st.markdown("**SBERT threshold**")
    threshold_sbert = st.slider(
        "SBERT minimum score",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.05,
        help="SBERT scores are higher as they capture semantic meaning.",
    )

    st.divider()
    st.caption(
        "A match is shown if it passes **either** threshold.  \n\n"
        "**TF-IDF labels**  \n"
        "Strong ≥ 0.35 · Good ≥ 0.20 · Possible ≥ threshold  \n\n"
        "**SBERT labels**  \n"
        "Strong ≥ 0.65 · Good ≥ 0.50 · Possible ≥ threshold"
    )
    

# ── Load models (cached after first run) ─────────────────────────────────────

vectorizers = load_and_fit_vectorizers()
sbert_model = load_sbert_model()

st.success(
    f"Ready for matching",
    
)
st.divider()

# ── File format instructions ──────────────────────────────────────────────────

with st.expander("How to prepare input files", expanded=False):
    st.markdown("""
**Candidate file (.txt):**
```
Position: Python Developer
Moreinfo: I have 4 years of experience in Python...
position_group: Engineering/IT
English Level: upper
Experience Years: 4
Primary Keyword: Python
```

**Job description file (.txt):**
```
Position: Senior Python Developer
Long Description: We are looking for an experienced Python developer...
position_group: Engineering/IT
Company Name: TechCorp
Exp Years: 3y
English Level: upper
Primary Keyword: Python
```


""")

# ── Upload columns ────────────────────────────────────────────────────────────

col_cv, col_jd = st.columns(2)

with col_cv:
    st.subheader("Candidate Profiles")
    cv_files = st.file_uploader(
        "Upload candidate files (.txt or .csv)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        key="cv_files",
    )

with col_jd:
    st.subheader("Job Descriptions")
    jd_files = st.file_uploader(
        "Upload job description files (.txt or .csv)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        key="jd_files",
    )

st.divider()
run = st.button("Run Matching", type="primary")

# ── Matching ──────────────────────────────────────────────────────────────────

if run:
    candidates = []
    jobs = []

    for f in cv_files:
        row = parse_uploaded_file(f.read(), f.name)
        fields = extract_candidate_fields(row)
        if not fields["text"]:
            st.warning(f"{f.name}: could not find 'Moreinfo' field.")
        else:
            candidates.append(fields)

    for f in jd_files:
        row = parse_uploaded_file(f.read(), f.name)
        fields = extract_job_fields(row)
        if not fields["text"]:
            st.warning(f"{f.name}: could not find 'Long Description' field.")
        else:
            jobs.append(fields)

    if not candidates:
        st.error("Please upload at least one candidate profile file.")
        st.stop()
    if not jobs:
        st.error("Please upload at least one job description file.")
        st.stop()

    with st.spinner("Running TF-IDF and SBERT matching..."):
        results = match_candidates_to_jobs(
            candidates, jobs, vectorizers,
            threshold_tfidf=threshold_tfidf,
            threshold_sbert=threshold_sbert,
            use_sbert=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────

    st.subheader("Results")

    total_matches = sum(len(m) for m in results.values())
    st.caption(
        f"{len(candidates)} candidate(s) · {len(jobs)} job(s) · "
        f"{total_matches} match(es) found"
    )

    for candidate in candidates:
        cname = candidate["name"]
        cgroup = candidate["position_group"]
        matches = results.get(cname, [])

        with st.expander(
            f"**{cname}** | Group: {cgroup} | {len(matches)} match(es)",
            expanded=True,
        ):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption(f"English: {candidate.get('english_level', '—')}")
            with col_b:
                st.caption(f"Experience: {candidate.get('experience_years', '—')} years")
            with col_c:
                st.caption(f"Keyword: {candidate.get('primary_keyword', '—')}")

            st.markdown("---")

            if not matches:
                group_jobs = [j for j in jobs if j["position_group"] == cgroup]
                if not group_jobs:
                    st.info(
                        f"No uploaded jobs belong to the **{cgroup}** group. "
                        "Matching is group-aware, candidates are only compared "
                        "to jobs in the same position group."
                    )
                else:
                    st.info(
                        "No matches found above the threshold. "
                        "Try lowering the thresholds in the sidebar."
                    )
            else:
                for m in matches:
                    ts = m["tfidf_score"]
                    ss = m["sbert_score"]
                    tl = m["tfidf_label"]
                    sl = m["sbert_label"]

                    # Badge colors
                    def badge_color(label):
                        if label == "Strong match":
                            return "green"
                        elif label == "Good match":
                            return "orange"
                        return "blue"

                    st.markdown(f"**{m['job']}**")
                    if m.get("company"):
                        st.caption(m["company"])

                    # TF-IDF row
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.markdown("TF-IDF")
                        st.progress(min(ts / 0.5, 1.0))
                    with col2:
                        st.markdown(f":{badge_color(tl)}[{tl}]")
                    with col3:
                        st.metric("Score", f"{ts:.3f}")

                    # SBERT row
                    col4, col5, col6 = st.columns([2, 2, 1])
                    with col4:
                        st.markdown("SBERT")
                        st.progress(min(ss, 1.0))
                    with col5:
                        st.markdown(f":{badge_color(sl)}[{sl}]")
                    with col6:
                        st.metric("Score", f"{ss:.3f}")

                    st.divider()