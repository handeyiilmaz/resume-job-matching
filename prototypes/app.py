import streamlit as st
from utils import parse_uploaded_file, extract_candidate_fields, extract_job_fields
from matcher import load_and_fit_vectorizers, load_sbert_model, match_candidates_to_jobs

st.set_page_config(
    page_title="Resume–Job Matcher",
    layout="wide",
)

# ── Custom CSS (visual only) ───────────────────────────────────────────────

st.markdown(
    "<style>"
    ".block-container { padding-top: 2rem; }"
    ".card { background: #ffffff; border: 1px solid #e6e9ef; border-radius: 12px; "
    "padding: 20px 22px; margin-bottom: 18px; }"
    ".card-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 2px; }"
    ".card-subtitle { color: #6b7280; font-size: 0.9rem; margin-bottom: 14px; }"
    ".meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; margin-bottom: 14px; }"
    ".meta-label { color: #6b7280; font-size: 0.78rem; }"
    ".meta-value { font-weight: 600; font-size: 0.95rem; }"
    ".section-label { color: #1d4ed8; font-weight: 700; font-size: 0.95rem; margin: 10px 0 8px 0; }"
    ".tag { display: inline-block; background: #eef2ff; color: #3730a3; border-radius: 6px; "
    "padding: 3px 10px; margin: 3px 4px 3px 0; font-size: 0.82rem; font-weight: 500; }"
    ".results-header { background: #eefbf2; border: 1px solid #d6f0dd; border-radius: 12px; "
    "padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; "
    "margin-bottom: 4px; }"
    ".results-header-title { color: #15803d; font-weight: 700; font-size: 1.35rem; }"
    ".rank-badge { width: 34px; height: 34px; border-radius: 50%; display: flex; "
    "align-items: center; justify-content: center; font-weight: 700; color: white; font-size: 0.95rem; }"
    ".rank-1 { background: #d4a017; }"
    ".rank-2 { background: #9ca3af; }"
    ".rank-3 { background: #b45309; }"
    ".rank-other { background: #374151; }"
    ".match-name { font-weight: 700; color: #1d4ed8; font-size: 1.25rem; margin-bottom: 4px; }"
    ".match-sub { color: #6b7280; font-size: 1.05rem; margin-top: 0px; }"
    ".badge-strong { color: #15803d; font-weight: 800; font-size: 1.35rem; text-align: right; display: block; }"
    ".badge-good { color: #b45309; font-weight: 800; font-size: 1.35rem; text-align: right; display: block; }"
    ".badge-low { color: #2563eb; font-weight: 800; font-size: 1.35rem; text-align: right; display: block; }"
    ".score-text { color: #6b7280; font-size: 1.05rem; margin-bottom: 2px; }"
    "hr.thin { margin: 16px 0; border: none; border-top: 1px solid #eef0f3; }"
    "</style>",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────

st.title("Resume–Job Matching")
st.markdown(
    "Upload candidate profiles and job descriptions from the dataset. "
    "Candidates are matched to jobs within the same **position group** using "
    "**TF-IDF** and **SBERT** cosine similarity."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    st.markdown("**TF-IDF threshold**")
    threshold_tfidf = st.slider(
        "TF-IDF minimum score",
        min_value=0.05,
        max_value=0.80,
        value=0.026,
        step=0.05,
        help="TF-IDF cosine similarity scores are naturally lower due to vector sparsity.",
    )

    st.markdown("**SBERT threshold**")
    threshold_sbert = st.slider(
        "SBERT minimum score",
        min_value=0.05,
        max_value=0.95,
        value=0.349,
        step=0.05,
        help="SBERT scores are higher as they capture semantic meaning.",
    )

    st.divider()
    st.caption(
        "A match is shown if it passes **either** threshold.  \n\n"
        "**TF-IDF labels**  \n"
        "Strong ≥ 0.053 · Good ≥ 0.044 · Possible ≥ 0.026  \n\n"
        "**SBERT labels**  \n"
        "Strong ≥ 0.429 · Good ≥ 0.417 · Possible ≥ 0.349"
    )

# ── Load models (cached after first run) ───────────────────────────────────

vectorizers = load_and_fit_vectorizers()
sbert_model = load_sbert_model()

st.success("Ready for matching")
st.divider()

# ── File format instructions ────────────────────────────────────────────────

with st.expander("How to prepare input files", expanded=False):
    st.markdown("""
**Candidate file (.txt):**
Position: Python Developer
Moreinfo: I have 4 years of experience in Python...
position_group: Engineering/IT
English Level: upper
Experience Years: 4
Primary Keyword: Python

**Job description file (.txt):**
Position: Senior Python Developer
Long Description: We are looking for an experienced Python developer...
position_group: Engineering/IT
Company Name: TechCorp
Exp Years: 3y
English Level: upper
Primary Keyword: Python
""")

# ── Upload columns ───────────────────────────────────────────────────────────

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

# ── Matching ────────────────────────────────────────────────────────────────

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

    # ── Results ──────────────────────────────────────────────────────────

    st.subheader("Results")

    total_matches = sum(len(m) for m in results.values())
    st.caption(
        f"{len(candidates)} candidate(s) · {len(jobs)} job(s) · "
        f"{total_matches} match(es) found"
    )

    def badge_class(label):
        if "Strong" in label:
            return "badge-strong"
        elif "Good" in label:
            return "badge-good"
        return "badge-low"

    def rank_class(i):
        return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(i, "rank-other")

    job_to_matches = {}
    for candidate in candidates:
        cname = candidate["name"]
        for m in results.get(cname, []):
            job_to_matches.setdefault(m["job"], []).append({**m, "candidate": candidate})

    for job in jobs:
        jtitle = job["title"]
        jgroup = job["position_group"]
        matches = job_to_matches.get(jtitle, [])

        col_left, col_right = st.columns([1, 2.2], gap="large")

        # ── Left: job detail card ────────────────────────────────────────
        with col_left:
            skills_html = ""
            if job.get("primary_keyword"):
                skills_html = "".join(
                    f'<span class="tag">{kw.strip()}</span>'
                    for kw in job["primary_keyword"].split(",") if kw.strip()
                )

            raw_desc = job.get("long_description_raw", "")
            summary_lines = [
                line.strip() for line in raw_desc.splitlines()
                if line.strip() and len(line.strip()) > 10
            ][:5]
            summary_html = "".join(f"<li>{line}</li>" for line in summary_lines)

            card_html = (
                '<div class="card">'
                f'<div class="card-title">{jtitle}</div>'
                f'<div class="card-subtitle">{job.get("company", "")}</div>'
                '<div class="meta-grid">'
                '<div><div class="meta-label">Group</div>'
                f'<div class="meta-value">{jgroup}</div></div>'
                '<div><div class="meta-label">Experience</div>'
                f'<div class="meta-value">{job.get("exp_years", "—")}</div></div>'
                '<div><div class="meta-label">English</div>'
                f'<div class="meta-value">{job.get("english_level", "—")}</div></div>'
                '<div><div class="meta-label">Keyword</div>'
                f'<div class="meta-value">{job.get("primary_keyword", "—")}</div></div>'
                '</div>'
                '<div class="section-label">Job Description (Summary)</div>'
                f'<ul>{summary_html if summary_html else "<li>—</li>"}</ul>'
                '<div class="section-label">Required Skills</div>'
                f'<div>{skills_html if skills_html else "<span class=\'meta-value\'>—</span>"}</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

        # ── Right: matching candidates ───────────────────────────────────
        with col_right:
            header_html = (
                '<div class="results-header">'
                f'<div class="results-header-title">Top {len(matches)} Matching Candidates</div>'
                '</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)

            if not matches:
                group_candidates = [c for c in candidates if c["position_group"] == jgroup]
                if not group_candidates:
                    st.info(
                        f"No uploaded candidates belong to the **{jgroup}** group. "
                        "Matching is group-aware, jobs are only compared "
                        "to candidates in the same position group."
                    )
                else:
                    st.info(
                        "No matches found above the threshold. "
                        "Try lowering the thresholds in the sidebar."
                    )
            else:
                label_weights = {
                    "Strong match": 3,
                    "Good match": 2,
                    "Possible match": 1,
                    "No match": 0
                }

                for i, m in enumerate(matches, start=1):
                    ts = m["tfidf_score"]
                    ss = m["sbert_score"]
                    tl = m["tfidf_label"]
                    sl = m["sbert_label"]
                    cand = m["candidate"]

                    # Karşılaştırma ve yüksek etiketi seçme
                    w_tl = label_weights.get(tl, 0)
                    w_sl = label_weights.get(sl, 0)

                    if w_tl > w_sl:
                        display_label = tl
                        tfidf_suffix = ""
                        sbert_suffix = f" ({sl})"
                    elif w_sl > w_tl:
                        display_label = sl
                        tfidf_suffix = f" ({tl})"
                        sbert_suffix = ""
                    else:
                        display_label = tl
                        tfidf_suffix = ""
                        sbert_suffix = ""

                    # 4 kolon: [Sıra], [Aday Bilgisi], [Skor/Barlar], [Match Rozeti]
                    r1, r2, r3, r4 = st.columns([0.4, 2.8, 3.0, 2.4])

                    with r1:
                        st.markdown(
                            f'<div class="rank-badge {rank_class(i)}">{i}</div>',
                            unsafe_allow_html=True,
                        )
                    
                    with r2:
                        st.markdown(f'<div class="match-name">{cand["name"]}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="match-sub">Exp: {cand.get("experience_years", "—")} yrs · '
                            f'{cand.get("english_level", "—")}</div>',
                            unsafe_allow_html=True,
                        )

                    with r3:
                        # Değerleri küçülttük ve ayrı dar bir sütuna aldık
                        st.markdown(f'<div class="score-text">TF-IDF: {ts:.3f}{tfidf_suffix}</div>', unsafe_allow_html=True)
                        st.progress(min(ts / 0.5, 1.0))
                        
                        st.markdown(f'<div class="score-text" style="margin-top:8px;">SBERT: {ss:.3f}{sbert_suffix}</div>', unsafe_allow_html=True)
                        st.progress(min(ss, 1.0))

                    with r4:
                        # Rozeti büyük ve en sağa yaslı yapıyoruz, padding ile hizasını sabitliyoruz
                        st.markdown(
                            f'<div style="padding-top: 4px;">'
                            f'<span class="{badge_class(display_label)}">{display_label}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

        st.divider()
