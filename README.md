# Resume–Job Matching Prototype

A prototype resume–job matching system developed as part of the paper 
*Group-Aware Resume–Job Matching: A Comparative Study of Text Representation 
Methods on a Cleaned Large-Scale Recruitment Dataset* (Yılmaz et al., 2026).

The system matches candidate profiles to job descriptions using TF-IDF and 
SBERT cosine similarity with group-aware filtering. Candidates are only 
compared to jobs within the same position group.

## Files

- `app.py` — Streamlit web application
- `matcher.py` — TF-IDF and SBERT matching logic
- `utils.py` — File parsing and text cleaning utilities
- `requirements.txt` — Required Python packages

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datasets

The application loads the following cleaned datasets from HuggingFace at startup:

- Job descriptions: https://huggingface.co/datasets/handeyilmaz/job-descriptions-ready-to-use
- Candidate profiles: https://huggingface.co/datasets/handeyilmaz/candidate-profiles-ready-to-use

## How to Use

1. Upload candidate profile files (.txt or .csv)
2. Upload job description files (.txt or .csv)
3. Adjust TF-IDF and SBERT thresholds in the sidebar
4. Click "Run Matching"

See the "How to prepare input files" section in the app for file format details.
