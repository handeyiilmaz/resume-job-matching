import io
import re
import csv


def parse_uploaded_file(file_bytes: bytes, filename: str) -> dict:
    
    KNOWN_KEYS = [
        "Position", "Moreinfo", "Highlights", "Long Description",
        "position_group", "English Level", "Experience Years",
        "Primary Keyword", "Company Name", "Exp Years", "CV"
    ]

    text = file_bytes.decode("utf-8", errors="ignore")

    if filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            return dict(row)

    # Parse key: value format with multiline support
    result = {}
    current_key = None
    current_value_lines = []

    for line in text.splitlines():
        # Check if this line starts a new known key
        matched_key = None
        for key in KNOWN_KEYS:
            if line.startswith(f"{key}:"):
                matched_key = key
                break

        if matched_key:
            # Save previous key if exists
            if current_key:
                result[current_key] = "\n".join(current_value_lines).strip()
            # Start new key
            current_key = matched_key
            current_value_lines = [line[len(matched_key) + 1:].strip()]
        else:
            # Continuation of previous value
            if current_key:
                current_value_lines.append(line)

    # Save last key
    if current_key:
        result[current_key] = "\n".join(current_value_lines).strip()

    return result


def extract_candidate_fields(row: dict) -> dict:
   
    moreinfo = row.get("Moreinfo", "")
    highlights = row.get("Highlights", "")
    combined_text = " ".join(filter(None, [moreinfo, highlights]))
    return {
        "name": row.get("Position", "Unknown Candidate"),
        "text": combined_text,
        "position_group": row.get("position_group", "Other"),
        "english_level": row.get("English Level", ""),
        "experience_years": row.get("Experience Years", ""),
        "primary_keyword": row.get("Primary Keyword", ""),
    }


def extract_job_fields(row: dict) -> dict:
  
    return {
        "title": row.get("Position", "Unknown Job"),
        "text": row.get("Long Description", ""),
        "position_group": row.get("position_group", "Other"),
        "company": row.get("Company Name", ""),
        "english_level": row.get("English Level", ""),
        "exp_years": row.get("Exp Years", ""),
        "primary_keyword": row.get("Primary Keyword", ""),
    }


def clean_text(text: str) -> str:
   
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        "",
        text,
    )
    boilerplate = [
        "apply now", "click here to apply", "visit our website",
        "follow us on", "apply here", "send your cv",
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()