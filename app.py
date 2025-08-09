# Streamlit XLSX Labeling Studio — with DOCX refs + Fact Check
# ------------------------------------------------------------
# Features
# - Upload .xlsx/.csv and edit in a grid (st.data_editor)
# - Upload .docx, parse line markers like L12:T3 (fallback to paragraph P1..)
# - Add refs to a selected row; copy referenced text → Extractive Summary
# - Local fact-check heuristic + optional OpenAI LLM check (paste API key)
# - Export edited sheet to .xlsx or .csv
#
# Local run:
#   1) pip install -r requirements.txt
#   2) streamlit run streamlit_app.py
#
# requirements.txt (create this file next to the script):
#   streamlit>=1.33
#   pandas>=2.0
#   openpyxl>=3.1
#   mammoth>=1.6.0
#   python-docx>=1.1.0
#   requests>=2.31
#
# Free hosting (Streamlit Community Cloud):
#   - Push this script + requirements.txt to a public GitHub repo
#   - Go to https://streamlit.io/cloud, sign in, and deploy the repo
#
# Optional: Hugging Face Spaces → new Space → Streamlit → add these two files

import io
import re
import json
import time
from typing import List, Dict

import pandas as pd
import streamlit as st
import requests

# -------- Page config --------
st.set_page_config(page_title="XLSX Labeling Studio", layout="wide")
st.title("XLSX Labeling Studio — DOCX‑assisted labeling")

# -------- Constants --------
EXPECTED_COLS = [
    "รหัสเอกสาร",
    "ลำดับ",
    "คำสั่ง (Prompt)",
    "Feedback",
    "ผลสรุปแบบสกัดข้อมูล (Extractive Summary)",
    "ผลสรุปแบบเรียบเรียงข้อมูลใหม่ (Abstractive Summary)",
    "หมายเลขย่อหน้า",
    "สถานะ",
]

# -------- Helpers --------

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""
    if "สถานะ" in df.columns:
        df["สถานะ"] = df["สถานะ"].replace({"": "Unlabeled"}).fillna("Unlabeled")
    else:
        df["สถานะ"] = "Unlabeled"
    return df


def read_xlsx_or_csv(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    # default to Excel
    return pd.read_excel(uploaded, engine="openpyxl")


def extract_docx_text(file_bytes: bytes) -> str:
    # Try mammoth first
    try:
        import mammoth
        result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
        text = result.value or ""
        if text.strip():
            return text
    except Exception:
        pass
    # Fallback: python-docx
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text
    except Exception as e:
        raise RuntimeError(f"Could not parse DOCX: {e}")


def parse_docx_lines(text: str) -> List[Dict[str, str]]:
    """Parse lines with markers 'Lxx:Tyy'. Fallback to paragraphs P1.."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Find tags like L12:T3 at line starts or after a newline
    re_tag = re.compile(r"(\n|^)\s*(L\d+:T\d+)\s*[:\-]?\s*")
    tags = []
    for m in re_tag.finditer(text):
        start_idx = m.start() + (len(m.group(1)) if m.group(1) else 0)
        tags.append({"idx": start_idx, "tag": m.group(2)})

    if not tags:
        # paragraph fallback
        bits = [b.strip() for b in re.split(r"\n+", text) if b.strip()]
        return [{"line": f"P{i+1}", "text": b} for i, b in enumerate(bits)]

    parts = []
    for i, t in enumerate(tags):
        start = t["idx"]
        tag = t["tag"]
        tag_end = start + text[start:].find(tag) + len(tag)
        end = tags[i + 1]["idx"] if i + 1 < len(tags) else len(text)
        chunk = text[tag_end:end].strip()
        if chunk:
            parts.append({"line": tag, "text": chunk})
    return parts


def get_referenced_text(ref_str: str, doc_lines: List[Dict[str, str]]) -> str:
    if not ref_str:
        return ""
    chunks: List[str] = []
    parts = [p.strip() for p in re.split(r"[,，]", str(ref_str)) if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = [x.strip() for x in p.split("-", 1)]
            try:
                a_idx = next(i for i, x in enumerate(doc_lines) if x["line"] == a)
                b_idx = next(i for i, x in enumerate(doc_lines) if x["line"] == b)
                if b_idx >= a_idx:
                    for i in range(a_idx, b_idx + 1):
                        chunks.append(doc_lines[i]["text"])
            except StopIteration:
                pass
        else:
            for x in doc_lines:
                if x["line"] == p:
                    chunks.append(x["text"])
                    break
    return "\n".join(chunks).strip()


def tokenize_th_en(s: str) -> set:
    # keep Thai range + word chars
    cleaned = re.sub(r"[^\w\u0E00-\u0E7F\s]", " ", str(s).lower())
    return set([t for t in re.split(r"\s+", cleaned) if t])


def simple_fact_check(src: str, extractive: str, abstractive: str):
    if not src:
        return {"score": 0, "notes": ["Missing or unresolved source lines from DOCX."]}
    src_tok = tokenize_th_en(src)
    ext_tok = tokenize_th_en(extractive)
    abs_tok = tokenize_th_en(abstractive)
    def jaccard(a, b):
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0
    overlap_ext = jaccard(ext_tok, src_tok)
    overlap_abs = jaccard(abs_tok, src_tok)
    nums = re.findall(r"\b\d+[\d.,]*\b", src)
    missing_nums = [n for n in nums if n not in re.findall(r"\b\d+[\d.,]*\b", abstractive)]
    score = round(((overlap_ext * 0.6) + (overlap_abs * 0.4)) * 100)
    notes = [
        f"Extractive lexical overlap ~{round(overlap_ext*100)}%",
        f"Abstractive lexical overlap ~{round(overlap_abs*100)}%",
    ]
    if missing_nums:
        notes.append("Numbers missing in abstractive: " + ", ".join(missing_nums))
    return {"score": score, "notes": notes}


def llm_fact_check(src: str, extractive: str, abstractive: str, api_key: str, model: str):
    if not api_key:
        return {"error": "Missing OpenAI API key"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict, concise fact-checker."},
            {"role": "user", "content": (
                "You are a meticulous fact-checking agent for Thai parliamentary minutes.\n"
                f"SOURCE:\n\"\"\"\n{src}\n\"\"\"\n"
                f"EXTRACTIVE:\n\"\"\"\n{extractive}\n\"\"\"\n"
                f"ABSTRACTIVE:\n\"\"\"\n{abstractive}\n\"\"\"\n"
                "Return JSON: {\"extractive_exact\":bool, \"abstractive_faithful\":bool, \"issues\":[\"...\"], \"suggested_fix\":\"(Thai)\"}"
            )}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body),
            timeout=60,
        )
        data = res.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}


# -------- Session State --------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "doc_lines" not in st.session_state:
    st.session_state.doc_lines = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "doc_error" not in st.session_state:
    st.session_state.doc_error = ""
if "sel_idx" not in st.session_state:
    st.session_state.sel_idx = 1  # 1-based for UI
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"


# -------- Sidebar: Uploads & Settings --------
with st.sidebar:
    st.header("Files")
    up = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "xls", "csv"], key="xlsx_up")
    if up is not None:
        try:
            df = read_xlsx_or_csv(up)
            df = ensure_columns(df)
            st.session_state.df = df
            st.success(f"Loaded {len(df)} rows from {up.name}")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    docx = st.file_uploader("Upload .docx (contains Lxx:Tyy markers)", type=["docx"], key="docx_up")
    if docx is not None:
        try:
            raw = docx.read()
            text = extract_docx_text(raw)
            st.session_state.doc_lines = parse_docx_lines(text)
            st.session_state.doc_name = docx.name
            st.session_state.doc_error = "" if st.session_state.doc_lines else "No markers found — using paragraph fallback."
            st.success(f"DOCX parsed • {len(st.session_state.doc_lines)} lines")
        except Exception as e:
            st.session_state.doc_lines = []
            st.session_state.doc_name = docx.name
            st.session_state.doc_error = str(e)
            st.error(f"DOCX error: {e}")

    st.header("LLM (optional)")
    st.session_state.api_key = st.text_input("OpenAI API key", value=st.session_state.api_key, type="password", help="Only used when you click 'Run with LLM'. Stored in session only.")
    st.session_state.model = st.text_input("Model", value=st.session_state.model)


# -------- Main layout --------
left, mid, right = st.columns([1.2, 1.4, 2.2])

# ----- Left: DOCX panel -----
with left:
    st.subheader("DOCX Source")
    if st.session_state.doc_name:
        st.caption(f"{st.session_state.doc_name} • {len(st.session_state.doc_lines)} lines")
        if st.session_state.doc_error:
            st.warning(st.session_state.doc_error)
    doc_q = st.text_input("Search DOCX (line or text)", "")
    limit = st.number_input("Show first N results", min_value=1, max_value=2000, value=300, step=50)

    # filter
    lines = st.session_state.doc_lines
    if doc_q.strip():
        q = doc_q.lower()
        lines = [x for x in lines if q in x["line"].lower() or q in str(x["text"]).lower()]

    # select lines view
    box_enabled = st.checkbox("Use multiselect to collect refs", value=False)
    if box_enabled:
        options = [x["line"] for x in lines[:limit]]
        selected_tags = st.multiselect("Select tags (Lxx:Tyy)", options=options)
        if st.button("Append selected → หมายเลขย่อหน้า") and len(st.session_state.df) > 0:
            idx = max(1, min(st.session_state.sel_idx, len(st.session_state.df))) - 1
            cur = str(st.session_state.df.at[idx, "หมายเลขย่อหน้า"]).strip()
            combo = ", ".join(selected_tags)
            new_val = f"{cur}, {combo}".strip(", ") if cur else combo
            st.session_state.df.at[idx, "หมายเลขย่อหน้า"] = new_val
            st.success("Appended refs")
    else:
        for i, row in enumerate(lines[:limit]):
            cols = st.columns([0.6, 0.4])
            cols[0].markdown(f"**{row['line']}** — {row['text'][:180]}{'…' if len(row['text'])>180 else ''}")
            if cols[1].button(f"Add {row['line']}", key=f"add_{i}") and len(st.session_state.df) > 0:
                idx = max(1, min(st.session_state.sel_idx, len(st.session_state.df))) - 1
                cur = str(st.session_state.df.at[idx, "หมายเลขย่อหน้า"]).strip()
                new_val = f"{cur}, {row['line']}".strip(", ") if cur else row['line']
                st.session_state.df.at[idx, "หมายเลขย่อหน้า"] = new_val

# ----- Middle: Focus card -----
with mid:
    st.subheader("Focus")
    df = st.session_state.df
    total = len(df)
    if total == 0:
        st.info("Upload an .xlsx or .csv to start.")
    else:
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            st.write("")
        with c2:
            if st.button("◀ Prev"):
                st.session_state.sel_idx = max(1, st.session_state.sel_idx - 1)
        with c3:
            if st.button("Next ▶"):
                st.session_state.sel_idx = min(total, st.session_state.sel_idx + 1)
        with c4:
            st.session_state.sel_idx = st.number_input("Row #", 1, max(1,total), value=st.session_state.sel_idx)
        idx = max(1, min(st.session_state.sel_idx, total)) - 1

        row = df.iloc[idx]
        st.caption(f"Row {idx+1} of {total}")

        # Editable fields
        prompt = st.text_area("คำสั่ง (Prompt)", value=str(row.get("คำสั่ง (Prompt)", "")), height=120)
        refs = st.text_input("หมายเลขย่อหน้า (e.g., L10:T2-L12:T4, L20:T1)", value=str(row.get("หมายเลขย่อหน้า", "")))

        # Doc preview
        doc_text = get_referenced_text(refs, st.session_state.doc_lines)
        with st.expander("DOCX Source Preview", expanded=True):
            st.text_area("", value=doc_text if doc_text else "—", height=180)

        extractive = st.text_area("ผลสรุปแบบสกัดข้อมูล (Extractive Summary)", value=str(row.get("ผลสรุปแบบสกัดข้อมูล (Extractive Summary)", "")), height=140)
        abstractive = st.text_area("ผลสรุปแบบเรียบเรียงข้อมูลใหม่ (Abstractive Summary)", value=str(row.get("ผลสรุปแบบเรียบเรียงข้อมูลใหม่ (Abstractive Summary)", "")), height=140)

        colA, colB, colC = st.columns([1,1,1])
        if colA.button("↘ Copy referenced DOCX → Extractive"):
            extractive = doc_text
        if colB.button("Mark Done"):
            df.at[idx, "สถานะ"] = "Done"
        if colC.button("Needs Review"):
            df.at[idx, "สถานะ"] = "Needs Review"

        # Write back edited values
        df.at[idx, "คำสั่ง (Prompt)"] = prompt
        df.at[idx, "หมายเลขย่อหน้า"] = refs
        df.at[idx, "ผลสรุปแบบสกัดข้อมูล (Extractive Summary)"] = extractive
        df.at[idx, "ผลสรุปแบบเรียบเรียงข้อมูลใหม่ (Abstractive Summary)"] = abstractive
        st.session_state.df = df

        # Fact check
        st.markdown("---")
        st.subheader("Fact Check")
        col1, col2 = st.columns([1,1])
        if col1.button("Run Local"):
            res = simple_fact_check(doc_text, extractive, abstractive)
            st.write(res)
        if col2.button("Run with LLM"):
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                with st.spinner("Asking the model…"):
                    res = llm_fact_check(doc_text, extractive, abstractive, st.session_state.api_key, st.session_state.model)
                st.write(res)

# ----- Right: Data grid + Export -----
with right:
    st.subheader("Data Grid")
    if len(st.session_state.df) == 0:
        st.info("No data yet. Upload a sheet in the sidebar.")
    else:
        edited = st.data_editor(
            st.session_state.df,
            use_container_width=True,
            num_rows="dynamic",
            height=540,
            key="grid",
        )
        st.session_state.df = edited

        # Progress
        done = int((edited["สถานะ"].astype(str).str.lower() == "done").sum())
        total = len(edited)
        pct = int(round(100 * done / total)) if total else 0
        st.progress(pct, text=f"Progress: {done}/{total} ({pct}%)")

        # Exports
        st.markdown("### Export")
        # CSV
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="edited.csv", mime="text/csv")
        # XLSX
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            edited.to_excel(writer, index=False, sheet_name="Sheet1")
        st.download_button("Download XLSX", data=xbuf.getvalue(), file_name="edited.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Shortcuts: Use the Focus panel to edit one row at a time, and the left DOCX panel to collect refs. Export from the right after editing.")
