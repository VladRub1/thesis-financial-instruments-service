"""Streamlit frontend for Document AI PoC.

Run with:
    uv run streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import streamlit as st

API_BASE = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.set_page_config(page_title="Document AI — Bank Guarantees", layout="wide")
st.title("Document AI — Bank Guarantee Extraction")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("Settings")
pipeline = st.sidebar.selectbox("Pipeline", ["ocr+extract", "ocr_only"])
engine_ocr = st.sidebar.selectbox("OCR engine", ["tesseract", "paddleocr"])
lang = st.sidebar.selectbox("Language", ["rus+eng", "rus", "eng"])

# ── Upload tab / Path tab ────────────────────────────────────
tab_upload, tab_path, tab_results = st.tabs(["Upload PDF", "By file path", "Job results"])

# ── helpers ──────────────────────────────────────────────────

def poll_job(job_id: str) -> dict:
    """Poll until job finishes."""
    progress_bar = st.progress(0, text="Processing…")
    while True:
        resp = requests.get(f"{API_BASE}/v1/jobs/{job_id}")
        data = resp.json()
        job_status = data.get("status", "unknown")
        pages_done = data.get("progress_pages", 0)
        pages_total = data.get("page_count")

        if job_status == "succeeded":
            progress_bar.progress(1.0, text=f"Done ({pages_total or pages_done} pages)")
            return data
        if job_status == "failed":
            progress_bar.progress(1.0, text="Failed")
            return data

        if pages_total and pages_total > 0:
            progress_bar.progress(
                min(pages_done / pages_total, 0.99),
                text=f"Processing… ({pages_done}/{pages_total} pages)",
            )
        else:
            progress_bar.progress(0, text="Processing…")
        time.sleep(1)


def submit_job(job_id: str) -> None:
    """Store job_id in session state and show results."""
    st.session_state["last_job_id"] = job_id
    result = poll_job(job_id)
    if result["status"] == "succeeded":
        st.success("Processing complete!")
    else:
        st.error(f"Job failed: {result.get('error_message', 'unknown error')}")


def show_result(job_id: str, ctx: str = "") -> None:
    resp = requests.get(f"{API_BASE}/v1/jobs/{job_id}/result")
    if resp.status_code != 200:
        st.error(f"Could not fetch result: {resp.text}")
        return
    result = resp.json()

    col_ocr, col_ext = st.columns(2)

    with col_ocr:
        st.subheader("OCR output (Markdown)")
        md_path = (result.get("artifacts") or {}).get("ocr_md")
        if md_path:
            try:
                with open(md_path, encoding="utf-8") as f:
                    st.text_area("OCR Markdown", f.read(), height=500, key=f"ocr_md_{ctx}_{job_id}")
            except FileNotFoundError:
                st.info(f"Artifact file not accessible locally: {md_path}")

    with col_ext:
        st.subheader("Extraction")
        ext = result.get("extraction")
        if ext and ext.get("json_validated"):
            st.json(ext["json_validated"])
        elif ext:
            st.warning(f"Extraction status: {ext.get('status')}")
        else:
            st.info("No extraction available (ocr_only pipeline?)")


def show_correction_form(job_id: str) -> None:
    """Render correction form — works independently of the processing button."""
    resp = requests.get(f"{API_BASE}/v1/jobs/{job_id}/result")
    if resp.status_code != 200:
        return
    result = resp.json()
    ext = result.get("extraction")
    current = (ext or {}).get("json_validated") or {}
    if not current:
        return

    st.subheader("Submit corrections")
    EDITABLE = [
        "guarantee_number", "issue_date", "start_date", "end_date",
        "amount", "currency", "principal_inn", "beneficiary_inn",
        "contract_number", "contract_date", "contract_name", "ikz",
        "bank_name", "bank_bic", "registry_number", "claim_period_days",
    ]
    with st.form(key=f"corrections_{job_id}"):
        fields: dict = {}
        for key in EDITABLE:
            val = current.get(key)
            fields[key] = st.text_input(key, value="" if val is None else str(val))

        comment = st.text_area("Comment (optional)")
        submitted_by = st.text_input("Your name (optional)")
        submit = st.form_submit_button("Submit corrections")

    if submit:
        cleaned = {k: (v if v != "" else None) for k, v in fields.items()}
        payload = {"fields": cleaned, "comment": comment or None, "submitted_by": submitted_by or None}
        corr_resp = requests.post(f"{API_BASE}/v1/jobs/{job_id}/corrections", json=payload)
        if corr_resp.status_code == 201:
            st.success(f"Correction saved (version {corr_resp.json().get('version')})")
        else:
            st.error(f"Error: {corr_resp.text}")


# ── Upload tab ───────────────────────────────────────────────
with tab_upload:
    SAMPLES_DIR = Path(__file__).parent / "samples"
    sample_pdfs = sorted(SAMPLES_DIR.glob("*.pdf")) if SAMPLES_DIR.is_dir() else []
    if sample_pdfs:
        st.markdown("**Sample documents** — download one, then upload it below:")
        cols = st.columns(len(sample_pdfs))
        for col, pdf in zip(cols, sample_pdfs):
            with col:
                st.download_button(
                    label=pdf.stem,
                    data=pdf.read_bytes(),
                    file_name=pdf.name,
                    mime="application/pdf",
                )
        st.divider()

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded:
        import fitz
        pdf_bytes = uploaded.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        preview_cols = st.columns(min(len(doc), 4))
        for i, page in enumerate(doc):
            if i >= 4:
                st.caption(f"… and {len(doc) - 4} more page(s)")
                break
            pix = page.get_pixmap(dpi=120)
            preview_cols[i].image(pix.tobytes("png"), caption=f"Page {i + 1}", width="stretch")
        st.caption(f"**{uploaded.name}** — {len(doc)} page(s), {len(pdf_bytes) / 1024:.0f} KB")
        doc.close()

    if uploaded and st.button("Process uploaded file"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data = {"pipeline": pipeline, "engine_ocr": engine_ocr, "lang": lang}
        resp = requests.post(f"{API_BASE}/v1/jobs", files=files, data=data)
        if resp.status_code == 201:
            job = resp.json()
            st.info(f"Job created: `{job['job_id']}`")
            submit_job(job["job_id"])
        else:
            st.error(f"API error: {resp.text}")

# ── Path tab ─────────────────────────────────────────────────
with tab_path:
    file_path = st.text_input("Server-side file path")
    if file_path and st.button("Process by path"):
        body = {
            "file_path": file_path,
            "pipeline": pipeline,
            "engine_ocr": engine_ocr,
            "lang": lang,
        }
        resp = requests.post(f"{API_BASE}/v1/jobs/by-path", json=body)
        if resp.status_code == 201:
            job = resp.json()
            st.info(f"Job created: `{job['job_id']}`")
            submit_job(job["job_id"])
        else:
            st.error(f"API error: {resp.text}")

# ── Results tab ──────────────────────────────────────────────
with tab_results:
    lookup_id = st.text_input("Job ID to look up")
    if lookup_id and st.button("Fetch result"):
        st.session_state["last_job_id"] = lookup_id

# ── Results section (shared, rendered once) ──────────────────
last_job = st.session_state.get("last_job_id")
if last_job:
    st.divider()
    show_result(last_job)
    show_correction_form(last_job)
