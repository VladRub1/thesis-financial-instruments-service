"""Streamlit frontend for Document AI PoC.

Run with:
    uv run streamlit run streamlit_app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import os

import requests
import streamlit as st

from app.core.config import settings
from app.llm.schemas import ExtractionV1, ExtractionV2
from app.ui.login_gate import render_login_gate


def _correction_form_field_names(payload: dict) -> list[str]:
    """Match editable fields to extraction schema (v2 default; v1 for legacy jobs)."""
    ver = str(payload.get("schema_version") or "").strip().lower()
    if ver == "v1":
        skip = {"schema_version", "evidence", "warnings"}
        return [k for k in ExtractionV1.model_fields if k not in skip]
    skip = {"schema_version"}
    return [k for k in ExtractionV2.model_fields if k not in skip]

st.set_page_config(
    page_title="Document AI — Bank Guarantees",
    page_icon=".streamlit/icon.png",
    layout="wide",
)

_DEFAULT_API = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_BASE = _DEFAULT_API
DEMO_PASSWORD = settings.DEMO_PASSWORD.strip()
PUBLIC_DEMO_UI = bool(DEMO_PASSWORD)
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

if "demo_authenticated" not in st.session_state:
    st.session_state["demo_authenticated"] = False
if DEMO_PASSWORD and not st.session_state["demo_authenticated"]:
    render_login_gate(DEMO_PASSWORD)
    st.stop()

st.title("Document AI — Bank Guarantee Extraction")
st.info("Public demo note: processing usually takes 2–3 minutes. If the worker is busy, jobs may wait in queue.")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("Settings")
pipeline = st.sidebar.selectbox("Pipeline", ["ocr+extract", "ocr_only"])
engine_ocr = st.sidebar.selectbox("OCR engine", ["tesseract", "paddleocr"])
lang = st.sidebar.selectbox("Language", ["rus+eng", "rus", "eng"])
if DEMO_PASSWORD and st.sidebar.button("Lock demo"):
    for key in ("demo_authenticated", "demo_password_input", "demo_auth_error", "demo_ascii", "demo_haiku"):
        st.session_state.pop(key, None)
    st.session_state["demo_authenticated"] = False
    st.rerun()

# ── Upload tab / Path tab ────────────────────────────────────
if PUBLIC_DEMO_UI:
    tab_upload, tab_results = st.tabs(["Upload PDF", "Job results"])
    tab_path = None
else:
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

        if job_status == "queued":
            progress_bar.progress(0.01, text="Queued… worker is busy, waiting to start")
            time.sleep(1)
            continue

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
    st.session_state["active_job_id"] = job_id
    try:
        result = poll_job(job_id)
    except Exception as exc:
        st.error(f"Job polling failed: {exc}")
        return
    finally:
        if st.session_state.get("active_job_id") == job_id:
            st.session_state["active_job_id"] = None
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
    editable = _correction_form_field_names(current)
    with st.form(key=f"corrections_{job_id}"):
        fields: dict = {}
        for key in editable:
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
if "active_job_id" not in st.session_state:
    st.session_state["active_job_id"] = None

session_busy = bool(st.session_state.get("active_job_id"))
if session_busy:
    st.warning(
        f"A job is currently running in this session (`{st.session_state['active_job_id']}`). "
        "Wait until it finishes before submitting another."
    )

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

    _ALLOWED_TYPES = ["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
    _MIME_MAP = {
        ".pdf": "application/pdf", ".tif": "image/tiff", ".tiff": "image/tiff",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    }
    uploaded = st.file_uploader("Choose a document", type=_ALLOWED_TYPES)
    file_bytes = b""
    if uploaded:
        import fitz
        from PIL import Image as _PILImage
        file_bytes = uploaded.getvalue()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            st.error(
                f"File is too large ({len(file_bytes) / (1024 * 1024):.1f} MB). "
                f"Public demo limit is {MAX_UPLOAD_MB} MB."
            )
            uploaded = None
            file_bytes = b""

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()

        if suffix == ".pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            preview_cols = st.columns(min(len(doc), 4))
            for i, page in enumerate(doc):
                if i >= 4:
                    st.caption(f"… and {len(doc) - 4} more page(s)")
                    break
                pix = page.get_pixmap(dpi=120)
                preview_cols[i].image(pix.tobytes("png"), caption=f"Page {i + 1}", width="stretch")
            st.caption(f"**{uploaded.name}** — {len(doc)} page(s), {len(file_bytes) / 1024:.0f} KB")
            doc.close()
        elif suffix in (".tif", ".tiff"):
            import io
            img = _PILImage.open(io.BytesIO(file_bytes))
            n_frames = getattr(img, "n_frames", 1)
            preview_cols = st.columns(min(n_frames, 4))
            for i in range(min(n_frames, 4)):
                img.seek(i)
                preview_cols[i].image(img.copy().convert("RGB"), caption=f"Page {i + 1}", width="stretch")
            if n_frames > 4:
                st.caption(f"… and {n_frames - 4} more page(s)")
            st.caption(f"**{uploaded.name}** — {n_frames} page(s), {len(file_bytes) / 1024:.0f} KB")
        else:
            st.image(file_bytes, caption=uploaded.name, width="stretch")
            st.caption(f"**{uploaded.name}** — {len(file_bytes) / 1024:.0f} KB")

    if uploaded and st.button("Process uploaded file", disabled=session_busy):
        suffix = Path(uploaded.name).suffix.lower()
        mime = _MIME_MAP.get(suffix, "application/octet-stream")
        files = {"file": (uploaded.name, uploaded.getvalue(), mime)}
        data = {"pipeline": pipeline, "engine_ocr": engine_ocr, "lang": lang}
        resp = requests.post(f"{API_BASE}/v1/jobs", files=files, data=data)
        if resp.status_code == 201:
            job = resp.json()
            st.info(f"Job created: `{job['job_id']}`")
            submit_job(job["job_id"])
        else:
            st.error(f"API error: {resp.text}")

# ── Path tab ─────────────────────────────────────────────────
if tab_path is not None:
    with tab_path:
        file_path = st.text_input("Server-side file path")
        if file_path and st.button("Process by path", disabled=session_busy):
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
