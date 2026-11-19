"""
DataFix - Upload and Ingestion Module
Handles dataset ingestion, sample loading, structural preview, and immutable state initialization.
"""

import os
import random
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from typing import Optional, Tuple
from utils.file_handler import load_dataset, get_file_info
from utils.data_integrity import compute_dataframe_hash, create_safe_working_copy
from config import MAX_PREVIEW_ROWS

SAMPLE_DATASETS = {
    "Customer Demographics & Contacts": "sample_data/sample_customers.csv",
    "E-Commerce Sales Transactions": "sample_data/sample_sales.csv",
    "HR & Staff Records": "sample_data/sample_employees.csv"
}


def generate_synthetic_dirty_dataset(domain: str = "Healthcare / Patient Records", n_rows: int = 50, corruption_pct: float = 0.25) -> pd.DataFrame:
    """Generates realistic synthetic datasets infused with intentional real-world defects on the fly."""
    random.seed(42)
    rows = []
    
    first_names = ["James", "Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia", "Benjamin", "Isabella"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    states = ["California", "CA", "calif", "New York", "NY", "Texas", "TX", "Florida", "FL"]
    departments = ["Cardiology", "Neurology", "Pediatrics", "cardiology", "NEUROLOGY", "Orthopedics"]
    
    for i in range(1, n_rows + 1):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        full_name = f"{fn} {ln}"
        domain_email = random.choice(["gmail.com", "yahoo.com", "outlook.com", "hospital.org"])
        email = f"{fn.lower()}.{ln.lower()}@{domain_email}"
        phone = f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}"
        state = random.choice(states)
        dept = random.choice(departments)
        
        # Base dates
        birth_year = random.randint(1955, 2002)
        birth_date = f"{birth_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        adm_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        
        base_fee = round(random.uniform(500.0, 5000.0), 2)
        meds_fee = round(random.uniform(50.0, 500.0), 2)
        total_bill = round(base_fee + meds_fee, 2)
        
        # Controlled Intentional Corruptions based on corruption_pct
        if random.random() < corruption_pct:
            defect_type = random.choice(["whitespace", "casing", "email_typo", "date_format", "cross_math", "negative", "currency_str", "null"])
            if defect_type == "whitespace":
                full_name = f"  {full_name}  "
            elif defect_type == "casing":
                full_name = f"{fn.lower()} {ln.upper()}"
            elif defect_type == "email_typo":
                email = f"{fn.lower()}.{ln.lower()}@gmaill.com"
            elif defect_type == "date_format":
                adm_date = f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/2023"
            elif defect_type == "cross_math":
                total_bill = round(total_bill + random.choice([150.0, -200.0, 500.0]), 2)
            elif defect_type == "negative":
                base_fee = -abs(base_fee)
            elif defect_type == "currency_str":
                total_bill = f"${total_bill:,.2f}"
            elif defect_type == "null":
                phone = None

        rows.append({
            "record_id": f"REC-{1000 + i}",
            "patient_name": full_name,
            "email": email,
            "phone": phone,
            "state": state,
            "department": dept,
            "birth_date": birth_date,
            "admission_date": adm_date,
            "consultation_fee": base_fee,
            "medication_fee": meds_fee,
            "total_billing": total_bill
        })
        
    df = pd.DataFrame(rows)
    # Inject 1 exact duplicate row to demonstrate deduplication
    if len(df) > 5:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def initialize_session_state():
    """Initializes standard session state keys if not already present."""
    default_states = {
        "original_df": None,
        "original_hash": "",
        "working_df": None,
        "current_filename": "",
        "file_metadata": {},
        "detected_issues": [],
        "diagnoses": [],
        "audit_log": [],
        "undo_stack": [],
        "quality_score_pre": None,
        "quality_score_post": None,
        "validation_report": None,
        "active_page": "Overview Dashboard"
    }
    for key, val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = val


def set_dataset(df: pd.DataFrame, filename: str, metadata: dict):
    """Safely registers a loaded dataset, creates the immutable baseline hash, and resets working state."""
    # Ensure standard string column headers
    df.columns = [str(c).strip() for c in df.columns]
    
    st.session_state["original_df"] = df.copy(deep=True)
    st.session_state["original_hash"] = compute_dataframe_hash(df)
    st.session_state["working_df"] = create_safe_working_copy(df)
    st.session_state["current_filename"] = filename
    st.session_state["file_metadata"] = metadata
    
    # Reset processing states
    st.session_state["detected_issues"] = []
    st.session_state["diagnoses"] = []
    st.session_state["audit_log"] = []
    st.session_state["undo_stack"] = []
    st.session_state["quality_score_pre"] = None
    st.session_state["quality_score_post"] = None
    st.session_state["validation_report"] = None


def render_upload_page():
    """Renders the Upload and Ingestion page."""
    initialize_session_state()
    
    st.markdown("""
    <div class="datafix-card">
        <h2 style="color: #0B192C; margin-bottom: 0.25rem;">Data Ingestion & Dataset Selection</h2>
        <p style="color: #64748B; font-size: 0.95rem;">
            Upload your own real-world dataset or select a pre-configured sample dataset loaded with intentional errors for instant demonstration.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_upload, tab_sample, tab_synth = st.tabs([
        "📁 1. Upload Custom File (CSV, XLSX, TSV, Parquet, JSON)",
        "📊 2. Curated Sample Datasets",
        "🎲 3. Live Synthetic Dirty Data Generator"
    ])
    
    with tab_upload:
        st.subheader("Upload Any Data File")
        uploaded_file = st.file_uploader(
            "Upload CSV, XLSX, XLS, JSON, TSV, TXT, or Parquet",
            type=["csv", "xlsx", "xls", "json", "tsv", "txt", "parquet"],
            help="Files up to 200MB are supported. Datasets are parsed locally with auto-delimiter sniffing and strict immutability guarantees."
        )
        if uploaded_file is not None:
            if st.button("Ingest Uploaded File", key="btn_ingest_custom", type="primary"):
                with st.spinner("Decoding and profiling dataset structure..."):
                    meta = get_file_info(uploaded_file, uploaded_file.name)
                    success, df, msg = load_dataset(uploaded_file, uploaded_file.name)
                    if success:
                        set_dataset(df, uploaded_file.name, meta)
                        st.success(f"Loaded: {msg}")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {msg}")
                        
    with tab_sample:
        st.subheader("Quick Load Pre-Curated Domain Sample")
        sample_choice = st.selectbox(
            "Choose a curated dataset with intentional defects:",
            options=list(SAMPLE_DATASETS.keys()),
            index=0
        )
        st.caption("Each sample demonstrates diverse real-world issues: missing values, mixed dates, casing drift, typos, and cross-column logic errors.")
        
        if st.button("Load Selected Sample", key="btn_load_sample", type="secondary"):
            sample_rel_path = SAMPLE_DATASETS[sample_choice]
            if os.path.exists(sample_rel_path):
                with open(sample_rel_path, "rb") as f:
                    fname = os.path.basename(sample_rel_path)
                    meta = get_file_info(f, fname)
                    success, df, msg = load_dataset(sample_rel_path, fname)
                    if success:
                        set_dataset(df, fname, meta)
                        st.success(f"Loaded Sample '{sample_choice}': {msg}")
                        st.rerun()
                    else:
                        st.error(f"Sample load error: {msg}")
            else:
                st.error(f"File '{sample_rel_path}' not found on server.")

    with tab_synth:
        st.subheader("Live Dirty Data Generator (Hackathon Judge Demo)")
        st.caption("Generate an entirely new synthetic dataset on demand with controlled corruption levels to test DataFix in real time.")
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            synth_domain = st.selectbox("Select Domain:", ["Healthcare / Patient Records", "Fintech / Transactions", "Retail / E-Commerce"])
        with sc2:
            n_records = st.slider("Record Count:", min_value=25, max_value=200, value=60, step=25)
        with sc3:
            corruption_rate = st.slider("Corruption Intensity:", min_value=10, max_value=60, value=30, step=5)
            
        if st.button("🎲 Generate & Ingest Dirty Data Now", type="primary"):
            with st.spinner("Generating dirty synthetic data with intentional defects..."):
                synth_df = generate_synthetic_dirty_dataset(synth_domain, n_records, corruption_rate / 100.0)
                gen_name = f"synthetic_{synth_domain.split('/')[0].strip().lower()}_{n_records}rows.csv"
                meta = {
                    "filename": gen_name,
                    "extension": ".csv",
                    "size_bytes": len(synth_df) * 100,
                    "size_mb": 0.05,
                    "is_supported": True
                }
                set_dataset(synth_df, gen_name, meta)
                st.success(f"Generated and ingested {len(synth_df)} rows with {corruption_rate}% controlled defect rate!")
                st.rerun()

    # Dataset Inspection and Preview
    if st.session_state["working_df"] is not None:
        df = st.session_state["working_df"]
        st.markdown("<hr style='margin: 2rem 0; border-color: #E2E8F0;' />", unsafe_allow_html=True)
        st.markdown(f"### Ingested Dataset: `{st.session_state['current_filename']}`")
        
        # Metadata metric cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{len(df):,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{len(df.columns)}</div>
                <div class="stat-label">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{mem_mb} MB</div>
                <div class="stat-label">Memory Usage</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            null_count = int(df.isna().sum().sum())
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val" style="color: {'#EF4444' if null_count > 0 else '#10B981'};">{null_count:,}</div>
                <div class="stat-label">Raw Null Cells</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col5:
            dup_count = int(df.duplicated().sum())
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val" style="color: {'#EF4444' if dup_count > 0 else '#10B981'};">{dup_count:,}</div>
                <div class="stat-label">Exact Duplicates</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        st.subheader("Data Preview (First 50 Rows)")
        st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)
        
        st.markdown(f"""
        <div class="datafix-callout">
            <strong>Immutable Baseline Secured:</strong> SHA-256 fingerprint 
            <code style="background: #E2E8F0; padding: 2px 6px; border-radius: 4px;">{st.session_state['original_hash'][:16]}...</code>
            recorded. All downstream detection and repair actions operate on an isolated working copy.
        </div>
        """, unsafe_allow_html=True)
