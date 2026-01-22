# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import zipfile
import io

import source

st.set_page_config(
    page_title="QuLab: Lab 8: LLM Evaluation Harness (Inference & Fine-Tuning Risk)",
    layout="wide",
)
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 8: LLM Evaluation Harness (Inference & Fine-Tuning Risk)")
st.divider()


def _ensure_prompt_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required + optional columns exist for prompts table
    for c in source.REQUIRED_PROMPTS_COLS:
        if c not in df.columns:
            df[c] = ""
    # Optional columns
    if hasattr(source, "OPTIONAL_PROMPTS_COLS"):
        for c in source.OPTIONAL_PROMPTS_COLS:
            if c not in df.columns:
                df[c] = ""
    # Order columns nicely
    ordered = source.REQUIRED_PROMPTS_COLS + \
        (source.OPTIONAL_PROMPTS_COLS if hasattr(
            source, "OPTIONAL_PROMPTS_COLS") else [])
    ordered = [c for c in ordered if c in df.columns] + \
        [c for c in df.columns if c not in ordered]
    return df[ordered]


def _append_new_entry(
    prompt_id: str,
    prompt_text: str,
    expected_answer: str,
    allowed_sources: str,
    finetuned_prompt_text: str,
    baseline_gen: dict,
    finetuned_gen: dict,
):
    # Prompts row
    prompts_row = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "expected_answer": expected_answer,
        "allowed_sources": allowed_sources,
    }
    if hasattr(source, "OPTIONAL_PROMPTS_COLS") and "finetuned_prompt_text" in source.OPTIONAL_PROMPTS_COLS:
        prompts_row["finetuned_prompt_text"] = finetuned_prompt_text

    # Outputs rows
    baseline_row = {
        "prompt_id": prompt_id,
        "llm_output": baseline_gen.get("llm_output", ""),
        "sources_cited": baseline_gen.get("sources_cited", ""),
    }
    finetuned_row = {
        "prompt_id": prompt_id,
        "llm_output": finetuned_gen.get("llm_output", ""),
        "sources_cited": finetuned_gen.get("sources_cited", ""),
    }

    # Append (and dedupe by prompt_id)
    st.session_state.prompts_df = pd.concat(
        [st.session_state.prompts_df, pd.DataFrame([prompts_row])],
        ignore_index=True
    ).drop_duplicates(subset=["prompt_id"], keep="last")

    st.session_state.baseline_df = pd.concat(
        [st.session_state.baseline_df, pd.DataFrame([baseline_row])],
        ignore_index=True
    ).drop_duplicates(subset=["prompt_id"], keep="last")

    # Only append finetuned if the lab includes it
    st.session_state.finetuned_df = pd.concat(
        [st.session_state.finetuned_df, pd.DataFrame([finetuned_row])],
        ignore_index=True
    ).drop_duplicates(subset=["prompt_id"], keep="last")

    # Clean columns
    st.session_state.prompts_df = _ensure_prompt_columns(
        st.session_state.prompts_df)

    # Reset run outputs because dataset changed
    _reset_run_outputs()


def render_llm_judge_card(judge: dict, title: str = "LLM Judge Validation"):
    """
    Dashboard-style rendering for LLM judge structured output.
    Expected keys:
      hallucination_risk (0..1), faithfulness (0..1),
      refusal_ok (bool), inappropriate_refusal (bool),
      flags (list[str]), notes (list[str])
    """
    if not isinstance(judge, dict) or not judge:
        st.info("No LLM judge data available.")
        return

    halluc = float(judge.get("hallucination_risk", 0.0) or 0.0)
    faith = float(judge.get("faithfulness", 0.0) or 0.0)
    refusal_ok = bool(judge.get("refusal_ok", False))
    inapp_ref = bool(judge.get("inappropriate_refusal", False))
    flags = judge.get("flags", []) or []
    notes = judge.get("notes", []) or []

    # Header
    st.markdown(f"**{title}**")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Hallucination Risk", f"{halluc:.2f}")
        st.caption("Higher = more likely hallucination/overreach.")

    with m2:
        st.metric("Faithfulness", f"{faith:.2f}")
        st.caption("Higher = better citation/scope compliance.")

    with m3:
        st.metric("Refusal OK", "Yes" if refusal_ok else "No")
        st.caption("Is refusal appropriate for the prompt?")

    with m4:
        st.metric("Inappropriate Refusal", "Yes" if inapp_ref else "No")
        st.caption("Refused a prompt that should be answerable.")

    # Status ribbon
    if halluc >= 0.7 or faith <= 0.3 or inapp_ref:
        st.error("Judge Summary: High risk / needs review.")
    elif halluc >= 0.4 or faith <= 0.6:
        st.warning("Judge Summary: Medium risk / review recommended.")
    else:
        st.success("Judge Summary: Looks acceptable under judge rubric.")

    # Flags + notes in expanders
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Flags", expanded=bool(flags)):
            if flags:
                st.dataframe(pd.DataFrame(
                    {"flag": flags}), hide_index=True, width='stretch')
            else:
                st.info("No flags returned by the judge.")

    with c2:
        with st.expander("Notes", expanded=bool(notes)):
            if notes:
                # Notes can be long; keep readable
                for i, n in enumerate(notes, start=1):
                    st.write(f"{i}. {n}")
            else:
                st.info("No notes returned by the judge.")

    # Optional: raw view without st.json
    with st.expander("Raw Judge Output (read-only)", expanded=False):
        st.code(json.dumps(judge, indent=2), language="json")

# -----------------------------
# Session state init
# -----------------------------


def _init_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.getenv(
            "OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else None

    if "current_page" not in st.session_state:
        st.session_state.current_page = "1. Dataset Builder"

    if "eval_config" not in st.session_state:
        st.session_state.eval_config = source.DEFAULT_EVAL_CONFIG.copy()

    # Dataset builder frames
    if "prompts_df" not in st.session_state:
        st.session_state.prompts_df = pd.DataFrame(
            columns=source.REQUIRED_PROMPTS_COLS)
    if "baseline_df" not in st.session_state:
        st.session_state.baseline_df = pd.DataFrame(
            columns=source.REQUIRED_OUTPUTS_COLS)
    if "finetuned_df" not in st.session_state:
        st.session_state.finetuned_df = pd.DataFrame(
            columns=source.REQUIRED_OUTPUTS_COLS)

    if "use_finetuned" not in st.session_state:
        st.session_state.use_finetuned = True

    # Run outputs
    if "eval_df" not in st.session_state:
        st.session_state.eval_df = pd.DataFrame()
    if "is_finetuned_comparison" not in st.session_state:
        st.session_state.is_finetuned_comparison = False
    if "aggregate_metrics" not in st.session_state:
        st.session_state.aggregate_metrics = {}
    if "regression_analysis_results" not in st.session_state:
        st.session_state.regression_analysis_results = {}

    # LLM judge
    if "run_llm_judge" not in st.session_state:
        st.session_state.run_llm_judge = False
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "gpt-4o-mini"

    # Export
    if "artifact_dir" not in st.session_state:
        st.session_state.artifact_dir = None
    if "zip_buffer" not in st.session_state:
        st.session_state.zip_buffer = None
    if "manifest_content" not in st.session_state:
        st.session_state.manifest_content = None


_init_state()

# -----------------------------
# Sidebar: config + nav
# -----------------------------
st.sidebar.header("Configuration")
openai_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Required only if you enable LLM Judge validation (Structured Outputs).",
    value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
)

if openai_key_input:
    st.session_state.openai_api_key = openai_key_input
    os.environ["OPENAI_API_KEY"] = openai_key_input
else:
    st.session_state.openai_api_key = None

st.session_state.run_llm_judge = True

st.session_state.llm_model = "gpt-4o-mini"

st.sidebar.divider()
st.sidebar.header("Navigation")
page_options = [
    "1. Dataset Builder",
    "2. Configure Evaluation Rules",
    "3. Run Evaluation",
    "4. View Scorecards",
    "5. Inspect Failure Exemplars",
    "6. Export Artifacts",
]
try:
    current_page_index = page_options.index(st.session_state.current_page)
except ValueError:
    current_page_index = 0

st.session_state.current_page = st.sidebar.selectbox(
    "Go to...", options=page_options, index=current_page_index)

# -----------------------------
# Helpers
# -----------------------------


def _reset_run_outputs():
    st.session_state.eval_df = pd.DataFrame()
    st.session_state.is_finetuned_comparison = False
    st.session_state.aggregate_metrics = {}
    st.session_state.regression_analysis_results = {}
    st.session_state.artifact_dir = None
    st.session_state.zip_buffer = None
    st.session_state.manifest_content = None


def _csv_download_button(df: pd.DataFrame, label: str, filename: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes,
                       file_name=filename, mime="text/csv")


# -----------------------------
# Pages
# -----------------------------
if st.session_state.current_page == "1. Dataset Builder":
    st.markdown("### 1. Dataset Builder")
    st.markdown(
        "Create, view, edit, and load evaluation data **inside the app**.")

    # Ensure prompt columns exist (required + optional)
    st.session_state.prompts_df = _ensure_prompt_columns(
        st.session_state.prompts_df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["View", "Add", "Edit", "Load"],
    )

    # -------------------------
    # LOAD
    # -------------------------
    with tab4:
        st.markdown("#### Load Sample Dataset")
        st.info("Click to load pre-generated sample prompts and outputs.")

        if st.button("Load Samples"):
            source.generate_mock_data()
            st.session_state.prompts_df = _ensure_prompt_columns(
                pd.read_csv("sample_prompts.csv"))
            st.session_state.baseline_df = pd.read_csv(
                "sample_baseline_outputs.csv")
            st.session_state.finetuned_df = pd.read_csv(
                "sample_finetuned_outputs.csv")
            st.session_state.use_finetuned = True
            _reset_run_outputs()
            st.success("Samples loaded.")
            st.toast("Sample dataset loaded into session.")
            import time
            time.sleep(1)
            st.rerun()

    # -------------------------
    # VIEW
    # -------------------------
    with tab1:
        st.markdown("#### Dataset Overview")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Prompts", str(len(st.session_state.prompts_df)))
        with c2:
            st.metric("Baseline Outputs", str(
                len(st.session_state.baseline_df)))
        with c3:
            st.metric("Fine-tuned Outputs",
                      str(len(st.session_state.finetuned_df)))

        st.markdown("##### Prompts")
        st.dataframe(st.session_state.prompts_df, width='stretch')

        st.markdown("##### Baseline Outputs")
        st.dataframe(st.session_state.baseline_df, width='stretch')

        st.markdown("##### Fine-tuned Outputs")
        st.dataframe(st.session_state.finetuned_df, width='stretch')

    # -------------------------
    # EDIT (with DELETE)
    # -------------------------
    with tab3:
        st.markdown("#### Edit Dataset")
        st.caption(
            "You can edit tables inline. Use Delete controls below to remove entries by prompt_id.")

        st.markdown("##### Edit Prompts")
        st.session_state.prompts_df = st.data_editor(
            st.session_state.prompts_df,
            num_rows="dynamic",
            width='stretch',
            key="edit_prompts",
        )

        st.markdown("##### Edit Baseline Outputs")
        st.session_state.baseline_df = st.data_editor(
            st.session_state.baseline_df,
            num_rows="dynamic",
            width='stretch',
            key="edit_baseline",
        )

        st.markdown("##### Edit Fine-tuned Outputs")
        st.session_state.finetuned_df = st.data_editor(
            st.session_state.finetuned_df,
            num_rows="dynamic",
            width='stretch',
            key="edit_finetuned",
        )

        st.markdown("---")
        st.markdown("#### Delete Entries")
        all_ids = sorted(set(st.session_state.prompts_df["prompt_id"].astype(
            str).tolist())) if "prompt_id" in st.session_state.prompts_df.columns else []
        delete_ids = st.multiselect(
            "Select prompt_id(s) to delete", options=all_ids)

        if st.button("Delete Selected", disabled=(len(delete_ids) == 0)):
            st.session_state.prompts_df = st.session_state.prompts_df[~st.session_state.prompts_df["prompt_id"].isin(
                delete_ids)].reset_index(drop=True)
            st.session_state.baseline_df = st.session_state.baseline_df[~st.session_state.baseline_df["prompt_id"].isin(
                delete_ids)].reset_index(drop=True)
            st.session_state.finetuned_df = st.session_state.finetuned_df[~st.session_state.finetuned_df["prompt_id"].isin(
                delete_ids)].reset_index(drop=True)
            _reset_run_outputs()
            st.success(f"Deleted {len(delete_ids)} entries.")
            st.rerun()

        st.markdown("---")
        st.markdown("#### Save Changes")
        st.info(
            "Edits automatically persist in session state. Run evaluation again after edits.")

    # -------------------------
    # ADD (with LLM structured generation + confirm add)
    # -------------------------
    with tab2:
        st.markdown("#### Add a New Prompt + Auto-generate Outputs")
        st.caption(
            "Fill prompt fields, click Generate Response, then confirm whether to add the entry to the dataset.")

        if st.session_state.run_llm_judge and not st.session_state.openai_api_key:
            st.warning(
                "You have LLM features enabled but OPENAI_API_KEY is not set. Add a key in the sidebar.")

        # Add form
        with st.form("add_prompt_form", clear_on_submit=False):
            prompt_id = "P00" + str(len(st.session_state.prompts_df) + 1)
            prompt_text = st.text_area("Prompt", height=120)
            expected_answer = st.text_area(
                "Expected Answer", height=80)
            allowed_sources = st.text_input(
                "Allowed Sources (comma-separated, optional)")
            finetuned_prompt_text = st.text_area(
                "Fine-tuned Prompt", height=120)

            gen_model = st.text_input(
                "Generation Model", value=st.session_state.llm_model)
            generate_clicked = st.form_submit_button(
                "Generate Response (Baseline + Fine-tuned)")

        if generate_clicked:
            if not prompt_text.strip():
                st.error("Prompt is required.")
                st.stop()
            if not expected_answer.strip():
                st.error("Expected Answer is required.")
                st.stop()
            if not finetuned_prompt_text.strip():
                st.error("Fine-tuned Prompt is required.")
                st.stop()
            if not st.session_state.openai_api_key:
                st.error(
                    "OPENAI_API_KEY is required to generate structured responses.")
                st.stop()

            with st.spinner("Generating structured baseline output..."):
                baseline_gen = source.generate_structured_llm_answer(
                    prompt_text=prompt_text,
                    allowed_sources=allowed_sources,
                    model=gen_model,
                )

            with st.spinner("Generating structured fine-tuned output..."):
                finetuned_gen = source.generate_structured_llm_answer(
                    prompt_text=finetuned_prompt_text,
                    allowed_sources=allowed_sources,
                    model=gen_model,
                )

            # Store pending candidate in session_state for confirmation step
            st.session_state.pending_new_entry = {
                "prompt_id": prompt_id.strip(),
                "prompt_text": prompt_text.strip(),
                "expected_answer": expected_answer.strip(),
                "allowed_sources": allowed_sources.strip(),
                "finetuned_prompt_text": finetuned_prompt_text.strip(),
                "baseline_gen": baseline_gen,
                "finetuned_gen": finetuned_gen,
            }

            st.success(
                "Generated structured outputs. Review below and confirm to add to dataset.")

        # Confirmation UI if pending exists
        if "pending_new_entry" in st.session_state and st.session_state.pending_new_entry:
            entry = st.session_state.pending_new_entry

            st.markdown("---")
            st.markdown("### Review Generated Outputs")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Baseline Generated Output")
                st.info(entry["baseline_gen"].get("llm_output", ""))
                st.caption(
                    f"Sources cited: {entry['baseline_gen'].get('sources_cited', '')}")

            with c2:
                st.markdown("#### Fine-tuned Generated Output")
                st.info(entry["finetuned_gen"].get("llm_output", ""))
                st.caption(
                    f"Sources cited: {entry['finetuned_gen'].get('sources_cited', '')}")

            st.markdown("---")
            decision = st.radio(
                "Do you want to add this entry to the dataset?",
                ["No", "Yes"],
                horizontal=True,
                key="confirm_add_radio",
            )

            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button("Confirm Addition", disabled=(decision != "Yes"), width='stretch'):
                    _append_new_entry(
                        prompt_id=entry["prompt_id"],
                        prompt_text=entry["prompt_text"],
                        expected_answer=entry["expected_answer"],
                        allowed_sources=entry["allowed_sources"],
                        finetuned_prompt_text=entry["finetuned_prompt_text"],
                        baseline_gen=entry["baseline_gen"],
                        finetuned_gen=entry["finetuned_gen"],
                    )
                    st.session_state.pending_new_entry = None
                    st.toast("New entry added to dataset.")
                    import time
                    time.sleep(1)
                    st.rerun()

            with b2:
                if st.button("Discard Pending Entry", width='stretch'):
                    st.session_state.pending_new_entry = None
                    st.info("Pending entry discarded.")
                    import time
                    time.sleep(1)
                    st.rerun()

elif st.session_state.current_page == "2. Configure Evaluation Rules":
    st.markdown("### 2. Configure Evaluation Rules")
    st.markdown(
        "Tune thresholds and phrase lists used by deterministic evaluation proxies.")
    st.markdown("---")

    st.markdown("#### Hallucination Proxy Settings")
    st.session_state.eval_config["hallucination_threshold_ratio"] = st.number_input(
        r"Hallucination Threshold Ratio ($$ R = \frac{\text{length}(\text{LLM Output})}{\text{length}(\text{Prompt Text})} $$)",
        value=float(st.session_state.eval_config.get(
            "hallucination_threshold_ratio", 0.2)),
        min_value=0.0, max_value=10.0, step=0.01,
    )

    kw_str = ", ".join(st.session_state.eval_config.get(
        "over_specificity_keywords", []))
    kw_new = st.text_area(
        "Over-Specificity Keywords (comma-separated)", value=kw_str)
    st.session_state.eval_config["over_specificity_keywords"] = [
        k.strip() for k in kw_new.split(",") if k.strip()]

    st.markdown("---")
    st.markdown("#### Refusal and Over-Compliance Settings")
    rp = "\n".join(st.session_state.eval_config.get("refusal_phrases", []))
    rp_new = st.text_area(
        "Refusal Phrases (one per line)", value=rp, height=150)
    st.session_state.eval_config["refusal_phrases"] = [
        p.strip() for p in rp_new.split("\n") if p.strip()]

    dp = "\n".join(st.session_state.eval_config.get(
        "excessive_safety_disclaimers", []))
    dp_new = st.text_area(
        "Excessive Safety Disclaimers (one per line)", value=dp, height=150)
    st.session_state.eval_config["excessive_safety_disclaimers"] = [
        p.strip() for p in dp_new.split("\n") if p.strip()]

    st.markdown("---")
    st.markdown("#### Regression Analysis Settings")
    st.session_state.eval_config["regression_threshold_delta"] = st.number_input(
        "Regression Threshold Delta",
        value=float(st.session_state.eval_config.get(
            "regression_threshold_delta", 0.05)),
        min_value=0.0, max_value=1.0, step=0.01,
    )

    st.markdown("---")
    st.markdown("#### Citation Pattern")
    st.session_state.eval_config["citation_pattern"] = st.text_input(
        "Citation Pattern (regex)",
        value=st.session_state.eval_config.get("citation_pattern", r"\[\d+\]"),
    )

    st.markdown("---")
    st.markdown("Current Evaluation Configuration:")

    # --- Dashboard-style "Current Evaluation Configuration" (replace st.json) ---

    st.markdown("---")
    st.markdown("### Current Evaluation Configuration Dashboard")

    cfg = st.session_state.eval_config

    # Row 1: headline metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Hallucination Threshold Ratio",
            value=f"{cfg.get('hallucination_threshold_ratio', 0.0):.2f}",
            help="R = length(LLM Output) / length(Prompt). Higher R flags potential verbosity/hallucination."
        )
    with c2:
        st.metric(
            label="Regression Threshold Delta",
            value=f"{cfg.get('regression_threshold_delta', 0.0):.2f}",
            help="Increase in negative metrics beyond this delta triggers regression flags."
        )
    with c3:
        st.metric(
            label="Citation Pattern",
            value=str(cfg.get("citation_pattern", "")),
            help="Regex used to detect citations in outputs."
        )
    with c4:
        st.metric(
            label="Output Directory",
            value=str(cfg.get("output_dir", "")),
            help="Base folder where evidence/artifacts are written."
        )

    st.markdown("")

    # Row 2: quick counts / inventory
    kws = cfg.get("over_specificity_keywords", []) or []
    ref_phrases = cfg.get("refusal_phrases", []) or []
    disc_phrases = cfg.get("excessive_safety_disclaimers", []) or []

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Over-Specificity Keywords", value=str(len(kws)))
        st.caption(
            "Signals over-precision risk (esp. when no sources are allowed).")
    with d2:
        st.metric("Refusal Phrases", value=str(len(ref_phrases)))
        st.caption(
            "Used to detect refusals and potential inappropriate refusals.")
    with d3:
        st.metric("Safety Disclaimer Phrases", value=str(len(disc_phrases)))
        st.caption("Used to detect over-compliance / excessive disclaimers.")

    st.markdown("")

    # Row 3: structured display in sections
    left, right = st.columns([1, 1])

    with left:
        with st.expander("Over-Specificity Keywords", expanded=True):
            if len(kws) == 0:
                st.info("No keywords configured.")
            else:
                kw_df = pd.DataFrame({"keyword": kws})
                st.dataframe(kw_df, width='stretch', hide_index=True)

        with st.expander("Refusal Phrases", expanded=False):
            if len(ref_phrases) == 0:
                st.info("No refusal phrases configured.")
            else:
                rp_df = pd.DataFrame({"phrase": ref_phrases})
                st.dataframe(rp_df, width='stretch', hide_index=True)

    with right:
        with st.expander("Excessive Safety Disclaimer Phrases", expanded=False):
            if len(disc_phrases) == 0:
                st.info("No disclaimer phrases configured.")
            else:
                dp_df = pd.DataFrame({"phrase": disc_phrases})
                st.dataframe(dp_df, width='stretch', hide_index=True)

        with st.expander("Raw Config (read-only)", expanded=False):
            # Still not st.json; keep it readable and compact
            st.code(json.dumps(cfg, indent=2), language="json")
            st.caption(
                "This is a read-only view for debugging. Main dashboard above is the preferred view.")


elif st.session_state.current_page == "3. Run Evaluation":
    st.markdown("### 3. Run Evaluation")
    st.markdown(
        "Runs deterministic proxies and LLM judge validation.")
    st.markdown("---")

    # Pre-flight dataset schema checks
    can_run = True
    errs = []
    ok, e = source.validate_dataframe_schema(
        st.session_state.prompts_df, source.REQUIRED_PROMPTS_COLS, "prompts_df")
    if not ok:
        can_run = False
        errs.extend(e)
    ok, e = source.validate_dataframe_schema(
        st.session_state.baseline_df, source.REQUIRED_OUTPUTS_COLS, "baseline_df")
    if not ok:
        can_run = False
        errs.extend(e)
    if st.session_state.use_finetuned and not st.session_state.finetuned_df.empty:
        ok, e = source.validate_dataframe_schema(
            st.session_state.finetuned_df, source.REQUIRED_OUTPUTS_COLS, "finetuned_df")
        if not ok:
            can_run = False
            errs.extend(e)

    if errs:
        st.warning("Fix schema issues before running:\n- " + "\n- ".join(errs))

    if st.session_state.run_llm_judge and not st.session_state.openai_api_key:
        st.warning(
            "LLM Judge is enabled but OPENAI_API_KEY is not set. Disable LLM Judge or add a key.")
        can_run = False

    if st.button("Run Evaluation", disabled=not can_run):
        with st.spinner("Running evaluation pipeline..."):
            finetuned_df = st.session_state.finetuned_df if (
                st.session_state.use_finetuned and not st.session_state.finetuned_df.empty) else None

            result = source.run_pipeline_from_dataframes(
                prompts_df=st.session_state.prompts_df,
                baseline_df=st.session_state.baseline_df,
                finetuned_df=finetuned_df,
                cfg=st.session_state.eval_config,
                run_llm_judge=st.session_state.run_llm_judge,
                llm_model=st.session_state.llm_model,
            )

            st.session_state.eval_df = result["eval_df"]
            st.session_state.is_finetuned_comparison = result["is_finetuned_comparison"]
            st.session_state.aggregate_metrics = result["aggregate_metrics"]
            st.session_state.regression_analysis_results = result["regression_analysis"]

            # clear export cache
            st.session_state.artifact_dir = None
            st.session_state.zip_buffer = None
            st.session_state.manifest_content = None

        st.success(
            "Evaluation complete! Proceed to Scorecards / Failure Exemplars / Export.")
        st.rerun()

    if not st.session_state.eval_df.empty:
        st.markdown("#### Evaluation Data Preview")
        st.dataframe(st.session_state.eval_df.head(), width='stretch')

elif st.session_state.current_page == "4. View Scorecards":
    st.markdown("### 4. View Scorecards")
    st.markdown("---")

    if not st.session_state.aggregate_metrics:
        st.warning("Run the evaluation first.")
    else:
        st.markdown("#### Aggregate Scorecard")
        metrics_to_display = {
            model: {k: v for k, v in data.items() if k != "total_prompts"}
            for model, data in st.session_state.aggregate_metrics.items()
        }
        metrics_df = pd.DataFrame.from_dict(metrics_to_display, orient="index")
        st.dataframe(metrics_df.style.format("{:.2%}").set_caption(
            "LLM Evaluation Metrics"), width='stretch')

        st.markdown("#### Comparison of Key Metrics")
        metrics_to_plot = ["hallucination_rate", "faithfulness_score",
                           "refusal_rate", "inappropriate_refusal_rate"]
        plot_data = []
        for model_prefix in ["baseline"] + (["finetuned"] if st.session_state.is_finetuned_comparison else []):
            for metric in metrics_to_plot:
                plot_data.append({
                    "Model": model_prefix.capitalize(),
                    "Metric": metric.replace("_", " ").title(),
                    "Value": st.session_state.aggregate_metrics[model_prefix].get(metric, 0),
                })

        plot_df = pd.DataFrame(plot_data)
        if not plot_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Metric", y="Value", hue="Model", data=plot_df)
            plt.title("Comparison of Key LLM Evaluation Metrics")
            plt.ylabel("Rate / Score")
            plt.ylim(0, 1)
            st.pyplot(plt)
            plt.clf()

        if st.session_state.run_llm_judge:
            st.markdown("---")
            st.markdown("#### LLM Judge Aggregates (if enabled)")
            cols = []
            if "baseline_llm_hallucination_risk" in st.session_state.eval_df.columns:
                cols.append("baseline_llm_hallucination_risk")
            if "baseline_llm_faithfulness" in st.session_state.eval_df.columns:
                cols.append("baseline_llm_faithfulness")
            if st.session_state.is_finetuned_comparison and "finetuned_llm_hallucination_risk" in st.session_state.eval_df.columns:
                cols.append("finetuned_llm_hallucination_risk")
            if st.session_state.is_finetuned_comparison and "finetuned_llm_faithfulness" in st.session_state.eval_df.columns:
                cols.append("finetuned_llm_faithfulness")
            if cols:
                agg = st.session_state.eval_df[cols].mean(
                    numeric_only=True).to_frame("mean").T
                st.dataframe(agg, width='stretch')
            else:
                st.info(
                    "No LLM judge columns found. Ensure evaluation ran with LLM Judge enabled.")

        if st.session_state.is_finetuned_comparison:
            st.markdown("---")
            st.markdown("#### Fine-tuning Regression Analysis")
            ra = st.session_state.regression_analysis_results
            if ra:
                for metric, delta in ra.get("deltas", {}).items():
                    st.markdown(
                        f"**{metric.replace('_', ' ').title()} Delta:** `{delta:+.4f}`")

                if ra.get("regressions_flagged"):
                    st.error("!!! REGRESSION DETECTED !!!")
                    for flagged_metric in ra.get("flagged_metrics", []):
                        st.markdown(f"- {flagged_metric}")
                else:
                    st.success(
                        "No regressions detected beyond threshold. Fine-tuning appears stable or improved.")
            else:
                st.info("No regression analysis available.")

elif st.session_state.current_page == "5. Inspect Failure Exemplars":
    st.markdown("### 5. Inspect Failure Exemplars")
    st.markdown("---")

    if st.session_state.eval_df.empty or not st.session_state.aggregate_metrics:
        st.warning("Run the evaluation first to identify failure exemplars.")
    else:
        df = st.session_state.eval_df.copy()

        flag_suffixes = [
            "excessive_length_flag",
            "unsupported_factual_claim_flag",
            "over_specificity_flag",
            "missing_allowed_source_flag",
            "out_of_scope_reference_flag",
            "uncited_assertion_flag",
            "refusal_flag",
            "excessive_disclaimer_flag",
            "inappropriate_refusal_flag",
        ]

        df["is_any_high_risk"] = False
        for suffix in flag_suffixes:
            base_col = f"baseline_{suffix}"
            if base_col in df.columns:
                df["is_any_high_risk"] = df["is_any_high_risk"] | df[base_col]
            if st.session_state.is_finetuned_comparison:
                ft_col = f"finetuned_{suffix}"
                if ft_col in df.columns:
                    df["is_any_high_risk"] = df["is_any_high_risk"] | df[ft_col]

        # If LLM judge enabled, also treat high hallucination risk as exemplar-worthy
        if st.session_state.run_llm_judge and "baseline_llm_hallucination_risk" in df.columns:
            df["is_any_high_risk"] = df["is_any_high_risk"] | (
                df["baseline_llm_hallucination_risk"] >= 0.7)
            if st.session_state.is_finetuned_comparison and "finetuned_llm_hallucination_risk" in df.columns:
                df["is_any_high_risk"] = df["is_any_high_risk"] | (
                    df["finetuned_llm_hallucination_risk"] >= 0.7)

        exemplars = df[df["is_any_high_risk"]].head(5)

        if exemplars.empty:
            st.info("No high-risk prompts identified based on current evaluation.")
        else:
            for _, row in exemplars.iterrows():
                st.markdown(f"##### Prompt ID: `{row['prompt_id']}`")
                st.markdown(f"**Prompt Text:** {row.get('prompt_text', '')}")
                st.markdown(
                    f"**Allowed Sources:** {row.get('allowed_sources', '')}")
                st.markdown(
                    f"**Expected Answer:** {row.get('expected_answer', '')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline Output**")
                    st.info(row.get("baseline_output", ""))
                    st.caption(
                        f"Sources: {row.get('baseline_sources_cited', '')}")
                    base_issues = []
                    for suffix in flag_suffixes:
                        c = f"baseline_{suffix}"
                        if c in row and bool(row[c]):
                            base_issues.append(
                                suffix.replace("_", " ").title())
                    if base_issues:
                        st.error("Detected issues: " + ", ".join(base_issues))
                    if st.session_state.run_llm_judge and "baseline_llm_validation" in row:
                        judge_obj = row.get("baseline_llm_validation", None)
                        if judge_obj:
                            st.markdown("---")
                            render_llm_judge_card(
                                judge_obj, title="LLM Judge (Baseline)")

                if st.session_state.is_finetuned_comparison:
                    with col2:
                        st.markdown("**Fine-tuned Output**")
                        st.info(row.get("finetuned_output", ""))
                        st.caption(
                            f"Sources: {row.get('finetuned_sources_cited', '')}")
                        ft_issues = []
                        for suffix in flag_suffixes:
                            c = f"finetuned_{suffix}"
                            if c in row and bool(row[c]):
                                ft_issues.append(
                                    suffix.replace("_", " ").title())
                        if ft_issues:
                            st.warning("Detected issues: " +
                                       ", ".join(ft_issues))
                        if st.session_state.run_llm_judge and "finetuned_llm_validation" in row:
                            judge_obj = row.get(
                                "finetuned_llm_validation", None)
                            if judge_obj:
                                st.markdown("---")
                                render_llm_judge_card(
                                    judge_obj, title="LLM Judge (Fine-tuned)")

                st.divider()

elif st.session_state.current_page == "6. Export Artifacts":
    st.markdown("### 6. Export Artifacts (Release-ready package)")
    st.markdown("---")

    if not st.session_state.aggregate_metrics or st.session_state.eval_df.empty:
        st.warning("Run the evaluation first to generate artifacts.")
    else:
        if st.button("Generate & Download Artifacts"):
            with st.spinner("Generating artifacts..."):
                artifact_dir = source.generate_artifacts(
                    df=st.session_state.eval_df.copy(),
                    aggregate_metrics=st.session_state.aggregate_metrics,
                    regression_analysis_results=st.session_state.regression_analysis_results if st.session_state.is_finetuned_comparison else {},
                    eval_config=st.session_state.eval_config,
                    is_finetuned_comparison=st.session_state.is_finetuned_comparison,
                )
                st.session_state.artifact_dir = artifact_dir
                run_id = os.path.basename(artifact_dir.rstrip("/\\"))
                # Zip it
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(artifact_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, artifact_dir)
                            zipf.write(file_path, arcname)
                zip_buffer.seek(0)
                st.session_state.zip_buffer = zip_buffer.getvalue()

                manifest_file = os.path.join(
                    artifact_dir, f"prompt_{run_id}_evidence_manifest.json")
                if os.path.exists(manifest_file):
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        st.session_state.manifest_content = f.read()
                else:
                    st.session_state.manifest_content = None

            st.success(f"Artifacts generated for run ID: `{run_id}`")
            st.rerun()

        if st.session_state.zip_buffer and st.session_state.artifact_dir:
            run_id = os.path.basename(
                st.session_state.artifact_dir.rstrip("/\\"))
            st.download_button(
                label=f"Download All Artifacts (Session_08_{run_id}.zip)",
                data=st.session_state.zip_buffer,
                file_name=f"Session_08_{run_id}.zip",
                mime="application/zip",
            )
        else:
            st.info(
                "Click 'Generate & Download Artifacts' to create your audit reports.")


# License
st.caption(
    '''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
'''
)
