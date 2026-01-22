

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import zipfile
import io
from datetime import datetime

# Import all business logic functions from source.py
# FIX: Change 'from source import *' to 'import source'
# This makes the 'source' module itself available,
# allowing access to its attributes (like EVAL_CONFIG) via source.EVAL_CONFIG
# and its functions via source.function_name().
import source 

st.set_page_config(page_title="QuLab: Lab 8: LLM Evaluation Harness (Inference & Fine-Tuning Risk)", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 8: LLM Evaluation Harness (Inference & Fine-Tuning Risk)")
st.divider()

# Your code starts here

# --- 0. st.session_state Initialization ---
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else None
if "current_page" not in st.session_state:
    st.session_state.current_page = "1. Data Upload"
if "prompts_df_raw" not in st.session_state:
    st.session_state.prompts_df_raw = None
if "baseline_outputs_df_raw" not in st.session_state:
    st.session_state.baseline_outputs_df_raw = None
if "finetuned_outputs_df_raw" not in st.session_state:
    st.session_state.finetuned_outputs_df_raw = None
if "eval_df" not in st.session_state:
    st.session_state.eval_df = pd.DataFrame()
if "is_finetuned_comparison" not in st.session_state:
    st.session_state.is_finetuned_comparison = False
if "aggregate_metrics" not in st.session_state:
    st.session_state.aggregate_metrics = {}
if "regression_analysis_results" not in st.session_state:
    st.session_state.regression_analysis_results = {}
if "eval_config" not in st.session_state:
    # Ensure EVAL_CONFIG from source.py is accessible and copied.
    # Assuming EVAL_CONFIG is a global dictionary in source.py.
    if hasattr(source, 'EVAL_CONFIG'):
        st.session_state.eval_config = source.EVAL_CONFIG.copy()
    else:
        st.session_state.eval_config = {} # Fallback if EVAL_CONFIG is not found in source
        st.error("Error: EVAL_CONFIG not found in source.py. Please ensure it's defined.")

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "zip_buffer" not in st.session_state:
    st.session_state.zip_buffer = None
if "manifest_content" not in st.session_state:
    st.session_state.manifest_content = None

# --- Sidebar Configuration and Navigation ---
st.sidebar.header("Configuration")
openai_key_input = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    help="Enter your OpenAI API key. This is optional if you are only evaluating pre-generated outputs (i.e., not calling OpenAI APIs from source.py).", 
    value=st.session_state.openai_api_key if st.session_state.openai_api_key else ""
)

if openai_key_input:
    st.session_state.openai_api_key = openai_key_input
    # Set the API key for `source.py` functions that might use it
    # This assumes `openai` module is available and imported/used within source.py
    if hasattr(source, 'openai') and hasattr(source.openai, 'api_key'):
        source.openai.api_key = st.session_state.openai_api_key
    else:
        st.warning("`source.openai.api_key` could not be set. Ensure `openai` module is imported and accessible in `source.py`.")
else:
    # Ensure a placeholder is set if not provided to avoid potential errors in source functions.
    # Actual OpenAI API calls will fail without a real key.
    if hasattr(source, 'openai') and hasattr(source.openai, 'api_key'):
        source.openai.api_key = "sk-placeholder"
    st.session_state.openai_api_key = None


st.sidebar.header("Navigation")
page_options = [
    "1. Data Upload",
    "2. Configure Evaluation Rules",
    "3. Run Evaluation",
    "4. View Scorecards",
    "5. Inspect Failure Exemplars",
    "6. Export Artifacts"
]
# Ensure index is an integer based on the current_page value
try:
    current_page_index = page_options.index(st.session_state.current_page)
except ValueError:
    current_page_index = 0 # Default to first page if not found
st.session_state.current_page = st.sidebar.selectbox("Go to...", options=page_options, index=current_page_index)

# --- Main Content Area ---

if st.session_state.current_page == "1. Data Upload":
    st.markdown(f"### 1. Setting the Stage: Loading Prompts and Model Outputs")
    st.markdown(f"As Alex, my first step in auditing InnovateCorp's RAG application is to gather the data. This includes a set of test prompts that cover various scenarios (simple questions, RAG-specific queries, sensitive topics), alongside the corresponding outputs from our current **baseline LLM** and a recently **fine-tuned LLM**. This allows me to compare performance and detect any regressions from fine-tuning.")
    st.markdown(f"The input data for our evaluation harness consists of:")
    st.markdown(f"*   **Prompt Set (`sample_prompts.csv`):** Contains `prompt_id`, `prompt_text`, an optional `expected_answer` for direct comparison, and `allowed_sources` which specifies the relevant documents for RAG contexts.")
    st.markdown(f"*   **Model Outputs (`sample_baseline_outputs.csv`, `sample_finetuned_outputs.csv`):** Each contains `prompt_id`, the `llm_output` (the generated response), and `sources_cited` (any sources explicitly mentioned by the LLM in its response).")
    st.markdown(f"This structure ensures that for each prompt, we have the original query, the desired outcome (if applicable), the context (allowed sources), and the actual LLM responses from different models.")

    st.markdown(f"---")
    st.markdown(f"#### Load Sample Data")
    st.info("You can quickly load pre-generated sample data to explore the application.")
    if st.button("Load Sample Data"):
        with st.spinner("Generating and loading mock data..."):
            source.generate_mock_data() # Ensure mock files exist
            st.session_state.eval_df, st.session_state.is_finetuned_comparison = source.load_evaluation_data(
                "sample_prompts.csv", "sample_baseline_outputs.csv", "sample_finetuned_outputs.csv"
            )
            st.success("Sample data loaded successfully!")
            st.rerun() # Rerun to refresh the dataframe display and move to next state

    st.markdown(f"---")
    st.markdown(f"#### Upload Your Own Data")
    uploaded_prompts_file = st.file_uploader("Upload Prompts CSV", type=["csv"], key="prompts_uploader")
    uploaded_baseline_file = st.file_uploader("Upload Baseline Outputs CSV", type=["csv"], key="baseline_outputs_uploader")
    uploaded_finetuned_file = st.file_uploader("Upload Fine-Tuned Outputs CSV (Optional)", type=["csv"], key="finetuned_outputs_uploader")

    # Read uploaded files into raw dataframes in session_state
    # Check if files are uploaded before attempting to read them.
    if uploaded_prompts_file:
        st.session_state.prompts_df_raw = pd.read_csv(uploaded_prompts_file)
    else:
        st.session_state.prompts_df_raw = None # Clear if file is removed
    if uploaded_baseline_file:
        st.session_state.baseline_outputs_df_raw = pd.read_csv(uploaded_baseline_file)
    else:
        st.session_state.baseline_outputs_df_raw = None # Clear if file is removed
    if uploaded_finetuned_file:
        st.session_state.finetuned_outputs_df_raw = pd.read_csv(uploaded_finetuned_file)
    else: # If finetuned file is removed, clear the raw df
        st.session_state.finetuned_outputs_df_raw = None

    load_uploaded_data_button_enabled = (
        st.session_state.prompts_df_raw is not None and
        st.session_state.baseline_outputs_df_raw is not None
    )

    if st.button("Load Uploaded Data", disabled=not load_uploaded_data_button_enabled):
        with st.spinner("Loading uploaded data..."):
            # Save uploaded files temporarily to disk for source.load_evaluation_data to read
            temp_dir = "temp_uploads_data_page"
            os.makedirs(temp_dir, exist_ok=True)
            
            prompts_path = os.path.join(temp_dir, "uploaded_prompts.csv")
            st.session_state.prompts_df_raw.to_csv(prompts_path, index=False)
            
            baseline_path = os.path.join(temp_dir, "uploaded_baseline_outputs.csv")
            st.session_state.baseline_outputs_df_raw.to_csv(baseline_path, index=False)

            finetuned_path = None
            if st.session_state.finetuned_outputs_df_raw is not None:
                finetuned_path = os.path.join(temp_dir, "uploaded_finetuned_outputs.csv")
                st.session_state.finetuned_outputs_df_raw.to_csv(finetuned_path, index=False)
            
            st.session_state.eval_df, st.session_state.is_finetuned_comparison = source.load_evaluation_data(
                prompts_path, baseline_path, finetuned_path
            )
            st.success("Uploaded data loaded successfully!")
            st.rerun() # Rerun to refresh the dataframe display and move to next state
    else:
        if not load_uploaded_data_button_enabled:
            st.warning("Please upload at least the Prompts CSV and Baseline Outputs CSV to load your own data.")


    if not st.session_state.eval_df.empty:
        st.markdown(f"#### Combined Evaluation Data Sample:")
        st.dataframe(st.session_state.eval_df.head())
        st.markdown(f"By displaying the head of the `eval_df` DataFrame, Alex quickly verifies that all prompt details are correctly matched with their respective baseline and fine-tuned LLM outputs. This ensures that subsequent evaluation steps will operate on a complete and correctly aligned dataset, preventing misattribution of scores or erroneous comparisons. The columns like `allowed_sources` and `sources_cited` are crucial for the faithfulness checks.")
    else:
        st.info("No data loaded yet. Please upload files or load sample data to proceed.")

elif st.session_state.current_page == "2. Configure Evaluation Rules":
    st.markdown(f"### 2. Configure Evaluation Rules")
    st.markdown(f"As Alex, I need to configure the specific parameters that will drive the LLM evaluation. These rules allow me to tailor the sensitivity of hallucination detection, define what constitutes over-specificity, and identify how the LLM should handle refusals. This customization ensures the evaluation aligns with InnovateCorp's risk profile and the specific use case of the internal knowledge assistant.")

    st.markdown(f"---")
    st.markdown(f"#### Hallucination Proxy Settings")
    st.markdown(f"Configure thresholds and keywords to detect potential hallucinations.")
    
    current_hallucination_threshold_ratio = st.session_state.eval_config.get("hallucination_threshold_ratio", 0.2)
    st.session_state.eval_config['hallucination_threshold_ratio'] = st.number_input(
        r"Hallucination Threshold Ratio ($$ R = \frac{\text{length}(\text{LLM Output})}{\text{length}(\text{Prompt Text})} $$)",
        value=current_hallucination_threshold_ratio,
        min_value=0.0, max_value=10.0, step=0.01,
        help="If the ratio of LLM output length (word count) to prompt length (word count) exceeds this, it may indicate excessive verbosity or hallucination. Where $R$ is the ratio, length(LLM Output) is the word count of the LLM's response, and length(Prompt Text) is the word count of the user's prompt."
    )
    
    current_over_specificity_keywords = ", ".join(st.session_state.eval_config.get("over_specificity_keywords", []))
    new_over_specificity_keywords = st.text_area(
        "Over-Specificity Keywords (comma-separated)",
        value=current_over_specificity_keywords,
        help="Keywords that, when present with unverified numerical details, may indicate over-specificity hallucinations. E.g., 'exactly', 'precisely'."
    )
    st.session_state.eval_config['over_specificity_keywords'] = [
        kw.strip() for kw in new_over_specificity_keywords.split(',') if kw.strip()
    ]

    st.markdown(f"---")
    st.markdown(f"#### Refusal and Over-Compliance Settings")
    st.markdown(f"Define phrases that indicate the LLM is refusing to answer or providing excessive disclaimers.")
    
    current_refusal_phrases = "\n".join(st.session_state.eval_config.get("refusal_phrases", []))
    new_refusal_phrases = st.text_area(
        "Refusal Phrases (one per line)",
        value=current_refusal_phrases,
        height=150,
        help="Phrases that identify a refusal to answer a prompt. E.g., 'I cannot answer questions that involve'."
    )
    st.session_state.eval_config['refusal_phrases'] = [
        phrase.strip() for phrase in new_refusal_phrases.split('\n') if phrase.strip()
    ]
    
    current_excessive_safety_disclaimers = "\n".join(st.session_state.eval_config.get("excessive_safety_disclaimers", []))
    new_excessive_safety_disclaimers = st.text_area(
        "Excessive Safety Disclaimers (one per line)",
        value=current_excessive_safety_disclaimers,
        height=150,
        help="Phrases that indicate overly cautious or unnecessary disclaimers. E.g., 'Please consult a professional'."
    )
    st.session_state.eval_config['excessive_safety_disclaimers'] = [
        phrase.strip() for phrase in new_excessive_safety_disclaimers.split('\n') if phrase.strip()
    ]

    st.markdown(f"---")
    st.markdown(f"#### Regression Analysis Settings")
    st.markdown(f"Set the sensitivity for detecting regressions when comparing fine-tuned models.")
    current_regression_threshold_delta = st.session_state.eval_config.get("regression_threshold_delta", 0.05)
    st.session_state.eval_config['regression_threshold_delta'] = st.number_input(
        "Regression Threshold Delta",
        value=current_regression_threshold_delta,
        min_value=0.0, max_value=1.0, step=0.01,
        help="A percentage increase in negative metrics (e.g., hallucination rate) beyond this threshold will flag a regression."
    )

    st.markdown(f"---")
    st.markdown(f"#### Citation Pattern")
    st.markdown(f"Specify the regex pattern used to detect citations in LLM outputs.")
    current_citation_pattern = st.session_state.eval_config.get("citation_pattern", r"\[\d+\]")
    st.session_state.eval_config['citation_pattern'] = st.text_input(
        "Citation Pattern (regex)",
        value=current_citation_pattern,
        help="Regular expression for detecting citations (e.g., [1], [2])."
    )
    st.markdown(f"Current Evaluation Configuration:")
    st.json(st.session_state.eval_config)

elif st.session_state.current_page == "3. Run Evaluation":
    st.markdown(f"### 3. Running the Evaluation Harness")
    st.markdown(f"With all individual checks defined and configured, Alex is now ready to run the complete evaluation harness across both the baseline and fine-tuned models. This will consolidate all per-response flags and calculate aggregate metrics that summarize the LLM's performance across the entire test set. This aggregated view is crucial for Maria, the Model Validator, to get a high-level understanding of the model's trustworthiness.")
    st.markdown(f"---")

    if st.button("Run Evaluation", disabled=st.session_state.eval_df.empty):
        if st.session_state.eval_df.empty:
            st.warning("Please load evaluation data first on the 'Data Upload' page.")
        else:
            # Update source.EVAL_CONFIG with the user-configured session state before running evaluations
            if hasattr(source, 'EVAL_CONFIG') and isinstance(source.EVAL_CONFIG, dict):
                source.EVAL_CONFIG.update(st.session_state.eval_config)
            else:
                st.error("Error: source.EVAL_CONFIG is not a modifiable dictionary. Evaluation cannot proceed.")
                st.stop()

            with st.spinner("Applying hallucination checks..."):
                st.session_state.eval_df = source.apply_hallucination_checks(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)
            with st.spinner("Applying faithfulness checks..."):
                st.session_state.eval_df = source.apply_faithfulness_checks(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)
            with st.spinner("Applying refusal and over-compliance checks..."):
                st.session_state.eval_df = source.apply_refusal_checks(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)
            
            with st.spinner("Aggregating metrics..."):
                st.session_state.eval_df, st.session_state.aggregate_metrics = source.run_evaluation_and_aggregate(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)
            
            if st.session_state.is_finetuned_comparison:
                with st.spinner("Performing regression analysis..."):
                    st.session_state.regression_analysis_results = source.perform_regression_analysis(st.session_state.aggregate_metrics)
            else:
                st.session_state.regression_analysis_results = {}
            
            st.success("Evaluation complete! You can now view scorecards and inspect failure exemplars.")
            st.rerun() # Rerun to refresh relevant displays

    else:
        if st.session_state.eval_df.empty:
            st.info("Load data on the 'Data Upload' page before running the evaluation.")
        else:
            st.info("Click 'Run Evaluation' to start the analysis.")

elif st.session_state.current_page == "4. View Scorecards":
    st.markdown(f"### 4. Visualizing Results and Aggregated Metrics")
    st.markdown(f"To effectively communicate the evaluation findings to Maria and David, Alex needs more than just raw numbers; he needs clear visualizations. This helps stakeholders quickly grasp the model's strengths and weaknesses, fostering transparency and trust.")
    st.markdown(f"---")

    if not st.session_state.aggregate_metrics:
        st.warning("Please run the evaluation first on the 'Run Evaluation' page.")
    else:
        st.markdown(f"#### Aggregate Scorecard")
        metrics_to_display = {model: {k: v for k, v in data.items() if k != 'total_prompts'} for model, data in st.session_state.aggregate_metrics.items()}
        metrics_df = pd.DataFrame.from_dict(metrics_to_display, orient='index')
        st.dataframe(metrics_df.style.format("{:.2%}").set_caption("LLM Evaluation Metrics"))
        st.markdown(f"Alex now has a high-level scorecard for both models. For the baseline model, he sees specific hallucination, faithfulness, and refusal rates. These aggregated numbers are immediately actionable for Maria and David, highlighting the overall trust posture of the LLM.")

        st.markdown(f"#### Comparison of Key LLM Evaluation Metrics")
        plt.figure(figsize=(12, 6))
        metrics_to_plot = ['hallucination_rate', 'faithfulness_score', 'refusal_rate', 'inappropriate_refusal_rate']
        
        # FIX: Build plot_df more efficiently to avoid FutureWarning with pd.concat in loop
        plot_data = []
        for model_prefix in ['baseline'] + (['finetuned'] if st.session_state.is_finetuned_comparison else []):
            for metric in metrics_to_plot:
                metric_value = st.session_state.aggregate_metrics[model_prefix].get(metric, 0) # Use .get with default 0 in case metric is missing
                plot_data.append({'Model': model_prefix.capitalize(), 'Metric': metric.replace('_', ' ').title(), 'Value': metric_value})
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            sns.barplot(x='Metric', y='Value', hue='Model', data=plot_df)
            plt.title('Comparison of Key LLM Evaluation Metrics')
            plt.ylabel('Rate / Score')
            plt.ylim(0, 1)
            st.pyplot(plt) # Display the plot
            plt.clf() # Clear the current figure to prevent layering issues on subsequent runs
        else:
            st.info("No metrics to plot. Ensure evaluation was run and metrics were generated.")
        
        if st.session_state.is_finetuned_comparison:
            st.markdown(f"#### Fine-Tuning Regression Analysis")
            if st.session_state.regression_analysis_results:
                for metric, delta in st.session_state.regression_analysis_results['deltas'].items():
                    st.markdown(f"**{metric.replace('_', ' ').title()} Delta:** `{delta:+.4f}`")
                
                if st.session_state.regression_analysis_results['regressions_flagged']:
                    st.error("!!! REGRESSION DETECTED !!!")
                    for flagged_metric in st.session_state.regression_analysis_results['flagged_metrics']:
                        st.markdown(f"- {flagged_metric.replace('_', ' ').title()}")
                else:
                    st.success("No regressions detected beyond threshold. Fine-tuning appears stable or improved.")
            else:
                st.info("Regression analysis not available. Ensure fine-tuned outputs were provided and evaluation was run.")
            st.markdown(f"The regression analysis output provides critical delta values for each metric. Alex can see if the `hallucination_rate` or `refusal_rate` increased after fine-tuning. If `REGRESSION DETECTED` is flagged, it's a clear signal that the fine-tuning introduced an undesirable behavior. This quantitative evidence is essential for Maria to make an informed decision on whether the fine-tuned model is fit for release or requires further iterations.")
        else:
            st.info("Fine-tuning regression analysis skipped as no fine-tuned outputs were provided.")

elif st.session_state.current_page == "5. Inspect Failure Exemplars":
    st.markdown(f"### 5. Inspect Failure Exemplars")
    st.markdown(f"To effectively communicate the evaluation findings to Maria and David, Alex needs concrete examples of failures. This helps stakeholders quickly grasp the model's strengths and weaknesses, fostering transparency and trust.")
    st.markdown(f"---")

    if st.session_state.eval_df.empty or not st.session_state.aggregate_metrics:
        st.warning("Please run the evaluation first on the 'Run Evaluation' page to identify failure exemplars.")
    else:
        st.markdown(f"#### Selected High-Risk Prompts")
        
        filtered_df = st.session_state.eval_df.copy() # Work on a copy
        
        # Collect all possible flag columns to determine "high risk"
        flag_suffixes = ['excessive_length_flag', 'unsupported_factual_claim_flag', 'over_specificity_flag',
                            'missing_allowed_source_flag', 'out_of_scope_reference_flag', 'uncited_assertion_flag',
                            'refusal_flag', 'excessive_disclaimer_flag', 'inappropriate_refusal_flag']
        
        # Create a combined high-risk flag for easier filtering
        filtered_df['is_any_high_risk'] = False
        for flag_suffix in flag_suffixes:
            baseline_col = f'baseline_{flag_suffix}'
            if baseline_col in filtered_df.columns:
                filtered_df['is_any_high_risk'] = filtered_df['is_any_high_risk'] | filtered_df[baseline_col]
            
            if st.session_state.is_finetuned_comparison:
                finetuned_col = f'finetuned_{flag_suffix}'
                if finetuned_col in filtered_df.columns:
                    filtered_df['is_any_high_risk'] = filtered_df['is_any_high_risk'] | filtered_df[finetuned_col]
        
        failure_ids_df = filtered_df[filtered_df['is_any_high_risk']].head(5) # Show up to 5 examples

        if not failure_ids_df.empty:
            for idx, row in failure_ids_df.iterrows():
                st.markdown(f"##### --- Prompt ID: `{row['prompt_id']}` ---")
                st.markdown(f"**Prompt Text:** {row['prompt_text']}")
                
                # Check if 'allowed_sources' or 'expected_answer' exist and are not NaN/None before displaying
                if 'allowed_sources' in row and pd.notna(row['allowed_sources']):
                    st.markdown(f"**Allowed Sources:** {row['allowed_sources']}")
                if 'expected_answer' in row and pd.notna(row['expected_answer']):
                    st.markdown(f"**Expected Answer:** {row['expected_answer']}")
                
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Baseline Model Output:**")
                    st.info(f"{row['baseline_output']}")
                    if 'baseline_sources_cited' in row and pd.notna(row['baseline_sources_cited']):
                        st.markdown(f"Sources Cited: {row['baseline_sources_cited']}")
                    
                    baseline_flags = []
                    for flag_suffix in flag_suffixes:
                        col_name = f'baseline_{flag_suffix}'
                        if col_name in row and row[col_name]: # Check if column exists and flag is True
                            baseline_flags.append(flag_suffix.replace('_', ' ').title())
                    if baseline_flags:
                        st.error(f"Detected Baseline Issues: {', '.join(baseline_flags)}")
                    else:
                        st.success("No Baseline Issues Detected")
                
                if st.session_state.is_finetuned_comparison:
                    with col2:
                        st.markdown(f"**Fine-Tuned Model Output:**")
                        st.info(f"{row['finetuned_output']}")
                        if 'finetuned_sources_cited' in row and pd.notna(row['finetuned_sources_cited']):
                            st.markdown(f"Sources Cited: {row['finetuned_sources_cited']}")
                        finetuned_flags = []
                        for flag_suffix in flag_suffixes:
                            col_name = f'finetuned_{flag_suffix}'
                            if col_name in row and row[col_name]: # Check if column exists and flag is True
                                finetuned_flags.append(flag_suffix.replace('_', ' ').title())
                        if finetuned_flags:
                            st.warning(f"Detected Fine-Tuned Issues: {', '.join(finetuned_flags)}")
                        else:
                            st.success("No Fine-Tuned Issues Detected")
                    
                    baseline_had_issue = bool(baseline_flags)
                    finetuned_had_issue = bool(finetuned_flags)

                    if finetuned_had_issue and not baseline_had_issue:
                        st.markdown("  ***Note: This prompt showed a regression in the fine-tuned model compared to baseline.***")
                    elif baseline_had_issue and not finetuned_had_issue:
                        st.markdown("  ***Note: Fine-tuned model improved for this prompt compared to baseline.***")
                st.markdown("---")
        else:
            st.info("No high-risk prompts identified based on current evaluation. All models appear to be performing well.")
    st.markdown(f"The 'Failure Exemplars' section is particularly valuable: for `P002`, Alex sees how the baseline hallucinated a wrong percentage and source, while the fine-tuned model corrected it – a clear improvement. However, for `P006`, the fine-tuned model introduced an inappropriate refusal, confirming the regression seen in the aggregate metrics. This granular view allows Alex to perform root cause analysis: Was the fine-tuning data flawed? Did new safety guardrails overcorrect? This directly informs subsequent model iterations and prompt engineering strategies, bridging the gap between raw data and actionable model improvements.")

elif st.session_state.current_page == "6. Export Artifacts":
    st.markdown(f"### 6. Generating Evaluation Artifacts")
    st.markdown(f"The final step for Alex is to generate all required evaluation artifacts. These artifacts serve as an auditable record for Maria (Model Validator) and David (AI Risk Lead), providing comprehensive evidence for model approval, compliance, and future risk assessments. Each artifact will be saved with a SHA-256 hash to ensure data integrity and traceability. This adheres to InnovateCorp's strict governance requirements.")
    st.markdown(f"---")

    if not st.session_state.aggregate_metrics:
        st.warning("Please run the evaluation first on the 'Run Evaluation' page to generate artifacts.")
    else:
        # Check if the 'run_id' is from a previous generation in the current session or generate a new one
        if st.session_state.run_id is None:
            st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # This button triggers the artifact generation and subsequent download.
        if st.button("Generate & Download Artifacts"):
            with st.spinner(f"Generating artifacts for run ID: `{st.session_state.run_id}`..."):
                # Update source.EVAL_CONFIG with the user-configured session state before generating artifacts
                if hasattr(source, 'EVAL_CONFIG') and isinstance(source.EVAL_CONFIG, dict):
                    source.EVAL_CONFIG.update(st.session_state.eval_config)
                else:
                    st.error("Error: source.EVAL_CONFIG is not a modifiable dictionary. Artifact generation cannot proceed.")
                    st.stop()

                # Define the specific output path for this run
                # Assuming source.EVAL_CONFIG["output_dir"] provides the base path, e.g., "reports"
                output_base_path_from_config = source.EVAL_CONFIG.get("output_dir", "reports") # Default to "reports"
                current_output_path = os.path.join(output_base_path_from_config, st.session_state.run_id)
                os.makedirs(current_output_path, exist_ok=True)
                
                # Temporarily modify EVAL_CONFIG in source.py to point to our specific run directory
                original_output_dir_in_source_config = source.EVAL_CONFIG.get("output_dir")
                if original_output_dir_in_source_config is not None: 
                    source.EVAL_CONFIG["output_dir"] = current_output_path

                try:
                    source.generate_artifacts(
                        st.session_state.eval_df.copy(),
                        st.session_state.aggregate_metrics,
                        st.session_state.regression_analysis_results if st.session_state.is_finetuned_comparison else {},
                        st.session_state.eval_config, # Passing `st.session_state.eval_config` as argument as per spec
                        st.session_state.is_finetuned_comparison
                    )
                finally:
                    # Always restore original config value after use, even if generation fails
                    if original_output_dir_in_source_config is not None:
                        source.EVAL_CONFIG["output_dir"] = original_output_dir_in_source_config

                # Now, zip the generated folder
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk through the directory we just created artifacts in
                    for root, _, files in os.walk(current_output_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Archive name should be relative to the run_id folder itself
                            # This means files like `reports/timestamp/file.csv` become `file.csv` in the zip root.
                            arcname = os.path.relpath(file_path, current_output_path)
                            zipf.write(file_path, arcname)
                zip_buffer.seek(0)
                st.session_state.zip_buffer = zip_buffer.getvalue()

                # Read the manifest content for display
                manifest_file = os.path.join(current_output_path, f"prompt_{st.session_state.run_id}_evidence_manifest.json")
                if os.path.exists(manifest_file):
                    with open(manifest_file, 'r') as f:
                        st.session_state.manifest_content = f.read()
                
                st.success(f"Artifacts generated and bundled for run ID: `{st.session_state.run_id}`")
                st.rerun() # Rerun to enable the download button (download_button is evaluated on each rerun)

        # Download button should be separate from generate button to allow state refresh
        if st.session_state.zip_buffer:
            st.download_button(
                label=f"Download All Artifacts (Session_08_{st.session_state.run_id}.zip)",
                data=st.session_state.zip_buffer,
                file_name=f"Session_08_{st.session_state.run_id}.zip",
                mime="application/zip",
                disabled=not st.session_state.zip_buffer
            )
            
            if st.session_state.manifest_content:
                st.markdown(f"---")
                st.markdown(f"#### Evidence Manifest (`evidence_manifest.json`)")
                # Ensure it's parsed as JSON before displaying with st.json
                st.json(json.loads(st.session_state.manifest_content))
                st.markdown(f"Alex has successfully generated a suite of audit-ready artifacts. The console output confirms each file's creation and its SHA-256 hash. These files provide comprehensive documentation: the `executive_summary.md` gives Maria and David a quick, high-level overview, while the detailed JSON files offer the granular data needed for deep dives or compliance audits. The `evidence_manifest.json` with its cryptographic hashes ensures the integrity and non-repudiation of all generated evaluation evidence, fulfilling InnovateCorp's strict security and governance requirements. This complete package enables informed decision-making and establishes a clear audit trail for the LLM's trustworthiness.")
        else:
            st.info("Click 'Generate & Download Artifacts' to create your audit reports.")

