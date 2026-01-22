
# Streamlit Application Specification: LLM Evaluation Harness

## 1. Application Overview

### Purpose of the Application

This Streamlit application serves as an LLM evaluation harness, designed to provide deterministic and repeatable evaluation of LLM inference-time behavior and fine-tuning regressions. It aims to quantify hallucination risk, verify citation faithfulness, detect refusal and over-compliance behavior, and compare fine-tuned models against baselines. The application generates auditable artifacts suitable for model approval, vendor assessment, and release gating, ensuring trust and stability in AI solutions for enterprise use cases like compliance assistants or internal knowledge bases.

### High-Level Story Flow

The application walks Alex, an LLM Engineer at InnovateCorp, through a structured workflow to assess the quality and reliability of an LLM.

1.  **Data Ingestion**: Alex starts by uploading a set of test prompts and the corresponding LLM outputs from a baseline model and, optionally, a fine-tuned model. The app also provides sample data for quick exploration.
2.  **Configuration**: Alex then fine-tunes evaluation parameters such as hallucination thresholds, over-specificity keywords, and refusal phrases to align with InnovateCorp's specific risk posture and requirements.
3.  **Evaluation Execution**: With data loaded and rules configured, Alex initiates the evaluation process. The application applies various proxy metrics for hallucination, faithfulness, and refusal, and performs a regression analysis if a fine-tuned model is provided.
4.  **Results Review**: Alex then views an aggregated scorecard, visualizing key metrics and comparing baseline vs. fine-tuned model performance. This gives an immediate high-level understanding of model behavior.
5.  **Failure Analysis**: To drill down, Alex inspects "failure exemplars" – specific high-risk prompts that highlight identified issues and potential regressions, providing concrete evidence for root cause analysis.
6.  **Artifact Export**: Finally, Alex generates a suite of auditable reports and artifacts, complete with SHA-256 hashes for integrity, ready to be presented to Maria (Model Validator) and David (AI Risk Lead) for model approval and risk assessment.

This workflow enables Alex to systematically evaluate LLMs, identify areas for improvement, and provide objective, verifiable evidence for stakeholders, ensuring the trustworthy deployment of AI within InnovateCorp.

## 2. Code Requirements

### Import Statement

```python
from source import *
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import zipfile
import io
```

### `st.session_state` Design

The following `st.session_state` keys will be used to maintain state across user interactions and simulated pages:

| Key                           | Initial Value                                                                                                        | Description                                                               |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| `openai_api_key`              | `None` or `os.getenv("OPENAI_API_KEY")`                                                                              | Stores the OpenAI API key provided by the user.                           |
| `current_page`                | `"1. Data Upload"`                                                                                                   | Controls the currently displayed content section in the main area.        |
| `prompts_df_raw`              | `None`                                                                                                               | Raw DataFrame from uploaded `sample_prompts.csv`.                         |
| `baseline_outputs_df_raw`     | `None`                                                                                                               | Raw DataFrame from uploaded `sample_baseline_outputs.csv`.                |
| `finetuned_outputs_df_raw`    | `None`                                                                                                               | Raw DataFrame from uploaded `sample_finetuned_outputs.csv`.               |
| `eval_df`                     | `pd.DataFrame()`                                                                                                     | The merged DataFrame containing prompts and LLM outputs, with evaluation flags and results. |
| `is_finetuned_comparison`     | `False`                                                                                                              | Boolean, `True` if fine-tuned outputs are provided and loaded.            |
| `aggregate_metrics`           | `{}`                                                                                                                 | Dictionary storing aggregated evaluation metrics for baseline and fine-tuned models. |
| `regression_analysis_results` | `{}`                                                                                                                 | Dictionary storing results of the regression analysis.                    |
| `eval_config`                 | `source.EVAL_CONFIG.copy()`                                                                                          | Modifiable copy of the evaluation configuration.                          |
| `run_id`                      | `None`                                                                                                               | Unique identifier for a specific evaluation run (timestamp-based).        |
| `zip_buffer`                  | `None`                                                                                                               | In-memory buffer for the generated `.zip` artifact file.                  |
| `manifest_content`            | `None`                                                                                                               | Content of the `evidence_manifest.json` as a string.                      |

**Initialization:**
All `st.session_state` keys will be initialized at the start of the `app.py` script using `if "key" not in st.session_state: st.session_state["key"] = initial_value`.

**Update Points:**

*   `openai_api_key`: Updated by `st.sidebar.text_input` on user input.
*   `current_page`: Updated by `st.sidebar.selectbox` on user selection.
*   `prompts_df_raw`, `baseline_outputs_df_raw`, `finetuned_outputs_df_raw`: Updated when files are uploaded or sample data is loaded.
*   `eval_df`, `is_finetuned_comparison`: Updated after `load_evaluation_data` is called.
*   `eval_config`: Updated by input widgets on the "Configure Evaluation Rules" page.
*   `aggregate_metrics`, `regression_analysis_results`: Updated after the "Run Evaluation" button is clicked and `run_evaluation_and_aggregate` and `perform_regression_analysis` are called.
*   `run_id`, `zip_buffer`, `manifest_content`: Updated after the "Generate & Download Artifacts" button is clicked and `generate_artifacts` is called.

**Read Across Pages:**
All keys in `st.session_state` are implicitly available to all parts of the Streamlit script, allowing data to persist and be read across different UI sections as the `current_page` changes.

### UI Interactions and Function Invocations (`source.py` calls)

**Sidebar:**

*   **OpenAI API Key Input:**
    *   `st.sidebar.text_input("OpenAI API Key", type="password")`
    *   **Updates:** `st.session_state.openai_api_key`
    *   **Interaction:** Sets `openai.api_key` directly from `source.py` after `source.py` import.
*   **Page Navigation:**
    *   `st.sidebar.selectbox("Navigate", options=[...], key="current_page")`
    *   **Updates:** `st.session_state.current_page`

**Main Content - "1. Data Upload" Page:**

*   **Load Sample Data Button:**
    *   `st.button("Load Sample Data")`
    *   **Calls:** `source.generate_mock_data()` (to ensure files exist).
    *   **Calls:** `st.session_state.eval_df, st.session_state.is_finetuned_comparison = source.load_evaluation_data("sample_prompts.csv", "sample_baseline_outputs.csv", "sample_finetuned_outputs.csv")`
    *   **Updates:** `st.session_state.eval_df`, `st.session_state.is_finetuned_comparison`
*   **File Uploaders:**
    *   `st.file_uploader("Upload Prompts CSV", type=["csv"], key="prompts_uploader")`
    *   `st.file_uploader("Upload Baseline Outputs CSV", type=["csv"], key="baseline_outputs_uploader")`
    *   `st.file_uploader("Upload Fine-Tuned Outputs CSV (Optional)", type=["csv"], key="finetuned_outputs_uploader")`
    *   **Updates:** `st.session_state.prompts_df_raw`, `st.session_state.baseline_outputs_df_raw`, `st.session_state.finetuned_outputs_df_raw` (after `pd.read_csv`).
*   **Load Uploaded Data Button:**
    *   `st.button("Load Uploaded Data")` (Only enabled if all required files are uploaded)
    *   **Calls:** `st.session_state.eval_df, st.session_state.is_finetuned_comparison = source.load_evaluation_data(prompts_file_path, baseline_file_path, finetuned_file_path)` (using temporary paths for uploaded files).
    *   **Updates:** `st.session_state.eval_df`, `st.session_state.is_finetuned_comparison`

**Main Content - "2. Configure Evaluation Rules" Page:**

*   **Configuration Widgets:**
    *   `st.number_input("Hallucination Threshold Ratio", key="hallucination_threshold_ratio_input")`
    *   `st.text_area("Over-Specificity Keywords (comma-separated)", key="over_specificity_keywords_input")`
    *   `st.text_area("Refusal Phrases (line-separated)", key="refusal_phrases_input")`
    *   `st.text_area("Excessive Safety Disclaimers (line-separated)", key="excessive_safety_disclaimers_input")`
    *   `st.number_input("Regression Threshold Delta", key="regression_threshold_delta_input")`
    *   `st.text_input("Citation Pattern (regex)", key="citation_pattern_input")`
    *   **Updates:** `st.session_state.eval_config` based on widget values.

**Main Content - "3. Run Evaluation" Page:**

*   **Run Evaluation Button:**
    *   `st.button("Run Evaluation", disabled=not st.session_state.eval_df.empty)`
    *   **Calls (in order):**
        1.  `st.session_state.eval_df = source.apply_hallucination_checks(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)`
        2.  `st.session_state.eval_df = source.apply_faithfulness_checks(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)`
        3.  `st.session_state.eval_df = source.apply_refusal_checks(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)`
        4.  `st.session_state.eval_df, st.session_state.aggregate_metrics = source.run_evaluation_and_aggregate(st.session_state.eval_df.copy(), st.session_state.is_finetuned_comparison)`
        5.  `st.session_state.regression_analysis_results = source.perform_regression_analysis(st.session_state.aggregate_metrics)` (if `is_finetuned_comparison` is True)
    *   **Updates:** `st.session_state.eval_df`, `st.session_state.aggregate_metrics`, `st.session_state.regression_analysis_results`

**Main Content - "6. Export Artifacts" Page:**

*   **Generate & Download Artifacts Button:**
    *   `st.button("Generate & Download Artifacts", disabled=not st.session_state.aggregate_metrics)`
    *   **Calls:** `source.generate_artifacts(st.session_state.eval_df.copy(), st.session_state.aggregate_metrics, st.session_state.regression_analysis_results, st.session_state.eval_config, st.session_state.is_finetuned_comparison)` (This function needs to be adapted to return the zip file content and manifest for Streamlit download, rather than saving to disk. A helper wrapper function will be used if `source.py` cannot be directly changed to return these. *Self-correction: The problem states "do not redefine, rewrite, stub, or duplicate them." So, I cannot change `generate_artifacts` to return a zip_buffer directly. Instead, `generate_artifacts` will save to a *temporary* local directory, and then the Streamlit app will zip *that directory* into a `zip_buffer` for download.*)
    *   **Updates:** `st.session_state.run_id`, `st.session_state.zip_buffer`, `st.session_state.manifest_content`.
*   **Download Button:**
    *   `st.download_button("Download All Artifacts as .zip", data=st.session_state.zip_buffer, file_name=f"Session_08_{st.session_state.run_id}.zip", mime="application/zip", disabled=not st.session_state.zip_buffer)`

### Markdown Content

#### Application Title and Sidebar

```python
st.set_page_config(layout="wide", page_title="LLM Evaluation Harness")
st.title("🛡️ LLM Evaluation Harness: Auditing for Hallucination, Faithfulness, and Regression Risk")

st.sidebar.header("Configuration")
st.session_state.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key. This is optional if you are only evaluating pre-generated outputs.")
if st.session_state.openai_api_key:
    # Set the API key for `source.py` functions that might use it
    source.openai.api_key = st.session_state.openai_api_key
else:
    source.openai.api_key = "sk-placeholder" # Ensure a placeholder is set if not provided to avoid errors.

st.sidebar.header("Navigation")
page_options = [
    "1. Data Upload",
    "2. Configure Evaluation Rules",
    "3. Run Evaluation",
    "4. View Scorecards",
    "5. Inspect Failure Exemplars",
    "6. Export Artifacts"
]
st.session_state.current_page = st.sidebar.selectbox("Go to...", options=page_options, index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0)
```

#### Page: 1. Data Upload

```python
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

st.markdown(f"---")
st.markdown(f"#### Upload Your Own Data")
uploaded_prompts_file = st.file_uploader("Upload Prompts CSV", type=["csv"], key="prompts_uploader")
uploaded_baseline_file = st.file_uploader("Upload Baseline Outputs CSV", type=["csv"], key="baseline_outputs_uploader")
uploaded_finetuned_file = st.file_uploader("Upload Fine-Tuned Outputs CSV (Optional)", type=["csv"], key="finetuned_outputs_uploader")

if uploaded_prompts_file and uploaded_baseline_file:
    st.session_state.prompts_df_raw = pd.read_csv(uploaded_prompts_file)
    st.session_state.baseline_outputs_df_raw = pd.read_csv(uploaded_baseline_file)
    if uploaded_finetuned_file:
        st.session_state.finetuned_outputs_df_raw = pd.read_csv(uploaded_finetuned_file)
    else:
        st.session_state.finetuned_outputs_df_raw = None

    if st.button("Load Uploaded Data"):
        with st.spinner("Loading uploaded data..."):
            # Save uploaded files temporarily to disk for source.load_evaluation_data to read
            temp_dir = "temp_uploads"
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
else:
    st.warning("Please upload at least the Prompts CSV and Baseline Outputs CSV to load your own data.")

if not st.session_state.eval_df.empty:
    st.markdown(f"#### Combined Evaluation Data Sample:")
    st.dataframe(st.session_state.eval_df.head())
    st.markdown(f"By displaying the head of the `eval_df` DataFrame, Alex quickly verifies that all prompt details are correctly matched with their respective baseline and fine-tuned LLM outputs. This ensures that subsequent evaluation steps will operate on a complete and correctly aligned dataset, preventing misattribution of scores or erroneous comparisons. The columns like `allowed_sources` and `sources_cited` are crucial for the faithfulness checks.")
else:
    st.info("No data loaded yet. Please upload files or load sample data to proceed.")
```

#### Page: 2. Configure Evaluation Rules

```python
st.markdown(f"### 2. Configure Evaluation Rules")
st.markdown(f"As Alex, I need to configure the specific parameters that will drive the LLM evaluation. These rules allow me to tailor the sensitivity of hallucination detection, define what constitutes over-specificity, and identify how the LLM should handle refusals. This customization ensures the evaluation aligns with InnovateCorp's risk profile and the specific use case of the internal knowledge assistant.")

st.markdown(f"---")
st.markdown(f"#### Hallucination Proxy Settings")
st.markdown(f"Configure thresholds and keywords to detect potential hallucinations.")
st.session_state.eval_config['hallucination_threshold_ratio'] = st.number_input(
    f"Answer Length vs. Prompt Length Ratio Threshold ($$ R = \\frac{{\\text{{length}}(\\text{{LLM Output}})}}{{\\text{{length}}(\\text{{Prompt Text}})}} $$)",
    value=st.session_state.eval_config.get("hallucination_threshold_ratio", 0.2),
    min_value=0.0, max_value=1.0, step=0.01,
    help="If the ratio of LLM output length to prompt length exceeds this, it may indicate excessive verbosity or hallucination. Where $R$ is the ratio, length(LLM Output) is the word count of the LLM's response, and length(Prompt Text) is the word count of the user's prompt."
)
st.session_state.eval_config['over_specificity_keywords'] = [
    kw.strip() for kw in st.text_area(
        "Over-Specificity Keywords (comma-separated)",
        value=", ".join(st.session_state.eval_config.get("over_specificity_keywords", [])),
        help="Keywords that, when present with unverified numerical details, may indicate over-specificity hallucinations. E.g., 'exactly', 'precisely'."
    ).split(',') if kw.strip()
]

st.markdown(f"---")
st.markdown(f"#### Refusal and Over-Compliance Settings")
st.markdown(f"Define phrases that indicate the LLM is refusing to answer or providing excessive disclaimers.")
st.session_state.eval_config['refusal_phrases'] = [
    phrase.strip() for phrase in st.text_area(
        "Refusal Phrases (one per line)",
        value="\n".join(st.session_state.eval_config.get("refusal_phrases", [])),
        height=150,
        help="Phrases that identify a refusal to answer a prompt. E.g., 'I cannot answer questions that involve'."
    ).split('\n') if phrase.strip()
]
st.session_state.eval_config['excessive_safety_disclaimers'] = [
    phrase.strip() for phrase in st.text_area(
        "Excessive Safety Disclaimers (one per line)",
        value="\n".join(st.session_state.eval_config.get("excessive_safety_disclaimers", [])),
        height=150,
        help="Phrases that indicate overly cautious or unnecessary disclaimers. E.g., 'Please consult a professional'."
    ).split('\n') if phrase.strip()
]

st.markdown(f"---")
st.markdown(f"#### Regression Analysis Settings")
st.markdown(f"Set the sensitivity for detecting regressions when comparing fine-tuned models.")
st.session_state.eval_config['regression_threshold_delta'] = st.number_input(
    "Regression Threshold Delta",
    value=st.session_state.eval_config.get("regression_threshold_delta", 0.05),
    min_value=0.0, max_value=1.0, step=0.01,
    help="A percentage increase in negative metrics (e.g., hallucination rate) beyond this threshold will flag a regression."
)

st.markdown(f"---")
st.markdown(f"#### Citation Pattern")
st.markdown(f"Specify the regex pattern used to detect citations in LLM outputs.")
st.session_state.eval_config['citation_pattern'] = st.text_input(
    "Citation Pattern (regex)",
    value=st.session_state.eval_config.get("citation_pattern", r"\[\d+\]"),
    help="Regular expression for detecting citations (e.g., [1], [2])."
)
st.markdown(f"Current Evaluation Configuration:")
st.json(st.session_state.eval_config)
```

#### Page: 3. Run Evaluation

```python
st.markdown(f"### 3. Running the Evaluation Harness")
st.markdown(f"With all individual checks defined and configured, Alex is now ready to run the complete evaluation harness across both the baseline and fine-tuned models. This will consolidate all per-response flags and calculate aggregate metrics that summarize the LLM's performance across the entire test set. This aggregated view is crucial for Maria, the Model Validator, to get a high-level understanding of the model's trustworthiness.")
st.markdown(f"---")

if st.button("Run Evaluation", disabled=st.session_state.eval_df.empty):
    if st.session_state.eval_df.empty:
        st.warning("Please load evaluation data first on the 'Data Upload' page.")
    else:
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
else:
    if st.session_state.eval_df.empty:
        st.info("Load data on the 'Data Upload' page before running the evaluation.")
    else:
        st.info("Click 'Run Evaluation' to start the analysis.")
```

#### Page: 4. View Scorecards

```python
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
    plot_df = pd.DataFrame()
    for model_prefix in ['baseline'] + (['finetuned'] if st.session_state.is_finetuned_comparison else []):
        for metric in metrics_to_plot:
            plot_df = pd.concat([plot_df, pd.DataFrame({'Model': model_prefix.capitalize(), 'Metric': metric.replace('_', ' ').title(), 'Value': st.session_state.aggregate_metrics[model_prefix][metric]}, index=[0])])
    
    sns.barplot(x='Metric', y='Value', hue='Model', data=plot_df)
    plt.title('Comparison of Key LLM Evaluation Metrics')
    plt.ylabel('Rate / Score')
    plt.ylim(0, 1)
    st.pyplot(plt) # Display the plot

    if st.session_state.is_finetuned_comparison:
        st.markdown(f"#### Fine-Tuning Regression Analysis")
        if st.session_state.regression_analysis_results:
            for metric, delta in st.session_state.regression_analysis_results['deltas'].items():
                st.markdown(f"**{metric.replace('_', ' ').title()} Delta:** `{delta:+.4f}`")
            
            if st.session_state.regression_analysis_results['regressions_flagged']:
                st.error("!!! REGRESSION DETECTED !!!")
                for flagged_metric in st.session_state.regression_analysis_results['flagged_metrics']:
                    st.markdown(f"- {flagged_metric}")
            else:
                st.success("No regressions detected beyond threshold. Fine-tuning appears stable or improved.")
        else:
            st.info("Regression analysis not available. Ensure fine-tuned outputs were provided and evaluation was run.")
        st.markdown(f"The regression analysis output provides critical delta values for each metric. Alex can see if the `hallucination_rate` or `refusal_rate` increased after fine-tuning. If `REGRESSION DETECTED` is flagged, it's a clear signal that the fine-tuning introduced an undesirable behavior. This quantitative evidence is essential for Maria to make an informed decision on whether the fine-tuned model is fit for release or requires further iterations.")
    else:
        st.info("Fine-tuning regression analysis skipped as no fine-tuned outputs were provided.")
```

#### Page: 5. Inspect Failure Exemplars

```python
st.markdown(f"### 5. Inspect Failure Exemplars")
st.markdown(f"To effectively communicate the evaluation findings to Maria and David, Alex needs concrete examples of failures. This helps stakeholders quickly grasp the model's strengths and weaknesses, fostering transparency and trust.")
st.markdown(f"---")

if st.session_state.eval_df.empty or not st.session_state.aggregate_metrics:
    st.warning("Please run the evaluation first on the 'Run Evaluation' page to identify failure exemplars.")
else:
    st.markdown(f"#### Selected High-Risk Prompts")
    # Identify unique failure IDs
    failure_ids_df = st.session_state.eval_df[
        (st.session_state.eval_df['baseline_high_risk_prompt']) | 
        (st.session_state.is_finetuned_comparison and st.session_state.eval_df['finetuned_high_risk_prompt'])
    ].head(5) # Show up to 5 examples

    if not failure_ids_df.empty:
        for idx, row in failure_ids_df.iterrows():
            st.markdown(f"##### --- Prompt ID: `{row['prompt_id']}` ---")
            st.markdown(f"**Prompt Text:** {row['prompt_text']}")
            if row['allowed_sources']:
                st.markdown(f"**Allowed Sources:** {row['allowed_sources']}")
            if row['expected_answer']:
                st.markdown(f"**Expected Answer:** {row['expected_answer']}")
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Baseline Model Output:**")
                st.info(f"{row['baseline_output']}")
                if row['baseline_sources_cited']:
                    st.markdown(f"Sources Cited: {row['baseline_sources_cited']}")
                
                baseline_flags = []
                for flag in ['excessive_length_flag', 'unsupported_factual_claim_flag', 'over_specificity_flag',
                             'missing_allowed_source_flag', 'out_of_scope_reference_flag', 'uncited_assertion_flag',
                             'refusal_flag', 'excessive_disclaimer_flag', 'inappropriate_refusal_flag']:
                    if row[f'baseline_{flag}']:
                        baseline_flags.append(flag.replace('_', ' ').title())
                if baseline_flags:
                    st.error(f"Detected Baseline Issues: {', '.join(baseline_flags)}")
                else:
                    st.success("No Baseline Issues Detected")
            
            if st.session_state.is_finetuned_comparison:
                with col2:
                    st.markdown(f"**Fine-Tuned Model Output:**")
                    st.info(f"{row['finetuned_output']}")
                    if row['finetuned_sources_cited']:
                        st.markdown(f"Sources Cited: {row['finetuned_sources_cited']}")
                    finetuned_flags = []
                    for flag in ['excessive_length_flag', 'unsupported_factual_claim_flag', 'over_specificity_flag',
                                 'missing_allowed_source_flag', 'out_of_scope_reference_flag', 'uncited_assertion_flag',
                                 'refusal_flag', 'excessive_disclaimer_flag', 'inappropriate_refusal_flag']:
                        if row[f'finetuned_{flag}']:
                            finetuned_flags.append(flag.replace('_', ' ').title())
                    if finetuned_flags:
                        st.warning(f"Detected Fine-Tuned Issues: {', '.join(finetuned_flags)}")
                    else:
                        st.success("No Fine-Tuned Issues Detected")
                
                if row['finetuned_high_risk_prompt'] and not row['baseline_high_risk_prompt']:
                    st.markdown("  ***Note: This prompt showed a regression in the fine-tuned model compared to baseline.***")
                elif row['baseline_high_risk_prompt'] and not row['finetuned_high_risk_prompt']:
                    st.markdown("  ***Note: Fine-tuned model improved for this prompt compared to baseline.***")
            st.markdown("---")
    else:
        st.info("No high-risk prompts identified based on current evaluation. All models appear to be performing well.")
    st.markdown(f"The 'Failure Exemplars' section is particularly valuable: for `P002`, Alex sees how the baseline hallucinated a wrong percentage and source, while the fine-tuned model corrected it – a clear improvement. However, for `P006`, the fine-tuned model introduced an inappropriate refusal, confirming the regression seen in the aggregate metrics. This granular view allows Alex to perform root cause analysis: Was the fine-tuning data flawed? Did new safety guardrails overcorrect? This directly informs subsequent model iterations and prompt engineering strategies, bridging the gap between raw data and actionable model improvements.")
```

#### Page: 6. Export Artifacts

```python
st.markdown(f"### 6. Generating Evaluation Artifacts")
st.markdown(f"The final step for Alex is to generate all required evaluation artifacts. These artifacts serve as an auditable record for Maria (Model Validator) and David (AI Risk Lead), providing comprehensive evidence for model approval, compliance, and future risk assessments. Each artifact will be saved with a SHA-256 hash to ensure data integrity and traceability. This adheres to InnovateCorp's strict governance requirements.")
st.markdown(f"---")

if not st.session_state.aggregate_metrics:
    st.warning("Please run the evaluation first on the 'Run Evaluation' page to generate artifacts.")
else:
    if st.button("Generate & Download Artifacts"):
        with st.spinner("Generating artifacts..."):
            # Ensure output directory exists for source.py
            output_base_dir = st.session_state.eval_config["output_dir"]
            os.makedirs(output_base_dir, exist_ok=True)

            # Generate artifacts using the source function, which writes to disk
            # We need to temporarily redirect output_dir if source.py doesn't return the path
            # Or assume source.generate_artifacts creates a timestamped folder, and we find it.
            # Given the existing source.py, it creates reports/session08/<run_id>/
            # We need to capture that run_id to zip the correct folder.
            
            # This is a critical point. `generate_artifacts` in source.py creates a new dir each time.
            # We need to pass a specific output_dir and then zip its content.
            # Re-calling it is fine, but we need to control the `run_id` for consistency in Streamlit.
            
            # Generate a consistent run_id for this session
            st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_output_path = os.path.join(output_base_dir, st.session_state.run_id)
            os.makedirs(current_output_path, exist_ok=True)
            
            # Temporarily modify EVAL_CONFIG to use our chosen run_id path
            original_output_dir = st.session_state.eval_config["output_dir"]
            st.session_state.eval_config["output_dir"] = current_output_path
            
            source.generate_artifacts(
                st.session_state.eval_df.copy(),
                st.session_state.aggregate_metrics,
                st.session_state.regression_analysis_results if st.session_state.is_finetuned_comparison else {},
                st.session_state.eval_config,
                st.session_state.is_finetuned_comparison
            )
            
            # Restore original config after use
            st.session_state.eval_config["output_dir"] = original_output_dir

            # Now, zip the generated folder
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(current_output_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Archive name should be relative to the run_id folder
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
                st.json(json.loads(st.session_state.manifest_content))
                st.markdown(f"Alex has successfully generated a suite of audit-ready artifacts. The console output confirms each file's creation and its SHA-256 hash. These files provide comprehensive documentation: the `executive_summary.md` gives Maria and David a quick, high-level overview, while the detailed JSON files offer the granular data needed for deep dives or compliance audits. The `evidence_manifest.json` with its cryptographic hashes ensures the integrity and non-repudiation of all generated evaluation evidence, fulfilling InnovateCorp's strict security and governance requirements. This complete package enables informed decision-making and establishes a clear audit trail for the LLM's trustworthiness.")
    else:
        st.info("Click 'Generate & Download Artifacts' to create your audit reports.")
```
