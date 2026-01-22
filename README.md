Here's a comprehensive `README.md` for your Streamlit application lab project, designed to be professional and informative.

---

# QuLab: Lab 8: LLM Evaluation Harness (Inference & Fine-Tuning Risk)

![Streamlit App Screenshot Placeholder](https://via.placeholder.com/1200x600?text=LLM+Evaluation+Harness+Screenshot)
*(Replace this placeholder with an actual screenshot or GIF of your running Streamlit application)*

## Table of Contents

1.  [Project Title and Description](#1-project-title-and-description)
2.  [Features](#2-features)
3.  [Getting Started](#3-getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [OpenAI API Key Configuration](#openai-api-key-configuration)
4.  [Usage](#4-usage)
    *   [Application Workflow](#application-workflow)
    *   [Navigation](#navigation)
    *   [Loading Data](#loading-data)
    *   [Configuring Evaluation Rules](#configuring-evaluation-rules)
    *   [Running Evaluation](#running-evaluation)
    *   [Viewing Results](#viewing-results)
    *   [Inspecting Exemplars](#inspecting-exemplars)
    *   [Exporting Artifacts](#exporting-artifacts)
5.  [Project Structure](#5-project-structure)
6.  [Technology Stack](#6-technology-stack)
7.  [Contributing](#7-contributing)
8.  [License](#8-license)
9.  [Contact](#9-contact)

---

## 1. Project Title and Description

This project, **"QuLab: Lab 8: LLM Evaluation Harness (Inference & Fine-Tuning Risk)"**, is a Streamlit-based application designed to provide a comprehensive framework for evaluating Large Language Models (LLMs), particularly focusing on assessing risks introduced by inference and fine-tuning processes.

Developed for professionals like Alex (AI Auditor), Maria (Model Validator), and David (AI Risk Lead) at InnovateCorp, this tool addresses the critical need for robust validation of RAG (Retrieval-Augmented Generation) applications. It enables users to systematically audit LLM behavior by analyzing outputs from both baseline and fine-tuned models against a set of prompts. The harness identifies potential risks such as hallucinations, lack of faithfulness, inappropriate refusals, and over-compliance, offering a quantitative and qualitative view of model trustworthiness. Crucially, it provides a mechanism to detect regressions in model behavior post-fine-tuning, ensuring that model improvements in one area don't inadvertently degrade performance or introduce new risks in others.

The application culminates in the generation of auditable artifacts, complete with cryptographic hashes, to ensure data integrity and traceability for compliance and governance purposes.

## 2. Features

The LLM Evaluation Harness offers a suite of functionalities to facilitate thorough model assessment:

*   **Flexible Data Upload:**
    *   Upload custom CSV files for prompts, baseline model outputs, and optional fine-tuned model outputs.
    *   Quickly load pre-generated sample data for immediate exploration and testing.
*   **Configurable Evaluation Rules:**
    *   **Hallucination Proxy Settings:** Define thresholds for excessive verbosity, over-specificity keywords, and patterns indicating unverified numerical details.
    *   **Faithfulness Checks:** Configure regex patterns for citation detection to verify source attribution.
    *   **Refusal & Over-Compliance Settings:** Specify phrases for detecting model refusals and excessive safety disclaimers.
    *   **Regression Analysis Sensitivity:** Set a delta threshold for flagging performance regressions between baseline and fine-tuned models.
*   **Automated Evaluation Run:**
    *   Execute a comprehensive set of checks including hallucination detection, faithfulness verification against `allowed_sources`, and analysis of refusal/over-compliance behaviors.
*   **Interactive Scorecards:**
    *   View aggregate metrics for both baseline and fine-tuned models (if provided), including hallucination rate, faithfulness score, refusal rate, and inappropriate refusal rate.
    *   Visualize key metric comparisons using interactive bar charts.
*   **Fine-Tuning Regression Analysis:**
    *   Automatically compare performance metrics between baseline and fine-tuned models.
    *   Identify and flag specific metrics where fine-tuning has led to a degradation (regression) beyond a defined threshold.
*   **Failure Exemplar Inspection:**
    *   Drill down into individual high-risk prompts where models exhibited issues (e.g., hallucination, unfaithfulness, refusal).
    *   Compare baseline and fine-tuned outputs side-by-side, along with detected issues, prompt text, expected answers, and allowed sources.
    *   Identify specific instances of improvement or regression at a granular level.
*   **Auditable Artifact Generation:**
    *   Generate a ZIP archive containing all evaluation results, aggregate scorecards, regression analysis, detailed evaluation dataframes, and the evaluation configuration.
    *   Includes an `evidence_manifest.json` with SHA-256 hashes of all generated files, ensuring data integrity and traceability for audit and compliance.
*   **OpenAI API Key Integration (Optional):** Input an OpenAI API key in the sidebar for `source.py` functions that might leverage OpenAI models (e.g., for real-time inference or more sophisticated evaluation criteria if extended). Not strictly required if only evaluating pre-generated outputs.

## 3. Getting Started

Follow these steps to set up and run the Streamlit application on your local machine.

### Prerequisites

Before you begin, ensure you have met the following requirements:

*   **Python 3.8+**: Download and install Python from [python.org](https://www.python.org/downloads/).
*   **`pip`**: Python's package installer, usually comes with Python installation.
*   **Git**: For cloning the repository.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/quslab-llm-eval-harness.git
    cd quslab-llm-eval-harness
    ```
    *(Replace `https://github.com/your-username/quslab-llm-eval-harness.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Create a `requirements.txt` file in your project root with the following content)*:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    matplotlib>=3.0.0
    seaborn>=0.13.0
    openai # Potentially needed by source.py
    python-dotenv # If using .env for API key
    ```

### OpenAI API Key Configuration

While the application primarily focuses on evaluating *pre-generated* outputs, some internal functions in `source.py` or future extensions might require an OpenAI API key.

You can provide your OpenAI API key in two ways:

1.  **Environment Variable (Recommended):**
    Create a `.env` file in the project root directory and add your key:
    ```
    OPENAI_API_KEY="sk-your_openai_api_key_here"
    ```
    The application will automatically try to load this.

2.  **Direct Input in Streamlit Sidebar:**
    Upon running the application, you will find an "OpenAI API Key" input field in the sidebar. You can paste your key directly there. This will override any key loaded from environment variables for the current session.

    **Note:** If you are only evaluating pre-generated outputs and `source.py` does not perform live API calls, providing an API key is optional.

## 4. Usage

To run the application, ensure your virtual environment is active and then execute:

```bash
streamlit run app.py
```

This will open the application in your default web browser (usually at `http://localhost:8501`).

### Application Workflow

The application guides you through a logical workflow, accessible via the sidebar navigation:

1.  **Data Upload**: Load your prompt and model output datasets.
2.  **Configure Evaluation Rules**: Customize evaluation parameters.
3.  **Run Evaluation**: Execute the analysis based on loaded data and rules.
4.  **View Scorecards**: Review aggregate metrics and regression analysis.
5.  **Inspect Failure Exemplars**: Examine detailed cases of model failure.
6.  **Export Artifacts**: Download all generated reports and evidence.

### Navigation

Use the "Go to..." selectbox in the left sidebar to navigate between the different stages of the evaluation process.

### Loading Data (1. Data Upload)

*   **Load Sample Data**: Click the "Load Sample Data" button to quickly populate the application with pre-generated mock datasets for demonstration purposes. This is a great way to explore the features without needing your own data.
*   **Upload Your Own Data**: Use the file uploaders to provide your `Prompts CSV`, `Baseline Outputs CSV`, and optionally `Fine-Tuned Outputs CSV`.
    *   **Prompts CSV** must contain: `prompt_id`, `prompt_text`, `expected_answer` (optional), `allowed_sources` (optional, for RAG faithfulness checks).
    *   **Model Outputs CSV** (Baseline and Fine-Tuned) must contain: `prompt_id`, `llm_output`, `sources_cited` (optional).
    *   Click "Load Uploaded Data" once files are selected.

After successful loading, a sample of the combined evaluation DataFrame will be displayed.

### Configuring Evaluation Rules (2. Configure Evaluation Rules)

This section allows you to fine-tune the evaluation logic:

*   **Hallucination Proxy Settings**: Adjust the `Hallucination Threshold Ratio` and define `Over-Specificity Keywords`.
*   **Refusal and Over-Compliance Settings**: Input `Refusal Phrases` and `Excessive Safety Disclaimers` (one per line).
*   **Regression Analysis Settings**: Set the `Regression Threshold Delta` to determine the sensitivity for flagging performance regressions.
*   **Citation Pattern**: Provide a regular expression (e.g., `\[\d+\]`) to identify citations within LLM outputs for faithfulness checks.

Your current configuration is displayed in JSON format at the bottom of the page. These settings are stored in `st.session_state` and applied during the evaluation run.

### Running Evaluation (3. Run Evaluation)

Once data is loaded and rules are configured, click the "Run Evaluation" button. The application will:

1.  Apply hallucination, faithfulness, and refusal checks to both baseline and fine-tuned model outputs (if provided).
2.  Aggregate per-prompt flags into overall metrics.
3.  Perform regression analysis if fine-tuned outputs are present.

Progress will be shown via spinner messages. A success message will appear upon completion.

### Viewing Results (4. View Scorecards)

After running the evaluation, this page displays:

*   **Aggregate Scorecard**: A table showing key metrics (e.g., hallucination rate, faithfulness score, refusal rate) for the baseline and fine-tuned models.
*   **Comparison Plot**: A bar chart visualizing the comparison of key metrics between models.
*   **Fine-Tuning Regression Analysis**: If fine-tuned outputs were evaluated, this section will highlight delta changes in metrics and explicitly flag any regressions detected based on your configured threshold.

### Inspecting Exemplars (5. Inspect Failure Exemplars)

This section provides detailed views of prompts where models exhibited "high-risk" behaviors. It displays:

*   **Prompt ID and Text**: The original query.
*   **Allowed Sources & Expected Answer**: Contextual information for RAG and correctness checks.
*   **Baseline Model Output**: The response from the baseline model, its cited sources, and a list of detected issues.
*   **Fine-Tuned Model Output**: The response from the fine-tuned model, its cited sources, and a list of detected issues (if applicable).
*   **Comparison Notes**: Highlights if a prompt showed improvement or regression after fine-tuning.

This granular view is crucial for understanding the root causes of issues and informing model improvements.

### Exporting Artifacts (6. Export Artifacts)

This final stage allows you to generate and download a comprehensive audit package:

1.  Click **"Generate & Download Artifacts"**.
2.  The application will create a unique `run_id` (e.g., `20231027_153045`).
3.  It will then generate various reports, including:
    *   `executive_summary.md`: High-level overview of findings.
    *   `aggregate_metrics.json`: Detailed aggregate scores.
    *   `regression_analysis.json`: Regression findings.
    *   `evaluation_dataframe.csv`: The complete DataFrame with all flags.
    *   `evaluation_config.json`: The exact configuration used for the run.
    *   `prompt_{run_id}_evidence_manifest.json`: A manifest file listing all generated artifacts with their SHA-256 hashes for integrity verification.
4.  All these files will be bundled into a `.zip` archive named `Session_08_{run_id}.zip`.
5.  A **"Download All Artifacts"** button will appear, allowing you to save the ZIP file.
6.  The content of the `evidence_manifest.json` will also be displayed in the application for quick review.

This ensures a complete, traceable, and auditable record of each evaluation run.

## 5. Project Structure

```
quslab-llm-eval-harness/
├── app.py                      # Main Streamlit application
├── source.py                   # Contains all core business logic and evaluation functions
├── requirements.txt            # Python dependencies
├── .env                        # (Optional) Environment variables, e.g., OPENAI_API_KEY
├── data/                       # (Optional) Directory for sample data or user-uploaded data
│   ├── sample_prompts.csv
│   ├── sample_baseline_outputs.csv
│   └── sample_finetuned_outputs.csv
└── reports/                    # Directory for generated evaluation artifacts
    └── <run_id>/               # Subdirectory for each evaluation run
        ├── executive_summary.md
        ├── aggregate_metrics.json
        ├── regression_analysis.json
        ├── evaluation_dataframe.csv
        ├── evaluation_config.json
        └── prompt_<run_id>_evidence_manifest.json
```

## 6. Technology Stack

*   **Frontend & Application Framework**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Plotting & Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Core Language**: Python 3.8+
*   **LLM Interaction (via `source.py`)**: Potentially the `openai` Python library, configurable via API key.
*   **Utilities**: `os`, `json`, `zipfile`, `io`, `datetime`

## 7. Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate documentation and tests where applicable.

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Create a `LICENSE` file in your project root if you haven't already)*

## 9. Contact

For questions, feedback, or support, please reach out to:

*   **QuantUniversity:** [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Email:** info@quantuniversity.com
*   **GitHub Issues:** Feel free to open an issue in this repository.

---

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
