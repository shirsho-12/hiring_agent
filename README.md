# Agentic Hiring Pipeline

This project implements an advanced agentic pipeline for resume screening and evaluation, featuring a multi-agent system, a PDF-based RAG for contextual analysis, and experiment tracking with Weights & Biases.

## Features

- **Multi-Agent System**: A collaborative team of agents for extraction, evaluation, and summarization.
- **RAG-Powered Evaluation**: The evaluator uses a knowledge base of past successful candidates (in PDF format) to provide more accurate, context-aware scores.
- **Resume Anonymization**: Automatically removes PII, hyperlinks, and standardizes location/company information.
- **Resume Reformatting**: Ensures consistent and professional formatting of resume content.
- **Configurable Models & Prompts**: Easily swap out LLMs and prompts through a centralized configuration.
- **Experiment Tracking**: Integrated with Weights & Biases to log and visualize pipeline runs, making it ideal for research and iteration.
- **Robust Logging**: Detailed logging for clear visibility and easier debugging.

## Pipeline Stages

1.  **Resume Extractor**: Extracts key details from a resume.
2.  **Resume Evaluator**: Evaluates the extracted information using a RAG system built from historical PDF resumes.
3.  **Resume Summarizer**: A panel of sub-agents (CEO, CTO, HR) collaborates to generate personalized feedback.
4.  **Score Formatter**: Standardizes the final scores into a consistent format.

## Setup

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up Environment Variables**:

    - Rename `.env.example` to `.env`.
    - Add your API keys for OpenAI and Weights & Biases:
      ```
      OPENAI_API_KEY="your_openai_api_key_here"
      WANDB_API_KEY="your_wandb_api_key_here"
      ```

3.  **Populate the RAG Knowledge Base**:
    - The RAG system now supports multiple data sources to provide a comprehensive context for evaluation. Add your documents to the appropriate subdirectories within `data/rag_sources/`:
      - `past_resumes/`: Add PDF resumes of successful past candidates.
      - `company_criteria/`: Add Markdown files detailing company-wide hiring criteria, values, and policies.
      - `job_descriptions/`: Add Markdown files with detailed job descriptions.
    - The pipeline will automatically scan these directories, process all supported files (PDF and Markdown), and build a unified vector store for the evaluator agent.

## Resume Processing Features

The pipeline now includes advanced resume processing capabilities:

### Resume Anonymization

- Removes all personal identifiable information (PII)
- Strips out hyperlinks and social media handles
- Standardizes location information to a specified country
- Replaces company names with generic placeholders

### Resume Reformatting

- Ensures consistent spacing and layout
- Standardizes section headers
- Maintains a clean, professional appearance
- Preserves a placeholder for candidate names

<!-- ### Example Usage -->

## How to Run

1.  **Configure the Pipeline** (Optional):

    - To change the models used by the agents, modify the settings in `src/config/config.py`.
    - To update the prompts, edit the files in the `src/prompts/` directory.
    - For resume processing, ensure you provide an LLM instance when initializing the pipeline

2.  **Run the Pipeline**:

    For the full pipeline:

    ```bash
    python main.py
    ```

3.  **View Results in Weights & Biases**:
    - After running the pipeline, a link to your W&B dashboard will be printed in the console.
    - Open the link to view the detailed logs, including extracted details, evaluation scores, and the final summary for each run.
