# Smart CPG Decision Support Agent

This repository contains a full-stack Decision Support Agent designed for Consumer Packaged Goods (CPG) companies. It integrates **Databricks/PySpark** for data ingestion, **Python/Pandas** for complex metric extraction and simulation, **LangChain** for GenAI orchestration, and **Streamlit** for the UI.

## Architecture
- **Data Layer**: Ingests Parquet/CSV sales data. Supports both PySpark (for Databricks environments) and Pandas (for local UI).
- **Tool Layer**: Python modules to analyze trends, flag anomalies, and simulate "what-if" business scenarios.
- **GenAI Layer**: Integrates multiple LLM providers (Google Gemini, HuggingFace, OpenAI).
- **Agent Layer**: A LangChain ReAct agent orchestrates tool selection, parses outputs, and maintains conversational memory.
- **UI Layer**: Includes both a Streamlit dashboard and a robust Command Line Interface (CLI).

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- (Optional but recommended) A virtual environment

### 2. Installation
1. Clone the repository.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### 3. Environment Variables
To run the GenAI Agent, you need an API key from a supported provider. Set **ONE** of the following in your terminal:
```bash
# For Google Gemini (Free tier available via Google AI Studio)
export GOOGLE_API_KEY="your_api_key"

# For HuggingFace Models
export HUGGINGFACEHUB_API_TOKEN="your_hf_token"

# For OpenAI
export OPENAI_API_KEY="your_api_key"
```

## Running the Application

### Option A: Local Streamlit Web UI
To run the interactive web interface on your local machine:
```bash
streamlit run src/ui/streamlit_app.py
```

### Option B: Local Command Line Interface (CLI)
To run a fast terminal-based chat interface:
```bash
python src/ui/cli.py
```

### Option C: Databricks Free Edition
Since you already have your data loaded in Databricks Community Edition:
1. Upload the `smart-cpg-decision-agent/` folder to your Databricks Workspace (or clone it via Repos if supported).
2. Open the notebooks in `notebooks/`.
3. In your first cell of any notebook, install the local package: `%pip install -e .`
4. Set your API keys in the notebook using `os.environ`.
5. Run the cells to interact with the agent directly inside Databricks!

## Project Structure
```text
smart-cpg-decision-agent/
├── data/                  # Place your cpg_sales_data.parquet here
├── notebooks/             # Jupyter notebooks for Databricks usage
├── src/
│   ├── data_loader.py     # Unified spark/pandas data loader
│   ├── tools/             # Analytics and Simulation functions
│   ├── genai/             # LLM setup and routing
│   ├── agent/             # LangChain core and memory
│   └── ui/                # Streamlit and CLI entrypoints
├── tests/                 # Pytest test cases
├── requirements.txt       # Dependencies
└── setup.py               # Package installer
```
