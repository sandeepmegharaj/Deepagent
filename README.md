# 🚀 Deep Agent

> **Live Demo:** [https://deepagent.streamlit.app/](https://deepagent.streamlit.app/) *(or deploy via Streamlit Cloud)*

A multi-agent workspace built on **LangChain**, **LangGraph**, and **DeepAgents**. It plans multi-step tasks, conducts live web research with citations, extracts and profiles data files (CSV, Excel, PDF), and renders interactive Plotly charts directly inside a ChatGPT-style conversational UI.

---

## ⚡ Key Features

- **🤖 3 Autonomous Specialist Agents:**
  - **Research Agent:** Live web search and citation synthesis via Tavily.
  - **Data Analyst:** Deterministic data quality checks, statistical profiling, and interactive chart generation.
  - **Structured Researcher:** Generates structured Pydantic reports with confidence scores and source URLs.
- **💬 ChatGPT-Grade Interface:** Clean dark UI, full-width composer with model selector (`ChatGPT-5`), file attachment chips, and collapsible step-by-step reasoning logs.
- **📊 Interactive Data Visualizations:** Automatically creates inline Plotly charts (bar, line, scatter, box, pie, heatmap) from uploaded CSV/Excel files.
- **📄 Document Extraction:** Bounded text and structural extraction from PDFs and TXT files without external OCR dependencies.
- **🧠 Flexible Memory & Backends:**
  - `StateBackend`: Isolated per-thread in-memory workspace.
  - `FilesystemBackend`: Persistent local workspace with disk access.
  - `StoreBackend`: Cross-thread persistent memory using LangGraph Store.

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend / UI** | Streamlit, Custom Vanilla CSS (Dark Minimalist Theme) |
| **Agent Framework** | LangChain, LangGraph, DeepAgents |
| **LLM Provider** | OpenAI / Azure OpenAI (`openai:gpt-5`), Groq (fallback) |
| **Search & Research** | Tavily Web Search API |
| **Data & Viz** | Pandas, Plotly Express, OpenPyXL, PyPDF |
| **State & Memory** | LangGraph `MemorySaver` (thread checkpointing), `InMemoryStore` |

---

## 🏁 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/sandeepmegharaj/Deepagent.git
cd Deepagent
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your keys:
```ini
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://your-endpoint.openai.azure.com/openai/v1  # optional
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key                                  # optional
```

### 3. Run Locally
```bash
streamlit run streamlit_app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New app**:
   - **Repository:** `sandeepmegharaj/Deepagent`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. In **Advanced settings → Secrets**, paste:
```toml
OPENAI_API_KEY = "your_key"
OPENAI_BASE_URL = "https://your-endpoint.openai.azure.com/openai/v1"
TAVILY_API_KEY = "your_key"
GROQ_API_KEY = "your_key"
```
4. Click **Deploy**.
