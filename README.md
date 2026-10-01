# 🚀 Deep Agent

> **Live Demo:** [https://deepagent.streamlit.app/](https://deepagent.streamlit.app/)

A multi-agent autonomous platform built on **LangChain**, **LangGraph**, and **DeepAgents**. It orchestrates multiple specialized sub-agents to plan multi-step workflows, conduct live web research with citations, inspect and profile data files (CSV, Excel, PDF), and generate interactive Plotly visualizations directly inside a ChatGPT-style conversational workspace.

---

## ⚡ Key Features

- **🤖 3 Autonomous Specialist Agents:**
  - **Research Agent:** Conducts live web research and source citation synthesis via Tavily.
  - **Data Analyst:** Executes deterministic data-quality checks, statistical profiling, and interactive chart generation.
  - **Structured Researcher:** Generates structured Pydantic reports with confidence scoring and validated source URLs.
- **💬 ChatGPT-Grade Interface:** Clean dark UI, full-width message composer with model selector (`ChatGPT-5`), file attachment chips, and collapsible step-by-step reasoning logs.
- **📊 Interactive Data Visualizations:** Automatically renders inline Plotly charts (bar, line, scatter, box, pie, heatmap) from uploaded CSV/Excel files.
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
