# 🚀 Deep Agent

> **Live Demo:** [https://deepagent.streamlit.app/](https://deepagent.streamlit.app/)

An enterprise-ready, multi-agent AI system powered by **Azure AI Services** and **Azure OpenAI Service**. It orchestrates specialized autonomous agents powered by **GPT-5** on Azure AI to execute complex reasoning, live web research with citations, automated data profiling on uploaded datasets (CSV, Excel, PDF), and inline interactive visual analytics within a ChatGPT-grade workspace.

---

## ⚡ Key Features

- **☁️ Azure AI-Powered Intelligence:**
  - Deployed on **Azure AI Foundry / Azure OpenAI Services** using enterprise-grade models (`ChatGPT-5 / GPT-5`).
  - Secure API gateway integration supporting custom endpoints, high throughput, and seamless fallback routing.
- **🤖 3 Autonomous Specialist Agents:**
  - **Research Agent:** Performs multi-step web investigation, source validation, and citation synthesis.
  - **Data Analyst:** Deterministic data-quality auditing, statistical profiling, missing value diagnosis, and automated interactive chart generation.
  - **Structured Researcher:** Schema-enforced research extraction outputting confidence metrics and validated reference sources.
- **💬 Modern Conversational UI:** Clean, dark ChatGPT-style interface with full-width composer, model selector, file attachment badges, and collapsible execution step logs.
- **📊 Interactive Data Visualizations:** Generates dynamic, responsive Plotly charts (bar, line, scatter, box, pie, heatmap) directly within the conversation flow.
- **📄 Document & Tabular Processing:** High-speed text extraction from PDFs and multi-sheet Excel workbooks with bounded resource consumption.
- **🧠 Scalable Memory & Context Management:**
  - Stateful session tracking and thread checkpointing.
  - In-memory workspace and persistent backend storage options for long-horizon agent tasks.

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **AI & Model Layer** | **Azure AI Services**, **Azure OpenAI Service** (`GPT-5`), Groq |
| **Agent Architecture** | DeepAgents Multi-Agent Engine, Stateful Orchestration |
| **Frontend / UI** | Streamlit, Custom Vanilla CSS (Dark Minimalist Architecture) |
| **Data & Visual Analytics** | Pandas, Plotly Express, OpenPyXL, PyPDF |
| **Search & Discovery** | Tavily Web Intelligence API |
| **State & Checkpointing** | Thread Memory Saver, Persistent Key-Value Store |
