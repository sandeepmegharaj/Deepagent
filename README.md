# Deep Agents Streamlit App

This app reproduces the repository's Deep Agents demo as a provider-configurable
Streamlit application. The default model is OpenAI GPT-5. The agent graph
remains based on `deepagents` and LangGraph, including planning, virtual file
tools, context files, skills, subagents, structured research output,
checkpointers, and selectable storage backends.

Read [`DEEP_AGENT_GUIDE.md`](DEEP_AGENT_GUIDE.md) for the capability
classification, sub-agent roles, task expectations, API-key placement, and a
run checklist. Read [`ANTIGRAVITY_HANDOFF.md`](ANTIGRAVITY_HANDOFF.md) before
opening the project in Antigravity.

## Run locally

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` for the default GPT-5 model.
3. Set `TAVILY_API_KEY` if web research should be enabled.
4. Install dependencies and start Streamlit:

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The same command works with a normal Python virtual environment and
`pip install -r requirements.txt`.

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Streamlit Cloud

Add provider keys under the app's Secrets settings instead of committing a
`.env` file:

```toml
OPENAI_API_KEY = "..."
TAVILY_API_KEY = "..."
```

## Agent behavior

- The default model is `openai:gpt-5`; GPT-5.4 and GPT-5.5 are also selectable.
- The one-retry fallback logic is retained, but the current workspace UI
  intentionally selects `None` and does not expose fallback model choices.
- The main agent can plan with `write_todos`, use virtual file tools, load
  `AGENTS.md` and skills, search through Tavily, and delegate to
  `research-agent`, `structured-researcher`, and `data-analyst`.
- The data analyst can inspect CSV, JSON, and Excel worksheets for profiles,
  deterministic quality checks, numeric summaries, common values, and grouped
  aggregations. Analysis is capped at the first 100,000 rows.
- The agent can create interactive bar, line, area, scatter, histogram, box,
  violin, pie, and correlation-heatmap charts from uploaded data. Chart payloads
  are capped at 500 points and render directly in the conversation.
- PDF and TXT uploads support bounded text extraction. PDF extraction reads at
  most 20 pages and 20,000 characters per request; scanned PDFs require OCR.
- Data Lab shows quality findings, an interactive missingness chart, and
  previews, then downloads an HTML dashboard and a CSV of findings. The HTML
  chart loads Plotly from its pinned CDN and needs an internet connection.
- The interface uses a readable Inter/system sans stack inspired by Claude's
  conversation typography, aligns user bubbles right, and keeps work details
  and each created file collapsed until opened.
- The left sidebar shows the files available to the current backend and lets
  you preview their contents.
- `StateBackend` is the safe default for in-thread scratch files.
- `FilesystemBackend` maps virtual paths into `deepagentsdemo/` on disk.
- `StoreBackend` keeps files across threads for the lifetime of the Streamlit
  process through the in-memory LangGraph store.

Uploads live under the ignored `deepagentsdemo/uploads/` folder and are not
included in the Antigravity source ZIP. Do not place API keys in source files
or commit `.env`.
