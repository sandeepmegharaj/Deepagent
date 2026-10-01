# Antigravity Handoff

Use this checklist when importing the project into Antigravity for further
changes.

## 1. Import the complete project

Use `deep-agents-with-langchain-streamlit.zip`, extract it, and open the
extracted `deep-agents-with-langchain` folder as the project. Do not import
only `streamlit_app.py` because the agent needs the skills, `AGENTS.md`,
notebooks, `styles.css`, `.streamlit/config.toml`, and `uv.lock`.

The main application file is `streamlit_app.py`.

## 2. Keep these files together

- `streamlit_app.py`: Streamlit frontend and deep-agent wiring
- `styles.css`: dark theme, responsive layout, and readable sans typography
- `deepagentsdemo/`: project context, skills, notebooks, and demo files
- `deepagentsdemo/uploads/`: created on first upload; ignored local data and documents
- `DEEP_AGENT_GUIDE.md`: capabilities and expected behavior
- `README.md`: setup overview
- `pyproject.toml` and `uv.lock`: dependency definition and lock file
- `.env.example`: safe placeholder configuration only

## 3. Configure secrets safely

Create a local `.env` beside `streamlit_app.py` and add:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Use the Antigravity environment/secrets settings instead if Antigravity
provides them. Never paste a real key into source code, a prompt, a commit,
or a public project file. The repository ignores `.env` and
`.streamlit/secrets.toml`.

`OPENAI_API_KEY` is required for the default `openai:gpt-5` model.
`TAVILY_API_KEY` is optional for normal chat but required for live web search.
Set the key for whichever provider you select in Agent settings.

## 4. Install and run

From the project root, use one of these paths:

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Or with a normal Python environment:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Use Python 3.11 or newer. The default model is `openai:gpt-5`.

## 5. First smoke test

1. Start the app.
2. Leave `StateBackend`, AGENTS context, skills, and sub-agents enabled.
3. Ask: `Create /notes/hello.txt with the text Hello from the agent, then read it back.`
4. Confirm the answer appears and the file is visible in the left explorer.
5. Open Data files and upload a small CSV, JSON, XLSX, PDF, or TXT file.
6. Ask for a chart in normal chat and confirm it renders interactively.
7. In Data Lab, check tabular quality findings and download the HTML dashboard.
8. Extract text from a selectable-text PDF; scanned PDFs require OCR.
9. Ask a current-information question only after adding the Tavily key.

## 6. Rules for further changes

- Preserve `create_deep_agent`, the checkpointer, the backend options, skills,
  and sub-agent definitions unless the change explicitly targets agent
  behavior.
- Put visual changes in `styles.css` and keep the agent logic independent from
  the styling layer.
- Keep the data analyst bounded to uploaded-file tools. Tabular analysis reads
  at most 100,000 rows; document extraction is limited to 20 pages and 20,000
  characters per request.
- Keep chart output interactive and bounded to 500 points. The downloaded
  dashboard loads Plotly from its pinned CDN and requires internet access.
- Preserve the one-retry fallback behavior. Once fallback is activated, later
  turns in the session stay on the fallback model.
- Keep secrets in environment settings, never in Python or Markdown files.
- When changing prompts, skills, or sub-agents, update `DEEP_AGENT_GUIDE.md`.
- Run `uv run python -m unittest discover -s tests -v`, then smoke-test a
  normal chat, a file-writing task, and a research task after meaningful changes.
- Before pushing, run `git status` and confirm that `.env`, API keys, cache
  files, and generated private data are not staged.

## 7. Suggested Antigravity instruction

Use this as the first instruction after opening the project:

```text
Read README.md, DEEP_AGENT_GUIDE.md, and ANTIGRAVITY_HANDOFF.md first.
Preserve the deepagents/LangGraph behavior, skills, virtual file explorer,
sub-agents, checkpointer, and safe secret handling. Make the requested change
with the smallest correct edit, then run the Streamlit smoke tests before
reporting completion. Never expose or commit environment secrets.
```
