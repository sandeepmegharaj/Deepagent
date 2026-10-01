# Deep Agent Guide

This note explains what the application is, which skills and sub-agents it
contains, what tasks it is designed for, and how to configure its model
provider before running it.

## 1. What "deep agent" means

A deep agent is a stateful, long-running agent for work that takes more than
one simple question-and-answer step. This project uses `deepagents` on top of
LangGraph. The agent can plan work, store intermediate context in files,
delegate focused work to child agents, and remember a conversation through a
thread checkpointer.

The important difference from a basic chatbot is that the agent has an
operating loop:

1. Understand the request and decide whether it needs a plan.
2. Break a multi-step request into explicit todo items.
3. Use tools and skills to complete the work.
4. Offload large notes, drafts, and intermediate results to its virtual file
   system.
5. Delegate specialized research when isolation or a separate context is
   useful.
6. Synthesize a final answer, citing sources when web research was used.

## 2. Skills included

### LangGraph

Use this skill for LangGraph and agent-workflow questions. It covers state
schemas, nodes, edges, conditional routing, checkpointers, thread memory,
stores, streaming, interrupts, and multi-agent graph patterns.

### Python

Use this skill for Python coding, debugging, refactoring, typing, packaging,
testing, data structures, and practical implementation questions.

### AWS

Use this skill for AWS architecture, service selection, boto3, AWS CLI,
security, IAM, deployment, and cost considerations. It is instructed to avoid
hard-coded credentials and to prefer least privilege.

### Report writer

Use this skill after substantive work to create a structured Markdown report
with the question, approach, findings, answer, sources, caveats, and next
steps. The report is written through the agent's file tools.

### Project context

`deepagentsdemo/projects/AGENTS.md` is the standing operating guide. It tells
the agent what a deep agent is, when to plan, when to offload context, how to
delegate, and how to present research answers.

## 3. Sub-agents included

### research-agent

The research agent handles in-depth research tasks in an isolated context. It
can use the `internet_search` tool and is instructed to research thoroughly
and cite its sources. Its final findings are returned to the main agent.

### structured-researcher

The structured researcher handles research that should be returned in a
predictable shape. Its result contains:

- `summary`: the main findings
- `confidence`: a confidence score from 0 to 1
- `sources`: source URLs

The main agent can use that structured result while preparing the final answer.

### data-analyst

The data analyst handles uploaded CSV, JSON, and XLSX work. It can list files,
profile columns and data types, report missing values, identify duplicate rows,
empty or constant columns and whitespace-padded text, flag possible numeric
outliers using the 1.5 x IQR rule, calculate numeric summaries, find common
values, and calculate grouped count, mean, sum, minimum, or maximum results.
Analysis is bounded to the first 100,000 rows. Excel work can select a
worksheet.

For PDF and TXT uploads, it can extract bounded text. PDF extraction is capped
at 20 pages and 20,000 characters per request. Scanned PDFs need OCR and may
return no selectable text. Outliers are review signals, not assumed errors.

Sub-agents are context-isolated workers. They are not separate users, separate
API accounts, or independent long-running services.

## 4. What the agent can do

- Answer normal questions using the selected model and project context.
- Plan and execute multi-step work with `write_todos`.
- Read, write, edit, list, search, and organize virtual files.
- Create research notes, drafts, reports, code snippets, and project documents.
- Research current information through Tavily when `TAVILY_API_KEY` is set.
- Ask the research sub-agent for a deep-dive.
- Ask the structured sub-agent for findings with confidence and sources.
- Upload CSV, JSON, or XLSX files and ask for data-quality checks or analysis.
- Ask for interactive bar, line, area, scatter, histogram, box, violin, pie,
  or correlation-heatmap charts in the conversation. Chart data is capped at
  500 points; analysis remains capped at 100,000 rows.
- Upload PDFs or text files for bounded document-text extraction.
- Open Data Lab to preview quality findings, inspect an interactive missingness
  chart, and download an HTML dashboard or findings CSV. The HTML chart uses a
  pinned Plotly CDN and needs an internet connection.
- Use the LangGraph, Python, AWS, and report-writer skills when relevant.
- Continue a conversation using the same `thread_id` and checkpointer.
- Show intermediate planning, tool calls, sub-agent work, and virtual files in
  the chat interface.

## 5. Good task examples

- "Research the current LangGraph memory patterns and write a short report."
- "Create a Python implementation of binary search, add tests, and explain it."
- "Compare AWS Lambda and ECS for this workload and include cost and security
  tradeoffs."
- "Read the project guide, create `/notes/architecture.md`, and summarize it."
- "Research three options, compare their sources, and return a recommendation."
- "Draft a technical design document and save the intermediate notes in files."
- "Explain a LangGraph conditional-routing design with a minimal example."

## 6. What to expect

- The agent is strongest at research, planning, code, documentation, and
  technical analysis.
- A simple question may receive a direct answer without visible delegation.
- A complex request may take longer because the agent plans, searches, writes
  files, or calls a sub-agent before answering.
- Web citations require a working Tavily key. Without it, the agent can still
  chat, plan, use skills, and work with virtual files, but web search returns a
  setup message.
- The default `StateBackend` keeps files in the current conversation thread.
- `FilesystemBackend` writes inside `deepagentsdemo/` on the local machine.
- `StoreBackend` shares files across threads while the Streamlit process is
  running; the demo store is in memory and is not a production database.
- The model can make mistakes. Review generated code, research conclusions,
  citations, and any file changes before using them in production.
- The selected primary provider handles the supervisor. If a fallback is
  configured and the primary fails, the app retries once and keeps using the
  fallback for the current session.
- The agent does not automatically deploy production systems, spend money,
  access private accounts, or know secrets that were not provided to it.

## 7. Interface

The Streamlit frontend preserves the dark Chat/Blog workspace, adds Data Lab,
and uses a ChatGPT-like layout: user messages sit in a right-aligned gray
bubble, while assistant responses remain left-aligned on the canvas. The
stylesheet uses an Inter/system sans stack for readable Claude-like conversation
typography without bundling Anthropic's own font file. Work details and each
created file stay collapsed until opened. The left file explorer and
application status footer remain available across pages.

## 8. Where to configure provider keys

### Local run

Place provider keys in a file named `.env` at the project root, next to
`streamlit_app.py` and `pyproject.toml`. The current default is `openai:gpt-5`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Start from the supplied template:

```bash
cp .env.example .env
```

Then replace the placeholder. Never commit `.env`; it is ignored by the
project's `.gitignore`.

### Streamlit Cloud

Open the deployed app's Settings, then Secrets, and add:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
```

Tavily is optional for basic chat, but required for live web research. An
OpenAI key is needed for the default model; configure the selected provider's
key if you change models.

Do not put a provider key in `streamlit_app.py`, `README.md`, notebooks,
GitHub issues, screenshots, or chat messages that will be committed.

## 9. Run checklist

1. Use Python 3.11 or newer.
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` for the default model.
3. Set `TAVILY_API_KEY` if web search is required.
4. Install dependencies with `uv sync`, or use a virtual environment and
   `pip install -r requirements.txt`.
5. Start the app with `uv run streamlit run streamlit_app.py`, or run
   `streamlit run streamlit_app.py` inside the activated environment.
6. Keep the default GPT-5 model, `StateBackend`, AGENTS context, skills, and
   sub-agents enabled for the first test.
7. Configure another provider key if you change the model selection.
8. Ask: `Create /notes/hello.txt with the text Hello from the agent, then read it back.`
9. Confirm the file appears in the left explorer and the assistant explains
   what it did.
10. Upload CSV, JSON, XLSX, PDF, or TXT files in Data files. Check an Excel
    worksheet, run quality checks in Data Lab, or extract PDF text.
11. Download and open the HTML dashboard to verify it works without the app.
12. If testing web research, ask a current-events question and confirm sources
    appear after configuring Tavily.
13. Before pushing to GitHub, confirm `.env` is not listed by `git status` and
    no key appears in the diff.

## 10. GitHub and deployment checklist

- Push the source code, skills, notebooks, `.env.example`, and lock file.
- Do not push `.env` or any real secret.
- In Streamlit Cloud, select `streamlit_app.py` as the main file.
- Add the secrets in the Streamlit Secrets panel.
- Use the default GPT-5 model for the first deployment.
- Test a normal question, a file-writing task, and a research task after
  deployment.
