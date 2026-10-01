"""
Deep Agents Chatbot - Streamlit app
===================================
A conversational chatbot built on the `deepagents` library that keeps the
agent behavior from the accompanying notebooks with a provider-configurable
dark workspace, a left-hand file explorer, and a built-in Data Lab.

1-basicsdeepagent.ipynb   -> create_deep_agent, custom model, custom system
                             prompt, custom tools (Tavily web search),
                             built-in planning (write_todos) + virtual files
2-contextengineering.ipynb -> AGENTS.md context file, memory=, checkpointer +
                             thread_id conversation memory, Skills (/skills/)
3-backends.ipynb          -> StateBackend / FilesystemBackend / StoreBackend
4-subagents.ipynb         -> custom subagents (research-agent) + structured
                             output subagent (Pydantic response_format)

Run with:  streamlit run streamlit_app.py
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from data_tools import (
    DOCUMENT_EXTENSIONS,
    MAX_ANALYSIS_ROWS,
    MAX_DOCUMENT_CHARS,
    SUPPORTED_UPLOAD_EXTENSIONS,
    TABULAR_EXTENSIONS,
    build_quality_dashboard_html,
    create_chart_spec,
    data_quality_report,
    extract_document_text as extract_uploaded_document_text,
    list_excel_sheets as get_excel_sheets,
    list_upload_files,
    load_tabular_file,
    safe_upload_path,
)

# ---------------------------------------------------------------------------
# Environment (notebook 1: load API keys from .env or Streamlit secrets)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DEMO_DIR = ROOT_DIR / "deepagentsdemo"
UPLOAD_DIR = DEMO_DIR / "uploads"
DATA_EXTENSIONS = TABULAR_EXTENSIONS
UPLOAD_EXTENSIONS = SUPPORTED_UPLOAD_EXTENSIONS

load_dotenv(ROOT_DIR / ".env")

# Propagate custom endpoint and normalize Azure AI / OpenAI base URL
for env_k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "GROQ_API_KEY", "TAVILY_API_KEY"):
    if os.getenv(env_k):
        os.environ[env_k] = os.getenv(env_k).strip()

openai_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
if openai_base:
    clean_base = openai_base.rstrip("/")
    if clean_base.endswith("/responses"):
        clean_base = clean_base[:-len("/responses")]
    os.environ["OPENAI_BASE_URL"] = clean_base

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from tavily import TavilyClient


def get_secret(name: str) -> str | None:
    """Read a setting from environment variables, then Streamlit secrets."""
    value = os.getenv(name)
    if not value:
        try:
            value = st.secrets.get(name)
        except (FileNotFoundError, KeyError, TypeError):
            value = None

    if not value:
        return None

    str_val = str(value).strip()
    if (str_val.startswith("your_") and str_val.endswith("_here")) or not str_val:
        return None

    return str_val


def load_styles() -> str:
    """Load the app's small, responsive dark-theme stylesheet."""
    return (ROOT_DIR / "styles.css").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Custom tool (notebook 1: Tavily internet search)
# ---------------------------------------------------------------------------
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search when TAVILY_API_KEY is configured."""
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": (
                "TAVILY_API_KEY is not configured. Add it to .env or Streamlit "
                "secrets to enable internet search."
            )
        }

    return TavilyClient(api_key=api_key).search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def _safe_data_path(file_name: str) -> Path:
    """Resolve an uploaded data file without allowing path traversal."""
    return safe_upload_path(UPLOAD_DIR, file_name, DATA_EXTENSIONS)


@st.cache_data(show_spinner=False)
def _cached_load_data_frame(
    file_name: str,
    sheet_name: str | None,
    modified_ns: int,
    file_size: int,
) -> tuple[pd.DataFrame, str]:
    return load_tabular_file(UPLOAD_DIR, file_name, sheet_name)


def _load_data_frame(
    file_name: str, sheet_name: str | None = None
) -> tuple[pd.DataFrame, str]:
    """Load a bounded tabular file and return its dataframe plus filename."""
    path = _safe_data_path(file_name)
    metadata = path.stat()
    return _cached_load_data_frame(
        path.name, sheet_name, metadata.st_mtime_ns, metadata.st_size
    )


@st.cache_data(show_spinner=False)
def _cached_extract_document(
    file_name: str,
    start_page: int,
    max_pages: int,
    max_chars: int,
    modified_ns: int,
    file_size: int,
) -> dict[str, Any]:
    return extract_uploaded_document_text(
        UPLOAD_DIR,
        file_name,
        start_page=start_page,
        max_pages=max_pages,
        max_chars=max_chars,
    )


def extract_document_text(
    file_name: str,
    start_page: int = 1,
    max_pages: int = 10,
    max_chars: int = MAX_DOCUMENT_CHARS,
) -> dict[str, Any]:
    """Extract bounded text from an uploaded PDF or TXT document."""
    try:
        path = safe_upload_path(UPLOAD_DIR, file_name, DOCUMENT_EXTENSIONS)
        metadata = path.stat()
    except (OSError, TypeError, ValueError) as exc:
        return {"error": str(exc)}
    return _cached_extract_document(
        path.name,
        start_page,
        max_pages,
        max_chars,
        metadata.st_mtime_ns,
        metadata.st_size,
    )


@st.cache_data(show_spinner=False)
def _cached_quality_report(
    file_name: str,
    sheet_name: str | None,
    modified_ns: int,
    file_size: int,
) -> dict[str, Any]:
    frame, _ = _cached_load_data_frame(
        file_name, sheet_name, modified_ns, file_size
    )
    return data_quality_report(frame)


@st.cache_data(show_spinner=False)
def _cached_dashboard_html(
    file_name: str,
    sheet_name: str | None,
    modified_ns: int,
    file_size: int,
) -> str:
    frame, _ = _cached_load_data_frame(
        file_name, sheet_name, modified_ns, file_size
    )
    return build_quality_dashboard_html(frame, file_name, sheet_name)


@st.cache_data(show_spinner=False)
def _cached_chart_spec(
    file_name: str,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    color_column: str | None,
    aggregation: str,
    top_n: int,
    sheet_name: str | None,
    title: str | None,
    modified_ns: int,
    file_size: int,
) -> dict[str, Any]:
    return create_chart_spec(
        UPLOAD_DIR,
        file_name,
        chart_type=chart_type,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
        aggregation=aggregation,
        top_n=top_n,
        sheet_name=sheet_name,
        title=title,
    )


def list_data_files() -> dict[str, Any]:
    """List uploaded tabular files and documents available for local analysis."""
    return list_upload_files(UPLOAD_DIR)


def analyze_tabular_file(
    file_name: str,
    operation: Literal[
        "profile", "summary", "missing", "quality", "top_values", "group_summary"
    ] = "profile",
    column: str | None = None,
    metric: str | None = None,
    aggregation: Literal["mean", "sum", "count", "min", "max"] = "mean",
    top_n: int = 10,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Analyze an uploaded CSV, JSON, or selected XLSX worksheet.

    Operations include profile, numeric summary, missing values, deterministic
    quality checks, common values, and grouped aggregation. Analysis is bounded
    to the first 100,000 rows.
    """
    try:
        frame, loaded_name = _load_data_frame(file_name, sheet_name)
    except (ImportError, OSError, ValueError, TypeError) as exc:
        return {"error": str(exc)}

    if frame.empty:
        return {"file": loaded_name, "error": "The data file is empty."}

    top_n = max(1, min(int(top_n), 50))
    columns = [str(value) for value in frame.columns]
    result: dict[str, Any] = {
        "file": loaded_name,
        "rows_analyzed": int(len(frame)),
        "columns": columns,
    }
    if sheet_name and loaded_name.lower().endswith(".xlsx"):
        result["worksheet"] = sheet_name

    if operation == "profile":
        sample = frame.head(5).astype(object).where(pd.notna(frame.head(5)), None)
        sample = sample.apply(
            lambda series: series.map(
                lambda value: value.isoformat() if hasattr(value, "isoformat") else value
            )
        )
        result.update(
            {
                "column_types": {
                    str(key): str(value) for key, value in frame.dtypes.items()
                },
                "missing_values": {
                    str(key): int(value) for key, value in frame.isna().sum().items()
                },
                "numeric_columns": [
                    str(value) for value in frame.select_dtypes("number").columns
                ],
                "sample_rows": sample.to_dict(orient="records"),
            }
        )
        return result

    if operation == "summary":
        numeric = frame.select_dtypes("number")
        if numeric.empty:
            return {**result, "message": "No numeric columns were found."}
        summary = numeric.describe().round(4).transpose().reset_index()
        summary = summary.rename(columns={"index": "column"})
        result["numeric_summary"] = summary.to_dict(orient="records")
        return result

    if operation == "missing":
        result["missing_values"] = [
            {
                "column": str(key),
                "missing": int(value),
                "missing_percent": round(float(value / len(frame) * 100), 2),
            }
            for key, value in frame.isna().sum().items()
        ]
        return result

    if operation == "quality":
        result["quality"] = data_quality_report(frame)
        return result

    if operation == "top_values":
        if not column or column not in frame.columns:
            return {**result, "error": f"Choose one column from: {columns}"}
        counts = frame[column].value_counts(dropna=False).head(top_n)
        result["top_values"] = [
            {"value": "<missing>" if pd.isna(key) else str(key), "count": int(value)}
            for key, value in counts.items()
        ]
        return result

    if operation == "group_summary":
        if not column or column not in frame.columns:
            return {**result, "error": f"Choose a grouping column from: {columns}"}
        if aggregation != "count" and (not metric or metric not in frame.columns):
            return {**result, "error": f"Choose a metric column from: {columns}"}
        groups = frame.groupby(column, dropna=False)
        values = groups.size() if aggregation == "count" else getattr(groups[metric], aggregation)()
        result["group_summary"] = [
            {
                "group": "<missing>" if pd.isna(key) else str(key),
                "value": float(value),
            }
            for key, value in values.head(top_n).items()
        ]
        return result

    return {**result, "error": f"Unsupported operation: {operation}"}


def create_interactive_chart(
    file_name: str,
    chart_type: Literal[
        "bar", "line", "area", "scatter", "histogram", "box", "violin", "pie", "heatmap"
    ] = "bar",
    x_column: str | None = None,
    y_column: str | None = None,
    color_column: str | None = None,
    aggregation: Literal["none", "count", "sum", "mean"] = "none",
    top_n: int = 40,
    sheet_name: str | None = None,
    title: str | None = None,
) -> str:
    """Create an interactive chart from an uploaded CSV, JSON, or Excel file.

    Supported charts are bar, line, area, scatter, histogram, box, violin,
    pie, and numeric correlation heatmap. Chart data is bounded to 500 points.
    """
    try:
        path = _safe_data_path(file_name)
        metadata = path.stat()
        spec = _cached_chart_spec(
            path.name,
            chart_type,
            x_column,
            y_column,
            color_column,
            aggregation,
            top_n,
            sheet_name,
            title,
            metadata.st_mtime_ns,
            metadata.st_size,
        )
    except (OSError, TypeError, ValueError) as exc:
        spec = {"kind": "interactive_chart", "error": str(exc)}
    return json.dumps(spec, ensure_ascii=True, allow_nan=False)


# ---------------------------------------------------------------------------
# Structured output schema (notebook 4: structured output with subagents)
# ---------------------------------------------------------------------------
class ResearchFindings(BaseModel):
    """Structured findings from a research task."""

    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description="Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")


# ---------------------------------------------------------------------------
# Context engineering helpers (notebook 2: AGENTS.md + skills seeding)
# ---------------------------------------------------------------------------
def load_agents_md() -> str:
    path = DEMO_DIR / "projects" / "AGENTS.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_skill_seed_files() -> dict:
    """Read every file under deepagentsdemo/skills/ and convert it to in-state
    file data so the StateBackend agent can discover and read skills."""
    files = {}
    skills_root = DEMO_DIR / "skills"
    if skills_root.exists():
        for f in skills_root.rglob("*.md"):
            virtual = "/skills/" + f.relative_to(skills_root).as_posix()
            files[virtual] = create_file_data(f.read_text(encoding="utf-8"))
    return files


# ---------------------------------------------------------------------------
# Agent factory — assembles ALL the features based on sidebar config
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI assistant and researcher. You conduct thorough "
    "research using your internet_search tool when needed, plan multi-step "
    "work with write_todos, offload bulky content to files, use your skills "
    "when a query matches one, and delegate deep-dive research to your "
    "subagents. For uploaded tabular data, use deterministic data-quality and "
    "analysis tools; for PDFs and text files, use document extraction. Do not "
    "invent data. For chart requests on uploaded data, call "
    "create_interactive_chart with an appropriate type and columns; the "
    "workspace renders it interactively. Summarize the takeaway instead of "
    "repeating raw chart data. Keep final responses concise; refer to created "
    "artifacts instead of pasting their contents unless asked. Always cite "
    "sources when research was involved."
)
MODEL_OPTIONS = ("openai:gpt-5",)
FALLBACK_OPTIONS = ("None",)


def model_label(model_name: str) -> str:
    """Show a short model name while retaining its provider-qualified value."""
    if model_name == "None":
        return "None"
    return "ChatGPT-5"


def sync_model_selection(source_key: str, target_key: str) -> None:
    """Keep the sidebar setting and composer model selector in sync."""
    st.session_state[target_key] = st.session_state[source_key]

SUBAGENT_DOC = """
- **research-agent** — in-depth research with web search (context quarantine)
- **structured-researcher** — returns `ResearchFindings` (summary, confidence, sources)
- **data-analyst** — CSV, JSON, Excel quality checks, interactive charts, and PDF/text extraction
"""


def model_key_name(model_name: str) -> str | None:
    """Return the environment key required by a provider:model name."""
    provider = model_name.split(":", 1)[0].lower()
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider)


def model_is_configured(model_name: str) -> bool:
    """Check whether a provider model has the secret it needs."""
    key_name = model_key_name(model_name)
    return bool(key_name and get_secret(key_name))


def build_agent(cfg: dict):
    """Create a deep agent wired up according to the sidebar configuration."""
    seed_files = {}

    # --- backend selection (notebook 3) -----------------------------------
    if cfg["backend"] == "StateBackend (in-state, per thread)":
        backend = StateBackend()
        # StateBackend has no disk access -> seed AGENTS.md + skills into state
        if cfg["use_agents_md"]:
            seed_files["/projects/AGENTS.md"] = create_file_data(load_agents_md())
        if cfg["use_skills"]:
            seed_files.update(load_skill_seed_files())
        memory_paths = ["/projects/AGENTS.md"] if cfg["use_agents_md"] else None

    elif cfg["backend"] == "FilesystemBackend (real disk)":
        # virtual_mode=True confines the agent inside deepagentsdemo/
        backend = FilesystemBackend(root_dir=str(DEMO_DIR), virtual_mode=True)
        # AGENTS.md and skills/ already exist on disk — nothing to seed
        memory_paths = ["/projects/AGENTS.md"] if cfg["use_agents_md"] else None

    else:  # StoreBackend (cross-thread memory)
        store = st.session_state.store
        backend = StoreBackend(store=store, namespace=lambda rt: ("memories",))
        # Seed durable memory into the store once per session
        if not st.session_state.get("store_seeded"):
            if cfg["use_agents_md"]:
                store.put(("memories",), "/projects/AGENTS.md",
                          create_file_data(load_agents_md()))
            if cfg["use_skills"]:
                for path, data in load_skill_seed_files().items():
                    store.put(("memories",), path, data)
            st.session_state.store_seeded = True
        memory_paths = ["/projects/AGENTS.md"] if cfg["use_agents_md"] else None

    # --- subagents (notebook 4) --------------------------------------------
    subagents = []
    if cfg["use_subagents"]:
        subagents.append({
            "name": "research-agent",
            "description": "Used to research more in depth questions",
            "system_prompt": "You are a great researcher. Research thoroughly "
                             "and cite your sources.",
            "tools": [internet_search],
        })
        subagents.append({
            "name": "structured-researcher",
            "description": "Researches topics and returns structured findings "
                           "(summary, confidence score, source URLs)",
            "system_prompt": "Research the given topic thoroughly. "
                             "Return your findings.",
            "tools": [internet_search],
            "response_format": ResearchFindings,
        })
        if cfg["use_data_analyst"]:
            data_analyst = {
                "name": "data-analyst",
                "description": (
                    "Inspects uploaded CSV, JSON, and Excel worksheets for "
                    "profiles, quality issues, summaries, grouped results, "
                    "and interactive charts; extracts bounded text from PDF "
                    "or TXT documents."
                ),
                "system_prompt": (
                    "You are a careful data analyst. First list the available "
                    "files. Use the quality operation for data-quality requests, "
                    "specify the worksheet for Excel workbooks, and use document "
                    "extraction for PDFs. Use create_interactive_chart for "
                    "visualization requests. Explain row/page limits, distinguish "
                    "possible outliers from errors, and never invent values."
                ),
                "tools": [
                    list_data_files,
                    analyze_tabular_file,
                    extract_document_text,
                    create_interactive_chart,
                ],
            }
            # Prefer a configured fallback for data work; otherwise inherit
            # the selected primary model.
            if (
                cfg["analyst_model"] == "Fallback model (recommended)"
                and cfg["fallback_model"] != "None"
                and model_is_configured(cfg["fallback_model"])
            ):
                data_analyst["model"] = cfg["fallback_model"]
            subagents.append(data_analyst)

    # --- assemble the deep agent --------------------------------------------
    agent_tools = [internet_search]
    if cfg["use_data_analyst"]:
        agent_tools.extend(
            [
                list_data_files,
                analyze_tabular_file,
                extract_document_text,
                create_interactive_chart,
            ]
        )

    kwargs = dict(
        model=cfg["model"],
        tools=agent_tools,
        system_prompt=cfg["system_prompt"],
        backend=backend,
        checkpointer=st.session_state.checkpointer,  # notebook 2: thread memory
    )
    if subagents:
        kwargs["subagents"] = subagents
    if cfg["use_skills"]:
        kwargs["skills"] = ["/skills/"]  # notebook 2: Agent Skills
    if memory_paths:
        kwargs["memory"] = memory_paths  # notebook 2: memory= context loading
    if cfg["backend"].startswith("StoreBackend"):
        kwargs["store"] = st.session_state.store

    return create_deep_agent(**kwargs), seed_files


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    """AIMessage.content may be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def file_content(data: Any) -> str:
    """Extract displayable text from a deepagents file-data value."""
    if isinstance(data, dict):
        content = data.get("content", "")
    else:
        content = data

    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content or "")


def normalize_files(files: dict[str, Any] | None) -> dict[str, str]:
    """Convert backend file data into path-to-text values for the UI."""
    if not files:
        return {}
    return {path: file_content(data) for path, data in files.items()}


TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sql",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
CHART_COLORS = ["#54d2a0", "#7da9ff", "#e8b86d", "#b78bf2", "#ef8181", "#55c4cf"]


def compact_text(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def save_uploaded_files(uploaded_files: list[Any]) -> list[str]:
    """Save supported tabular and document uploads into the bounded data folder."""
    if not uploaded_files:
        return []

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    saved_uploads = st.session_state.setdefault("saved_upload_signatures", {})
    for uploaded_file in uploaded_files:
        name = Path(uploaded_file.name).name
        if Path(name).suffix.lower() not in UPLOAD_EXTENSIONS:
            continue
        target = UPLOAD_DIR / name
        file_id = getattr(uploaded_file, "file_id", None)
        signature = (file_id, getattr(uploaded_file, "size", None))
        if file_id and saved_uploads.get(name) == signature and target.is_file():
            saved.append(name)
            continue
        target.write_bytes(uploaded_file.getvalue())
        if file_id:
            saved_uploads[name] = signature
        saved.append(name)
    return saved


def load_uploaded_data_files() -> dict[str, str]:
    """Expose uploaded data files as explorer entries without reading binaries."""
    files: dict[str, str] = {}
    if not UPLOAD_DIR.exists():
        return files

    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in UPLOAD_EXTENSIONS:
            continue
        virtual_path = "/uploads/" + path.name
        if path.suffix.lower() in {".csv", ".json", ".txt"} and path.stat().st_size <= 1_000_000:
            try:
                files[virtual_path] = path.read_text(encoding="utf-8")
                continue
            except (OSError, UnicodeDecodeError):
                pass
        files[virtual_path] = (
            f"Binary data file: {path.name}\n"
            "Use the data-analyst sub-agent to inspect this file."
        )
    return files


def load_disk_files() -> dict[str, str]:
    """Read text files available to the FilesystemBackend for the explorer."""
    files: dict[str, str] = {}
    if not DEMO_DIR.exists():
        return files

    for path in DEMO_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            virtual_path = "/" + path.relative_to(DEMO_DIR).as_posix()
            files[virtual_path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return files


def visible_files(cfg: dict[str, Any], seed_files: dict[str, Any]) -> dict[str, str]:
    """Combine backend files and the latest agent result for the explorer."""
    files: dict[str, str] = {}
    if cfg["backend"].startswith("FilesystemBackend"):
        files.update(load_disk_files())
    else:
        files.update(normalize_files(seed_files))
    files.update(load_uploaded_data_files())
    files.update(st.session_state.get("virtual_files", {}))
    return dict(sorted(files.items()))


def file_language(path: str) -> str:
    """Return a small syntax-highlighting hint for a file preview."""
    return {
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".sql": "sql",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(Path(path).suffix.lower(), "text")


def render_file_explorer(files: dict[str, str]) -> None:
    """Render a selectable, left-sidebar view of the agent's virtual files."""
    st.sidebar.divider()
    st.sidebar.markdown(
        '<div class="sidebar-section-label">FILE EXPLORER</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"{len(files)} file(s) available to the agent")
    query = st.sidebar.text_input(
        "Filter files",
        placeholder="Search paths",
        key="file_filter",
    ).strip().lower()

    filtered = {
        path: content
        for path, content in files.items()
        if not query or query in path.lower()
    }
    if not filtered:
        st.sidebar.caption("No files match the current filter.")
        return

    groups: dict[str, list[str]] = {}
    for path in filtered:
        parts = path.strip("/").split("/")
        group = parts[0] if len(parts) > 1 else "."
        groups.setdefault(group, []).append(path)

    for group, paths in groups.items():
        if group == ".":
            group_paths = paths
        else:
            with st.sidebar.expander(f"{group}/", expanded=False):
                group_paths = paths
                for path in group_paths:
                    label = "/".join(path.strip("/").split("/")[1:])
                    if st.button(
                        label,
                        key=f"open-file-{path}",
                        width="stretch",
                    ):
                        st.session_state.selected_file = path
                        st.rerun()
            continue

        for path in group_paths:
            if st.sidebar.button(
                path,
                key=f"open-file-{path}",
                width="stretch",
            ):
                st.session_state.selected_file = path
                st.rerun()

    selected = st.session_state.get("selected_file")
    if selected not in filtered:
        selected = None
        st.session_state.selected_file = None
    if selected:
        with st.sidebar.expander(Path(selected).name, expanded=True):
            st.caption(selected)
            st.code(filtered[selected], language=file_language(selected))


def _tool_activity_label(name: str, args: dict[str, Any]) -> str:
    """Summarize a tool call without exposing its full arguments or output."""
    if name == "internet_search":
        return "Searching the web"
    if name == "task":
        specialist = args.get("subagent_type")
        return f"Delegating to {specialist}" if specialist else "Delegating specialist work"
    if name in {"write_file", "edit_file"}:
        path = args.get("file_path") or args.get("path")
        return f"Updating {Path(str(path)).name}" if path else "Updating a file"
    if name in {"read_file", "ls", "glob", "grep"}:
        path = args.get("file_path") or args.get("path")
        return f"Reviewing {Path(str(path)).name}" if path else "Reviewing project files"
    if name == "analyze_tabular_file":
        file_name = args.get("file_name", "uploaded data")
        operation = args.get("operation", "profile")
        return f"Analyzing {file_name} ({operation.replace('_', ' ')})"
    if name == "extract_document_text":
        return f"Reading {args.get('file_name', 'the document')}"
    if name == "create_interactive_chart":
        return f"Building an interactive {args.get('chart_type', 'data')} chart"
    return f"Using {name.replace('_', ' ')}"


def _chart_payload(message: Any) -> dict[str, Any] | None:
    content = getattr(message, "content", "")
    if isinstance(content, dict):
        return content
    try:
        return json.loads(extract_text(content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _style_chart(
    figure: go.Figure,
    title: str,
    x_title: str | None,
    y_title: str | None,
    height: int | None = None,
) -> go.Figure:
    font = 'Inter, "Segoe UI", system-ui, sans-serif'
    figure.update_layout(
        template="plotly_dark",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"family": font, "size": 17, "color": "#f2f4f5"},
        },
        font={"family": font, "size": 12, "color": "#c8d0d4"},
        colorway=CHART_COLORS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 12, "r": 18, "t": 62, "b": 22},
        hoverlabel={
            "bgcolor": "#171b1d",
            "bordercolor": "#41494d",
            "font": {"family": font, "size": 12, "color": "#f2f4f5"},
        },
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        height=height,
    )
    figure.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor="#3a4246",
        tickfont={"color": "#9da8ae"},
        automargin=True,
    )
    figure.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor="rgba(156, 170, 176, 0.14)",
        zeroline=False,
        linecolor="#3a4246",
        tickfont={"color": "#9da8ae"},
        automargin=True,
    )
    figure.update_traces(hoverlabel_namelength=-1)
    return figure


def render_interactive_chart(spec: dict[str, Any], key: str) -> None:
    """Render a validated chart payload returned by the agent tool."""
    if spec.get("error"):
        st.info(spec["error"])
        return

    chart_type = spec.get("chart_type")
    title = str(spec.get("title") or "Data chart")
    if chart_type == "heatmap":
        labels = spec.get("x_labels", [])
        figure = go.Figure(
            go.Heatmap(
                z=spec.get("matrix", []),
                x=labels,
                y=spec.get("y_labels", labels),
                zmin=-1,
                zmax=1,
                colorscale=[
                    [0, "#dc7772"],
                    [0.5, "#222a2d"],
                    [1, "#54d2a0"],
                ],
                colorbar={"title": "Correlation"},
                texttemplate="%{z:.2f}",
                hovertemplate="%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            )
        )
        figure.update_yaxes(autorange="reversed")
        figure = _style_chart(
            figure,
            title,
            None,
            None,
            height=max(360, min(720, 70 * len(labels) + 150)),
        )
    else:
        data = pd.DataFrame(spec.get("data", []))
        if data.empty:
            st.info("There is not enough data to render this chart.")
            return

        x_column = spec.get("x")
        y_column = spec.get("y")
        color_column = spec.get("color")
        labels = {"__value__": spec.get("y_label", "Value")}
        if x_column:
            labels[x_column] = spec.get("x_label") or x_column
        if y_column and y_column != "__value__":
            labels[y_column] = spec.get("y_label", y_column)
        common = {
            "data_frame": data,
            "color_discrete_sequence": CHART_COLORS,
            "labels": labels,
        }
        if color_column:
            common["color"] = color_column

        if chart_type == "bar":
            if not pd.api.types.is_numeric_dtype(data[x_column]):
                figure = px.bar(
                    **common,
                    x=y_column,
                    y=x_column,
                    orientation="h",
                    barmode="group",
                )
                figure.update_yaxes(categoryorder="total ascending")
                x_title, y_title = spec.get("y_label"), None
                chart_height = max(360, min(900, 34 * data[x_column].nunique() + 130))
            else:
                figure = px.bar(**common, x=x_column, y=y_column, barmode="group")
                x_title, y_title = x_column, spec.get("y_label")
                chart_height = None
        elif chart_type == "line":
            figure = px.line(**common, x=x_column, y=y_column, markers=True)
            figure.update_traces(line={"width": 2.5})
        elif chart_type == "area":
            figure = px.area(**common, x=x_column, y=y_column, line_group=color_column)
        elif chart_type == "scatter":
            figure = px.scatter(**common, x=x_column, y=y_column, opacity=0.82)
            figure.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "#121618"}})
        elif chart_type == "histogram":
            figure = px.bar(**common, x=x_column, y=y_column, barmode="overlay")
            figure.update_traces(opacity=0.76, width=spec.get("bin_width"))
        elif chart_type == "box":
            figure = px.box(**common, x=x_column, y=y_column, points="outliers")
        elif chart_type == "violin":
            figure = px.violin(**common, x=x_column, y=y_column, box=True, points="outliers")
        elif chart_type == "pie":
            figure = px.pie(
                **common,
                names=x_column,
                values=y_column,
                hole=0.48,
            )
            figure.update_traces(
                textinfo="label+percent",
                textposition="inside",
                hovertemplate="%{label}<br>%{value:,.3g} (%{percent})<extra></extra>",
            )
        else:
            st.info("This chart type is not supported by the current renderer.")
            return

        if chart_type != "bar":
            x_title = (
                spec.get("x_label") or x_column
                if chart_type == "histogram"
                else x_column
            )
            y_title = None if chart_type == "pie" else spec.get("y_label")
            if chart_type == "histogram":
                y_title = spec.get("y_label", "Records")
            chart_height = None
        figure = _style_chart(figure, title, x_title, y_title, height=chart_height)

    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    source = spec.get("file") or "Uploaded data"
    rows = spec.get("rows_analyzed")
    plotted = spec.get("rows_plotted")
    note = f"{source} · {rows:,} rows analyzed" if isinstance(rows, int) else source
    if isinstance(plotted, int) and spec.get("sampled"):
        available = spec.get("rows_available")
        if chart_type in {"bar", "pie"} and isinstance(available, int):
            note += f" · top {plotted:,} of {available:,} categories"
        else:
            note += f" · {plotted:,} points shown"
    if spec.get("columns_limited"):
        note += " · first 32 numeric columns"
    st.caption(note)


def render_steps(messages, key_prefix: str = "activity"):
    """Show concise, collapsed activity plus any charts created for the turn."""
    actions = []
    todos = None
    charts = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name, args = tc["name"], tc.get("args", {})
                if name == "write_todos":
                    todos = args.get("todos", [])
                else:
                    label = _tool_activity_label(name, args)
                    if label not in actions:
                        actions.append(label)
        elif msg_type == "tool" and getattr(msg, "name", "") == "create_interactive_chart":
            spec = _chart_payload(msg)
            if spec:
                charts.append(spec)

    items = todos or actions
    if items:
        with st.status(f"Work details · {len(items)} steps", state="complete", expanded=False):
            for index, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    label = compact_text(item.get("content") or "Task")
                    status = str(item.get("status") or "pending").replace("_", " ")
                    st.markdown(f"**{index}.** {label} · {status.title()}")
                else:
                    st.markdown(f"**{index}.** {item}")

    for index, spec in enumerate(charts):
        render_interactive_chart(spec, key=f"{key_prefix}-chart-{index}")


def render_files(files: dict):
    if not files:
        return
    st.markdown(f'<div class="artifact-heading">Files · {len(files)}</div>', unsafe_allow_html=True)
    for path, data in files.items():
        content = file_content(data)
        with st.expander(Path(path).name, expanded=False):
            st.caption(path)
            st.code(content, language=file_language(path))


def invoke_with_fallback(
    cfg: dict[str, Any],
    payload: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Invoke the primary model once, then use a configured fallback later."""
    fallback_model = cfg["fallback_model"]

    def invoke_fallback() -> dict[str, Any]:
        fallback_cfg = {
            **cfg,
            "model": fallback_model,
            "fallback_model": "None",
            "analyst_model": "Primary model",
        }
        fallback_agent, _ = build_agent(fallback_cfg)
        return fallback_agent.invoke(payload, config=config)

    if st.session_state.get("fallback_active"):
        return invoke_fallback(), True

    try:
        return st.session_state.agent.invoke(payload, config=config), False
    except Exception as primary_error:
        if (
            fallback_model == "None"
            or fallback_model == cfg["model"]
            or not model_is_configured(fallback_model)
        ):
            raise primary_error

        st.session_state.fallback_active = True
        try:
            return invoke_fallback(), True
        except Exception as fallback_error:
            raise RuntimeError(
                "The primary model failed and the fallback model also failed. "
                f"Primary error: {primary_error}. Fallback error: {fallback_error}"
            ) from fallback_error


def start_new_chat() -> None:
    """Start a fresh thread without clearing the shared in-memory store."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.virtual_files = {}
    st.session_state.selected_file = None
    st.session_state.active_page = "chat"
    st.session_state.pop("attached_file", None)
    st.session_state.pop("queued_prompt", None)


def render_blog_page() -> None:
    """Render the current workspace guide, capabilities, and runnable prompts."""
    if st.button("← Back to Chat", key="blog_back_top"):
        st.session_state.active_page = "chat"
        st.rerun()

    st.markdown(
        """
        <div class="blog-hero">
          <div class="eyebrow">A practical multi-agent workspace</div>
          <h1>Meet Your Deep Agent</h1>
          <p>Plan multi-step work, research with sources, manage project files, and analyze local data with transparent checks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Specialist agents")
    columns = st.columns(3)
    cards = [
        ("Research Agent", "Investigates a topic with web search and returns cited findings."),
        ("Structured Researcher", "Returns a consistent summary, confidence level, and source URLs."),
        ("Data Analyst", "Profiles CSV, JSON, and Excel data; runs quality checks; extracts PDF and text."),
    ]
    for column, (title, description) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="blog-card"><div class="blog-card-title">{title}</div>'
                f'<div class="blog-card-desc">{description}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### What you can do")
    capabilities = [
        "Multi-step planning", "Web research", "File creation and editing",
        "Conversation memory", "Data quality checks", "Excel worksheets",
        "PDF text extraction", "Downloadable HTML dashboards",
    ]
    cap_cols = st.columns(4)
    for index, label in enumerate(capabilities):
        with cap_cols[index % 4]:
            st.markdown(f'<div class="capability-badge">{label}</div>', unsafe_allow_html=True)

    sample_path = UPLOAD_DIR / "covid_data.csv"
    st.markdown("### Demo dataset")
    if sample_path.is_file():
        try:
            sample = pd.read_csv(sample_path, nrows=MAX_ANALYSIS_ROWS)
            metrics = st.columns(3)
            metrics[0].metric("Rows available", f"{len(sample):,}")
            metrics[1].metric("Columns", f"{len(sample.columns):,}")
            metrics[2].metric("File", "covid_data.csv")
            with st.expander("Preview the first 10 rows"):
                st.dataframe(sample.head(10), width="stretch")
            if st.button("Profile COVID sample", key="blog_covid_profile"):
                st.session_state.queued_prompt = (
                    "Profile covid_data.csv and run the data-quality checks. "
                    "Summarize missing values and numeric columns."
                )
                st.session_state.active_page = "chat"
                st.rerun()
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"The optional sample file could not be read: {exc}")
    else:
        st.info(
            "No COVID demo CSV is bundled with this project. Upload a CSV or "
            "Excel file in Data files to use the Data Lab."
        )

    # --- Example Report Section ---
    st.markdown("---")
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.85rem;">
            <div style="background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(59,130,246,0.1));
                        border: 1px solid rgba(16,185,129,0.3); border-radius:9999px;
                        padding:0.25rem 0.85rem; font-size:0.72rem; font-weight:700;
                        letter-spacing:0.1em; color:#6ee7b7; text-transform:uppercase;">
                Example Output
            </div>
            <span style="color:#a3a3a3; font-size:0.85rem;">What Deep Agent generated from covid_data.csv</span>
        </div>
        <h3 style="margin:0 0 0.5rem; font-size:1.25rem; color:#f0f0f0;">
            📊 COVID-19 Data Report — Generated by Deep Agent
        </h3>
        <p style="color:#9ca3af; font-size:0.9rem; margin-bottom:1rem;">
            Uploaded <code>covid_data.csv</code> with the prompt <em>"make a report on this"</em> — the agent
            automatically profiled 1,682 rows × 21 columns, ran quality checks, produced an interactive
            time-series chart of US cumulative confirmed cases, and ranked the top 15 states by confirmed count.
        </p>
        """,
        unsafe_allow_html=True,
    )

    import os as _os
    _report_img = _os.path.join(_os.path.dirname(__file__), "assets", "covid_report_example.png")
    if _os.path.isfile(_report_img):
        st.image(
            _report_img,
            caption="Deep Agent report: US COVID-19 trends, state rankings, and data quality summary",
            use_container_width=True,
        )
        st.markdown(
            """
            <div style="background:#1a1a1a; border:1px solid #2d2d2d; border-radius:0.85rem;
                        padding:0.9rem 1.1rem; margin-top:0.85rem;">
                <div style="font-size:0.82rem; font-weight:700; color:#9de7c2; margin-bottom:0.5rem;">
                    📋 What the agent produced:
                </div>
                <ul style="margin:0; padding-left:1.25rem; color:#c9d1d9; font-size:0.85rem; line-height:1.75;">
                    <li>Interactive time-series: <strong>US cumulative confirmed cases over time</strong></li>
                    <li>Bar chart: <strong>Top 15 states by max cumulative confirmed</strong></li>
                    <li>Data profile: 1,682 rows · 21 columns · 84 states/territories</li>
                    <li>Quality flags: 23.1% cells missing · identified empty columns</li>
                    <li>12 reasoning steps completed end-to-end</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Try a prompt")
    prompts = [
        "Run data-quality checks on my uploaded file and explain each finding.",
        "Profile my uploaded Excel workbook and compare its worksheets.",
        "Extract the main points from the PDF I uploaded.",
        "Create a project plan and save it as /notes/project-plan.md.",
        "Research a current topic and include source links.",
    ]
    for index, prompt in enumerate(prompts):
        left, right = st.columns([5, 1.3])
        with left:
            st.markdown(prompt)
        with right:
            if st.button("Run in Agent", key=f"blog_prompt_{index}", width="stretch"):
                st.session_state.queued_prompt = prompt
                st.session_state.active_page = "chat"
                st.rerun()


def render_data_lab() -> None:
    """Render deterministic data checks and bounded document extraction."""
    st.markdown("### Data Lab")
    st.caption(
        f"Local inspection, capped at {MAX_ANALYSIS_ROWS:,} rows. "
        "No opaque quality score is used."
    )
    mode = st.radio(
        "Choose a workspace",
        ["Tabular data", "PDF and text"],
        horizontal=True,
        label_visibility="collapsed",
    )
    uploaded = list_data_files().get("files", [])

    if mode == "Tabular data":
        files = [item["name"] for item in uploaded if f'.{item["type"]}' in DATA_EXTENSIONS]
        if not files:
            st.info("Upload a CSV, JSON, or XLSX file from the Data files panel to begin.")
            return

        file_name = st.selectbox("Data file", files, key="data_lab_file")
        sheet_name = None
        if Path(file_name).suffix.lower() == ".xlsx":
            try:
                sheet_names = get_excel_sheets(UPLOAD_DIR, file_name)
            except (ImportError, OSError, ValueError) as exc:
                st.error(f"Could not read workbook sheets: {exc}")
                return
            if not sheet_names:
                st.warning("This workbook has no worksheets.")
                return
            sheet_name = st.selectbox("Worksheet", sheet_names, key="data_lab_sheet")

        try:
            frame, _ = _load_data_frame(file_name, sheet_name)
            path = _safe_data_path(file_name)
            metadata = path.stat()
            quality = _cached_quality_report(
                file_name, sheet_name, metadata.st_mtime_ns, metadata.st_size
            )
        except (ImportError, OSError, ValueError, TypeError) as exc:
            st.error(f"Could not analyze this file: {exc}")
            return
        if frame.empty:
            st.warning("This file or worksheet is empty.")
            return

        metrics = st.columns(4)
        metrics[0].metric("Rows analyzed", f"{quality['rows_analyzed']:,}")
        metrics[1].metric("Columns", f"{quality['columns_analyzed']:,}")
        metrics[2].metric("Missing cells", f"{quality['missing_cells']:,}")
        metrics[3].metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
        st.caption(f"Source: {file_name}" + (f" / {sheet_name}" if sheet_name else ""))

        if quality["findings"]:
            st.markdown("#### Findings")
            st.dataframe(
                pd.DataFrame(quality["findings"]), width="stretch", hide_index=True
            )
        else:
            st.success("No common data-quality issues were detected in the analyzed rows.")

        missing_rows = [item for item in quality["missing_by_column"] if item["missing"]]
        if missing_rows:
            st.markdown("#### Missingness profile")
            missing_frame = (
                pd.DataFrame(missing_rows)
                .sort_values("missing_percent", ascending=True)
                .tail(20)
            )
            chart_height = max(300, min(680, 52 * len(missing_frame) + 110))
            figure = px.bar(
                missing_frame,
                x="missing_percent",
                y="column",
                orientation="h",
                color="missing_percent",
                color_continuous_scale=["#28584c", "#54d2a0"],
                custom_data=["missing"],
            )
            figure.update_traces(
                marker_line_width=0,
                texttemplate="%{x:.1f}%",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>%{x:.2f}% missing"
                    "<br>%{customdata[0]:,} cells<extra></extra>"
                ),
            )
            maximum = float(missing_frame["missing_percent"].max())
            figure.update_xaxes(range=[0, min(100, max(1, maximum * 1.25))])
            figure.update_layout(coloraxis_showscale=False)
            figure = _style_chart(
                figure,
                "Missing values by column",
                "Missing cells (%)",
                None,
                height=chart_height,
            )
            st.plotly_chart(
                figure,
                width="stretch",
                key=f"quality-missing-{file_name}-{sheet_name or 'default'}",
                config={"responsive": True, "displaylogo": False},
            )

        with st.expander("Preview the first 100 rows", expanded=False):
            st.dataframe(frame.head(100), width="stretch", hide_index=True)

        dashboard = _cached_dashboard_html(
            file_name, sheet_name, metadata.st_mtime_ns, metadata.st_size
        )
        safe_stem = Path(file_name).stem.replace(" ", "-")
        st.download_button(
            "Download HTML dashboard",
            dashboard,
            file_name=f"{safe_stem}-quality-dashboard.html",
            mime="text/html",
            width="stretch",
        )
        if quality["findings"]:
            findings_csv = pd.DataFrame(quality["findings"]).to_csv(index=False)
            st.download_button(
                "Download findings CSV",
                findings_csv,
                file_name=f"{safe_stem}-quality-findings.csv",
                mime="text/csv",
                width="stretch",
            )
        return

    documents = [item["name"] for item in uploaded if f'.{item["type"]}' in DOCUMENT_EXTENSIONS]
    if not documents:
        st.info("Upload a PDF or TXT file from the Data files panel to extract its text.")
        return

    file_name = st.selectbox("Document", documents, key="document_lab_file")
    is_pdf = Path(file_name).suffix.lower() == ".pdf"
    page_cols = st.columns(2) if is_pdf else []
    start_page = page_cols[0].number_input("Start page", min_value=1, value=1, step=1) if is_pdf else 1
    max_pages = page_cols[1].number_input(
        "Pages to read (max 20)", min_value=1, max_value=20, value=10, step=1
    ) if is_pdf else 10
    if st.button("Extract text", type="primary", key="extract_document_button"):
        try:
            st.session_state.document_extract = extract_document_text(
                file_name, start_page=int(start_page), max_pages=int(max_pages)
            )
            st.session_state.document_extract_file = file_name
        except (ImportError, OSError, ValueError, TypeError) as exc:
            st.session_state.document_extract = {"error": str(exc)}
            st.session_state.document_extract_file = file_name

    result = st.session_state.get("document_extract")
    if st.session_state.get("document_extract_file") == file_name and result:
        if result.get("error"):
            st.error(result["error"])
        else:
            if result.get("message"):
                st.caption(result["message"])
            text = result.get("text", "")
            if text:
                st.text_area("Extracted text preview", text, height=320, disabled=True)
                st.download_button(
                    "Download extracted text",
                    text,
                    file_name=f"{Path(file_name).stem}-extracted.txt",
                    mime="text/plain",
                    width="stretch",
                )
            elif result.get("page_count", 0):
                st.info("This PDF contains no selectable text. Scanned pages need OCR.")


def render_chat_workspace(cfg: dict[str, Any]) -> None:
    """Render the chat view and invoke the configured deep agent on submit."""
    top_c1, top_c2, top_c3, top_c4 = st.columns([3.4, 1.2, 1.2, 1.2])
    with top_c1:
        st.markdown("### Agent Workspace")
        st.caption("Plan, research, analyze data, inspect PDFs, and create files")
    with top_c2:
        if st.button("New chat", key="top_new_chat_btn", width="stretch"):
            start_new_chat()
            st.rerun()
    with top_c3:
        if st.button("Data Lab", key="top_data_lab_btn", width="stretch"):
            st.session_state.active_page = "data"
            st.rerun()
    with top_c4:
        if st.button("Read Blog", key="top_blog_btn", width="stretch"):
            st.session_state.active_page = "blog"
            st.rerun()
    st.divider()

    for history_index, (role, text, steps, files) in enumerate(st.session_state.history):
        if role == "user":
            st.markdown(
                f'<div class="chat-row user-row"><div class="user-bubble">{text}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant", avatar=None):
                if steps:
                    render_steps(steps, key_prefix=f"history-{history_index}")
                st.markdown(text)
                if files:
                    render_files(files)



    with st.container(key="composer_model_selector"):
        st.selectbox(
            "Model",
            MODEL_OPTIONS,
            key="composer_model",
            format_func=model_label,
            on_change=sync_model_selection,
            args=("composer_model", "sidebar_model"),
            label_visibility="collapsed",
        )
    chat_submission = st.chat_input(
        "Message DeepAgent",
        accept_file="multiple",
        file_type=sorted(extension.lstrip(".") for extension in UPLOAD_EXTENSIONS),
    )
    uploaded_files_from_input = []
    text_prompt = ""
    if chat_submission is not None:
        if getattr(chat_submission, "files", None):
            uploaded_files_from_input = chat_submission.files
        if hasattr(chat_submission, "text"):
            text_prompt = chat_submission.text
        elif isinstance(chat_submission, str):
            text_prompt = chat_submission

    saved_names = save_uploaded_files(uploaded_files_from_input)
    queued_prompt = st.session_state.pop("queued_prompt", None)
    base_prompt = text_prompt or queued_prompt
    if not base_prompt and not saved_names:
        return

    # Seed files into session state and StateBackend
    if saved_names:
        for name in saved_names:
            target_path = UPLOAD_DIR / name
            if target_path.is_file():
                try:
                    content = target_path.read_text(encoding="utf-8")
                    st.session_state.seed_files[f"/uploads/{name}"] = create_file_data(content)
                    st.session_state.virtual_files[f"/uploads/{name}"] = content
                except Exception:
                    pass

    # Build the agent prompt with explicit file context
    if saved_names:
        files_summary = ", ".join(saved_names)
        if base_prompt:
            agent_prompt = (
                f"{base_prompt}\n\n"
                f"[Uploaded file(s): {files_summary}. Analyze these datasets using "
                f"the data analysis tools and provide a clear, comprehensive report with metrics and key findings.]"
            )
        else:
            agent_prompt = (
                f"Profile, inspect, and summarize the uploaded file(s): {files_summary}. "
                f"Provide a comprehensive data quality and summary report."
            )
    else:
        agent_prompt = base_prompt

    if not agent_prompt:
        return

    if st.session_state.get("agent") is None:
        key_name = model_key_name(cfg["model"]) or "the selected provider's API key"
        st.error(f"Add {key_name} in .env or Streamlit secrets, then try again.")
        st.session_state.queued_prompt = base_prompt
        return

    # Build visual chat bubble for the user message showing uploaded file badges
    if saved_names:
        file_tags = "".join(
            f'<div class="chat-file-attachment">'
            f'<span class="chat-file-icon">📄</span>'
            f'<span class="chat-file-name">{name}</span>'
            f'</div>'
            for name in saved_names
        )
        user_display = (
            f'<div class="chat-file-group">{file_tags}</div>'
            f'<div class="chat-user-text">{base_prompt or "Analyze uploaded file(s)"}</div>'
        )
    else:
        user_display = base_prompt

    st.markdown(
        f'<div class="chat-row user-row"><div class="user-bubble">{user_display}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state.history.append(("user", user_display, None, saved_names or None))

    payload = {"messages": [{"role": "user", "content": agent_prompt}]}
    if st.session_state.seed_files and cfg["backend"].startswith("StateBackend"):
        payload["files"] = st.session_state.seed_files
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 100,
    }
    previous_files = dict(st.session_state.virtual_files)
    if cfg["backend"].startswith("FilesystemBackend"):
        previous_files = load_disk_files()

    with st.chat_message("assistant", avatar=None):
        with st.spinner("Analyzing and preparing report..."):
            try:
                result, fallback_used = invoke_with_fallback(cfg, payload, config)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()
        if fallback_used:
            st.info("The primary model encountered an issue, so the fallback model handled it.")

        all_messages = result["messages"]
        turn_start = max(
            (index for index, message in enumerate(all_messages)
             if getattr(message, "type", "") == "human"),
            default=0,
        )
        new_messages = all_messages[turn_start + 1:]
        render_steps(new_messages, key_prefix=f"turn-{len(st.session_state.history)}")
        answer = extract_text(all_messages[-1].content) or "*(no text response)*"
        st.markdown(answer)

        result_files = result.get("files", {}) or {}
        st.session_state.virtual_files.update(normalize_files(result_files))
        if cfg["backend"].startswith("FilesystemBackend"):
            st.session_state.virtual_files = load_disk_files()
        files = {
            path: content
            for path, content in st.session_state.virtual_files.items()
            if previous_files.get(path) != content
            and path not in st.session_state.seed_files
        }
        render_files(files)
    st.session_state.history.append(("assistant", answer, new_messages, files))


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Agent Using LangChain",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{load_styles()}</style>", unsafe_allow_html=True)

# --- session state init ------------------------------------------------------
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, text, steps_messages, files)]
if "virtual_files" not in st.session_state:
    st.session_state.virtual_files = {}
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "active_page" not in st.session_state:
    st.session_state.active_page = "chat"
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None
if "sidebar_model" not in st.session_state:
    st.session_state.sidebar_model = MODEL_OPTIONS[0]
if "composer_model" not in st.session_state:
    st.session_state.composer_model = st.session_state.sidebar_model

# --- sidebar configuration ---------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand-box">'
        '<div class="sidebar-pill">MULTI-AGENT PLATFORM</div>'
        '<div class="sidebar-kicker">DEEP AGENT</div>'
        '<div class="sidebar-subtitle">Research Agent • Data Analyst • Structured Researcher</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("➕ New chat", key="sidebar_new_chat", width="stretch", type="primary"):
        start_new_chat()
        st.rerun()

    nav_cols = st.columns(3)
    current_page = st.session_state.active_page
    for column, page, label in zip(
        nav_cols, ("chat", "data", "blog"), ("Chat", "Data Lab", "Blog")
    ):
        with column:
            if st.button(
                label,
                key=f"sidebar_nav_{page}",
                width="stretch",
                type="primary" if current_page == page else "secondary",
            ):
                st.session_state.active_page = page
                st.rerun()

    with st.expander("Agent settings", expanded=False):
        model = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            key="sidebar_model",
            format_func=model_label,
            on_change=sync_model_selection,
            args=("sidebar_model", "composer_model"),
            help="Provider:model identifier used by create_deep_agent.",
        )

        fallback_model = st.selectbox(
            "Fallback model",
            FALLBACK_OPTIONS,
            index=0,
            format_func=model_label,
            help="Configured fallback model used if primary model fails or for data analyst.",
        )

        backend = st.radio(
            "Backend",
            [
                "StateBackend (in-state, per thread)",
                "FilesystemBackend (real disk)",
                "StoreBackend (cross-thread store)",
            ],
            help="Choose where the agent's virtual files and memory live.",
        )

        st.markdown('<div class="settings-label">FEATURES</div>', unsafe_allow_html=True)
        use_agents_md = st.checkbox(
            "Load AGENTS.md context (memory=)", value=True,
            help="Load the durable project operating guide.",
        )
        use_skills = st.checkbox(
            "Skills (/skills/)", value=True,
            help="Enable the bundled LangGraph, Python, AWS, and report skills.",
        )
        use_subagents = st.checkbox(
            "Subagents", value=True,
            help="Enable research-agent, structured researcher, and data analyst.",
        )
        use_data_analyst = st.checkbox(
            "Data analyst", value=True,
            help="Enable CSV/JSON/Excel checks and PDF/text extraction.",
        )
        analyst_model = st.radio(
            "Data analyst model",
            ["Fallback model (recommended)", "Primary model"],
            index=0,
            help="Uses the fallback when configured; otherwise uses the selected primary model.",
        )
        if use_subagents:
            st.markdown(SUBAGENT_DOC)

        system_prompt = st.text_area(
            "System prompt", DEFAULT_SYSTEM_PROMPT, height=160,
            help="Project-specific instructions layered on the deep-agent prompt.",
        )

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1rem;">ACTIVE SPECIALISTS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #222222; border: 1px solid #333333; border-radius: 0.75rem; padding: 0.75rem 0.85rem; margin-bottom: 0.85rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1rem;">🔍</span>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #f0f0f0;">Research Agent</div>
                    <div style="font-size: 0.72rem; color: #888888;">Live Web Search & Synthesis</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1rem;">📊</span>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #f0f0f0;">Data Analyst</div>
                    <div style="font-size: 0.72rem; color: #888888;">Data Profiling, Quality & Charts</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1rem;">📑</span>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #f0f0f0;">Structured Researcher</div>
                    <div style="font-size: 0.72rem; color: #888888;">Structured Findings & Schema</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(f"Session Thread: `{st.session_state.thread_id[:8]}...`")
    col1, col2 = st.columns(2)
    if col1.button("New thread", width="stretch"):
        start_new_chat()
        st.rerun()
    if col2.button("Reset all", width="stretch", help="Wipe session memory and chat history."):
        for k in (
            "checkpointer",
            "store",
            "store_seeded",
            "thread_id",
            "history",
            "virtual_files",
            "selected_file",
            "fallback_active",
            "cfg_key",
            "agent",
            "seed_files",
            "active_page",
            "queued_prompt",
            "document_extract",
            "document_extract_file",
            "attached_file",
        ):
            st.session_state.pop(k, None)
        st.rerun()

cfg = {
    "model": model,
    "backend": backend,
    "use_agents_md": use_agents_md,
    "use_skills": use_skills,
    "use_subagents": use_subagents,
    "use_data_analyst": use_data_analyst,
    "analyst_model": analyst_model,
    "fallback_model": fallback_model,
    "system_prompt": system_prompt,
}

# Rebuild the agent only when the configuration changes
cfg_key = f"{sorted(cfg.items())}|configured={model_is_configured(cfg['model'])}"
if st.session_state.get("cfg_key") != cfg_key:
    st.session_state.fallback_active = False
    if model_is_configured(cfg["model"]):
        st.session_state.agent, st.session_state.seed_files = build_agent(cfg)
    else:
        st.session_state.agent = None
        st.session_state.seed_files = {}
    st.session_state.cfg_key = cfg_key

st.sidebar.markdown(
    """
    <div class="brand-footer" aria-label="Application identity and status">
        <div class="brand-title">Deep Agent Using LangChain</div>
        <div class="brand-meta">
            <span>Created by Sandeep Megharaj</span>
            <span class="status-pill">
                <span class="status-dot" aria-hidden="true"></span>
                Status: active
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.active_page == "blog":
    render_blog_page()
elif st.session_state.active_page == "data":
    render_data_lab()
else:
    render_chat_workspace(cfg)
