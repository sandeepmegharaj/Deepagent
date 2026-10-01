"""Bounded file inspection helpers for the Streamlit data workspace."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

TABULAR_EXTENSIONS = frozenset({".csv", ".json", ".xlsx"})
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".txt"})
SUPPORTED_UPLOAD_EXTENSIONS = TABULAR_EXTENSIONS | DOCUMENT_EXTENSIONS
MAX_ANALYSIS_ROWS = 100_000
MAX_DOCUMENT_PAGES = 20
MAX_DOCUMENT_CHARS = 20_000
MAX_CHART_POINTS = 500
CHART_TYPES = frozenset(
    {"bar", "line", "area", "scatter", "histogram", "box", "violin", "pie", "heatmap"}
)


def safe_upload_path(
    upload_dir: Path | str,
    file_name: str,
    allowed_extensions: frozenset[str] = SUPPORTED_UPLOAD_EXTENSIONS,
) -> Path:
    """Resolve an uploaded filename inside its directory and validate type."""
    root = Path(upload_dir).resolve()
    candidate = (root / Path(str(file_name)).name).resolve()
    if candidate.parent != root or candidate.suffix.lower() not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Only files with these extensions are supported: {allowed}.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {Path(file_name).name}")
    return candidate


def list_upload_files(upload_dir: Path | str) -> dict[str, Any]:
    """List supported upload metadata without reading file contents."""
    root = Path(upload_dir)
    if not root.exists():
        return {"files": [], "message": "No files have been uploaded."}

    files = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS:
            files.append(
                {
                    "name": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"files": files, "message": f"Found {len(files)} file(s)."}


def load_tabular_file(
    upload_dir: Path | str,
    file_name: str,
    sheet_name: str | None = None,
    max_rows: int = MAX_ANALYSIS_ROWS,
) -> tuple[pd.DataFrame, str]:
    """Read at most ``max_rows`` from a supported tabular upload."""
    path = safe_upload_path(upload_dir, file_name, TABULAR_EXTENSIONS)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, nrows=max_rows)
    elif path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name=sheet_name or 0, nrows=max_rows)
    else:
        with path.open(encoding="utf-8-sig") as source:
            prefix = source.read(4096)
        first_char = next(
            (char for char in prefix if not char.isspace() and char != "\ufeff"),
            "",
        )
        if first_char == "[":
            frame = pd.DataFrame(_read_json_array(path, max_rows))
        else:
            is_json_lines = False
            nonempty_lines = []
            with path.open(encoding="utf-8-sig") as source:
                for _ in range(20):
                    line = source.readline(1_000_000)
                    if not line:
                        break
                    if line.strip():
                        nonempty_lines.append(line)
                    if len(nonempty_lines) == 2:
                        break
            if len(nonempty_lines) == 2 and all(
                line.endswith(("\n", "\r")) for line in nonempty_lines
            ):
                try:
                    json.loads(nonempty_lines[0])
                    json.loads(nonempty_lines[1])
                    is_json_lines = True
                except json.JSONDecodeError:
                    pass
            if is_json_lines:
                frame = pd.read_json(path, lines=True, nrows=max_rows)
            else:
                frame = pd.read_json(path).head(max_rows)
    return frame, path.name


def _read_json_array(path: Path, max_rows: int) -> list[Any]:
    """Stream a top-level JSON array without parsing records beyond the cap."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    array_started = False
    eof = False
    rows = []

    with path.open(encoding="utf-8-sig") as source:
        while len(rows) < max_rows:
            if position > 64_000:
                buffer = buffer[position:]
                position = 0
            if not eof and position >= len(buffer):
                chunk = source.read(64_000)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not array_started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError("The JSON file does not contain an array.")
                    continue
                if buffer[position] != "[":
                    raise ValueError("The JSON file does not contain an array.")
                position += 1
                array_started = True
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                break
            if position >= len(buffer):
                if eof:
                    break
                continue
            try:
                record, next_position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = source.read(64_000)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            rows.append(record)
            position = next_position
    return rows


def list_excel_sheets(upload_dir: Path | str, file_name: str) -> list[str]:
    """Return worksheet names for a validated XLSX upload."""
    path = safe_upload_path(upload_dir, file_name, frozenset({".xlsx"}))
    with pd.ExcelFile(path) as workbook:
        return list(workbook.sheet_names)


def create_chart_spec(
    upload_dir: Path | str,
    file_name: str,
    chart_type: str = "bar",
    x_column: str | None = None,
    y_column: str | None = None,
    color_column: str | None = None,
    aggregation: str = "none",
    top_n: int = 40,
    sheet_name: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, JSON-safe chart specification from an uploaded file."""
    try:
        frame, loaded_name = load_tabular_file(upload_dir, file_name, sheet_name)
    except (ImportError, OSError, ValueError, TypeError) as exc:
        return {"kind": "interactive_chart", "error": str(exc)}
    if frame.empty:
        return {"kind": "interactive_chart", "error": "The data file is empty."}

    chart_type = str(chart_type).strip().lower()
    aggregation = str(aggregation).strip().lower()
    if chart_type not in CHART_TYPES:
        return {
            "kind": "interactive_chart",
            "error": f"Choose a chart type from: {', '.join(sorted(CHART_TYPES))}.",
        }
    if aggregation not in {"none", "count", "sum", "mean"}:
        return {
            "kind": "interactive_chart",
            "error": "Aggregation must be none, count, sum, or mean.",
        }

    data = frame.copy()
    data.columns = [str(column) for column in data.columns]
    if data.columns.duplicated().any():
        return {
            "kind": "interactive_chart",
            "error": "Rename duplicate columns before creating a chart.",
        }
    columns = list(data.columns)
    numeric_columns = [
        column for column in columns if pd.api.types.is_numeric_dtype(data[column])
    ]
    category_columns = [column for column in columns if column not in numeric_columns]
    try:
        row_limit = max(2, min(int(top_n), MAX_CHART_POINTS))
    except (TypeError, ValueError):
        row_limit = 100

    if chart_type == "heatmap":
        if len(numeric_columns) < 2:
            return {
                "kind": "interactive_chart",
                "error": "A correlation heatmap needs at least two numeric columns.",
            }
        heatmap_columns = numeric_columns[:32]
        correlation = data[heatmap_columns].corr(numeric_only=True).round(3)
        return {
            "kind": "interactive_chart",
            "chart_type": chart_type,
            "title": title or f"Correlation matrix - {loaded_name}",
            "file": loaded_name,
            "worksheet": sheet_name,
            "rows_analyzed": int(len(data)),
            "columns_limited": len(numeric_columns) > len(heatmap_columns),
            "x_labels": list(correlation.columns),
            "y_labels": list(correlation.index),
            "matrix": json.loads(correlation.to_json(orient="values")),
        }

    histogram_x_label = None
    histogram_bin_count = None
    histogram_bin_width = None
    histogram_binned = False
    if chart_type == "histogram":
        x_column = x_column or (numeric_columns[0] if numeric_columns else None)
    elif chart_type == "scatter":
        x_column = x_column or (numeric_columns[0] if numeric_columns else None)
        y_column = y_column or next(
            (column for column in numeric_columns if column != x_column), None
        )
    elif chart_type in {"box", "violin"}:
        y_column = y_column or (numeric_columns[0] if numeric_columns else None)
        x_column = x_column or (category_columns[0] if category_columns else None)
    else:
        x_column = x_column or (category_columns[0] if category_columns else columns[0])
        if not (chart_type == "pie" and aggregation == "count"):
            y_column = y_column or next(
                (column for column in numeric_columns if column != x_column), None
            )
        if chart_type == "bar" and aggregation == "none":
            aggregation = "sum" if y_column else "count"
        if chart_type in {"bar", "line", "area"} and not y_column and aggregation == "none":
            aggregation = "count"

    for label, column in (("x", x_column), ("y", y_column), ("color", color_column)):
        if column is not None and column not in columns:
            return {
                "kind": "interactive_chart",
                "error": f"Unknown {label} column. Choose from: {', '.join(columns)}.",
            }

    if chart_type == "histogram":
        if not x_column or x_column not in numeric_columns:
            return {
                "kind": "interactive_chart",
                "error": "A histogram needs a numeric x column.",
            }
        if aggregation != "none":
            return {"kind": "interactive_chart", "error": "Histograms use their own bin counts."}
        values = pd.to_numeric(data[x_column], errors="coerce")
        valid = values.notna() & ~values.isin([float("inf"), float("-inf")])
        if not valid.any():
            return {"kind": "interactive_chart", "error": "No numeric values are available to plot."}
        histogram_x_label = x_column
        histogram_bin_count = min(48, max(8, int(int(valid.sum()) ** 0.5)))
        histogram_frame = data.loc[
            valid,
            list(dict.fromkeys([x_column] + ([color_column] if color_column else []))),
        ].copy()
        bin_column = "__chart_bin__"
        while bin_column in columns:
            bin_column += "_"
        count_column = "__chart_count__"
        while count_column in columns or count_column == bin_column:
            count_column += "_"
        histogram_frame[bin_column] = pd.cut(
            values.loc[valid], bins=histogram_bin_count, include_lowest=True
        )
        histogram_bin_width = float(
            histogram_frame[bin_column].cat.categories[0].length
        )
        group_columns = [bin_column]
        if color_column:
            histogram_frame[color_column] = histogram_frame[color_column].astype(object)
            popular_colors = histogram_frame[color_column].value_counts(dropna=False).head(8).index
            if histogram_frame[color_column].nunique(dropna=False) > 8:
                histogram_frame[color_column] = histogram_frame[color_column].where(
                    histogram_frame[color_column].isin(popular_colors), "Other"
                )
            group_columns.append(color_column)
        chart_data = (
            histogram_frame.groupby(
                group_columns, dropna=False, sort=False, observed=True
            )
            .size()
            .reset_index(name=count_column)
        )
        chart_data[bin_column] = chart_data[bin_column].map(
            lambda interval: float(interval.mid) if isinstance(interval, pd.Interval) else None
        )
        chart_data = chart_data.sort_values(bin_column, kind="stable")
        x_column = bin_column
        y_column = count_column
        y_label = "Records"
        histogram_binned = True
    elif chart_type in {"box", "violin"}:
        if not y_column or y_column not in numeric_columns:
            return {
                "kind": "interactive_chart",
                "error": f"A {chart_type} chart needs a numeric y column.",
            }
        selected = [column for column in (x_column, y_column, color_column) if column]
        chart_data = data[selected]
        y_label = y_column
    elif chart_type == "pie":
        if not x_column:
            return {"kind": "interactive_chart", "error": "A pie chart needs a category column."}
        if y_column and y_column not in numeric_columns:
            return {"kind": "interactive_chart", "error": "Pie values must use a numeric column."}
        if aggregation == "none" and y_column:
            aggregation = "sum"
        if aggregation == "count" or not y_column:
            chart_data = (
                data.groupby(x_column, dropna=False, sort=False)
                .size()
                .reset_index(name="__value__")
            )
            y_label = "Records"
        else:
            pie_aggregation = aggregation if aggregation in {"sum", "mean"} else "sum"
            chart_data = (
                data.groupby(x_column, dropna=False, sort=False)[y_column]
                .agg(pie_aggregation)
                .reset_index(name="__value__")
            )
            y_label = f"{pie_aggregation.title()} {y_column}"
        y_column = "__value__"
    elif chart_type in {"bar", "line", "area"}:
        if not x_column:
            return {"kind": "interactive_chart", "error": "Choose an x column for this chart."}
        if aggregation == "count":
            group_columns = list(dict.fromkeys([x_column] + ([color_column] if color_column else [])))
            chart_data = (
                data.groupby(group_columns, dropna=False, sort=False)
                .size()
                .reset_index(name="__value__")
            )
            y_column = "__value__"
            y_label = "Records"
        elif aggregation in {"sum", "mean"}:
            if not y_column or y_column not in numeric_columns:
                return {
                    "kind": "interactive_chart",
                    "error": f"{aggregation.title()} aggregation needs a numeric y column.",
                }
            group_columns = list(dict.fromkeys([x_column] + ([color_column] if color_column else [])))
            chart_data = (
                data.groupby(group_columns, dropna=False, sort=False)[y_column]
                .agg(aggregation)
                .reset_index(name="__value__")
            )
            y_label = f"{aggregation.title()} {y_column}"
            y_column = "__value__"
        else:
            if not y_column or y_column not in numeric_columns:
                return {
                    "kind": "interactive_chart",
                    "error": "Choose a numeric y column or use count aggregation.",
                }
            selected = list(dict.fromkeys([x_column, y_column] + ([color_column] if color_column else [])))
            chart_data = data[selected]
            y_label = y_column
    else:
        if not x_column or not y_column:
            return {
                "kind": "interactive_chart",
                "error": f"A {chart_type} chart needs numeric x and y columns.",
            }
        if aggregation != "none":
            return {
                "kind": "interactive_chart",
                "error": f"Aggregation is not supported for {chart_type} charts.",
            }
        if x_column not in numeric_columns or y_column not in numeric_columns:
            return {
                "kind": "interactive_chart",
                "error": "Scatter charts need numeric x and y columns.",
            }
        selected = list(dict.fromkeys([x_column, y_column] + ([color_column] if color_column else [])))
        chart_data = data[selected]
        y_label = y_column

    if chart_type in {"bar", "pie"} and "__value__" in chart_data.columns:
        chart_data = chart_data.sort_values("__value__", ascending=False)
    if chart_type in {"line", "area"} and x_column in chart_data.columns:
        if pd.api.types.is_datetime64_any_dtype(chart_data[x_column]):
            chart_data = chart_data.sort_values(x_column)

    rows_available = len(chart_data)
    sampled = rows_available > row_limit and not histogram_binned
    if chart_type in {"bar", "pie"} and "__value__" in chart_data.columns:
        chart_data = chart_data.head(row_limit)
    elif sampled:
        positions = [
            round(index * (rows_available - 1) / (row_limit - 1))
            for index in range(row_limit)
        ]
        chart_data = chart_data.iloc[positions]

    if title:
        chart_title = title
    elif chart_type == "histogram":
        chart_title = f"Distribution of {histogram_x_label}"
    else:
        chart_title = f"{y_label} by {x_column}" if x_column else "Data distribution"
    return {
        "kind": "interactive_chart",
        "chart_type": chart_type,
        "title": chart_title,
        "file": loaded_name,
        "worksheet": sheet_name,
        "x": x_column,
        "x_label": histogram_x_label or x_column,
        "y": y_column,
        "color": color_column if chart_type not in {"pie"} else None,
        "y_label": y_label,
        "aggregation": aggregation,
        "rows_analyzed": int(len(data)),
        "rows_available": int(rows_available),
        "rows_plotted": int(len(chart_data)),
        "sampled": sampled,
        "histogram_binned": histogram_binned,
        "bin_count": histogram_bin_count,
        "bin_width": histogram_bin_width,
        "data": json.loads(chart_data.to_json(orient="records", date_format="iso")),
    }


def data_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate explainable quality checks without assigning an opaque score."""
    row_count, column_count = frame.shape
    missing_by_column = []
    empty_columns = []
    constant_columns = []
    whitespace_padded_values = 0

    for index, label in enumerate(frame.columns):
        series = frame.iloc[:, index]
        missing_count = int(series.isna().sum())
        missing_by_column.append(
            {
                "column": str(label),
                "column_index": index,
                "missing": missing_count,
                "missing_percent": (
                    round(missing_count / row_count * 100, 2) if row_count else 0.0
                ),
            }
        )
        if row_count and missing_count == row_count:
            empty_columns.append(str(label))
        try:
            unique_count = int(series.nunique(dropna=False))
        except TypeError:
            unique_count = int(series.map(repr).nunique(dropna=False))
        if row_count and missing_count < row_count and unique_count <= 1:
            constant_columns.append(str(label))
        if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
            text_values = series.astype("string")
            whitespace_padded_values += int(
                text_values.str.strip().ne(text_values).fillna(False).sum()
            )

    try:
        duplicate_rows = int(frame.duplicated().sum())
    except (TypeError, ValueError):
        duplicate_rows = 0

    duplicate_column_names = [
        str(label)
        for label, is_duplicate in zip(frame.columns, frame.columns.duplicated())
        if is_duplicate
    ]
    total_missing = int(frame.isna().sum().sum())
    all_empty_rows = int(frame.isna().all(axis=1).sum()) if column_count else 0

    numeric_outliers = []
    for index, label in enumerate(frame.columns):
        series = frame.iloc[:, index]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            continue
        values = series.dropna()
        values = values[~values.isin([float("inf"), float("-inf")])]
        if len(values) < 4:
            continue
        lower_quartile = values.quantile(0.25)
        upper_quartile = values.quantile(0.75)
        spread = upper_quartile - lower_quartile
        outlier_count = int(
            ((values < lower_quartile - 1.5 * spread)
             | (values > upper_quartile + 1.5 * spread)).sum()
        )
        if outlier_count:
            numeric_outliers.append(
                {"column": str(label), "count": outlier_count, "rule": "1.5 x IQR"}
            )

    findings = []
    if total_missing:
        findings.append(
            {
                "severity": "Review",
                "check": "Missing values",
                "count": total_missing,
                "details": f"{total_missing} missing cell(s) across {row_count} row(s).",
            }
        )
    if duplicate_rows:
        findings.append(
            {
                "severity": "Review",
                "check": "Duplicate rows",
                "count": duplicate_rows,
                "details": "Rows identical across all columns after the first occurrence.",
            }
        )
    if duplicate_column_names:
        findings.append(
            {
                "severity": "Review",
                "check": "Duplicate column names",
                "count": len(duplicate_column_names),
                "details": ", ".join(duplicate_column_names),
            }
        )
    if empty_columns:
        findings.append(
            {
                "severity": "Review",
                "check": "Empty columns",
                "count": len(empty_columns),
                "details": ", ".join(empty_columns),
            }
        )
    if all_empty_rows:
        findings.append(
            {
                "severity": "Review",
                "check": "Empty rows",
                "count": all_empty_rows,
                "details": "Rows that contain no values in any column.",
            }
        )
    if constant_columns:
        findings.append(
            {
                "severity": "Info",
                "check": "Constant columns",
                "count": len(constant_columns),
                "details": ", ".join(constant_columns),
            }
        )
    if whitespace_padded_values:
        findings.append(
            {
                "severity": "Review",
                "check": "Padded text",
                "count": whitespace_padded_values,
                "details": "Text values have leading or trailing whitespace.",
            }
        )
    for outlier in numeric_outliers:
        findings.append(
            {
                "severity": "Inspect",
                "check": f"Possible outliers: {outlier['column']}",
                "count": outlier["count"],
                "details": "Flagged by the 1.5 x IQR rule; these may be valid values.",
            }
        )

    return {
        "rows_analyzed": int(row_count),
        "columns_analyzed": int(column_count),
        "missing_cells": total_missing,
        "missing_percent": (
            round(total_missing / (row_count * column_count) * 100, 2)
            if row_count and column_count
            else 0.0
        ),
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "duplicate_column_names": duplicate_column_names,
        "empty_columns": empty_columns,
        "empty_rows": all_empty_rows,
        "constant_columns": constant_columns,
        "whitespace_padded_values": whitespace_padded_values,
        "numeric_outliers_iqr": numeric_outliers,
        "findings": findings,
    }


def extract_document_text(
    upload_dir: Path | str,
    file_name: str,
    start_page: int = 1,
    max_pages: int = 10,
    max_chars: int = MAX_DOCUMENT_CHARS,
) -> dict[str, Any]:
    """Extract bounded text from a PDF or plain-text upload."""
    path = safe_upload_path(upload_dir, file_name, DOCUMENT_EXTENSIONS)
    max_chars = max(1, min(int(max_chars), MAX_DOCUMENT_CHARS))
    if path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8", errors="replace") as source:
            text = source.read(max_chars + 1)
        return {
            "file": path.name,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
            "message": "Text file extraction is limited to 20,000 characters.",
        }

    try:
        from pypdf import PdfReader
    except ImportError:
        return {"error": "PDF extraction requires pypdf. Install project dependencies."}

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return {"file": path.name, "error": "This PDF is password-protected."}
        page_count = len(reader.pages)
        if page_count == 0:
            return {"file": path.name, "page_count": 0, "text": ""}
        first_page = max(1, min(int(start_page), page_count))
        page_limit = max(1, min(int(max_pages), MAX_DOCUMENT_PAGES))
        last_page = min(page_count, first_page + page_limit - 1)
        chunks = []
        has_extractable_text = False
        for page_number in range(first_page, last_page + 1):
            page_text = reader.pages[page_number - 1].extract_text() or ""
            has_extractable_text = has_extractable_text or bool(page_text.strip())
            chunks.append(f"--- Page {page_number} ---\n{page_text.strip()}")
        text = "\n\n".join(chunks) if has_extractable_text else ""
    except Exception as exc:
        return {"file": path.name, "error": f"Could not extract PDF text: {exc}"}

    truncated = len(text) > max_chars
    return {
        "file": path.name,
        "page_count": page_count,
        "pages_extracted": list(range(first_page, last_page + 1)),
        "text": text[:max_chars],
        "truncated": truncated,
        "message": (
            "No selectable text found. Scanned PDFs require OCR."
            if not has_extractable_text
            else "PDF extraction is limited to 20 pages and 20,000 characters per request."
        ),
    }


def build_quality_dashboard_html(
    frame: pd.DataFrame,
    file_name: str,
    sheet_name: str | None = None,
) -> str:
    """Create an escaped HTML report with an interactive missingness chart."""
    report = data_quality_report(frame)
    title = f"{file_name}{f' - {sheet_name}' if sheet_name else ''}"
    missing_rows = [item for item in report["missing_by_column"] if item["missing"]]
    missing_chart = {
        "columns": [item["column"] for item in missing_rows[:20]],
        "percentages": [item["missing_percent"] for item in missing_rows[:20]],
        "counts": [item["missing"] for item in missing_rows[:20]],
    }
    chart_json = json.dumps(missing_chart, ensure_ascii=True, allow_nan=False).replace(
        "</", "<\\/"
    )
    if missing_rows:
        missing_html = pd.DataFrame(missing_rows).to_html(
            index=False, escape=True, classes="data-table", border=0
        )
    else:
        missing_html = '<p class="quiet">No missing values detected.</p>'
    schema = pd.DataFrame(
        [
            {
                "column": str(label),
                "type": str(frame.iloc[:, index].dtype),
                "missing": report["missing_by_column"][index]["missing"],
                "missing_percent": report["missing_by_column"][index]["missing_percent"],
            }
            for index, label in enumerate(frame.columns)
        ]
    )
    if schema.empty:
        schema_html = '<p class="quiet">No columns available.</p>'
    else:
        schema_html = schema.to_html(
            index=False, escape=True, classes="data-table", border=0
        )
    if report["findings"]:
        findings_html = pd.DataFrame(report["findings"]).to_html(
            index=False, escape=True, classes="data-table", border=0
        )
    else:
        findings_html = '<p class="good">No common quality issues detected.</p>'

    preview = frame.head(10).astype(object).where(pd.notna(frame.head(10)), "-")
    preview_html = preview.to_html(
        index=False, escape=True, classes="data-table", border=0
    )
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        numeric_html = '<p class="quiet">No numeric columns to summarize.</p>'
    else:
        summary = numeric.describe().transpose().reset_index().rename(
            columns={"index": "column"}
        )
        numeric_html = summary.round(3).to_html(
            index=False, escape=True, classes="data-table", border=0
        )

    safe_title = html.escape(title)
    missing_percent = report["missing_percent"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title} | Data Quality Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{ color-scheme: dark; font-family: anthropic-sans, "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background: #111315; color: #eef1f4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: clamp(1rem, 4vw, 3rem); }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    header {{ border-bottom: 1px solid #30343a; padding-bottom: 1.5rem; margin-bottom: 1.5rem; }}
    h1 {{ font-size: clamp(1.7rem, 3vw, 2.4rem); margin: .2rem 0 .5rem; }}
    h2 {{ font-size: 1.1rem; margin: 1.8rem 0 .8rem; }}
    p, .quiet {{ color: #9ca3af; }}
    .eyebrow {{ color: #75d9a6; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; }}
    .card, section {{ background: #191c20; border: 1px solid #2b3037; border-radius: 14px; padding: 1rem; }}
    .card span {{ color: #9ca3af; font-size: .82rem; }}
    .card strong {{ display: block; font-size: 1.55rem; margin-top: .35rem; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: .88rem; overflow-wrap: anywhere; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #30343a; padding: .65rem .7rem; text-align: left; }}
    .data-table th {{ color: #b5bdc7; font-weight: 600; }}
    .data-table td {{ color: #e3e7eb; }}
    .table-wrap {{ overflow-x: auto; }}
    .chart-panel {{ min-height: 330px; padding: .5rem .25rem; }}
    #missingChart {{ min-height: 310px; width: 100%; }}
    .chart-empty {{ color: #8f9aa1; display: grid; min-height: 290px; place-items: center; text-align: center; }}
    .good {{ color: #75d9a6; }}
    footer {{ color: #737b86; font-size: .78rem; margin: 1.2rem 0; }}
    @media (max-width: 600px) {{ .data-table {{ min-width: 520px; }} .chart-panel {{ padding: .25rem 0; }} }}
  </style>
</head>
<body><main>
  <header><div class="eyebrow">Data Quality Dashboard</div><h1>{safe_title}</h1>
    <p>Deterministic checks on the first {report['rows_analyzed']:,} analyzed row(s); no opaque quality score.</p>
  </header>
  <div class="cards">
    <div class="card"><span>Rows analyzed</span><strong>{report['rows_analyzed']:,}</strong></div>
    <div class="card"><span>Columns</span><strong>{report['columns_analyzed']:,}</strong></div>
    <div class="card"><span>Missing cells</span><strong>{report['missing_cells']:,}</strong></div>
    <div class="card"><span>Duplicate rows</span><strong>{report['duplicate_rows']:,}</strong></div>
    <div class="card"><span>Missing rate</span><strong>{missing_percent:.2f}%</strong></div>
  </div>
  <h2>Quality findings</h2><section>{findings_html}</section>
  <h2>Schema</h2><section class="table-wrap">{schema_html}</section>
  <h2>Missing values by column</h2>
  <section class="chart-panel"><div id="missingChart" role="img" aria-label="Missing values by column"></div></section>
  <section class="table-wrap">{missing_html}</section>
  <h2>Numeric summary</h2><section class="table-wrap">{numeric_html}</section>
  <h2>Preview (up to 10 rows)</h2><section class="table-wrap">{preview_html}</section>
  <footer>Analysis is local to the app process. Outliers use the 1.5 x IQR rule and should be reviewed, not assumed to be errors.</footer>
</main>
<script type="application/json" id="missing-chart-data">{chart_json}</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
  const qualityChartData = JSON.parse(document.getElementById("missing-chart-data").textContent);
  const chartNode = document.getElementById("missingChart");
  if (!qualityChartData.columns.length) {{
    chartNode.innerHTML = '<div class="chart-empty">No missing values detected.</div>';
  }} else if (!window.Plotly) {{
    chartNode.innerHTML = '<div class="chart-empty">Connect to the internet to load the interactive chart.</div>';
  }} else {{
    Plotly.newPlot(chartNode, [{{
      type: "bar",
      orientation: "h",
      x: qualityChartData.percentages,
      y: qualityChartData.columns,
      customdata: qualityChartData.counts,
      marker: {{ color: qualityChartData.percentages, colorscale: [[0, "#28584c"], [1, "#54d2a0"]], line: {{ width: 0 }} }},
      texttemplate: "%{{x:.1f}}%",
      textposition: "outside",
      cliponaxis: false,
      hovertemplate: "<b>%{{y}}</b><br>%{{x:.2f}}% missing<br>%{{customdata:,}} cells<extra></extra>"
    }}], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{ family: 'Inter, system-ui, sans-serif', color: "#c8d0d4", size: 12 }},
      margin: {{ l: 12, r: 56, t: 20, b: 42 }},
      xaxis: {{ title: "Missing cells (%)", range: [0, Math.min(100, Math.max(1, Math.max(...qualityChartData.percentages) * 1.25))], gridcolor: "rgba(156,170,176,.14)", zeroline: false }},
      yaxis: {{ automargin: true, gridcolor: "rgba(0,0,0,0)", zeroline: false }},
      showlegend: false,
      height: Math.max(310, Math.min(680, qualityChartData.columns.length * 42 + 100))
    }}, {{ responsive: true, displaylogo: false, scrollZoom: true }});
  }}
</script>
</body></html>"""
