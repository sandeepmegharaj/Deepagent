import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pypdf import PdfWriter

from data_tools import (
    build_quality_dashboard_html,
    create_chart_spec,
    data_quality_report,
    extract_document_text,
    list_excel_sheets,
    load_tabular_file,
    safe_upload_path,
)


class DataToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_quality_report_flags_common_issues(self):
        frame = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "score": [1, 1, 2, 100],
                "note": ["clean", "clean", " padded ", "padded"],
                "empty": [None, None, None, None],
                "constant": ["x", "x", "x", "x"],
            }
        )

        report = data_quality_report(frame)

        self.assertEqual(report["missing_cells"], 4)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["empty_columns"], ["empty"])
        self.assertIn("constant", report["constant_columns"])
        self.assertNotIn("empty", report["constant_columns"])
        self.assertEqual(report["whitespace_padded_values"], 1)
        self.assertEqual(report["numeric_outliers_iqr"][0]["count"], 1)

    def test_load_csv_and_excel_sheets(self):
        csv_path = self.upload_dir / "sample.csv"
        pd.DataFrame({"value": [1, 2]}).to_csv(csv_path, index=False)
        csv_frame, csv_name = load_tabular_file(self.upload_dir, "sample.csv")
        self.assertEqual(csv_name, "sample.csv")
        self.assertEqual(csv_frame["value"].tolist(), [1, 2])

        excel_path = self.upload_dir / "sample.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame({"value": [3]}).to_excel(writer, sheet_name="North", index=False)
            pd.DataFrame({"value": [4]}).to_excel(writer, sheet_name="South", index=False)
        self.assertEqual(list_excel_sheets(self.upload_dir, "sample.xlsx"), ["North", "South"])
        south_frame, _ = load_tabular_file(self.upload_dir, "sample.xlsx", "South")
        self.assertEqual(south_frame["value"].tolist(), [4])

    def test_loads_json_records(self):
        json_path = self.upload_dir / "records.json"
        json_path.write_text(
            '[{"value": 5}, {"value": 6}, invalid trailing data]',
            encoding="utf-8",
        )
        frame, _ = load_tabular_file(self.upload_dir, "records.json", max_rows=2)
        self.assertEqual(frame["value"].tolist(), [5, 6])

        jsonl_path = self.upload_dir / "records-lines.json"
        jsonl_path.write_text('{"value": 7}\n{"value": 8}\n', encoding="utf-8")
        jsonl_frame, _ = load_tabular_file(self.upload_dir, "records-lines.json")
        self.assertEqual(jsonl_frame["value"].tolist(), [7, 8])

    def test_rejects_symlinks_outside_upload_directory(self):
        outside = self.upload_dir.parent / "outside.csv"
        outside.write_text("value\n1\n", encoding="utf-8")
        link = self.upload_dir / "link.csv"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Symlinks require administrator privilege on this system")
        with self.assertRaises(ValueError):
            safe_upload_path(self.upload_dir, "link.csv")

    def test_extracts_text_file_with_character_limit(self):
        text_path = self.upload_dir / "notes.txt"
        text_path.write_text("abcdef", encoding="utf-8")
        result = extract_document_text(self.upload_dir, "notes.txt", max_chars=3)
        self.assertEqual(result["text"], "abc")
        self.assertTrue(result["truncated"])

    def test_extracts_pdf_page_metadata(self):
        pdf_path = self.upload_dir / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with pdf_path.open("wb") as output:
            writer.write(output)

        result = extract_document_text(self.upload_dir, "blank.pdf")

        self.assertEqual(result["page_count"], 1)
        self.assertEqual(result["pages_extracted"], [1])
        self.assertIn("OCR", result["message"])

    def test_dashboard_escapes_uploaded_values(self):
        frame = pd.DataFrame({"<column>": ["<script>alert(1)</script>"]})
        html = build_quality_dashboard_html(frame, "<unsafe>.csv")
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;unsafe&gt;.csv", html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_chart_spec_groups_and_bounds_uploaded_data(self):
        pd.DataFrame(
            {
                "region": ["North", "South", "North", "West"],
                "revenue": [12, 7, 18, 3],
            }
        ).to_csv(self.upload_dir / "sales.csv", index=False)

        spec = create_chart_spec(
            self.upload_dir,
            "sales.csv",
            chart_type="bar",
            x_column="region",
            y_column="revenue",
            aggregation="sum",
            top_n=2,
        )

        self.assertEqual(spec["kind"], "interactive_chart")
        self.assertEqual(spec["chart_type"], "bar")
        self.assertEqual(spec["rows_analyzed"], 4)
        self.assertEqual(spec["rows_plotted"], 2)
        self.assertEqual(spec["data"][0], {"region": "North", "__value__": 30})

    def test_bar_chart_defaults_to_category_counts_without_numeric_columns(self):
        pd.DataFrame({"team": ["A", "B", "A"]}).to_csv(
            self.upload_dir / "teams.csv", index=False
        )

        spec = create_chart_spec(self.upload_dir, "teams.csv", chart_type="bar")

        self.assertEqual(spec["aggregation"], "count")
        self.assertEqual(spec["y_label"], "Records")
        self.assertEqual(spec["data"][0], {"team": "A", "__value__": 2})

    def test_histogram_bins_finite_values_without_sampling_bins(self):
        pd.DataFrame(
            {
                "amount": list(range(200)) + [float("inf"), None],
                "segment": [f"Group {index % 10}" for index in range(200)]
                + ["Ignored", "Ignored"],
            }
        ).to_csv(self.upload_dir / "amounts.csv", index=False)

        spec = create_chart_spec(
            self.upload_dir,
            "amounts.csv",
            chart_type="histogram",
            x_column="amount",
            color_column="segment",
            top_n=4,
        )

        self.assertTrue(spec["histogram_binned"])
        self.assertEqual(spec["x_label"], "amount")
        self.assertEqual(spec["title"], "Distribution of amount")
        self.assertEqual(spec["bin_count"], 14)
        self.assertGreater(spec["bin_width"], 0)
        self.assertFalse(spec["sampled"])
        self.assertGreater(spec["rows_plotted"], 4)
        self.assertEqual(
            sum(row["__chart_count__"] for row in spec["data"]), 200
        )
        self.assertIn("Other", {row["segment"] for row in spec["data"]})
        centers = [row["__chart_bin__"] for row in spec["data"]]
        self.assertEqual(centers, sorted(centers))

    def test_heatmap_spec_uses_numeric_correlation(self):
        pd.DataFrame(
            {"first": [1, 2, 3], "second": [2, 4, 6], "group": ["a", "b", "a"]}
        ).to_csv(self.upload_dir / "metrics.csv", index=False)

        spec = create_chart_spec(
            self.upload_dir, "metrics.csv", chart_type="heatmap"
        )

        self.assertEqual(spec["x_labels"], ["first", "second"])
        self.assertEqual(spec["matrix"][0][1], 1.0)

    def test_dashboard_escapes_script_terminators_in_chart_labels(self):
        label = "</script><script>alert(1)</script>"
        frame = pd.DataFrame({label: [None, 1]})

        html = build_quality_dashboard_html(frame, "safe.csv")

        self.assertNotIn(label, html)
        self.assertIn("<\\/script>", html)
        self.assertIn("plotly-2.35.2.min.js", html)


if __name__ == "__main__":
    unittest.main()
