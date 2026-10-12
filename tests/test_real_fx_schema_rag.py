"""Tests use fixture exchange rates, never presented as actual market data."""
import tempfile
from pathlib import Path

import pytest

from diffragsql.real_fx_sql import build_original_sqlite
from diffragsql.real_fx_date_windows import EVIDENCE_QUESTIONS
from diffragsql.real_fx_schema_rag import grounded_sql_answer, source_schema_evidence


def test_schema_retrieval_agrees_with_four_safe_sql_date_operators():
    with tempfile.TemporaryDirectory() as folder:
        con = build_original_sqlite(
            Path(folder) / "fixture.sqlite3", [
                ("2025-01-02", 1.4), ("2025-01-03", 1.5),
                ("2025-04-02", 1.3), ("2025-07-02", 1.2),
                ("2025-10-02", 1.1),
            ])
        try:
            assert source_schema_evidence(con)["source_schema_checked_in_sqlite"]
            for query, expected in zip(EVIDENCE_QUESTIONS, [2, 1.3, 1.2, 1.1]):
                result = grounded_sql_answer(con, query)
                assert result["actual_sql_answer"] == pytest.approx(expected)
                assert result["retrieved_operator"] == result["operator"]
                assert result["retrieval_similarity"] > 0
                assert result["retrieval_is_evidence_not_a_free_sql_generator"]
        finally:
            con.close()


def test_unapproved_sql_and_date_injection_are_refused():
    with tempfile.TemporaryDirectory() as folder:
        con = build_original_sqlite(
            Path(folder) / "fixture.sqlite3", [("2025-01-02", 1.4)]
        )
        try:
            for query in ["DROP TABLE fx_usdcad",
                          "What was the average official USD to CAD rate between 2025-01-01 and 2025-12-31?; DELETE FROM fx_usdcad"]:
                with pytest.raises(ValueError):
                    grounded_sql_answer(con, query)
            assert con.execute("SELECT COUNT(*) FROM fx_usdcad").fetchone()[0] == 1
        finally:
            con.close()


def test_fail_closed_when_expected_original_schema_or_series_missing():
    import sqlite3
    with tempfile.TemporaryDirectory() as folder:
        con = sqlite3.connect(Path(folder) / "bad.sqlite3")
        con.execute("CREATE TABLE fx_usdcad (date TEXT, other REAL)")
        with pytest.raises(ValueError, match="schema"):
            source_schema_evidence(con)
        con.close()
