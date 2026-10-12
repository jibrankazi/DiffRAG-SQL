"""In-memory software fixtures only. No invented Bank of Canada observations."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from diffragsql.real_fx_date_windows import EVIDENCE_QUESTIONS, answer, interpret
from diffragsql.real_fx_sql import build_original_sqlite

class DateWindowQueryContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.connection = build_original_sqlite(
            Path(self.temp.name) / "fixture.sqlite3",
            [("2025-01-02", 1.4), ("2025-01-03", 1.5),
             ("2025-04-02", 1.3), ("2025-07-02", 1.2),
             ("2025-10-02", 1.1)])
    def tearDown(self):
        self.connection.close()
        self.temp.cleanup()

    def test_four_bounded_operator_types(self):
        for question, expected in zip(EVIDENCE_QUESTIONS, [2, 1.3, 1.2, 1.1]):
            with self.subTest(question=question):
                row = answer(self.connection, question)
                self.assertAlmostEqual(row["actual_sql_answer"], expected)
                self.assertEqual(row["executed_sql"].count("?"), 2)
                self.assertEqual(len(row["bound_parameters"]), 2)

    def test_invalid_dates_and_arbitrary_language_are_refused(self):
        invalid = [
            "What was the average official USD to CAD rate between 2024-01-01 and 2025-03-31?",
            "What was the average official USD to CAD rate between 2025-04-01 and 2025-03-01?",
            "What was the average official USD to CAD rate between 2025-02-29 and 2025-03-31?",
            "What was the average official USD to CAD rate between 2025-01-01 and 2026-01-01?",
            "What was the average official USD to CAD rate between 2025-01-01 and 2025-03-31?; DROP TABLE fx_usdcad",
            "Delete the FX table",
        ]
        for question in invalid:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    interpret(question)

    def test_query_is_read_only_even_when_direct_db_write_attempted(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.connection.execute("DELETE FROM fx_usdcad")
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM fx_usdcad").fetchone()[0], 5)

    def test_missing_rates_fail_closed_for_average(self):
        question = "What was the average official USD to CAD rate between 2025-02-01 and 2025-02-28?"
        with self.assertRaises(ValueError):
            answer(self.connection, question)

if __name__ == "__main__":
    unittest.main()
