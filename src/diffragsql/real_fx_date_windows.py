"""Audited date-bounded English-to-SQL over *real* 2025 Bank of Canada data.

This is a deterministic safe grammar, NOT trained NL-to-SQL, differentiable
retrieval, or financial forecasting. Queries never access external tables.
"""
import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from sqlite3 import Connection

from diffragsql.real_fx_sql import SOURCE, build_original_sqlite, fetch_actual_observations

OPERATORS = {
    "How many official USD to CAD daily observations were published":
        ("SELECT COUNT(*) FROM fx_usdcad WHERE date >= ? AND date <= ?", "count"),
    "What was the average official USD to CAD rate":
        ("SELECT AVG(cad_per_usd) FROM fx_usdcad WHERE date >= ? AND date <= ?", "average"),
    "What was the maximum official USD to CAD rate":
        ("SELECT MAX(cad_per_usd) FROM fx_usdcad WHERE date >= ? AND date <= ?", "maximum"),
    "What was the minimum official USD to CAD rate":
        ("SELECT MIN(cad_per_usd) FROM fx_usdcad WHERE date >= ? AND date <= ?", "minimum"),
}
PATTERN = re.compile(
    r"(?P<operation>.+?) between (?P<start>[0-9]{4}-[0-9]{2}-[0-9]{2}) "
    r"and (?P<end>[0-9]{4}-[0-9]{2}-[0-9]{2})\?"
)
EVIDENCE_QUESTIONS = [
    "How many official USD to CAD daily observations were published between 2025-01-01 and 2025-03-31?",
    "What was the average official USD to CAD rate between 2025-04-01 and 2025-06-30?",
    "What was the maximum official USD to CAD rate between 2025-07-01 and 2025-09-30?",
    "What was the minimum official USD to CAD rate between 2025-10-01 and 2025-12-31?",
]


def interpret(question):
    match = PATTERN.fullmatch(question)
    if match is None or match["operation"] not in OPERATORS:
        raise ValueError("Unsupported question: only four approved date-bounded operators")
    try:
        first = date.fromisoformat(match["start"])
        last = date.fromisoformat(match["end"])
    except ValueError as exc:
        raise ValueError("Invalid historical calendar date") from exc
    if not date(2025, 1, 1) <= first <= last <= date(2025, 12, 31):
        raise ValueError("Date range must be ordered and fully inside 2025")
    sql, operation = OPERATORS[match["operation"]]
    return sql, operation, (first.isoformat(), last.isoformat())


def answer(conn: Connection, question):
    sql, operation, parameters = interpret(question)
    # Query parameter binding, fixed SQL templates and query_only sqlite mode.
    values = conn.execute(sql, parameters).fetchall()
    if len(values) != 1 or values[0][0] is None:
        raise ValueError("No observed exchange-rate values for this question")
    return {
        "question": question,
        "operator": operation,
        "executed_sql": sql,
        "bound_parameters": list(parameters),
        "actual_sql_answer": values[0][0],
        "original_series": "FXUSDCAD",
        "source_url": SOURCE,
    }


def evaluate(output="runs/real_fx_date_windows", extra_question=None):
    """Re-download the official source, then independently cross-check all SQL."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    observations, body_hash = fetch_actual_observations()
    conn = build_original_sqlite(root / "original_2025_usdcad.sqlite3", observations)
    try:
        queries = list(EVIDENCE_QUESTIONS)
        if extra_question is not None and extra_question not in queries:
            queries.append(extra_question)
        records = []
        for question in queries:
            result = answer(conn, question)
            start, end = result["bound_parameters"]
            observed = [rate for day, rate in observations if start <= day <= end]
            if result["operator"] == "count":
                reference = len(observed)
            elif not observed:
                raise ValueError("No original market observation available for the requested date range")
            elif result["operator"] == "average":
                reference = sum(observed) / len(observed)
            elif result["operator"] == "maximum":
                reference = max(observed)
            else:
                reference = min(observed)
            if abs(result["actual_sql_answer"] - reference) > 1e-9:
                raise AssertionError("Original-source Python reference does not match SQLite")
            result["independent_source_python_verified"] = True
            result["original_observations_in_window"] = len(observed)
            records.append(result)
    finally:
        conn.close()
    report = {
        "publisher": "Bank of Canada Valet",
        "source_url": SOURCE,
        "downloaded_body_sha256": body_hash,
        "extraction_utc": datetime.now(timezone.utc).isoformat(),
        "original_observed_rows": len(observations),
        "date_bounded_queries_executed": len(records),
        "independently_verified_sql_answers": len(records),
        "raw_sql_or_unrecognized_question_allowed": False,
        "learned_natural_language_or_differentiable_rag_implemented": False,
        "records": records,
    }
    (root / "date_window_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", default=None, help="Exact approved English date-range question")
    ap.add_argument("--output", default="runs/real_fx_date_windows")
    ap.add_argument("--list-operators", action="store_true")
    options = ap.parse_args()
    if options.list_operators:
        print(json.dumps({"operators": list(OPERATORS), "examples": EVIDENCE_QUESTIONS,
                          "year": 2025}, indent=2))
    else:
        evaluate(output=options.output, extra_question=options.question)
