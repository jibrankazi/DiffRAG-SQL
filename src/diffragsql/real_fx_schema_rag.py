"""Source-backed schema retrieval + authorized deterministic SQL evaluation.

A *non-differentiable* TF-IDF retrieval module grounds the approved query in
the original read-only Bank of Canada FX SQLite table before execution. No
learned text-to-SQL, broad NL parsing or LLM generation is claimed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from diffragsql.real_fx_date_windows import EVIDENCE_QUESTIONS, answer, interpret
from diffragsql.real_fx_sql import (
    SOURCE, build_original_sqlite, fetch_actual_observations,
)
from diffragsql.retriever import TFIDFRetriever

DOCS = {
    "count": "How many count number of official Bank of Canada FXUSDCAD dated rows daily observations are published in a specified time window?",
    "average": "What is the arithmetic average mean FXUSDCAD official daily CAD per USD foreign exchange rate over a date interval?",
    "maximum": "What is the maximum highest peak largest official historical FXUSDCAD CAD per USD exchange rate in a time window?",
    "minimum": "What is the minimum lowest smallest official historical FXUSDCAD CAD per USD exchange rate in a time window?",
}


def source_schema_evidence(conn):
    columns = conn.execute("PRAGMA table_info(fx_usdcad)").fetchall()
    available = {record[1] for record in columns}
    if not {"date", "cad_per_usd", "original_series"}.issubset(available):
        raise ValueError("Original Bank of Canada FX source schema cannot be confirmed")
    names = conn.execute("SELECT DISTINCT original_series FROM fx_usdcad").fetchall()
    if names != [("FXUSDCAD",)]:
        raise ValueError("SQL data are not solely original FXUSDCAD observations")
    return {"source_url": SOURCE, "table_name": "fx_usdcad",
            "verified_original_column_names": sorted(available),
            "actual_database_series": "FXUSDCAD",
            "retrieval_method": "TF-IDF over four explanatory schema/operator notes (not trained)",
            "source_schema_checked_in_sqlite": True}


def grounded_sql_answer(conn, question):
    """Require schema, lexical supporting note and exact vetted query operator."""
    schema = source_schema_evidence(conn)
    sql, expected_operator, params = interpret(question)
    documents = list(DOCS.values())
    retriever = TFIDFRetriever(documents)
    hit = retriever.search(question, k=1)[0]
    if hit["score"] <= 0:
        raise ValueError("No schema/operator evidence retrieved for question")
    retrieved_operator = next(k for k, v in DOCS.items() if v == hit["doc"])
    if retrieved_operator != expected_operator:
        raise ValueError("Retrieved schema operation disagrees with the safe query parser")
    resolved = answer(conn, question)
    if resolved["executed_sql"] != sql or resolved["bound_parameters"] != list(params):
        raise AssertionError("Retrieved instruction changed the vetted SQL")
    resolved["schema_evidence"] = schema
    resolved["retrieved_schema_note"] = hit["doc"]
    resolved["retrieved_operator"] = retrieved_operator
    resolved["retrieval_similarity"] = hit["score"]
    resolved["retrieval_is_evidence_not_a_free_sql_generator"] = True
    return resolved


def run(output="runs/real_fx_schema_grounded_sql"):
    """Real original rows -> SQLite -> schema evidence -> retrieval -> vetted SQL."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    observations, digest = fetch_actual_observations()
    conn = build_original_sqlite(folder / "genuine_2025_fx.sqlite3", observations)
    evidence = []
    try:
        for question in EVIDENCE_QUESTIONS:
            record = grounded_sql_answer(conn, question)
            start, end = record["bound_parameters"]
            original = [rate for day, rate in observations if start <= day <= end]
            if not original:
                raise AssertionError("Unsupported empty historical observation interval")
            reference = {
                "count": len(original),
                "average": sum(original) / len(original),
                "maximum": max(original),
                "minimum": min(original),
            }[record["operator"]]
            if abs(reference - record["actual_sql_answer"]) > 1e-9:
                raise AssertionError("Schema-retrieved answer does not match independent original data")
            record["original_observation_count_in_window"] = len(original)
            record["independent_python_reference_verified"] = True
            evidence.append(record)
    finally:
        conn.close()
    report = {
        "actual_publisher": "Bank of Canada Valet",
        "original_source_url": SOURCE,
        "original_download_sha256": digest,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "genuine_source_rows": len(observations),
        "grounded_retrieval_and_sql_answers": len(evidence),
        "verified_against_actual_original_source_rows": len(evidence),
        "integrated_tf_idf_schema_retrieval": True,
        "trained_text_to_sql_model": False,
        "end_to_end_differentiable_rag": False,
        "arbitrary_sql_accepted": False,
        "records": evidence,
    }
    (folder / "source_grounded_sql_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/real_fx_schema_grounded_sql")
    run(p.parse_args().output)
