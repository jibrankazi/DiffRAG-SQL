"""Real Bank of Canada Valet observations -> SQLite -> grounded question answers.

This implements an honestly scoped deterministic *read-only text-to-SQL*
baseline, not differentiable retrieval, gradient training, or an LLM.
The SQL executes against original public numeric FX observations.
"""
import argparse
import gzip
from hashlib import sha256
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = ("https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json"
          "?start_date=2025-01-01&end_date=2025-12-31")
QUESTIONS = {
    "How many official USD to CAD daily observations were published during 2025?":
        ("SELECT COUNT(*) FROM fx_usdcad", "count"),
    "What was the average official USD to CAD rate for 2025?":
        ("SELECT AVG(cad_per_usd) FROM fx_usdcad", "average"),
    "What was the maximum official USD to CAD rate in 2025?":
        ("SELECT MAX(cad_per_usd) FROM fx_usdcad", "maximum"),
    "What was the minimum official USD to CAD rate in 2025?":
        ("SELECT MIN(cad_per_usd) FROM fx_usdcad", "minimum"),
    "What was the first available 2025 USD to CAD observation?":
        ("SELECT date, cad_per_usd FROM fx_usdcad ORDER BY date ASC LIMIT 1", "first"),
    "What was the last available 2025 USD to CAD observation?":
        ("SELECT date, cad_per_usd FROM fx_usdcad ORDER BY date DESC LIMIT 1", "last"),
    "Which five 2025 dates had the highest official USD to CAD rates?":
        ("SELECT date, cad_per_usd FROM fx_usdcad ORDER BY cad_per_usd DESC, date ASC LIMIT 5", "top_5"),
}


def fetch_actual_observations():
    request = Request(SOURCE, headers={
        "User-Agent": "EvidenceFirstSQLResearch/1.0 (https://github.com/jibrankazi/DiffRAG-SQL)",
        "Accept-Encoding": "identity",
    })
    with urlopen(request, timeout=50) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
    payload = json.loads(raw)
    rows = []
    for item in payload.get("observations", []):
        value = item.get("FXUSDCAD", {}).get("v")
        if value is None:
            continue
        date = item["d"]
        if not ("2025-01-01" <= date <= "2025-12-31"):
            raise ValueError(f"Outside the official requested observation year: {date}")
        rate = float(value)
        if not 0.5 <= rate <= 3.0:
            raise ValueError(f"Unexpected Canadian dollars per US dollar: {rate}")
        rows.append((date, rate))
    rows.sort()
    if len(rows) < 200 or len(rows) > 260 or len({r[0] for r in rows}) != len(rows):
        raise ValueError("2025 Bank of Canada USD/CAD daily source does not meet expected historical contract")
    return rows, sha256(raw).hexdigest()


def build_original_sqlite(db_path, observations):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("""CREATE TABLE fx_usdcad (
        date TEXT PRIMARY KEY NOT NULL,
        cad_per_usd REAL NOT NULL CHECK (cad_per_usd > 0),
        original_series TEXT NOT NULL DEFAULT 'FXUSDCAD'
    )""")
    connection.executemany(
        "INSERT INTO fx_usdcad(date,cad_per_usd) VALUES(?,?)", observations
    )
    connection.commit()
    connection.execute("PRAGMA query_only = ON")
    return connection


def interpret(question):
    """Only authored, auditable SQL templates; refuse arbitrary unapproved SQL."""
    if question not in QUESTIONS:
        raise ValueError("Unsupported question: no verified SQL template; refusing to guess")
    return QUESTIONS[question]


def answer(connection, question):
    sql, kind = interpret(question)
    if not sql.startswith("SELECT ") or ";" in sql or "fx_usdcad" not in sql:
        raise ValueError("Only a single audited read-only SELECT can run")
    rows = connection.execute(sql).fetchall()
    if not rows or rows[0][0] is None:
        raise ValueError("SQL returned no grounded result")
    return {"question": question, "generated_from": "whitelisted deterministic templates",
            "executed_sql": sql, "answer_type": kind, "actual_sql_rows": rows,
            "support_table": "fx_usdcad", "source_url": SOURCE}


def evaluate(output_dir="runs/real_fx_sql"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    originals, digest = fetch_actual_observations()
    conn = build_original_sqlite(output / "real_fx_2025.sqlite3", originals)
    try:
        validated = []
        numbers = [value for _, value in originals]
        reference = {
            "count": len(originals),
            "average": sum(numbers) / len(numbers),
            "maximum": max(numbers), "minimum": min(numbers),
            "first": originals[0], "last": originals[-1],
            "top_5": sorted(originals, key=lambda r: (-r[1], r[0]))[:5],
        }
        for question in QUESTIONS:
            observed = answer(conn, question)
            kind = observed["answer_type"]
            actual = observed["actual_sql_rows"]
            if kind in ("count", "average", "maximum", "minimum"):
                if abs(actual[0][0] - reference[kind]) > 1e-9:
                    raise AssertionError(f"SQL/Python ground-truth mismatch: {kind}")
            elif kind in ("first", "last"):
                expected = reference[kind]
                if actual[0][0] != expected[0] or abs(actual[0][1] - expected[1]) > 1e-12:
                    raise AssertionError(f"SQL row mismatch: {kind}")
            else:
                for actual_row, expected_row in zip(actual, reference["top_5"]):
                    if actual_row[0] != expected_row[0] or abs(actual_row[1] - expected_row[1]) > 1e-12:
                        raise AssertionError("Top five SQL rows not supported by actual source")
            observed["independent_python_reference_verified"] = True
            validated.append(observed)
        try:
            answer(conn, "Delete the FX table")
        except ValueError:
            refusal_passed = True
        else:
            raise AssertionError("Non-whitelisted dangerous question was not refused")
    finally:
        conn.close()
    report = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "publisher": "Bank of Canada Valet public official foreign exchange series",
        "source_url": SOURCE,
        "original_api_body_sha256": digest,
        "original_2025_real_observation_count": len(originals),
        "first_original_date": originals[0][0],
        "last_original_date": originals[-1][0],
        "sqlite_original_table": "fx_usdcad",
        "query_system": "Deterministic, bounded natural-language to read-only SQLite SELECT",
        "actual_grounded_questions": len(validated),
        "verified_against_independent_python": len(validated),
        "unrecognized_question_refused": refusal_passed,
        "records": validated,
        "limitations": (
            "These are 7 explicitly supported English question templates, "
            "not a trained natural-language-to-SQL model or differentiable RAG. "
            "No arbitrary LLM-generated SQL, no test of broad NL generalization."
        ),
    }
    (output / "verified_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/real_fx_sql")
    evaluate(parser.parse_args().output)
