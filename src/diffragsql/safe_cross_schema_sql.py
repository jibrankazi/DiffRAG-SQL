"""Read-only learned-intent, schema-grounded aggregate SQL across SQLite databases.

Uses existing supervised four-intent model as an *agreement check*, a
strictly bounded question grammar, and dynamic live SQLite table/column
inspection to generate a single safe aggregate SELECT. This is NOT arbitrary
text-to-SQL, general schema linking, multi-table joins or differentiable RAG.
The Chinook database used in CI is a published SAMPLE DB, not actual business
activity; the Bank of Canada source contains actual published measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.request import Request, urlopen

from diffragsql.real_fx_sql import build_original_sqlite, fetch_actual_observations
from diffragsql.trained_fx_rag_sql import model, predict

CHINOOK_COMMIT = "7f67772503d71ba90f19283c38e93923addb43fa"
CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    + CHINOOK_COMMIT + "/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
QUESTION = re.compile(
    r"^(?P<intent>[A-Za-z][A-Za-z ]{5,130}?)(?: of column (?P<column>[A-Za-z][A-Za-z0-9_]*))?"
    r" in table (?P<table>[A-Za-z][A-Za-z0-9_]*)\?$",
    re.I,
)
VALID_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
INTENT_MARKERS = {
    "count": r"\b(?:count|how many|number of)\b",
    "average": r"\b(?:average|mean)\b",
    "maximum": r"\b(?:maximum|max|highest|largest|peak)\b",
    "minimum": r"\b(?:minimum|min|lowest|smallest|bottom)\b",
}
AGGREGATES = {"count": "COUNT", "average": "AVG", "maximum": "MAX", "minimum": "MIN"}


def sqlite_schema(db: sqlite3.Connection) -> dict[str, dict[str, str]]:
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    result = {}
    for (table,) in tables:
        if not VALID_IDENTIFIER.fullmatch(table):
            continue
        columns = db.execute(f'PRAGMA table_info("{table}")').fetchall()
        result[table.lower()] = {
            str(column[1]).lower(): str(column[2]).upper()
            for column in columns if VALID_IDENTIFIER.fullmatch(str(column[1]))
        }
    if not result:
        raise ValueError("No supported real SQLite table schema exists")
    return result


def grounded_aggregate(db: sqlite3.Connection, estimator, question: str) -> dict:
    if "--" in question or ";" in question or "/*" in question:
        raise ValueError("Comments and multiple SQL statements are refused")
    match = QUESTION.fullmatch(question.strip())
    if not match:
        raise ValueError("Unsupported bounded read-only aggregate question")
    phrase = match.group("intent")
    intents = [label for label, pattern in INTENT_MARKERS.items()
               if re.search(pattern, phrase, re.I)]
    if len(intents) != 1:
        raise ValueError("Exactly one supported aggregate intent is required")
    explicit_intent = intents[0]
    learned_intent = predict(estimator, phrase)
    if learned_intent != explicit_intent:
        raise ValueError("Trained intent model disagrees; refusing unsafe guess")
    schema = sqlite_schema(db)
    table = match.group("table").lower()
    if table not in schema:
        raise ValueError("Table is not in the actual SQLite schema")
    column = match.group("column")
    if explicit_intent == "count":
        if column:
            raise ValueError("COUNT(*) is supported without a column only")
        select = "COUNT(*)"
        raw_values = db.execute(f'SELECT 1 FROM "{table}"').fetchall()
        verified = len(raw_values)
    else:
        if not column:
            raise ValueError("Numeric aggregate requires an explicit schema column")
        name = column.lower()
        if name not in schema[table]:
            raise ValueError("Column does not exist in selected SQLite table")
        type_name = schema[table][name]
        if not any(token in type_name for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
            raise ValueError("Non-numeric column not eligible for aggregation")
        select = f'{AGGREGATES[explicit_intent]}("{name}")'
        values = [row[0] for row in db.execute(
            f'SELECT "{name}" FROM "{table}"'
        ).fetchall() if row[0] is not None]
        if not values:
            raise ValueError("No actual numeric observations found")
        verified = {
            "average": lambda: sum(values) / len(values),
            "maximum": lambda: max(values),
            "minimum": lambda: min(values),
        }[explicit_intent]()
    sql = f'SELECT {select} FROM "{table}"'
    answer = db.execute(sql).fetchone()[0]
    if answer is None or not math.isclose(float(answer), float(verified),
                                           abs_tol=1e-8, rel_tol=1e-12):
        raise AssertionError("SQL answer differs from independent Python aggregation")
    return {
        "question": question,
        "trained_intent": learned_intent,
        "explicit_user_intent": explicit_intent,
        "schema_confirmed_table": table,
        "schema_confirmed_column": column.lower() if column else None,
        "safe_generated_sql": sql,
        "actual_sql_answer": answer,
        "independent_python_check_passed": True,
        "multi_table_or_arbitrary_sql_supported": False,
    }


CASES = {
    "published_chinook_sample": [
        "How many records in table Track?",
        "What is the average value of column UnitPrice in table Track?",
        "Find the largest value of column Total in table Invoice?",
        "Find the smallest value of column UnitPrice in table Track?",
    ],
    "original_bank_of_canada": [
        "How many observations in table fx_usdcad?",
        "Find the average rate of column cad_per_usd in table fx_usdcad?",
        "Find the largest rate of column cad_per_usd in table fx_usdcad?",
        "Find the lowest rate of column cad_per_usd in table fx_usdcad?",
    ],
}


def _download_chinook(target: Path) -> str:
    request = Request(CHINOOK_URL, headers={"User-Agent": "DiffRAG-SQL/1.0"})
    with urlopen(request, timeout=60) as response:
        data = response.read(1_500_001)
    if not 100_000 < len(data) < 1_500_000 or not data.startswith(b"SQLite format 3\0"):
        raise ValueError("Pinned Chinook public SQLite sample has unexpected contents")
    target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def run(output: str = "runs/cross_schema_aggregate_sql") -> dict:
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    chinook_file = folder / "Chinook_Sqlite.sqlite"
    source_sha = _download_chinook(chinook_file)
    observations, original_digest = fetch_actual_observations()
    fx_db = build_original_sqlite(folder / "original_fx.sqlite3", observations)
    chinook_db = sqlite3.connect(
        f"file:{chinook_file.resolve().as_posix()}?mode=ro", uri=True,
    )
    classifier = model()
    traces = []
    try:
        for source_name, db in [
            ("published_chinook_sample", chinook_db),
            ("original_bank_of_canada", fx_db),
        ]:
            db.execute("PRAGMA query_only = ON")
            for question in CASES[source_name]:
                try:
                    record = grounded_aggregate(db, classifier, question)
                except ValueError as err:
                    traces.append({
                        "source": source_name, "question": question,
                        "outcome": "refused", "reason": str(err),
                    })
                else:
                    record.update({"source": source_name, "outcome": "verified"})
                    traces.append(record)
    finally:
        chinook_db.close()
        fx_db.close()
    verified = sum(item["outcome"] == "verified" for item in traces)
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            "published_chinook_sample_sqlite": source_sha,
            "original_bank_of_canada_json": original_digest,
        },
        "source_urls": {
            "chinook_published_sample": CHINOOK_URL,
            "bank_of_canada": "https://www.bankofcanada.ca/valet/docs/",
        },
        "source_types": {
            "chinook": "Published fictional/sample relational database; NOT real business transactions",
            "bank_of_canada": "Authentic publisher FX observations for 2025",
        },
        "distinct_live_sqlite_schemas_tested": 2,
        "attempted_questions": len(traces),
        "verified_sql_answers": verified,
        "refused_unsafe_or_model_disagreement": len(traces) - verified,
        "all_executed_sql_independently_checked": True,
        "trained_intent_model": "32 authored FX training phrases, no Chinook fine-tuning",
        "schema_introspection": "Live SQLite sqlite_master + PRAGMA table_info",
        "bounded_dynamic_sql_generation": True,
        "arbitrary_sql_or_general_nl_to_sql_proven": False,
        "traces": traces,
    }
    (folder / "evidence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/cross_schema_aggregate_sql")
    args = parser.parse_args()
    run(args.output)
