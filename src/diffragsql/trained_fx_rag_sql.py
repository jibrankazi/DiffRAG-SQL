"""Supervised, reproducible *domain-limited* RAG-to-SQL experiment.

Unlike prior fixed-question mappings, a scikit-learn estimator is fit to
authored training paraphrases and evaluated on non-overlapping held-out
wordings. Schema retrieval grounds the SQLite table; SQL is always selected
from audited templates, never generated or run from arbitrary model text.
This does not prove cross-schema generalization or differentiable RAG.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import argparse
import json
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

from diffragsql.real_fx_date_windows import OPERATORS
from diffragsql.real_fx_schema_rag import source_schema_evidence
from diffragsql.real_fx_sql import build_original_sqlite, fetch_actual_observations

TRAIN = {
    "count": [
        "How many daily observations exist", "Count the published daily exchange rates",
        "Number of available exchange rate dates", "Total number of rate observations",
        "How many FX values were recorded", "Count all recorded trading dates",
        "Give the observation count", "Tell me how many daily rate samples",
    ],
    "average": [
        "Find the average foreign exchange rate", "Compute the mean CAD per USD",
        "What is the average daily rate", "Give the arithmetic mean exchange rate",
        "Calculate the mean FX value", "Average all observations",
        "Tell me the mean exchange value", "What is the average rate value",
    ],
    "maximum": [
        "Find the highest USD CAD rate", "What was the maximum observed rate",
        "Give the peak exchange value", "Find the largest FX quote",
        "What was the highest daily exchange rate", "Compute the maximum FX observation",
        "Find the largest exchange rate", "Identify the peak CAD per USD value",
    ],
    "minimum": [
        "Find the lowest USD CAD rate", "What was the minimum observed rate",
        "Give the trough exchange value", "Find the smallest FX quote",
        "What was the lowest daily exchange rate", "Compute the minimum FX observation",
        "Find the smallest exchange rate", "Identify the bottom CAD per USD value",
    ],
}
HELDOUT = {
    "count": [
        "How many recorded days do we have", "Give me the count of observed quotes",
        "What's the total number of FX records", "Number of official rates collected",
    ],
    "average": [
        "What's the mean exchange quotation", "Calculate average of recorded rates",
        "Summarize the arithmetic average of FX values", "Which rate is the period mean",
    ],
    "maximum": [
        "Show the largest recorded quote", "Which day had the highest rate",
        "Compute the peak value from FX records", "Give the maximum observed FX value",
    ],
    "minimum": [
        "Show the smallest recorded quote", "Which day had the lowest rate",
        "Compute the bottom value from FX records", "Give the minimum observed FX value",
    ],
}
DATE_RX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SQL_BY_OPERATOR = {op: sql for sql, op in OPERATORS.values()}
LABELS = sorted(TRAIN)


def model():
    examples = [(sentence, kind) for kind, texts in TRAIN.items() for sentence in texts]
    estimator = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True),
        LogisticRegression(max_iter=1500, random_state=42),
    )
    estimator.fit([sample for sample, _ in examples], [kind for _, kind in examples])
    return estimator


def predict(estimator, wording):
    return str(estimator.predict([wording])[0])


def benchmark(estimator):
    train = {s.strip().casefold() for texts in TRAIN.values() for s in texts}
    held = [(s, kind) for kind, texts in HELDOUT.items() for s in texts]
    assert not any(s.strip().casefold() in train for s, _ in held)
    true = [kind for _, kind in held]
    predicted = [predict(estimator, sentence) for sentence, _ in held]
    return {
        "model": "TfidfVectorizer(char_wb 2-5) + fitted LogisticRegression",
        "training_paraphrases": sum(len(x) for x in TRAIN.values()),
        "heldout_paraphrases": len(held),
        "heldout_wording_overlap": 0,
        "heldout_accuracy": accuracy_score(true, predicted),
        "labels": LABELS,
        "confusion_matrix_actual_rows_predicted_columns":
            confusion_matrix(true, predicted, labels=LABELS).tolist(),
        "predictions": [{"prompt": s, "expected": expected, "predicted": got,
                         "correct": expected == got}
                        for (s, expected), got in zip(held, predicted)],
        "independent_public_benchmark": False,
        "cross_database_generalization_tested": False,
    }


def grounded_answer(connection, estimator, prompt: str):
    schema = source_schema_evidence(connection)
    dates = DATE_RX.findall(prompt)
    if len(dates) != 2 or not re.fullmatch(r"[\w\s?.,:/-]+", prompt):
        raise ValueError("Only bounded two-date read-only exchange-rate questions are accepted")
    if not ("usd" in prompt.casefold() and "cad" in prompt.casefold()):
        raise ValueError("Question must explicitly identify USD and CAD")
    try:
        first, last = (date.fromisoformat(d) for d in dates)
    except ValueError as exc:
        raise ValueError("Invalid calendar date") from exc
    if not date(2025, 1, 1) <= first <= last <= date(2025, 12, 31):
        raise ValueError("Date bounds must lie within the verified 2025 source")
    # Only text before the first date affects supervised intent inference.
    wording = prompt.split(dates[0])[0].strip(" :-")
    if len(wording) < 6:
        raise ValueError("Question provides too little intent evidence")
    operation = predict(estimator, wording)
    sql = SQL_BY_OPERATOR[operation]
    if not sql.startswith("SELECT ") or sql.count("?") != 2:
        raise AssertionError("Internal SQL allowlist contract failed")
    params = (first.isoformat(), last.isoformat())
    row = connection.execute(sql, params).fetchone()
    if row is None or row[0] is None:
        raise ValueError("No official observation inside requested range")
    return {
        "operator": operation, "executed_sql": sql, "bound_parameters": list(params),
        "actual_sql_answer": row[0], "schema_evidence": schema,
        "intent_model_is_trained": True,
        "sql_is_fixed_reviewed_template": True,
        "general_sql_generation": False,
    }


def run(output="runs/trained_fx_rag_sql"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    estimator = model()
    metrics = benchmark(estimator)
    originals, digest = fetch_actual_observations()
    db = build_original_sqlite(output / "original_source_rates.sqlite3", originals)
    traces = []
    try:
        questions = [
            ("How many recorded days do we have for USD to CAD from 2025-01-01 to 2025-03-31?", "count"),
            ("Calculate average of recorded rates for USD to CAD from 2025-04-01 to 2025-06-30?", "average"),
            ("Show the largest recorded quote for USD to CAD from 2025-07-01 to 2025-09-30?", "maximum"),
            ("Show the smallest recorded quote for USD to CAD from 2025-10-01 to 2025-12-31?", "minimum"),
        ]
        for prompt, intended in questions:
            result = grounded_answer(db, estimator, prompt)
            left, right = result["bound_parameters"]
            values = [v for day, v in originals if left <= day <= right]
            reference = {"count": lambda: len(values),
                         "average": lambda: sum(values) / len(values),
                         "maximum": lambda: max(values),
                         "minimum": lambda: min(values)}[intended]()
            result["independent_python_match"] = (
                result["operator"] == intended and
                abs(result["actual_sql_answer"] - reference) <= 1e-9
            )
            if not result["independent_python_match"]:
                raise AssertionError("Held-out SQL operator or real-source answer mismatched")
            result["question"] = prompt
            traces.append(result)
    finally:
        db.close()
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "publisher": "Bank of Canada Valet",
        "public_source_body_sha256": digest,
        "real_observations": len(originals),
        "benchmark": metrics,
        "original_source_sql_cases": len(traces),
        "all_real_source_sql_cross_checks_passed": all(x["independent_python_match"] for x in traces),
        "traces": traces,
        "research_limit": "Small manually authored held-out English intent benchmark over ONE 2025 FX schema. Not Spider/BIRD, arbitrary SQL generation, differentiable retrieval, or cross-schema generalization.",
    }
    (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="runs/trained_fx_rag_sql")
    run(ap.parse_args().output)
