"""Independent, database-disjoint SQL *structure* evaluation, not SQL generation.

Source: Defog sql-eval's human-assembled question/gold SQL collections.
Train on seven unrelated DBs from questions_gen_sqlite.csv and test on four
different DBs in instruct_basic_sqlite.csv. Original gold SQL is used ONLY for
scoring. The estimator predicts six structural SQL properties of a question.
It cannot produce arbitrary executable SQL, identify schema entities or
replace evaluation against live databases.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

UPSTREAM_COMMIT = "b83332416ea2d424b0cd12994a0a9ba100f4b4af"
BASE = f"https://raw.githubusercontent.com/defog-ai/sql-eval/{UPSTREAM_COMMIT}/data"
TRAIN_FILE = "questions_gen_sqlite.csv"
EVAL_FILE = "instruct_basic_sqlite.csv"
TRAIN_DATABASES = frozenset({
    "academic", "advising", "atis", "geography", "restaurants", "scholar", "yelp"
})
EVAL_DATABASES = frozenset({
    "broker", "car_dealership", "derm_treatment", "ewallet"
})
FEATURES = ("join", "where", "group_by", "order_by", "limit", "aggregate")
PATTERNS = (
    r"\\bjoin\\b",
    r"\\bwhere\\b",
    r"\\bgroup\\s+by\\b",
    r"\\border\\s+by\\b",
    r"\\blimit\\b",
    r"\\b(?:count|sum|avg|min|max)\\s*\\(",
)
# The gold-query feature extractor measures clauses, not query equivalence.
COMPILED = tuple(re.compile(pattern, re.I) for pattern in PATTERNS)


def gold_features(sql: str) -> list[int]:
    if not isinstance(sql, str) or not sql.lstrip().lower().startswith(("select", "with")):
        raise ValueError("Expected published SELECT/WITH gold SQL")
    return [int(bool(pattern.search(sql))) for pattern in COMPILED]


def parse_csv(text: str, *, source: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if not {"db_name", "db_type", "query", "question"}.issubset(reader.fieldnames or []):
        raise ValueError(f"Unexpected {source} columns")
    rows = []
    for row in reader:
        if row["db_type"] != "sqlite" or not row["db_name"] or not row["question"]:
            raise ValueError(f"Invalid {source} record")
        if not row["query"] or not row["query"].lstrip().lower().startswith(("select", "with")):
            raise ValueError(f"Invalid gold query in {source}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty {source}")
    return rows


def select_disjoint(
    train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if TRAIN_DATABASES & EVAL_DATABASES:
        raise AssertionError("Train/evaluation databases must be disjoint")
    train = [row for row in train_rows if row["db_name"] in TRAIN_DATABASES]
    test = [row for row in eval_rows if row["db_name"] in EVAL_DATABASES]
    train_db = {row["db_name"] for row in train}
    test_db = {row["db_name"] for row in test}
    if train_db != TRAIN_DATABASES or test_db != EVAL_DATABASES:
        raise ValueError("A pinned public-source database is absent")
    if len(train) < 100 or len(test) < 20:
        raise ValueError("Incomplete pinned public benchmark records")
    train_questions = {row["question"].casefold().strip() for row in train}
    if len(train_questions) != len(train):
        raise ValueError("Duplicate training wordings")
    if any(row["question"].casefold().strip() in train_questions for row in test):
        raise ValueError("Question wording leakage into unseen-schema evaluation")
    return train, test


def measures(true: list[list[int]], pred: list[list[int]]) -> dict:
    return {
        "micro_f1": float(f1_score(true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "exact_structure_match_rate": sum(
            actual == guessed for actual, guessed in zip(true, pred)
        ) / len(true),
        "by_feature": {
            name: {
                "support": sum(row[i] for row in true),
                "precision": float(precision_score(
                    [row[i] for row in true], [row[i] for row in pred],
                    zero_division=0,
                )),
                "recall": float(recall_score(
                    [row[i] for row in true], [row[i] for row in pred],
                    zero_division=0,
                )),
                "f1": float(f1_score(
                    [row[i] for row in true], [row[i] for row in pred],
                    zero_division=0,
                )),
            }
            for i, name in enumerate(FEATURES)
        },
    }


def evaluate(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> dict:
    train, test = select_disjoint(train_rows, eval_rows)
    x_train = [row["question"] for row in train]
    x_test = [row["question"] for row in test]
    y_train = [gold_features(row["query"]) for row in train]
    y_test = [gold_features(row["query"]) for row in test]
    # Only source-domain train examples are fitted. The evaluation databases
    # and SQL answers never enter the model or its vocabulary.
    estimator = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), strip_accents="unicode",
            sublinear_tf=True, lowercase=True,
        ),
        OneVsRestClassifier(LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42,
        )),
    )
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test).astype(int).tolist()

    # Legitimate reference: six majority-class predictions learned from
    # training labels, not a tuned or schema-aware SQL generator.
    baseline = []
    for i in range(len(FEATURES)):
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit([[0]] * len(y_train), [y[i] for y in y_train])
        baseline.append([int(z) for z in dummy.predict([[0]] * len(test))])
    naive_predictions = [list(row) for row in zip(*baseline)]

    # Retrieval evidence: nearest *training* wording and its source database.
    vectorizer = estimator.named_steps["tfidfvectorizer"]
    train_vectors = vectorizer.transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    similarity = (test_vectors @ train_vectors.T).tocsr()
    results = []
    for j, row in enumerate(test):
        dense = similarity.getrow(j).toarray()[0]
        best = int(dense.argmax())
        results.append({
            "test_database": row["db_name"],
            "question": row["question"],
            "gold_sql_structures": dict(zip(FEATURES, y_test[j])),
            "predicted_sql_structures": dict(zip(FEATURES, predictions[j])),
            "exact_structure_match": y_test[j] == predictions[j],
            "retrieved_train_example": {
                "database": train[best]["db_name"],
                "question": train[best]["question"],
                "similarity": float(dense[best]),
            },
        })
    return {
        "benchmark_name": "Public Defog SQL-Eval unseen-database SQL-structure probe",
        "upstream_repository": "https://github.com/defog-ai/sql-eval",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_files": [TRAIN_FILE, EVAL_FILE],
        "train_databases": sorted(TRAIN_DATABASES),
        "evaluation_databases": sorted(EVAL_DATABASES),
        "train_examples": len(train),
        "evaluation_examples": len(test),
        "feature_names": list(FEATURES),
        "model": "Supervised TF-IDF 1/2-grams + one-vs-rest logistic regression",
        "training_only_nearest_question_retrieval": True,
        "trained_model_scores": measures(y_test, predictions),
        "majority_training_label_baseline_scores": measures(y_test, naive_predictions),
        "eval_gold_sql_is_used_only_to_score": True,
        "model_generated_executable_sql": False,
        "execution_accuracy_measured": False,
        "general_cross_schema_sql_generation_proven": False,
        "example_results": results,
        "limitations": [
            "Six SQL-structure tags are predicted, NOT complete SQL programs.",
            "Unseen database names test partial transfer; actual test schemas are not passed to the model.",
            "Published text and gold SQL are independently authored, but underlying data is not executed.",
            "No Spider/BIRD exact-match, execution accuracy or generalizable SQL generation is claimed.",
        ],
    }


def fetch_file(name: str) -> tuple[list[dict[str, str]], str]:
    if name not in {TRAIN_FILE, EVAL_FILE}:
        raise ValueError("Only pinned public evaluation sources are accepted")
    request = Request(BASE + "/" + name, headers={"User-Agent": "DiffRAG-SQL-public-eval/1.0"})
    with urlopen(request, timeout=50) as response:
        content = response.read(2_000_001)
    if len(content) > 2_000_000:
        raise ValueError("Public source exceeds the expected size")
    return parse_csv(content.decode("utf-8-sig"), source=name), hashlib.sha256(content).hexdigest()


def run(output: str = "runs/defog_cross_schema") -> dict:
    train, train_sha = fetch_file(TRAIN_FILE)
    test, test_sha = fetch_file(EVAL_FILE)
    report = evaluate(train, test)
    report["source_sha256"] = {TRAIN_FILE: train_sha, EVAL_FILE: test_sha}
    report["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "evidence.json").write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "example_results"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/defog_cross_schema")
    args = parser.parse_args()
    run(args.output)
