"""Offline leakage, label extraction and cross-schema benchmark invariants."""
import csv
import io

import pytest

from diffragsql.public_cross_schema_benchmark import (
    EVAL_DATABASES, FEATURES, TRAIN_DATABASES,
    evaluate, gold_features, parse_csv, select_disjoint,
)


def sample(db: str, i: int, sql: str) -> dict[str, str]:
    return {
        "db_name": db, "db_type": "sqlite", "question": f"Question {db} {i}",
        "query": sql,
    }


def test_gold_sql_structure_labels_and_rejects_mutations():
    tags = gold_features(
        "SELECT AVG(price) FROM purchases JOIN stores ON purchases.id=stores.id "
        "WHERE units > 0 GROUP BY store ORDER BY price LIMIT 3"
    )
    assert dict(zip(FEATURES, tags)) == {
        "join": 1, "where": 1, "group_by": 1, "order_by": 1,
        "limit": 1, "aggregate": 1,
    }
    assert gold_features("SELECT name FROM books") == [0] * len(FEATURES)
    with pytest.raises(ValueError):
        gold_features("DELETE FROM books")


def test_csv_parsing_preserves_quoted_comma_and_rejects_invalid_gold():
    content = "db_name,db_type,query,question\n" + (
        'academic,sqlite,"SELECT a, b FROM people","List all people"\n'
    )
    parsed = parse_csv(content.replace("\n", "\n"), source="sample")
    assert parsed[0]["query"] == "SELECT a, b FROM people"
    with pytest.raises(ValueError):
        parse_csv("wrong,header\na,b\n".replace("\n", "\n"), source="bad")


def test_enforce_database_partition_and_no_wording_leakage():
    training = [sample(db, i, "SELECT COUNT(*) FROM a")
                for db in TRAIN_DATABASES for i in range(20)]
    evaluation = [sample(db, i, "SELECT x FROM b")
                  for db in EVAL_DATABASES for i in range(6)]
    selected_train, selected_test = select_disjoint(training, evaluation)
    assert len(selected_train) == len(TRAIN_DATABASES) * 20
    assert len(selected_test) == len(EVAL_DATABASES) * 6
    assert {r["db_name"] for r in selected_train}.isdisjoint(
        {r["db_name"] for r in selected_test}
    )
    evaluation[0]["question"] = training[0]["question"]
    with pytest.raises(ValueError, match="leakage"):
        select_disjoint(training, evaluation)


def test_real_estimator_on_small_disjoint_schemas_has_no_execution_claim():
    patterns = [
        ("Count objects in table", "SELECT COUNT(*) FROM a"),
        ("Find objects by tag", "SELECT a FROM a WHERE tag='yes'"),
        ("Show sorted objects", "SELECT a FROM a ORDER BY a"),
        ("Show joined objects", "SELECT a FROM a JOIN b ON a.id=b.id"),
    ]
    train = [sample(db, j * 20 + i, sql)
             for db in TRAIN_DATABASES
             for j in range(5)
             for i, (_, sql) in enumerate(patterns)]
    for row in train:
        row["question"] = patterns[row["question"].split()[-1] and 0][0] + " " + row["question"]
    test = [sample(db, i, patterns[i % len(patterns)][1])
            for db in EVAL_DATABASES for i in range(6)]
    report = evaluate(train, test)
    assert report["train_examples"] == 140
    assert report["evaluation_examples"] == 24
    assert not report["model_generated_executable_sql"]
    assert not report["execution_accuracy_measured"]
    assert len(report["example_results"]) == 24
    assert report["selected_train_only_feature_style"] in ("word", "char", "hybrid")
    assert report["evaluation_data_used_for_model_selection"] is False
    variants = report["train_database_leave_one_out_candidate_scores"]
    assert len(variants) == 3
    assert all(len(v["folds"]) == len(TRAIN_DATABASES) for v in variants)
    assert all(all(f["all_fitted_rows_exclude_heldout_database"] for f in v["folds"]) for v in variants)
    assert set(report["train_databases"]).isdisjoint(set(report["evaluation_databases"]))
