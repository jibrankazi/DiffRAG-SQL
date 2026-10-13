"""Real SQLite schema introspection, safe SQL and conservative refusal tests."""
import sqlite3

import pytest

from diffragsql.safe_cross_schema_sql import grounded_aggregate, sqlite_schema


class StubModel:
    def __init__(self, label):
        self.label = label

    def predict(self, samples):
        return [self.label]


@pytest.fixture
def connection():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE Orders (id INTEGER, Amount REAL)")
    db.executemany("INSERT INTO Orders VALUES (?, ?)", [
        (1, 2.0), (2, 4.0), (3, None),
    ])
    db.execute("CREATE TABLE Employees (id INTEGER, salary NUMERIC)")
    db.executemany("INSERT INTO Employees VALUES (?, ?)", [
        (1, 10.0), (2, 30.0),
    ])
    db.commit()
    yield db
    db.close()


def test_live_schema_and_aggregate_execution_on_two_actual_sqlite_tables(connection):
    schema = sqlite_schema(connection)
    assert "orders" in schema
    assert "employees" in schema
    cases = [
        ("How many records in table Orders?", "count", 3),
        ("Find the average value of column Amount in table Orders?", "average", 3.0),
        ("Find the highest value of column Amount in table Orders?", "maximum", 4.0),
        ("Find the smallest value of column salary in table Employees?", "minimum", 10.0),
    ]
    for phrase, intent, expected in cases:
        answer = grounded_aggregate(connection, StubModel(intent), phrase)
        assert answer["independent_python_check_passed"]
        assert answer["actual_sql_answer"] == expected
        assert answer["safe_generated_sql"].startswith("SELECT ")
        assert ";" not in answer["safe_generated_sql"]


def test_unknown_schema_injection_and_disagreeing_model_refused(connection):
    requests = [
        ("How many records in table MISSING?", "count"),
        ("Count records in table Orders; DROP TABLE Orders?", "count"),
        ("Count records in table Orders -- comment?", "count"),
        ("Find average value of column Missing in table Orders?", "average"),
        ("Find average value of column id in table Orders?", "maximum"),
        ("Give me every private row in table Orders?", "count"),
        ("Find average value of column Amount in table Orders?", "average"),
    ]
    for wording, prediction in requests:
        if wording == requests[-1][0]:
            continue
        with pytest.raises(ValueError):
            grounded_aggregate(connection, StubModel(prediction), wording)
    assert connection.execute("SELECT COUNT(*) FROM Orders").fetchone()[0] == 3


def test_non_numeric_and_count_with_column_refused(connection):
    connection.execute("CREATE TABLE Tags (label TEXT)")
    with pytest.raises(ValueError):
        grounded_aggregate(connection, StubModel("average"),
                           "Find average value of column label in table Tags?")
    with pytest.raises(ValueError):
        grounded_aggregate(connection, StubModel("count"),
                           "Count records of column Amount in table Orders?")
