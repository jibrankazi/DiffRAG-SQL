"""Source rows are local software fixtures, not historical exchange-rate proof."""
from pathlib import Path
import pytest

from diffragsql.real_fx_sql import build_original_sqlite
from diffragsql.trained_fx_rag_sql import benchmark, grounded_answer, model


def test_fitted_supervised_model_has_disjoint_heldout_set():
    metrics = benchmark(model())
    assert metrics["training_paraphrases"] == 32
    assert metrics["heldout_paraphrases"] == 16
    assert metrics["heldout_wording_overlap"] == 0
    assert 0 <= metrics["heldout_accuracy"] <= 1
    assert not metrics["cross_database_generalization_tested"]


def test_trained_schema_grounded_parameterized_sql(tmp_path: Path):
    db = build_original_sqlite(tmp_path / "sample.sqlite3", [
        ("2025-01-02", 1.4), ("2025-01-03", 1.5),
        ("2025-01-06", 1.3), ("2025-02-01", 1.2)])
    classifier = model()
    try:
        answer = grounded_answer(db, classifier,
            "Count the published daily exchange rates for USD to CAD from 2025-01-01 to 2025-01-31?")
        assert answer["operator"] == "count"
        assert answer["actual_sql_answer"] == 3
        assert answer["sql_is_fixed_reviewed_template"]
        assert answer["schema_evidence"]["actual_database_series"] == "FXUSDCAD"
    finally:
        db.close()


def test_injection_bad_dates_and_unsupported_series_refused(tmp_path: Path):
    db = build_original_sqlite(tmp_path / "sample.sqlite3", [("2025-01-02", 1.4)])
    classifier = model()
    try:
        for prompt in (
            "Delete from fx_usdcad; DROP TABLE x",
            "Average USD CAD from 2025-01-01 to 2025-01-31?; DELETE TABLE",
            "Average USD CAD from 2025-01-01 to 2025-01-31 -- comment",
            "Average USD CAD from 2025-13-01 to 2025-01-31",
            "Average EUR GBP from 2025-01-01 to 2025-01-31",
            "Average USD CAD from 2024-01-01 to 2025-01-31",
        ):
            with pytest.raises(ValueError):
                grounded_answer(db, classifier, prompt)
        assert db.execute("SELECT COUNT(*) FROM fx_usdcad").fetchone()[0] == 1
    finally:
        db.close()
