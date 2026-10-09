from diffragsql.retriever import TFIDFRetriever
from diffragsql.metrics import exact_match, f1_score
from diffragsql.to_latex import export_metrics
import json
from pathlib import Path
import yaml


def test_retriever_ranks_relevant_document():
    docs = ["interest rate bank funding", "cat mouse pet animal", "payment card fraud alert"]
    retriever = TFIDFRetriever(docs)
    hits = retriever.search("fraud card", k=1)
    assert hits[0]["doc"] == docs[2]


def test_em_and_f1():
    assert exact_match("The bank", ["bank"]) == 1.0
    assert f1_score("bank fraud alert", ["fraud alert"]) > 0.7


def test_export(tmp_path):
    results = tmp_path / "metrics.json"
    dest = tmp_path / "report.tex"
    config = tmp_path / "config.yaml"
    results.write_text(json.dumps({"EM": 0.75, "F1_score": 0.81}))
    config.write_text(yaml.safe_dump({"outputs": {"results_json": str(results), "latex_table": str(dest)}}))
    assert export_metrics(config) == dest
    out = dest.read_text()
    assert "0.7500" in out and "F1\\_score" in out
