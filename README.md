# DiffRAG-SQL — verified real-data SQL grounding and extractive QA research

The repository currently contains **two separately executed research baselines**. The original project name describes a longer-term differentiable retrieval plus SQL research goal. **There is not yet a trained differentiable retrieval/SQL model or integrated RAG→SQL agent.** Do not attribute such a system or fabricated paper results to these executions.

## 1. Working genuine public data → SQLite → question-to-SQL → independently checked answers

**[Actual successful GitHub Actions execution, October 9, 2026](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/37969551978)**

The new `src/diffragsql/real_fx_sql.py` uses the original publicly available [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs/), no credentials, to retrieve its official `FXUSDCAD` exchange-rate records for 2025. It verifies the source responses, builds an actual read-only SQLite database and maps seven explicitly supported natural-language questions to SQLite `SELECT` statements, executing every query over the original rows. It checks each result independently against direct Python calculations rather than just checking that the SQL parses.

The executed run produced:

| Verified artifact/observation | Actual result |
| --- | --- |
| Original Bank of Canada `FXUSDCAD` 2025 published dates | **249** |
| First / last observed dates | **2025-01-02 / 2025-12-31** |
| Queries actually executed, each independently cross-checked | **7 / 7 passed** |
| Sample real result: mean Canadian dollars per USD in those dates | **1.39776225** |
| Unrecognized/unapproved question | **Refused**, not silently mapped to dangerous SQL |
| Saved reproducibility artifacts | Original-row SQLite database + raw-source SHA-256 + SQL traces / results JSON |

The seven supported questions calculate observation count, annual average, maximum, minimum, first available rate, last available rate and highest-five-rate dates. All seven SQL strings and their actual supported row answers are saved in `runs/real_fx_sql/verified_results.json`. The database is saved as `runs/real_fx_sql/real_fx_2025.sqlite3` and uploaded to the successful workflow artifact.

**Run locally:**

```bash
# Show all seven supported genuine-source questions; no API call required:
PYTHONPATH=src python -m diffragsql.real_fx_sql --list-questions

# Download genuine official data, build and query SQLite, verify and save traces:
PYTHONPATH=src python -m diffragsql.real_fx_sql --question "What was the average official USD to CAD rate for 2025?"
```

No external Python dependencies are needed for this component. It uses the built-in `sqlite3`, `urllib` and JSON libraries. **This is deterministic whitelisted text-to-SQL**, not a neural question-parsing model. Seven predefined templates are not evidence of general text-to-SQL ability, differentiability, SQL safety under arbitrary models, or live production reliability.

## 2. Real Stanford SQuAD extractive QA (separate component)

[Successful real SQuAD workflow](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/37935263617) downloaded genuine Stanford SQuAD v1.1 *development* questions and ran TF-IDF retrieval plus the pretrained DistilBERT extractive question answering reader. The independent evaluation of **16** selected questions / **16** distinct passages found:

| Measure | Observed |
| --- | ---: |
| Source passage among top 3 retrieved | 16 / 16 |
| Exact answer match | 11 / 16 |
| Token-level F1 | 0.747 |

These are small, selected and unusually easy retrieval conditions, **not a general-purpose RAG or SQL benchmark**. They do not involve training or executing database queries and do not have the same source observations as the Bank of Canada experiment.

**Run the existing reader after installing `requirements.txt` and the package:**

```bash
pip install -r requirements.txt
pip install -e .
python -m diffragsql.real_squad_experiment --count 16
```

## Still unimplemented

- Joint differentiable retriever–reader gradient propagation and any trained SQL-query policy.
- Validated general-domain natural-language-to-SQL performance on a held-out set such as Spider/BIRD.
- Combining the SQuAD reader and SQL engine into a single learned, multi-source SQL-grounded RAG system.
- Verified research figures beyond the above actual experiment records.

**Project status:** working independently tested **real Bank of Canada SQLite/SQL execution** and **separate pretrained extractive QA**, not a complete differentiable SQL-RAG architecture. Earlier README draft references to synthetic relational training, peer-reviewed proof, 71.4% EM, 78.6 F1, 0.91 faithfulness and automatically synchronized manuscript outcomes were **unverified proposals**, not measured or reproducible findings.
