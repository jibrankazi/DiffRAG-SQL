## New independent public SQL execution evaluation — October 13, 2026

**Actual passing GitHub Actions workflows:**
[Original public-database execution baseline](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/38331293970) and
[expanded foreign-key joins and grouped-ranking implementation](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/38331488243).
The generator, workflow and tests are on `main`.

### Execution benchmark (independently published gold SQL)

The executable, schema-aware bounded SQL compiler in
[`src/diffragsql/independent_sql_exec.py`](src/diffragsql/independent_sql_exec.py)
downloads three source-pinned SQLite databases and their separate natural-language
question/gold-SQL suites from AWS Samples' `natural-language-queries-for-datalakes`
project: Chinook (55 questions), Northwind (57), MySports (58), **170 published
questions** in total. The inputs are pinned to external commit
`9823ae6fe0a83951294dd68aa3d74003e0b480d1` and hashed in the artifact.
Only executable, read-only gold results are scored; each generated answer must
match **all columns and full result-row multiplicity**, with numerical cells
rounded to six decimals. Refusals count as failures in the denominator.

| Measured outcome | Initial run | Expanded generator run |
| --- | ---: | ---: |
| Public questions attempted | 170 | 170 |
| Independently executable gold references | 168 | 167 |
| Original, unclarified questions answered correctly | **9** | **12** |
| Original-question execution accuracy | 5.36% | **7.19%** |
| Correct original-question queries involving generated FK joins | 2 | **4** |
| Publisher-disambiguated questions answered correctly | 9 | **14** |
| Disambiguated-question execution accuracy | 5.36% | **8.38%** |
| Correct publisher-disambiguated queries with joins | 3 | **7** |

The gold-reference denominator differed slightly between runs because the
read-only reference scorer has a query timeout and a few upstream gold queries
are incompatible with/timed out on the bundled databases. The exact reasons
and all per-question SQL, successes, refusals and source hashes are preserved
in the workflow artifacts. **These accuracy numbers are low**, not evidence of
full NL→SQL generalization, and are **not** directly comparable to structure
classification accuracy.

The compiler discovers SQLite tables/columns and declared FK paths at runtime,
allows up to three edges per join path, generates safe identifier-quoted SELECTs,
uses SQLite-bound parameters for quoted filter literals, and supports restricted
COUNT/SUM/AVG, numeric extrema, first/last-style projections, grouped top counts
and year/group filters. Unsupported ambiguity and SQL-injection syntax are
rejected. This is a **rule-constrained, schema-aware SQL generator**.
The prior 32-example supervised four-intent classifier is called as a
*diagnostic only*: it **does not** train or control the SQL decoder. The
benchmark uses the original question and the third-party author's clarified
question **as separate conditions**; clarified questions are not evidence of
raw-query accuracy.

### Proper train-only cross-schema learning validation

The separately trained public Defog [structure prediction benchmark](
https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/38331672006)
was extended with leave-one-training-database-out selection over word, character,
and hybrid TF-IDF representations. It selected the **original word-based
representation** using the seven training databases without using the four test
databases or their SQL to choose hyperparameters. On the unchanged 40 independent
test questions it still measured **micro-F1 0.6724**, **macro-F1 0.6552**, and
**full six-tag match 2/40 (5%)**. The cross-validation did **not** improve the
unseen-database score, and the structure predictor remains separate from
generation. The honest outcome is that feature tuning alone did not solve the
generalization gap.

```sh
python -m pip install numpy scipy scikit-learn pytest
PYTHONPATH=src python -m pytest -q tests/test_independent_sql_exec.py tests/test_public_cross_schema_benchmark.py
PYTHONPATH=src python -m diffragsql.independent_sql_exec
PYTHONPATH=src python -m diffragsql.public_cross_schema_benchmark
```

**Open research requirements:** pretrained or more extensively trained
schema-conditioned sequence-to-SQL generation; unrestricted projections and
complex joins/subqueries/window functions; alignment of unstructured raw
questions to unseen columns/tables; external full Spider/BIRD execution
accuracy; hard-set adversarial testing; and substantial independent
accuracy improvements. Do not describe this version as a complete,
production-ready, generalizable, differentiable RAG-to-SQL system.

---

## Verified expanded public evaluation — October 13, 2026

Two additional GitHub Actions experiments have now succeeded:

1. **[Independent Defog SQL-Eval, unseen-database evaluation](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/38324968571):** fit a TF-IDF + one-vs-rest logistic-regression model to **190 independently published** NL questions from **7 training databases**. Test on **40 independently published** questions from **4 entirely different databases** (broker, car_dealership, derm_treatment, ewallet), withholding their SQL until scoring. Six SQL-structure labels (join, where, group-by, order-by, limit, aggregate): trained **micro-F1 0.6724**, **macro-F1 0.6552**, **complete six-label match 2/40 (5%)**. The training-label majority baseline scored micro-F1 **0.2567**, macro-F1 **0.125**, complete-label match **0/40**. These figures describe SQL **structure prediction, not SQL execution accuracy or full generated SQL**. The frozen public upstream revision and both SHA-256 hashes, each evaluation question, predictions, retrieved training precedent, confusion-style per-label precision/recall/F1 and baseline scores are saved in the workflow artifact. Gold SQL is used only for scoring. Alternative gold SQL strings can contribute clause presence; this is a deliberately coarse structural measure.

2. **[Real-source two-schema bounded SQL execution](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/38325203663):** reuse the existing **32-phrase-trained** four-operation classifier without Chinook fine-tuning; require agreement with an explicit single aggregate intent; read real SQLite schema using `sqlite_master` and `PRAGMA table_info`, verify the specified table and numeric column, construct a single read-only aggregate `SELECT`, execute it and independently recompute the answer from source rows. Tests used the **real original Bank of Canada 2025 observations** and **Chinook's published *fictional sample* relational database**, not real business activity. **8/8 narrowly worded demonstration queries verified** (4 per schema), zero rejected in this selected demonstration. The sample database and original FX-source SQLite, SHA-256 source digests, concrete SQL queries, and answers are in the CI artifact. Additional local tests verify malformed input, unknown tables/columns, model disagreement and SQL injection refusal.

Run manually (the workflows download the original external sources):

```sh
python -m pip install numpy scipy scikit-learn pytest
PYTHONPATH=src python -m diffragsql.public_cross_schema_benchmark
PYTHONPATH=src python -m diffragsql.safe_cross_schema_sql
```

**Research assessment:** real public cross-database structure evaluation and working *bounded* aggregate SQL now exist. Nevertheless, a **5% complete structure-label match on unseen databases** is insufficient evidence of generalized NL→SQL competence. The bounded eight-question SQL demonstration is hand-selected; no genuine Spider/BIRD execution-accuracy benchmark, unrestricted joins, new SQL constructions, cross-database semantic entity linking, jointly trained differentiable RAG, or deployable unrestricted SQL agent exists. Treat those as **remaining research work**, not completed features.

---

## New executed trained SQL-intent baseline — October 12, 2026

A [successful authenticated public-source CI
run](https://github.com/jibrankazi/DiffRAG-SQL/actions/runs/38324028295)
now fits a supervised TF-IDF character-n-gram + logistic-regression
classifier to **32 authored training paraphrases**, and evaluates against
**16 nonidentical held-out paraphrases** across four supported operations:
count, average, maximum, minimum. Measured held-out accuracy was
**16/16 (100%)** on that small, authored, single-domain evaluation.
This is not an independent Spider/BIRD result and should not be used to
suggest broad generalizability.

The new [trained_fx_rag_sql.py](src/diffragsql/trained_fx_rag_sql.py)
combines model-fitted intent prediction with actual SQLite schema verification
and audited, parameter-bound SQL; the workflow retrieved original 2025 Bank
of Canada FXUSDCAD rate observations and independently cross-checked four
executed SQL answers against Python calculations from the original rows.
CI artifacts preserve the source download digest, confusion matrix, fitted
model evaluation, SQL traces and original-source SQLite table.

```sh
python -m pip install numpy scikit-learn
PYTHONPATH=src python -m diffragsql.trained_fx_rag_sql
```

**Limitations:** No jointly trained differentiable retriever, no open-ended
SQL generation, and no evaluation on independent databases, Spider/BIRD, or
SQL-join schema transfer. This is a genuinely **trained, measured restricted
SQL-intent baseline**, not a complete generalizable RAG-to-SQL system.

---

# DiffRAG-SQL — verified real-data SQL grounding and extractive QA research

The repository contains **multiple separately executed research baselines**, including the new domain-limited trained intent classifier. The original project name describes a longer-term differentiable retrieval plus SQL research goal. **There is not yet a trained differentiable retrieval/SQL model or integrated RAG→SQL agent.** Do not attribute such a system or fabricated paper results to these executions.

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


## New: safe calendar-bounded SQL over real Bank of Canada 2025 records

The separately runnable [date-range SQL component](src/diffragsql/real_fx_date_windows.py) now supports four whitelisted English operator patterns (count, average, minimum and maximum), with **any valid start/end date inside 2025**. Dates are validated, checked for correct order and bound as SQLite `?` parameters. Unrecognized English, arbitrary SQL, invalid dates, reversed windows and other years are refused. The original seven historical SQL templates have **not changed**.

Run:

```bash
PYTHONPATH=src python -m diffragsql.real_fx_date_windows --list-operators
PYTHONPATH=src python -m diffragsql.real_fx_date_windows --question "What was the average official USD to CAD rate between 2025-04-01 and 2025-06-30?"
```

The [real-source GitHub Actions workflow](https://github.com/jibrankazi/DiffRAG-SQL/actions/workflows/real-fx-date-windows.yml) independently downloads Bank of Canada Valet source observations, builds fresh query-only SQLite, validates four distinct quarter-year queries against original Python computations, independently reruns the resulting parameterized queries, and uploads the original database, source digest and actual answer traces. Local unit tests use synthetic **software fixtures only**. This remains a deterministic text-to-SQL baseline rather than an LLM, a differentiable SQL model or a complete RAG system.

## Schema-retrieved Bank of Canada SQL (evidence-constrained integration)

The new src/diffragsql/real_fx_schema_rag.py joins the existing TF-IDF
retriever with the official Bank of Canada original-source SQLite pipeline.
It inspects the actual database schema, retrieves a supporting operation
note, requires agreement with the vetted question parser, binds user date
values in read-only SQL, and independently calculates answers from the actual
downloaded original rows. The four distinct test prompts are reproduced by
the schema-rag-sql.yml GitHub Actions pipeline with a downloadable artifact.

This is a **retrieval-plus-SQL integration**, not a trained model. Schema
notes are authored interpretations and TF-IDF is not differentiable.
General natural-language-to-SQL, joint retriever training, multi-source
reasoning and broad benchmark claims remain unimplemented.

Run: PYTHONPATH=src python -m diffragsql.real_fx_schema_rag
