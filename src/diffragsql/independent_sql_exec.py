"""Foreign-key-aware, read-only SQLite SQL generator and third-party benchmark.

Supports explicitly named tables, COUNT/SUM/AVG, foreign-key joins, bounded
equality literals, grouping and year filters. Unsupported requests fail closed.
Published AWS gold SQL is NEVER accessed by the generator.
"""
from __future__ import annotations
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.request import Request, urlopen

from diffragsql.trained_fx_rag_sql import model, predict

COMMIT = "9823ae6fe0a83951294dd68aa3d74003e0b480d1"
ROOT = f"https://raw.githubusercontent.com/aws-samples/natural-language-queries-for-datalakes/{COMMIT}/src/"
SOURCES = {
    "Chinook": ("example_data/datalake/data/Chinook.db", "tests/test_sets/Chinook_tests.json"),
    "northwind": ("example_data/datalake/data/northwind.db", "tests/test_sets/northwind_tests.json"),
    "mysports": ("example_data/datalake/data/mysports.db", "tests/test_sets/mysports_tests.json"),
}
LITERAL = re.compile(r"['\"]([^'\"]{1,100})['\"]")
YEAR = re.compile(r"\b(?:in|during|year of)\s+(19\d{2}|20\d{2})\b", re.I)
NUMERIC = ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")
MAX_ROWS = 15000


def mentions_column(question: str, col: str) -> bool:
    """Link camel-case, snake-case and naturally spaced schema column names."""
    split = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", col).replace("_", " ")
    question_norm = normalize(question)
    for phrase in (normalize(col), normalize(split)):
        if phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", question_norm):
            return True
    return False


def primary_id(table: str, columns: dict[str, str]) -> str | None:
    """Conservative key naming fallback for table-level grouping."""
    expected = normalize(table).replace(" ", "") + "id"
    candidates = [col for col in columns if normalize(col).replace(" ", "") == expected]
    return candidates[0] if len(candidates) == 1 else None


def display_column(columns: dict[str, str]) -> str | None:
    """Find a unique common human-readable label column without inventing names."""
    names = [col for col in columns if col.casefold() in (
        "name", "title", "productname", "companyname", "teamname",
        "categoryname", "artistname", "playlistname", "sportname",
    )]
    return names[0] if len(names) == 1 else None


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize(s: str) -> str:
    def sing(t: str) -> str:
        if len(t) > 4 and t.endswith("ies"):
            return t[:-3] + "y"
        if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return t[:-1]
        return t
    return " ".join(sing(t) for t in re.findall(r"[A-Za-z0-9]+", s.lower()))


def structure(db: sqlite3.Connection):
    tables = {}
    edges = {}
    for (name,) in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ):
        tables[name] = {r[1]: (r[2] or "").upper()
                        for r in db.execute(f"PRAGMA table_info({quote(name)})")}
    for name in tables:
        edges[name] = []
        for row in db.execute(f"PRAGMA foreign_key_list({quote(name)})"):
            dest, col, target = row[2], row[3], row[4]
            if dest in tables and col in tables[name] and target in tables[dest]:
                edges[name].append((dest, col, target))
    for name in tables:
        for dest, col, target in list(edges[name]):
            edges[dest].append((name, target, col))
    return tables, edges


def mentioned_tables(question: str, tables: dict) -> list[str]:
    q = normalize(question)
    matches = []
    for name in tables:
        target = normalize(name)
        if len(target) < 3:
            continue
        m = re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", q)
        if m:
            matches.append((m.start(), name))
    return [t for _, t in sorted(matches)][:5]


def shortest_path(root: str, target: str, graph: dict) -> list[tuple] | None:
    queue = deque([(root, [])])
    visited = {root}
    while queue:
        item, path = queue.popleft()
        if item == target:
            return path
        if len(path) >= 3:
            continue
        for dest, src_col, dst_col in graph[item]:
            if dest not in visited:
                visited.add(dest)
                queue.append((dest, path + [(item, dest, src_col, dst_col)]))
    return None


def existing_literal(db, tables, included, value):
    matches = []
    for name in included:
        for col, kind in tables[name].items():
            if any(x in kind for x in ("INT", "REAL", "NUM", "DEC", "BLOB")):
                continue
            try:
                found = db.execute(
                    f"SELECT 1 FROM {quote(name)} WHERE {quote(col)} = ? LIMIT 1",
                    (value,),
                ).fetchone()
            except sqlite3.Error:
                continue
            if found:
                priority = int("name" in col.lower() or "country" in col.lower()
                               or "title" in col.lower())
                matches.append((priority, name, col))
    matches.sort(reverse=True)
    if not matches or (len(matches) > 1 and matches[0][0] == matches[1][0]):
        raise ValueError("Ambiguous or absent literal in named tables")
    return matches[0][1:]


def generate(db: sqlite3.Connection, question: str, estimator=None) -> dict:
    if len(question) > 1500 or not question.strip() or any(
        x in question for x in (";", "--", "/*", "*/")
    ):
        raise ValueError("Malformed/unsafe input")
    tables, graph = structure(db)
    named = mentioned_tables(question, tables)
    if not named:
        raise ValueError("Question does not identify a source table")
    first = named[0]
    included = [first]
    joins = []
    for name in named[1:]:
        if name in included:
            continue
        path = shortest_path(first, name, graph)
        if path is None:
            raise ValueError("No declared FK join path between referenced tables")
        for src, dest, c1, c2 in path:
            if dest not in included:
                joins.append((src, dest, c1, c2))
                included.append(dest)
    q = question.casefold()
    detected = {
        "count": bool(re.search(r"\b(?:count|how many|number of)\b", q)),
        "average": bool(re.search(r"\b(?:average|mean)\b", q)),
        "sum": bool(re.search(r"\b(?:sum of|total amount|total size)\b", q)),
    }
    intents = [key for key, flag in detected.items() if flag]
    if len(intents) > 1:
        raise ValueError("Multiple competing operations")
    if intents:
        operation = intents[0]
    elif re.search(r"\b(?:most|highest|largest|top|earliest|oldest|lowest|smallest)\b", q):
        operation = "extremum"
    else:
        raise ValueError("Question outside supported query patterns")
    top_count = (
        bool(re.search(r"\b(?:most|highest|largest)\b", q))
        and not detected["average"] and not detected["sum"]
        and len(included) > 1
        and bool(re.search(r"\b(?:has|associated|number of|count of)\b", q))
    )
    if top_count:
        operation = "top_count"
    def ref(table, col):
        return f"{quote(table)}.{quote(col)}"
    from_part = " FROM " + quote(first)
    for src, dst, left, right in joins:
        from_part += f" JOIN {quote(dst)} ON {ref(src, left)} = {ref(dst, right)}"
    where = []
    args = []
    for value in LITERAL.findall(question):
        name, col = existing_literal(db, tables, included, value)
        where.append(f"{ref(name, col)} = ?")
        args.append(value)
    year = YEAR.search(question)
    if year:
        possible = [(table, col) for table in included
                    for col in tables[table] if "date" in col.lower()]
        if len(possible) != 1:
            raise ValueError("Cannot resolve date column uniquely")
        tab, col = possible[0]
        where.append(f"strftime('%Y', {ref(tab, col)}) = ?")
        args.append(year.group(1))
    group_part = ""
    order_part = ""
    if operation == "count":
        expression = "COUNT(*)"
        if re.search(r"\b(each|per|grouped by)\b", q):
            cols = [(table, col) for table in included for col in tables[table]
                    if mentions_column(question, col) and not col.lower().endswith("id")]
            if len(cols) != 1:
                raise ValueError("Grouping column not unique")
            table, col = cols[0]
            expression = ref(table, col) + ", COUNT(*)"
            group_part = f" GROUP BY {ref(table, col)}"
            if "descending" in q:
                order_part = " ORDER BY COUNT(*) DESC"
    elif operation == "top_count":
        if not joins:
            raise ValueError("Top group count requires a declared relationship")
        group_key = primary_id(first, tables[first])
        display = display_column(tables[first])
        if group_key is None or display is None:
            raise ValueError("Unambiguous grouped entity and label not identified")
        expression = f"{ref(first, display)}, COUNT(*)"
        group_part = f" GROUP BY {ref(first, group_key)}"
        order_part = " ORDER BY COUNT(*) DESC LIMIT 1"
    elif operation in ("average", "sum"):
        matched = [(tab, col) for tab in included for col, typ in tables[tab].items()
                   if any(word in typ for word in NUMERIC)
                   and mentions_column(question, col)]
        if len(matched) != 1:
            raise ValueError("Aggregate needs exactly one named numeric column")
        tab, col = matched[0]
        expression = ("AVG" if operation == "average" else "SUM") + f"({ref(tab, col)})"
    elif operation == "extremum":
        # Order-by / extremum is compiled from explicit schema column evidence.
        # Unsupported computed measures, multiple keys and implicit arithmetic
        # are refused rather than returning a convenient unrelated ranking.
        named_columns = [
            (tab, col) for tab in included for col in tables[tab]
            if mentions_column(question, col)
        ]
        if len(named_columns) != 1:
            raise ValueError("Extremum requires exactly one explicit sort column")
        tab, col = named_columns[0]
        date_sort = "DATE" in col.upper()
        descending = bool(re.search(
            r"\b(?:most|highest|largest|latest|greatest|top|newest)\b", q))
        ascending = bool(re.search(
            r"\b(?:earliest|oldest|lowest|smallest)\b", q))
        if descending == ascending:
            raise ValueError("Ambiguous ordering direction")
        is_projection = bool(re.search(r"\b(?:who|which|name|first and last)\b", q))
        if is_projection:
            labels = []
            if "FirstName" in tables[first] and "LastName" in tables[first]:
                labels = [ref(first, "FirstName"), ref(first, "LastName")]
            else:
                candidate = display_column(tables[first])
                if candidate:
                    labels = [ref(first, candidate)]
            if not labels:
                raise ValueError("Cannot project a clearly identified label")
            expression = ", ".join(labels)
            order_part = f" ORDER BY {ref(tab, col)} {'DESC' if descending else 'ASC'} LIMIT 1"
        else:
            if date_sort:
                raise ValueError("Date extremum needs an explicit output entity")
            if not any(x in tables[tab][col] for x in NUMERIC):
                raise ValueError("Non-numeric minimum or maximum without projection")
            expression = f"{'MAX' if descending else 'MIN'}({ref(tab, col)})"
    else:
        raise ValueError("Unimplemented structural intent")
    sql = "SELECT " + expression + from_part
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += group_part + order_part
    result = db.execute(sql, args).fetchmany(MAX_ROWS + 1)
    if len(result) > MAX_ROWS:
        raise ValueError("Too many rows")
    return {
        "sql": sql, "bound_parameters": args, "rows": result,
        "selected_tables": included, "join_count": len(joins),
        "intent": operation,
        "prior_trained_model_prediction": predict(estimator, question) if estimator else None,
    }


def fetch(relative: str, destination: Path, limit: int) -> str:
    request = Request(ROOT + relative, headers={
        "User-Agent": "DiffRAG-SQL-independent-execution/1.0",
        "Accept-Encoding": "identity",
    })
    with urlopen(request, timeout=90) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Pinned source exceeds expected bound")
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_only(path: Path):
    conn = sqlite3.connect("file:" + path.resolve().as_posix() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def gold_execute(db, sql: str, timeout=2.5):
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        raise ValueError("External reference SQL must be SELECT/WITH")
    deadline = time.monotonic() + timeout
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 3000)
    try:
        rows = db.execute(sql).fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            raise ValueError("Gold query result too large")
        return rows
    finally:
        db.set_progress_handler(None, 0)


def rowbag(rows):
    def atom(v):
        if isinstance(v, (float, int)) and not isinstance(v, bool):
            return ("number", round(float(v), 6))
        if v is None:
            return ("null", "")
        return ("text", str(v))
    return Counter(tuple(atom(v) for v in row) for row in rows)


def run(folder="runs/aws_independent_exec", max_items=None):
    output = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    estimator = model()  # Prior 32 manually authored FX phrases, no AWS training.
    reports = []
    source_sha = {}
    for dataset, (db_path, cases_path) in SOURCES.items():
        db_target = output / (dataset + ".sqlite3")
        test_target = output / (dataset + "_reference.json")
        source_sha[db_path] = fetch(db_path, db_target, 30_000_000)
        source_sha[cases_path] = fetch(cases_path, test_target, 200_000)
        if db_target.read_bytes()[:16] != b"SQLite format 3\x00":
            raise ValueError("Original source does not contain SQLite bytes")
        tasks = json.loads(test_target.read_text())
        if len(tasks) < 30:
            raise ValueError("Incomplete source benchmark")
        if max_items:
            tasks = tasks[:max_items]
        with read_only(db_target) as connection:
            for item in tasks:
                row = {"database": dataset, "id": item["QUESTION_NO"],
                       "question": item["QUESTION"]}
                try:
                    expected = gold_execute(connection, item["ANSWER_SQL"])
                except (ValueError, sqlite3.Error) as error:
                    row.update({"gold_executable": False, "gold_error": str(error)[:150]})
                    reports.append(row)
                    continue
                row["gold_executable"] = True
                row["gold_row_count"] = len(expected)
                for variant, prompt in (
                    ("original_question", item["QUESTION"]),
                    ("publisher_disambiguated_question", item.get("DISAMBIGUATED_QUESTION") or ""),
                ):
                    try:
                        answer = generate(connection, prompt, estimator=estimator)
                        row[variant] = {
                            "status": ("correct" if rowbag(answer["rows"]) == rowbag(expected)
                                       else "incorrect"),
                            "generated_sql": answer["sql"],
                            "parameters": answer["bound_parameters"],
                            "joined_tables": answer["selected_tables"],
                            "join_count": answer["join_count"],
                            "trained_intent_diagnostic": answer["prior_trained_model_prediction"],
                        }
                    except (ValueError, sqlite3.Error) as error:
                        row[variant] = {"status": "refused", "reason": str(error)[:150]}
                reports.append(row)
    eligible = [r for r in reports if r["gold_executable"]]
    summary = {}
    for variant in ("original_question", "publisher_disambiguated_question"):
        counts = Counter(r[variant]["status"] for r in eligible)
        summary[variant] = {
            "correct": counts["correct"], "incorrect": counts["incorrect"],
            "refused": counts["refused"], "evaluated_gold_executable": len(eligible),
            "accuracy_with_refusals": counts["correct"] / len(eligible) if eligible else 0,
            "correct_queries_with_generated_joins": sum(
                r[variant]["status"] == "correct"
                and r[variant].get("join_count", 0) > 0 for r in eligible
            ),
        }
    evidence = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "third_party_source": "AWS Samples natural-language-queries-for-datalakes",
        "upstream_commit": COMMIT,
        "source_sha256": source_sha,
        "external_gold_attempted": len(reports),
        "external_gold_executable": len(eligible),
        "external_gold_not_executable": len(reports) - len(eligible),
        "scoring": "Exact multiset of full SQLite result rows, numeric values rounded to 6 digits",
        "training_gold_leakage": False,
        "prior_training_only_fx_32_phrases": True,
        "evaluation_variants": summary,
        "schema_aware_fk_join_generation": True,
        "unrestricted_general_sql_generation_proven": False,
        "limitations": "Only bounded aggregation and FK joins, literal equality/group/year; does not implement arbitrary NL->SQL. Benchmark-owned disambiguated prompts are reported separately and are not zero-shot raw queries.",
        "cases": reports,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "cases"}, indent=2))
    return evidence


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/aws_independent_exec")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()
    run(args.output, args.max_items)
