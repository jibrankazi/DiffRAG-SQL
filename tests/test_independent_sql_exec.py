"""Local fixtures validate FK path compilation; AWS source runs are separate."""
from pathlib import Path
import sqlite3
import pytest

from diffragsql.independent_sql_exec import (
    generate, mentioned_tables, normalize, read_only, rowbag, structure,
)


@pytest.fixture
def music(tmp_path: Path):
    path = tmp_path / "fixture.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
    CREATE TABLE Album (
      AlbumId INTEGER PRIMARY KEY, Title TEXT,
      ArtistId INTEGER REFERENCES Artist(ArtistId));
    CREATE TABLE Track (
      TrackId INTEGER PRIMARY KEY, AlbumId INTEGER REFERENCES Album(AlbumId),
      Price REAL);
    CREATE TABLE Invoice (
      InvoiceId INTEGER PRIMARY KEY, Total REAL, InvoiceDate TEXT);
    INSERT INTO Artist VALUES (1,'AC/DC'), (2,'Metallica');
    INSERT INTO Album VALUES (1,'High Voltage',1),(2,'Back in Black',1),(3,'Ride the Lightning',2);
    INSERT INTO Track VALUES (1,1,1.1),(2,2,2.3),(3,3,4.5);
    INSERT INTO Invoice VALUES (1,2.0,'2021-06-02'), (2,4.0,'2022-02-01');
    """)
    conn.commit()
    conn.close()
    with read_only(path) as db:
        yield db


def test_schema_adapts_to_arbitrary_identifiers_and_fk_paths(music):
    tables, edges = structure(music)
    assert len(tables) == 4
    assert len(edges["Artist"]) == 1
    result = generate(
        music,
        "Count the number of albums in the Album table associated with the artist named 'AC/DC' in the Artist table.",
    )
    assert result["rows"] == [(2,)]
    assert result["join_count"] == 1
    assert "JOIN" in result["sql"]
    assert "AC/DC" not in result["sql"]
    assert result["bound_parameters"] == ["AC/DC"]
    result = generate(
        music, "Count number of tracks in Track table linked to Artist table")
    assert result["join_count"] == 2
    assert result["rows"] == [(3,)]


def test_compiles_aggregate_and_calendar_on_unseen_schema(music):
    out = generate(music, "Calculate sum of Total from Invoice table")
    assert out["rows"] == [(6.0,)]
    avg = generate(music, "Find average Price in Track table")
    assert avg["rows"][0][0] == pytest.approx((1.1+2.3+4.5)/3)
    year = generate(music, "Count Invoice rows in Invoice table in 2021")
    assert year["rows"] == [(1,)]
    assert year["bound_parameters"] == ["2021"]


def test_fails_closed_on_unsupported_or_dangerous_sql(music):
    for phrase in (
        "", "DELETE FROM Artist;", "How many Albums; DROP TABLE Artist?",
        "Count the rows in secret table", "Count albums with unknown 'NOT_REAL'",
        "Please find first five albums", "Find the maximum Price in Track",
    ):
        with pytest.raises(ValueError):
            generate(music, phrase)
    assert music.execute("SELECT COUNT(*) FROM Artist").fetchone()[0] == 2


def test_rowbag_keeps_column_count_multiplicity_and_numeric_values():
    assert rowbag([(1, 'A'), (2, 'B')]) == rowbag([(2.0, 'B'), (1, 'A')])
    assert rowbag([(1,), (1,)]) != rowbag([(1,)])
    assert rowbag([(1, "A")]) != rowbag([("A", 1)])
    assert normalize("categories") == "category"
