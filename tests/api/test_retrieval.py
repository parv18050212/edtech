import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import psycopg2
from dotenv import load_dotenv

from api.retrieval import search_chunks
from ingestion.embed import embed_query

load_dotenv()


def _connect():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def test_search_chunks_returns_only_the_requested_chapter():
    conn = _connect()
    try:
        query_embedding = embed_query("What is a star?")
        results = search_chunks(
            conn,
            query_embedding,
            subject="Environmental Studies",
            class_="5",
            chapter_number=1,
            top_k=5,
        )
        assert len(results) == 5
        for r in results:
            assert r["chunk_id"].startswith("MSB_EVS5_CH01_")
    finally:
        conn.close()


def test_search_chunks_excludes_other_chapters():
    conn = _connect()
    try:
        query_embedding = embed_query("What is a star?")
        results = search_chunks(
            conn,
            query_embedding,
            subject="Environmental Studies",
            class_="5",
            chapter_number=2,  # "Motions of the Earth", not the Stars chapter
            top_k=5,
        )
        for r in results:
            assert r["chunk_id"].startswith("MSB_EVS5_CH02_")
    finally:
        conn.close()


def test_search_chunks_orders_by_similarity_descending():
    conn = _connect()
    try:
        query_embedding = embed_query("What is a star?")
        results = search_chunks(
            conn,
            query_embedding,
            subject="Environmental Studies",
            class_="5",
            chapter_number=1,
            top_k=5,
        )
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    finally:
        conn.close()
