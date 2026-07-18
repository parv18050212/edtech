import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from ingestion.db import INSERT_SQL, INSERT_TEMPLATE, row_from_record

load_dotenv()


def load_file(conn, path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        rows = [row_from_record(json.loads(line)) for line in f]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, INSERT_SQL, rows, template=INSERT_TEMPLATE
        )
    conn.commit()
    return len(rows)


def main():
    database_url = os.environ["DATABASE_URL"]
    processed_dir = Path("data/processed")
    files = sorted(processed_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit("No JSONL files found in data/processed/")

    conn = psycopg2.connect(database_url)
    try:
        total = 0
        for path in files:
            n = load_file(conn, path)
            total += n
            print(f"{path.name}: loaded {n} rows")
        print(f"Total: {total} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
