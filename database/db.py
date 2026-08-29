"""Small SQLite store for web-app prediction history."""

import os
import sqlite3
from typing import Optional


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def _connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                real_probability REAL NOT NULL,
                fake_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                gradcam_filename TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def create_prediction(record: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                original_filename, stored_filename, predicted_label,
                real_probability, fake_probability, confidence,
                gradcam_filename, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["original_filename"],
                record["stored_filename"],
                record["predicted_label"],
                record["real_probability"],
                record["fake_probability"],
                record["confidence"],
                record.get("gradcam_filename"),
                record["created_at"],
            ),
        )
        return int(cursor.lastrowid)


def get_prediction(record_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[sqlite3.Row]:
    with _connect(db_path) as connection:
        return connection.execute(
            "SELECT * FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()


def list_predictions(db_path: str = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    with _connect(db_path) as connection:
        return connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()
