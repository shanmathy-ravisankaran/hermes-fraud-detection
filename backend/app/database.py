import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "hermes.db"


def get_connection():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_amount REAL NOT NULL,
                transaction_dt INTEGER NOT NULL,
                product_cd TEXT NOT NULL,
                card4 TEXT,
                card6 TEXT,
                email_domain TEXT,
                fraud_score REAL NOT NULL,
                threshold REAL NOT NULL,
                is_suspicious INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                analyst_notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS investigation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                investigation_id INTEGER NOT NULL,
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                previous_notes TEXT,
                updated_notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )
        """)

        connection.commit()


def save_investigation(transaction, prediction):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO investigations (
                transaction_amount,
                transaction_dt,
                product_cd,
                card4,
                card6,
                email_domain,
                fraud_score,
                threshold,
                is_suspicious,
                model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.transaction_amount,
                transaction.transaction_dt,
                transaction.product_cd,
                transaction.card4,
                transaction.card6,
                transaction.email_domain,
                prediction.fraud_score,
                prediction.threshold,
                int(prediction.is_suspicious),
                prediction.model_name,
            ),
        )

        connection.commit()

        return cursor.lastrowid


def list_investigations(limit=100, investigation_status=None):
    with get_connection() as connection:
        if investigation_status is None:
            rows = connection.execute(
                """
                SELECT *
                FROM investigations
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM investigations
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (investigation_status, limit),
            ).fetchall()

        return [dict(row) for row in rows]


def get_investigation(investigation_id: int):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM investigations
            WHERE id = ?
            """,
            (investigation_id,),
        ).fetchone()

        return dict(row) if row else None


def update_investigation(
    investigation_id: int,
    investigation_status: str,
    analyst_notes=None,
    notes_provided=False,
):
    with get_connection() as connection:
        current = connection.execute(
            """
            SELECT *
            FROM investigations
            WHERE id = ?
            """,
            (investigation_id,),
        ).fetchone()

        if current is None:
            return None

        previous_notes = current["analyst_notes"]
        updated_notes = analyst_notes if notes_provided else previous_notes

        connection.execute(
            """
            UPDATE investigations
            SET status = ?, analyst_notes = ?
            WHERE id = ?
            """,
            (investigation_status, updated_notes, investigation_id),
        )

        if (
            current["status"] != investigation_status
            or previous_notes != updated_notes
        ):
            connection.execute(
                """
                INSERT INTO investigation_history (
                    investigation_id,
                    previous_status,
                    new_status,
                    previous_notes,
                    updated_notes
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    investigation_id,
                    current["status"],
                    investigation_status,
                    previous_notes,
                    updated_notes,
                ),
            )

        updated = connection.execute(
            """
            SELECT *
            FROM investigations
            WHERE id = ?
            """,
            (investigation_id,),
        ).fetchone()

        connection.commit()
        return dict(updated)


def list_investigation_history(investigation_id: int):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM investigation_history
            WHERE investigation_id = ?
            ORDER BY id ASC
            """,
            (investigation_id,),
        ).fetchall()

        return [dict(row) for row in rows]
