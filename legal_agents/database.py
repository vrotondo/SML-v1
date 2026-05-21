import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
from .config import DB_PATH


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _row(row) -> Optional[Dict]:
    return dict(row) if row else None


def _rows(rows) -> List[Dict]:
    return [dict(r) for r in rows]


def init_db():
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS clients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        email       TEXT,
        phone       TEXT,
        address     TEXT,
        notes       TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS cases (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id       INTEGER NOT NULL,
        case_number     TEXT UNIQUE,
        case_type       TEXT,
        description     TEXT,
        opposing_party  TEXT,
        jurisdiction    TEXT,
        status          TEXT DEFAULT 'intake',
        created_at      TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (client_id) REFERENCES clients(id)
    );

    CREATE TABLE IF NOT EXISTS research_notes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id     INTEGER NOT NULL,
        title       TEXT,
        content     TEXT,
        source      TEXT,
        created_at  TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (case_id) REFERENCES cases(id)
    );

    CREATE TABLE IF NOT EXISTS case_briefs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id     INTEGER NOT NULL,
        content     TEXT,
        version     INTEGER DEFAULT 1,
        status      TEXT DEFAULT 'draft',
        created_at  TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (case_id) REFERENCES cases(id)
    );

    CREATE TABLE IF NOT EXISTS documents (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id      INTEGER NOT NULL,
        doc_type     TEXT,
        title        TEXT,
        content      TEXT,
        status       TEXT DEFAULT 'draft',
        review_notes TEXT,
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (case_id) REFERENCES cases(id)
    );

    CREATE TABLE IF NOT EXISTS reminders (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id        INTEGER NOT NULL,
        reminder_type  TEXT,
        description    TEXT,
        due_date       TEXT,
        completed      INTEGER DEFAULT 0,
        created_at     TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (case_id) REFERENCES cases(id)
    );

    CREATE TABLE IF NOT EXISTS agent_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        agent       TEXT,
        action      TEXT,
        case_id     INTEGER,
        details     TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    conn.close()


# ── Clients ──────────────────────────────────────────────────────────────────

def create_client(name: str, email: str = None, phone: str = None,
                  address: str = None, notes: str = None) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO clients (name, email, phone, address, notes) VALUES (?,?,?,?,?)",
        (name, email, phone, address, notes)
    )
    client_id = cur.lastrowid
    conn.commit()
    conn.close()
    return client_id


def get_client(client_id: int) -> Optional[Dict]:
    conn = _conn()
    row = conn.execute("SELECT * FROM clients WHERE id=?", (client_id,)).fetchone()
    conn.close()
    return _row(row)


def list_clients() -> List[Dict]:
    conn = _conn()
    rows = conn.execute("SELECT * FROM clients ORDER BY created_at DESC").fetchall()
    conn.close()
    return _rows(rows)


def search_clients(query: str) -> List[Dict]:
    like = f"%{query}%"
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM clients WHERE name LIKE ? OR email LIKE ? OR phone LIKE ?",
        (like, like, like)
    ).fetchall()
    conn.close()
    return _rows(rows)


# ── Cases ─────────────────────────────────────────────────────────────────────

def create_case(client_id: int, case_type: str, description: str,
                opposing_party: str = None, jurisdiction: str = None) -> int:
    conn = _conn()
    count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    case_number = f"CASE-{datetime.now().year}-{count + 1:04d}"
    cur = conn.execute(
        "INSERT INTO cases (client_id, case_number, case_type, description, opposing_party, jurisdiction) "
        "VALUES (?,?,?,?,?,?)",
        (client_id, case_number, case_type, description, opposing_party, jurisdiction)
    )
    case_id = cur.lastrowid
    conn.commit()
    conn.close()
    return case_id


def get_case(case_id: int) -> Optional[Dict]:
    conn = _conn()
    row = conn.execute("""
        SELECT cases.*, clients.name as client_name,
               clients.email as client_email, clients.phone as client_phone
        FROM cases
        JOIN clients ON cases.client_id = clients.id
        WHERE cases.id = ?
    """, (case_id,)).fetchone()
    conn.close()
    return _row(row)


def list_cases(status: str = None) -> List[Dict]:
    conn = _conn()
    if status:
        rows = conn.execute("""
            SELECT cases.*, clients.name as client_name
            FROM cases JOIN clients ON cases.client_id = clients.id
            WHERE cases.status = ? ORDER BY cases.created_at DESC
        """, (status,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT cases.*, clients.name as client_name
            FROM cases JOIN clients ON cases.client_id = clients.id
            ORDER BY cases.created_at DESC
        """).fetchall()
    conn.close()
    return _rows(rows)


def update_case_status(case_id: int, status: str) -> bool:
    conn = _conn()
    conn.execute("UPDATE cases SET status=? WHERE id=?", (status, case_id))
    conn.commit()
    conn.close()
    return True


# ── Research Notes ────────────────────────────────────────────────────────────

def save_research_note(case_id: int, title: str, content: str, source: str = None) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO research_notes (case_id, title, content, source) VALUES (?,?,?,?)",
        (case_id, title, content, source)
    )
    note_id = cur.lastrowid
    conn.commit()
    conn.close()
    return note_id


def get_research_notes(case_id: int) -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM research_notes WHERE case_id=? ORDER BY created_at DESC",
        (case_id,)
    ).fetchall()
    conn.close()
    return _rows(rows)


# ── Case Briefs ───────────────────────────────────────────────────────────────

def save_brief(case_id: int, content: str) -> int:
    conn = _conn()
    max_ver = conn.execute(
        "SELECT MAX(version) FROM case_briefs WHERE case_id=?", (case_id,)
    ).fetchone()[0]
    version = (max_ver or 0) + 1
    cur = conn.execute(
        "INSERT INTO case_briefs (case_id, content, version) VALUES (?,?,?)",
        (case_id, content, version)
    )
    brief_id = cur.lastrowid
    conn.commit()
    conn.close()
    return brief_id


def get_latest_brief(case_id: int) -> Optional[Dict]:
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM case_briefs WHERE case_id=? ORDER BY version DESC LIMIT 1",
        (case_id,)
    ).fetchone()
    conn.close()
    return _row(row)


# ── Documents ─────────────────────────────────────────────────────────────────

def save_document(case_id: int, doc_type: str, title: str, content: str) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO documents (case_id, doc_type, title, content) VALUES (?,?,?,?)",
        (case_id, doc_type, title, content)
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def get_document(doc_id: int) -> Optional[Dict]:
    conn = _conn()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return _row(row)


def list_documents(case_id: int) -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM documents WHERE case_id=? ORDER BY created_at DESC",
        (case_id,)
    ).fetchall()
    conn.close()
    return _rows(rows)


def update_document(doc_id: int, status: str, review_notes: str = None) -> bool:
    conn = _conn()
    conn.execute(
        "UPDATE documents SET status=?, review_notes=? WHERE id=?",
        (status, review_notes, doc_id)
    )
    conn.commit()
    conn.close()
    return True


# ── Reminders ─────────────────────────────────────────────────────────────────

def add_reminder(case_id: int, reminder_type: str, description: str, due_date: str) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO reminders (case_id, reminder_type, description, due_date) VALUES (?,?,?,?)",
        (case_id, reminder_type, description, due_date)
    )
    reminder_id = cur.lastrowid
    conn.commit()
    conn.close()
    return reminder_id


def list_upcoming_reminders(days_ahead: int = 30) -> List[Dict]:
    conn = _conn()
    rows = conn.execute("""
        SELECT reminders.*, cases.case_number, clients.name as client_name
        FROM reminders
        JOIN cases ON reminders.case_id = cases.id
        JOIN clients ON cases.client_id = clients.id
        WHERE reminders.completed = 0
          AND date(reminders.due_date) <= date('now', '+' || ? || ' days')
        ORDER BY reminders.due_date ASC
    """, (days_ahead,)).fetchall()
    conn.close()
    return _rows(rows)


def complete_reminder(reminder_id: int) -> bool:
    conn = _conn()
    conn.execute("UPDATE reminders SET completed=1 WHERE id=?", (reminder_id,))
    conn.commit()
    conn.close()
    return True


def get_case_reminders(case_id: int) -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM reminders WHERE case_id=? ORDER BY due_date ASC",
        (case_id,)
    ).fetchall()
    conn.close()
    return _rows(rows)


# ── Agent Log ─────────────────────────────────────────────────────────────────

def log_action(agent: str, action: str, case_id: int = None, details: str = None):
    conn = _conn()
    conn.execute(
        "INSERT INTO agent_log (agent, action, case_id, details) VALUES (?,?,?,?)",
        (agent, action, case_id, details)
    )
    conn.commit()
    conn.close()
