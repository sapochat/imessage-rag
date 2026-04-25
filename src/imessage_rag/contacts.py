"""Local macOS Contacts resolution for iMessage handles."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from imessage_rag.config import CONTACTS_DB, CONTACTS_ENABLED


def normalize_handle(value: str) -> str:
    """Normalize phone numbers and emails for contact matching."""
    cleaned = value.strip()
    if "@" in cleaned:
        return cleaned.casefold()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return digits or cleaned.casefold()


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _looks_like_handle(value: object) -> bool:
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    if "@" in cleaned and "." in cleaned:
        return True
    return sum(ch.isdigit() for ch in cleaned) >= 5


@dataclass
class ContactRecord:
    display_name: str
    handles: tuple[str, ...]


@dataclass
class ContactResolver:
    """In-memory map from iMessage handles to human contact names."""

    records: tuple[ContactRecord, ...] = ()
    errors: tuple[str, ...] = ()
    _handle_to_name: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _name_to_handles: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for record in self.records:
            name_key = _normalize_name(record.display_name)
            if name_key:
                self._name_to_handles.setdefault(name_key, set()).update(record.handles)
            for handle in record.handles:
                key = normalize_handle(handle)
                if key and key not in self._handle_to_name:
                    self._handle_to_name[key] = record.display_name

    @classmethod
    def empty(cls, errors: Iterable[str] = ()) -> "ContactResolver":
        return cls(records=(), errors=tuple(errors))

    @property
    def contact_count(self) -> int:
        return len(self.records)

    @property
    def handle_count(self) -> int:
        return len(self._handle_to_name)

    def label_for_handle(self, handle: str) -> str:
        """Return a display name for a handle, or the original handle if unknown."""
        return self._handle_to_name.get(normalize_handle(handle), handle)

    def label_for_participants(self, handles: Iterable[str]) -> tuple[str, ...]:
        return tuple(self.label_for_handle(handle) for handle in handles)

    def handles_for_contact(self, query: str) -> tuple[str, ...]:
        """Return handles for an exact or unique partial contact-name query."""
        query_key = _normalize_name(query)
        exact = self._name_to_handles.get(query_key)
        if exact is not None:
            return tuple(sorted(exact))

        partial_matches = [
            handles
            for name_key, handles in self._name_to_handles.items()
            if query_key and query_key in name_key
        ]
        if len(partial_matches) == 1:
            return tuple(sorted(partial_matches[0]))
        return ()


def default_contact_db_paths() -> list[Path]:
    """Return likely local macOS Contacts database paths."""
    if CONTACTS_DB is not None:
        return [CONTACTS_DB]

    root = Path.home() / "Library" / "Application Support" / "AddressBook"
    paths = [root / "AddressBook-v22.abcddb"]
    try:
        paths.extend(sorted(root.glob("Sources/*/AddressBook-v*.abcddb")))
    except OSError:
        pass
    return paths


def load_contacts(db_paths: Iterable[Path] | None = None) -> ContactResolver:
    """Load Contacts from local AddressBook SQLite DBs.

    This function never raises for permission or schema failures. It returns an
    empty resolver with sanitized errors so ingestion can proceed without
    blocking on Contacts permissions.
    """
    if not CONTACTS_ENABLED:
        return ContactResolver.empty()

    records: list[ContactRecord] = []
    errors: list[str] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for db_path in db_paths or default_contact_db_paths():
        if not db_path.exists():
            continue
        try:
            loaded = _load_contacts_from_db(db_path)
        except (OSError, sqlite3.Error) as exc:
            errors.append(f"{db_path}: {type(exc).__name__}")
            continue
        for record in loaded:
            key = (record.display_name, record.handles)
            if key not in seen:
                records.append(record)
                seen.add(key)

    return ContactResolver(records=tuple(records), errors=tuple(errors))


_DEFAULT_RESOLVER: ContactResolver | None = None


def get_default_resolver() -> ContactResolver:
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = load_contacts()
    return _DEFAULT_RESOLVER


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row[0] for row in rows}


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    return [row[1] for row in rows]


def _pick(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _load_contacts_from_db(db_path: Path) -> tuple[ContactRecord, ...]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        records: list[ContactRecord] = []
        table_names = _tables(conn)
        if "ZABCDRECORD" in table_names:
            records.extend(_load_modern_contacts(conn, "ZABCDRECORD", table_names))
        if "ABPerson" in table_names:
            records.extend(_load_legacy_contacts(conn, "ABPerson", table_names))
        return tuple(records)
    finally:
        conn.close()


def _load_people(
    conn: sqlite3.Connection,
    table: str,
    name_candidates: Iterable[str],
) -> dict[object, str]:
    columns = _columns(conn, table)
    pk_col = _pick(columns, ("Z_PK", "ROWID", "id"))
    if pk_col:
        pk_expr = _quote(pk_col)
    else:
        pk_expr = "rowid"

    selected_name_cols = [
        column for column in name_candidates if column.lower() in {c.lower() for c in columns}
    ]
    if not selected_name_cols:
        return {}

    select_cols = [f"{pk_expr} AS __pk"]
    select_cols.extend(_quote(column) for column in selected_name_cols)
    rows = conn.execute(
        f"SELECT {', '.join(select_cols)} FROM {_quote(table)}"
    ).fetchall()

    people: dict[object, str] = {}
    for row in rows:
        person_name_cols = {"ZFIRSTNAME", "ZMIDDLENAME", "ZLASTNAME", "FIRST", "MIDDLE", "LAST"}
        primary_cols = [
            column for column in selected_name_cols if column.upper() in person_name_cols
        ]
        fallback_cols = [
            column for column in selected_name_cols if column.upper() not in person_name_cols
        ]
        pieces = [
            str(row[column]).strip()
            for column in primary_cols
            if row[column] is not None and str(row[column]).strip()
        ]
        if not pieces:
            pieces = [
                str(row[column]).strip()
                for column in fallback_cols
                if row[column] is not None and str(row[column]).strip()
            ][:1]
        name = " ".join(pieces)
        if name:
            people[row["__pk"]] = name
    return people


def _records_from_owner_tables(
    conn: sqlite3.Connection,
    people: dict[object, str],
    table_names: Iterable[str],
    table_predicate,
) -> tuple[ContactRecord, ...]:
    handles_by_person: dict[object, set[str]] = {person_id: set() for person_id in people}

    for table in table_names:
        if not table_predicate(table):
            continue
        columns = _columns(conn, table)
        owner_col = _pick(columns, ("ZOWNER", "ZRECORD", "record_id", "person_id", "owner_id"))
        if owner_col is None:
            continue
        value_cols = [
            column
            for column in columns
            if any(token in column.upper() for token in ("ADDRESS", "EMAIL", "NUMBER", "VALUE"))
            and not any(token in column.upper() for token in ("LABEL", "UUID", "TYPE"))
        ]
        if not value_cols:
            continue
        selected = [_quote(owner_col), *(_quote(column) for column in value_cols)]
        rows = conn.execute(
            f"SELECT {', '.join(selected)} FROM {_quote(table)}"
        ).fetchall()
        for row in rows:
            owner = row[owner_col]
            if owner not in handles_by_person:
                continue
            for column in value_cols:
                value = row[column]
                if _looks_like_handle(value):
                    handles_by_person[owner].add(str(value).strip())

    return tuple(
        ContactRecord(display_name=people[person_id], handles=tuple(sorted(handles)))
        for person_id, handles in handles_by_person.items()
        if handles
    )


def _load_modern_contacts(
    conn: sqlite3.Connection,
    person_table: str,
    table_names: Iterable[str],
) -> tuple[ContactRecord, ...]:
    people = _load_people(
        conn,
        person_table,
        (
            "ZFIRSTNAME",
            "ZMIDDLENAME",
            "ZLASTNAME",
            "ZNICKNAME",
            "ZORGANIZATION",
            "ZDISPLAYNAME",
        ),
    )
    return _records_from_owner_tables(
        conn,
        people,
        table_names,
        lambda table: "PHONE" in table.upper() or "EMAIL" in table.upper(),
    )


def _load_legacy_contacts(
    conn: sqlite3.Connection,
    person_table: str,
    table_names: Iterable[str],
) -> tuple[ContactRecord, ...]:
    people = _load_people(
        conn,
        person_table,
        ("First", "Middle", "Last", "Nickname", "Organization"),
    )
    return _records_from_owner_tables(
        conn,
        people,
        table_names,
        lambda table: table in {"ABMultiValue", "ABMultiValueEntry"}
        or "PHONE" in table.upper()
        or "EMAIL" in table.upper(),
    )
