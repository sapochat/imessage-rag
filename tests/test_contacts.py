import sqlite3

from imessage_rag.contacts import ContactRecord, ContactResolver, load_contacts, normalize_handle


class TestContactResolver:
    def test_normalizes_phone_numbers(self):
        assert normalize_handle("+1 (555) 123-4567") == "15551234567"

    def test_resolves_handles_and_names(self):
        resolver = ContactResolver(
            records=(
                ContactRecord(
                    display_name="Alice Example",
                    handles=("+1 (555) 123-4567", "alice@example.test"),
                ),
            )
        )

        assert resolver.label_for_handle("15551234567") == "Alice Example"
        assert resolver.label_for_handle("ALICE@example.test") == "Alice Example"
        assert resolver.handles_for_contact("alice example") == (
            "+1 (555) 123-4567",
            "alice@example.test",
        )
        assert resolver.handles_for_contact("alice") == (
            "+1 (555) 123-4567",
            "alice@example.test",
        )
        assert resolver.label_for_handle("+15550000000") == "+15550000000"

    def test_ambiguous_partial_name_does_not_match(self):
        resolver = ContactResolver(
            records=(
                ContactRecord("Alice Example", ("+15551234567",)),
                ContactRecord("Alice Other", ("+15557654321",)),
            )
        )

        assert resolver.handles_for_contact("alice") == ()


class TestLoadContacts:
    def test_loads_modern_addressbook_schema(self, tmp_path):
        db_path = tmp_path / "AddressBook-v22.abcddb"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ZABCDPHONENUMBER (
                ZOWNER INTEGER,
                ZFULLNUMBER TEXT,
                ZLABEL TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ZABCDEMAILADDRESS (
                ZOWNER INTEGER,
                ZADDRESS TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO ZABCDRECORD (Z_PK, ZFIRSTNAME, ZLASTNAME) VALUES (1, 'Alice', 'Example')"
        )
        conn.execute(
            "INSERT INTO ZABCDPHONENUMBER (ZOWNER, ZFULLNUMBER, ZLABEL) VALUES (1, '+1 (555) 123-4567', '_$!<Mobile>!$_')"
        )
        conn.execute(
            "INSERT INTO ZABCDEMAILADDRESS (ZOWNER, ZADDRESS) VALUES (1, 'alice@example.test')"
        )
        conn.commit()
        conn.close()

        resolver = load_contacts([db_path])

        assert resolver.errors == ()
        assert resolver.contact_count == 1
        assert resolver.handle_count == 2
        assert resolver.label_for_handle("+15551234567") == "Alice Example"
        assert resolver.label_for_handle("alice@example.test") == "Alice Example"

    def test_loads_legacy_addressbook_schema(self, tmp_path):
        db_path = tmp_path / "AddressBook-v6.abcddb"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE ABPerson (
                ROWID INTEGER PRIMARY KEY,
                First TEXT,
                Last TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ABMultiValue (
                record_id INTEGER,
                value TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO ABPerson (ROWID, First, Last) VALUES (1, 'Bob', 'Example')"
        )
        conn.execute(
            "INSERT INTO ABMultiValue (record_id, value) VALUES (1, '+1 555 765 4321')"
        )
        conn.commit()
        conn.close()

        resolver = load_contacts([db_path])

        assert resolver.errors == ()
        assert resolver.contact_count == 1
        assert resolver.label_for_handle("15557654321") == "Bob Example"
