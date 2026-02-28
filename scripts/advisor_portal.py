# scripts/advisor_portal.py
import os
import psycopg2

# ================================
# DATABASE CONFIG
# ================================
DB_NAME = "omicron_db"
DB_USER = "postgres"
DB_PASSWORD = "2719"
DB_HOST = "localhost"
DB_PORT = "5432"

# Advisor username from backend/session (no prompting)
ADVISOR_USERNAME = os.getenv("OMICRON_ADVISOR", "adv01")


def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        
        host=DB_HOST,
        port=DB_PORT
    )


def require_advisor_exists(cur, username: str):
    cur.execute("SELECT role FROM users WHERE username = %s;", (username,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Advisor '{username}' not found in users table.")
    if (row[0] or "").lower() != "advisor":
        raise ValueError(f"User '{username}' is not an advisor (role={row[0]}).")


def list_tickets(cur, statuses=("pending", "assigned"), limit=30):
    cur.execute(
        """
        SELECT
            e.escalation_id,
            e.query_id,
            q.username AS student_username,
            q.query_text,
            e.status,
            e.escalated_at
        FROM escalations e
        JOIN queries q ON q.query_id = e.query_id
        WHERE e.status = ANY(%s)
        ORDER BY e.escalated_at ASC
        LIMIT %s;
        """,
        (list(statuses), limit)
    )
    return cur.fetchall()


def get_ticket(cur, escalation_id: int):
    cur.execute(
        """
        SELECT
            e.escalation_id,
            e.query_id,
            q.username AS student_username,
            q.query_text,
            q.created_at,
            e.status,
            e.advisor_username,
            e.advisor_response,
            e.escalated_at,
            e.responded_at
        FROM escalations e
        JOIN queries q ON q.query_id = e.query_id
        WHERE e.escalation_id = %s;
        """,
        (escalation_id,)
    )
    return cur.fetchone()


def assign_ticket(cur, escalation_id: int, advisor_username: str):
    """
    If pending -> assigned, set advisor_username if empty.
    """
    cur.execute(
        """
        UPDATE escalations
        SET
            advisor_username = COALESCE(advisor_username, %s),
            status = CASE WHEN status='pending' THEN 'assigned' ELSE status END
        WHERE escalation_id = %s;
        """,
        (advisor_username, escalation_id)
    )


def resolve_ticket(cur, escalation_id: int, advisor_username: str, response_text: str) -> bool:
    """
    Mark resolved + save advisor response + also update queries.chatbot_response
    so student can see it.
    Returns True if success, False if escalation not found.
    """
    # 1) Make sure escalation exists and get query_id
    cur.execute(
        "SELECT query_id FROM escalations WHERE escalation_id = %s;",
        (escalation_id,)
    )
    row = cur.fetchone()
    if not row:
        return False
    query_id = row[0]

    # 2) Update escalations row
    cur.execute(
        """
        UPDATE escalations
        SET
            advisor_username = %s,
            advisor_response = %s,
            status = 'resolved',
            responded_at = NOW()
        WHERE escalation_id = %s;
        """,
        (advisor_username, response_text, escalation_id)
    )

    # 3) Push final answer to queries table for student display
    cur.execute(
        """
        UPDATE queries
        SET chatbot_response = %s
        WHERE query_id = %s;
        """,
        (response_text, query_id)
    )

    return True


def main():
    print("\n✅ Omicron Advisor Portal")
    print("Commands: list | open <id> | resolve <id> | exit\n")

    conn = connect()
    cur = conn.cursor()

    try:
        require_advisor_exists(cur, ADVISOR_USERNAME)
        print(f"✅ Logged in advisor (session): {ADVISOR_USERNAME}\n")

        while True:
            cmd = input("advisor> ").strip()

            if cmd.lower() in {"exit", "quit", "q"}:
                print("👋 Exiting advisor portal.")
                break

            if cmd.lower() in {"list", "ls"}:
                rows = list_tickets(cur)
                if not rows:
                    print("🎉 No pending/assigned tickets.\n")
                    continue

                print("\n================= TICKETS =================")
                for esc_id, qid, stu, qtext, status, escalated_at in rows:
                    print(f"\nEscalation ID : {esc_id}")
                    print(f"Query ID      : {qid}")
                    print(f"Student       : {stu}")
                    print(f"Status        : {status}")
                    print(f"Escalated At  : {escalated_at}")
                    print(f"Question      : {qtext[:220]}{'...' if len(qtext) > 220 else ''}")
                    print("------------------------------------------")
                print()
                continue

            if cmd.lower().startswith("open "):
                parts = cmd.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("❌ Usage: open <escalation_id>\n")
                    continue

                esc_id = int(parts[1])
                t = get_ticket(cur, esc_id)
                if not t:
                    print("❌ Ticket not found.\n")
                    continue

                (escalation_id, query_id, student_username, query_text, created_at,
                 status, advisor_username, advisor_response, escalated_at, responded_at) = t

                print("\n================= TICKET DETAILS =================")
                print(f"Escalation ID  : {escalation_id}")
                print(f"Query ID       : {query_id}")
                print(f"Student        : {student_username}")
                print(f"Question Time  : {created_at}")
                print(f"Escalated At   : {escalated_at}")
                print(f"Status         : {status}")
                print(f"Advisor        : {advisor_username}")
                print(f"Responded At   : {responded_at}")
                print("\nQuestion:")
                print(query_text)
                if advisor_response:
                    print("\nAdvisor Response:")
                    print(advisor_response)
                print("==================================================\n")
                continue

            if cmd.lower().startswith("resolve "):
                parts = cmd.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("❌ Usage: resolve <escalation_id>\n")
                    continue

                esc_id = int(parts[1])
                t = get_ticket(cur, esc_id)
                if not t:
                    print("❌ Ticket not found.\n")
                    continue

                status = t[5]
                if status == "resolved":
                    print("✅ Already resolved.\n")
                    continue

                # assign first (so advisor_username gets filled)
                assign_ticket(cur, esc_id, ADVISOR_USERNAME)
                conn.commit()

                print("Type your response. End with a blank line:\n")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)

                response_text = "\n".join(lines).strip()
                if not response_text:
                    print("❌ Empty response; nothing saved.\n")
                    continue

                ok = resolve_ticket(cur, esc_id, ADVISOR_USERNAME, response_text)
                if not ok:
                    print("❌ Could not resolve: escalation_id not found.\n")
                    continue

                conn.commit()
                print("✅ Resolved and saved. Student will see it in chatbot.\n")
                continue

            print("❌ Unknown command. Use: list | open <id> | resolve <id> | exit\n")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
