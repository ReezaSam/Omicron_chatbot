--1) users

--Purpose: stores system users and their roles (Student / Advisor / Admin).
--Why important: every query, feedback, and escalation must be linked to a user.--

--Key columns:--

username (PK): unique identifier (as SDS says)

first_name, last_name

email (unique)

password_hash

role (student/admin/advisor) with CHECK constraint

studentid (nullable; only for students)

department (CHECK constraint)

Panel explanation: “We use username as PK to match the SDS. Roles control access (students ask queries, advisors respond to escalations, admin manages).”--

--Show to panel:--

SELECT username, role, department FROM users LIMIT 10;

---------------------------
--2) queries--

--Purpose: logs every chatbot interaction.
--Why important: this is your chat history + auditing and also supports escalation.

--Key columns:

query_id (PK, serial)

username (FK → users.username)

question (text)

answer (text)

is_escalated (boolean default false)

created_at

--Panel explanation: “Each query belongs to exactly one user; if confidence is low, we set is_escalated=true.”

--Show to panel:

SELECT query_id, username, is_escalated, created_at
FROM queries
ORDER BY created_at DESC
LIMIT 10;

---------------------------------
--3) escalations

--Purpose: stores cases where chatbot couldn’t answer confidently and a human advisor must reply.
Why important: matches SDS escalation entity and flow.

--Key columns:

escalation_id (PK)

query_id (FK → queries.query_id, UNIQUE so one escalation per query)

advisor_id or advisor_username (FK → users; can be NULL until assigned)

status (pending/assigned/resolved)

advisor_response

escalated_at, responded_at

--Panel explanation: “Escalation is created when AI confidence is low; advisor later responds and status changes to resolved.”

--Show to panel:

SELECT escalation_id, query_id, status, escalated_at, responded_at
FROM escalations
ORDER BY escalated_at DESC
LIMIT 10;

-------------------
--4) feedback

--Purpose: collects user feedback about chatbot performance.
--Why important: supports improvement & evaluation.

--Key columns:

feedback_id (PK)

username (FK → users.username)

message

created_at

--Panel explanation: “Feedback is linked to the user who submitted it; we can analyze feedback later.”

--Show to panel:

SELECT feedback_id, username, created_at, LEFT(message, 80) AS preview
FROM feedback
ORDER BY created_at DESC
LIMIT 10;

-----------------------------
--B) Knowledge Base Tables (for RAG + pgvector)

--These are what you (AI/DB side) add to support “handbook Q&A”.

--5) kb_documents

--Purpose: stores metadata about each source document (Faculty handbook PDF, policies, etc.)
--Why important: you can have multiple documents in the future.

--Your current columns (fine):

doc_id (PK)

title

department (optional classification)

content (optional; many teams keep this as “full doc text”, but chunks are what matter)

uploaded_by (FK to users.username) nullable

created_at

--Panel explanation: “Each uploaded handbook becomes one kb_document. Then we split it into many chunks for semantic search.”

--Show to panel:

SELECT doc_id, title, department, uploaded_by, created_at
FROM kb_documents;


--Tip for panel: If you don’t really use content, say: “We keep content for archival, but retrieval uses chunks.”

--6) kb_chunks ✅ (MOST IMPORTANT FOR RAG)

--Purpose: stores chunked text pieces + their embeddings (vectors).
--Why important: this is what makes “semantic search” possible.

--Columns you should have:

chunk_id (PK)

doc_id (FK → kb_documents.doc_id)

chunk_text (the chunk)

section (which heading/section it came from)

chunk_type (text or table) so tables are preserved

embedding vector(384) (pgvector column)

created_at

--Panel explanation (simple):

--“We split the handbook into 334 chunks.”

--“We generate SBERT embeddings for each chunk (384-dim).”

--“When user asks a question, we embed the question and retrieve closest chunks using cosine similarity.”

--Show to panel (proof):

SELECT COUNT(*) FROM kb_chunks WHERE doc_id = 1;

SELECT chunk_id, section, chunk_type,
       (embedding IS NOT NULL) AS has_embedding,
       LEFT(chunk_text, 100) AS preview
FROM kb_chunks
LIMIT 10;

--C) pgvector Setup (What to explain)
--Why pgvector?

--Because SQL can do similarity search directly inside Postgres.

--What you did:

Enable extension:

CREATE EXTENSION IF NOT EXISTS vector;


--Create embedding column:

ALTER TABLE kb_chunks
ADD COLUMN embedding vector(384);


--Use cosine distance search:

SELECT chunk_id, chunk_text,
       (embedding <=> '[...]'::vector(384)) AS distance
FROM kb_chunks
WHERE doc_id = 1
ORDER BY distance
LIMIT 5;


--Panel explanation (one line):
--“pgvector lets us store embeddings and run vector similarity search inside PostgreSQL.”

--D) How to show tables to panel (pgAdmin steps)
--Option 1: Show table list (GUI)

pgAdmin → Databases → omicron_db

Schemas → public → Tables

--Expand and show:
users, queries, escalations, feedback, kb_documents, kb_chunks

---📸 Screenshot: the table list expanded.

Option 2: Show table definitions (SQL = best for viva)

Run these and screenshot results:

--Show columns:

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'kb_chunks'
ORDER BY ordinal_position;


--Show constraints:

SELECT conname, pg_get_constraintdef(c.oid)
FROM pg_constraint c
JOIN pg_class t ON c.conrelid = t.oid
WHERE t.relname = 'kb_chunks';

