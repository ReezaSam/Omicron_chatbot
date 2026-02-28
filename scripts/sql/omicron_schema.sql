-- =========================================================
-- OMICRON_DB : COMPLETE SCHEMA (SDS/SRS + KB + PGVECTOR)
-- =========================================================

-- 0) Create DB (run this in postgres database / server level if needed)
-- CREATE DATABASE omicron_db;

-- Connect to: omicron_db before running the rest

-- =========================================================
-- 1) EXTENSIONS
-- =========================================================
-- pgvector (for embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: pgcrypto if you want UUID generation later
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================================================
-- 2) USERS (SDS: Username is Primary Key)
-- =========================================================
DROP TABLE IF EXISTS feedback CASCADE;
DROP TABLE IF EXISTS escalations CASCADE;
DROP TABLE IF EXISTS queries CASCADE;
DROP TABLE IF EXISTS kb_chunks CASCADE;
DROP TABLE IF EXISTS kb_documents CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    username VARCHAR(50) PRIMARY KEY,

    first_name VARCHAR(100) NOT NULL,
    last_name  VARCHAR(100) NOT NULL,

    email VARCHAR(100) UNIQUE NOT NULL,

    password_hash TEXT NOT NULL,

    role VARCHAR(20) NOT NULL
        CHECK (role IN ('student', 'admin', 'advisor')),

    studentid VARCHAR(30), -- nullable (only for students)

    department VARCHAR(50) NOT NULL
        CHECK (department IN (
            'Computing',
            'Health Promotion',
            'Applied Science - Biology',
            'Applied Science - Physical'
        ))
);

-- =========================================================
-- 3) QUERIES (SDS Query Entity)
-- =========================================================
CREATE TABLE queries (
    query_id SERIAL PRIMARY KEY,

    username VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,

    is_escalated BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_queries_user
        FOREIGN KEY (username)
        REFERENCES users(username)
        ON DELETE CASCADE
);

-- =========================================================
-- 4) ESCALATIONS (SDS Escalation Entity)
-- =========================================================
CREATE TABLE escalations (
    escalation_id SERIAL PRIMARY KEY,

    query_id INT NOT NULL UNIQUE,   -- one escalation per query
    escalated_to VARCHAR(50),       -- advisor username (nullable until assigned)

    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'assigned', 'resolved')),

    response_text TEXT,
    escalation_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    responded_at TIMESTAMP,

    CONSTRAINT fk_escalations_query
        FOREIGN KEY (query_id)
        REFERENCES queries(query_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_escalations_advisor
        FOREIGN KEY (escalated_to)
        REFERENCES users(username)
        ON DELETE SET NULL
);

-- =========================================================
-- 5) FEEDBACK (SDS Feedback Entity)
-- =========================================================
CREATE TABLE feedback (
    feedback_id SERIAL PRIMARY KEY,

    username VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,

    date_submitted TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_feedback_user
        FOREIGN KEY (username)
        REFERENCES users(username)
        ON DELETE CASCADE
);

-- =========================================================
-- 6) KNOWLEDGE BASE DOCUMENTS (KB)
-- =========================================================
CREATE TABLE kb_documents (
    doc_id SERIAL PRIMARY KEY,

    title VARCHAR(200) NOT NULL,

    department VARCHAR(50)
        CHECK (department IN (
            'Computing',
            'Health Promotion',
            'Applied Science - Biology',
            'Applied Science - Physical'
        )),

    -- Optional: store full doc text (you can keep short placeholder if using chunks)
    content TEXT NOT NULL,

    uploaded_by VARCHAR(50),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_kb_uploaded_by
        FOREIGN KEY (uploaded_by)
        REFERENCES users(username)
        ON DELETE SET NULL
);

-- =========================================================
-- 7) KNOWLEDGE BASE CHUNKS (KB + PGVECTOR)
-- =========================================================
CREATE TABLE kb_chunks (
    chunk_id SERIAL PRIMARY KEY,

    doc_id INT NOT NULL,

    -- Extra metadata from your JSONL chunker
    section TEXT,
    chunk_type VARCHAR(10) NOT NULL DEFAULT 'text'
        CHECK (chunk_type IN ('text','table')),

    chunk_text TEXT NOT NULL,

    -- SBERT all-MiniLM-L6-v2 => 384 dimensions
    embedding vector(384),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_kb_chunks_doc
        FOREIGN KEY (doc_id)
        REFERENCES kb_documents(doc_id)
        ON DELETE CASCADE
);

-- =========================================================
-- 8) (Recommended) Vector index for faster/better retrieval
-- =========================================================
-- Use cosine distance operators (best for SBERT)
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
ON kb_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

ANALYZE kb_chunks;

-- =========================================================
-- DONE ✅
-- =========================================================


-- =========================
-- Verification
-- =========================
--What to show to panel after running this--
--Show all tables exist--
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;

--Show kb_chunks has pgvector embedding--
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name='kb_chunks'
ORDER BY ordinal_position;

--Show you inserted chunks + embeddings--
SELECT COUNT(*) AS total_chunks,
       SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS embedding_nulls
FROM kb_chunks;




-- Optional: add source_file to documents (nice to have)--

ALTER TABLE kb_documents
ADD COLUMN IF NOT EXISTS source_file TEXT;