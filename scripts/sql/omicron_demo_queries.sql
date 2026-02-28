-- ==========================================
-- OMICRON_DB DEMO QUERIES (Panel Proof)
-- ==========================================

-- 1) Show all tables
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;


-- 2) View kb_documents
SELECT * FROM kb_documents;

-- 3) Count kb_chunks
SELECT COUNT(*) AS total_chunks
FROM kb_chunks;

-- 4) Check embeddings exist
SELECT COUNT(*) AS total_rows,
       COUNT(embedding) AS rows_with_embeddings
FROM kb_chunks;

-- 5) Check if any embedding is NULL
SELECT COUNT(*) AS embedding_nulls
FROM kb_chunks
WHERE embedding IS NULL;

-- 6) View sample chunks (preview)
SELECT chunk_id, section, chunk_type,
       LEFT(chunk_text, 120) AS preview
FROM kb_chunks
LIMIT 10;

-- 7) View actual embedding (first row)
SELECT chunk_id, LEFT(embedding::text, 200) AS embedding_preview
FROM kb_chunks
LIMIT 5;

-- 8) Check embedding column type
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'kb_chunks'
AND column_name = 'embedding';

--See the Actual Vector (Optional)

--To see part of the vector:

SELECT chunk_id,
       embedding
FROM kb_chunks
LIMIT 1;

-- 9) Show last 5 user queries
SELECT query_id, username, created_at, is_escalated
FROM queries
ORDER BY created_at DESC
LIMIT 5;

-- 10) Show escalations
SELECT escalation_id, query_id, escalated_to, status, escalation_date
FROM escalations
ORDER BY escalation_date DESC;

-- 11) Show feedback
SELECT feedback_id, username, date_submitted, LEFT(message, 80) AS preview
FROM feedback
ORDER BY date_submitted DESC;
