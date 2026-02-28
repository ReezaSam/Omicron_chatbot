# Omicron UniBot — Handover Document
> **For:** Teammate (LoRA Fine-tuning Task)  
> **From:** Implementation 1–3 (Data Pipeline + Chat System + Evaluation)  
> **Date:** 2026-02-28

---

## 0) Project Goal

Build a university chatbot that:
1. **Routes** student questions → `answer / clarify / escalate`
2. Uses **RAG** (handbook → chunks → embeddings → pgvector) to fetch relevant context
3. Uses **Groq Llama-3.3-70B** to generate the final answer
4. If question is vague → asks for **Department + Year + Semester**
5. If not confident or policy/special request → **Escalates** to human

---

## 1) What Has Been Done (Do NOT redo these)

| Implementation | Status | What was done |
|---|---|---|
| Implementation 1 | ✅ Done | PDF → clean text → chunks → embeddings → pgvector |
| Implementation 2 | ✅ Done | ML Router + RAG chat pipeline + Groq integration |
| Implementation 3 | ✅ Done | Router eval + RAG eval with hybrid FTS + vector retrieval |
| Embedding fine-tuning | ✅ Done | Fine-tuned `all-MiniLM-L6-v2` on 487 university QA pairs |

---

## 2) Setup Instructions

### Step 1: Install requirements
```bash
pip install -r requirements.txt
pip install datasets accelerate>=1.1.0
```

### Step 2: Configure `.env`
Copy `.env.example` to `.env` and fill in:
```
DB_NAME=omicron_db
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=5432
DOC_ID=1
GROQ_API_KEY=your_groq_key_here
```

### Step 3: Start Postgres + enable pgvector
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 4: Enable Hybrid FTS (run once in your DB)
```sql
ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS chunk_tsv tsvector;
UPDATE kb_chunks SET chunk_tsv = to_tsvector('english', coalesce(chunk_text, ''));
CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING GIN (chunk_tsv);
```

---

## 3) Pipeline Run Order

### Pipeline 1: Build Knowledge Base
```bash
python scripts/pdf_to_text.py
python scripts/clean_and_fix_handbook.py
python scripts/chunk_handbook.py
python scripts/embed_and_insert_chunks.py
```

### Pipeline 2: Train Router
```bash
python scripts/make_router_train_csv.py
python scripts/train_router_classifier.py
```
> Output model: `models/router_classifier.joblib`

### Pipeline 3: Fine-tune Embedding Model (already done)
```bash
python scripts/finetune_embedding.py
```
> Output model: `models/rusl_embedding_model`  
> ⚠️ After fine-tuning, always re-run `embed_and_insert_chunks.py` to re-embed chunks

### Pipeline 4: Run Evaluation
```bash
python scripts/evaluate_ml_router.py
python scripts/evaluate_rag_retrieval_v2.py
```

---

## 4) Frozen Baseline Numbers (Beat These!)

These are the **final numbers after embedding fine-tuning**.  
Your LoRA fine-tuned model must **improve on these.**

### Router Accuracy
| Metric | Baseline |
|---|---|
| Router Accuracy (90 eval questions) | **0.867** |

### RAG Retrieval — Final Baseline (487 QA pairs)
| Metric | @K=5 | @K=10 | Notes |
|---|---|---|---|
| ExactAnswerHit | 0.014 | 0.014 | Low by design — short answers |
| NumericHit | — | **0.764** | On 89 numeric answers ← KEY METRIC |
| KeywordHit | **0.522** | **0.569** | All 487 answers ← KEY METRIC |
| SoftHit | **0.201** | **0.240** | Soft overlap |
| MRR@5 | **0.143** | — | Mean Reciprocal Rank |

> ⚠️ Focus on beating **NumericHit@10 > 0.764** and **KeywordHit@10 > 0.569**

---

## 5) Embedding Model Info

| Item | Detail |
|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Fine-tuned on | 487 university QA pairs (`qa_pairs.csv`) |
| Training epochs | 3 |
| Loss function | `MultipleNegativesRankingLoss` |
| Output dim | 384 |
| Saved to | `models/rusl_embedding_model/` |

---

## 6) Your Task (LoRA Fine-tuning)

| Task | Details |
|---|---|
| Replace/adjust LLM | Use local model or keep Groq |
| Fine-tune with LoRA | Train LoRA adapter on university Q&A data |
| Keep pipeline same | Router + retrieval + confidence logic stays unchanged |
| Evaluate | Run `evaluate_rag_retrieval_v2.py` and compare to baseline above |

**Data available for fine-tuning:**
- `data/answer_questions_500.csv` — ~487 answer-type QA pairs
- `data/clarify_questions_200.csv` — 200 clarify questions
- `data/escalate_questions_200.csv` — 200 escalate questions
- `chunks/faculty_handbook_chunks_metadata.jsonl` — raw chunks with metadata

---

## 7) Folder Structure

```
Omicron_Bot/
  scripts/
    pdf_to_text.py
    clean_and_fix_handbook.py
    chunk_handbook.py
    embed_and_insert_chunks.py
    finetune_embedding.py           ← embedding fine-tuning
    Rag_Retriever.py
    EmbeddingModel.py
    chat_service.py
    make_router_train_csv.py
    train_router_classifier.py
    evaluate_ml_router.py
    evaluate_rag_retrieval_v2.py    ← USE THIS for eval
    debug_rag_failures.py
  data/
    answer_questions_500.csv
    clarify_questions_200.csv
    escalate_questions_200.csv
    router_train.csv
    qa_pairs.csv
  chunks/
    faculty_handbook_chunks_metadata.jsonl
  models/
    router_classifier.joblib        ← trained ML router
    rusl_embedding_model/           ← fine-tuned embedding model
  eval_rag_outputs_v2/
    full_eval_results.csv           ← baseline results
    retrieval_metrics_bar.png       ← baseline chart
  .env.example
  README_HANDOVER.md
```

---

## 8) Key Technical Notes

- **Embedding model:** Fine-tuned `all-MiniLM-L6-v2` at `models/rusl_embedding_model` (384 dim)
- **Vector DB:** Postgres + pgvector, table: `kb_chunks`
- **Retrieval:** Hybrid — pgvector (cosine) + Postgres FTS (tsvector + GIN index)
- **Router model:** Scikit-learn classifier saved as `router_classifier.joblib`
- **LLM:** Groq API with `llama-3.3-70b` (set `GROQ_API_KEY` in `.env`)
- **Escalation threshold:** distance > 0.65 OR admin policy question with distance > 0.55

---

## 9) Known Issues / Limitations

| Issue | Status |
|---|---|
| Vector search weak for short facts (years, IDs) | ✅ Fixed with hybrid FTS |
| SoftHit metric unfair for short gold answers | ⚠️ Use NumericHit + KeywordHit instead |
| Router trained on similar splits — test on harder unseen set | ⚠️ Note for teammate |
| HF Token warning on embedding load | ✅ Harmless, can be ignored |
| Only 487 QA pairs for embedding fine-tuning | ⚠️ More data would help |

---

*Good luck! Reach out if any script fails on setup.*