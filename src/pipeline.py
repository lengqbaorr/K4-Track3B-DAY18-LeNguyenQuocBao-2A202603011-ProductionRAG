"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

from __future__ import annotations

import os
import sys
import time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical  # noqa: E402
from src.m2_search import HybridSearch  # noqa: E402
from src.m3_rerank import CrossEncoderReranker  # noqa: E402
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report  # noqa: E402
from src.m5_enrichment import enrich_chunks  # noqa: E402
from config import RERANK_TOP_K  # noqa: E402


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_texts = {}
    for doc_index, doc in enumerate(docs):
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_id = f"{doc_index}:{parent.metadata['parent_id']}"
            parent_texts[parent_id] = parent.text
        for child in children:
            parent_id = f"{doc_index}:{child.parent_id}"
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": parent_id}})
    chunk_ms = (time.perf_counter() - t0) * 1000
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({chunk_ms / 1000:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text,
                       "metadata": {**e.auto_metadata, **chunk["metadata"]}}
                      for chunk, e in zip(all_chunks, enriched)]
        print(f"  ✓ Enriched {len(enriched)} chunks ({(time.perf_counter()-t0):.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    enrich_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    index_ms = (time.perf_counter() - t0) * 1000
    search.parents = parent_texts
    search.latency = {"chunk_ms": chunk_ms, "enrich_ms": enrich_ms, "index_ms": index_ms}
    search.query_times = {"search": 0.0, "rerank": 0.0, "llm": 0.0, "count": 0}
    print(f"  ✓ Indexed ({index_ms / 1000:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    results = search.search(query, top_k=20)
    search_ms = (time.perf_counter() - t0) * 1000
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    rerank_ms = (time.perf_counter() - t0) * 1000
    contexts = []
    seen = set()
    for result in reranked:
        parent_id = result.metadata.get("parent_id")
        key = parent_id or result.text
        if key in seen:
            continue
        seen.add(key)
        contexts.append(getattr(search, "parents", {}).get(parent_id, result.text))
        if len(contexts) == RERANK_TOP_K:
            break

    from config import OPENAI_API_KEY
    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
                {"role": "system", "content": (
                    "Trả lời ngắn gọn bằng tiếng Việt và chỉ dựa trên ngữ cảnh. "
                    "Nêu chính xác các con số và ngày tháng. Nếu các phiên bản chính sách mâu thuẫn, "
                    "ưu tiên phiên bản mới nhất. Nếu không có thông tin, nói 'Không tìm thấy thông tin.'"
                )},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    llm_ms = (time.perf_counter() - t0) * 1000
    if hasattr(search, "query_times"):
        search.query_times["search"] += search_ms
        search.query_times["rerank"] += rerank_ms
        search.query_times["llm"] += llm_ms
        search.query_times["count"] += 1
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    latency = dict(getattr(search, "latency", {}))
    times = getattr(search, "query_times", {})
    count = times.get("count", 0)
    for step in ("search", "rerank", "llm"):
        latency[f"{step}_avg_ms"] = times.get(step, 0.0) / count if count else 0.0
    print("\nLATENCY (ms)")
    for step, value in latency.items():
        print(f"  {step}: {value:.1f}")
    save_report(results, failures, latency=latency)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
