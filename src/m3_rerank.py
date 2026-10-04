"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

from __future__ import annotations

import os
import sys
import time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K  # noqa: E402


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  Warning: cross-encoder unavailable ({exc}); using lexical fallback.", flush=True)

                class LexicalReranker:
                    @staticmethod
                    def predict(pairs):
                        import re

                        return [len(set(re.findall(r"\w+", query.lower())) &
                                    set(re.findall(r"\w+", document.lower())))
                                for query, document in pairs]

                self._model = LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        scores = self._load_model().predict([(query, d["text"]) for d in documents])
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [RerankResult(doc["text"], doc.get("score", 0.0), float(score),
                             doc.get("metadata", {}), rank)
                for rank, (score, doc) in enumerate(ranked[:top_k])]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        from flashrank import Ranker, RerankRequest

        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        ranked = Ranker().rerank(RerankRequest(query=query, passages=passages))
        return [RerankResult(documents[int(item["id"])]["text"],
                             documents[int(item["id"])].get("score", 0.0),
                             float(item["score"]),
                             documents[int(item["id"])].get("metadata", {}), rank)
                for rank, item in enumerate(ranked[:top_k])]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
