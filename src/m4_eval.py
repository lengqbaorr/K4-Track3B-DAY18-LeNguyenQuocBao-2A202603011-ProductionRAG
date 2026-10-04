"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

from __future__ import annotations

import os
import sys
import json
import math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH  # noqa: E402


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    def score(value):
        number = float(value)
        return 0.0 if math.isnan(number) else number

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

        answer_relevancy.strictness = 1  # proxy rejects n>1 completions (400) -> NaN scores
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        per_question = [EvalResult(row["question"], row["answer"], row["contexts"],
                                   row["ground_truth"],
                                   *(score(row.get(name, 0.0)) for name in names))
                        for _, row in result.to_pandas().iterrows()]
        return {**{name: score(result[name]) for name in names},
                "per_question": per_question}
    except Exception as exc:
        print(f"  RAGAS evaluation failed: {exc}")
        return {**dict.fromkeys(names, 0.0), "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
    }
    names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    worst = sorted(eval_results, key=lambda row: sum(getattr(row, name) for name in names) / 4)
    failures = []
    for row in worst[:bottom_n]:
        metric = min(names, key=lambda name: getattr(row, name))
        diagnosis, fix = tree[metric]
        failures.append({"question": row.question, "worst_metric": metric,
                         "score": getattr(row, metric), "diagnosis": diagnosis,
                         "suggested_fix": fix})
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json",
                latency: dict | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "latency": latency or {},
        "per_question": [vars(r) for r in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=list)  # ragas returns ndarray contexts
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
