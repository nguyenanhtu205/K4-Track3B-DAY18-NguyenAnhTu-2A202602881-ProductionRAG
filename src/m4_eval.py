from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (EMBEDDING_MODEL, OPENAI_API_KEY, OPENROUTER_BASE_URL,
                    OPENROUTER_MODEL, TEST_SET_PATH)


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
    metric_names = (
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    )
    empty_result = {name: 0.0 for name in metric_names} | {"per_question": []}

    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        print("  ⚠️  RAGAS evaluation skipped: input lists have different lengths.")
        return empty_result
    if not questions:
        return empty_result

    try:
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)
        from ragas.run_config import RunConfig

        if not OPENAI_API_KEY:
            raise RuntimeError("OpenRouter API key is not configured")

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        llm = LangchainLLMWrapper(ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            tiktoken_model_name="gpt-3.5-turbo",
        ))
        embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        ))
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            # Free OpenRouter capacity is rate-limited. Keep the evaluator
            # responsive rather than allowing RAGAS's 16 concurrent workers
            # and long retries to stall the whole pipeline.
            run_config=RunConfig(timeout=30, max_retries=1, max_wait=10, max_workers=2),
        )
        dataframe = result.to_pandas()

        def score(row, metric: str) -> float:
            value = row.get(metric, 0.0)
            return float(value) if value == value else 0.0  # NaN becomes zero.

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=score(row, "faithfulness"),
                answer_relevancy=score(row, "answer_relevancy"),
                context_precision=score(row, "context_precision"),
                context_recall=score(row, "context_recall"),
            )
            for _, row in dataframe.iterrows()
        ]
        return {
            name: sum(getattr(item, name) for item in per_question) / len(per_question)
            for name in metric_names
        } | {"per_question": per_question}
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed ({type(error).__name__}): {error}")
        return empty_result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa thông tin ngoài context.",
            "Thắt chặt system prompt và đặt temperature = 0.",
        ),
        "context_recall": (
            "Retrieval bỏ sót chunk chứa thông tin cần thiết.",
            "Cải thiện chunking hoặc bổ sung từ khóa BM25.",
        ),
        "context_precision": (
            "Chunk không liên quan bị xếp quá cao.",
            "Bổ sung reranking hoặc lọc theo metadata.",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa tập trung trực tiếp vào câu hỏi.",
            "Cải thiện prompt để yêu cầu trả lời trực tiếp hơn.",
        ),
    }
    metric_names = tuple(diagnostic_tree)
    failures = []
    for result in eval_results:
        scores = {name: getattr(result, name) for name in metric_names}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(failures, key=lambda failure: failure["score"])[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "evaluation_note": (
            f"LLM evaluation is configured for OpenRouter free routing "
            f"({OPENROUTER_MODEL}), not OpenAI; embeddings use local {EMBEDDING_MODEL}."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
