from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


_api_unavailable = False


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _chat(messages: list[dict], max_tokens: int) -> str | None:
    """Call the configured OpenAI-compatible provider, with a safe fallback."""
    global _api_unavailable
    if not OPENAI_API_KEY or _api_unavailable:
        return None
    try:
        from openai import OpenAI

        response = OpenAI(
            api_key=OPENAI_API_KEY, base_url=OPENROUTER_BASE_URL
        ).chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as error:
        # Avoid repeatedly retrying an unavailable/rate-limited free endpoint
        # for every chunk in the corpus.
        _api_unavailable = True
        print(f"  ⚠️  Enrichment LLM unavailable ({type(error).__name__}); using fallback.")
        return None


def _fallback_summary(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    return " ".join(sentences[:2]) if sentences else text


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    candidates = [
        "Đoạn văn này quy định điều gì?",
        "Quy định này áp dụng cho ai?",
        "Những điều kiện hoặc thời hạn quan trọng là gì?",
    ]
    return candidates[:max(n_questions, 0)] if text.strip() else []


def _fallback_context(source: str) -> str:
    return f"Đoạn trích thuộc tài liệu {source}." if source else "Đoạn trích từ một quy định nội bộ."


def _fallback_metadata(text: str) -> dict:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "general")
    return {"topic": first_line[:100], "entities": [], "category": "policy", "language": "vi"}


def _parse_json(content: str | None) -> dict:
    if not content:
        return {}
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {}


def _normalise_enrichment(result: dict, text: str, source: str) -> dict:
    questions = result.get("questions", [])
    if isinstance(questions, str):
        questions = [line.strip(" -•0123456789.\t") for line in questions.splitlines() if line.strip()]
    metadata = result.get("metadata", {})
    return {
        "summary": str(result.get("summary") or _fallback_summary(text)),
        "questions": [str(question).strip() for question in questions if str(question).strip()][:3]
        or _fallback_questions(text),
        "context": str(result.get("context") or _fallback_context(source)),
        "metadata": metadata if isinstance(metadata, dict) and metadata else _fallback_metadata(text),
    }


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    response = _chat([
        {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
        {"role": "user", "content": text},
    ], max_tokens=150)
    return response or _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    response = _chat([
        {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên một dòng."},
        {"role": "user", "content": text},
    ], max_tokens=200)
    if response:
        questions = [line.strip(" -•0123456789.\t") for line in response.splitlines() if line.strip()]
        if questions:
            return questions[:n_questions]
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    response = _chat([
        {"role": "system", "content": "Viết một câu ngắn mô tả vị trí và chủ đề của đoạn văn trong tài liệu. Chỉ trả về một câu."},
        {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=80)
    return f"{response or _fallback_context(document_title)}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    response = _chat([
        {"role": "system", "content": 'Trích xuất metadata và chỉ trả JSON: {"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}.'},
        {"role": "user", "content": text},
    ], max_tokens=150)
    return _parse_json(response) or _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    response = _chat([
        {"role": "system", "content": """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "một câu mô tả vị trí và chủ đề của đoạn văn trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
        {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=400)
    return _normalise_enrichment(_parse_json(response), text, source)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
