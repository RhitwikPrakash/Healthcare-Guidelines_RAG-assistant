import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    label: int
    source: str
    page: int
    text: str


def split_text(text: str, max_chars: int = 1200, overlap: int = 180) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)

            if len(para) <= max_chars:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunks.append(para[start:end].strip())
                    start = max(0, end - overlap)
                current = ""

    if current:
        chunks.append(current)

    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 80:
            final_chunks.append(chunk)

    return final_chunks


def build_chunks(pages: List[Dict]) -> List[Chunk]:
    chunks = []
    label = 1

    for page in pages:
        page_chunks = split_text(page["text"])

        for text in page_chunks:
            chunks.append(
                Chunk(
                    label=label,
                    source=page["source"],
                    page=page["page"],
                    text=text,
                )
            )
            label += 1

    return chunks


def build_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, object]:
    texts = [chunk.text for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=1,
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_chunks(
    question: str,
    chunks: List[Chunk],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 8,
) -> List[Chunk]:
    if not chunks:
        return []

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(chunks[idx])

    if not results:
        results = chunks[: min(top_k, len(chunks))]

    for new_label, chunk in enumerate(results, start=1):
        chunk.label = new_label

    return results


def build_evidence_text(chunks: List[Chunk]) -> str:
    evidence_parts = []

    for chunk in chunks:
        evidence_parts.append(
            f"[{chunk.label}] Source: {chunk.source}, page {chunk.page}\n{chunk.text}"
        )

    return "\n\n---\n\n".join(evidence_parts)


def build_answer_prompt(question: str, evidence_text: str, answer_mode: str) -> str:
    if answer_mode == "Fast":
        style = "Answer briefly in 5-8 bullet points."
    elif answer_mode == "Deep Research":
        style = "Give a detailed, structured answer with headings and bullet points."
    else:
        style = "Give a clear, balanced answer with enough detail."

    return f"""
You are a careful Healthcare Guidelines RAG Assistant.

You must answer ONLY using the evidence given below.
Do not use outside knowledge.
Do not invent facts.
If the evidence does not contain the answer, say:
"The uploaded document does not provide enough evidence to answer this."

Citation rules:
- Every factual sentence must include an inline evidence label like [1], [2], or [3].
- Use only labels that appear in the evidence.
- Do not write citation labels that are not provided.
- Do not put all citations only at the end.
- Avoid uncited medical claims.

Answer style:
{style}

Question:
{question}

Evidence:
{evidence_text}

Now write the final answer with inline citations.
""".strip()


def valid_citation_labels(answer: str, allowed_labels: List[int]) -> List[int]:
    found = re.findall(r"\[(\d+)\]", answer)
    allowed = set(allowed_labels)

    valid = []
    for item in found:
        num = int(item)
        if num in allowed and num not in valid:
            valid.append(num)

    return valid


def looks_incomplete(answer: str) -> bool:
    if not answer or len(answer.strip()) < 40:
        return True

    text = answer.strip()
    lowered = text.lower()

    bad_endings = (
        "including",
        "such as",
        "for example",
        "and",
        "or",
        "with",
        "because",
        "therefore",
        "including:",
        "such as:",
    )

    if lowered.endswith(bad_endings):
        return True

    if text[-1] not in ".!?)]}":
        return True

    return False


def build_repair_prompt(
    question: str,
    draft_answer: str,
    evidence_text: str,
) -> str:
    return f"""
Rewrite the answer below so that it is complete and has valid inline evidence labels.

Rules:
- Keep the meaning the same.
- Use only the evidence.
- Every factual sentence must include labels like [1], [2], or [3].
- Use only labels that appear in the evidence.
- Do not add unsupported claims.
- Do not mention that you are repairing the answer.

Question:
{question}

Evidence:
{evidence_text}

Draft answer:
{draft_answer}

Final repaired answer:
""".strip()


def force_basic_citations(answer: str, evidence_labels: List[int]) -> str:
    if not answer.strip() or not evidence_labels:
        return answer

    existing = valid_citation_labels(answer, evidence_labels)
    if existing:
        return answer

    main_label = evidence_labels[0]
    backup_label = evidence_labels[1] if len(evidence_labels) > 1 else main_label

    lines = answer.splitlines()
    fixed_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            fixed_lines.append(line)
            continue

        if stripped.startswith(("#", "-", "*")):
            if re.search(r"\[\d+\]", stripped):
                fixed_lines.append(line)
            else:
                fixed_lines.append(f"{line} [{main_label}]")
            continue

        if len(stripped) > 20 and not re.search(r"\[\d+\]", stripped):
            if stripped.endswith("."):
                fixed_lines.append(f"{line} [{main_label}]")
            else:
                fixed_lines.append(f"{line}. [{backup_label}]")
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def format_evidence_cards(chunks: List[Chunk]) -> List[Dict]:
    cards = []

    for chunk in chunks:
        preview = chunk.text.strip()

        if len(preview) > 1200:
            preview = preview[:1200].rstrip() + "..."

        cards.append(
            {
                "label": chunk.label,
                "source": chunk.source,
                "page": chunk.page,
                "text": preview,
            }
        )

    return cards