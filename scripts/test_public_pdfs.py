from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))

from app.services.bm25_store import BM25Okapi, tokenize  # noqa: E402
from app.services.chunking import chunk_pages  # noqa: E402
from app.services.pdf_loader import load_pdf  # noqa: E402


CASES = [
    {
        "name": "who_hypertension.pdf",
        "url": "https://www.ncbi.nlm.nih.gov/books/n/who344424/pdf/",
        "question": "When should pharmacological treatment for hypertension be initiated?",
        "retrieval_query": "hypertension pharmacological treatment initiation threshold systolic diastolic blood pressure",
        "expected": ["140", "90", "hypertension", "pharmacological"],
    },
    {
        "name": "who_tb_preventive_treatment.pdf",
        "url": "https://www.ncbi.nlm.nih.gov/books/n/who378536/pdf/",
        "question": "Who should receive tuberculosis preventive treatment?",
        "retrieval_query": "tuberculosis preventive treatment eligible populations HIV household contacts TB disease excluded",
        "expected": ["preventive treatment", "hiv", "household contacts", "tb disease"],
    },
    {
        "name": "who_physical_activity.pdf",
        "url": "https://www.ncbi.nlm.nih.gov/books/n/who336656/pdf/",
        "question": "What physical activity is recommended for adults?",
        "retrieval_query": "adults physical activity recommendation aerobic moderate vigorous minutes per week muscle strengthening",
        "expected": ["150", "300", "moderate-intensity", "adults"],
    },
]


def download(case: dict, directory: Path) -> Path:
    target = directory / case["name"]
    if target.exists() and target.stat().st_size > 10000:
        return target
    response = requests.get(case["url"], timeout=120, allow_redirects=True)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def local_retrieval_test(directory: Path) -> dict:
    report = {"cases": [], "passed": True}
    for case in CASES:
        path = download(case, directory)
        pages = load_pdf(path)
        chunks = chunk_pages(pages)
        bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
        scores = bm25.get_scores(tokenize(case["retrieval_query"]))
        top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:5]
        top_text = " ".join(chunks[index]["text"] for index in top_indices).lower()
        hits = [term for term in case["expected"] if term.lower() in top_text]
        passed = len(hits) >= 3
        report["cases"].append(
            {
                "name": case["name"],
                "pages": len(pages),
                "chunks": len(chunks),
                "question": case["question"],
                "retrieval_query": case["retrieval_query"],
                "keyword_hits": hits,
                "passed": passed,
                "top_pages": [chunks[index]["page"] for index in top_indices],
                "top_preview": " ".join(top_text.split()[:20]),
            }
        )
        report["passed"] = report["passed"] and passed
    return report


def api_test(directory: Path, api_url: str) -> dict:
    session_id = f"public-test-{int(time.time())}"
    files = []
    handles = []
    try:
        for case in CASES:
            path = download(case, directory)
            handle = path.open("rb")
            handles.append(handle)
            files.append(("files", (path.name, handle, "application/pdf")))
        response = requests.post(f"{api_url}/documents/upload", data={"session_id": session_id}, files=files, timeout=180)
        response.raise_for_status()
        upload_job = response.json()["job_id"]
        upload_result = poll(api_url, upload_job)
        answers = []
        for case in CASES:
            response = requests.post(
                f"{api_url}/query",
                json={"session_id": session_id, "question": case["question"], "mode": "deep", "history": []},
                timeout=30,
            )
            response.raise_for_status()
            result = poll(api_url, response.json()["job_id"])
            answers.append(
                {
                    "question": case["question"],
                    "citation_count": len(result.get("citations", [])),
                    "confidence": result.get("confidence"),
                    "answer_preview": result.get("answer", "")[:500],
                }
            )
        return {"upload": upload_result, "answers": answers}
    finally:
        for handle in handles:
            handle.close()
        try:
            requests.delete(f"{api_url}/documents/{session_id}", timeout=60)
        except Exception:
            pass


def poll(api_url: str, job_id: str) -> dict:
    while True:
        response = requests.get(f"{api_url}/jobs/{job_id}", timeout=20)
        response.raise_for_status()
        job = response.json()
        print(f"{job['phase']}: {job.get('detail', '')}")
        if job["status"] == "complete":
            return job["result"]
        if job["status"] == "failed":
            raise RuntimeError(job.get("error"))
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="", help="Optional running API URL, e.g. http://localhost:8000")
    args = parser.parse_args()
    directory = ROOT / "data" / "public_test_pdfs"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"local_retrieval": local_retrieval_test(directory)}
    if args.api:
        report["api_end_to_end"] = api_test(directory, args.api.rstrip("/"))
    output = ROOT / "public_pdf_test_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")
    if not report["local_retrieval"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
