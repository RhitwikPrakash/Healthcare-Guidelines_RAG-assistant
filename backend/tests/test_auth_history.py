import time
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import _needs_history, _result_quality, app
from app.services.answer_engine import _answer_prompt, _repair_prompt
from app.services.chat_store import chat_store
from app.services.job_manager import JobManager

def _register(client: TestClient, prefix: str) -> dict:
    unique = uuid4().hex
    response = client.post(
        "/auth/register",
        json={
            "email": f"{prefix}.{unique}@example.com",
            "display_name": f"{prefix.title()} User",
            "password": "strong-password-123",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()

def _trace(question: str, *, semantic_passed: bool = True) -> dict:
    return {
        "subqueries": [question],
        "citation_coverage": 1.0,
        "citation_support_ratio": 1.0,
        "semantic_verifier": {
            "available": True,
            "passed": semantic_passed,
            "aligned": semantic_passed,
            "complete": semantic_passed,
            "unrelated_sections": [] if semantic_passed else ["Unrelated material"],
            "missing_requirements": [],
            "score": 0.95 if semantic_passed else 0.2,
        },
    }

def test_authenticated_history_is_user_specific() -> None:
    with TestClient(app) as client:
        first = _register(client, "first")
        first_headers = {"Authorization": f"Bearer {first['access_token']}"}
        created = client.post("/conversations", json={"title": "Private guideline review"}, headers=first_headers)
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        assert client.get(f"/conversations/{conversation_id}/messages", headers=first_headers).json() == []

        second = _register(client, "second")
        second_headers = {"Authorization": f"Bearer {second['access_token']}"}
        forbidden = client.get(f"/conversations/{conversation_id}/messages", headers=second_headers)
        assert forbidden.status_code == 404


def test_history_is_used_only_for_real_followups() -> None:
    assert not _needs_history("Compare barriers in Step 1, Step 2, and Step 3 of early diagnosis")
    assert _needs_history("How does that compare with screening?")
    assert _needs_history("Why so?")


def test_generation_prompt_never_contains_a_previous_assistant_answer() -> None:
    messages = _answer_prompt(
        "How does that compare with screening?",
        "[S1] Screening uses a target population.",
        "What is early diagnosis?",
        "auto",
    )
    prompt_text = "\n".join(message["content"] for message in messages)
    assert "What is early diagnosis?" in prompt_text
    assert "previous assistant" not in prompt_text.casefold()
    assert "CURRENT QUESTION" in prompt_text


def test_repair_prompt_is_bound_to_current_question() -> None:
    messages = _repair_prompt(
        "What numerical survival benefit is reported?",
        "An unrelated screening answer. [S1]",
        "[S1] A short delay was associated with 7% greater survival.",
    )
    prompt_text = "\n".join(message["content"] for message in messages)
    assert "What numerical survival benefit is reported?" in prompt_text
    assert "Remove every unrelated section" in prompt_text


def test_final_quality_gate_rejects_an_answer_to_a_previous_topic() -> None:
    question = "Compare the main barriers at Step 1, Step 2, and Step 3 of early diagnosis."
    wrong_result = {
        "answer": "Breast cancer warning symptoms include a breast lump and nipple discharge. [S1]",
        "citations": [{"id": "S1"}],
        "trace": _trace(question),
    }
    quality = _result_quality(question, wrong_result)
    assert quality["passed"] is False
    assert quality["intent_passed"] is False


def test_final_quality_gate_rejects_appended_previous_answer() -> None:
    question = "What evidence shows that reducing diagnostic delay improves cancer survival? Include numerical findings."
    previous = (
        "Countries with weak health systems should strengthen early diagnosis before screening because screening "
        "requires additional infrastructure, population invitations, follow-up and treatment capacity. [S2] " * 4
    )
    contaminated = {
        "answer": (
            "Patients with delays under three months had better survival than those delayed longer. [S1]\n\n"
            + previous
        ),
        "citations": [{"id": "S1"}, {"id": "S2"}],
        "trace": _trace(question),
    }
    quality = _result_quality(question, contaminated, [previous])
    assert quality["passed"] is False
    assert quality["previous_answer_contamination_passed"] is False


def test_numerical_request_requires_numerical_content() -> None:
    question = "What numerical findings show that reducing diagnostic delay improves survival?"
    result = {
        "answer": "Shorter delay was associated with better survival. [S1]",
        "citations": [{"id": "S1"}],
        "trace": _trace(question),
    }
    quality = _result_quality(question, result)
    assert quality["passed"] is False
    assert quality["numerical_requested"] is True
    assert quality["numerical_passed"] is False


def test_semantic_verifier_failure_is_a_hard_rejection() -> None:
    question = "What numerical survival benefit is reported?"
    result = {
        "answer": "A 7% greater likelihood of survival was reported. [S1]",
        "citations": [{"id": "S1"}],
        "trace": _trace(question, semantic_passed=False),
    }
    quality = _result_quality(question, result)
    assert quality["passed"] is False
    assert quality["semantic_verifier_passed"] is False


def test_final_quality_gate_accepts_a_relevant_cited_answer() -> None:
    question = "Compare the main barriers at Step 1, Step 2, and Step 3 of early diagnosis."
    correct_result = {
        "answer": (
            "Step 1 barriers include poor health literacy, cancer stigma, and limited access to primary care. [S1]\n\n"
            "Step 2 barriers include inaccurate clinical assessment, inaccessible pathology and staging, and poor coordination. [S2]\n\n"
            "Step 3 barriers are mainly financial, geographic, logistical, and sociocultural obstacles to treatment. [S3]"
        ),
        "citations": [{"id": "S1"}, {"id": "S2"}, {"id": "S3"}],
        "trace": _trace(question),
    }
    quality = _result_quality(question, correct_result)
    assert quality["passed"] is True


def test_job_result_keeps_exact_request_and_document_binding() -> None:
    manager = JobManager(max_workers=1)
    binding = {
        "request_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "user_message_id": str(uuid4()),
        "question_hash": "a" * 64,
        "document_set_hash": "d" * 64,
    }
    job_id = manager.create("query", ["Answer"], owner_id="user-1", binding=binding)
    manager.complete(job_id, {"answer": "Bound response"})
    job = manager.get(job_id, owner_id="user-1")
    assert job is not None
    for key, value in binding.items():
        assert job[key] == value
        assert job["result"][key] == value


def test_only_previous_user_question_can_be_used_for_reference_resolution() -> None:
    with TestClient(app) as client:
        token_payload = _register(client, "history")
        user_id = token_payload["user"]["id"]
        conversation = chat_store.create_conversation(user_id, "Bound history")
        conversation_id = conversation["id"]

        request_id = str(uuid4())
        question_hash = "b" * 64
        user_message = chat_store.add_message(
            user_id,
            conversation_id,
            "user",
            "What are the three early diagnosis steps?",
            {"request_id": request_id, "question_hash": question_hash, "document_set_hash": "e" * 64},
        )
        chat_store.add_bound_assistant_message(
            user_id,
            conversation_id,
            "Awareness, diagnosis and staging, and access to treatment. [S1]",
            {"quality": {"passed": True}},
            parent_message_id=user_message["id"],
            request_id=request_id,
            question_hash=question_hash,
        )

        assert chat_store.previous_user_question(user_id, conversation_id) == "What are the three early diagnosis steps?"
        history = chat_store.recent_history(user_id, conversation_id, limit=8)
        assert history == [{"role": "user", "content": "What are the three early diagnosis steps?"}]
        assert "Awareness" not in str(history)


def test_bound_assistant_save_is_atomic_and_idempotent() -> None:
    with TestClient(app) as client:
        token_payload = _register(client, "atomic")
        user_id = token_payload["user"]["id"]
        conversation_id = chat_store.create_conversation(user_id, "Atomic test")["id"]
        request_id = str(uuid4())
        question_hash = "c" * 64
        user_message = chat_store.add_message(
            user_id,
            conversation_id,
            "user",
            "What is the target interval?",
            {"request_id": request_id, "question_hash": question_hash},
        )
        first = chat_store.add_bound_assistant_message(
            user_id,
            conversation_id,
            "The target interval is under 90 days. [S1]",
            {"quality": {"passed": True}},
            parent_message_id=user_message["id"],
            request_id=request_id,
            question_hash=question_hash,
        )
        second = chat_store.add_bound_assistant_message(
            user_id,
            conversation_id,
            "A duplicate answer that must not be inserted. [S1]",
            {"quality": {"passed": True}},
            parent_message_id=user_message["id"],
            request_id=request_id,
            question_hash=question_hash,
        )
        assert first["id"] == second["id"]
        assert [item["role"] for item in chat_store.list_messages(user_id, conversation_id)] == ["user", "assistant"]


def test_query_binding_retry_idempotency_and_stateless_retry(monkeypatch) -> None:
    import app.main as main_module

    calls: list[dict] = []
    question = "Compare the main barriers at Step 1, Step 2, and Step 3 of early diagnosis."

    def fake_answer_job(job_id, session_id, submitted_question, mode, followup_question):
        calls.append({"question": submitted_question, "mode": mode, "followup_question": followup_question})
        if len(calls) == 1:
            return {
                "answer": "Breast cancer warning symptoms include a breast lump and nipple discharge. [S1]",
                "citations": [{"id": "S1"}],
                "confidence": {"label": "moderate", "score": 0.7},
                "trace": _trace(question),
            }
        return {
            "answer": (
                "Step 1 barriers include poor health literacy, cancer stigma, and limited primary-care access. [S1]\n\n"
                "Step 2 barriers include inaccurate assessment, unavailable pathology and staging, and poor coordination. [S2]\n\n"
                "Step 3 barriers include financial, geographic, logistical, and sociocultural obstacles to treatment. [S3]"
            ),
            "citations": [{"id": "S1"}, {"id": "S2"}, {"id": "S3"}],
            "confidence": {"label": "high", "score": 0.9},
            "trace": _trace(question),
        }

    document = {"file_name": "guide.pdf", "sha256": "f" * 64, "pages": 48, "chunks": 108}
    monkeypatch.setattr(main_module.store, "list_documents", lambda _: [document])
    monkeypatch.setattr(main_module, "answer_job", fake_answer_job)

    with TestClient(app) as client:
        token_payload = _register(client, "binding")
        headers = {"Authorization": f"Bearer {token_payload['access_token']}"}
        conversation = client.post("/conversations", json={"title": "Binding test"}, headers=headers).json()
        request_id = str(uuid4())
        payload = {"session_id": conversation["id"], "request_id": request_id, "question": question, "mode": "fast"}
        created = client.post("/query", json=payload, headers=headers)
        assert created.status_code == 202, created.text
        created_payload = created.json()
        assert created_payload["document_set_hash"]

        job = None
        for _ in range(100):
            response = client.get(f"/jobs/{created_payload['job_id']}", headers=headers)
            assert response.status_code == 200, response.text
            job = response.json()
            if job["status"] in {"complete", "failed"}:
                break
            time.sleep(0.02)
        assert job is not None
        assert job["status"] == "complete", job
        result = job["result"]
        for key in ("request_id", "conversation_id", "user_message_id", "question_hash", "document_set_hash"):
            assert result[key] == created_payload[key]
        assert result["quality"]["passed"] is True
        assert result["quality"]["retry_used"] is True
        assert len(calls) == 2
        assert calls[1]["followup_question"] is None
        assert calls[1]["mode"] == "auto"

        messages = client.get(f"/conversations/{conversation['id']}/messages", headers=headers).json()
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[1]["metadata"]["in_reply_to_message_id"] == messages[0]["id"]
        assert messages[1]["metadata"]["request_id"] == request_id
        assert messages[1]["metadata"]["document_set_hash"] == created_payload["document_set_hash"]

        repeated = client.post("/query", json=payload, headers=headers)
        assert repeated.status_code == 202
        assert repeated.json()["job_id"] == created_payload["job_id"]
        assert len(client.get(f"/conversations/{conversation['id']}/messages", headers=headers).json()) == 2