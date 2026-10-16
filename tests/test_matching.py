import fitz
import pytest
from fastapi.testclient import TestClient

import main


JOB = """Job Title: Software Engineer

Required Skills:
Python, SQL, AWS

Preferred Skills:
React, Kubernetes

At least 3 years of experience. Excellent digital communication."""


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


# --- Skill matching -------------------------------------------------------

def test_has_skill_matches_whole_words_only():
    assert main.has_skill("I use Java daily", "java")
    assert not main.has_skill("I use JavaScript daily", "java")
    assert not main.has_skill("digital marketing", "git")
    assert not main.has_skill("excellent communicator", "excel")


def test_has_skill_handles_symbols():
    assert main.has_skill("Wrote C++ and C# services", "c++")
    assert main.has_skill("Wrote C++ and C# services", "c#")
    assert main.has_skill("APIs in Node.js", "node.js")


def test_classify_jd_skills_splits_required_and_preferred():
    required, preferred, other = main.classify_jd_skills(JOB)

    assert required == ["python", "sql", "aws"]
    assert sorted(preferred) == ["kubernetes", "react"]
    # "excellent" and "digital" must not be read as Excel and Git
    assert other == []


# --- Extraction -----------------------------------------------------------

def test_extract_required_experience():
    assert main.extract_required_experience(JOB) == 3
    assert main.extract_required_experience("Minimum of 5 years") == 5
    assert main.extract_required_experience("No experience needed") == 0


def test_extract_target_job_title():
    assert main.extract_target_job_title(JOB) == "Software Engineer"
    assert main.extract_target_job_title("Position: data analyst\n...") == "Data Analyst"


def test_extract_education():
    text = "Bachelor of Science in Computer Science"
    assert main.extract_education(text) == ["Bachelor's"]


def test_extract_certifications():
    text = "AWS Certified Solutions Architect, CompTIA Security+"
    assert main.extract_certifications(text) == ["AWS Certified", "CompTIA Security+"]


def test_title_match_ignores_seniority():
    assert main.calculate_title_match("Software Engineer", "Senior Software Engineer") == 100
    assert main.calculate_title_match("Software Engineer", "Software Developer") == 100
    assert main.calculate_title_match("Software Engineer", "Chef") == 0


# --- Scoring --------------------------------------------------------------

def test_strong_resume_scores_higher_than_weak_resume():
    job = main.analyze_job_description(JOB)

    strong = main.parse_resume(
        "Software Engineer\n"
        "5 years of experience. Built APIs with Python and SQL, deployed on AWS. "
        "Developed dashboards in React. Bachelor of Science.",
        "strong.pdf",
        "TEST"
    )
    weak = main.parse_resume(
        "Chef\nCooked meals for 2 years.",
        "weak.pdf",
        "TEST"
    )

    strong_result = main.score_resume(strong, job)
    weak_result = main.score_resume(weak, job)

    assert strong_result["required_missing"] == []
    assert strong_result["experience_met"]
    assert strong_result["fit_score"] > weak_result["fit_score"]


def test_fit_score_is_between_0_and_100():
    job = main.analyze_job_description(JOB)
    resume = main.parse_resume("Python SQL AWS React Kubernetes " * 20, "x.pdf", "TEST")

    score = main.score_resume(resume, job)["fit_score"]

    assert 0 <= score <= 100


def test_suggestions_mention_missing_required_skill():
    job = main.analyze_job_description(JOB)
    resume = main.parse_resume("Software Engineer\nPython and SQL", "x.pdf", "TEST")
    result = main.score_resume(resume, job)

    suggestions = main.build_suggestions(result, job, resume["text"])

    assert any("AWS" in s for s in suggestions)


# --- API ------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    # Skip loading the real dataset; use two small fake resumes instead
    monkeypatch.setattr(main, "build_resume_cache", lambda: None)
    monkeypatch.setattr(main, "RESUME_CACHE", [
        main.parse_resume("Software Engineer\nPython SQL AWS, 4 years of experience", "a.pdf", "IT"),
        main.parse_resume("Chef\nCooking", "b.pdf", "CHEF"),
    ])

    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_match_job_ranks_relevant_resume_first(client):
    response = client.post("/match-job", json={"job_description": JOB})

    assert response.status_code == 200
    assert response.json()["top_candidates"][0]["file"] == "a.pdf"


def test_analyze_resume_returns_score_and_suggestions(client):
    pdf = make_pdf("Software Engineer\nBuilt services with Python and SQL.")

    response = client.post(
        "/analyze-resume",
        files={"resume": ("me.pdf", pdf, "application/pdf")},
        data={"job_description": JOB}
    )

    assert response.status_code == 200
    data = response.json()
    assert "aws" in data["result"]["required_missing"]
    assert data["suggestions"]
    assert data["out_of"] == 3


def test_analyze_resume_rejects_non_pdf(client):
    response = client.post(
        "/analyze-resume",
        files={"resume": ("me.txt", b"hello", "text/plain")},
        data={"job_description": JOB}
    )

    assert response.status_code == 400
