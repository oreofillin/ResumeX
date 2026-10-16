from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import fitz
import glob
import os
import re
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel


class JobRequest(BaseModel):
    job_description: str

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_resume_cache()
    yield

app = FastAPI(lifespan=lifespan)

DATASET_PATH = os.environ.get(
    "RESUMEX_DATASET_PATH",
    os.path.expanduser(
        "~/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/data/data"
    )
)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
KNOWN_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "react",
    "angular",
    "node.js",
    "fastapi",
    "django",
    "flask",
    "spring boot",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "power bi",
    "tableau",
    "excel"
]
RESUME_CACHE = []

def extract_years_experience(text: str) -> float:
    text_lower = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(?:over|more than|greater than|approximately|about)\s+(\d+(?:\.\d+)?)\s*\+?\s*years?"
    ]

    years_found = []

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)

        for match in matches:
            try:
                years = float(match)

                if 0 < years <= 50:
                    years_found.append(years)

            except ValueError:
                pass

    if not years_found:
        return 0

    return max(years_found)


def extract_employers(text: str) -> list:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    employers = []

    # Words commonly found in organization names
    org_keywords = [
        "inc", "inc.",
        "llc",
        "ltd", "ltd.",
        "corp", "corp.",
        "corporation",
        "technologies",
        "technology",
        "solutions",
        "systems",
        "services",
        "group",
        "bank",
        "labs",
        "consulting",
        "associates"
    ]

    # Dataset placeholders and section headings to ignore
    reject_phrases = [
        "company name",
        "city, state",
        "city , state",
        "responsibilities",
        "professional summary",
        "summary",
        "education",
        "skills",
        "technical skills",
        "certifications",
        "projects"
    ]

    # Usually indicates a responsibility sentence, not an employer
    action_words = [
        "managed",
        "developed",
        "created",
        "implemented",
        "maintained",
        "reconciled",
        "worked",
        "performed",
        "prepared",
        "assisted",
        "responsible",
        "communicated",
        "consulted",
        "reviewed"
    ]

    for line in lines:
        line_lower = line.lower()

        # Employer names should normally be short
        if len(line) < 3 or len(line) > 70:
            continue

        # Ignore anonymized placeholders and headings
        if any(
            phrase in line_lower
            for phrase in reject_phrases
        ):
            continue

        # Ignore long sentence-like lines
        if len(line.split()) > 10:
            continue

        # Ignore responsibility statements
        if any(
            line_lower.startswith(action)
            for action in action_words
        ):
            continue

        # Keep education institutions out of employer extraction
        if any(
            word in line_lower
            for word in [
                "university",
                "college",
                "school",
                "institute"
            ]
        ):
            continue

        # Check whether line looks like an organization
        has_org_keyword = any(
            re.search(
                rf"(?<!\w){re.escape(keyword)}(?!\w)",
                line_lower
            )
            for keyword in org_keywords
        )

        if not has_org_keyword:
            continue

        # Remove month names
        cleaned = re.sub(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
            r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
            "",
            line,
            flags=re.IGNORECASE
        )

        # Remove years
        cleaned = re.sub(
            r"\b(?:19|20)\d{2}\b",
            "",
            cleaned
        )

        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" ,-–|")

        if (
            cleaned
            and cleaned.lower() != "company"
            and cleaned not in employers
        ):
            employers.append(cleaned)

    return employers[:10]

def extract_certifications(text: str) -> list:
    text_lower = text.lower()

    certification_patterns = {
        "AWS Certified": [
            r"\baws certified\b",
            r"\bamazon web services certified\b"
        ],
        "CompTIA A+": [
            r"\bcomptia a\+(?!\w)"
        ],
        "CompTIA Network+": [
            r"\bcomptia network\+(?!\w)"
        ],
        "CompTIA Security+": [
            r"\bcomptia security\+(?!\w)"
        ],
        "CISSP": [
            r"\bcissp\b",
            r"\bcertified information systems security professional\b"
        ],
        "PMP": [
            r"\bpmp\b",
            r"\bproject management professional\b"
        ],
        "CCNA": [
            r"\bccna\b",
            r"\bcisco certified network associate\b"
        ],
        "CCNP": [
            r"\bccnp\b",
            r"\bcisco certified network professional\b"
        ],
        "Azure Certified": [
            r"\bmicrosoft certified.*azure\b",
            r"\bazure certified\b"
        ],
        "Google Cloud Certified": [
            r"\bgoogle cloud certified\b",
            r"\bgcp certified\b"
        ],
        "CEH": [
            r"\bceh\b",
            r"\bcertified ethical hacker\b"
        ],
        "Scrum Master": [
            r"\bcertified scrum master\b",
            r"\bcsm\b"
        ]
    }

    certifications = []

    for certification, patterns in certification_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                certifications.append(certification)
                break

    return certifications

def extract_projects(text: str) -> list:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    projects = []

    project_headers = [
        "projects",
        "project experience",
        "academic projects",
        "personal projects",
        "technical projects",
        "selected projects"
    ]

    start_index = None

    # Find a project section
    for i, line in enumerate(lines):
        line_lower = line.lower().strip(" :-")

        if line_lower in project_headers:
            start_index = i + 1
            break

    if start_index is None:
        return []

    # Common headings that indicate the project section has ended
    stop_headers = [
        "education",
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "skills",
        "technical skills",
        "certifications",
        "awards",
        "interests",
        "references"
    ]

    for line in lines[start_index:]:
        line_lower = line.lower().strip(" :-")

        if line_lower in stop_headers:
            break

        # Ignore long description/bullet lines
        if len(line) > 100:
            continue

        # Likely project title
        if 2 <= len(line.split()) <= 12:
            cleaned = line.strip("•-–| ")

            if cleaned and cleaned not in projects:
                projects.append(cleaned)

        if len(projects) == 5:
            break

    return projects

def extract_education(text: str) -> list:
    text_lower = text.lower()

    degree_patterns = {
        "PhD": [
            r"\bph\.?d\.?\b",
            r"\bdoctor of philosophy\b",
            r"\bdoctorate\b"
        ],
        "Master's": [
            r"\bmaster'?s\b",
            r"\bmaster of science\b",
            r"\bmaster of arts\b",
            r"\bm\.?s\.?\b",
            r"\bm\.?sc\.?\b",
            r"\bmba\b",
            r"\bm\.?b\.?a\.?\b"
        ],
        "Bachelor's": [
            r"\bbachelor'?s\b",
            r"\bbachelor of science\b",
            r"\bbachelor of arts\b",
            r"\bbachelor of technology\b",
            r"\bb\.?s\.?\b",
            r"\bb\.?sc\.?\b",
            r"\bb\.?a\.?\b",
            r"\bb\.?tech\b"
        ],
        "Associate": [
            r"\bassociate'?s\b",
            r"\bassociate of science\b",
            r"\bassociate of arts\b"
        ]
    }

    education = []

    for degree, patterns in degree_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                education.append(degree)
                break

    return education

def extract_required_experience(job_description: str) -> float:
    text = job_description.lower()

    patterns = [
        r"at least\s+(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s+years?\s+of\s+experience"
    ]

    years_found = []

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                years = float(match)

                if 0 < years <= 50:
                    years_found.append(years)

            except ValueError:
                pass

    if not years_found:
        return 0

    return max(years_found)

def extract_pdf_text(doc) -> str:
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


def parse_resume(text: str, file: str, category: str) -> dict:
    return {
        "file": file,
        "category": category,
        "text": text,
        "text_lower": text.lower(),
        "job_title": extract_job_title(text),
        "years_experience": extract_years_experience(text),
        "education": extract_education(text),
        "certifications": extract_certifications(text)
    }


def build_resume_cache():
    global RESUME_CACHE

    print("Building resume cache...")

    pdfs = glob.glob(os.path.join(DATASET_PATH, "*", "*.pdf"))
    cache = []

    for pdf in pdfs:
        try:
            text = extract_pdf_text(fitz.open(pdf))

            cache.append(parse_resume(
                text,
                os.path.basename(pdf),
                os.path.basename(os.path.dirname(pdf))
            ))

        except Exception as e:
            print(f"CACHE ERROR {pdf}: {e}")

    RESUME_CACHE = cache

    print(f"Resume cache ready: {len(RESUME_CACHE)} resumes")

def count_skill_occurrences(text: str, skill: str) -> int:
    pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def has_skill(text: str, skill: str) -> bool:
    # Whole-word match so "java" doesn't match "javascript"
    # and "git" doesn't match "digital"
    pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/resumes")
def resumes():
    categories = {}

    for resume in RESUME_CACHE:
        categories[resume["category"]] = categories.get(resume["category"], 0) + 1

    return {
        "count": len(RESUME_CACHE),
        "categories": dict(sorted(categories.items()))
    }

def extract_job_title(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return "Unknown"

    for line in lines[:10]:
        if 2 <= len(line.split()) <= 10 and len(line) <= 100:
            title = line
            break
    else:
        title = lines[0][:100]

    # Remove common date ranges from titles
    title = re.sub(
        r"\b\d{1,2}/\d{4}\s*(?:to|-|–|—)\s*(?:current|present|\d{1,2}/\d{4})\b",
        "",
        title,
        flags=re.IGNORECASE
    )

    title = re.sub(
        r"\b\d{4}\s*(?:to|-|–|—)\s*(?:current|present|\d{4})\b",
        "",
        title,
        flags=re.IGNORECASE
    )

    title = re.sub(r"\s*,\s*$", "", title)
    title = re.sub(r"\s{2,}", " ", title).strip()

    return title.title()
@app.post("/extract-skills")
def extract_skills(request: JobRequest):
    job_description = request.job_description

    extracted_skills = []

    for skill in KNOWN_SKILLS:
        if has_skill(job_description, skill):
            extracted_skills.append(skill)

    return {
        "extracted_skills": extracted_skills,
        "count": len(extracted_skills)
    }

def extract_target_job_title(job_description: str) -> str:
    lines = [
        line.strip()
        for line in job_description.splitlines()
        if line.strip()
    ]

    if not lines:
        return "Unknown"

    # Look for an explicit title
    for line in lines[:10]:
        lower = line.lower()

        if lower.startswith("job title:"):
            return line.split(":", 1)[1].strip().title()

        if lower.startswith("position:"):
            return line.split(":", 1)[1].strip().title()

        if lower.startswith("role:"):
            return line.split(":", 1)[1].strip().title()

    # Otherwise assume first short line is the title
    first = lines[0]

    if len(first.split()) <= 8 and len(first) <= 80:
        return first.title()

    return "Unknown"

def classify_jd_skills(job_description: str):
    jd_lower = job_description.lower()

    required_skills = []
    preferred_skills = []
    other_skills = []

    required_section = ""
    preferred_section = ""

    required_match = re.search(
        r"required\s+skills?\s*:?",
        jd_lower
    )

    preferred_match = re.search(
        r"preferred\s+skills?\s*:?",
        jd_lower
    )

    if required_match and preferred_match:

        if required_match.start() < preferred_match.start():
            required_section = jd_lower[
                required_match.end():preferred_match.start()
            ]

            preferred_section = jd_lower[
                preferred_match.end():
            ]

        else:
            preferred_section = jd_lower[
                preferred_match.end():required_match.start()
            ]

            required_section = jd_lower[
                required_match.end():
            ]

    elif required_match:
        required_section = jd_lower[required_match.end():]

    elif preferred_match:
        preferred_section = jd_lower[preferred_match.end():]

    for skill in KNOWN_SKILLS:

        if has_skill(required_section, skill):
            required_skills.append(skill)

        elif has_skill(preferred_section, skill):
            preferred_skills.append(skill)

        elif has_skill(jd_lower, skill):
            other_skills.append(skill)

    return required_skills, preferred_skills, other_skills

def calculate_title_match(target_title: str, candidate_title: str) -> int:
    if not target_title or not candidate_title:
        return 0

    if target_title == "Unknown" or candidate_title == "Unknown":
        return 0

    def normalize_title_words(title: str):
        words = re.findall(r"\b[a-zA-Z]+\b", title.lower())

        aliases = {
            "engineering": "engineer",
            "engineered": "engineer",
            "developer": "engineer",
            "development": "engineer",
            "dev": "engineer",
            "programmer": "engineer"
        }

        ignore_words = {
            "senior", "junior", "sr", "jr",
            "lead", "principal", "staff",
            "intern", "internship", "co", "op"
        }

        normalized = set()

        for word in words:
            if word in ignore_words:
                continue

            normalized.add(aliases.get(word, word))

        return normalized

    target_words = normalize_title_words(target_title)
    candidate_words = normalize_title_words(candidate_title)

    if not target_words or not candidate_words:
        return 0

    common_words = target_words & candidate_words

    return round(
        (len(common_words) / len(target_words)) * 100
    )

def calculate_fit_score(
    required_skills,
    preferred_skills,
    other_skills,
    required_matched,
    preferred_matched,
    other_matched,
    title_match,
    target_job_title,
    candidate_experience,
    required_experience
):
    earned = 0.0
    possible = 0.0

    # -------------------------
    # Required skills: 45 pts
    # -------------------------
    if required_skills:
        required_ratio = (
            len(required_matched) / len(required_skills)
        )

        earned += required_ratio * 45
        possible += 45

    # -------------------------
    # Preferred skills: 15 pts
    # -------------------------
    if preferred_skills:
        preferred_ratio = (
            len(preferred_matched) / len(preferred_skills)
        )

        earned += preferred_ratio * 15
        possible += 15

    # -------------------------
    # Other JD skills: 5 pts
    # -------------------------
    if other_skills:
        other_ratio = (
            len(other_matched) / len(other_skills)
        )

        earned += other_ratio * 5
        possible += 5

    # -------------------------
    # Job title relevance: 20 pts
    # -------------------------
    if target_job_title and target_job_title != "Unknown":
        earned += (title_match / 100) * 20
        possible += 20

    # -------------------------
    # Experience: 15 pts
    # -------------------------
    if required_experience > 0:
        experience_ratio = min(
            candidate_experience / required_experience,
            1.0
        )

        earned += experience_ratio * 15
        possible += 15

    # Normalize applicable criteria to 0-100
    if possible == 0:
        return 0.0

    fit_score = (earned / possible) * 100

    return round(
        min(max(fit_score, 0), 100),
        1
    )

ACTION_WORDS = [
    "developed",
    "built",
    "implemented",
    "deployed",
    "managed",
    "designed",
    "created",
    "configured",
    "administered",
    "maintained",
    "worked with",
    "used",
    "integrated"
]


def analyze_job_description(job_description: str) -> dict:
    required_skills, preferred_skills, other_skills = classify_jd_skills(job_description)

    return {
        "target_job_title": extract_target_job_title(job_description),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "other_skills": other_skills,
        "skills": required_skills + preferred_skills + other_skills,
        "required_experience": extract_required_experience(job_description)
    }


def score_resume(resume: dict, job: dict) -> dict:
    text = resume["text"]

    required_skills = job["required_skills"]
    preferred_skills = job["preferred_skills"]
    other_skills = job["other_skills"]
    skills = job["skills"]
    required_experience = job["required_experience"]
    target_job_title = job["target_job_title"]

    candidate_experience = resume["years_experience"]

    experience_met = (
        required_experience == 0
        or candidate_experience >= required_experience
    )
    experience_score = 0

    if required_experience > 0:
        if candidate_experience >= required_experience:
            experience_score = 30
        elif candidate_experience > 0:
            experience_score = round(
                30 * (candidate_experience / required_experience)
            )

    skill_scores = {}
    context_scores = {}

    for skill in skills:
        count = count_skill_occurrences(text, skill)
        skill_scores[skill] = count

        context_hits = 0

        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            context_start = max(0, match.start() - 80)
            context_end = min(len(text), match.end() + 80)

            context = text[context_start:context_end].lower()

            if any(action in context for action in ACTION_WORDS):
                context_hits += 1

        context_scores[skill] = context_hits

    matched_skills = [
        skill for skill, count in skill_scores.items()
        if count > 0
    ]

    missing_skills = [
        skill for skill in skills
        if skill not in matched_skills
    ]

    required_matched = [
        skill for skill in required_skills
        if skill in matched_skills
    ]

    preferred_matched = [
        skill for skill in preferred_skills
        if skill in matched_skills
    ]

    other_matched = [
        skill for skill in other_skills
        if skill in matched_skills
    ]

    mention_score = sum(
        min(count, 5)
        for count in skill_scores.values()
    )

    coverage_score = (
        len(required_matched) * 40
        + len(preferred_matched) * 20
        + len(other_matched) * 10
    )

    context_score = sum(
        min(hits, 3) * 10
        for hits in context_scores.values()
    )

    required_bonus = 0
    if required_skills and len(required_matched) == len(required_skills):
        required_bonus = 50

    weighted_possible = (
        len(required_skills) * 3
        + len(preferred_skills) * 1
        + len(other_skills) * 1
    )

    weighted_earned = (
        len(required_matched) * 3
        + len(preferred_matched) * 1
        + len(other_matched) * 1
    )

    match_percentage = 0
    if weighted_possible > 0:
        match_percentage = round(
            (weighted_earned / weighted_possible) * 100,
            1
        )

    title_match = calculate_title_match(
        target_job_title,
        resume["job_title"]
    )

    title_score = round(title_match * 0.5)

    total_score = (
        mention_score
        + coverage_score
        + context_score
        + required_bonus
        + title_score
        + title_match
        + experience_score
    )

    fit_score = calculate_fit_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        other_skills=other_skills,
        required_matched=required_matched,
        preferred_matched=preferred_matched,
        other_matched=other_matched,
        title_match=title_match,
        target_job_title=target_job_title,
        candidate_experience=candidate_experience,
        required_experience=required_experience
    )

    return {
        "file": resume["file"],
        "category": resume["category"],
        "job_title": resume["job_title"],
        "title_match": title_match,
        "score": total_score,
        "match_percentage": match_percentage,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "required_experience": required_experience,
        "years_experience": candidate_experience,
        "experience_met": experience_met,
        "experience_score": experience_score,
        "education": resume["education"],
        "certifications": resume["certifications"],
        "fit_score": fit_score,

        "required_matched": required_matched,
        "preferred_matched": preferred_matched,
        "required_missing": [
            skill for skill in required_skills
            if skill not in matched_skills
        ],
        "preferred_missing": [
            skill for skill in preferred_skills
            if skill not in matched_skills
        ],

        "skill_scores": skill_scores,
        "context_scores": context_scores,
        "preview": " ".join(text.split())[:300]
    }


SKILL_LABELS = {
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "aws": "AWS",
    "gcp": "GCP",
    "node.js": "Node.js",
    "fastapi": "FastAPI",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "c++": "C++",
    "c#": "C#",
    "power bi": "Power BI"
}


def display_skill(skill: str) -> str:
    return SKILL_LABELS.get(skill, skill.title())


def build_suggestions(result: dict, job: dict, text: str) -> list:
    suggestions = []

    for skill in result["required_missing"]:
        suggestions.append(
            f"The job requires {display_skill(skill)}, but it isn't on your resume. "
            f"If you have this skill, add it with an example of how you used it."
        )

    missing_preferred = [display_skill(s) for s in result["preferred_missing"]]

    if missing_preferred:
        suggestions.append(
            "Preferred skills you could add: " + ", ".join(missing_preferred) + "."
        )

    weak_skills = [
        display_skill(skill) for skill in result["matched_skills"]
        if result["context_scores"].get(skill, 0) == 0
    ]

    if weak_skills:
        pronoun = "it" if len(weak_skills) == 1 else "them"
        suggestions.append(
            "You list " + ", ".join(weak_skills) + f" but never show using {pronoun}. "
            "Add bullet points like \"Built ... using " + weak_skills[0] + "\"."
        )

    if not result["experience_met"]:
        if result["years_experience"] == 0:
            suggestions.append(
                f"The job asks for {result['required_experience']:g}+ years of experience. "
                f"State your total years of experience clearly (e.g. \"3 years of experience in ...\")."
            )
        else:
            suggestions.append(
                f"The job asks for {result['required_experience']:g}+ years of experience "
                f"and your resume shows {result['years_experience']:g}. "
                f"Highlight internships, projects or freelance work that count toward it."
            )

    if (
        job["target_job_title"] != "Unknown"
        and result["title_match"] < 50
    ):
        suggestions.append(
            f"Your headline doesn't match the role \"{job['target_job_title']}\". "
            f"Consider a summary line at the top that names the role you're targeting."
        )

    if not re.search(r"\d+\s*%|\$\s*\d|\b\d+\s*(?:users|customers|clients|projects)\b", text, re.IGNORECASE):
        suggestions.append(
            "Add measurable results, such as \"cut load time by 40%\" or \"served 2,000 users\"."
        )

    if not result["education"]:
        suggestions.append(
            "No degree was detected. Make sure your Education section names your degree clearly."
        )

    return suggestions


@app.post("/match-job")
def match_job(request: JobRequest):
    job = analyze_job_description(request.job_description)

    results = []

    for resume in RESUME_CACHE:
        try:
            result = score_resume(resume, job)

            if result["score"] > 0:
                results.append(result)

        except Exception as e:
            print(f"ERROR PROCESSING {resume['file']}: {e}")

    results = sorted(
        results,
        key=lambda x: (
            x["fit_score"],
            len(x["required_matched"]),
            x["title_match"],
            x["score"]
        ),
        reverse=True
    )[:20]

    return {
        "target_job_title": job["target_job_title"],
        "extracted_skills": job["skills"],
        "required_skills": job["required_skills"],
        "preferred_skills": job["preferred_skills"],
        "other_skills": job["other_skills"],
        "skill_count": len(job["skills"]),
        "top_candidates": results,
        "count": len(results)
    }


@app.post("/analyze-resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Please paste a job description.")

    data = await resume.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Resume must be smaller than 5 MB.")

    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        text = extract_pdf_text(fitz.open(stream=data, filetype="pdf"))
    except Exception:
        raise HTTPException(status_code=400, detail="That PDF couldn't be read.")

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text found in that PDF. Scanned image resumes aren't supported yet."
        )

    parsed = parse_resume(text, resume.filename or "resume.pdf", "Uploaded")
    job = analyze_job_description(job_description)
    result = score_resume(parsed, job)

    # Where this resume would rank against the dataset for the same job
    better = sum(
        1 for other in RESUME_CACHE
        if score_resume(other, job)["fit_score"] > result["fit_score"]
    )

    return {
        "target_job_title": job["target_job_title"],
        "required_skills": job["required_skills"],
        "preferred_skills": job["preferred_skills"],
        "result": result,
        "employers": extract_employers(text),
        "projects": extract_projects(text),
        "suggestions": build_suggestions(result, job, text),
        "rank": better + 1,
        "out_of": len(RESUME_CACHE) + 1
    }

@app.get("/search")
def search(keywords: str):
    results = []

    skills = [
        skill.strip().lower()
        for skill in keywords.split(",")
        if skill.strip()
    ]

    action_words = [
     "developed",
     "built",
     "implemented",
     "deployed",
     "managed",
     "designed",
     "created",
     "configured",
     "administered",
     "maintained",
     "worked with",
     "used",
     "integrated"
    ]

    for resume in RESUME_CACHE:
        try:
            text = resume["text"]
            text_lower = resume["text_lower"]

            skill_scores = {}
            context_scores = {}

            for skill in skills:
                count = text_lower.count(skill)
                skill_scores[skill] = count

                context_hits = 0
                start_pos = 0

                while True:
                    position = text_lower.find(skill, start_pos)

                    if position == -1:
                        break

                    context_start = max(0, position - 80)
                    context_end = min(
                        len(text_lower),
                        position + len(skill) + 80
                    )

                    context = text_lower[context_start:context_end]

                    if any(action in context for action in action_words):
                        context_hits += 1

                    start_pos = position + len(skill)

                context_scores[skill] = context_hits

            matched_skills = [
                skill for skill, count in skill_scores.items()
                if count > 0
            ]

            mention_score = sum(
                min(count, 5)
                for count in skill_scores.values()
            )

            coverage_score = len(matched_skills) * 20

            context_score = sum(
                min(hits, 3) * 10
                for hits in context_scores.values()
            )

            all_skills_bonus = 0
            if len(matched_skills) == len(skills):
                all_skills_bonus = 30

            total_score = (
                mention_score
                + coverage_score
                + context_score
                + all_skills_bonus
            )

            if total_score > 0:
                preview = " ".join(text.split())[:300]

                results.append({
                    "file": resume["file"],
                    "category": resume["category"],
                    "score": total_score,
                    "matched_skills": matched_skills,
                    "skill_scores": skill_scores,
                    "context_scores": context_scores,
                    "context_score": context_score,
                    "preview": preview
                })

        except Exception as e:
            print(f"ERROR SEARCHING {resume['file']}: {e}")

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )[:20]

    return {
        "keywords": skills,
        "matches": results,
        "count": len(results)
    }
