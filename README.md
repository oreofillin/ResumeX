# ResumeX

ResumeX matches resumes to job descriptions. Paste a job posting and it ranks 2,484 real resumes by how well they fit, or upload your own resume to see your score, how you'd rank against the dataset and what to improve.

**Live demo:** http://3.150.92.72:8000

## Features

- **Find Candidates:** paste a job description and get the top 20 resumes ranked by an overall fit score, with matched and missing skills for each.
- **Check My Resume:** upload a PDF resume and get:
  - a fit score for the job
  - your rank against the 2,484 dataset resumes
  - specific suggestions, such as missing required skills, skills listed without evidence, experience gaps and missing measurable results
- **Job description parsing:** detects the job title, required vs. preferred skills and years of experience required.
- **Resume parsing:** extracts job title, years of experience, education level, certifications, employers and projects from PDF text.

## How the scoring works

Each resume gets an **Overall Fit** score from 0 to 100, built from the criteria that apply to the job:

| Criterion | Weight | How it's measured |
|---|---|---|
| Required skills | 45 | Share of required skills found in the resume |
| Job title relevance | 20 | Word overlap between the job title and the resume's title, ignoring seniority words and treating "developer" and "engineer" as the same |
| Preferred skills | 15 | Share of preferred skills found |
| Experience | 15 | Candidate's years ÷ years required (capped at 100%) |
| Other skills | 5 | Other skills mentioned in the job |

If a job doesn't specify a criterion (for example, no experience requirement), that criterion is left out and the score is normalized over the rest.

Skills are matched as **whole words**, so "Java" doesn't match "JavaScript" and "Excel" doesn't match "excellent". Ties are broken by a secondary score that also rewards skills mentioned next to action verbs ("built", "deployed", "designed"), because that is evidence of real use rather than a keyword list.

## Tech stack

- **Backend:** Python, FastAPI, PyMuPDF (PDF text extraction)
- **Frontend:** HTML, CSS and vanilla JavaScript (no build step), with dark mode and mobile layout
- **Data:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), 2,484 resumes across 24 industries
- **Deployment:** AWS EC2 (Amazon Linux 2023), systemd
- **Testing:** pytest, FastAPI TestClient

All resume PDFs are parsed once at startup and cached in memory, so ranking the full dataset takes a few seconds and keyword search is instant.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `GET` | `/resumes` | Number of resumes loaded, by category |
| `POST` | `/match-job` | Body `{"job_description": "..."}`; returns the top 20 candidates |
| `POST` | `/analyze-resume` | Multipart form with `resume` (PDF) and `job_description`; returns score, rank and suggestions |
| `POST` | `/extract-skills` | Body `{"job_description": "..."}`; returns detected skills |
| `GET` | `/search?keywords=python,sql` | Keyword search across all resumes |

Interactive API docs are available at `/docs`.

## Run locally

```bash
git clone https://github.com/<your-username>/ResumeX.git
cd ResumeX
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the dataset (requires a free Kaggle account)
python -c "import kagglehub; print(kagglehub.dataset_download('snehaanbhawal/resume-dataset'))"

uvicorn main:app --reload
```

Then open http://localhost:8000. If the dataset is somewhere other than the default kagglehub cache, set `RESUMEX_DATASET_PATH` to the folder that contains the category folders (`ACCOUNTANT`, `ENGINEERING` and so on).

## Run tests

```bash
pytest
```

## Deploy on EC2

```bash
sudo cp deploy/resumex.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now resumex
```

Open port 8000 in the instance's security group.

## Limitations and next steps

- Skill detection uses a fixed list of about 30 technical skills, so non-technical roles get less precise matches. Next step: semantic matching with embeddings.
- Scanned (image-only) PDFs have no text to extract. Next step: OCR.
- The dataset is anonymized, so employer names often appear as "Company Name".
