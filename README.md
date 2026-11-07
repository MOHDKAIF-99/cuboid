# Cuboid

Upload previous year question papers (text PDFs or photos) and get a ranked list of the questions most likely to come in the next exam.

## How it works

1. Photos are read in the browser with Tesseract.js. PDFs are read on the server with PyMuPDF.
2. The text is split into individual questions.
3. Similar questions are grouped with TF-IDF and scikit-learn.
4. Groups are ranked by how many papers they appear in, how recent they are, and their marks.
5. Groq (Llama 3.3) cleans up the wording and names the topic. This step is optional.
6. The list can be copied or downloaded as a PDF.

## Run on your computer

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      (then paste your Groq key into .env)
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Deploy on Render (free)

- New > Web Service > pick this GitHub repo
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variables: `GROQ_API_KEY` = your key, `PYTHON_VERSION` = `3.12.7`

The free plan sleeps after 15 minutes without visitors, so the first request afterwards takes about a minute.

## Limits

- Scanned PDFs (text cannot be selected) are not supported. Upload photos of the pages instead.
- Papers need numbered questions such as `1.`, `Q1` or `(a)`.
- Put the year in each file name (for example `DS_2023.pdf`) so recent years rank higher.
