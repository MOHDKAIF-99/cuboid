"""Cuboid: predicts the most likely exam questions from previous year papers.

Flow: read text (PDF here, photos in the browser) -> split into questions ->
group similar ones -> rank by repeats, recent years and marks -> let an LLM
(Groq) clean up the wording and name the topic.
"""
import json
import os
import re
import traceback
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fpdf import FPDF
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent
MAX_SOURCES = 10          # papers per request
MAX_FILE_MB = 15          # per PDF
MAX_QUESTIONS = 1500      # keeps memory low on the free Render plan
TOP_N = 12                # questions in the final list
SIMILARITY_CUTOFF = 0.65  # cosine distance; lower = stricter grouping
# Groq retired llama-3.3-70b-versatile on 16 Aug 2026. Set GROQ_MODEL to change this.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Words that start many exam questions but say nothing about the topic.
EXAM_WORDS = {
    "explain", "define", "write", "describe", "discuss", "compare", "differentiate",
    "list", "state", "give", "draw", "illustrate", "briefly", "short", "note", "notes",
    "example", "examples", "using", "given", "also", "following", "marks", "mark",
}

app = FastAPI(title="Cuboid")


# ---------------------------------------------------------------- reading text
def pdf_to_text(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


START = re.compile(
    r"^\s*(?:q\.?\s*\d{1,2}\s*[.):\-]?|question\s*\d{1,2}\s*[.):\-]?"
    r"|\d{1,2}\s*[.)]|\(?[a-h]\)|\(?[ivx]{1,4}\))\s+",
    re.I,
)
MARKS = re.compile(
    r"[\[(]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\])]|\b(\d{1,2})\s*marks?\b", re.I
)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")  # works for DS_2023.pdf too


def split_questions(text: str) -> List[dict]:
    """Cut a paper's text into questions, using numbering like 1. Q2 (a)."""
    raw_questions, current = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if START.match(line):
            if current:
                raw_questions.append(current)
            current = START.sub("", line, count=1)
        elif current is not None:
            current += " " + line
    if current:
        raw_questions.append(current)

    questions = []
    for q in raw_questions:
        marks = None
        found = list(MARKS.finditer(q))
        if found:
            last = found[-1]
            value = int(last.group(1) or last.group(2))
            if 1 <= value <= 30:
                marks = value
                q = q[: last.start()] + " " + q[last.end():]
        q = re.sub(r"\s+", " ", q).strip(" -:;.")
        if len(q) < 15 or len(q.split()) < 4:
            continue
        questions.append({"text": q[:400], "marks": marks})
    return questions


def find_year(name: str, text: str) -> Optional[int]:
    match = YEAR.search(name) or YEAR.search(text[:1500])
    return int(match.group(0)) if match else None


# -------------------------------------------------------------------- analysis
def analyse(papers: List[dict]) -> Optional[dict]:
    items = []
    for paper in papers:
        for q in split_questions(paper["text"]):
            items.append({**q, "paper": paper["name"], "year": paper["year"]})
    items = items[:MAX_QUESTIONS]
    if not items:
        return None

    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | EXAM_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only, no numbers
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform([i["text"] for i in items])
    except ValueError:
        return None

    # Questions with no topic words (only "explain", "with examples", numbers...)
    # have an all-zero vector, which cosine grouping cannot handle. Drop them.
    keep = np.asarray(matrix.getnnz(axis=1) > 0).ravel()
    items = [item for item, k in zip(items, keep) if k]
    if not items:
        return None
    matrix = matrix[keep]

    if len(items) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=SIMILARITY_CUTOFF,
        ).fit_predict(matrix.toarray())

    similarity = (matrix @ matrix.T).toarray()
    terms = vectorizer.get_feature_names_out()
    paper_names = {i["paper"] for i in items}
    all_years = [p["year"] for p in papers if p["year"]]
    year_min, year_max = (min(all_years), max(all_years)) if all_years else (0, 0)

    clusters = []
    for label in set(labels.tolist()):
        idx = np.where(labels == label)[0]
        sub = similarity[np.ix_(idx, idx)]
        medoid = int(idx[sub.sum(axis=1).argmax()])  # the most typical wording
        members = [items[i] for i in idx]
        years = sorted({m["year"] for m in members if m["year"]})
        marks = [m["marks"] for m in members if m["marks"]]
        row = matrix[medoid].toarray().ravel()
        topic = terms[row.argmax()].title() if row.max() > 0 else "General"
        clusters.append(
            {
                "question": items[medoid]["text"],
                "topic": topic,
                "marks": max(marks) if marks else None,
                "years": years,
                "papers": {m["paper"] for m in members},
                "times": len(members),
            }
        )

    max_marks = max((c["marks"] or 0 for c in clusters), default=0)
    for c in clusters:
        freq = len(c["papers"]) / len(paper_names)
        if c["years"] and year_max > year_min:
            recency = (max(c["years"]) - year_min) / (year_max - year_min)
        else:
            recency = 0.5
        mark_part = (c["marks"] / max_marks) if c["marks"] and max_marks else 0.5
        c["score"] = int(min(99, max(1, round(100 * (0.6 * freq + 0.25 * recency + 0.15 * mark_part)))))

    clusters.sort(key=lambda c: (c["score"], c["times"]), reverse=True)
    return {
        "clusters": clusters[:TOP_N],
        "questions_found": len(items),
        "papers": len(paper_names),
    }


def refine_with_groq(rows: List[dict], subject: str) -> Optional[dict]:
    """Ask Groq (Llama 3.3) to fix wording and name each topic. Optional."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=key, timeout=30)
        payload = [
            {"id": r["id"], "question": r["question"], "years": r["years"], "marks": r["marks"]}
            for r in rows
        ]
        prompt = (
            "These are exam questions that repeat across previous year papers"
            + (f" for the subject '{subject}'" if subject else "")
            + ". For each one: fix spelling or OCR mistakes and write it as one clear exam "
            "question without changing its meaning, and give a short topic name of 1 to 3 words. "
            'Reply with JSON only: {"items":[{"id":0,"question":"...","topic":"..."}]}\n\n'
            + json.dumps(payload)
        )
        reply = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You help students prepare for exams. Reply with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(reply.choices[0].message.content)
        return {int(x["id"]): x for x in data.get("items", [])}
    except Exception as exc:  # network, quota, bad JSON: keep the basic result
        print("Groq step skipped:", exc)
        return None


# ------------------------------------------------------------------------ API
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"])  # uptime monitors send HEAD
def health():
    return {"ok": True}


@app.post("/api/predict")
def predict(
    files: List[UploadFile] = File(default=[]),
    ocr_texts: str = Form("[]"),
    subject: str = Form(""),
):
    try:
        ocr = json.loads(ocr_texts)
        if not isinstance(ocr, list):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "Photo text was not sent correctly. Please try again.")
    if len(files) + len(ocr) == 0:
        raise HTTPException(400, "Add at least one paper first.")
    if len(files) + len(ocr) > MAX_SOURCES:
        raise HTTPException(400, f"Please upload at most {MAX_SOURCES} papers at a time.")

    papers, skipped = [], []
    for f in files:
        name = f.filename or "paper.pdf"
        if not name.lower().endswith(".pdf"):
            skipped.append({"name": name, "reason": "Only PDF files are read on the server."})
            continue
        data = f.file.read(MAX_FILE_MB * 1024 * 1024 + 1)
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            skipped.append({"name": name, "reason": f"File is larger than {MAX_FILE_MB} MB."})
            continue
        try:
            text = pdf_to_text(data)
        except Exception:
            skipped.append({"name": name, "reason": "This PDF could not be opened."})
            continue
        if len(text.strip()) < 30:
            skipped.append(
                {"name": name, "reason": "Scanned PDF with no selectable text. Upload photos of its pages instead."}
            )
            continue
        papers.append({"name": name, "year": find_year(name, text), "text": text})

    for item in ocr:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "photo"))
        text = str(item.get("text", ""))[:200000]
        if len(text.strip()) < 30:
            skipped.append({"name": name, "reason": "No readable text found in this photo."})
            continue
        papers.append({"name": name, "year": find_year(name, text), "text": text})

    try:
        result = analyse(papers) if papers else None
    except Exception as exc:
        traceback.print_exc()  # full details appear in the terminal
        raise HTTPException(
            500, f"Could not analyse these papers ({type(exc).__name__}). Try different files."
        )
    if not result:
        raise HTTPException(
            422,
            "No questions found. Check that the papers have numbered questions like 1., Q1 or (a).",
        )

    rows = [
        {"id": i, "question": c["question"], "years": c["years"], "marks": c["marks"]}
        for i, c in enumerate(result["clusters"])
    ]
    refined = refine_with_groq(rows, subject.strip())
    items = []
    for i, c in enumerate(result["clusters"]):
        r = (refined or {}).get(i, {})
        question = r.get("question") if isinstance(r.get("question"), str) and r.get("question").strip() else c["question"]
        topic = r.get("topic") if isinstance(r.get("topic"), str) and r.get("topic").strip() else c["topic"]
        items.append(
            {
                "rank": i + 1,
                "question": question.strip(),
                "topic": topic.strip(),
                "marks": c["marks"],
                "years": c["years"],
                "times": c["times"],
                "score": c["score"],
            }
        )

    topic_counts = {}
    for it in items:
        topic_counts[it["topic"]] = topic_counts.get(it["topic"], 0) + it["times"]
    total = sum(topic_counts.values()) or 1
    topics = [
        {"name": name, "share": round(100 * n / total)}
        for name, n in sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    ]

    warning = None
    if result["papers"] == 1:
        warning = "Only one paper was read. Predictions get much better with 3 or more papers."

    return {
        "stats": {
            "papers": result["papers"],
            "questions": result["questions_found"],
            "topics": len(topic_counts),
            "ai": refined is not None,
        },
        "items": items,
        "topics": topics,
        "skipped": skipped,
        "warning": warning,
    }


class PdfItem(BaseModel):
    rank: int
    question: str
    topic: str = ""
    marks: Optional[int] = None
    years: List[int] = []
    score: int = 0


class PdfRequest(BaseModel):
    subject: str = ""
    items: List[PdfItem]


def safe(text: str) -> str:
    """The built-in PDF fonts only know Latin-1 characters."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


@app.post("/api/pdf")
def make_pdf(body: PdfRequest):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Cuboid - predicted questions", new_x="LMARGIN", new_y="NEXT")
    if body.subject.strip():
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, safe(body.subject.strip()), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for it in body.items:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 6, safe(f"{it.rank}. {it.question}"), new_x="LMARGIN", new_y="NEXT")
        details = [it.topic] if it.topic else []
        if it.marks:
            details.append(f"{it.marks} marks")
        if it.years:
            details.append("Asked in " + ", ".join(str(y) for y in it.years))
        details.append(f"{it.score}% likely")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, safe("   " + "  |  ".join(details)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    return Response(
        bytes(pdf.output()),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cuboid-predicted-questions.pdf"'},
    )

