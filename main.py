"""Cuboid: predicts the most likely exam questions from previous year papers.

Flow: read the papers (PDFs here, photos are read in the browser) -> pull out the
individual questions -> group similar ones -> rank by how many years they appeared,
how recent they are and their marks -> let an LLM (Groq) name the topic of each.
The question text always stays exactly as printed in the paper.
"""
import io
import json
import os
import re
import traceback
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pdfplumber
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
MAX_PAGES = 30            # pages read per PDF
MAX_QUESTIONS = 1500      # keeps memory low on the free Render plan
TOP_N = 15                # questions in the final list
SIMILARITY_CUTOFF = 0.6   # cosine distance; lower = stricter grouping
# Groq retired llama-3.3-70b-versatile on 16 Aug 2026. Set GROQ_MODEL to change this.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Words that start many exam questions but say nothing about the topic.
EXAM_WORDS = {
    "explain", "define", "write", "describe", "discuss", "compare", "differentiate",
    "list", "state", "give", "draw", "illustrate", "briefly", "short", "note", "notes",
    "example", "examples", "using", "given", "also", "following", "marks", "mark",
    "justify", "show", "prove", "find", "solve", "let", "attempt", "calculate", "compute",
    "ii", "iii", "iv", "ensure", "determine", "identify", "express", "whether", "exist", "exists",
}

app = FastAPI(title="Cuboid")


# ------------------------------------------------------------ small text helpers
CID = re.compile(r"\(cid:\d+\)")
CUT_AT = re.compile(r"[ऀ-ॿ]|\(cid:\d+\)")  # Hindi copy of a question starts here
MARKS_HEADER = re.compile(r"(\d{1,2})\s*[xX×]\s*(\d{1,2})\s*=\s*(\d{1,3})")
SESSION = re.compile(r"(20\d{2})\s*[-–]\s*(\d{2})(?!\d)")
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")  # works for DS_2023.pdf too
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def clean_question(text: str) -> str:
    """Tidy one question: drop the Hindi copy, stray labels and diagram numbers."""
    if len(CID.findall(text)) >= 3:  # many (cid:N) = unreadable Hindi copy: cut there
        text = CUT_AT.split(text, maxsplit=1)[0]
    else:  # one stray glyph inside a maths symbol: only drop the Hindi letters
        text = CID.sub("", text)
        text = re.split(r"[\u0900-\u097F]", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?:\s+(?:[ivx]{1,4}|[a-h])[.)])+\s*$", "", text)  # "i)" left before a Hindi copy
    text = re.sub(r"(?:\s+[A-Za-z]?\d{1,3}){3,}\s*$", "", text)       # numbers from a diagram
    return text.strip(" -:;|").lstrip(". ")


def keep_question(text: str) -> bool:
    return len(text) >= 12 and len(text.split()) >= 3


def marks_from_header(line: str) -> Optional[int]:
    """'2 x 07 = 14' -> 2 marks each. '3x10 = 30' with 'any three' -> 10 marks each."""
    m = MARKS_HEADER.search(line)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    word = re.search(r"\bany\s+(one|two|three|four|five)\b", line, re.I)
    marks = b if word and NUMBER_WORDS[word.group(1).lower()] == a and a != b else a
    return marks if 1 <= marks <= 30 else None


def find_year(name: str, text: str) -> Optional[int]:
    match = YEAR.search(name) or YEAR.search(text[:1500])
    return int(match.group(0)) if match else None


def find_session(name: str, text: str):
    """Returns (start year, label) such as (2022, '2022-23'), or (None, None)."""
    for m in SESSION.finditer(text[:3000]):
        year, tail = int(m.group(1)), int(m.group(2))
        if (year + 1) % 100 == tail:
            return year, f"{year}-{tail:02d}"
    year = find_year(name, text)
    return (year, str(year)) if year else (None, None)


# --------------------------------------------------- finding questions in text
START = re.compile(
    r"^\s*(?:q\.?\s*\d{1,2}\s*[.):\-]?|question\s*\d{1,2}\s*[.):\-]?"
    r"|\d{1,2}\s*[.)]|\(?[a-h]\)|\(?[ivx]{1,4}\))\s+",
    re.I,
)
MARKS = re.compile(
    r"[\[(]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\])]|\b(\d{1,2})\s*marks?\b", re.I
)
LETTER_START = re.compile(r"^\s*(?:\(([a-z])\)|([a-z])[.)])\s+(\S.*)$")
SECTION_LINE = re.compile(r"^\s*SECTION\b", re.I)
ATTEMPT_LINE = re.compile(r"^\s*\d{1,2}[.)]\s+Attempt\b", re.I)


def split_questions(text: str) -> List[dict]:
    """For plain papers whose questions are numbered 1. Q2 (a) and so on."""
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
        q = clean_question(q)
        if keep_question(q):
            questions.append({"text": q[:400], "marks": marks})
    return questions


def split_by_letters(text: str) -> List[dict]:
    """For papers that list questions as (a) (b) (c)... under SECTION headings."""
    questions, current, expected, marks = [], None, "a", None

    def flush():
        nonlocal current
        if current:
            q = clean_question(current["text"])
            if keep_question(q):
                questions.append({"text": q[:400], "marks": current["marks"]})
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header_marks = marks_from_header(line)
        if header_marks is not None or MARKS_HEADER.search(line):
            flush()
            marks, expected = header_marks, "a"
            continue
        if SECTION_LINE.match(line) or ATTEMPT_LINE.match(line):
            flush()
            expected = "a"
            continue
        m = LETTER_START.match(line)
        if m:
            letter = (m.group(1) or m.group(2)).lower()
            if letter == expected or letter == "a":
                flush()
                current = {"text": m.group(3), "marks": marks}
                expected = chr(ord(letter) + 1)
                continue
        if current is not None and not re.match(r"^(printed page|subject code|roll no)", line, re.I):
            current["text"] += " " + line
    flush()
    return questions


def questions_from_text(text: str) -> List[dict]:
    """Plain text (photos, simple PDFs): use whichever style finds more questions."""
    lettered = split_by_letters(text)
    best = lettered if len(lettered) >= 8 else split_questions(text)
    return best[:MAX_QUESTIONS]


# ------------------------------------------------------------------ reading PDFs
LABEL = re.compile(r"^(?:\(?[A-Za-z]\)?[.)]?|\d{1,2}[.)]?)$")
QNO_HEADER = re.compile(r"^q\s*\.?\s*no\.?$", re.I)


def drop_watermark(page):
    """The diagonal watermark is drawn 45 degrees tilted; real text is straight."""
    return page.filter(
        lambda o: o.get("object_type") != "char"
        or (abs(o["matrix"][1]) < 0.05 and abs(o["matrix"][2]) < 0.05)
    )


def questions_from_tables(pages) -> List[dict]:
    """Question papers laid out as tables: one row per question."""
    questions, current_marks, marks_col = [], None, None
    for page in pages:
        events = []
        for line in page.extract_text_lines():
            if MARKS_HEADER.search(line["text"]):
                events.append((line["top"], 0, marks_from_header(line["text"])))
        for table in page.find_tables():
            events.append((table.bbox[1], 1, table))
        events.sort(key=lambda e: (e[0], e[1]))
        for _, kind, payload in events:
            if kind == 0:
                current_marks = payload
                continue
            for row in payload.extract():
                cells = [(c or "").strip() for c in row]
                if len(cells) < 2:
                    continue
                if QNO_HEADER.match(re.sub(r"\s+", " ", cells[0])):
                    marks_col = next((i for i, c in enumerate(cells) if c.lower() == "marks"), None)
                    continue
                if not LABEL.match(cells[0]):
                    continue
                text = clean_question(cells[1])
                if not keep_question(text):
                    continue
                marks = current_marks
                if marks_col is not None and marks_col < len(cells) and re.fullmatch(r"\d{1,2}", cells[marks_col]):
                    marks = int(cells[marks_col])
                questions.append({"text": text[:400], "marks": marks})
    return questions


def read_pdf(data: bytes, name: str) -> dict:
    """Returns {'has_text', 'questions', 'year', 'label', 'method'} for one PDF."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [drop_watermark(p) for p in pdf.pages[:MAX_PAGES]]
        text = "\n".join((p.extract_text() or "") for p in pages)
        if len(text.strip()) < 30:
            return {"has_text": False, "questions": [], "year": None, "label": None, "method": None}
        questions, method = questions_from_tables(pages), "table"
        if len(questions) < 8:  # not a table paper: read it as plain text instead
            questions, method = questions_from_text(text), "text"
    year, label = find_session(name, text)
    return {"has_text": True, "questions": questions[:MAX_QUESTIONS], "year": year, "label": label, "method": method}


# -------------------------------------------------------------------- analysis
def analyse(papers: List[dict]) -> Optional[dict]:
    items = []
    for paper in papers:
        questions = paper.get("questions")
        if questions is None:
            questions = questions_from_text(paper["text"])
        label = paper.get("label") or (str(paper["year"]) if paper.get("year") else None)
        for q in questions:
            items.append({**q, "paper": paper["name"], "year": paper.get("year"), "label": label})
    items = items[:MAX_QUESTIONS]
    if not items:
        return None

    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | EXAM_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only, no numbers
        ngram_range=(1, 2),
        sublinear_tf=True,
        preprocessor=lambda t: re.sub(r"k[\s-]?map|karnaugh\s+map", "kmap", t.lower()),
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
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    paper_names = {i["paper"] for i in items}
    all_years = [p["year"] for p in papers if p.get("year")]
    year_min, year_max = (min(all_years), max(all_years)) if all_years else (0, 0)

    clusters = []
    for label_id in set(labels.tolist()):
        idx = np.where(labels == label_id)[0]
        sub = similarity[np.ix_(idx, idx)]
        medoid = int(idx[sub.sum(axis=1).argmax()])  # the most typical wording
        members = [items[i] for i in idx]
        sessions = sorted({(m["year"] or 0, m["label"]) for m in members if m["label"]})
        marks = [m["marks"] for m in members if m["marks"]]
        row = matrix[medoid].toarray().ravel()
        common = row * (doc_freq >= 2)  # a word shared by several questions names a topic better
        if common.max() > 0:
            row = common
        topic = terms[row.argmax()].title() if row.max() > 0 else "General"
        clusters.append(
            {
                "question": items[medoid]["text"],
                "topic": topic,
                "marks": items[medoid]["marks"] or (max(marks) if marks else None),  # the shown question's own marks
                "years": [s[1] for s in sessions],
                "year_values": [s[0] for s in sessions if s[0]],
                "papers": {m["paper"] for m in members},
                "times": len(members),
            }
        )

    max_marks = max((c["marks"] or 0 for c in clusters), default=0)
    for c in clusters:
        freq = len(c["papers"]) / len(paper_names)
        if c["year_values"] and year_max > year_min:
            recency = (max(c["year_values"]) - year_min) / (year_max - year_min)
        else:
            recency = 0.5
        mark_part = (c["marks"] / max_marks) if c["marks"] and max_marks else 0.5
        c["score"] = int(min(99, max(1, round(100 * (0.6 * freq + 0.25 * recency + 0.15 * mark_part)))))

    clusters.sort(key=lambda c: (c["score"], c["times"]), reverse=True)
    return {
        "clusters": clusters[:TOP_N],
        "questions_found": len(items),
        "papers": len(paper_names),
        "used": sorted(paper_names),
    }


def name_topics_with_groq(rows: List[dict], subject: str) -> Optional[dict]:
    """Ask Groq for a short topic name per question. The question text is not touched."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=key, timeout=30)
        payload = [{"id": r["id"], "question": r["question"]} for r in rows]
        prompt = (
            "These are exam questions"
            + (f" for the subject '{subject}'" if subject else "")
            + ". For each one give the syllabus topic it belongs to, as a short name of 1 to 3 words "
            "(for example 'Lattices', 'Boolean algebra', 'Graph theory'). "
            "Do not rewrite or repeat the questions. "
            'Reply with JSON only: {"items":[{"id":0,"topic":"..."}]}\n\n'
            + json.dumps(payload)
        )
        reply = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You label exam questions. Reply with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(reply.choices[0].message.content)
        topics = {}
        for x in data.get("items", []):
            if isinstance(x.get("topic"), str) and x["topic"].strip():
                topics[int(x["id"])] = x["topic"].strip()[:40]
        return topics or None
    except Exception as exc:  # network, quota, bad JSON: keep the basic topic names
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
            info = read_pdf(data, name)
        except Exception:
            traceback.print_exc()
            skipped.append({"name": name, "reason": "This PDF could not be opened."})
            continue
        if not info["has_text"]:
            skipped.append(
                {"name": name, "reason": "Scanned PDF with no selectable text. Upload photos of its pages instead."}
            )
        elif not info["questions"]:
            skipped.append({"name": name, "reason": "No questions could be found in this paper."})
        else:
            papers.append({"name": name, **info})

    for item in ocr:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "photo"))
        text = str(item.get("text", ""))[:200000]
        if len(text.strip()) < 30:
            skipped.append({"name": name, "reason": "No readable text found in this photo."})
            continue
        questions = questions_from_text(text)
        year, label = find_session(name, text)
        if not questions:
            skipped.append({"name": name, "reason": "No questions could be found in this photo."})
        else:
            papers.append({"name": name, "questions": questions, "year": year, "label": label})

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
            "No questions found. Check that the papers are readable and list their questions clearly.",
        )

    used = set(result["used"])
    for p in papers:
        if p["name"] not in used:
            skipped.append({"name": p["name"], "reason": "No usable questions were found in this paper."})

    rows = [{"id": i, "question": c["question"]} for i, c in enumerate(result["clusters"])]
    topic_names = name_topics_with_groq(rows, subject.strip())
    items = []
    for i, c in enumerate(result["clusters"]):
        items.append(
            {
                "rank": i + 1,
                "question": c["question"],
                "topic": (topic_names or {}).get(i, c["topic"]),
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
            "ai": topic_names is not None,
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
    years: List[Union[str, int]] = []
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

