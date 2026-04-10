import os
import re
import tempfile
from functools import lru_cache
from html import escape
from typing import List

import fitz
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

try:
    import pytesseract
except Exception:
    pytesseract = None

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "my_vector_store")

app = FastAPI(title="LawGPT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LEGAL_TERMS = [
    "law", "legal", "court", "judge", "advocate", "lawyer", "attorney", "petition",
    "case", "bail", "fir", "complaint", "summons", "notice", "agreement", "contract",
    "deed", "property", "tenant", "landlord", "rent", "divorce", "custody", "maintenance",
    "alimony", "ipc", "bns", "bharatiya nyaya sanhita", "crpc", "bnss", "evidence",
    "witness", "appeal", "writ", "civil", "criminal", "plaint", "affidavit", "injunction",
    "cheque", "cheque bounce", "section", "police", "arrest", "rights", "employment",
    "termination", "labour", "salary", "consumer", "cyber", "fraud", "defamation",
    "will", "inheritance", "succession", "partnership", "company", "tax"
]

LEGAL_PHRASES = [
    "legal advice", "legal help", "law related", "court case", "file a case", "file complaint",
    "legal notice", "bail application", "property dispute", "employment dispute", "divorce case",
    "consumer complaint", "police complaint", "contract dispute", "criminal case", "civil case"
]

NON_LEGAL_RESPONSE = (
    "I can only answer legal or law-related questions. Please ask about courts, contracts, cases, "
    "rights, notices, property, family law, criminal law, or other legal matters."
)

uploaded_db = None
uploaded_file_summaries: List[dict] = []
messages: List[dict] = []


class ChatRequest(BaseModel):
    question: str
    response_language: str = "English"
    mode: str = "GENERAL"


class ChatResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


@lru_cache(maxsize=1)
def get_main_db() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing in environment.")
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")


def is_legal_related_question(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    if not normalized:
        return False

    if any(phrase in normalized for phrase in LEGAL_PHRASES):
        return True

    word_hits = sum(1 for term in LEGAL_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized))
    if word_hits >= 1:
        return True

    legal_action_patterns = [
        r"\bmy\s+(?:rights?|property|case|complaint|agreement|contract)\b",
        r"\b(?:file|draft|send|serve|respond to)\s+(?:a\s+)?(?:case|complaint|notice|petition|bail|appeal)\b",
        r"\b(?:arrest|detain|summon|evict|eviction|terminate|termination|sue|lawsuit|inherit|inheritance)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in legal_action_patterns)


def normalize_llm_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join([p.strip() for p in parts if str(p).strip()]).strip()
    return str(content).strip()


def format_chat_history(max_turns: int = 4) -> str:
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "No previous chat history."


def build_context_from_docs(retriever, question: str) -> str:
    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No relevant document context found."

    context_parts = []
    for idx, doc in enumerate(docs[:6], start=1):
        source = doc.metadata.get("source", "unknown")
        content = str(doc.page_content or "").strip()
        if content:
            context_parts.append(f"[Source {idx}: {source}]\n{content}")

    return "\n\n".join(context_parts) if context_parts else "No relevant document context found."


def extract_text_from_image(image_file: UploadFile) -> str:
    if pytesseract is None:
        # OCR support is optional in serverless/container deploys.
        return ""
    image = Image.open(image_file.file)
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.file.read())
        tmp_path = tmp_file.name

    try:
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def build_text_summary(text: str, file_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable text found in this file."

    def clean_summary_text(raw: str) -> str:
        if not raw:
            return ""
        parts = []
        for line in str(raw).splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if line:
                parts.append(line)
        merged = " ".join(parts)
        merged = re.sub(r"\s+", " ", merged).strip()
        return merged

    # First preference: LLM paragraph summary with legal-priority coverage.
    try:
        source = cleaned[:14000]
        prompt = f"""
You are a legal-document analyst.
Prepare one meaningful paragraph summary of this legal document.

Rules:
1. Output must be a single coherent paragraph, not points.
2. Cover key points in this order naturally within the paragraph:
   parties/background, issues/facts, findings/holdings, relief/orders/next steps.
3. Include important legal details, obligations, risks, and outcomes if present.
4. Keep the paragraph factual and easy to understand.
5. Do not invent information.
6. No bullet points, no numbering, no headings.
7. Target length: 8-12 sentences.

File name: {file_name}
Document text:
{source}
"""
        llm_response = get_llm().invoke(prompt)
        llm_summary = normalize_llm_content(getattr(llm_response, "content", ""))
        paragraph = clean_summary_text(llm_summary)
        if len(paragraph.split()) >= 80:
            return paragraph
    except Exception:
        pass

    # Fallback: deterministic extractive summary with legal-priority ordering.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]

    def pick_by_keywords(source_sentences, keywords, limit):
        matches = []
        for sentence in source_sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                matches.append(sentence)
            if len(matches) >= limit:
                break
        return matches

    parties = pick_by_keywords(
        sentences,
        ["petitioner", "respondent", "appellant", "defendant", "plaintiff", "party", "court", "tribunal"],
        5,
    )
    issues = pick_by_keywords(
        sentences,
        ["issue", "question", "fact", "claim", "dispute", "evidence", "incident", "agreement"],
        5,
    )
    findings = pick_by_keywords(
        sentences,
        ["held", "find", "observation", "reason", "analysis", "conclusion", "violation", "liable"],
        6,
    )
    relief = pick_by_keywords(
        sentences,
        ["relief", "order", "direct", "penalty", "compensation", "fine", "sentence", "dismissed", "allowed"],
        4,
    )

    ordered = []
    for group in [parties, issues, findings, relief]:
        for sentence in group:
            if sentence not in ordered:
                ordered.append(sentence)

    if len(ordered) < 15:
        for sentence in sentences:
            if sentence not in ordered:
                ordered.append(sentence)
            if len(ordered) >= 20:
                break

    if len(ordered) < 15:
        words = cleaned.split()
        chunk_size = max(14, len(words) // 16) if words else 14
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        for chunk in chunks:
            if chunk not in ordered:
                ordered.append(chunk)
            if len(ordered) >= 20:
                break

    selected = ordered[:10]
    if not selected:
        selected = ["Limited readable legal content was extracted from this file."]

    paragraph = " ".join(selected)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "service": "lawgpt-api"}


@app.post("/reset")
def reset_chat():
    global messages
    messages = []
    return {"status": "reset"}


@app.get("/history")
def get_history():
    return {"messages": messages[-10:]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global messages

    mode = req.mode.upper().strip()
    question = req.question.strip()

    if not question:
        return ChatResponse(answer="Please enter a question.")

    if mode == "GENERAL" and not is_legal_related_question(question):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": NON_LEGAL_RESPONSE})
        return ChatResponse(answer=NON_LEGAL_RESPONSE)

    if mode == "DOCUMENT" and uploaded_db is None:
        return ChatResponse(answer="PLEASE UPLOAD THE DOCUMENT")
    try:
        retriever = (uploaded_db if mode == "DOCUMENT" and uploaded_db is not None else get_main_db()).as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        context = build_context_from_docs(retriever, question)
        chat_history = format_chat_history()

        prompt = f"""
<s>[INST]This is a chat template and As a legal chat bot, your primary objective is to provide accurate and concise information.
CRITICAL DOMAIN RULE: If the question is not legal or law-related, reply only with a brief refusal.
RESPONSE LANGUAGE: {req.response_language}
IMPORTANT: Your final answer must be entirely in the RESPONSE LANGUAGE.
MODE: {mode}
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

        response = get_llm().invoke(prompt)
        answer = normalize_llm_content(getattr(response, "content", ""))
    except Exception:
        return ChatResponse(
            answer=(
                "Chat service is temporarily unavailable. "
                "Please verify GROQ_API_KEY in Railway Variables and try again."
            )
        )

    if not answer:
        answer = "I could not generate a response for this question."

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    messages = messages[-20:]

    return ChatResponse(answer=answer)


@app.post("/upload")
def upload_documents(files: List[UploadFile] = File(...)):
    global uploaded_db, uploaded_file_summaries

    documents = []
    uploaded_file_summaries = []

    for uploaded_file in files:
        file_name = uploaded_file.filename
        ext = os.path.splitext(file_name)[1].lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif ext in [".jpg", ".jpeg", ".png"]:
            text = extract_text_from_image(uploaded_file)
        else:
            continue

        if text.strip():
            uploaded_file_summaries.append(
                {
                    "file_name": escape(file_name),
                    "word_count": len(text.split()),
                    "summary": build_text_summary(text, file_name),
                }
            )
            documents.append(Document(page_content=text, metadata={"source": file_name, "file_type": ext}))

    if not documents:
        return {"status": "no-docs", "summaries": []}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    embeddings = get_embeddings()

    main_db = get_main_db()
    main_db.add_documents(split_docs)
    main_db.save_local(VECTOR_STORE_PATH)

    if uploaded_db is None:
        uploaded_db = FAISS.from_documents(split_docs, embeddings)
    else:
        uploaded_db.add_documents(split_docs)

    return {
        "status": "ok",
        "processed_files": len(files),
        "chunks_added": len(split_docs),
        "summaries": uploaded_file_summaries,
    }
