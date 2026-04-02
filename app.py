import streamlit as st
import os
import time
import tempfile
import re
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()

# Get API keys with defaults to avoid NoneType errors
google_api_key = os.getenv("GOOGLE_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY", "")

if google_api_key:
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Streamlit UI setup
st.set_page_config(page_title="LawGPT", layout="wide")
col1, col2, col3 = st.columns([1, 4, 1])
st.title("Llama Model Legal ChatBot")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Check for required API keys and vector store
if not groq_api_key:
    st.error("⚠️ GROQ API key is missing! Please add your GROQ_API_KEY to the .env file.")
    st.info("📝 Create a .env file in the project root with:\n```\nGROQ_API_KEY=your_key_here\n```")
    st.stop()

if not os.path.exists("my_vector_store"):
    st.error("⚠️ Vector store not found! Please run ingestion.py first to create the vector store.")
    st.stop()

# Helper function to extract text from images using OCR
def extract_text_from_image(image_file):
    """Extract text from image using pytesseract OCR"""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Extract text using PyMuPDF
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Clean up temp file
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def build_text_summary(text):
    """Create a detailed 5-10 line extractive summary from uploaded text."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable text found in this file."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    candidates = [s.strip() for s in sentences if s.strip()]

    # Ensure summary has between 5 and 10 lines when possible.
    target_lines = max(5, min(10, len(candidates))) if candidates else 5

    if len(candidates) < 5:
        words = cleaned.split()
        chunk_size = max(12, len(words) // 5) if words else 12
        candidates = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                candidates.append(chunk)

    selected = []
    for sentence in candidates:
        selected.append(sentence)
        if len(selected) >= target_lines:
            break

    if not selected:
        return cleaned

    lines = [f"{idx + 1}. {line}" for idx, line in enumerate(selected)]
    return "\n".join(lines)


def build_llm_summary(text, file_name):
    """Generate a clean 5-10 line summary using the configured Groq model."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable text found in this file."

    # Limit input size to keep summarization fast and stable.
    summary_source = cleaned[:12000]

    prompt = f"""
You are a legal document summarizer.
Create a clear MAIN SUMMARY of the uploaded document in 5 to 10 numbered lines.

Rules:
1. Focus on key facts only: parties, incident/background, key legal sections, timeline, and current status.
2. Keep each line concise and readable.
3. If the text is noisy (OCR artifacts), infer carefully and ignore obvious noise.
4. Do not add facts that are not present.

File name: {file_name}
Document text:
{summary_source}
"""

    try:
        summary_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        response = summary_llm.invoke(prompt)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join([str(item) for item in content])
        content = str(content).strip()

        if content:
            return content
    except Exception:
        # Fall back to deterministic extractive summary if LLM summary fails.
        pass

    return build_text_summary(text)

# Function to process uploaded documents and add to vector store
def process_and_add_documents(uploaded_files, embeddings, db):
    """Process uploaded files and add to existing vector store"""
    if not uploaded_files:
        return db, [], []
    
    documents = []
    file_summaries = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        st.info(f"Processing: {file_name}")
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            text = extract_text_from_image(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue
        
        if text.strip():
            word_count = len(text.split())
            file_summaries.append({
                "file_name": file_name,
                "file_type": file_extension,
                "word_count": word_count,
                "summary": build_llm_summary(text, file_name)
            })

            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={"source": file_name, "file_type": file_extension}
            )
            documents.append(doc)
        else:
            st.warning(f"No text extracted from {file_name}")
    
    if documents:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Add to existing vector store
        st.info(f"Adding {len(split_docs)} chunks to vector store...")
        db.add_documents(split_docs)
        
        # Save updated vector store
        db.save_local("my_vector_store")
        st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")

        return db, file_summaries, split_docs

    return db, file_summaries, []

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_summaries" not in st.session_state:
    st.session_state.uploaded_file_summaries = []

if "uploaded_file_sources" not in st.session_state:
    st.session_state.uploaded_file_sources = []

if "uploaded_db" not in st.session_state:
    st.session_state.uploaded_db = None

# Initialize embeddings and vector store with HuggingFace (free & unlimited!)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    
    # Store embeddings and db in session state for reuse
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = embeddings
    if 'db' not in st.session_state:
        st.session_state.db = db
        
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    st.error(f"❌ Error loading embeddings: {str(e)}")
    st.info("💡 Make sure you've run `python ingestion.py` to create the vector store first.")
    st.stop()

# ==================== FIRST HALF: FILE UPLOAD SECTION ====================
st.markdown("---")
st.header("📄 Upload Legal Documents")
st.markdown("Upload PDF, JPG, JPEG, or PNG files (up to 200MB per file)")

uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=['pdf', 'jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Upload legal documents to add to the knowledge base"
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Selected {len(uploaded_files)} file(s):**")
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            st.write(f"- {file.name} ({file_size_mb:.2f} MB)")
    
    with col2:
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing documents..."):
                updated_db, processed_summaries, split_docs = process_and_add_documents(
                    uploaded_files, 
                    st.session_state.embeddings, 
                    st.session_state.db
                )
                st.session_state.db = updated_db

                # Store file summaries for display and add source names for file-focused Q&A.
                if processed_summaries:
                    st.session_state.uploaded_file_summaries = processed_summaries
                    st.session_state.uploaded_file_sources = [
                        item["file_name"] for item in processed_summaries
                    ]

                if split_docs:
                    st.session_state.uploaded_db = FAISS.from_documents(
                        split_docs,
                        st.session_state.embeddings
                    )

                # Update retriever with new documents
                db_retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                st.rerun()

if st.session_state.uploaded_file_summaries:
    st.subheader("📌 Uploaded File Summaries")
    for item in st.session_state.uploaded_file_summaries:
        with st.expander(f"{item['file_name']} ({item['word_count']} words)", expanded=False):
            st.markdown(item["summary"].replace("\n", "  \n"))
    st.markdown("---")

st.markdown("---")
st.header("💬 Ask Legal Questions")
st.markdown("Chat with the legal assistant about your uploaded documents or existing legal knowledge")
st.markdown("---")

# Response language selector
language_options = {
    "English": "English",
    "Telugu": "Telugu",
    "Hindi": "Hindi"
}
selected_language = st.selectbox(
    "Select response language",
    options=list(language_options.keys()),
    index=0,
    help="Assistant responses will be generated in the selected language"
)
response_language = language_options[selected_language]

# Define the prompt template
prompt_template = """
<s>[INST]This is a chat template and As a legal chat bot , your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code and uploaded legal documents.
RESPONSE LANGUAGE: {response_language}
IMPORTANT: Your final answer must be entirely in the RESPONSE LANGUAGE.
MODE: {response_mode}
ANSWER FORMAT: {answer_length_instruction}
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question', 'chat_history', 'response_language', 'response_mode', 'answer_length_instruction']
)

# Initialize the LLM with updated model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Mode selector requested by user.
selected_mode = st.selectbox(
    "Ask questions from",
    options=["GENERAL", "DOCUMENT"],
    index=0,
    help="GENERAL: legal answers from complete knowledge base. DOCUMENT: answers only from uploaded file(s)."
)

if selected_mode == "DOCUMENT" and not st.session_state.uploaded_db:
    st.warning("PLEASE UPLOAD THE DOCUMENT")

# Set up mode-specific retriever and instructions.
if selected_mode == "GENERAL":
    selected_retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    answer_length_instruction = "Provide a detailed legal answer in 10 to 15 lines."
else:
    selected_retriever = st.session_state.uploaded_db.as_retriever(search_type="similarity", search_kwargs={"k": 4}) if st.session_state.uploaded_db else st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    answer_length_instruction = "Answer only using the uploaded document context in 10 to 15 lines. If context is insufficient, state that clearly."

prompt_with_mode = prompt.partial(
    response_language=response_language,
    response_mode=selected_mode,
    answer_length_instruction=answer_length_instruction
)


def format_chat_history(messages, max_turns=4):
    """Format recent chat history for prompt context."""
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "No previous chat history."


def build_context_from_docs(retriever, question):
    """Retrieve relevant chunks and combine them into prompt context."""
    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No relevant document context found.", []

    context_parts = []
    for idx, doc in enumerate(docs[:6], start=1):
        source = doc.metadata.get("source", "unknown")
        content = str(doc.page_content or "").strip()
        if not content:
            continue
        context_parts.append(f"[Source {idx}: {source}]\n{content}")

    return "\n\n".join(context_parts) if context_parts else "No relevant document context found.", docs


def generate_answer(question, retriever, prompt_template):
    """Generate answer from retriever context + chat history using ChatGroq."""
    context, _ = build_context_from_docs(retriever, question)
    chat_history = format_chat_history(st.session_state.messages)
    formatted_prompt = prompt_template.format(
        context=context,
        chat_history=chat_history,
        question=question,
    )
    response = llm.invoke(formatted_prompt)
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "\n".join([str(item) for item in content])
    return str(content).strip() if str(content).strip() else "I could not generate a response for this question."

# Display previous messages
for message in st.session_state.messages:
    role = message.get("role")
    avatar = "⚖️" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Say something")

if input_prompt:
    if selected_mode == "DOCUMENT" and not st.session_state.uploaded_db:
        with st.chat_message("assistant", avatar="⚖️"):
            st.write("PLEASE UPLOAD THE DOCUMENT")
        st.session_state.messages.append({"role": "assistant", "content": "PLEASE UPLOAD THE DOCUMENT"})
        st.stop()

    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("Thinking 💡...", expanded=True):
            answer_text = generate_answer(input_prompt, selected_retriever, prompt_with_mode)
            message_placeholder = st.empty()
            full_response = "\n\n\n"

            # Print the result dictionary to inspect its structure
            #st.write(result)

            for chunk in answer_text:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

            # Print the answer
            #st.write(result["answer"])

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    st.session_state.messages.append({"role": "assistant", "content": answer_text})