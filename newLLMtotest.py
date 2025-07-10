import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import dotenv
import faiss
import html
import streamlit as st
import pytesseract
from datetime import datetime
import re
import docx
import requests
import pdfplumber
from PIL import Image
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import gc

st.set_page_config(page_title="TalkTonic", layout="centered")

@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")
embed_model = get_embed_model()

def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

def retrieve_relevant_chunks_faiss(query, faiss_index, chunk_map, top_k=4):
    query_vec = embed_model.encode([query])[0]
    query_vec = normalize_vector(query_vec).astype("float32")

    D, I = faiss_index.search(np.array([query_vec]), top_k)
    results = [chunk_map.get(i, "") for i in I[0]]

    return "\n".join(results)

def chunk_text(text, chunk_size=400, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    image = page.to_image(resolution=150).original
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text + "\n"
            return text or "No text found, even with OCR."
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return "Unsupported file type."
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_here")
def call_groq_model(message):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": message}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def get_file_hash(file):
    content = file.read()
    hash_val = hashlib.md5(content).hexdigest()
    file.seek(0)
    return hash_val

def strip_html_tags(text):
    return re.sub(r'<[^>]*>', '', text)

def get_theme_colors(theme):
    themes = {
        "Light": {"chat_bg": "#f0f0f0", "user_bg": "#4caf50", "user_color": "white",
                  "bot_bg": "#d3d3d3", "bot_color": "black", "clear_btn_bg": "#f44336", "clear_btn_color": "white"},
        "Midnight": {"chat_bg": "#0b0c10", "user_bg": "#66fcf1", "user_color": "#0b0c10",
                     "bot_bg": "#1f2833", "bot_color": "#c5c6c7", "clear_btn_bg": "#45a29e", "clear_btn_color": "#0b0c10"},
        "Dark": {"chat_bg": "#1f1f1f", "user_bg": "#4caf50", "user_color": "white",
                 "bot_bg": "#333", "bot_color": "#f1f1f1", "clear_btn_bg": "#f44336", "clear_btn_color": "white"}
    }
    return themes.get(theme, themes["Dark"])

# ===== Session State =====
for key, default in {
    "last_file_hash": None,
    "messages": [],
    "pending_input": "",
    "theme": "Dark",
    "faiss_index": None,
    "chunk_map": {},
    "preview_text": "",
    "show_summary_input": True
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===== File Upload =====
uploaded_file = st.file_uploader("Upload a PDF, Image, or Text File", type=["pdf", "png", "jpg", "jpeg", "txt", "docx"])
if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    uploaded_file.seek(0)
    if file_hash != st.session_state.last_file_hash:
        with st.spinner("Processing file and preparing chatbot..."):
            extracted_text = extract_text_from_file(uploaded_file)
            chunks = chunk_text(extracted_text)

            embeddings = embed_model.encode(chunks, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)

            dimension = embeddings.shape[1]
            st.session_state.faiss_index = faiss.IndexFlatIP(dimension)
            st.session_state.faiss_index.add(embeddings)

            st.session_state.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
            st.session_state.preview_text = extracted_text[:2000]
            st.session_state.last_file_hash = file_hash
            st.session_state.show_summary_input = True

        st.success(f"✅ File processed.")
        st.write(f"📄 Current File: {uploaded_file.name}")

st.markdown("<h2 style='text-align: left; margin-top: 0;'>   🤖 TalkTonic</h2>", unsafe_allow_html=True)

# ===== Chat Page =====
colors = get_theme_colors(st.session_state.theme)
st.markdown(f"""
    <style>
    .chat-container {{height: 400px; overflow-y: auto; border: 1px solid #444; padding: 10px;
        border-radius: 10px; background-color: {colors['chat_bg']}; color: {colors['bot_color']};}}
    .user-message {{background-color: {colors['user_bg']}; color: {colors['user_color']};
        padding: 8px 12px; border-radius: 10px; margin-bottom: 8px; max-width: 70%;
        float: right; clear: both; font-size: 15px;}}
    .bot-message {{background-color: {colors['bot_bg']}; color: {colors['bot_color']};
        padding: 8px 12px; border-radius: 10px; margin-bottom: 8px; max-width: 70%;
        float: left; clear: both; font-size: 15px;}}
    small {{display: block; font-size: 11px; opacity: 0.7; margin-top: 3px;}}
    </style>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([4, 4, 5])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            for key in ["faiss_index", "chunk_map", "messages", "pending_input", "last_file_hash", "preview_text"]:
                st.session_state[key] = None if key != "messages" else []
            st.session_state.show_summary_input = True
            gc.collect()

# Show preview if available


# Query input after upload
if st.session_state.show_summary_input and st.session_state.faiss_index:
    query = st.text_input("Ask something about this file:", placeholder="E.g. Summarize this document", key="summary_input")
    if st.button("Send Extracted Text to Bot"):
        query = query.strip()
        if query:
            context = retrieve_relevant_chunks_faiss(query, st.session_state.faiss_index, st.session_state.chunk_map)
            prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
            st.session_state.messages.append(("user", query))
        else:
            sample_chunks = list(st.session_state.chunk_map.values())[:5]
            prompt = "\n".join(sample_chunks)
            st.session_state.messages.append(("user", "[Full Document Sent]"))

        prompt = prompt[:8000] if len(prompt) > 8000 else " ".join(prompt.split()[:2000])
        reply = call_groq_model(prompt)
        st.session_state.messages.append(("bot", reply))
        st.session_state.show_summary_input = False

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.pending_input = user_input.strip()

if st.session_state.pending_input:
    input_text = st.session_state.pending_input
    if st.session_state.faiss_index:
        context = retrieve_relevant_chunks_faiss(input_text, st.session_state.faiss_index, st.session_state.chunk_map)
        prompt = f"Context:\n{context}\n\nUser Query: {input_text}\n\nAnswer:"
    else:
        prompt = input_text

    reply = call_groq_model(prompt)
    st.session_state.messages.append(("user", input_text))
    st.session_state.messages.append(("bot", reply))
    st.session_state.pending_input = ""

# Chat Display
chat_html = """<div id="chatbox" class="chat-container">"""
for sender, msg in st.session_state.messages:
    chat_html += f'<div class="{sender}-message">{html.escape(msg)}</div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)
st.markdown("""<script>var chatbox = document.getElementById("chatbox");if(chatbox){chatbox.scrollTop = chatbox.scrollHeight;}</script>""", unsafe_allow_html=True)
