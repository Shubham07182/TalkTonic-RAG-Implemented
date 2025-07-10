from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber
import pytesseract
import docx
from PIL import Image
import io
import requests

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session state
embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
faiss_index = None
chunk_map = {}

# ============================ UTILITIES ============================
def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

def chunk_text(text, chunk_size=400, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_text(file_bytes, content_type):
    if content_type == "application/pdf":
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
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
    elif content_type.startswith("image/"):
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    elif content_type == "text/plain":
        return file_bytes.decode("utf-8")
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    return "Unsupported file type."

def retrieve_chunks(query, top_k=4):
    global faiss_index, chunk_map
    if faiss_index is None:
        return "❌ No file uploaded."

    query_vec = embed_model.encode([query])[0]
    query_vec = normalize_vector(query_vec).astype("float32")
    D, I = faiss_index.search(np.array([query_vec]), top_k)
    return "\n".join([chunk_map.get(i, "") for i in I[0]])

# ========================== ROUTES ==========================

class Query(BaseModel):
    message: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global faiss_index, chunk_map
    try:
        contents = await file.read()
        extracted_text = extract_text(contents, file.content_type)
        chunks = chunk_text(extracted_text)

        embeddings = embed_model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

        return {"preview": extracted_text[:2000]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/query")
def handle_query(data: Query):
    try:
        context = retrieve_chunks(data.message)
        prompt = f"Context:\n{context}\n\nUser Query: {data.message}\n\nAnswer:"
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer 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"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        result = response.json()
        return {"reply": result['choices'][0]['message']['content']}
    except Exception as e:
        return {"error": str(e)}
