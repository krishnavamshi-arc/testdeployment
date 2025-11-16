import os
import re
import json
import uuid
import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# API Configuration
# ----------------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAAlV9eAGcg4yShhU0o6CE0-cFKPV8FsnY")

# Qdrant Configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

if not GEMINI_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY not set in environment variables")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("⚠️ Warning: QDRANT_URL and QDRANT_API_KEY not set in environment variables")

# ----------------------------
# Initialize Global Resources
# ----------------------------
qdrant_client = None
embedder = None

def initialize_resources():
    """Initialize Qdrant client and embedder"""
    global qdrant_client, embedder
    
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
    
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Helper Functions
# ----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-\s+\n", "-", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def read_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return clean_text(" ".join(pages))

def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    return splitter.split_text(text)

def create_qdrant_collection(collection_name: str, vector_size: int = 384):
    """Create a new Qdrant collection"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name in collection_names:
            qdrant_client.delete_collection(collection_name=collection_name)
        
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return True
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

def index_to_qdrant(chunks: list, collection_name: str):
    """Index text chunks to Qdrant"""
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "chunk_index": idx
            }
        )
        points.append(point)
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )

def retrieve_from_qdrant(collection_name: str, query: str, top_k: int = 5):
    """Retrieve top-k most relevant chunks from Qdrant"""
    query_vector = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k
    )
    
    results = []
    for hit in search_result:
        results.append({
            "text": hit.payload["text"],
            "score": hit.score,
            "chunk_index": hit.payload["chunk_index"]
        })
    
    return results

def build_prompt(query: str, contexts: list, max_context_chars: int = 6000):
    ctx = ""
    for item in contexts:
        chunk = item["text"]
        if len(ctx) + len(chunk) + 2 > max_context_chars:
            break
        ctx += chunk + "\n\n"
    
    prompt = (
        f"You are a helpful assistant answering questions based on the provided context.\n\n"
        f"Context from the document:\n{ctx}\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"- Answer based ONLY on the context provided above\n"
        f"- Be concise and accurate\n"
        f"- If the answer is not in the context, say 'I cannot find this information in the document'\n"
        f"- Provide specific details when available\n\n"
        f"Answer:"
    )
    return prompt

def call_google_ai_api(prompt: str, temperature: float = 0.3):
    """Call Google AI REST API"""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1024,
            "topP": 0.8,
            "topK": 40
        }
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=body, timeout=30)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "No response text.")
            return "No response generated."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception while calling API: {e}"

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    """Main page"""
    collection_name = session.get('collection_name')
    num_chunks = session.get('num_chunks', 0)
    
    # Get Qdrant status
    qdrant_status = "disconnected"
    num_collections = 0
    try:
        if qdrant_client:
            collections = qdrant_client.get_collections()
            qdrant_status = "connected"
            num_collections = len(collections.collections)
    except:
        pass
    
    return render_template('index.html', 
                         collection_name=collection_name,
                         num_chunks=num_chunks,
                         qdrant_status=qdrant_status,
                         num_collections=num_collections)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and indexing"""
    if 'pdf' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['pdf']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Get parameters from form
        chunk_size = int(request.form.get('chunk_size', 600))
        chunk_overlap = int(request.form.get('chunk_overlap', 120))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read and process PDF
            text = read_pdf(filepath)
            if not text:
                flash('Could not extract text from the PDF', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if len(chunks) == 0:
                flash('No text chunks were created', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Create collection and index
            collection_name = f"pdf_{filename.replace('.pdf', '').replace(' ', '_').lower()}_{hash(filename) % 10000}"
            
            if create_qdrant_collection(collection_name, vector_size=384):
                index_to_qdrant(chunks, collection_name)
                session['collection_name'] = collection_name
                session['num_chunks'] = len(chunks)
                session['pdf_name'] = filename
                flash(f'Successfully indexed {len(chunks)} chunks!', 'success')
            else:
                flash('Error creating Qdrant collection', 'error')
            
            # Clean up uploaded file
            os.remove(filepath)
            
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}', 'error')
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        flash('Invalid file type. Only PDF files are allowed', 'error')
    
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question answering"""
    query = request.form.get('query', '').strip()
    
    if not query:
        flash('Please enter a question', 'error')
        return redirect(url_for('index'))
    
    collection_name = session.get('collection_name')
    if not collection_name:
        flash('Please upload a PDF first', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get parameters
        top_k = int(request.form.get('top_k', 5))
        temperature = float(request.form.get('temperature', 0.3))
        
        # Retrieve and generate
        contexts = retrieve_from_qdrant(collection_name, query, top_k=top_k)
        prompt = build_prompt(query, contexts)
        answer = call_google_ai_api(prompt, temperature=temperature)
        
        # Store in session
        session['last_query'] = query
        session['last_answer'] = answer
        session['last_contexts'] = contexts
        
    except Exception as e:
        flash(f'Error processing question: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    """Reset session"""
    session.clear()
    flash('Session reset successfully', 'success')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'qdrant_connected': qdrant_client is not None,
        'embedder_loaded': embedder is not None
    })

# ----------------------------
# Initialize and Run
# ----------------------------
if __name__ == '__main__':
    initialize_resources()
    app.run(debug=True, host='0.0.0.0', port=5000)
