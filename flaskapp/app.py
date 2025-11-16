from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import os
import re
import json
import uuid
import requests
from io import BytesIO
from werkzeug.utils import secure_filename

# Only import what's available
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except:
    QDRANT_AVAILABLE = False

# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-12345')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ----------------------------
# API Configuration
# ----------------------------
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

# Initialize Qdrant (lazy loading)
_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None and QDRANT_AVAILABLE and QDRANT_URL and QDRANT_API_KEY:
        try:
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        except Exception as e:
            print(f"Qdrant connection error: {e}")
            _qdrant_client = False
    return _qdrant_client if _qdrant_client else None

# ----------------------------
# Helper Functions
# ----------------------------
def get_embeddings(texts):
    """Get embeddings from HuggingFace API"""
    if not HF_API_KEY:
        print("HF_API_KEY not set")
        return None
        
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    embeddings = []
    
    # Process in smaller batches
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        for text in batch:
            try:
                # Clean and limit text
                clean = text.strip()[:500]
                if not clean:
                    continue
                    
                response = requests.post(
                    HF_EMBED_URL, 
                    headers=headers, 
                    json={"inputs": clean, "options": {"wait_for_model": True}},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings.append(result)
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    print("Model loading, waiting...")
                    import time
                    time.sleep(2)
                    response = requests.post(
                        HF_EMBED_URL, 
                        headers=headers, 
                        json={"inputs": clean, "options": {"wait_for_model": True}},
                        timeout=30
                    )
                    if response.status_code == 200:
                        embeddings.append(response.json())
                    else:
                        print(f"HF Error after retry: {response.status_code} - {response.text}")
                        return None
                else:
                    print(f"HF Error: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                print(f"Embedding error: {e}")
                return None
    
    print(f"Generated {len(embeddings)} embeddings")
    return embeddings if embeddings else None

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-\s+\n", "-", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def read_pdf_from_bytes(file_bytes):
    """Read PDF from bytes - Vercel compatible"""
    if not PDF_AVAILABLE:
        return None
    
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages[:50]:  # Limit pages for Vercel
            try:
                text = page.extract_text()
                if text:
                    pages.append(text)
            except:
                continue
        return clean_text(" ".join(pages))
    except Exception as e:
        print(f"PDF read error: {e}")
        return None

def split_text_simple(text: str, chunk_size: int = 500):
    """Simple text splitter - no external dependencies"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks[:50]  # Limit chunks for Vercel

def create_collection(collection_name: str):
    client = get_qdrant_client()
    if not client:
        return False
    
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name in collection_names:
            client.delete_collection(collection_name=collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        return True
    except Exception as e:
        print(f"Collection error: {e}")
        return False

def index_chunks(chunks: list, collection_name: str):
    client = get_qdrant_client()
    if not client:
        print("Qdrant client not available")
        return False
    
    print(f"Indexing {len(chunks)} chunks...")
    
    embeddings = get_embeddings(chunks)
    if not embeddings:
        print("Failed to get embeddings")
        return False
    
    if len(embeddings) != len(chunks):
        print(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
        return False
    
    try:
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Validate embedding is a list of numbers
            if not isinstance(embedding, list) or len(embedding) == 0:
                print(f"Invalid embedding at index {idx}")
                return False
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk, "chunk_index": idx}
            ))
        
        print(f"Upserting {len(points)} points to Qdrant...")
        client.upsert(collection_name=collection_name, points=points)
        print("Successfully indexed to Qdrant")
        return True
    except Exception as e:
        print(f"Index error: {e}")
        import traceback
        traceback.print_exc()
        return False

def search_qdrant(collection_name: str, query: str, top_k: int = 5):
    client = get_qdrant_client()
    if not client:
        return []
    
    query_embedding = get_embeddings([query])
    if not query_embedding:
        return []
    
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding[0],
            limit=top_k
        )
        
        return [{
            "text": hit.payload["text"],
            "score": hit.score,
            "chunk_index": hit.payload["chunk_index"]
        } for hit in results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def call_gemini(prompt: str, temperature: float = 0.3):
    if not GEMINI_API_KEY:
        return "Error: Google API key not configured"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 800
        }
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=body, timeout=25)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "No response")
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def build_prompt(query: str, contexts: list):
    ctx_text = "\n\n".join([item["text"][:400] for item in contexts[:3]])
    return (
        f"Context from document:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer based only on the context above. Be concise.\n\n"
        f"Answer:"
    )

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    collection_name = session.get('collection_name')
    num_chunks = session.get('num_chunks', 0)
    
    client = get_qdrant_client()
    qdrant_status = "connected" if client else "disconnected"
    num_collections = 0
    
    try:
        if client:
            num_collections = len(client.get_collections().collections)
    except:
        pass
    
    return render_template('index.html',
                         collection_name=collection_name,
                         num_chunks=num_chunks,
                         qdrant_status=qdrant_status,
                         num_collections=num_collections)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['pdf']
    if not file or file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    try:
        print(f"Processing file: {file.filename}")
        
        # Read PDF in memory
        file_bytes = file.read()
        print(f"File size: {len(file_bytes)} bytes")
        
        text = read_pdf_from_bytes(file_bytes)
        
        if not text:
            flash('Could not extract text from PDF', 'error')
            return redirect(url_for('index'))
        
        print(f"Extracted text length: {len(text)} chars")
        
        # Split into chunks
        chunk_size = int(request.form.get('chunk_size', 500))
        chunks = split_text_simple(text, chunk_size=chunk_size)
        
        if not chunks:
            flash('No text chunks created', 'error')
            return redirect(url_for('index'))
        
        print(f"Created {len(chunks)} chunks")
        
        # Limit chunks for free tier
        if len(chunks) > 20:
            chunks = chunks[:20]
            flash(f'Processing first 20 chunks only (free tier limit)', 'warning')
        
        # Create collection and index
        safe_name = secure_filename(file.filename).replace('.pdf', '').replace('.', '_')
        collection_name = f"pdf_{safe_name}_{hash(file.filename) % 10000}"
        
        print(f"Creating collection: {collection_name}")
        
        if create_collection(collection_name):
            print("Collection created, starting indexing...")
            if index_chunks(chunks, collection_name):
                session['collection_name'] = collection_name
                session['num_chunks'] = len(chunks)
                flash(f'Successfully indexed {len(chunks)} chunks!', 'success')
            else:
                flash('Error indexing chunks. Please check: 1) HuggingFace API key is valid, 2) Model is loaded, 3) Try again in a moment', 'error')
        else:
            flash('Error creating collection. Check Qdrant connection.', 'error')
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error processing PDF: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form.get('query', '').strip()
    
    if not query:
        flash('Please enter a question', 'error')
        return redirect(url_for('index'))
    
    collection_name = session.get('collection_name')
    if not collection_name:
        flash('Please upload a PDF first', 'error')
        return redirect(url_for('index'))
    
    try:
        top_k = int(request.form.get('top_k', 5))
        temperature = float(request.form.get('temperature', 0.3))
        
        # Search for relevant chunks
        contexts = search_qdrant(collection_name, query, top_k=top_k)
        
        if contexts:
            # Generate answer
            prompt = build_prompt(query, contexts)
            answer = call_gemini(prompt, temperature=temperature)
            
            session['last_query'] = query
            session['last_answer'] = answer
            session['last_contexts'] = contexts
        else:
            flash('No relevant information found', 'error')
            
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    flash('Session reset successfully', 'success')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'pdf_available': PDF_AVAILABLE,
        'qdrant_available': QDRANT_AVAILABLE,
        'qdrant_connected': get_qdrant_client() is not None,
        'hf_configured': bool(HF_API_KEY),
        'gemini_configured': bool(GEMINI_API_KEY)
    })

@app.route('/test-embed')
def test_embed():
    """Test endpoint to verify embeddings work"""
    if not HF_API_KEY:
        return jsonify({'error': 'HF_API_KEY not set'})
    
    test_text = ["Hello world", "This is a test"]
    embeddings = get_embeddings(test_text)
    
    if embeddings:
        return jsonify({
            'success': True,
            'num_embeddings': len(embeddings),
            'embedding_length': len(embeddings[0]) if embeddings else 0,
            'sample': embeddings[0][:5] if embeddings else []
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to get embeddings'})

# For local testing
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
