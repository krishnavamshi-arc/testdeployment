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
HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en"

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
def get_embeddings(texts, max_retries=3):
    """Get embeddings from HuggingFace API with retries"""
    if not HF_API_KEY:
        print("HF_API_KEY not set")
        return None
        
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    embeddings = []
    
    for text_idx, text in enumerate(texts):
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Clean and limit text
                clean = text.strip()[:500]
                if not clean:
                    embeddings.append([0.0] * 384)
                    success = True
                    continue
                
                print(f"Getting embedding {text_idx + 1}/{len(texts)}, attempt {retry_count + 1}")
                
                # HuggingFace Inference API format
                payload = {
                    "inputs": clean,
                    "options": {"wait_for_model": True}
                }
                
                response = requests.post(
                    HF_EMBED_URL, 
                    headers=headers, 
                    json=payload,
                    timeout=60
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Result type: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
                    
                    # HuggingFace feature extraction returns array of shape [1, 384] or just [384]
                    embedding = None
                    
                    if isinstance(result, list):
                        if len(result) > 0:
                            # Check structure
                            if isinstance(result[0], (int, float)):
                                # Flat array [0.1, 0.2, ...]
                                embedding = result
                                print(f"✓ Got flat embedding, length: {len(result)}")
                            elif isinstance(result[0], list) and len(result[0]) > 0:
                                if isinstance(result[0][0], (int, float)):
                                    # Nested [[0.1, 0.2, ...]]
                                    embedding = result[0]
                                    print(f"✓ Got nested embedding, length: {len(result[0])}")
                                elif isinstance(result[0][0], list):
                                    # Double nested [[[0.1, ...]]]
                                    embedding = result[0][0]
                                    print(f"✓ Got double nested embedding, length: {len(result[0][0])}")
                    
                    if embedding and len(embedding) == 384:
                        embeddings.append(embedding)
                        success = True
                        print(f"✓ Embedding {text_idx + 1} successful")
                    else:
                        print(f"Invalid embedding: {embedding[:5] if embedding else 'None'}")
                        retry_count += 1
                        
                elif response.status_code == 503:
                    print(f"Model loading (503), waiting 10 seconds...")
                    import time
                    time.sleep(10)
                    retry_count += 1
                    
                elif response.status_code == 401 or response.status_code == 403:
                    print(f"Authentication error ({response.status_code})")
                    print(f"Response: {response.text[:200]}")
                    return None
                    
                else:
                    print(f"HF Error {response.status_code}: {response.text[:300]}")
                    retry_count += 1
                    
            except Exception as e:
                print(f"Exception: {e}")
                import traceback
                traceback.print_exc()
                retry_count += 1
                if retry_count < max_retries:
                    import time
                    time.sleep(2)
        
        if not success:
            print(f"Failed after {max_retries} attempts")
            return None
    
    print(f"✓ Successfully generated {len(embeddings)} embeddings")
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
    """Simple text splitter - splits by characters with overlap"""
    if not text:
        return []
    
    chunks = []
    chunk_overlap = min(100, chunk_size // 5)  # 20% overlap
    
    # Split by sentences first for better coherence
    sentences = []
    for separator in ['. ', '! ', '? ', '\n']:
        if separator in text:
            parts = text.split(separator)
            for part in parts:
                if part.strip():
                    sentences.append(part.strip() + separator.strip())
            break
    
    # If no sentence separators found, split by words
    if not sentences:
        words = text.split()
        sentences = [' '.join(words[i:i+50]) for i in range(0, len(words), 50)]
    
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds chunk_size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous chunk
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Limit total chunks for Vercel/free tier
    if len(chunks) > 30:
        chunks = chunks[:30]
    
    print(f"Split text into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks

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
    
    print(f"Got {len(embeddings)} embeddings, expected {len(chunks)}")
    
    if len(embeddings) != len(chunks):
        print(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
        return False
    
    try:
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Debug: check embedding structure
            print(f"Chunk {idx}: embedding type={type(embedding)}, len={len(embedding) if isinstance(embedding, list) else 'N/A'}")
            
            # Validate embedding is a list of numbers
            if not isinstance(embedding, list):
                print(f"ERROR: Embedding at index {idx} is not a list, it's {type(embedding)}")
                return False
            
            if len(embedding) == 0:
                print(f"ERROR: Embedding at index {idx} is empty")
                return False
            
            # Check if all elements are numbers
            if not all(isinstance(x, (int, float)) for x in embedding[:5]):  # Check first 5
                print(f"ERROR: Embedding at index {idx} contains non-numeric values")
                return False
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk, "chunk_index": idx}
            ))
        
        print(f"✓ Created {len(points)} points, now upserting to Qdrant...")
        client.upsert(collection_name=collection_name, points=points)
        print("✓ Successfully indexed to Qdrant")
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
        f"Answer based context above and elaborate a little for better understanding for the user. Be concise.\n\n"
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
        
        print(f"Extracted text length: {len(text)} characters")
        
        # Split into chunks with user-specified size
        chunk_size = int(request.form.get('chunk_size', 600))
        chunks = split_text_simple(text, chunk_size=chunk_size)
        
        if not chunks:
            flash('No text chunks created', 'error')
            return redirect(url_for('index'))
        
        print(f"Created {len(chunks)} chunks")
        
        # Show sample of first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i} length: {len(chunk)} chars, preview: {chunk[:100]}...")
        
        # Limit chunks for free tier
        if len(chunks) > 30:
            original_count = len(chunks)
            chunks = chunks[:30]
            flash(f'Limited to first 30 chunks (document had {original_count} chunks)', 'warning')
        
        # Create collection and index
        safe_name = secure_filename(file.filename).replace('.pdf', '').replace('.', '_')
        collection_name = f"pdf_{safe_name}_{hash(file.filename) % 10000}"
        
        print(f"Creating collection: {collection_name}")
        
        if create_collection(collection_name):
            print("Collection created, starting indexing...")
            if index_chunks(chunks, collection_name):
                session['collection_name'] = collection_name
                session['num_chunks'] = len(chunks)
                flash(f'Successfully indexed {len(chunks)} chunks! Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars', 'success')
            else:
                flash('Error indexing chunks. Check HuggingFace API key and try again.', 'error')
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
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    test_input = "Hello world"
    
    try:
        # Test with HuggingFace Inference API
        payload = {
            "inputs": test_input,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(
            HF_EMBED_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'status_code': response.status_code,
                'error_message': response.text,
                'url_used': HF_EMBED_URL,
                'model': HF_MODEL
            })
        
        result = response.json()
        
        # Extract embedding
        embedding = None
        structure = "unknown"
        
        if isinstance(result, list):
            if len(result) > 0:
                if isinstance(result[0], (int, float)):
                    embedding = result
                    structure = "flat"
                elif isinstance(result[0], list) and len(result[0]) > 0:
                    if isinstance(result[0][0], (int, float)):
                        embedding = result[0]
                        structure = "nested"
                    elif isinstance(result[0][0], list):
                        embedding = result[0][0]
                        structure = "double_nested"
        
        return jsonify({
            'success': True,
            'status_code': response.status_code,
            'structure': structure,
            'embedding_length': len(embedding) if embedding else 0,
            'first_5_values': embedding[:5] if embedding else 'N/A',
            'works': embedding is not None and len(embedding) == 384,
            'url_used': HF_EMBED_URL,
            'model': HF_MODEL
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc(),
            'url_used': HF_EMBED_URL
        })

@app.route('/debug-upload', methods=['POST'])
def debug_upload():
    """Debug version of upload that shows detailed info"""
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file'})
    
    file = request.files['pdf']
    
    try:
        # Read PDF
        file_bytes = file.read()
        text = read_pdf_from_bytes(file_bytes)
        
        if not text:
            return jsonify({'error': 'No text extracted'})
        
        # Create just 2 chunks for testing
        chunks = split_text_simple(text, chunk_size=500)[:2]
        
        # Get embeddings
        embeddings = get_embeddings(chunks)
        
        if not embeddings:
            return jsonify({'error': 'Failed to get embeddings', 'chunks': chunks})
        
        return jsonify({
            'success': True,
            'num_chunks': len(chunks),
            'num_embeddings': len(embeddings),
            'embedding_0_type': str(type(embeddings[0])),
            'embedding_0_length': len(embeddings[0]) if isinstance(embeddings[0], list) else 'N/A',
            'embedding_0_sample': str(embeddings[0][:5]) if isinstance(embeddings[0], list) else str(embeddings[0])[:100],
            'chunks_preview': [c[:100] for c in chunks]
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

# For local testing
if __name__ == '__main__':
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

