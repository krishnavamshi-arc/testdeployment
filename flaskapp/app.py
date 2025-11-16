import os
import re
import json
import uuid
import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Use /tmp for Vercel
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# API Configuration
# ----------------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

# Qdrant Configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# ----------------------------
# Initialize Global Resources
# ----------------------------
qdrant_client = None

def initialize_resources():
    """Initialize Qdrant client"""
    global qdrant_client
    
    if qdrant_client is None and QDRANT_URL and QDRANT_API_KEY:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

# ----------------------------
# Embedding Functions (OpenAI Alternative)
# ----------------------------
def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings from OpenAI API - Vercel compatible!"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "input": texts,
        "model": model
    }
    
    try:
        response = requests.post(OPENAI_EMBED_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            return embeddings
        else:
            print(f"OpenAI API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

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
