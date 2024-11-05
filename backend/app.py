from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory, abort
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import CSRFProtect
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tika import parser
from docx import Document
import requests
from bs4 import BeautifulSoup
import logging
import pickle
import json
from datetime import datetime
import atexit
import spacy
import torch

# ----------------------------
# 1. Environment Setup
# ----------------------------

import tika
tika.initVM()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploaded_documents')
os.makedirs(UPLOAD_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

# Initialize NLP Models
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
nlp.add_pipe("sentencizer")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
response_refiner = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_here')
csrf = CSRFProtect(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. User Management
# ----------------------------

class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {
    'admin': {'password': generate_password_hash('admin')}
}

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# ----------------------------
# 4. Knowledge Base Initialization
# ----------------------------

faiss_index = None
kb_list = []
embedder = None
qa_pipeline = None
url_list = []

def initialize():
    global faiss_index, kb_list, url_list, embedder, qa_pipeline

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = embedder.get_sentence_embedding_dimension()

    qa_pipeline = pipeline(
        'question-answering',
        model='distilbert-base-cased-distilled-squad',
        tokenizer='distilbert-base-cased-distilled-squad'
    )

    # Load the FAISS index and knowledge base list from disk if available
    faiss_index_path = os.path.join(DATA_DIR, 'faiss_index.pkl')
    kb_list_path = os.path.join(DATA_DIR, 'kb_list.pkl')
    url_list_path = os.path.join(DATA_DIR, 'url_list.pkl')

    if os.path.exists(faiss_index_path) and os.path.exists(kb_list_path):
        try:
            with open(faiss_index_path, 'rb') as f:
                faiss_index = pickle.load(f)
            with open(kb_list_path, 'rb') as f:
                kb_list = pickle.load(f)
            logger.info("Knowledge base loaded from disk.")
        except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load knowledge base from disk: {e}. Reinitializing.")
            faiss_index = faiss.IndexFlatL2(dimension)
            kb_list = []
    else:
        faiss_index = faiss.IndexFlatL2(dimension)
        kb_list = []
        logger.info("Initialized empty knowledge base.")

    # Load the URL list from disk if available
    if os.path.exists(url_list_path):
        try:
            with open(url_list_path, 'rb') as f:
                url_list = pickle.load(f)
            logger.info(f"URL list loaded from disk. Current URLs: {[entry['url'] for entry in url_list]}")
        except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load URL list from disk: {e}. Initializing empty URL list.")
            url_list = []
    else:
        url_list = []
        logger.info("Initialized empty URL list.")

initialize()

# ----------------------------
# 5. Document and URL Processing Functions
# ----------------------------

def process_document(filepath):
    logger.info(f"Processing document: {filepath}")
    try:
        if filepath.lower().endswith('.pdf'):
            raw = parser.from_file(filepath)
            text = raw['content'] or ''
        elif filepath.lower().endswith('.docx'):
            doc = Document(filepath)
            text = '\n'.join([para.text for para in doc.paragraphs])
        else:
            flash(f"Unsupported file type: {filepath}", 'error')
            return

        if not text.strip():
            flash(f"No text extracted from document: {filepath}", 'error')
            return

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        global faiss_index, kb_list, embedder
        embeddings = embedder.encode(sentences, show_progress_bar=True)
        faiss_index.add(np.array(embeddings).astype('float32'))
        kb_list.extend(sentences)

        # Save the updated knowledge base files
        faiss_index_path = os.path.join(DATA_DIR, 'faiss_index.pkl')
        kb_list_path = os.path.join(DATA_DIR, 'kb_list.pkl')

        with open(faiss_index_path, 'wb') as f:
            pickle.dump(faiss_index, f)
        with open(kb_list_path, 'wb') as f:
            pickle.dump(kb_list, f)
    except Exception as e:
        logger.error(f"Error processing document '{filepath}': {e}")
        flash(f"Error processing document '{filepath}': {e}", 'error')

def process_url(url):
    normalized_url = url.strip().lower().rstrip('/')
    logger.info(f"Processing URL: {normalized_url}")

    try:
        # Attempt to send a request to the URL
        response = requests.get(normalized_url, timeout=5)

        # Check if the response is OK (200)
        if response.status_code != 200:
            flash("Failed to retrieve content from the URL. Please check the URL and try again.", 'error')
            return False

        # Extract text from HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join([para.get_text() for para in paragraphs])

        # Check if any content was extracted
        if not text.strip():
            flash("No readable content found at the provided URL. Please check the URL.", 'error')
            return False

        # Process the extracted text
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        global faiss_index, kb_list, url_list, embedder
        embeddings = embedder.encode(sentences, show_progress_bar=True)
        faiss_index.add(np.array(embeddings).astype('float32'))
        kb_list.extend(sentences)

        # Add the URL to url_list and save to disk
        url_list.append({"url": normalized_url, "sentences": sentences})

        # Save the updated knowledge base files
        faiss_index_path = os.path.join(DATA_DIR, 'faiss_index.pkl')
        kb_list_path = os.path.join(DATA_DIR, 'kb_list.pkl')
        url_list_path = os.path.join(DATA_DIR, 'url_list.pkl')

        with open(url_list_path, 'wb') as f:
            pickle.dump(url_list, f)
        with open(faiss_index_path, 'wb') as f:
            pickle.dump(faiss_index, f)
        with open(kb_list_path, 'wb') as f:
            pickle.dump(kb_list, f)

        return True

    except requests.exceptions.MissingSchema:
        flash("Invalid URL format. Please ensure it starts with http:// or https://.", 'error')
    except requests.exceptions.ConnectionError:
        flash("Unable to connect to the URL. Please check if the URL is correct and reachable.", 'error')
    except requests.exceptions.Timeout:
        flash("The request to the URL timed out. Please try again later.", 'error')
    except Exception as e:
        logger.error(f"Unexpected error retrieving URL '{normalized_url}': {e}")
        flash("An unexpected error occurred while processing the URL. Please try again.", 'error')

    return False  # Indicate failure in case of any exception

# ----------------------------
# 6. Ensure Knowledge Base is Saved on Exit
# ----------------------------

def save_kb_on_exit():
    try:
        faiss_index_path = os.path.join(DATA_DIR, 'faiss_index.pkl')
        kb_list_path = os.path.join(DATA_DIR, 'kb_list.pkl')

        with open(faiss_index_path, 'wb') as f:
            pickle.dump(faiss_index, f)
        with open(kb_list_path, 'wb') as f:
            pickle.dump(kb_list, f)
        logger.info("Knowledge base saved on exit.")
    except Exception as e:
        logger.error(f"Error saving knowledge base on exit: {e}")

atexit.register(save_kb_on_exit)

# ----------------------------
# 7. Routes
# ----------------------------

@app.route('/admin')
@login_required
def admin_panel():
    total_entries = faiss_index.ntotal if faiss_index else 0
    uploaded_files = get_uploaded_files()

    # Load URLs from url_list.pkl if it exists
    url_list_path = os.path.join(DATA_DIR, 'url_list.pkl')
    if os.path.exists(url_list_path):
        with open(url_list_path, 'rb') as f:
            url_list = pickle.load(f)
    else:
        url_list = []

    return render_template('admin.html', total_entries=total_entries, uploaded_files=uploaded_files, url_list=url_list)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and check_password_hash(users[username]['password'], password):
            login_user(User(username))
            flash('Logged in successfully.', 'success')
            return redirect(url_for('admin_panel'))
        else:
            flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/upload_document', methods=['POST'])
@login_required
def upload_document_route():
    if 'document' not in request.files:
        flash('No file part in the upload request.', 'error')
        return redirect(url_for('admin_panel'))

    file = request.files['document']
    if file.filename == '':
        flash('No selected file.', 'error')
        return redirect(url_for('admin_panel'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        process_document(filepath)
        flash('File uploaded and knowledge base updated.', 'success')
        return redirect(url_for('admin_panel'))

@app.route('/add_url', methods=['POST'])
@login_required
def add_url_route():
    url = request.form.get('url')
    if not url:
        flash('No URL provided.', 'error')
        return redirect(url_for('admin_panel'))

    # Attempt to process the URL and check if it succeeded
    if process_url(url):
        flash('URL content added and knowledge base updated.', 'success')
    else:
        flash("Failed to add URL content to the knowledge base.", 'error')

    return redirect(url_for('admin_panel'))

@app.route('/delete_file/<filename>', methods=['POST'])
@login_required
def delete_file(filename):
    filepath = os.path.join(UPLOAD_DIR, secure_filename(filename))

    if os.path.exists(filepath):
        os.remove(filepath)
        flash(f"Document '{filename}' deleted successfully.", 'success')

        # Refresh the knowledge base after deletion
        refresh_knowledge_base()
    else:
        flash(f"File '{filename}' does not exist.", 'error')

    return redirect(url_for('admin_panel'))

@app.route('/download_file/<filename>', methods=['GET'])
@login_required
def download_file_route(filename):
    return send_from_directory(directory=UPLOAD_DIR, path=secure_filename(filename), as_attachment=True)

@app.route('/knowledge_base', methods=['GET'])
@login_required
def knowledge_base_route():
    """
    Returns a structured HTML summary of the full knowledge base with pagination.
    """
    entries_per_page = 10  # Set the number of entries to display per page
    page = request.args.get('page', 1, type=int)  # Get the page number from the URL query parameter

    # Combine document and URL entries into a single list
    sample_entries = [{'content': entry, 'source': 'Document'} for entry in kb_list]
    for url_entry in url_list:
        for sentence in url_entry['sentences']:
            sample_entries.append({'content': sentence, 'source': url_entry['url']})

    # Calculate total pages
    total_entries = len(sample_entries)
    total_pages = (total_entries + entries_per_page - 1) // entries_per_page

    # Paginate entries
    start = (page - 1) * entries_per_page
    end = start + entries_per_page
    paginated_entries = sample_entries[start:end]

    # Render paginated knowledge base summary
    return render_template(
        'knowledge_base_summary.html',
        sample_entries=paginated_entries,
        total_entries=total_entries,
        total_pages=total_pages,
        current_page=page
    )

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
@csrf.exempt
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        language = data.get('language', 'en')

        # Placeholder response until Rasa is integrated
        response_text = "Chatbot functionality is under development. Please try again later."
        return jsonify({"responses": [response_text]})
    except Exception as e:
        logger.error(f"Unexpected error in /chat route: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/delete_url/<path:url>', methods=['POST'])
@login_required
def delete_url(url):
    global faiss_index, kb_list, url_list

    normalized_url = url.strip().lower().rstrip('/')
    logger.info(f"Attempting to delete URL: {normalized_url}")
    logger.info(f"Current URLs in url_list: {[entry['url'] for entry in url_list]}")

    # Find and remove the URL entry
    url_entry = next((entry for entry in url_list if entry['url'] == normalized_url), None)

    if url_entry:
        for sentence in url_entry['sentences']:
            if sentence in kb_list:
                kb_list.remove(sentence)

        # Update url_list and save it
        url_list = [entry for entry in url_list if entry['url'] != normalized_url]
        url_list_path = os.path.join(DATA_DIR, 'url_list.pkl')
        with open(url_list_path, 'wb') as f:
            pickle.dump(url_list, f)

        # Refresh the knowledge base to reflect the deletion
        refresh_knowledge_base()

        flash(f"URL '{url}' and its content deleted successfully.", 'success')
    else:
        flash(f"URL '{url}' not found.", 'error')

    return redirect(url_for('admin_panel'))

# ----------------------------
# 8. Helper Functions
# ----------------------------

def get_uploaded_files():
    files = []
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath):
                upload_time = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                files.append({'filename': filename, 'uploaded_at': upload_time})
    return files

def refresh_knowledge_base():
    """
    Reinitializes the FAISS index and knowledge base list by reprocessing all uploaded documents and URLs.
    If no documents or URLs exist, it clears the index and knowledge base list.
    """
    global faiss_index, kb_list, url_list, embedder

    # Reset the FAISS index and knowledge base list
    dimension = embedder.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatL2(dimension)
    kb_list = []

    # Reload documents
    if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath):
                process_document(filepath)  # Process each document

    # Reload URLs
    url_list_path = os.path.join(DATA_DIR, 'url_list.pkl')
    if os.path.exists(url_list_path):
        with open(url_list_path, 'rb') as f:
            url_list = pickle.load(f)
        for entry in url_list:
            sentences = entry['sentences']
            embeddings = embedder.encode(sentences, show_progress_bar=False)
            faiss_index.add(np.array(embeddings).astype('float32'))
            kb_list.extend(sentences)

    # Save updated knowledge base files
    faiss_index_path = os.path.join(DATA_DIR, 'faiss_index.pkl')
    kb_list_path = os.path.join(DATA_DIR, 'kb_list.pkl')

    with open(faiss_index_path, 'wb') as f:
        pickle.dump(faiss_index, f)
    with open(kb_list_path, 'wb') as f:
        pickle.dump(kb_list, f)
    with open(url_list_path, 'wb') as f:
        pickle.dump(url_list, f)

    logger.info("Knowledge base has been refreshed based on existing files and URLs.")

# ----------------------------
# 9. Run the Application
# ----------------------------

if __name__ == "__main__":
    app.run(debug=False)
