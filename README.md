# Multilingual Chatbot Project Backend

Welcome to the **Mardi Chatbot Backend** repository! This project is designed to support a multilingual chatbot with robust admin panel functionalities for document and URL management, enabling seamless integration into a broader web ecosystem.

## Project Overview
This backend application facilitates the integration of documents and web content into a chatbot’s knowledge base. Key features include secure admin authentication, document uploads, URL processing, and content management with real-time updates. The system is built using **Flask** with various integrated Python libraries for NLP, file processing, and more.

## Features

### ✅ **Admin Authentication**
- Secure login system ensuring that only authorized users can access the admin panel.
- **Framework**: Flask-Login for user session management.

### ✉️ **Document Uploads**
- Supports **DOCX** and **PDF** formats.
- Extracts and indexes textual content for the chatbot's knowledge base.
- Uses **Apache Tika** and **python-docx** for content extraction.

### 📁 **Knowledge Base Initialization**
- Utilizes **FAISS** for efficient semantic search and indexing.
- Embeddings created with **SentenceTransformer**.

### 📂 **URL Content Processing**
- Extracts and indexes content from external URLs.
- Integrates with **BeautifulSoup** for HTML parsing.

### 💎 **Enhanced NLP Capabilities**
- Summarization and text refinement powered by **Hugging Face Transformers**.
- Spacy-based sentence splitting for content processing.

### 🌐 **Multilingual Support** (Planned)
- Future updates to include support for handling content across multiple languages.

## Technologies Used
- **Framework**: Flask, Flask-Login, Flask-CORS, Flask-Limiter, Flask-WTF
- **NLP**: Spacy, Hugging Face Transformers
- **Search and Indexing**: FAISS, SentenceTransformer
- **File Processing**: Apache Tika, python-docx
- **Web Parsing**: BeautifulSoup, Requests
- **Security**: CSRF Protection, User Authentication

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anukitsim/mardi_chatbot_backend.git
   cd mardi_chatbot_backend
   ```

2. **Set up your environment**:
   Ensure Python 3.x and `pip` are installed. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Key Functionalities Explained

### Admin Authentication
- **Secure login** for accessing the admin panel, managed via Flask-Login.
- Admin credentials are stored securely with hashed passwords using **Werkzeug**.

### Document Processing
- Extracts content from **DOCX** and **PDF** files using Apache Tika and python-docx.
- Sentences are encoded and stored in **FAISS** for efficient retrieval.

### URL Integration
- Parses HTML content from URLs, allowing admin users to expand the knowledge base with web content.
- Supports URL management with add and delete functionalities.

### Knowledge Base Summary
- Provides a paginated overview of indexed content, detailing sources (document or URL).

## File Structure
```
mardi_chatbot_backend/
├── app.py
├── templates/
│   ├── admin.html
│   ├── login.html
│   └── knowledge_base_summary.html
├── static/
├── utils/
│   ├── document_processing.py
│   └── url_processing.py
├── data/
│   └── uploaded_documents/
└── requirements.txt
```

## Testing and Verification
### Admin Authentication
- **Tested with correct and incorrect credentials**, ensuring secure access.
- Successful login redirects to the admin panel; failed attempts display error messages.

### File Uploads
- **Verified** document uploads with DOCX and PDF formats.
- Uploaded files are processed and indexed, with confirmation messages.

### URL Management
- **Validated** URL content extraction and integration into the knowledge base.
- Checked for user-friendly error handling with invalid URLs.

### Knowledge Base Summary
- **Tested** pagination for easy navigation of content entries.
- Ensured accurate source indicators (Document/URL).

## Future Improvements
- ✨ **Multilingual capabilities** for indexing and processing content in multiple languages.
- ⚙️ **Chatbot Integration** with a user-facing front end for real-time queries.


## Contact
✉️ For more information, please contact:
- **Author**: Anuki Tsimintia
- **Email**: anukacim@gmail.com
- **LinkedIn**: https://linkedin.com/in/anuki-tsimintia

 🚀

