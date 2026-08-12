# Edumind

# 🎓 AI-Powered Student Learning Assistant

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27.0-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.14-1C1C1C)
![LangChain](https://img.shields.io/badge/LangChain-Integration-blue)

An intelligent, full-stack educational platform designed to elevate the learning experience for students and streamline material management for faculty. This application leverages advanced AI capabilities, including Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), to automatically process course content and generate interactive study materials, practice questions, and answer contextual student queries.

---

## 🌟 Comprehensive Feature Set

### 👨‍🏫 For Faculty
* **Centralized Material Management:** Securely upload course materials (supports `.pdf`, `.docx`, and `.txt`) organized intuitively by Subject and Unit.
* **Automated Content Generation:** Background processes automatically parse uploaded documents, extract text, and chunk it for AI analysis.
* **Content Preview Dashboard:**
  * **Summaries:** Generate and review comprehensive, structured summaries of entire units.
  * **MCQ Generation:** Automatically create diverse Multiple Choice Questions with varying difficulty levels and detailed explanations.
  * **Flashcards:** Auto-generate question/answer pairs for quick revision.

### 👩‍🎓 For Students
* **Smart Study Dashboard:** Access the AI-generated summaries for any subject and unit to quickly grasp core concepts.
* **Interactive Practice Center:** Test knowledge with AI-generated MCQs, complete with immediate grading and explanatory feedback.
* **Flashcard Revision Module:** Review key concepts quickly using digital flashcards.
* **AI Q&A Assistant (RAG Pipeline):** Ask natural language questions about the course material. The AI uses semantic search to find the exact context within the uploaded documents and provides accurate, sourced answers, preventing hallucinations.

---

## 🏗️ Architecture & Technology Stack

The application is built on a modern decoupled architecture:

### 1. Frontend (User Interface)
* **Framework:** [Streamlit](https://streamlit.io/)
* **Purpose:** Provides an interactive, fast, and responsive user interface for both Faculty and Students. Manages user sessions, authentication state, and API communication.

### 2. Backend (API & Business Logic)
* **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Purpose:** High-performance RESTful API server handling file uploads, user authentication, background processing tasks, and orchestration of the AI services.

### 3. AI & NLP Engine (The "Brain")
* **Orchestration:** [LangChain](https://python.langchain.com/) for building the RAG (Retrieval-Augmented Generation) chains and prompting the LLM.
* **Embeddings:** Local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) used to convert document text into vector representations.
* **LLM Inference:** Powered by Groq for high-speed, high-accuracy natural language generation.

### 4. Data Layer
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) - Stores document chunk embeddings and allows for ultra-fast semantic search during Q&A and content generation.
* **Relational Database:** SQLite - Manages user accounts, credentials, roles, and basic metadata.
* **File System:** Local hierarchical storage for raw uploaded documents (`storage/subjects/<Subject>/<Unit>/docs/`).

---

## 🚀 Getting Started

### Prerequisites
* **Python:** Version 3.9 or higher.
* **API Key:** A Groq API Key is required for the LLM inference.

### Installation & Setup

1. **Clone the repository** (or navigate to your project directory):
   ```bash
   git clone <repository-url>
   cd AI-Powered-Student-Learning-Assistant
   ```

2. **Set up Environment Variables:**
   Create a `.env` file in the root directory of the project and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the Application (Windows):**
   The project includes an automated startup script (`start_all.bat`). This script handles the entire lifecycle:
   * Checks for and creates a Python virtual environment (`venv`).
   * Installs all required dependencies from `requirements.txt`.
   * Initializes the SQLite database.
   * Pre-downloads the HuggingFace embedding models to ensure smooth runtime.
   * Starts the FastAPI backend server (port 8000).
   * Starts the Streamlit frontend application (port 8501).
   
   **To start, simply run:**
   ```cmd
   start_all.bat
   ```

---

## 💻 Usage Guide

### Accessing the Application
Once the `start_all.bat` script is running, open your web browser:
* **Application UI:** [http://localhost:8501](http://localhost:8501)
* **Backend API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

### Default Login Credentials
For testing purposes, the database is initialized with two default accounts:
* **Faculty Account:** 
  * Username: `faculty`
  * Password: `password`
* **Student Account:** 
  * Username: `student`
  * Password: `password`

### Workflow Example
1. Log in as **Faculty**.
2. Navigate to "Upload Materials". Type a Subject (e.g., "Computer Networks") and Unit (e.g., "Unit 1").
3. Upload a relevant PDF document. The backend will process this in the background, extracting text and storing vectors in ChromaDB.
4. Log out and log in as **Student**.
5. Navigate to the "Ask Question" tab, select the subject and unit, and ask a specific question based on the PDF. The AI will answer it accurately based *only* on the provided text!

---

## 📁 Detailed Project Structure

```text
AI-Powered-Student-Learning-Assistant/
│
├── backend/                        # FastAPI Backend Application
│   ├── api/                        # API Router definitions
│   │   ├── auth.py                 # Authentication endpoints
│   │   ├── faculty.py              # Faculty endpoints (uploads)
│   │   └── student.py              # Student endpoints (RAG queries)
│   ├── services/                   # Core business logic
│   │   ├── embedding_service.py    # ChromaDB & Vector management
│   │   ├── file_service.py         # File storage handling
│   │   └── rag_service.py          # Langchain orchestration (LLM interactions)
│   ├── utils/                      # Helper utilities
│   │   ├── hf_embeddings.py        # HuggingFace model wrapper
│   │   └── text_extractor.py       # PDF/DOCX parsing and chunking
│   ├── main.py                     # FastAPI application entry point
│   └── init_db.py                  # Database initialization script
│
├── frontend/                       # Streamlit Frontend Application
│   ├── streamlit_app.py            # Main frontend entry point & routing
│   ├── faculty_dashboard.py        # Faculty UI components
│   └── student_dashboard.py        # Student UI components
│
├── chroma/                         # ChromaDB local vector storage (auto-generated)
├── storage/                        # Raw document upload directory (auto-generated)
├── venv/                           # Python Virtual Environment (auto-generated)
├── .env                            # Environment variables (API keys)
├── requirements.txt                # Python dependencies
└── start_all.bat                   # Windows startup automation script
```

---

## 🛠️ Advanced Details: How the RAG Pipeline Works

When a student asks a question via the UI:
1. **Query Embedding:** The student's question is passed to the backend and converted into a numerical vector using the local `all-MiniLM-L6-v2` model.
2. **Semantic Search:** ChromaDB performs a similarity search, comparing the question vector against all document chunk vectors for that specific Subject/Unit.
3. **Context Retrieval:** The top most relevant text chunks are retrieved from the database.
4. **Prompt Construction:** A LangChain prompt is assembled, containing the student's question and the retrieved text chunks as context.
5. **LLM Generation:** The Groq LLM processes the prompt and generates a concise, accurate answer based strictly on the provided context, preventing hallucinations.

---

## 🔮 Future Enhancements
* **Docker Support:** Containerization for easier cross-platform deployment.
* **Support for Video/Audio Transcripts:** Automatically transcribe and embed multimedia lecture recordings.
* **Student Analytics:** Allow faculty to see which concepts students are struggling with based on MCQ performance and popular questions.
* **Chat History:** Persistent chat interfaces for students to refer back to previous RAG Q&A sessions.

---

*Built with ❤️ for better education.*
