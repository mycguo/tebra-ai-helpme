# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal AI chatbot that allows users to chat with their knowledge base using Retrieval-Augmented Generation (RAG). The application is built with Streamlit and supports ingesting various document types (PDFs, Word docs, Excel, text files, audio, video, URLs, and YouTube videos) into a FAISS vector store.

## Development Commands

### Local Development
```bash
# Setup virtual environment
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Testing
```bash
# Run tests
pytest
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t personal-ai-chatbot .
docker run -p 8080:8080 personal-ai-chatbot

# Or use the provided run script
./run.sh
```

## Architecture

### Core Components

- **Main App** (`app.py`): Primary Streamlit interface for querying the knowledge base
- **Admin Interface** (`pages/app_admin.py`): Document ingestion and knowledge base management
- **Web Crawler** (`webcrawer.py`): URL crawling and content extraction
- **Vector Store**: FAISS-based similarity search with Google Generative AI embeddings

### Key Technologies

- **Streamlit**: Web framework for the user interface
- **FAISS**: Vector store for document similarity search
- **LangChain**: RAG pipeline orchestration and document processing
- **Google Generative AI**: Embeddings (models/embedding-001) and chat model (gemini-2.0-flash)
- **AssemblyAI**: Audio transcription service
- **AWS S3**: Vector store persistence and backup

### Data Flow

1. Documents are ingested via the admin interface (`pages/app_admin.py`)
2. Text is extracted and chunked using `RecursiveCharacterTextSplitter` (chunk_size=5000, chunk_overlap=1000)
3. Embeddings are generated using Google's embedding-001 model
4. Documents are stored in FAISS vector store (saved to `faiss_index/`)
5. User queries trigger similarity search and LLM response generation

## Configuration Requirements

### Environment Variables/Secrets
The application requires these secrets (configured via Streamlit secrets):
- `GOOGLE_API_KEY`: For Google Generative AI embeddings and chat
- `NVIDIA_API_KEY`: Alternative LLM provider (currently commented out)
- `ASSEMBLYAI_API_KEY`: For audio/video transcription
- `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`: For S3 vector store backup
- `BUCKET_NAME`: S3 bucket for FAISS index storage

## Important Implementation Details

### Vector Store Management
- FAISS index is stored locally in `faiss_index/` directory
- Vector store has safety mechanisms for loading/saving to prevent corruption
- S3 backup functionality for persistence across deployments

### Document Processing
- Supports multiple file types with metadata extraction
- Audio/video files are transcribed before adding to knowledge base
- URL crawling respects depth limits and handles encoding issues
- YouTube videos require local ffmpeg installation

### Error Handling
- Safe vector store operations with atomic saves
- Graceful handling of malformed documents and network errors
- Encoding detection and fallback for web content

## File Structure Notes

- `pages/`: Streamlit multi-page app structure
- `tests/`: Contains pytest test files with mocking for external dependencies
- `k8s-personal-ai-agent.yaml`: Kubernetes deployment configuration
- Media files (`*.mp3`, `*.mp4`): Test/demo files for audio/video processing