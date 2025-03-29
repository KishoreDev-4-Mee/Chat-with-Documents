# Document Chat Assistant

A chatbot application that allows users to upload documents (PDF, DOCX, TXT) and interact with them using natural language queries. Built with LangChain and Gemini.

## Features

- Document upload support (PDF, DOCX, TXT)
- Persistent document storage using FAISS
- Chat history persistence across sessions
- Multi-document context support
- Real-time chat interface

## Setup Instructions

1. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file:
   - Copy `.env.example` to `.env`
   - Fill in your Azure OpenAI credentials and model deployment names

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

4. Configure your `.env` file with:
   - GOOGLE_API_KEY: Your Azure OpenAI API key
   - MODEL: Your Gemini Model Name

## Running the Application

1. Make sure your virtual environment is activated:

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Start the Streamlit app:

```bash
streamlit run app.py
```

The application will:
- Load any existing document index
- Display previously processed documents in the sidebar
- Show previous chat history
- Allow you to upload new documents and chat with them

## Usage

1. Upload a document using the file uploader
2. Wait for the document to be processed
3. Start asking questions about your document in the chat interface
4. Upload additional documents as needed - they will be added to the existing context

## Data Persistence

The application maintains several types of persistent data:
- Document embeddings (stored in `data/faiss_index/`)
- Document metadata (stored in `data/document_metadata.txt`)
- Chat history (stored in `data/chat_history.json`)

This ensures that your documents and conversations are preserved across application restarts.
