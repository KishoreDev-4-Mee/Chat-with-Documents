import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/faiss_index"):
    os.makedirs("data/faiss_index")

import json

# Function to load or create document metadata
def load_metadata():
    metadata_path = "data/document_metadata.txt"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

def save_metadata(metadata_list):
    metadata_path = "data/document_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata_list))

def load_chat_history():
    chat_path = "data/chat_history.json"
    if os.path.exists(chat_path):
        with open(chat_path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(messages):
    chat_path = "data/chat_history.json"
    with open(chat_path, "w") as f:
        json.dump(messages, f)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = load_metadata()
if "vector_store" not in st.session_state:
    # Try to load existing FAISS index
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model="models/embedding-001"
        )
        st.session_state.vector_store = FAISS.load_local(
            "data/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True  # Safe since we created these files
        )
        if st.session_state.processed_docs:
            st.success(f"Loaded existing index with {len(st.session_state.processed_docs)} documents!")
    except Exception:
        st.session_state.vector_store = None

def load_document(file):
    """Load document based on file type"""
    name = file.name
    if name.endswith('.pdf'):
        # Save PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        os.remove("temp.pdf")
    elif name.endswith('.docx'):
        # Save DOCX temporarily
        with open("temp.docx", "wb") as f:
            f.write(file.getbuffer())
        loader = Docx2txtLoader("temp.docx")
        docs = loader.load()
        os.remove("temp.docx")
    elif name.endswith('.txt'):
        # Save TXT temporarily
        with open("temp.txt", "wb") as f:
            f.write(file.getbuffer())
        loader = TextLoader("temp.txt")
        docs = loader.load()
        os.remove("temp.txt")
    else:
        st.error("Unsupported file format")
        return None
    return docs

def process_documents(docs):
    """Process documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def setup_vector_store(chunks, existing_store=None):
    """Set up or update FAISS vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/embedding-001"
    )
    
    if existing_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store = existing_store
        vector_store.add_documents(chunks)
    
    # Save the updated index
    vector_store.save_local("data/faiss_index")
    return vector_store

def get_response_chain():
    """Set up LangChain for chat responses"""
    llm = GoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("MODEL"),
        temperature=0.7
    )
    
    template = """You are a helpful AI assistant that answers questions based on provided documents.
    Use the following context to answer the user's question. If you don't know the answer, say so.
    
    Context: {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": st.session_state.vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# App UI
st.title("Document Chat Assistant")

# Sidebar with document list
with st.sidebar:
    st.header("Processed Documents")
    if st.session_state.processed_docs:
        for doc in st.session_state.processed_docs:
            st.text(f"📄 {doc}")
    else:
        st.text("No documents processed yet")

# File upload
uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Load and process document
        docs = load_document(uploaded_file)
        if docs:
            chunks = process_documents(docs)
            if uploaded_file.name not in st.session_state.processed_docs:
                st.session_state.vector_store = setup_vector_store(
                    chunks, 
                    existing_store=st.session_state.vector_store
                )
                st.session_state.processed_docs.append(uploaded_file.name)
                save_metadata(st.session_state.processed_docs)
                st.success(f"Document '{uploaded_file.name}' processed and added to index successfully!")
            else:
                st.warning(f"Document '{uploaded_file.name}' has already been processed!")

# Chat interface
if not st.session_state.processed_docs:
    st.info("Please upload a document to start chatting. The chat interface will automatically appear once documents are processed.")
else:
    # Display number of available documents
    st.write(f"Ready to chat with {len(st.session_state.processed_docs)} document(s)")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # Make sure vector store is initialized
            if st.session_state.vector_store is None:
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        google_api_key=os.getenv("GOOGLE_API_KEY"),
                        model="models/embedding-001"
                    )
                    st.session_state.vector_store = FAISS.load_local(
                        "data/faiss_index",
                        embeddings,
                        allow_dangerous_deserialization=True  # Safe since we created these files
                    )
                except Exception as e:
                    st.error(f"Error loading document index: {str(e)}")
                    st.stop()
            
            chain = get_response_chain()
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_chat_history(st.session_state.messages)
