import os
import glob
import streamlit as st
from google import genai
from PIL import Image
import html

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredPowerPointLoader,
    Docx2txtLoader
)
import pandas as pd


api_key = os.environ.get("GEMINI_API_KEY")

# Setup directories
base_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_dir = os.path.join(base_dir, ".faiss_index")

#DATA LOADERS

# 1. Text Loader 
def load_text_langchain(file_path):

    try:
        # LangChain handles opening, encoding, and reading automatically
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        return [Document(page_content=f"Error loading text file: {e}",
                         metadata={"source": file_path, "type": "text"})]

# 2. PDF Loader 
def load_pdf_langchain(file_path):

    try:
        # Replaces your manual PyPDF2 reader and loop
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        return [Document(page_content=f"Error loading PDF file: {e}",
                         metadata={"source": file_path, "type": "pdf"})]

# 3. CSV Loader
def load_csv_langchain(file_path):
    try:
        df = pd.read_csv(file_path)
        documents = []
        
        # Add a summary document
        summary = f"CSV File: {os.path.basename(file_path)}\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n"
        summary += f"Total Rows: {len(df)}\n\n"
        summary += "Sample Data:\n"
        summary += df.head(10).to_string(index=False)
        
        documents.append(Document(
            page_content=summary,
            metadata={"source": file_path, "type": "csv", "rows": len(df)}
        ))
        
        # Convert each row to a document
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(Document(
                page_content=row_text,
                metadata={"source": file_path, "type": "csv_row", "row_number": idx + 1}
            ))
        
        return documents
    except Exception as e:
        return [Document(page_content=f"Error loading CSV file: {e}",
                         metadata={"source": file_path, "type": "csv"})]

# 3b. Excel Loader
def load_excel_langchain(file_path):
    try:
        df = pd.read_excel(file_path)
        documents = []
        
        # Add a summary document
        summary = f"Excel File: {os.path.basename(file_path)}\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n"
        summary += f"Total Rows: {len(df)}\n\n"
        summary += "Sample Data:\n"
        summary += df.head(10).to_string(index=False)
        
        documents.append(Document(
            page_content=summary,
            metadata={"source": file_path, "type": "excel", "rows": len(df)}
        ))
        
        # Convert each row to a document
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(Document(
                page_content=row_text,
                metadata={"source": file_path, "type": "excel_row", "row_number": idx + 1}
            ))
        
        return documents
    except Exception as e:
        return [Document(page_content=f"Error loading Excel file: {e}",
                         metadata={"source": file_path, "type": "excel"})]

# 3c. PowerPoint Loader
def load_pptx_langchain(file_path):
    try:
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    except Exception as e:
        return [Document(page_content=f"Error loading PowerPoint file: {e}",
                         metadata={"source": file_path, "type": "pptx"})]

# 3d. Word Document Loader
def load_docx_langchain(file_path):
    try:
        loader = Docx2txtLoader(file_path)
        return loader.load()
    except Exception as e:
        return [Document(page_content=f"Error loading Word document: {e}",
                         metadata={"source": file_path, "type": "docx"})]

# 4. Image Loader
def load_image_as_document(file_path, client):
    
    try:
        image = Image.open(file_path)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=["Extract all text and describe the content of this image in detail:", image]
        )
        
        # Safe text extraction
        text_out = getattr(response, "text", None)
        if text_out is None:
            try:
                text_out = response.candidates[0].content.parts[0].text
            except:
                text_out = str(response)

        # Wrap in LangChain Document so it works with the pipeline
        return [Document(page_content=text_out, metadata={"source": file_path, "type": "image"})]
    except Exception as e:
        # Return error as a Document so the pipeline doesn't crash
        return [Document(page_content=f"Error processing image: {e}", metadata={"source": file_path})]

# 5. Video Loader
def load_video_as_document(file_path, client):
    """Extract information from video using Gemini's multimodal capabilities."""
    try:
        # Upload video file to Gemini
        with open(file_path, 'rb') as video_file:
            video_data = video_file.read()
        
        # Use Gemini to analyze the video
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[
                "Analyze this video and provide a detailed description including: "
                "1) Main subject and activities, "
                "2) Key visual elements, "
                "3) Any text visible in the video, "
                "4) Overall context and setting.",
                {"mime_type": "video/mp4", "data": video_data}
            ]
        )
        
        # Safe text extraction
        text_out = getattr(response, "text", None)
        if text_out is None:
            try:
                text_out = response.candidates[0].content.parts[0].text
            except:
                text_out = str(response)
        
        return [Document(
            page_content=f"Video Analysis: {text_out}",
            metadata={"source": file_path, "type": "video"}
        )]
    except Exception as e:
        return [Document(
            page_content=f"Error processing video: {e}",
            metadata={"source": file_path, "type": "video"}
        )]

# RAG PIPELINE (using LangChain) 

def build_vector_store(_client):
    "Scans data folder and builds the RAG index using LangChain."
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Check if data directory exists
    if not os.path.exists(data_dir):
        return None, None, "Folder 'data' not found."

    files = glob.glob(os.path.join(data_dir, "*.*"))
    if not files:
        return None, None, "No files found in 'data'."

    status_msg = ""
    documents = []
    
    # 2. Load files using LangChain document loaders
    for f in files:
        ext = f.lower().split('.')[-1]
        filename = os.path.basename(f)
        try:
            if ext == 'pdf':
                docs = load_pdf_langchain(f)
                documents.extend(docs)
                status_msg += f"✅ Loaded PDF: {filename} ({len(docs)} pages)\n"
            elif ext == 'csv':
                docs = load_csv_langchain(f)
                documents.extend(docs)
                status_msg += f"✅ Loaded CSV: {filename} ({len(docs)} rows)\n"
            elif ext in ['xlsx', 'xls']:
                docs = load_excel_langchain(f)
                documents.extend(docs)
                status_msg += f"✅ Loaded Excel: {filename} ({len(docs)} rows)\n"
            elif ext in ['pptx', 'ppt']:
                docs = load_pptx_langchain(f)
                documents.extend(docs)
                status_msg += f"✅ Loaded PowerPoint: {filename} ({len(docs)} slides)\n"
            elif ext in ['docx', 'doc']:
                docs = load_docx_langchain(f)
                documents.extend(docs)
                status_msg += f"✅ Loaded Word Document: {filename}\n"
            elif ext == 'txt':
                docs = load_text_langchain(f)
                documents.extend(docs)
                status_msg += f"✅ Loaded TXT: {filename}\n"
            elif ext in ['jpg', 'jpeg', 'png']:
                status_msg += f"⏳ Processing Image: {filename}...\n"
                docs = load_image_as_document(f, _client)
                documents.extend(docs)
                status_msg += f"✅ Loaded Image: {filename}\n"
            elif ext in ['mp4', 'avi', 'mov', 'mkv']:
                status_msg += f"⏳ Processing Video: {filename}...\n"
                docs = load_video_as_document(f, _client)
                documents.extend(docs)
                status_msg += f"✅ Loaded Video: {filename}\n"
            else:
                status_msg += f"⚠️ Skipped unsupported file: {filename}\n"
        except Exception as e:
            status_msg += f"❌ Error loading {filename}: {e}\n"

    if not documents:
        return None, None, "No documents loaded successfully."

    # 3. Split documents using RecursiveCharacterTextSplitter with better parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    status_msg += f"📊 Total documents loaded: {len(documents)}\n"
    status_msg += f"✅ Split into {len(chunks)} chunks (size: 500, overlap: 100)\n"

    # 4. Create embeddings and vector store using FAISS and Gemini Embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        
        # Create FAISS vector store from documents
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Optionally save the index
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)
        vector_store.save_local(faiss_index_dir)
        
        status_msg += f"✅ FAISS vector store created ({len(chunks)} chunks)\n"
        return vector_store, chunks, status_msg
    except Exception as e:
        status_msg += f"❌ Error creating vector store: {e}\n"
        return None, None, status_msg

# STREAMLIT UI

st.set_page_config(
    page_title="Document AI Assistant", 
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Corporate Styling
st.markdown("""
<style>
    /* Clean Typography */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Main Content - White Background with Dark Text */
    .main {
        background-color: #FFFFFF !important;
    }
    
    /* Force Dark Text on Light Background */
    .main .stMarkdown, 
    .main .stMarkdown p, 
    .main .stMarkdown div,
    .main h1, 
    .main h2, 
    .main h3, 
    .main h4, 
    .main h5, 
    .main h6 {
        color: #1F2937 !important;
    }
    
    /* Title and Caption */
    .main [data-testid="stCaptionContainer"],
    .main [data-testid="caption"] {
        color: #6B7280 !important;
    }
    
    /* Radio Buttons - Dark Labels on Light Background */
    .main [data-testid="stRadio"] label,
    .main [data-testid="stRadio"] div,
    .main [data-testid="stRadio"] span,
    .main [data-testid="stRadio"] p {
        color: #1F2937 !important;
    }
    
    /* All Main Content Text */
    .main p,
    .main span,
    .main div,
    .main label {
        color: #1F2937 !important;
    }
    
    /* Main Area Metrics - Dark Text on White */
    .main [data-testid="stMetricValue"],
    .main [data-testid="stMetricLabel"],
    .main [data-testid="stMetricDelta"] {
        color: #1F2937 !important;
    }
    
    /* Sidebar - Light Gray Background */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* Sidebar - Force ALL Text to Dark */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #333333 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {
        color: #333333 !important;
    }
    
    /* Sidebar Metrics - Force Dark Text */
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* Sidebar Button Text */
    [data-testid="stSidebar"] button {
        color: #FFFFFF !important;
    }
    
    /* Chat Messages - Force Readable Contrast */
    [data-testid="stChatMessage"] {
        padding: 16px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #f0f2f6 !important;
        border: 1px solid #ddd !important;
    }
    
    /* Force Black Text in All Chat Messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span {
        color: #000000 !important;
    }
    
    /* User Message - Subtle Gray */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #F3F4F6 !important;
        border: 1px solid #ddd !important;
    }
    
    /* Assistant Message - White with Border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #FFFFFF !important;
        border: 1px solid #ddd !important;
    }
    
    /* Chunk Display */
    .chunk-container {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-left: 3px solid #3B82F6;
        padding: 16px;
        margin: 8px 0;
        border-radius: 6px;
        color: #1F2937;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Simple Header
st.title("📄 Document AI Assistant")
st.caption("Intelligent document analysis powered by Google Gemini")
st.divider()

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found. Set it in terminal: $env:GEMINI_API_KEY='...'")
    st.stop()

client = genai.Client(api_key=api_key)

# System Information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("AI Model", "Gemini 2.5 Flash Lite", delta=None)
with col2:
    st.metric("Embeddings", "text-embedding-004", delta=None)
with col3:
    st.metric("Vector DB", "FAISS", delta=None)

st.divider()

# Sidebar: Knowledge Base Status
with st.sidebar:
    st.header("📂 Knowledge Base")
    
    if st.button("🔄 Re-Scan Folder", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    with st.spinner("🔨 Building RAG Index..."):
        vector_store, chunks, log = build_vector_store(client)
    
    if log:
        st.success("✅ Index Built Successfully")
        with st.expander("📋 View Build Log", expanded=False):
            st.code(log, language="text")
        
        # Display statistics
        if chunks:
            st.metric("Total Chunks", len(chunks))
            st.metric("Chunk Size", "500 chars")
            st.metric("Overlap", "100 chars")
    else:
        st.error("❌ Index build failed.")
    
    st.markdown("---")
    st.info("""
    💡 **Tips:**
    - Place files in the `data/` folder
    - Supported: PDF, CSV, Excel, PowerPoint, Word, Text, Images, Videos
    - Click Re-Scan after adding files
    """)

# Mode Selector
mode = st.radio(
    "⚙️ Operation Mode",
    ["📄 RAG Mode (Use Documents)", "🤖 Gemini Only (No Context)"],
    index=0,
    horizontal=True,
    help="RAG Mode uses your documents as context. Gemini Only responds without document context."
)
st.markdown("---")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents, images, or videos..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check mode selection
    use_rag_mode = (mode == "📄 RAG Mode (Use Documents)")
    
    # Make sure the vector store exists if RAG mode is selected
    if use_rag_mode and vector_store is None:
        with st.chat_message("assistant"):
            st.markdown("⚠️ The knowledge base is not ready. Re-scan the folder in the sidebar.")
        st.session_state.messages.append({"role": "assistant", "content": "Knowledge base not ready."})
        st.stop()

    # Process query based on mode
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    google_api_key=api_key,
                    convert_system_message_to_human=True,
                    temperature=0.7
                )
                
                if use_rag_mode:
                    # RAG Mode: Use RetrievalQA with document context
                    custom_prompt = PromptTemplate(
                        template="""You are an intelligent AI assistant answering questions based on provided documents and your knowledge.

❓ USER QUESTION:
{question}

📄 DOCUMENT CONTEXT:
{context}

📋 INSTRUCTIONS:
1. **Read the document context carefully** - Extract specific information, data, and facts
2. **Answer directly and accurately** - Use document information as primary source
3. **Add context when helpful** - Supplement with your knowledge only when it enhances understanding
4. **Be specific** - Reference exact details from the documents
5. **If uncertain** - Clearly state when information is not in the documents

✍️ YOUR ANSWER (be specific and accurate):""",
                        input_variables=["context", "question"]
                    )
                    
                    retrieval_qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 8}
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": custom_prompt}
                    )
                    
                    result = retrieval_qa.invoke({"query": prompt})
                    text_out = result["result"]
                    source_docs = result.get("source_documents", [])
                else:
                    # Gemini Only Mode: Direct query without RAG
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=prompt
                    )
                    
                    # Safe text extraction
                    text_out = getattr(response, "text", None)
                    if text_out is None:
                        try:
                            text_out = response.candidates[0].content.parts[0].text
                        except:
                            text_out = str(response)
                    
                    source_docs = []  # No source documents in Gemini-only mode

            st.markdown(text_out)
            
            # Enhanced chunk display (only in RAG mode)
            if use_rag_mode and source_docs:
                st.markdown("---")
                with st.expander(f"🔍 Retrieved Context ({len(source_docs)} chunks)", expanded=True):
                    for i, doc in enumerate(source_docs, 1):
                        # Create header with type badge
                        doc_type = doc.metadata.get('type', 'unknown') if doc.metadata else 'unknown'
                        st.markdown(f'<div style="background: linear-gradient(135deg, #232F3E 0%, #1A2535 100%); padding: 12px; border-radius: 6px 6px 0 0; margin-top: 10px;"><span style="color: white; font-weight: 600; font-size: 15px;">📄 Chunk {i}</span> <span style="background-color: #D4AF37; color: #1A2332; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 10px;">{doc_type.upper()}</span></div>', unsafe_allow_html=True)
                        
                        # Escape HTML in chunk content to prevent rendering issues
                        escaped_content = html.escape(doc.page_content)
                        st.markdown(f'<div class="chunk-container" style="border-radius: 0 0 6px 6px; margin-top: 0;">{escaped_content}</div>', unsafe_allow_html=True)
                        
                        if doc.metadata:
                            metadata_info = f"📍 **Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                            if 'page' in doc.metadata:
                                metadata_info += f" | 📄 Page: {doc.metadata['page']}"
                            if 'row_number' in doc.metadata:
                                metadata_info += f" | 🔢 Row: {doc.metadata['row_number']}"
                            st.caption(metadata_info)
                        st.markdown("")
            elif use_rag_mode and not source_docs:
                st.info("ℹ️ No relevant context found in documents.")
            elif not use_rag_mode:
                st.info("ℹ️ Gemini-Only Mode: Response without document context.")
                
            st.session_state.messages.append({"role": "assistant", "content": text_out})
        except Exception as e:
            error_msg = f"⚠️ API Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})