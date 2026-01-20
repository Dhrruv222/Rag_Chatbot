import os
import glob
import streamlit as st
from google import genai
from PIL import Image, ExifTags

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredImageLoader


api_key = os.environ.get("GEMINI_API_KEY")

# Setup Cache (keep downloads inside project to avoid permission issues)
base_dir = os.path.dirname(os.path.abspath(__file__))
project_cache = os.path.join(base_dir, ".cache")
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

# 3. Image Loader
def load_image_as_document(file_path, client):
    
    try:
        image = Image.open(file_path)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
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


if __name__ == "__main__":

    pass

# RAG PIPELINE (using LangChain) 

def build_vector_store(_client):
    "Scans task_2 folder and builds the RAG index using LangChain."
    data_dir = os.path.join(base_dir, "task_2")
    
    # 1. Check if data directory exists
    if not os.path.exists(data_dir):
        return None, None, "Folder 'task_2' not found."

    files = glob.glob(os.path.join(data_dir, "*.*"))
    if not files:
        return None, None, "No files found in 'task_2'."

    status_msg = ""
    documents = []
    
    # 2. Load files using LangChain document loaders
    for f in files:
        ext = f.lower().split('.')[-1]
        try:
            if ext == 'pdf':
                docs = load_pdf_langchain(f)
                documents.extend(docs)
                status_msg += f" Loaded PDF: {os.path.basename(f)}\n"
            elif ext == 'txt':
                docs = load_text_langchain(f)
                documents.extend(docs)
                status_msg += f" Loaded TXT: {os.path.basename(f)}\n"
            elif ext in ['jpg', 'jpeg', 'png']:
                status_msg += f" Processing Image: {os.path.basename(f)}...\n"
                docs = load_image_as_document(f, _client)
                documents.extend(docs)
                status_msg += f" Loaded Image: {os.path.basename(f)}\n"
        except Exception as e:
            status_msg += f" Error loading {os.path.basename(f)}: {e}\n"

    if not documents:
        return None, None, "No documents loaded successfully."

    # 3. Split documents using LangChain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    status_msg += f"✅ Split into {len(chunks)} chunks\n"

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

st.set_page_config(page_title="Multimodal RAG System", page_icon="🧠")
st.title("🧠 Multimodal RAG Chatbot")
st.caption("Processes: PDF 📄 | Text 📝 | Images 🖼️")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found. Set it in terminal: $env:GEMINI_API_KEY='...'")
    st.stop()

client = genai.Client(api_key=api_key)

# Show notice
st.info("""
**Using Gemini (Google) API for Generation & LangChain with Gemini Embeddings + FAISS for RAG:**
- Make sure your GEMINI_API_KEY is set
- FAISS provides fast similarity search for document retrieval
- If you get errors, check your API key and access permissions
""")

# Sidebar: Knowledge Base Status
with st.sidebar:
    st.header("📂 Knowledge Base")
    if st.button("🔄 Re-Scan Folder"):
        st.cache_resource.clear()
    
    with st.spinner("Building RAG Index..."):
        vector_store, chunks, log = build_vector_store(client)
    
    if log:
        st.text(log)
    else:
        st.error("Index build failed.")

# Demo Section: RAG vs. No-RAG Comparison
with st.expander("📊 : RAG vs. No-RAG Comparison", expanded=False):
    st.markdown("### Why RAG Matters")
    st.markdown("""
    **Without RAG:** The LLM relies only on its training data and may not know about your specific documents.
    
    **With RAG:** The LLM has access to your actual documents and can provide accurate, document-specific answers.
    """)
    
    demo_question = st.text_input("📝 Ask a question (e.g., 'What is the main topic in the documents?'):")
    
    if demo_question and vector_store is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❌ Without RAG (Knowledge Base Ignored)")
            try:
                no_rag_prompt = f"Answer this question: {demo_question}"
                response_no_rag = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=no_rag_prompt
                )
                text_no_rag = getattr(response_no_rag, "text", None)
                if text_no_rag is None:
                    try:
                        text_no_rag = response_no_rag.candidates[0].content[0].text
                    except Exception:
                        text_no_rag = str(response_no_rag)
                st.write(text_no_rag)
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col2:
            st.subheader("✅ With RAG (Knowledge Base Used)")
            try:
                # Use LangChain's RetrievalQA chain with Gemini
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    convert_system_message_to_human=True
                )
                
                retrieval_qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True
                )
                
                result = retrieval_qa.invoke({"query": demo_question})
                text_with_rag = result["result"]
                st.write(text_with_rag)
                
                # Show retrieved sources for verification
                if result.get("source_documents"):
                    with st.expander("📄 Retrieved Context"):
                        for i, doc in enumerate(result["source_documents"][:3], 1):
                            st.caption(f"Chunk {i}: {doc.page_content[:200]}...")
            except Exception as e:
                st.error(f"Error: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your PDF, Text, or Image..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Make sure the vector store exists
    if vector_store is None:
        with st.chat_message("assistant"):
            st.markdown("⚠️ The knowledge base is not ready. Re-scan the folder in the sidebar.")
        st.session_state.messages.append({"role": "assistant", "content": "Knowledge base not ready."})
        st.stop()

    # RAG Logic using LangChain RetrievalQA
    with st.chat_message("assistant"):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
            
            retrieval_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}
                ),
                return_source_documents=True
            )
            
            result = retrieval_qa.invoke({"query": prompt})
            text_out = result["result"]
            source_docs = result.get("source_documents", [])

            st.markdown(text_out)
            
            with st.expander("🔍 View Retrieved Context"):
                if source_docs:
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Document {i}:**")
                        st.code(doc.page_content)
                        if doc.metadata:
                            st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                else:
                    st.write("No source documents retrieved.")
                
            st.session_state.messages.append({"role": "assistant", "content": text_out})
        except Exception as e:
            error_msg = f"⚠️ API Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})