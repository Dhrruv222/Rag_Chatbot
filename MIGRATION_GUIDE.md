# Migration Guide: Old to New RAG System

## Quick Start

### 1. Install New Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Both API Keys
```powershell
# PowerShell (Windows)
$env:GEMINI_API_KEY = "your-gemini-key-here"
$env:OPENAI_API_KEY = "your-openai-key-here"

# Or in a .env file
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=your-openai-key-here
```

### 3. Run the App
```bash
streamlit run app.py
```

## What Changed in the Code

### Import Statements
Old:
```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
```

New:
```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
```

### File Loading Changes

**Old Pattern:**
```python
# Custom function for each format
def load_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
```

**New Pattern:**
```python
# LangChain document loaders
if ext == 'pdf':
    loader = PyPDFLoader(f)
    docs = loader.load()  # Returns List[Document]
elif ext == 'txt':
    loader = TextLoader(f, encoding='utf-8')
    docs = loader.load()  # Returns List[Document]
```

### Chunking Changes

**Old Pattern:**
```python
chunk_size = 500
step = max(1, chunk_size - 50)
chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), step)]
# Result: List[str]
```

**New Pattern:**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
# Result: List[Document] with metadata preserved
```

### Vector Store Changes

**Old Pattern:**
```python
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)  # numpy array
embeddings = np.array(embeddings, dtype=np.float32)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Search
query_vector = model.encode([query_text])
query_vector = np.array(query_vector, dtype=np.float32)
_, indices = index.search(query_vector, k=5)
```

**New Pattern:**
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=chroma_db_dir
)

# Search
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke({"query": query_text})
```

### RAG Chain Changes

**Old Pattern:**
```python
# Manual retrieval and prompt construction
query_vector = model.encode([prompt])
query_vector = np.array(query_vector, dtype=np.float32)
_, indices = index.search(query_vector, k=4)
valid_idxs = [int(i) for i in indices[0] if i >= 0 and i < len(chunks)]
retrieved_context = "\n\n".join([chunks[i] for i in valid_idxs])

rag_prompt = f"""
You are a helpful assistant. Answer based on the Context below.
If the answer is not in the context, say "I don't know".

Context:
{retrieved_context}

Question:
{prompt}
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=rag_prompt
)
```

**New Pattern:**
```python
# Unified RetrievalQA chain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

result = retrieval_qa.invoke({"query": prompt})
text_out = result["result"]
source_docs = result.get("source_documents", [])
```

### Return Values Update

**build_vector_store() function:**

Old:
```python
return index, chunks, model, status_msg
# Where:
# - index: faiss.IndexFlatL2
# - chunks: List[str]
# - model: SentenceTransformer
# - status_msg: str
```

New:
```python
return vector_store, chunks, status_msg
# Where:
# - vector_store: Chroma (can be None)
# - chunks: List[Document]
# - status_msg: str
```

## Configuration Changes

### Environment Variables

**Removed:**
- Cache folder setup for HuggingFace (no longer needed)

**Added:**
- `OPENAI_API_KEY` - Required for embeddings

**Still Required:**
- `GEMINI_API_KEY` - For generation

### Configuration Code
```python
# Old
os.environ["HF_HOME"] = project_cache
os.environ["TRANSFORMERS_CACHE"] = project_cache
os.environ["HF_HUB_CACHE"] = project_cache

# New (still applicable but simplified)
chroma_db_dir = os.path.join(base_dir, ".chroma_db")
# Chroma handles its own persistence
```

## File Structure Changes

### New Files Created
- `.chroma_db/` - Persistent vector store directory (auto-created)

### Files Modified
- `app.py` - Complete refactoring
- `requirements.txt` - Dependency updates

### Files Removed (not needed)
- `.cache/` usage for transformers (optional, can clean up)

## Backward Compatibility

⚠️ **Not backward compatible** - The old FAISS index will not work with the new Chroma store. The vector store will be rebuilt on first run.

## Cost Implications

### Embedding Costs
- **Old:** Free (local SentenceTransformer)
- **New:** ~$0.02 per 1M tokens (OpenAI API)

This is a tradeoff for:
- Higher quality embeddings
- Better semantic understanding
- Production-grade reliability
- Persistent storage

### Generation Costs
- **Unchanged:** Still uses Gemini API with same cost model

## Verification Checklist

- [ ] Both API keys are set and valid
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Old FAISS index is not expected to load
- [ ] First run will create `.chroma_db/` directory
- [ ] Vector store persists between app restarts
- [ ] Retrieved documents show in context expander
- [ ] Source metadata is displayed correctly
- [ ] RAG vs. No-RAG demo works in both modes

## Troubleshooting

### Error: "Import 'langchain_*' could not be resolved"
- Solution: Install dependencies with `pip install -r requirements.txt`

### Error: "OPENAI_API_KEY not found"
- Solution: Set the OpenAI API key in terminal or .env file

### Slow first run
- Expected: First embedding with OpenAI may take time
- Subsequent runs will use cached embeddings

### "No source documents retrieved"
- Check that documents were loaded (check sidebar status)
- Verify task_2 folder contains files
- Try a different query

### Vector store not persisting
- Check that `.chroma_db/` directory has write permissions
- Ensure the directory path is correct
- Try clearing cache and rebuilding

## Performance Tips

1. **Optimize chunk size** if retrieval is off-topic:
   - Increase for longer context (e.g., 1000)
   - Decrease for more precise matches (e.g., 250)

2. **Adjust retrieval count:**
   - k=4 for chat (faster)
   - k=5-7 for more comprehensive context

3. **Monitor embedding costs:**
   - Cache vector store to reduce rebuilds
   - Use persistent storage (already enabled)

4. **Improve retrieval quality:**
   - Add more diverse documents
   - Use clearer, more structured documents
   - Adjust separators in RecursiveCharacterTextSplitter

## Next Steps

1. Deploy with new LangChain stack
2. Monitor OpenAI API usage
3. Gather user feedback on retrieval quality
4. Consider alternative LLMs (Claude, Ollama)
5. Implement conversation history if needed
