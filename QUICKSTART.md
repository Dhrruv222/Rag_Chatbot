# Quick Start Guide - LangChain RAG System

## 🚀 Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Environment Variables

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
$env:OPENAI_API_KEY = "your-openai-api-key"
```

**Windows Command Prompt:**
```cmd
set GEMINI_API_KEY=your-gemini-api-key
set OPENAI_API_KEY=your-openai-api-key
```

**macOS/Linux:**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

**Using .env file (recommended for development):**
```
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

And load in app.py:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 3: Add Documents
Place your documents in the `task_2/` folder:
- 📄 PDF files (.pdf)
- 📝 Text files (.txt)
- 🖼️ Image files (.jpg, .png, .jpeg)

### Step 4: Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🎯 Usage

### Main Features

1. **Knowledge Base Sidebar**
   - Shows document loading status
   - "Re-Scan Folder" button to rebuild index
   - Real-time progress updates

2. **RAG vs. No-RAG Demo**
   - Compare answers with and without document context
   - Understand the value of RAG
   - See retrieved documents

3. **Chat Interface**
   - Ask questions about your documents
   - View retrieved source documents
   - Maintain conversation history

### Example Queries
```
"What is the main topic of these documents?"
"Summarize the key points"
"Find information about [specific topic]"
"How does [concept] work in the context of these documents?"
```

## 📊 What's Different from Before?

### ✅ Improvements
- **Better Embeddings**: OpenAI embeddings are more semantically accurate
- **Persistent Storage**: Vector store survives app restarts
- **Production-Ready**: Using industry-standard Chroma database
- **Cleaner Code**: LangChain handles orchestration
- **Better Debugging**: Source documents displayed with metadata

### ⚠️ Changes You Need to Know
- **New API Required**: OPENAI_API_KEY needed (in addition to GEMINI_API_KEY)
- **First Run Slow**: Initial embedding takes longer (API calls)
- **Embedding Costs**: Small cost per embedding (~$0.02 per 1M tokens)
- **Old Index Invalid**: FAISS index won't work with new system

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"
```powershell
# Check if set correctly
$env:GEMINI_API_KEY

# If empty, set it
$env:GEMINI_API_KEY = "your-key"
```

### "OPENAI_API_KEY not found"
```powershell
# Set OpenAI API key
$env:OPENAI_API_KEY = "your-key"
```

### "Module not found" errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Slow first run
**This is normal!** The app is:
1. Loading documents with LangChain loaders
2. Creating embeddings via OpenAI API
3. Building Chroma vector store
4. Saving to disk

Subsequent runs will be faster.

### No documents loaded
1. Check `task_2/` folder exists
2. Verify file formats (PDF, TXT, JPG, PNG)
3. Check file permissions (readable)
4. Look for error messages in sidebar status

### "No source documents retrieved"
1. Try a simpler query
2. Check that documents were actually loaded (sidebar)
3. Verify document content is relevant
4. Try rephrasing your question

### Vector store errors
1. Delete `.chroma_db/` folder
2. Restart the app
3. Rebuild index (click "Re-Scan Folder")

## 📈 Performance Tips

### Speed Up Retrieval
```python
# Reduce retrieval count
retriever=vector_store.as_retriever(search_kwargs={"k": 2})
```

### Improve Answer Quality
```python
# Increase retrieval count
retriever=vector_store.as_retriever(search_kwargs={"k": 6})
```

### Reduce Costs
```python
# Use smaller chunk size
chunk_size = 250

# Less frequent rebuilds (don't click Re-Scan often)
```

## 🔑 API Keys

### Getting GEMINI_API_KEY
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Click "Create API Key"
3. Copy the key
4. Set environment variable

### Getting OPENAI_API_KEY
1. Go to [OpenAI Platform](https://platform.openai.com)
2. Sign up or log in
3. Go to API keys section
4. Create new secret key
5. Set environment variable

## 📚 Documentation

For detailed information, see:
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - What changed and why
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - How to upgrade from old system
- [LANGCHAIN_COMPONENTS.md](LANGCHAIN_COMPONENTS.md) - Technical details

## 🚨 Important Notes

1. **API Costs**: 
   - Embedding: ~$0.02 per 1M tokens (OpenAI)
   - Generation: Depends on Gemini usage

2. **Data Privacy**:
   - Documents are processed by OpenAI APIs
   - Check OpenAI privacy policy
   - Vector store stored locally in `.chroma_db/`

3. **Rate Limits**:
   - May hit rate limits with many concurrent requests
   - Add delays between API calls if needed

4. **Disk Space**:
   - Vector store (`.chroma_db/`) grows with document size
   - Each document + embeddings ≈ 2-3x original size

## 🎓 Learning Resources

### LangChain
- [Official Docs](https://python.langchain.com/)
- [GitHub](https://github.com/langchain-ai/langchain)

### Vector Databases
- [Chroma Docs](https://docs.trychroma.com/)
- [Vector DB Comparison](https://github.com/qdrant/vector-db-benchmark)

### RAG (Retrieval-Augmented Generation)
- [RAG Papers](https://arxiv.org/abs/2005.11401)
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997)

## 💡 Next Steps

1. ✅ Get API keys
2. ✅ Install dependencies
3. ✅ Add documents to `task_2/`
4. ✅ Run the app
5. ✅ Test with your documents
6. 📖 Read detailed docs for customization
7. 🚀 Deploy to production

## 🆘 Getting Help

1. Check the documentation files
2. Review error messages (they're usually helpful)
3. Check `.chroma_db/` directory exists and has files
4. Verify API keys are correct
5. Look at Streamlit logs for more details

## 📝 Common Customizations

### Change LLM Temperature
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3  # Lower = more deterministic
)
```

### Use Different Embedding Model
```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Higher quality
    api_key=openai_api_key
)
```

### Adjust Chunk Parameters
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Larger chunks = more context
    chunk_overlap=100,    # Higher overlap = better continuity
    separators=["\n\n", "\n", " ", ""]
)
```

### Retrieval Settings
```python
# In demo section and chat section:
retriever=vector_store.as_retriever(
    search_kwargs={
        "k": 6,              # Number of documents
        "filter": {"type": "pdf"}  # Optional filtering
    }
)
```

## ✨ Tips for Best Results

1. **Quality Documents**: 
   - Use well-structured documents
   - Clear headings and sections
   - Avoid scanned images (use searchable PDFs)

2. **Relevant Queries**:
   - Be specific in your questions
   - Use document terminology
   - Provide context if needed

3. **Monitor Results**:
   - Check retrieved documents
   - Verify relevance
   - Adjust if needed

4. **Cost Management**:
   - Monitor API usage
   - Cache results when possible
   - Avoid repeated identical queries

---

**Version**: 2.0 (LangChain-based)  
**Last Updated**: December 2025  
**Status**: ✅ Production Ready
