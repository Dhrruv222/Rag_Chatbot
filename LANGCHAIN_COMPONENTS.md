# LangChain Components Reference

## Document Loaders

### PyPDFLoader
**Purpose:** Load and parse PDF files with page preservation

**Usage:**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/file.pdf")
docs = loader.load()  # Returns List[Document]
```

**Output Document Structure:**
```python
Document(
    page_content="Extracted text from page...",
    metadata={"source": "path/to/file.pdf", "page": 0}
)
```

**Advantages:**
- Handles complex PDF layouts
- Preserves page numbers in metadata
- Robust error handling
- Supports searchable and scanned PDFs

### TextLoader
**Purpose:** Load plain text files

**Usage:**
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("path/to/file.txt", encoding="utf-8")
docs = loader.load()  # Returns List[Document]
```

**Output Document Structure:**
```python
Document(
    page_content="Full text content...",
    metadata={"source": "path/to/file.txt"}
)
```

**Advantages:**
- Simple and efficient
- Full encoding support
- Metadata tracking

### Custom Image Loader (Gemini-based)
**Purpose:** Extract text from images using Gemini's multimodal capabilities

**Implementation:**
```python
def load_image_with_gemini(file_path, client):
    image = Image.open(file_path)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            "Extract all text and describe the content of this image in detail:",
            image
        ]
    )
    text_out = response.text
    
    # Convert to LangChain Document
    return Document(
        page_content=text_out,
        metadata={"source": file_path, "type": "image"}
    )
```

**Advantages:**
- Leverages Gemini's vision capabilities
- Extracts both text and descriptions
- Preserves metadata

## Text Splitters

### RecursiveCharacterTextSplitter
**Purpose:** Intelligently split documents while respecting natural boundaries

**Configuration:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # Max characters per chunk
    chunk_overlap=50,         # Overlap between chunks
    separators=[              # Try these in order
        "\n\n",              # Paragraph boundary (highest priority)
        "\n",                # Line boundary
        " ",                 # Word boundary
        ""                   # Character boundary (fallback)
    ]
)

chunks = splitter.split_documents(documents)
```

**Why This Approach:**
1. **Paragraph separation** (`\n\n`): Respects semantic boundaries first
2. **Line breaks** (`\n`): Maintains sentence context
3. **Spaces** (` `): Keeps words together
4. **Character fallback** (`""`): Ensures exact chunk_size when needed

**Output:**
```python
List[Document] where each Document has:
- page_content: Chunk text
- metadata: Original document metadata + splitter info
```

**Parameters Explained:**
- **chunk_size=500**: Balance between context (larger) and precision (smaller)
  - Larger (800-1000): More context, fewer chunks, may lose detail
  - Smaller (200-300): More precise, more chunks, retrieval overhead
- **chunk_overlap=50**: Prevents context loss at chunk boundaries
  - Higher overlap (100-200): Better continuity, more redundancy
  - Lower/none: More efficient, potential context loss

## Vector Stores

### Chroma
**Purpose:** Modern, persistent vector database with semantic search

**Initialization:**
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=".chroma_db"  # Optional but recommended
)
```

**Key Features:**
- **Persistent Storage**: `.chroma_db/` directory survives app restarts
- **Semantic Search**: Finds semantically similar documents
- **Metadata Filtering**: Can filter by document metadata
- **Similarity Scoring**: Returns results with relevance scores

**Common Operations:**
```python
# 1. Similarity search
results = vector_store.similarity_search("query text", k=5)

# 2. As LangChain retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4, "filter": {"type": "pdf"}}
)

# 3. Delete documents
vector_store.delete(ids=["doc_id_1", "doc_id_2"])

# 4. Add documents
vector_store.add_documents(new_documents)
```

**Advantages:**
- Production-ready reliability
- Built-in persistence
- Excellent documentation
- Active development
- SQLite backend (no external DB needed)

### Why Chroma over FAISS?
| Feature | FAISS | Chroma |
|---------|-------|--------|
| Persistence | Manual | Built-in |
| Metadata handling | Limited | Full support |
| Integration | Lower-level | LangChain-native |
| Ease of use | Complex | Simple |
| Search filtering | No | Yes |
| Production ready | Requires wrapper | Yes |

## Embeddings

### OpenAIEmbeddings
**Purpose:** Generate high-quality semantic embeddings using OpenAI API

**Usage:**
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Default, cheaper
    api_key=openai_api_key
)

# Embed a document
doc_embedding = embeddings.embed_documents(["text to embed"])

# Embed a query
query_embedding = embeddings.embed_query("query text")
```

**Models Available:**
- `text-embedding-3-small`: Faster, cheaper ($0.02/1M tokens)
- `text-embedding-3-large`: Higher quality ($0.13/1M tokens)

**Advantages over SentenceTransformer:**
- Superior semantic understanding
- Better cross-lingual support
- Optimized for semantic search
- Regular updates from OpenAI

**Why the Switch?**
- **Quality**: OpenAI embeddings are more semantically accurate
- **Consistency**: Aligned with modern LLMs
- **Scalability**: Handles enterprise use cases
- **Integration**: Native LangChain support

## Language Models

### ChatGoogleGenerativeAI
**Purpose:** Interface to Google's Gemini models for generation

**Configuration:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
    top_p=0.9
)
```

**Available Models:**
- `gemini-2.5-flash`: Fast, efficient (used here)
- `gemini-2.0-pro`: More powerful but slower
- `gemini-2.0-flash`: Latest optimized version

**Usage:**
```python
# Single response
response = llm.invoke("What is RAG?")

# With structured input
response = llm.invoke([
    ("human", "Question from user"),
    ("ai", "Previous response"),
    ("human", "Follow-up question")
])
```

**Advantages:**
- Multimodal (text + images)
- Fast and efficient
- Good for RAG tasks
- Well-integrated with LangChain

## Chains

### RetrievalQA
**Purpose:** Orchestrate retrieval + generation with unified error handling

**Initialization:**
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",              # Concatenate docs
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 4}       # Top 4 documents
    ),
    return_source_documents=True     # Include sources in output
)
```

**Chain Types:**
| Type | Approach | Best For |
|------|----------|----------|
| `stuff` | Concatenate all docs into prompt | Small context, fast |
| `map_reduce` | Process each doc separately, reduce | Large context, expensive |
| `refine` | Iteratively refine with each doc | Sequential processing |
| `multi_query` | Generate multiple queries | Complex questions |

**Invocation:**
```python
result = qa_chain.invoke({"query": "user question"})

# Result structure
result = {
    "query": "user question",
    "result": "Generated answer from LLM",
    "source_documents": [Document, Document, ...]
}
```

**Advantages:**
- Unified retrieval + generation
- Automatic source tracking
- Error handling built-in
- Configurable chain type
- Works with any LLM/Retriever combination

## Data Flow Diagram

```
┌─────────────────┐
│  Task_2 Files   │
│ (PDF/TXT/Image) │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ Document Loaders     │
│ - PyPDFLoader        │
│ - TextLoader         │
│ - Custom (Gemini)    │
└────────┬─────────────┘
         │
         ▼ List[Document]
┌──────────────────────┐
│RecursiveCharacter    │
│TextSplitter          │
└────────┬─────────────┘
         │
         ▼ List[Document]
┌──────────────────────┐
│ OpenAIEmbeddings     │
│ (Generate vectors)   │
└────────┬─────────────┘
         │
         ▼ Vector data
┌──────────────────────┐
│ Chroma VectorStore   │
│ (Persist in disk)    │
└────────┬─────────────┘
         │
    ┌────┴────┐
    │ Query   │
    └────┬────┘
         │
         ▼
┌──────────────────────┐
│RetrievalQA Chain     │
│ - Retriever (k=4)    │
│ - ChatGoogleGenAI    │
└────────┬─────────────┘
         │
         ▼
  {"result": "...",
   "source_documents": [...]}
```

## Configuration Best Practices

### Chunk Size
```python
# Short documents, high precision needed
chunk_size = 250

# Medium documents, balanced
chunk_size = 500  # ← Current setting

# Long documents, more context
chunk_size = 1000
```

### Overlap
```python
# Minimal redundancy
chunk_overlap = 0

# Standard (balanced)
chunk_overlap = 50  # ← Current setting

# High continuity
chunk_overlap = 100
```

### Retrieval Count (k)
```python
# Fast, focused
k = 2

# Balanced
k = 4  # ← Chat interface

# Comprehensive
k = 5  # ← Demo interface
k = 7  # More thorough
```

### Temperature
```python
# Deterministic, focused answers
temperature = 0.0

# Balanced (default Gemini)
temperature = 0.7

# Creative, diverse answers
temperature = 1.0
```

## Error Handling

Each component has built-in error handling:

```python
try:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
except Exception as e:
    logging.error(f"Failed to load PDF: {e}")
    # Gracefully skip to next file

try:
    vector_store = Chroma.from_documents(...)
except Exception as e:
    # Vector store creation failed
    return None, None, f"Error: {e}"
```

## Monitoring & Debugging

### Check loaded documents
```python
print(f"Loaded {len(documents)} documents")
for doc in documents[:1]:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:200]}")
```

### Verify chunks
```python
print(f"Created {len(chunks)} chunks")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f}")
```

### Test retriever
```python
test_query = "What is the main topic?"
results = vector_store.similarity_search(test_query, k=3)
print(f"Retrieved {len(results)} documents")
for doc in results:
    print(f"Score: {doc.metadata}")
```

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Community Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- [Chroma Vector Store](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/)
- [Gemini API](https://ai.google.dev/)
