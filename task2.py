# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
import os

# Use a project-local cache for Hugging Face to avoid permission issues
base_dir = os.path.dirname(os.path.abspath(__file__))
hf_cache = os.path.join(base_dir, ".cache", "huggingface")
os.environ.setdefault("HF_HOME", hf_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(base_dir, ".cache"))
os.makedirs(hf_cache, exist_ok=True)

import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define the file name (use the PDF stored in `task_2/`)
filename = os.path.join(base_dir, "task_2", "STUDENT_HANDOUT4Week4-8.pdf")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"❌ Error loading SentenceTransformer model: {e}")
    raise

print("✅ Setup complete. Libraries loaded.")

# ==========================================
# 2. LOAD PDF & EXTRACT TEXT
# ==========================================
raw_text = ""
try:
    with open(filename, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # Loop through every page and extract text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    print(f"✅ PDF Loaded. Total characters: {len(raw_text)}")
except FileNotFoundError:
    print("❌ Error: PDF file not found. Please upload the file.")

# ==========================================
# 3. CREATE TEXT CHUNKS
# ==========================================
# We split text into smaller pieces (chunks) so the search is precise.
chunk_size = 500  # Characters per chunk
overlap = 50      # Overlap to keep context

chunks = []
for i in range(0, len(raw_text), chunk_size - overlap):
    chunks.append(raw_text[i:i + chunk_size])

print(f"✅ Text split into {len(chunks)} chunks.")
print(f"   -> Example Chunk: {chunks[0][:100]}...")

# ==========================================
# 4. GENERATE EMBEDDINGS (The "Vectors")
# ==========================================
# Convert text chunks into lists of numbers (vectors)
print("⏳ Generating embeddings (this may take a few seconds)...")
embeddings = model.encode(chunks)

print(f"✅ Embeddings created. Shape: {embeddings.shape}")

# ==========================================
# 5. STORE IN FAISS (Vector Database)
# ==========================================
# Create the index (the database)
dimension = embeddings.shape[1]  # 384 dimensions for this model
index = faiss.IndexFlatL2(dimension)

# Add our vectors to the database
index.add(np.array(embeddings))
print(f"✅ Database ready. Stored {index.ntotal} vectors.")

# ==========================================
# 6. SEARCH (The Retrieval)
# ==========================================
# This solves the "Failure Example" from Task 1
query = "What is the weight percentage of the Preparedness area in the Final Presentation Rubric?"

# 1. Convert query to vector
query_vector = model.encode([query])

# 2. Search for the 3 closest chunks
k = 3
distances, indices = index.search(np.array(query_vector), k)

print("\n" + "="*40)
print(f"🔍 QUERY: {query}")
print("="*40)

for i in range(k):
    # Get the index of the result
    result_index = indices[0][i]
    # Get the actual text
    result_text = chunks[result_index]
    
    print(f"\n[Result {i+1}]")
    print(f"{result_text.strip()}")
    print("-" * 40)