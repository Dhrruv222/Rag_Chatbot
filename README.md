# rag_system — Setup

This project contains two demo scripts: `app.py` (Streamlit multimodal RAG UI) and `task2.py` (CLI RAG demo).

Recommended setup on Windows

1) Using a Python virtual environment (`.venv`)

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.25.3 --prefer-binary
python -m pip install -r requirements.txt
```

If you prefer automation, run the helper script:

```pwsh
.\scripts\setup_env.ps1
```

2) If pip still builds NumPy from source or you hit compilation failures (common on Windows), use conda/mamba:

```pwsh
# using conda (PowerShell)
.\scripts\setup_conda.ps1
# or run commands manually
# conda create -n rag python=3.11 -y
# conda activate rag
# conda install -c conda-forge numpy=1.25.3 faiss-cpu sentence-transformers pillow -y
# pip install -r requirements.txt
```

Troubleshooting
- If `numpy` tries to build from source, check your Python bitness (`python -c "import struct; print(struct.calcsize('P')*8)"`) — prefer 64-bit.
- If `faiss-cpu` fails on pip, use conda-forge to install `faiss-cpu`.
- If you want me to run the install steps for you, say so (I will create the venv and run installation commands and report back).
RAG System — Task 2

This small project extracts text from a PDF, creates embeddings with `sentence-transformers`, stores them in a FAISS index, and runs a simple similarity query.

Quick start (PowerShell):

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the script from the project root:

```powershell
python task2.py
```

Notes:
- The script expects the PDF `task_2/STUDENT_HANDOUT4Week4-8.pdf` to be present (it is included in the `task_2/` folder).
- A project-local Hugging Face cache is used (`./.cache/huggingface`) to avoid permission errors writing to the global user cache.
- If you prefer, run the script from `task_2/` or copy the PDF into the project root.

If you'd like, I can also:
- Create a `.gitignore` that excludes virtual environments and the `.cache` folder.
- Commit the changes to git now.
