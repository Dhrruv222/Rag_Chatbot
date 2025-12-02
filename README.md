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
