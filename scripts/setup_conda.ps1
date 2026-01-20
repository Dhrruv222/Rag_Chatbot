# Conda-based setup (PowerShell)
# Requires Miniconda/Anaconda installed and `conda` available in PATH.
# Usage: run from project root in PowerShell: .\\scripts\\setup_conda.ps1

Set-StrictMode -Version Latest

param(
    [string]$envName = "rag"
)

Write-Host "Creating conda env '$envName' with Python 3.11..."
conda create -n $envName python=3.11 -y

Write-Host "Activating env..."
conda activate $envName

Write-Host "Installing core binary packages from conda-forge (numpy, faiss-cpu, sentence-transformers, pillow)..."
conda install -c conda-forge numpy=1.25.3 faiss-cpu sentence-transformers pillow -y

Write-Host "Installing remaining pip requirements..."
pip install -r requirements.txt

Write-Host "Conda setup complete. Activate with: conda activate $envName" -ForegroundColor Green
