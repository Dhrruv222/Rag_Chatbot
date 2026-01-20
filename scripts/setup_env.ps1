# Setup script for pip-based environment (PowerShell)
# Usage: run from project root in PowerShell after activating your venv
# To run: .\\scripts\\setup_env.ps1

Set-StrictMode -Version Latest

if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

Write-Host "Activate the venv with: .\\.venv\\Scripts\\Activate.ps1"
Write-Host "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing a binary NumPy wheel (preferred, latest compatible)..."
python -m pip install numpy --upgrade --prefer-binary

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install a binary NumPy wheel. Try using the conda script: .\\scripts\\setup_conda.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Installing project requirements from requirements.txt..."
python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing requirements failed. Consider using conda (run scripts/setup_conda.ps1) or run the diagnostic commands in README.md." -ForegroundColor Yellow
    exit 1
}

Write-Host "Done. If you plan to use FAISS on Windows and see issues, prefer conda for faiss-cpu installation." -ForegroundColor Green
