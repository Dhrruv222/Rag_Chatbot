# scripts/setup.ps1
# Usage: ./scripts/setup.ps1 [-Recreate]
param([switch]$Recreate)

# Create or recreate virtual environment
if (Test-Path .venv -PathType Container) {
    if ($Recreate) {
        Write-Host "Removing existing .venv..."
        Remove-Item -Recurse -Force .venv
    }
}

if (-not (Test-Path .venv -PathType Container)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

# Activate venv for the current script/session
Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Write-Host 'Setup complete. To run the project: python task2.py'