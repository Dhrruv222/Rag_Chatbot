# Run Streamlit with proper configuration
# This script bypasses permission issues with ~/.streamlit

# Set environment variables to use project-local config
$env:STREAMLIT_CONFIG_DIR = (Resolve-Path ".\.streamlit").Path
$env:STREAMLIT_CLIENT_SHOWERRORDETAILS = "true"
$env:STREAMLIT_LOGGER_LEVEL = "error"
$env:STREAMLIT_BROWSER_GATHERSTATS = "false"
$env:STREAMLIT_CLIENT_TOOLBARMODE = "minimal"

# Also disable the global Streamlit telemetry that causes the permission error
# by setting the credentials file in the project directory
$env:STREAMLIT_SERVER_HEADLESS = "true"

Write-Host "Starting Streamlit RAG App..." -ForegroundColor Green
Write-Host "App will be available at: http://localhost:8501" -ForegroundColor Cyan

# Activate venv and run
.\.venv\Scripts\Activate.ps1
streamlit run .\app.py
