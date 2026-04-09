$ErrorActionPreference = "Stop"

# This script starts the Streamlit app with the correct venv and a single stable port.
$projectDir = $PSScriptRoot
$workspaceDir = Split-Path -Parent $projectDir
$pythonExe = Join-Path $workspaceDir ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found at: $pythonExe"
    exit 1
}

# Stop duplicate Streamlit app.py processes to avoid blank/unstable page behavior.
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*streamlit*app.py*"
}

foreach ($t in $targets) {
    try {
        Stop-Process -Id $t.ProcessId -Force -ErrorAction Stop
    } catch {
        # Ignore race conditions if process exits during cleanup.
    }
}

Set-Location $projectDir
& $pythonExe -m streamlit run app.py --server.port 8501 --server.headless true
