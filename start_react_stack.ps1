$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$workspaceDir = Split-Path -Parent $projectDir
$pythonExe = Join-Path $workspaceDir ".venv\Scripts\python.exe"
$frontendDir = Join-Path $projectDir "frontend"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found at: $pythonExe"
    exit 1
}

if (-not (Test-Path $frontendDir)) {
    Write-Error "Frontend directory not found at: $frontendDir"
    exit 1
}

# Stop stale API or Vite processes launched from this workspace.
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*api_server:app*" -or $_.CommandLine -like "*vite*"
}
foreach ($t in $targets) {
    try {
        Stop-Process -Id $t.ProcessId -Force -ErrorAction Stop
    } catch {
        # Ignore race conditions during process cleanup.
    }
}

# Start backend API.
Start-Process -FilePath $pythonExe -ArgumentList "-m uvicorn api_server:app --host 127.0.0.1 --port 8000" -WorkingDirectory $projectDir

# Start React frontend.
Start-Process -FilePath "npm" -ArgumentList "run dev" -WorkingDirectory $frontendDir

Write-Output "React stack started."
Write-Output "Frontend: http://localhost:5173"
Write-Output "Backend:  http://127.0.0.1:8000/health"
