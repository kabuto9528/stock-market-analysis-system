param(
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [int]$AutoStopAfterSeconds = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"
$logDirectory = Join-Path $projectRoot "artifacts\logs"
$runtimeDirectory = Join-Path $projectRoot "artifacts\runtime"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Virtual environment Python was not found: $pythonPath. See README first."
}
if (-not (Test-Path -LiteralPath $appPath -PathType Leaf)) {
    throw "Streamlit entry point was not found: $appPath."
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null

$selectedPort = $Port
while ($selectedPort -le ($Port + 20)) {
    $listener = Get-NetTCPConnection -LocalPort $selectedPort -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) {
        break
    }
    $selectedPort++
}
if ($selectedPort -gt ($Port + 20)) {
    throw "Ports $Port through $($Port + 20) are unavailable."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $logDirectory "streamlit-launch-$timestamp.stdout.log"
$stderrPath = Join-Path $logDirectory "streamlit-launch-$timestamp.stderr.log"
$serverProcess = $null
$browserProcess = $null
$listenerProcessId = $null

function Stop-LaunchedServer {
    if ($script:listenerProcessId) {
        Stop-Process -Id $script:listenerProcessId -Force -ErrorAction SilentlyContinue
    }
    if ($script:serverProcess -and -not $script:serverProcess.HasExited) {
        try {
            $script:serverProcess.Kill($true)
            $script:serverProcess.WaitForExit(5000)
        }
        catch {
            Stop-Process -Id $script:serverProcess.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    $arguments = @(
        "-m", "streamlit", "run", "app.py",
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", "$selectedPort",
        "--browser.gatherUsageStats", "false"
    )
    $serverProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    $url = "http://127.0.0.1:$selectedPort"
    $deadline = (Get-Date).AddSeconds(45)
    $healthy = $false
    while ((Get-Date) -lt $deadline) {
        if ($serverProcess.HasExited) {
            break
        }
        try {
            $response = Invoke-WebRequest `
                -Uri "$url/_stcore/health" `
                -UseBasicParsing `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 400
        }
    }
    if (-not $healthy) {
        $details = if (Test-Path -LiteralPath $stderrPath) {
            (Get-Content -LiteralPath $stderrPath -Tail 20) -join [Environment]::NewLine
        }
        else {
            "No error log was generated."
        }
        throw "Streamlit startup timed out. Log: $details"
    }

    $listener = Get-NetTCPConnection `
        -LocalAddress "127.0.0.1" `
        -LocalPort $selectedPort `
        -State Listen `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $listenerProcessId = $listener.OwningProcess
    }

    Write-Host "Application started: $url" -ForegroundColor Green
    Write-Host "Close the application browser window to stop the server." -ForegroundColor Cyan
    Write-Host "Runtime log: $stderrPath"

    if ($AutoStopAfterSeconds -gt 0) {
        Start-Sleep -Seconds $AutoStopAfterSeconds
    }
    elseif ($NoBrowser) {
        Read-Host "Press Enter to stop the application"
    }
    else {
        $edgeCandidates = @()
        $edgePath32 = Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"
        $edgePath64 = Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"
        foreach ($candidate in @($edgePath32, $edgePath64)) {
            if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                $edgeCandidates += $candidate
            }
        }

        if ($edgeCandidates) {
            $browserProfile = Join-Path $runtimeDirectory "edge-app-$PID-$timestamp"
            New-Item -ItemType Directory -Path $browserProfile -Force | Out-Null
            $browserProcess = Start-Process `
                -FilePath $edgeCandidates[0] `
                -ArgumentList @(
                    "--app=$url",
                    "--user-data-dir=$browserProfile",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-background-mode"
                ) `
                -PassThru
            Wait-Process -Id $browserProcess.Id
        }
        else {
            Start-Process $url | Out-Null
            Read-Host "Browser tracking is unavailable. Press Enter to stop the application"
        }
    }
}
finally {
    Stop-LaunchedServer
    Write-Host "Application server stopped." -ForegroundColor Yellow
}
