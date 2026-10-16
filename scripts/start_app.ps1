param(
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [int]$AutoStopAfterSeconds = 0
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"
$logDirectory = Join-Path $projectRoot "artifacts\logs"
$runtimeDirectory = Join-Path $projectRoot "artifacts\runtime"
$launchLog = Join-Path $logDirectory "launcher-last.txt"

function Write-LaunchLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    [System.IO.File]::AppendAllText($launchLog, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    Write-Host $Message
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
Set-Content -LiteralPath $launchLog -Value "" -Encoding UTF8

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    Write-LaunchLog "Virtual environment Python was not found: $pythonPath"
    throw "Virtual environment Python was not found: $pythonPath. See README first."
}
if (-not (Test-Path -LiteralPath $appPath -PathType Leaf)) {
    Write-LaunchLog "Streamlit entry point was not found: $appPath"
    throw "Streamlit entry point was not found: $appPath."
}

function Test-LocalPortOpen {
    param([int]$CandidatePort)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $wait = $client.BeginConnect("127.0.0.1", $CandidatePort, $null, $null)
        if (-not $wait.AsyncWaitHandle.WaitOne(400, $false)) {
            return $false
        }
        $client.EndConnect($wait) | Out-Null
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Get-ListeningProcessId {
    param([int]$CandidatePort)
    $lines = & netstat.exe -ano -p tcp
    foreach ($line in $lines) {
        $parts = @($line -split '\s+' | Where-Object { $_ })
        if ($parts.Count -lt 5 -or $parts[0] -ne "TCP" -or $parts[3] -ne "LISTENING") {
            continue
        }
        $localPortText = $parts[1].Substring($parts[1].LastIndexOf(":") + 1)
        if ($localPortText -eq "$CandidatePort" -and $parts[4] -match '^\d+$') {
            return [int]$parts[4]
        }
    }
    return $null
}

function Test-StreamlitHealth {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri ($Url + "/_stcore/health") -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Get-EdgeExecutable {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"),
        (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\Edge\Application\msedge.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    return $null
}

function Stop-ProcessTree {
    param([int]$ProcessId)
    if ($ProcessId -le 0) {
        return
    }
    try {
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = "taskkill.exe"
        $startInfo.Arguments = "/PID $ProcessId /T /F"
        $startInfo.CreateNoWindow = $true
        $startInfo.UseShellExecute = $false
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $killer = [System.Diagnostics.Process]::Start($startInfo)
        $null = $killer.StandardOutput.ReadToEnd()
        $null = $killer.StandardError.ReadToEnd()
        $null = $killer.WaitForExit(5000)
    }
    catch {
        Write-LaunchLog ("Ignoring process cleanup failure for PID {0}: {1}" -f $ProcessId, $_.Exception.Message)
    }
}

function Show-ApplicationWindow {
    param([IntPtr]$Handle)
    if ($Handle -eq [IntPtr]::Zero) {
        return
    }
    try {
        if (-not ("AppWindowFocus" -as [type])) {
            Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class AppWindowFocus {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@ | Out-Null
        }
        [AppWindowFocus]::ShowWindow($Handle, 9) | Out-Null
        [AppWindowFocus]::SetForegroundWindow($Handle) | Out-Null
    }
    catch {
        Write-LaunchLog ("Could not focus the application window: {0}" -f $_.Exception.Message)
    }
}

function Open-ApplicationWindow {
    param(
        [string]$Url,
        [string]$RuntimeDirectory,
        [string]$Timestamp
    )
    $edgePath = Get-EdgeExecutable
    if (-not $edgePath) {
        Write-LaunchLog "Edge was not found. Opening the default browser."
        Start-Process $Url | Out-Null
        Read-Host "Open $Url in a browser, then press Enter to stop the service"
        return
    }

    $browserProfile = Join-Path $RuntimeDirectory "edge-app-$PID-$Timestamp"
    New-Item -ItemType Directory -Path $browserProfile -Force | Out-Null
    Write-LaunchLog "Opening Edge window: $Url"
    $browserProcess = Start-Process `
        -FilePath $edgePath `
        -ArgumentList @(
            "--app=$Url",
            "--user-data-dir=$browserProfile",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-mode",
            "--disable-sync",
            "--window-position=80,40",
            "--window-size=1280,860"
        ) `
        -PassThru
    if (-not $browserProcess) {
        throw "Edge did not start: $edgePath"
    }

    $deadline = (Get-Date).AddSeconds(12)
    do {
        Start-Sleep -Milliseconds 200
        $browserProcess.Refresh()
        if ($browserProcess.HasExited) {
            break
        }
        if ($browserProcess.MainWindowHandle -ne 0) {
            break
        }
    } while ((Get-Date) -lt $deadline)

    if ($browserProcess.HasExited -or $browserProcess.MainWindowHandle -eq 0) {
        Write-LaunchLog "Edge application window did not stay open. Falling back to the default browser."
        Start-Process $Url | Out-Null
        Read-Host "The app URL is $Url. Press Enter here to stop the service"
        return
    }

    Show-ApplicationWindow -Handle $browserProcess.MainWindowHandle
    Write-LaunchLog ("Application window is open. PID {0}. Close that window to stop the service." -f $browserProcess.Id)
    Wait-Process -Id $browserProcess.Id
}

$selectedPort = $null
$reuseExisting = $false
foreach ($candidatePort in $Port..($Port + 20)) {
    if (-not (Test-LocalPortOpen -CandidatePort $candidatePort)) {
        $selectedPort = $candidatePort
        break
    }
    $candidateUrl = "http://127.0.0.1:$candidatePort"
    if (Test-StreamlitHealth -Url $candidateUrl) {
        $selectedPort = $candidatePort
        $reuseExisting = $true
        break
    }
}
if (-not $selectedPort) {
    Write-LaunchLog "No free port between $Port and $($Port + 20)."
    throw "Ports $Port through $($Port + 20) are unavailable."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $logDirectory "streamlit-launch-$timestamp.stdout.log"
$stderrPath = Join-Path $logDirectory "streamlit-launch-$timestamp.stderr.log"
$serverProcess = $null
$listenerProcessId = $null
$startedServer = $false

function Stop-LaunchedServer {
    if (-not $script:startedServer) {
        return
    }
    $ids = @()
    if ($script:listenerProcessId) {
        $ids += [int]$script:listenerProcessId
    }
    if ($script:serverProcess -and -not $script:serverProcess.HasExited) {
        $ids += [int]$script:serverProcess.Id
    }
    foreach ($processId in ($ids | Select-Object -Unique)) {
        Stop-ProcessTree -ProcessId $processId
    }
}

try {
    $url = "http://127.0.0.1:$selectedPort"
    Write-LaunchLog "Launcher started. Selected port $selectedPort. Reuse=$reuseExisting"
    if ($reuseExisting) {
        $listenerProcessId = Get-ListeningProcessId -CandidatePort $selectedPort
        Write-LaunchLog "Existing application is already running: $url"
    }
    else {
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
        $startedServer = $true
        Write-LaunchLog "Started Streamlit process $($serverProcess.Id)"

        $deadline = (Get-Date).AddSeconds(45)
        $healthy = $false
        while ((Get-Date) -lt $deadline) {
            if ($serverProcess.HasExited) {
                break
            }
            if (Test-StreamlitHealth -Url $url) {
                $healthy = $true
                break
            }
            Start-Sleep -Milliseconds 400
        }
        if (-not $healthy) {
            $details = if (Test-Path -LiteralPath $stderrPath) {
                (Get-Content -LiteralPath $stderrPath -Tail 20) -join [Environment]::NewLine
            }
            else {
                "No error log was generated."
            }
            Write-LaunchLog "Streamlit startup timed out. $details"
            throw "Streamlit startup timed out. Log: $details"
        }
        $listenerProcessId = Get-ListeningProcessId -CandidatePort $selectedPort
        if (-not $listenerProcessId) {
            $listenerProcessId = $serverProcess.Id
        }
        Write-LaunchLog "Application is healthy: $url"
    }

    if ($AutoStopAfterSeconds -gt 0) {
        Start-Sleep -Seconds $AutoStopAfterSeconds
    }
    elseif ($NoBrowser) {
        Read-Host "Press Enter to stop the application"
    }
    else {
        Open-ApplicationWindow -Url $url -RuntimeDirectory $runtimeDirectory -Timestamp $timestamp
    }
    Write-LaunchLog "Launcher finished normally."
}
catch {
    Write-LaunchLog ("Launcher failed: {0}" -f $_.Exception.Message)
    throw
}
finally {
    Stop-LaunchedServer
    if ($startedServer) {
        Write-LaunchLog "Application server stopped."
    }
}
