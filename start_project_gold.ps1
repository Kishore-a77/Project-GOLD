$ErrorActionPreference = "Stop"

# Portable local launcher. Set PROJECT_GOLD_PYTHON to an external Python
# executable when a dedicated runtime is preferred; never create a venv here.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = $env:PROJECT_GOLD_PYTHON
if (-not $python) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}

$port = if ($env:PROJECT_GOLD_STREAMLIT_PORT) { $env:PROJECT_GOLD_STREAMLIT_PORT } else { "8501" }
$url = "http://127.0.0.1:$port"
$log = Join-Path $env:TEMP "project-gold-streamlit.log"
$err = Join-Path $env:TEMP "project-gold-streamlit.err"
Remove-Item -LiteralPath $log, $err -Force -ErrorAction SilentlyContinue

$arguments = @(
    "-m", "streamlit", "run", (Join-Path $root "streamlit_app.py"),
    "--server.headless=true", "--server.port=$port"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $root -RedirectStandardOutput $log `
    -RedirectStandardError $err -PassThru

try {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Start-Process $url
                Write-Host "Project GOLD is running at $url"
                exit 0
            }
        } catch {
            # Streamlit is still starting.
        }
        if ($process.HasExited) {
            throw "Streamlit exited before becoming ready. See $log and $err"
        }
    } while ((Get-Date) -lt $deadline)
    throw "Streamlit did not become ready within 30 seconds. See $log and $err"
} catch {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    throw
}
