<#
.SYNOPSIS
    Bootstrap the jaxmech-demo environment on Windows + WSL.
.DESCRIPTION
    Creates or syncs Config/env.cfg, auto-detects Windows Python, optionally
    records a WSL Python interpreter, installs Windows web dependencies from
    requirements-web.txt, and installs WSL analysis dependencies from
    requirements-wsl.txt.
#>

param(
    [string]$WindowsPython,
    [string]$WslPython,
    [switch]$SkipWindowsDeps,
    [switch]$SkipWslDeps
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$ConfigDir = Join-Path $ProjectRoot "Config"
$EnvTemplate = Join-Path $ConfigDir "env.template.cfg"
$EnvCfg = Join-Path $ConfigDir "env.cfg"
$RequirementsWeb = Join-Path $ProjectRoot "requirements-web.txt"
$RequirementsWsl = Join-Path $ProjectRoot "requirements-wsl.txt"
$RunInWsl = Join-Path $ConfigDir "run_in_wsl.ps1"

function Ensure-EnvCfg {
    if (-not (Test-Path $EnvCfg)) {
        if (Test-Path $EnvTemplate) {
            Copy-Item $EnvTemplate $EnvCfg
        }
        else {
            Set-Content -Path $EnvCfg -Encoding UTF8 -Value @(
                "# Machine-local environment configuration",
                "windows_python_exe = ",
                "wsl_python = "
            )
        }
    }
}

function Get-CfgValue {
    param([string]$Key)

    if (-not (Test-Path $EnvCfg)) {
        return ""
    }

    foreach ($line in Get-Content $EnvCfg -Encoding UTF8) {
        $data = ($line -split '#', 2)[0].Trim()
        if (-not $data -or -not $data.Contains('=')) {
            continue
        }
        $pair = $data -split '=', 2
        if ($pair[0].Trim() -eq $Key) {
            return $pair[1].Trim()
        }
    }

    return ""
}

function Set-CfgValue {
    param(
        [string]$Key,
        [string]$Value
    )

    Ensure-EnvCfg
    $lines = @(Get-Content $EnvCfg -Encoding UTF8)
    $newLines = New-Object System.Collections.Generic.List[string]
    $found = $false

    foreach ($line in $lines) {
        $data = ($line -split '#', 2)[0].Trim()
        if ($data.Contains('=') -and (($data -split '=', 2)[0].Trim() -eq $Key)) {
            $newLines.Add("$Key = $Value")
            $found = $true
        }
        else {
            $newLines.Add($line)
        }
    }

    if (-not $found) {
        $newLines.Add("$Key = $Value")
    }

    Set-Content -Path $EnvCfg -Encoding UTF8 -Value $newLines
}

function Resolve-AbsolutePath {
    param([string]$PathValue)

    if (-not $PathValue) {
        return ""
    }

    try {
        return (Resolve-Path $PathValue).ProviderPath
    }
    catch {
        return $PathValue
    }
}

function Probe-PythonExecutable {
    param([string[]]$Command)

    if (-not $Command -or $Command.Count -eq 0) {
        return $null
    }

    try {
        $commandName = $Command[0]
        $commandArgs = @()
        if ($Command.Count -gt 1) {
            $commandArgs = $Command[1..($Command.Count - 1)]
        }
        $output = & $commandName @commandArgs -c "import sys; print(sys.executable)" 2>$null
        $exe = ($output | Select-Object -Last 1)
        if ($exe) {
            $exe = $exe.Trim()
            if ($exe -and (Test-Path $exe)) {
                return (Resolve-Path $exe).ProviderPath
            }
        }
    }
    catch {
    }

    return $null
}

function Resolve-WindowsPython {
    if ($WindowsPython) {
        $resolved = Resolve-AbsolutePath $WindowsPython
        if (-not (Test-Path $resolved)) {
            throw "Specified Windows Python not found: $WindowsPython"
        }
        return $resolved
    }

    $configured = Get-CfgValue "windows_python_exe"
    if ($configured -and (Test-Path $configured)) {
        return (Resolve-Path $configured).ProviderPath
    }

    foreach ($candidate in @(@("py", "-3"), @("python"), @("python3"))) {
        $resolved = Probe-PythonExecutable $candidate
        if ($resolved) {
            return $resolved
        }
    }

    throw "Cannot find a Windows Python interpreter. Install Python or pass -WindowsPython."
}

function Resolve-WslPython {
    if ($WslPython) {
        return $WslPython.Trim()
    }

    $configured = Get-CfgValue "wsl_python"
    if ($configured) {
        return $configured
    }

    try {
        $output = & $RunInWsl "-c" "import sys; print(sys.executable)" 2>$null
        $resolved = ($output | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1)
        if ($resolved) {
            return $resolved.Trim()
        }
    }
    catch {
    }

    return ""
}

Ensure-EnvCfg

$selectedWindowsPython = Resolve-WindowsPython
Set-CfgValue "windows_python_exe" $selectedWindowsPython
Write-Host "[demo setup] Windows Python: $selectedWindowsPython" -ForegroundColor Cyan

if (-not $SkipWindowsDeps) {
    if (-not (Test-Path $RequirementsWeb)) {
        throw "Missing requirements file: $RequirementsWeb"
    }
    Write-Host "[demo setup] Installing Windows dependencies from requirements-web.txt..." -ForegroundColor Cyan
    & $selectedWindowsPython -m pip install -r $RequirementsWeb
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not $SkipWslDeps) {
    if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
        throw "wsl.exe not found. Install WSL or rerun with -SkipWslDeps."
    }
    if (-not (Test-Path $RequirementsWsl)) {
        throw "Missing requirements file: $RequirementsWsl"
    }

    $selectedWslPython = Resolve-WslPython
    if ($selectedWslPython) {
        Set-CfgValue "wsl_python" $selectedWslPython
        Write-Host "[demo setup] WSL Python: $selectedWslPython" -ForegroundColor Cyan
    }
    else {
        Write-Warning "Could not resolve a concrete WSL Python path. Falling back to WSL PATH lookup during install."
    }

    Write-Host "[demo setup] Installing WSL dependencies from requirements-wsl.txt..." -ForegroundColor Cyan
    & $RunInWsl -m pip -- install -r $RequirementsWsl
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "" 
Write-Host "[demo setup] Config/env.cfg is ready." -ForegroundColor Green
Write-Host "[demo setup] Next steps:" -ForegroundColor Green
Write-Host "  1. Review Config/env.cfg if you want to pin a specific WSL environment."
Write-Host "  2. Run .\Config\start_web.bat for the Web UI."
Write-Host "  3. Run .\Config\run_in_wsl.ps1 -m jaxmech.tools.build.build_model_mat -- --config ... for analysis."