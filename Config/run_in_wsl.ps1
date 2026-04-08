<#
.SYNOPSIS
    Run Python scripts via WSL using the repository configuration.
.DESCRIPTION
    Loads machine-local settings from Config/env.cfg via Config/wsl_env.sh.
    WSL PYTHONPATH prefers repositories under external/ before falling back to
    the WSL environment site-packages.
#>

param(
    [switch]$m,

    [Parameter(Position=0)]
    [string]$ScriptPath,

    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$PROJECT_ROOT_WIN = Split-Path $PSScriptRoot -Parent
$WSL_ENV_SCRIPT_WIN = Join-Path $PSScriptRoot "wsl_env.sh"

function Convert-ToWslPath {
    param([string]$WinPath)
    if (-not [System.IO.Path]::IsPathRooted($WinPath)) {
        $WinPath = Join-Path (Get-Location) $WinPath
    }
    # Normalize forward-slash Windows paths like E:/foo/bar
    if ($WinPath -match '^([A-Za-z]):/(.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    $WinPath = [System.IO.Path]::GetFullPath($WinPath)
    if ($WinPath -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    Write-Error "Cannot convert path: $WinPath"
    exit 1
}

$PROJECT_ROOT_WSL = Convert-ToWslPath $PROJECT_ROOT_WIN
$WSL_ENV_SCRIPT = Convert-ToWslPath $WSL_ENV_SCRIPT_WIN
$PROJECT_PYTHONPATH = "$PROJECT_ROOT_WSL/external`:$PROJECT_ROOT_WSL"

if ($env:WSL_PYTHON) {
    $WSL_PYTHON = $env:WSL_PYTHON
} else {
    $detected = (wsl -e bash -c "source $WSL_ENV_SCRIPT 2>/dev/null && echo `$WSL_PYTHON")
    if ($detected -and $detected.Trim() -match "^/") {
        $WSL_PYTHON = $detected.Trim()
    } else {
        $WSL_PYTHON = "python3"
    }
}

# Detect the WSL user's real home directory.
# wsl -e inherits the Windows HOME which breaks MPI/PETSc, so we query
# the actual Linux home and inject it explicitly.
$WSL_HOME = (wsl -e bash -c 'getent passwd $(id -un) 2>/dev/null | cut -d: -f6')
if (-not $WSL_HOME -or $WSL_HOME.Trim() -eq "") { $WSL_HOME = "/root" }
$WSL_HOME = $WSL_HOME.Trim()

$env_prefix = "HOME=$WSL_HOME PYTHONPATH=$PROJECT_PYTHONPATH JAX_ENABLE_X64=1"

if ($ScriptPath -eq "-c") {
    $code = $ExtraArgs -join " "
    Write-Host "[MockWSL] Inline code..." -ForegroundColor Cyan
    wsl -e bash -c "$env_prefix $WSL_PYTHON -c '$code'"
}
elseif ($m) {
    $module = $ScriptPath
    $wslArgs = @()
    foreach ($arg in $ExtraArgs) {
        if ($arg -match '^[A-Za-z]:\\' -or $arg -match '^[A-Za-z]:/' -or ($arg -match '\\' -and (Test-Path $arg -ErrorAction SilentlyContinue))) {
            $wslArgs += Convert-ToWslPath $arg
        } else {
            $wslArgs += $arg
        }
    }
    $argsStr = if ($wslArgs.Count -gt 0) { " " + ($wslArgs -join " ") } else { "" }

    Write-Host "[MockWSL] Module: $module" -ForegroundColor Cyan
    Write-Host "[MockWSL] Python: $WSL_PYTHON" -ForegroundColor DarkGray
    Write-Host "[MockWSL] PYTHONPATH: $PROJECT_PYTHONPATH" -ForegroundColor DarkGray
    Write-Host ""
    wsl -e bash -c "$env_prefix $WSL_PYTHON -u -m $module$argsStr"
}
elseif ($ScriptPath) {
    $wslScript = Convert-ToWslPath $ScriptPath
    $argsStr = if ($ExtraArgs) { " " + ($ExtraArgs -join " ") } else { "" }

    Write-Host "[MockWSL] Script: $wslScript" -ForegroundColor Cyan
    Write-Host "[MockWSL] Python: $WSL_PYTHON" -ForegroundColor DarkGray
    Write-Host "[MockWSL] PYTHONPATH: $PROJECT_PYTHONPATH" -ForegroundColor DarkGray
    Write-Host ""
    wsl -e bash -c "$env_prefix $WSL_PYTHON $wslScript$argsStr"
}
else {
    Write-Host "[MockWSL] Environment:" -ForegroundColor Cyan
    Write-Host "  WSL Python:   $WSL_PYTHON" -ForegroundColor White
    Write-Host "  Project root: $PROJECT_ROOT_WSL" -ForegroundColor White
    Write-Host "  PYTHONPATH:   $PROJECT_PYTHONPATH" -ForegroundColor White
    Write-Host ""
    Write-Host "[MockWSL] Usage:" -ForegroundColor Yellow
    Write-Host "  .\Config\run_in_wsl.ps1 <script.py> [args...]"
    Write-Host "  .\Config\run_in_wsl.ps1 -m <module.name> [args...]"
    Write-Host "  .\Config\run_in_wsl.ps1 -c `"import numpy; print(numpy.__version__)`""
}

exit $LASTEXITCODE
