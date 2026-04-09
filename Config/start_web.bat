@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

echo ============================================================
echo  jaxmech Web Frontend Launcher
echo ============================================================

:: ============================================================
:: Step 1: Ensure Config\env.cfg exists (create from template)
:: ============================================================
if not exist "Config\env.cfg" (
    echo [INFO] Config\env.cfg not found, creating from template...
    if exist "Config\env.template.cfg" (
        copy "Config\env.template.cfg" "Config\env.cfg" >nul
    ) else (
        echo # Machine-local environment configuration > "Config\env.cfg"
        echo windows_python_exe = >> "Config\env.cfg"
    )
)

:: ============================================================
:: Step 2: Read windows_python_exe from Config\env.cfg
:: NOTE: Only strip ONE leading space to preserve paths with spaces
::       (e.g. C:\Program Files\...) — do NOT use "val: =!" which
::       removes ALL spaces and destroys such paths.
:: ============================================================
set "WIN_PYTHON="
for /f "usebackq tokens=1,* delims==" %%A in ("Config\env.cfg") do (
    set "_k=%%A"
    set "_k=!_k: =!"
    if "!_k!"=="windows_python_exe" (
        set "_v=%%B"
        if "!_v:~0,1!"==" " set "_v=!_v:~1!"
        if not "!_v!"=="" set "WIN_PYTHON=!_v!"
    )
)

:: ============================================================
:: Step 3: Validate — does the configured path actually exist?
:: ============================================================
if not "!WIN_PYTHON!"=="" (
    if not exist "!WIN_PYTHON!" (
        echo [WARN] Configured Python not found: !WIN_PYTHON!
        set "WIN_PYTHON="
    )
)

:: ============================================================
:: Step 4: Auto-detect Python if not configured / path invalid
:: ============================================================
if "!WIN_PYTHON!"=="" (
    echo [INFO] Auto-detecting Windows Python interpreter...

    :: 4a. Python Launcher (py.exe) — most reliable on Windows 10/11
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if "!WIN_PYTHON!"=="" if exist "%%P" set "WIN_PYTHON=%%P"
    )

    :: 4b. python command in PATH
    if "!WIN_PYTHON!"=="" (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
            if "!WIN_PYTHON!"=="" if exist "%%P" set "WIN_PYTHON=%%P"
        )
    )

    :: 4c. python3 command in PATH
    if "!WIN_PYTHON!"=="" (
        for /f "delims=" %%P in ('python3 -c "import sys; print(sys.executable)" 2^>nul') do (
            if "!WIN_PYTHON!"=="" if exist "%%P" set "WIN_PYTHON=%%P"
        )
    )

    if "!WIN_PYTHON!"=="" (
        echo.
        echo [ERROR] Cannot find a Windows Python interpreter.
        echo   Options:
        echo     1. Install Python from https://www.python.org/
        echo     2. Manually set windows_python_exe in Config\env.cfg
        echo.
        pause
        exit /b 1
    )

    echo [INFO] Detected: !WIN_PYTHON!

    :: Write detected path back to Config\env.cfg (preserves comments)
    "!WIN_PYTHON!" "Config\setup_env.py" --set windows_python_exe "!WIN_PYTHON!" 2>nul
    if errorlevel 1 (
        echo [WARN] Could not auto-update Config\env.cfg.
        echo        Please set windows_python_exe in Config\env.cfg manually.
    )
)

echo.
echo Python: !WIN_PYTHON!
echo Installing/checking web dependencies...
if exist "requirements-web.txt" (
    "!WIN_PYTHON!" -m pip install -r "requirements-web.txt" --quiet
) else (
    "!WIN_PYTHON!" -m pip install numpy scipy fastapi uvicorn pydantic websockets --quiet
)
if errorlevel 1 (
    echo [ERROR] Failed to install/check Windows web dependencies.
    echo        Try: powershell -ExecutionPolicy Bypass -File Config\setup_demo_env.ps1
    pause
    exit /b 1
)

echo Releasing port 8080 if occupied...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8080" ^| findstr "LISTEN"') do (
    echo   Releasing PID %%a ...
    taskkill /PID %%a /F >nul 2>&1
)

echo Starting jaxmech web...
"!WIN_PYTHON!" -m jaxmech.web.run
pause
