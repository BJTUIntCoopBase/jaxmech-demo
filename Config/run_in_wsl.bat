@echo off
setlocal
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_in_wsl.ps1" %*
exit /b %ERRORLEVEL%
