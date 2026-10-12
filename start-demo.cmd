@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-demo.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. See the message above and output\startup.log.
  pause
)
