@echo off
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%PROJECT_ROOT%scripts\start_app.ps1"

if errorlevel 1 (
  echo.
  echo 启动失败，请查看上方错误信息。
  pause
)
