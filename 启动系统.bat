@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title 股票价格预测系统

pushd "%~dp0"
if errorlevel 1 (
    echo 无法打开项目目录，请把启动文件放在项目根目录。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" goto missing_python
if not exist "scripts\start_app.ps1" goto missing_launcher

echo 正在启动系统，请稍候。首次打开可能需要几秒钟。
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_app.ps1" %*
set "LAUNCH_EXIT=%ERRORLEVEL%"
goto finish

:missing_python
echo 未找到项目 Python 环境，请先按 README.md 完成环境安装。
set "LAUNCH_EXIT=1"
goto finish

:missing_launcher
echo 缺少 scripts\start_app.ps1，请检查项目文件是否完整。
set "LAUNCH_EXIT=1"

:finish
popd
if "%LAUNCH_EXIT%"=="0" exit /b 0
echo.
echo 启动失败，请查看上方提示和 artifacts\logs 中的启动日志。
pause
exit /b %LAUNCH_EXIT%
