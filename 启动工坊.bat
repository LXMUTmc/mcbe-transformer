@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  chcp 65001 >nul
  echo 还没有安装运行环境：请先双击 安装环境.bat
  pause
  exit /b 1
)
start "" .venv\Scripts\pythonw.exe app\mct_studio.py
