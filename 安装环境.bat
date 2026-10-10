@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  MCBE Transformer 工坊：安装运行环境
echo  需要：Python 3.10 以上（python.org 下载，安装时勾选 Add to PATH）
echo        NVIDIA 显卡 + 驱动（训练用；没有显卡也能装，但训练极慢）
echo  会下载约 3 GB（PyTorch），可能要十几分钟。
echo ============================================================
set PYEXE=
where py >nul 2>nul && (py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul && set PYEXE=py -3)
if "%PYEXE%"=="" (
  where python >nul 2>nul && (python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul && set PYEXE=python)
)
if "%PYEXE%"=="" (
  echo 没有找到 Python 3.10 以上。请先安装 Python：https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo 正在创建虚拟环境 .venv ...
  %PYEXE% -m venv .venv || (echo 创建虚拟环境失败 & pause & exit /b 1)
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install numpy pypinyin || (echo 安装 numpy / pypinyin 失败 & pause & exit /b 1)
echo 正在安装 PyTorch（CUDA 12.8 版）...
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || (echo 安装 PyTorch 失败 & pause & exit /b 1)
echo 正在安装 triton-windows（训练加速，可选；失败也不影响使用）...
.venv\Scripts\python.exe -m pip install "triton-windows<3.7"
echo.
echo 安装完成。双击 启动工坊.bat 打开程序。
pause
