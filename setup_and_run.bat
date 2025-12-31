@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ==================================================
REM ASR Project – Dependency Setup + Terminal Launcher
REM ==================================================

echo.
echo ===============================
echo  ASR Environment Setup
echo ===============================
echo.

REM Ensure we are in the script directory
cd /d "%~dp0"

REM -------------------------------
REM Check Python
REM -------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM -------------------------------
REM Create virtual environment
REM -------------------------------
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM -------------------------------
REM Activate virtual environment
REM -------------------------------
call venv\Scripts\activate.bat

REM -------------------------------
REM Upgrade pip
REM -------------------------------
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM -------------------------------
REM Install core dependencies
REM -------------------------------
echo.
echo Installing dependencies...

pip install ^
    numpy ^
    torch ^
    sounddevice ^
    tqdm

REM -------------------------------
REM Torch CUDA note
REM -------------------------------
echo.
echo =========================================
echo Torch installed.
echo If you want CUDA support:
echo   pip uninstall torch
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo =========================================

REM -------------------------------
REM Final message + shell
REM -------------------------------
echo.
echo =========================================
echo Setup complete.
echo Virtual environment is ACTIVE.
echo =========================================
echo.
echo Common commands:
echo   python speech-recognition-main.py
echo   python speech-recognition-main.py --mode menu
echo.

REM Keep terminal open and interactive
cmd /k
