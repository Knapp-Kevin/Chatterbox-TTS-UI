@echo off
setlocal

set VENV_DIR=.venv
set SCRIPT_NAME=main.py
set PYTHON_GUI_EXE=%VENV_DIR%\Scripts\pythonw.exe
set PYTHON_CONSOLE_EXE=%VENV_DIR%\Scripts\python.exe

call "%~dp0setup_env.bat"
if %ERRORLEVEL% neq 0 (
    echo.
    echo Setup failed. See the newest log in the logs folder.
    pause
    exit /b 1
)

echo Launching %SCRIPT_NAME%...
if exist "%PYTHON_GUI_EXE%" (
    start "" "%PYTHON_GUI_EXE%" "%SCRIPT_NAME%"
) else (
    start "" "%PYTHON_CONSOLE_EXE%" "%SCRIPT_NAME%"
)

exit /b 0
