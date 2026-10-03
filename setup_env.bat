@echo off
setlocal enabledelayedexpansion

set VENV_DIR=.venv
set REQUIREMENTS_INPUT_FILE=requirements.in
set REQUIREMENTS_LOCK_FILE=requirements.lock.txt
set REQUIRED_PYTHON=3.11
set DEPS_STAMP=%VENV_DIR%\.deps_installed
set TORCH_STAMP=%VENV_DIR%\.torch_checked
set LOG_DIR=logs
set CREATED_VENV=0
set RUNTIME_LOCK_FILE=%VENV_DIR%\.requirements.runtime.lock.txt
set PYTHON_CONSOLE_EXE=%VENV_DIR%\Scripts\python.exe

if not defined UV_APP_DRY set "UV_APP_DRY=0"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set LOG_TIMESTAMP=%%i
set LOG_FILE=%LOG_DIR%\installer_%LOG_TIMESTAMP%.log
set CHATTERBOX_INSTALL_LOG=%CD%\%LOG_FILE%

echo.
echo === Starting This Voice Thing Installer ===
echo.
echo Installer log: %LOG_FILE%
call :LOG_MESSAGE "=== Starting This Voice Thing Installer ==="
call :LOG_MESSAGE "Dry Run Mode: %UV_APP_DRY%"

where python >nul
if %ERRORLEVEL% neq 0 (
    call :LOG_MESSAGE "ERROR: Python is missing from PATH."
    echo ERROR: Python is missing!
    echo Please install Python 3.11 or higher from: https://www.python.org/downloads/windows/
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation. After installing Python, re-run this script.
    exit /b 1
)

echo Checking for uv...
call :LOG_MESSAGE "Checking for uv..."
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    call :LOG_MESSAGE "ERROR: uv is missing."
    echo ERROR: UV is missing.
    echo Please install it first: https://github.com/astral-sh/uv#getting-started/installation
    echo Easiest way on Windows: pip install uv
    exit /b 1
)

if not exist "%VENV_DIR%" (
    echo Creating venv...
    call :LOG_MESSAGE "Creating venv at %VENV_DIR% with Python %REQUIRED_PYTHON%."
    uv venv "%VENV_DIR%" --python %REQUIRED_PYTHON% >> "%LOG_FILE%" 2>&1 || (
        call :LOG_MESSAGE "ERROR: Venv creation failed."
        echo ERROR: Venv creation failed!
        exit /b 1
    )
    set CREATED_VENV=1
    call :LOG_MESSAGE "Venv created successfully."
)

if exist "%PYTHON_CONSOLE_EXE%" (
    for /f %%i in ('call "%PYTHON_CONSOLE_EXE%" -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"') do set VENV_PYTHON_VERSION=%%i
    call :LOG_MESSAGE "Detected venv Python version !VENV_PYTHON_VERSION!."
    if not "!VENV_PYTHON_VERSION!"=="%REQUIRED_PYTHON%" (
        call :LOG_MESSAGE "Rebuilding venv because Python version does not match requirement."
        echo Existing venv uses Python !VENV_PYTHON_VERSION!, but this app requires Python %REQUIRED_PYTHON%.
        rmdir /s /q "%VENV_DIR%"
        uv venv "%VENV_DIR%" --python %REQUIRED_PYTHON% >> "%LOG_FILE%" 2>&1 || (
            call :LOG_MESSAGE "ERROR: Venv recreation failed."
            echo ERROR: Venv recreation failed!
            exit /b 1
        )
        set CREATED_VENV=1
        call :LOG_MESSAGE "Venv recreated successfully."
    )
)

call "%VENV_DIR%\Scripts\activate"
call :LOG_MESSAGE "Activated virtual environment."

if "%CREATED_VENV%"=="1" (
    call :LOG_MESSAGE "New virtual environment detected. Clearing installer stamp files."
    if exist "%DEPS_STAMP%" del /q "%DEPS_STAMP%"
    if exist "%TORCH_STAMP%" del /q "%TORCH_STAMP%"
)

set NEED_LOCK_COMPILE=0
if not exist "%REQUIREMENTS_LOCK_FILE%" set NEED_LOCK_COMPILE=1
if exist "%REQUIREMENTS_INPUT_FILE%" call :SET_NEWER_FLAG "%REQUIREMENTS_INPUT_FILE%" "%REQUIREMENTS_LOCK_FILE%" NEED_LOCK_COMPILE

if "%NEED_LOCK_COMPILE%"=="1" (
    echo Ensuring lock file is up to date...
    call :LOG_MESSAGE "Running uv pip compile because the lock file is missing or stale."
    uv pip compile "%REQUIREMENTS_INPUT_FILE%" -o "%REQUIREMENTS_LOCK_FILE%" >> "%LOG_FILE%" 2>&1 || (
        call :LOG_MESSAGE "ERROR: Lock file generation failed."
        echo ERROR: Lock file generation failed!
        exit /b 1
    )
    call :LOG_MESSAGE "Lock file generation succeeded."
) else (
    call :LOG_MESSAGE "Lock file already up to date. Skipping uv pip compile."
    echo Lock file already up to date.
)

call :BUILD_RUNTIME_LOCK
if %ERRORLEVEL% neq 0 (
    call :LOG_MESSAGE "ERROR: Failed to build runtime lock file without torch packages."
    echo ERROR: Failed to build runtime dependency lock.
    exit /b 1
)

set NEED_DEPS_SYNC=0
if not exist "%DEPS_STAMP%" set NEED_DEPS_SYNC=1
if exist "%REQUIREMENTS_LOCK_FILE%" call :SET_NEWER_FLAG "%REQUIREMENTS_LOCK_FILE%" "%DEPS_STAMP%" NEED_DEPS_SYNC
if "%CREATED_VENV%"=="1" set NEED_DEPS_SYNC=1

if "%NEED_DEPS_SYNC%"=="1" (
    echo Installing dependencies from lock file...
    call :LOG_MESSAGE "Running uv pip install from the runtime lock because dependencies are not stamped or the lock changed."
    uv pip install --python "%PYTHON_CONSOLE_EXE%" -r "%RUNTIME_LOCK_FILE%" --strict >> "%LOG_FILE%" 2>&1 || (
        call :LOG_MESSAGE "ERROR: Dependency installation from lock file failed."
        echo ERROR: Dependency installation from lock file failed!
        exit /b 1
    )
    > "%DEPS_STAMP%" echo Dependencies synced from %RUNTIME_LOCK_FILE%
    call :LOG_MESSAGE "Dependency sync succeeded. Updated %DEPS_STAMP%."
) else (
    call :LOG_MESSAGE "Dependencies already installed. Skipping runtime dependency install."
    echo Dependencies already installed. Skipping runtime dependency install.
)

if "%UV_APP_DRY%"=="0" (
    set NEED_TORCH_INSTALL=0
    if not exist "%TORCH_STAMP%" set NEED_TORCH_INSTALL=1
    call :CHECK_TORCH_RUNTIME
    if "!TORCH_RUNTIME_OK!"=="0" (
        call :LOG_MESSAGE "Torch runtime import check failed. Forcing reinstall."
        if exist "%TORCH_STAMP%" del /q "%TORCH_STAMP%"
        set NEED_TORCH_INSTALL=1
    )
    if exist scripts\install_torch.py call :SET_NEWER_FLAG "scripts\install_torch.py" "%TORCH_STAMP%" NEED_TORCH_INSTALL
    if exist "%REQUIREMENTS_LOCK_FILE%" call :SET_NEWER_FLAG "%REQUIREMENTS_LOCK_FILE%" "%TORCH_STAMP%" NEED_TORCH_INSTALL

    if "!NEED_TORCH_INSTALL!"=="1" (
        echo Installing PyTorch...
        call :LOG_MESSAGE "Running install_torch.py because Torch is not stamped or inputs changed."
        "%PYTHON_CONSOLE_EXE%" scripts\install_torch.py
        if !ERRORLEVEL! neq 0 (
            call :LOG_MESSAGE "WARNING: install_torch.py failed."
            echo WARNING: PyTorch install failed. App may lack GPU support.
            if exist "%TORCH_STAMP%" del /q "%TORCH_STAMP%"
            exit /b 1
        ) else (
            > "%TORCH_STAMP%" echo PyTorch checked by install_torch.py
            call :LOG_MESSAGE "Updated %TORCH_STAMP% after install_torch.py."
        )
    ) else (
        call :LOG_MESSAGE "PyTorch already checked. Skipping install_torch.py."
        echo PyTorch already checked. Skipping install_torch.py.
    )
) else (
    call :LOG_MESSAGE "[Dry Run] Skipped PyTorch installation."
    echo [Dry Run] Skipped PyTorch installation
)

echo Setup complete.
exit /b 0

:LOG_MESSAGE
>> "%LOG_FILE%" echo [%date% %time%] %~1
goto :EOF

:CHECK_TORCH_RUNTIME
set TORCH_RUNTIME_OK=0
"%PYTHON_CONSOLE_EXE%" -c "import torch, torchaudio, torchvision" >nul 2>nul
if %ERRORLEVEL%==0 set TORCH_RUNTIME_OK=1
goto :EOF

:BUILD_RUNTIME_LOCK
call :LOG_MESSAGE "Building runtime lock file without torch, torchvision, and torchaudio pins."
powershell -NoProfile -Command ^
    "$inputPath = '%REQUIREMENTS_LOCK_FILE%';" ^
    "$outputPath = '%RUNTIME_LOCK_FILE%';" ^
    "$lines = Get-Content -LiteralPath $inputPath;" ^
    "$filtered = $lines | Where-Object { $_ -notmatch '^(torch|torchaudio|torchvision)=='};" ^
    "Set-Content -LiteralPath $outputPath -Value $filtered -Encoding UTF8"
if %ERRORLEVEL% neq 0 exit /b 1
call :LOG_MESSAGE "Runtime lock file written to %RUNTIME_LOCK_FILE%."
goto :EOF

:SET_NEWER_FLAG
setlocal
set "SOURCE_FILE=%~1"
set "TARGET_FILE=%~2"
set "RESULT_FLAG=%~3"
if not exist "%SOURCE_FILE%" (
    >> "%LOG_FILE%" echo [%date% %time%] Skipping timestamp check because "%SOURCE_FILE%" does not exist.
    endlocal & goto :EOF
)
if not exist "%TARGET_FILE%" (
    >> "%LOG_FILE%" echo [%date% %time%] Marking "%RESULT_FLAG%=1" because "%TARGET_FILE%" does not exist.
    endlocal & set "%RESULT_FLAG%=1" & goto :EOF
)
powershell -NoProfile -Command ^
    "$src = Get-Item -LiteralPath '%SOURCE_FILE%';" ^
    "$dst = Get-Item -LiteralPath '%TARGET_FILE%';" ^
    "if ($src.LastWriteTimeUtc -gt $dst.LastWriteTimeUtc) { exit 0 } else { exit 1 }"
if %ERRORLEVEL%==0 (
    >> "%LOG_FILE%" echo [%date% %time%] "%SOURCE_FILE%" is newer than "%TARGET_FILE%". Marking "%RESULT_FLAG%=1".
    endlocal & set "%RESULT_FLAG%=1" & goto :EOF
)
>> "%LOG_FILE%" echo [%date% %time%] "%SOURCE_FILE%" is not newer than "%TARGET_FILE%". Leaving "%RESULT_FLAG%" unchanged.
endlocal & goto :EOF
