@echo off
rem Showroom Coach launcher (Windows): sets up .venv and the model weights on first run, then starts the web app.
rem Extra arguments are passed to "showroom-coach serve", e.g.  run.bat --port 8080
rem No labels or goto on purpose: the script keeps working even if a checkout turns its line endings into LF.
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "MODELS_ROOT=models"
if defined SHOWROOM_MODELS set "MODELS_ROOT=%SHOWROOM_MODELS%"

if not exist "%VENV_PY%" (
  echo Showroom Coach: creating the virtual environment in .venv - first run only
  if exist "%PY312%" (
    "%PY312%" -m venv .venv
  ) else (
    py -3.12 -m venv .venv
  )
  if not exist "%VENV_PY%" (
    echo ERROR: could not create .venv. Install Python 3.12 from https://www.python.org/downloads/ and run run.bat again.
    exit /b 1
  )
)

"%VENV_PY%" -c "import showroom_coach" >nul 2>&1
if errorlevel 1 (
  echo Showroom Coach: installing the package and its dependencies - first run only, needs internet
  "%VENV_PY%" -m pip install -e .
  if errorlevel 1 (
    echo ERROR: pip install failed. Check the internet connection and run run.bat again.
    exit /b 1
  )
)

set "NEED_MODELS="
for %%M in (yolox_nano rtmpose_s dinov2_small) do (
  if not exist "%MODELS_ROOT%\%%M\%%M.onnx" set "NEED_MODELS=1"
)
if defined NEED_MODELS (
  echo Showroom Coach: downloading the model weights - first run only, needs internet
  "%VENV_PY%" -m showroom_coach fetch-models yolox_nano rtmpose_s dinov2_small
  if errorlevel 1 (
    echo ERROR: the model download failed. Check the internet connection and run run.bat again.
    exit /b 1
  )
)

echo Showroom Coach: open http://localhost:8000 (Ctrl+C to stop)
"%VENV_PY%" -m showroom_coach serve --host 127.0.0.1 --port 8000 %*
exit /b %errorlevel%
