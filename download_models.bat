@echo off
setlocal

echo ==========================================
echo Downloading LaborRAG LLM models
echo ==========================================

REM Папка для моделей
set MODELS_DIR=C:\Projects\RAGIntelligentLaborSystem\models

if not exist "%MODELS_DIR%" (
    mkdir "%MODELS_DIR%"
)

cd /d "%MODELS_DIR%"

REM Hugging Face CLI + ускоренная загрузка через Xet
python -m pip install -U huggingface_hub hf_xet

echo.
echo ==========================================
echo 1/3 Qwen3.5-9B Q4_K_M
echo ==========================================

hf download unsloth/Qwen3.5-9B-GGUF ^
    Qwen3.5-9B-Q4_K_M.gguf ^
    --local-dir "%MODELS_DIR%\Qwen3.5-9B"

echo.
echo ==========================================
echo 2/3 Ministral 3 14B Instruct Q4_K_M
echo ==========================================

hf download mistralai/Ministral-3-14B-Instruct-2512-GGUF ^
    Ministral-3-14B-Instruct-2512-Q4_K_M.gguf ^
    --local-dir "%MODELS_DIR%\Ministral-3-14B"

echo.
echo ==========================================
echo 3/3 Gemma 3 12B IT Q4_K_M
echo ==========================================

hf download tensorblock/gemma-3-12b-it-GGUF ^
    gemma-3-12b-it-Q4_K_M.gguf ^
    --local-dir "%MODELS_DIR%\Gemma-3-12B"

echo.
echo ==========================================
echo Download complete
echo ==========================================

echo.
echo Models directory:
echo %MODELS_DIR%

echo.
echo Installed models:

pause