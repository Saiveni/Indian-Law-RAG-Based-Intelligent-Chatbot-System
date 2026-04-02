@echo off
echo ========================================
echo Legal ChatBot - Quick Setup
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo Step 2: Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)
echo.

echo Step 3: Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

echo Step 4: Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

echo Step 5: Checking .env file...
if not exist ".env" (
    echo WARNING: .env file not found
    echo Please create a .env file with your API keys:
    echo   GOOGLE_API_KEY=your_key_here
    echo   GROQ_API_KEY=your_key_here
    echo.
    echo You can copy .env.example to .env and fill in your keys
) else (
    echo .env file found
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the chatbot:
echo   1. Ensure you have added your API keys to .env file
echo   2. Place PDF files in LEGAL-DATA folder
echo   3. Run: python ingestion.py (if vector store doesn't exist)
echo   4. Run: streamlit run app.py
echo.
pause
