@echo off
echo [1/2] Checking and installing required dependencies...
python -m pip install -r requirements.txt
echo.
echo [2/2] Running main.py...
python main.py
pause
