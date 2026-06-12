@echo off
REM WC2026 prediction dashboard - double click to start
REM Browser opens automatically; close this window to stop the server.
cd /d "%~dp0"
C:\ProgramData\anaconda3\envs\python310\python.exe -m streamlit run app.py --server.port 8510
pause
