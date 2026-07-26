@echo off
title Road Cleanliness Analyzer Server
echo ==========================================================
echo Starting Road Cleanliness Analyzer Server...
echo (Using virtual environment Python)
echo ==========================================================
cd /d "%~dp0"
..\.venv\Scripts\python.exe flask_demo_app.py
pause
