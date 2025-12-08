@echo off
cd /d "%~dp0"

git fetch origin

git checkout Converter_v1

git pull

start "" pythonw main.py

exit