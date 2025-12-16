@echo off
cd /d "%~dp0"

git fetch origin

git checkout Converter_v4

git pull

start "" python main.py

exit