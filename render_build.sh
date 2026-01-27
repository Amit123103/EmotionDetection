#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Upgrade pip..."
python3 -m pip install --upgrade pip

echo "Installing dependencies..."
python3 -m pip install -r backend/requirements.txt

echo "Build finished!"
