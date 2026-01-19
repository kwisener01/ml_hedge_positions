#!/bin/bash
# Railway startup script

# Set Python path to include project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create logs directory if it doesn't exist
mkdir -p webapp/logs

# Change to webapp directory and start uvicorn
cd webapp
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
