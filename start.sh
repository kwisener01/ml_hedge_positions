#!/bin/bash
# Railway startup script

# Set Python path to include project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create required directories if they don't exist
mkdir -p webapp/logs
mkdir -p models/trained

# Change to webapp directory and start uvicorn
cd webapp
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
