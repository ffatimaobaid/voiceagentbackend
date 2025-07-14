#!/bin/bash
cd /mnt/c/Users/HP/Downloads/voice-agent-poc/backend
source venv/bin/activate
TORCH_HOME=$(pwd)/.torch_cache gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4

