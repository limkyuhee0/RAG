#!/bin/bash
set -e  # 에러 발생 시 스크립트 중단

export PYTHONUNBUFFERED=1

echo "[1] Preprocessing PDFs..."
poetry run python src/preprocess_pdf.py || { echo "Error in preprocess_pdf.py"; exit 1; }

echo "[2] Building Vector DB..."
poetry run python src/retriever.py || { echo "Error in retriever.py"; exit 1; }

echo "[3] Running Agent Evaluation..."
poetry run python src/agent.py || { echo "Error in agent.py"; exit 1; }

echo "Done!"
