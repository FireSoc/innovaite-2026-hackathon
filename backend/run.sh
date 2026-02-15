#!/usr/bin/env bash
# Run from repo root: ./backend/run.sh
# Or from backend: ./run.sh
cd "$(dirname "$0")" && uvicorn app.main:app --reload
