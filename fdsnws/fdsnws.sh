#!/usr/bin/env bash
uvicorn fdsnws:app --host localhost --port 8000 --reload
