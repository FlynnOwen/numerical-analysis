#!/usr/bin/env just --justfile
export PYTHON_VERSION := "3.11"

# Virtual environment
install-venv:
    [ -d venv ] || python{{PYTHON_VERSION}} -m venv venv

install-requirements: install-venv
    ./venv/bin/python -m pip install -r requirements.txt

# Linting
lint:
    ruff .

format:
    black .