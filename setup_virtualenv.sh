#!/bin/bash
set -e
set -x

virtualenv -p python3.7 venv
source venv/bin/activate

pip install -e .
