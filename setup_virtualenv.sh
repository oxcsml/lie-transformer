#!/bin/bash
set -e
set -x

virtualenv -p python3 venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r forge/requirements.txt