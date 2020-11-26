#!/bin/bash
set -e
set -x

virtualenv -p python3.7 venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e forge
pip install -e LieConv
pip install -e .
