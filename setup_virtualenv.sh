#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt
pip install -r forge/requirements.txt